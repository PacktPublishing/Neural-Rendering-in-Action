"""
slangpy_common.py -- shared plumbing for every *_slangpy.py training driver
in this chapter (the occupancy/SDF/appearance sections' GPU/SlangPy ports).
============================================================================
Three things every one of those scripts used to define locally, byte-for-
byte or near enough: Vulkan device creation, the NetworkParameters wrapper
(one differentiable dense layer's weights/biases, their Adam moment
buffers, and its .optimize() step -- the Python-side mirror of
common_network.slang's NetworkParameters<Inputs,Outputs> struct), and the
chunked/zero-padded batched-inference loop every Network.predict() repeats.

Each driver still defines its own Network wrapper class (the actual
architecture -- how many layers, what extra inputs like view direction or a
feature grid, the training loop) and its own *_train.slang module. Only the
identical plumbing lives here.
"""
import numpy as np
import os
import slangpy as spy
from pathlib import Path

# Vendor IDs Vulkan reports via AdapterInfo.vendor_id (PCI-SIG assigned) --
# recognized as shorthand for CHAPTER6_GPU, below.
_VENDOR_IDS = {"amd": 0x1002, "nvidia": 0x10DE, "intel": 0x8086}


def list_adapters():
    """Print every Vulkan adapter this machine exposes, with the vendor
    shorthand (if any) and the substring CHAPTER6_GPU would need to match
    it. Run this first if you're not sure what to pass:

        python -c "import slangpy_common as sc; sc.list_adapters()" """
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    if not adapters:
        print("No Vulkan adapters found.")
        return
    for a in adapters:
        vendor = next((n for n, vid in _VENDOR_IDS.items() if vid == a.vendor_id), None)
        tag = f"vendor={vendor}" if vendor else f"vendor_id=0x{a.vendor_id:04X}"
        print(f"  {a.name}  ({tag})")


def _preferred_adapter_luid():
    """Which GPU to use is picked by the CHAPTER6_GPU environment variable,
    not hardcoded -- this code runs on whatever machine and GPU(s) a reader
    has, not just the one it was developed on. Unset (the default): Vulkan
    picks its own default adapter, untouched.

    Set CHAPTER6_GPU to steer it -- either a vendor shorthand (amd, nvidia,
    intel) or any case-insensitive substring of the adapter's name (e.g.
    "3050", "radeon", "arc"), for multi-GPU-of-the-same-vendor machines or
    to sidestep an unfamiliar vendor string:

        (bash)       CHAPTER6_GPU=nvidia python train_one.py 4 1500
        (PowerShell) $env:CHAPTER6_GPU="nvidia"; python train_one.py 4 1500

    Run list_adapters() (above) to see what this machine offers and what
    name/substring would match it. Raises if CHAPTER6_GPU is set but
    nothing on the machine matches -- silently falling back to Vulkan's
    default would be more confusing than an error naming what IS
    available."""
    name = os.environ.get("CHAPTER6_GPU", "").strip().lower()
    if not name:
        return None
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    if name in _VENDOR_IDS:
        vendor_id = _VENDOR_IDS[name]
        matches = [a for a in adapters if a.vendor_id == vendor_id]
    else:
        matches = [a for a in adapters if name in a.name.lower()]
    if not matches:
        available = ", ".join(a.name for a in adapters) or "(none found)"
        raise ValueError(f"CHAPTER6_GPU={name!r} matched no Vulkan adapter. "
                         f"Available: {available}")
    return matches[0].luid


def create_device(this_file):
    """Vulkan device with this script's own directory AND the chapter root
    on the Slang search path -- the chapter root holds common_network.slang,
    imported by every section's own *_train.slang module.

    enable_hot_reload=False: this project lives on a Google Drive-synced
    path, and SlangPy's default hot-reload polls every imported source
    file's last-write-time to detect edits. Against that drive it
    intermittently failed outright ("Failed to get last write time") and
    may be a real contributor to the excessive compile times seen in this
    environment -- none of these batch training scripts edit shaders
    while running, so there's nothing hot-reload buys here."""
    here = Path(this_file).resolve().parent
    return spy.create_device(spy.DeviceType.vulkan, enable_debug_layers=False,
                             adapter_luid=_preferred_adapter_luid(),
                             enable_hot_reload=False,
                             include_paths=[here, here.parent])


class NetworkParameters(spy.InstanceList):
    """One differentiable dense layer: weights, biases, their gradients, and
    Adam's first/second moment buffers -- the Python-side counterpart of
    common_network.slang's NetworkParameters<Inputs,Outputs>."""

    def __init__(self, module, device, inputs, outputs, seed=0, out_scale=1.0, out_bias=0.0):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self._module = module
        rng = np.random.default_rng(seed)
        biases = np.full(outputs, out_bias, np.float32)
        weights = (rng.standard_normal((outputs, inputs))
                  * np.sqrt(2 / inputs) * out_scale).astype(np.float32)
        self.biases = spy.Tensor.from_numpy(device, biases)
        self.weights = spy.Tensor.from_numpy(device, weights)
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, lr, i):
        self._module.optimizer_step(self.biases, self.biases_grad, self.m_biases, self.v_biases, lr, i)
        self._module.optimizer_step(self.weights, self.weights_grad, self.m_weights, self.v_weights, lr, i)

    def n_params(self):
        return self.biases.shape[0] + self.weights.shape[0] * self.weights.shape[1]


def predict_chunked(pts, chunk, pts_t, result_t, dispatch):
    """The chunked/zero-padded batched-inference loop every *_slangpy.py
    driver's Network.predict() used to repeat: split `pts` into chunk-sized
    (zero-padded on the last, partial chunk) batches, upload each into the
    caller's persistent pts_t, call the caller's `dispatch()` closure (which
    runs the section-specific predict_batch kernel into the caller's
    persistent result_t), then trim the padding back off on the way out.

    pts_t / result_t: persistent GPU tensors, shaped (chunk, ...), already
    allocated by the caller (so repeated calls reuse them instead of
    allocating a fresh buffer every time -- see Section4/README.md on why
    that matters on this toolchain).
    dispatch: zero-argument callable; reads pts_t, writes result_t.
    """
    pts = np.asarray(pts, np.float32)
    # spy.Tensor.shape is a Shape object: supports int indexing, not slicing.
    extra_dims = tuple(result_t.shape[d] for d in range(1, len(result_t.shape)))
    out = np.empty((len(pts),) + extra_dims, np.float32)
    for i in range(0, len(pts), chunk):
        block = pts[i:i + chunk]
        n = len(block)
        if n < chunk:
            padded = np.zeros((chunk,) + pts.shape[1:], np.float32)
            padded[:n] = block
            block = padded
        pts_t.copy_from_numpy(block)
        dispatch()
        out[i:i + n] = result_t.to_numpy()[:n]
    return out
