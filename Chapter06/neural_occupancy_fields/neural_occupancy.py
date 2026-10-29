"""
neural_occupancy.py -- driving the occupancy field from Python
=====================================================
Paired with `neural_occupancy.slang`, which is the network; this file is
the Python side of it. There is one network implementation in the occupancy section and
it is the Slang one -- training, inference, checkpointing and meshing all go
through the module compiled here. `common_shapes.slang` (the shape) and
`common_rng.slang` (GPU-side randomness) supply everything the training
kernel needs to sample and label a batch itself: there is no CPU RNG and no
CPU shape math anywhere in this file's training path.

    CoordinateMLP   build / load the network, evaluate it, checkpoint it
    train           the training loop (Adam on binary cross-entropy)
    field_to_grid   sample the trained field on a lattice for marching cubes
    sample_labelled_points, occupancy_grid_gpu, detailed_torus_inside_batch
                    GPU-dispatched ground truth, for figures and scoring
                    only -- the trained networks never see any of this

The single most important experimental knob is L, the number of Fourier
bands in the positional encoding. L = 0 means "raw coordinates" -- the
baseline that fails to reproduce the ripples in Part 4. Everything else is
held fixed as L changes, so any difference is attributable to the encoding.

positional_encoding() below is a NumPy mirror of the encoding the shader
computes. It exists to DRAW the encoding for the Part 2 figure, and to
document the column order; the network never calls it.
"""

import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import slangpy_common as sc

HERE = Path(__file__).resolve().parent
SHADER = HERE / "neural_occupancy.slang"

# Width of each of the three hidden layers. One place, one number.
# The book uses 256, which compiles in ~10 s and trains fine; 64 is the
# default because it trains ~12x faster (see README.md, "Differences from
# the book").
HIDDEN = 64
BOUND = 1.10          # the shape fits inside [-BOUND, BOUND]^3

_device = None
_modules = {}


def _module(L, hidden):
    """Compile neural_occupancy.slang for one (L, hidden) pair, once.

    SlangPy's `options={"defines": ...}` never reaches the Slang
    preprocessor -- Module.load_from_file hands the path to
    device.load_module and keeps `options` for call configuration -- so the
    two compile-time constants are prepended to the source instead. Silent
    if you get it wrong: the shader keeps its own defaults while Python
    sizes the layers for the L you asked for, and the network trains on
    garbage.
    """
    global _device
    if _device is None:
        _device = sc.create_device(__file__)
    key = (L, hidden)
    if key not in _modules:
        _modules[key] = spy.Module.load_from_source(
            _device, f"neural_occupancy_L{L}_H{hidden}",
            f"#define OCC_L {L}\n#define OCC_HIDDEN {hidden}\n"
            + SHADER.read_text())
    return _modules[key]


# ----------------------------------------------------------------------------
# Part A -- Fourier positional encoding (the NumPy mirror, for figures)
# ----------------------------------------------------------------------------
# A ReLU MLP fed raw coordinates exhibits "spectral bias": it learns low
# frequencies quickly and high frequencies extremely slowly, if at all
# (Rahaman et al. 2019; Tancik et al. 2020). The fix is to lift each
# coordinate into a bank of sinusoids at geometrically spaced frequencies,
#
#   gamma(p) = [ p,
#                sin(2^0 pi p), cos(2^0 pi p),
#                ...
#                sin(2^{L-1} pi p), cos(2^{L-1} pi p) ].
#
# Each sin()/cos() term is a whole (x, y, z) triple, so one octave occupies
# six columns ordered [sin x, sin y, sin z, cos x, cos y, cos z]. The shader
# writes them in exactly that order.

def positional_encoding(p, L):
    """gamma(p): (N, 3) -> (N, 3 + 6L). L = 0 returns the raw coordinates."""
    out = [p]
    for k in range(L):
        out.append(np.sin(2.0 ** k * np.pi * p))
        out.append(np.cos(2.0 ** k * np.pi * p))
    return np.concatenate(out, axis=1)


def pe_dim(L):
    return 3 + 6 * L


# ----------------------------------------------------------------------------
# Part B -- the network
# ----------------------------------------------------------------------------

class CoordinateMLP(spy.InstanceList):
    """gamma(x) -> [hidden, ReLU] x3 -> logit, living on the GPU.

    Each layer is a NetworkParameters: weights, biases, their gradient
    buffers and Adam's two moment buffers. Slang stores a weight matrix as
    (outputs, inputs); the checkpoints on disk keep the (inputs, outputs)
    orientation the book's equations use, so save/load transpose.
    """

    N_LAYERS = 4

    def __init__(self, L=10, hidden=HIDDEN, seed=0):
        module = _module(L, hidden)
        super().__init__(module["Network"])
        self.L, self.hidden = L, hidden
        dims = [(pe_dim(L), hidden), (hidden, hidden), (hidden, hidden),
                (hidden, 1)]
        self.layer0, self.layer1, self.layer2, self.layer3 = [
            sc.NetworkParameters(module, _device, n_in, n_out, seed + i)
            for i, (n_in, n_out) in enumerate(dims)]
        self._predict_buffers = None

    @property
    def _layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3]

    def n_params(self):
        return sum(l.n_params() for l in self._layers)

    def memory_bytes(self):
        return self.n_params() * 4                    # float32

    def optimize(self, lr, step):
        for layer in self._layers:
            layer.optimize(lr, step)

    def occupancy(self, pts, chunk=65536):
        """P(inside) at each point. pts: (N, 3) -> (N,)."""
        if self._predict_buffers is None or self._predict_buffers[0] != chunk:
            self._predict_buffers = (
                chunk,
                spy.Tensor.empty(_device, (chunk,), "float3"),
                spy.Tensor.empty(_device, (chunk,), "float"))
        _, pts_t, result_t = self._predict_buffers
        module = _module(self.L, self.hidden)
        return sc.predict_chunked(
            pts, chunk, pts_t, result_t,
            lambda: module.predict_batch(idx=spy.grid((chunk,)), pts=pts_t,
                                         network=self, _result=result_t))

    def save(self, path):
        d = {"L": self.L, "hidden": self.hidden, "n_layers": self.N_LAYERS}
        for i, layer in enumerate(self._layers):
            d[f"W{i}"] = layer.weights.to_numpy().T      # -> (inputs, outputs)
            d[f"b{i}"] = layer.biases.to_numpy()
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        # Read the width off the weights rather than trusting a stored key:
        # checkpoints written before `hidden` was recorded (including the
        # book's original 256-wide ones) still load.
        hidden = int(d["hidden"]) if "hidden" in d.files else int(d["W0"].shape[1])
        m = cls(L=int(d["L"]), hidden=hidden)
        for i, layer in enumerate(m._layers):
            # np.array always copies; np.ascontiguousarray does not. For the
            # 1-output last layer the transpose is a (1, N) view NumPy calls
            # contiguous but whose strides slangpy's copy_from_numpy rejects.
            w = np.array(d[f"W{i}"].T, np.float32, order="C")
            w[np.abs(w) < 1e-20] = 0.0        # flush denormals (slow on CPU)
            layer.weights.copy_from_numpy(w)
            layer.biases.copy_from_numpy(
                np.ascontiguousarray(d[f"b{i}"], np.float32))
        return m


# ----------------------------------------------------------------------------
# Part C -- the training loop
# ----------------------------------------------------------------------------

def train(model, steps=1500, batch=4096, lr=1e-3, near_frac=0.5, jitter=0.05,
          seed=0, verbose=True):
    """Fit `model` to the rippled torus.

    One step is: dispatch calculate_grads, which samples the batch itself
    (half uniform in the box, half near the surface + jitter, via
    common_rng.slang + common_shapes.slang -- no CPU RNG, no pts/labels
    tensor uploaded from Python at all), backprops every point into the
    layers' atomic gradient buffers, then one Adam step per layer. The
    inference kernel is warmed up before the loop: on this driver, first
    use of a second kernel after many dispatches intermittently failed to
    allocate its buffers.
    """
    module = _module(model.L, model.hidden)
    n_uniform = batch - int(batch * near_frac)
    pts_t = spy.Tensor.empty(_device, (batch,), "float3")
    y_t = spy.Tensor.empty(_device, (batch,), "float")
    probe_t = spy.Tensor.empty(_device, (batch,), "float")
    grid = spy.grid((batch,))
    model.occupancy(np.zeros((1, 3), np.float32))        # warm up predict_batch

    history = {"loss": [], "acc": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        rng_seed = seed * 1_000_003 + step
        module.calculate_grads(idx=grid, seed=rng_seed, n_uniform=n_uniform,
                               jitter=jitter, network=model)
        model.optimize(lr, step)

        # Loss and accuracy are reporting-only (they drive the Part 3
        # curves): regenerate the SAME batch on the GPU -- sample_point
        # with the same seed reproduces the exact points, then
        # detailed_torus_inside_batch labels them -- and read back a
        # forward pass on it, rather than returning anything from the
        # training kernel itself.
        module.sample_point(idx=grid, seed=rng_seed, n_uniform=n_uniform,
                            jitter=jitter, _result=pts_t)
        module.detailed_torus_inside_batch(idx=grid, pts=pts_t, _result=y_t)
        module.predict_batch(idx=grid, pts=pts_t, network=model, _result=probe_t)
        p = np.clip(probe_t.to_numpy(), 1e-7, 1 - 1e-7)
        y = y_t.to_numpy()
        history["loss"].append(
            float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))
        history["acc"].append(float(((p > 0.5) == (y > 0.5)).mean()))
        if verbose and step % 500 == 0:
            print(f"    step {step:5d}  loss {history['loss'][-1]:.4f}  "
                  f"batch acc {history['acc'][-1]:.3f}  "
                  f"({time.time() - t0:.1f}s)")
    history["seconds"] = time.time() - t0
    return history


# ----------------------------------------------------------------------------
# Optional: Stanford bunny mesh support (train_bunny.py). Same CoordinateMLP,
# same bce_loss -- only the sampling/labeling kernels differ (they read a
# precomputed grid + surface cloud, built once by mesh_shape.load_and_cache,
# instead of the analytic torus). See calculate_grads_mesh's comment in
# neural_occupancy.slang for why: there is no closed-form inside test for an
# arbitrary mesh.
# ----------------------------------------------------------------------------

def train_mesh(model, grid, origin, spacing, res, cloud, steps=1500, batch=4096,
               lr=1e-3, near_frac=0.5, jitter=0.02, seed=0, verbose=True):
    """Fit `model` to a mesh's cached occupancy grid + surface cloud (both
    from mesh_shape.load_and_cache). Structurally identical to train():
    calculate_grads_mesh samples/labels/backprops one point per GPU thread
    per step; only which kernel it calls differs from the torus path.

    grid, origin, spacing, cloud: NumPy arrays from load_and_cache -- this
    function does the one-time upload to GPU Tensors itself.
    """
    module = _module(model.L, model.hidden)
    n_uniform = batch - int(batch * near_frac)
    n_cloud = cloud.shape[0]
    res = int(res)
    grid_t = spy.Tensor.from_numpy(_device, np.ascontiguousarray(grid.ravel(), np.float32))
    # spy.Tensor.from_numpy on a plain (N, 3) array infers a 2D float
    # tensor, not Tensor<float3, 1> -- match predict_chunked's own pattern
    # (slangpy_common.py) instead: an explicit "float3" tensor + copy.
    cloud_t = spy.Tensor.empty(_device, (n_cloud,), "float3")
    cloud_t.copy_from_numpy(np.ascontiguousarray(cloud, np.float32))
    origin_l, spacing_l = origin.tolist(), spacing.tolist()

    pts_t = spy.Tensor.empty(_device, (batch,), "float3")
    y_t = spy.Tensor.empty(_device, (batch,), "float")
    probe_t = spy.Tensor.empty(_device, (batch,), "float")
    grid_dispatch = spy.grid((batch,))
    model.occupancy(np.zeros((1, 3), np.float32))         # warm up predict_batch

    history = {"loss": [], "acc": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        rng_seed = seed * 1_000_003 + step
        module.calculate_grads_mesh(
            idx=grid_dispatch, seed=rng_seed, n_uniform=n_uniform, jitter=jitter,
            grid=grid_t, origin=origin_l, spacing=spacing_l, res=res,
            cloud=cloud_t, n_cloud=n_cloud, network=model)
        model.optimize(lr, step)

        module.sample_point_mesh(idx=grid_dispatch, seed=rng_seed, n_uniform=n_uniform,
                                 jitter=jitter, cloud=cloud_t, n_cloud=n_cloud,
                                 _result=pts_t)
        module.mesh_inside_batch(idx=grid_dispatch, pts=pts_t, grid=grid_t,
                                 origin=origin_l, spacing=spacing_l, res=res,
                                 _result=y_t)
        module.predict_batch(idx=grid_dispatch, pts=pts_t, network=model, _result=probe_t)
        p = np.clip(probe_t.to_numpy(), 1e-7, 1 - 1e-7)
        y = y_t.to_numpy()
        history["loss"].append(
            float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))
        history["acc"].append(float(((p > 0.5) == (y > 0.5)).mean()))
        if verbose and step % 500 == 0:
            print(f"    step {step:5d}  loss {history['loss'][-1]:.4f}  "
                  f"batch acc {history['acc'][-1]:.3f}  "
                  f"({time.time() - t0:.1f}s)")
    history["seconds"] = time.time() - t0
    return history


# ----------------------------------------------------------------------------
# Ground truth, entirely on the GPU (common_shapes.slang) -- for figures and
# scoring only. Any already-compiled (L, hidden) module works: none of
# these dispatches touch the network.
# ----------------------------------------------------------------------------

def sample_labelled_points(n, seed=0, near_frac=0.5, jitter=0.05, L=10, hidden=HIDDEN):
    """(points, labels) for Part 1's scatter figure -- the same sampler
    calculate_grads uses internally, exposed standalone since a figure
    needs the points back in Python rather than a gradient."""
    module = _module(L, hidden)
    n_uniform = n - int(n * near_frac)
    pts_t = spy.Tensor.empty(_device, (n,), "float3")
    y_t = spy.Tensor.empty(_device, (n,), "float")
    grid = spy.grid((n,))
    module.sample_point(idx=grid, seed=seed, n_uniform=n_uniform,
                       jitter=jitter, _result=pts_t)
    module.detailed_torus_inside_batch(idx=grid, pts=pts_t, _result=y_t)
    return pts_t.to_numpy(), y_t.to_numpy()


def occupancy_grid_gpu(resolution, bound=BOUND, L=10, hidden=HIDDEN):
    """Evaluate the exact inside test on a dense lattice, on the GPU --
    used only to mesh the GROUND TRUTH for figures (via marching cubes)
    and to score the trained networks against it."""
    module = _module(L, hidden)
    spacing = 2 * bound / (resolution - 1)
    origin = np.array([-bound] * 3, np.float32)
    n = resolution ** 3
    grid_t = spy.Tensor.empty(_device, (n,), "float")
    module.detailed_torus_grid_value(
        idx=spy.grid((n,)), origin=origin.tolist(),
        spacing=[spacing, spacing, spacing], res=resolution, _result=grid_t)
    occ = grid_t.to_numpy().reshape([resolution] * 3)
    return occ, origin, np.array([spacing] * 3, np.float32)


def detailed_torus_inside_batch(pts, L=10, hidden=HIDDEN, chunk=65536):
    """Exact inside test at arbitrary points, on the GPU -- the `inside_a`
    ground truth `metrics.volumetric_iou` scores a trained network against."""
    module = _module(L, hidden)
    pts_t = spy.Tensor.empty(_device, (chunk,), "float3")
    y_t = spy.Tensor.empty(_device, (chunk,), "float")
    out = sc.predict_chunked(
        pts, chunk, pts_t, y_t,
        lambda: module.detailed_torus_inside_batch(idx=spy.grid((chunk,)),
                                                   pts=pts_t, _result=y_t))
    return out > 0.5


def field_to_grid(model, resolution, bound=1.10):
    """Evaluate the trained field on a dense lattice, ready for marching cubes.

    The SAME network can be sampled at any resolution -- this is where the
    'continuous, resolution-free' claim becomes concrete (Part 5)."""
    xs = np.linspace(-bound, bound, resolution)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    occ = model.occupancy(pts).reshape([resolution] * 3).astype(np.float32)
    origin = np.array([-bound] * 3)
    spacing = np.array([xs[1] - xs[0]] * 3)
    return occ, origin, spacing
