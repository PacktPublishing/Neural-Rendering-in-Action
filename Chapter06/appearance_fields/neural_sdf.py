"""
neural_sdf.py -- driving the signed-distance field from Python [Section 5]
============================================================================
Paired with `neural_sdf.slang` (the network + its training kernels) and
`render.slang` (the sphere-tracing renderer). There is one network in this
section and it lives entirely on the GPU: this file only orchestrates
dispatches -- it never holds a NumPy mirror of the weights, never samples a
training batch on the CPU, and never runs the renderer's per-pixel loop
itself. `common_shapes.slang` supplies the smooth and knurled torus
math every kernel needs; `common_rng.slang` supplies GPU-side randomness, so
every point AND its label is generated on the GPU and never leaves it until
a checkpoint or a finished image is downloaded.

    Network         the GPU-resident weights (spy.InstanceList)
    train_supervised   5.2's regression recipe: exact distances, clamped-L1
    train_eikonal      5.5's recipe: surface points alone, IGR-style losses
    render             5.4: sphere-trace + shade a full image on the GPU
"""

import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import slangpy_common as sc

HERE = Path(__file__).resolve().parent
SHADER = HERE / "neural_sdf.slang"

HIDDEN = 64          # book: 256; 64 keeps eikonal training fast -- see README.md
BOUND = 1.10

_device = None
_modules = {}


def _module(L, hidden):
    """Compile neural_sdf.slang for one (L, hidden) pair, once. Both are
    #defines (not Slang generics -- a generic Network<let L: int> did not
    survive bwd_diff on this toolchain) prepended to the source text
    directly, NOT passed via `options={"defines": ...}`: that argument
    never reaches the Slang preprocessor for `Module.load_from_file` (see
    neural_occupancy_fields/README.md) -- an earlier version of this
    function used it anyway, so L silently stayed at the file's own
    fallback (6) no matter what was asked for. Harmless only because every
    caller in this chapter always asks for L=6."""
    global _device
    if _device is None:
        _device = sc.create_device(__file__)
    key = (L, hidden)
    if key not in _modules:
        _modules[key] = spy.Module.load_from_source(
            _device, f"neural_sdf_L{L}_H{hidden}",
            f"#define SDF_L {L}\n#define SDF_HIDDEN {hidden}\n"
            + SHADER.read_text())
    return _modules[key]


def pe_dim(L):
    return 3 + 6 * L


class Network(spy.InstanceList):
    """gamma(x) -> [ReLU hidden] x2 -> signed distance (linear output), all
    on the GPU. Weights live in NetworkParameters GPU buffers for the whole
    lifetime of this object -- there is no CPU-side mirror to keep in sync."""

    N_LAYERS = 3

    def __init__(self, L=6, hidden=HIDDEN, seed=0):
        module = _module(L, hidden)
        super().__init__(module["Network"])
        self.L, self.hidden = L, hidden
        dims = [(pe_dim(L), hidden), (hidden, hidden), (hidden, 1)]
        self.layer0, self.layer1, self.layer2 = [
            sc.NetworkParameters(module, _device, n_in, n_out, seed + i,
                                 out_scale=(0.1 if i == len(dims) - 1 else 1.0))
            for i, (n_in, n_out) in enumerate(dims)]
        self._predict_buffers = None

    @property
    def _layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def n_params(self):
        return sum(l.n_params() for l in self._layers)

    def memory_bytes(self):
        return self.n_params() * 4

    def optimize(self, lr, step):
        for layer in self._layers:
            layer.optimize(lr, step)

    def value(self, pts, chunk=65536):
        """Signed distance at each point. pts: (N, 3) -> (N,)."""
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
        # checkpoints written before `hidden` was recorded still load.
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
# 5.2 -- supervised regression: exact distances, clamped-L1
# ----------------------------------------------------------------------------
# Sampling AND labelling happen inside `calculate_supervised_grads` itself
# (GPU RNG + `common_shapes.torus_sdf`): the training loop here only ever
# passes a seed and a few scalars, never a batch of points.

def train_supervised(model, steps=1500, batch=4096, lr=1e-3, delta=0.10,
                     beta_far=0.2, jitter=0.08, near_frac=0.5, seed=0,
                     verbose=True):
    module = _module(model.L, model.hidden)
    grid = spy.grid((batch,))
    n_uniform = batch - int(batch * near_frac)
    val_t = spy.Tensor.empty(_device, (batch,), "float3")   # warm-up buffer
    model.value(np.zeros((1, 3), np.float32))                # warm up predict_batch

    history = {"loss": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        rng_seed = seed * 1_000_003 + step
        module.calculate_supervised_grads(
            idx=grid, seed=rng_seed, n_uniform=n_uniform, jitter=jitter,
            delta=delta, beta_far=beta_far, network=model)
        model.optimize(lr, step)
        if verbose and step % 500 == 0:
            print(f"    step {step:5d}  ({time.time() - t0:.1f}s)")
    history["seconds"] = time.time() - t0
    return history


# ----------------------------------------------------------------------------
# 5.5 -- eikonal: an SDF from a surface point cloud alone
# ----------------------------------------------------------------------------

def make_surface_cloud(module, n_points, seed=0):
    """Generate the fixed 'laser scan' point cloud once, on the GPU, via
    `common_shapes.detailed_torus_surface` -- the knurled torus, whose exact
    SDF has no closed form (that gap is this recipe's whole point)."""
    cloud_t = spy.Tensor.empty(_device, (n_points,), "float3")
    module.sample_surface_cloud(idx=spy.grid((n_points,)), seed=seed, _result=cloud_t)
    return cloud_t


def train_eikonal(model, cloud_t, steps=2700, batch_s=1024, batch_u=1024,
                  lr=1e-3, lam_eik=0.1, lam_rep=0.02, alpha=50.0, fd_h=5e-3,
                  jitter=0.002, warmup=200, seed=0, verbose=True,
                  checkpoint_every=900, on_checkpoint=None):
    """Learn an SDF from SURFACE POINTS ALONE (IGR-style): a surface term
    (sticks to zero, `calculate_surface_grads`) and an eikonal + repulsion
    term over 'space' points (`calculate_eikonal_grads`, seven GPU threads
    per logical point -- see the note at the top of neural_sdf.slang) -- see
    README.md. Every term's sampling, its finite-difference gradient, and
    its backward pass are GPU dispatches; `warmup` (only meaningful when the
    caller passes a freshly-initialized `model`) is the geometric warm-up
    toward a sphere SDF that gives the eikonal loss one clean inside region
    to deform."""
    module = _module(model.L, model.hidden)
    n_cloud = cloud_t.shape[0]
    grid_s = spy.grid((batch_s,))
    grid_u7 = spy.grid((batch_u * 7,))
    n_warmup = 2048
    grid_warmup = spy.grid((n_warmup,))
    model.value(np.zeros((1, 3), np.float32))                # warm up predict_batch

    t0 = time.time()
    for step in range(1, warmup + 1):
        module.calculate_warmup_grads(idx=grid_warmup, seed=seed * 7919 + step,
                                      n_warmup=n_warmup, network=model)
        model.optimize(lr, step)
    if warmup and verbose:
        print(f"    warm-up done ({time.time() - t0:.0f}s)")

    history = {"step": []}
    for step in range(1, steps + 1):
        rng_seed = seed * 1_000_003 + step
        module.calculate_surface_grads(
            idx=grid_s, seed=rng_seed, cloud=cloud_t, n_cloud=n_cloud,
            jitter=jitter, n_surface=batch_s, network=model)
        module.calculate_eikonal_grads(
            idx7=grid_u7, seed=rng_seed, cloud=cloud_t, n_cloud=n_cloud,
            n_space=batch_u, lam_eik=lam_eik, lam_rep=lam_rep, alpha=alpha,
            fd_h=fd_h, network=model)
        model.optimize(lr, step)
        history["step"].append(step)
        if step % checkpoint_every == 0 or step == steps:
            if verbose:
                print(f"    step {step:5d}  ({time.time() - t0:.0f}s)")
            if on_checkpoint is not None:
                on_checkpoint(step, model)
    history["seconds"] = time.time() - t0
    return history


# ----------------------------------------------------------------------------
# 5.4 -- sphere-traced rendering, straight from the GPU network
# ----------------------------------------------------------------------------

def _camera_basis(eye, target=(0, 0, 0), up=(0, 0, 1), fov=40.0):
    eye = np.asarray(eye, np.float32); target = np.asarray(target, np.float32)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    half_h = float(np.tan(np.radians(fov) / 2))
    return eye, fwd, right, true_up, half_h


def render(model, eye=(1.9, -1.9, 1.5), width=300, height=225,
          light=(0.5, -0.3, 0.85), base_color=(0.42, 0.6, 0.8),
          bg_color=(1.0, 1.0, 1.0)):
    """Sphere-trace and shade a full image against the trained field,
    entirely on the GPU (`render.slang`'s render_neural): march, normal,
    Lambert shade and background composite all happen per-pixel in one
    kernel. Returns an (H, W, 3) float32 image."""
    module = _module(model.L, model.hidden)
    eye_v, fwd, right, up, half_h = _camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_neural(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, light=list(light), base_color=list(base_color),
        bg_color=list(bg_color), network=model, _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


def render_reference(eye=(1.9, -1.9, 1.5), width=300, height=225,
                     light=(0.5, -0.3, 0.85), base_color=(0.42, 0.6, 0.8),
                     bg_color=(1.0, 1.0, 1.0), L=6):
    """The analytic smooth-torus reference render, same kernel family, no
    network involved -- `render.slang`'s render_analytic."""
    module = _module(L, HIDDEN)
    eye_v, fwd, right, up, half_h = _camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_analytic(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, light=list(light), base_color=list(base_color),
        bg_color=list(bg_color), _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


# ----------------------------------------------------------------------------
# 5.6 -- CSG: smooth union of the trained field and an analytic sphere
# ----------------------------------------------------------------------------

def render_smooth_union(model, k, sphere_center=(0.55, 0.0, 0.42), sphere_r=0.34,
                        eye=(1.9, -1.9, 1.5), width=300, height=225,
                        light=(0.5, -0.3, 0.85), base_color=(0.78, 0.55, 0.38),
                        bg_color=(1.0, 1.0, 1.0)):
    """smooth_union(trained_field, analytic_sphere) sphere-traced directly --
    `render.slang`'s render_smooth_union kernel; no weight is touched."""
    module = _module(model.L, model.hidden)
    eye_v, fwd, right, up, half_h = _camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_smooth_union(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, light=list(light), base_color=list(base_color),
        bg_color=list(bg_color), network=model, k=k,
        sphere_center=list(sphere_center), sphere_r=sphere_r, _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


# ----------------------------------------------------------------------------
# 5.3 -- normals as an RGB image (visualizing the field's own gradient)
# ----------------------------------------------------------------------------

def knurl_grid(resolution, bound=BOUND, L=6):
    """Evaluate the exact knurled-torus inside test on a dense lattice, on
    the GPU (`common_shapes.detailed_torus_grid_value`) -- used only to mesh
    the GROUND TRUTH for Figure 5.3's comparison panel via marching cubes."""
    module = _module(L, HIDDEN)
    spacing = 2 * bound / (resolution - 1)
    origin = np.array([-bound] * 3, np.float32)
    n = resolution ** 3
    grid_t = spy.Tensor.empty(_device, (n,), "float")
    module.detailed_torus_grid_value(
        idx=spy.grid((n,)), origin=origin.tolist(),
        spacing=[spacing, spacing, spacing], res=resolution, _result=grid_t)
    occ = grid_t.to_numpy().reshape([resolution] * 3)
    return occ, origin, np.array([spacing] * 3, np.float32)


def sdf_grid(model, resolution, bound=BOUND):
    """Evaluate the trained field on a dense lattice, ready for marching
    cubes at level 0 (a signed distance's zero level set is its surface)."""
    xs = np.linspace(-bound, bound, resolution)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    val = model.value(pts).reshape([resolution] * 3).astype(np.float32)
    origin = np.array([-bound] * 3)
    spacing = np.array([xs[1] - xs[0]] * 3)
    return val, origin, spacing


def render_normals(model, eye=(1.9, -1.9, 1.5), width=300, height=225):
    module = _module(model.L, model.hidden)
    eye_v, fwd, right, up, half_h = _camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_neural_normals(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, network=model, _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)
