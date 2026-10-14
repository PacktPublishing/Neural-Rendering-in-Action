"""
appearance.py -- driving the appearance fields from Python [Section 6]
============================================================================
Paired with `appearance_train.slang` (both networks + their training and
rendering kernels). Geometry and appearance are two independently trained
fields over the same coordinates: this file only ever trains/evaluates the
COLOUR networks; the geometry network (Section 5's trained SDF) is frozen
and imported unchanged from `neural_sdf.py`. Nothing here samples a
training batch or shades a pixel on the CPU -- sampling, labelling
(`texture_albedo`, `radiance_gt`), sphere tracing and shading are all GPU
dispatches in `appearance_train.slang`.

    AppearanceNetwork   the GPU-resident weights, one per field (albedo,
                        radiance)
    train_albedo        6.2's recipe: a procedural surface texture
    train_radiance      6.3's recipe: view-dependent radiance (+3 columns)
    render_albedo / render_radiance
                        6.2/6.3: sphere-trace the FROZEN geometry field,
                        colour each hit with this field
"""

import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import slangpy_common as sc

import neural_sdf as G   # the frozen geometry field (Section 5)

HERE = Path(__file__).resolve().parent
SHADER = HERE / "appearance_train.slang"
NEURAL_SDF_SHADER = HERE / "neural_sdf.slang"
_NEURAL_SDF_IMPORT_LINE = (
    "__exported import neural_sdf;      "
    "// the FROZEN geometry network, for rendering\n")

HIDDEN = 128         # the book's own width for this section; see README.md
BOUND = G.BOUND

_device = None
_modules = {}


def _module(L, has_view, sdf_L=6, hidden=None, sdf_hidden=None):
    """Compile appearance_train.slang for one (L, has_view, sdf_L, hidden,
    sdf_hidden) combination, once. All five are #defines (not Slang
    generics), prepended to the source text directly -- NOT passed via
    `options={"defines": ...}`: that argument never reaches the Slang
    preprocessor for `Module.load_from_file` (see
    neural_occupancy_fields/README.md). An earlier version of this
    function used it anyway, so every one of these silently stayed at the
    shader's own fallback (APP_L=6, HAS_VIEW=0, hidden=64) no matter what
    was asked for -- HAS_VIEW=0 in particular meant the radiance network
    (has_view=True) was silently trained and rendered as if it never saw
    the view direction at all; view-dependence never worked. Verify with
    a differential test (same point, two different view vectors) after
    touching this, not just that construction/predict don't error --
    SlangPy does not reliably raise on a mismatched generic binding.

    sdf_hidden separately controls the FROZEN geometry Network's width
    (neural_sdf.slang's SDF_HIDDEN). A `#define` alone cannot set it,
    though -- confirmed empirically, not just suspected: compiling with
    SDF_HIDDEN=256 requested but then binding hidden=64-shaped buffers
    (the checkpoint SDF_HIDDEN's own fallback default) rendered
    byte-for-byte identically to a normal hidden=64 run, proving the
    define never reached neural_sdf.slang's `#ifndef SDF_HIDDEN` guard at
    all. Unlike a textual `#include`, Slang's `__exported import` does not
    share the importer's preprocessor macro state with the imported
    translation unit -- only symbols neural_sdf.slang itself DECLARES
    (Network, HIDDEN the const, ...) cross that boundary, not `#define`s
    written before the `import` statement. Binding a hidden=256 geometry
    checkpoint to a struct the compiled kernel still silently thought was
    64-wide is exactly the kind of shape mismatch this module's own
    NUM_ENCODED/HIDDEN vs APP_NUM_ENCODED/APP_HIDDEN naming split was
    already defending against for genuine symbols -- it just didn't cover
    #define-only parameters like SDF_HIDDEN.

    The fix: don't `__exported import neural_sdf;` at all here -- inline
    its actual source text as part of THIS translation unit (see below),
    the same way neural_sdf.py's own `_module()` compiles it directly as
    root. Then the prepended `#define SDF_HIDDEN` lands before
    neural_sdf.slang's own `#ifndef SDF_HIDDEN` guard in the SAME
    preprocessing pass, the ordinary case that guard is for. Duplicate
    `import`s of common_network/common_shapes/common_rng/render (both
    files already import all four) are harmless -- render.slang was
    already being imported twice before this change, transitively via
    `__exported import neural_sdf;` AND directly, and that already
    worked; Slang treats a re-`import` of an already-imported module as a
    no-op, not a redefinition. Every render_* caller below passes
    `geometry.hidden` here."""
    global _device
    if _device is None:
        _device = sc.create_device(__file__)
    if hidden is None:
        hidden = HIDDEN
    if sdf_hidden is None:
        sdf_hidden = G.HIDDEN
    key = (L, bool(has_view), sdf_L, hidden, sdf_hidden)
    if key not in _modules:
        app_src = SHADER.read_text().replace(_NEURAL_SDF_IMPORT_LINE, "")
        assert app_src != SHADER.read_text(), (
            "_NEURAL_SDF_IMPORT_LINE no longer matches appearance_train.slang -- "
            "update it, or SDF_HIDDEN silently stops propagating again.")
        combined = (
            f"#define SDF_L {sdf_L}\n#define SDF_HIDDEN {sdf_hidden}\n"
            f"#define APP_L {L}\n#define HAS_VIEW {1 if has_view else 0}\n"
            f"#define APP_HIDDEN_DEFINE {hidden}\n"
            + NEURAL_SDF_SHADER.read_text() + "\n" + app_src)
        _modules[key] = spy.Module.load_from_source(
            _device,
            f"appearance_L{L}_V{int(bool(has_view))}_S{sdf_L}_H{hidden}_G{sdf_hidden}",
            combined)
    return _modules[key]


def pe_dim(L, has_view):
    return 3 + 6 * L + (3 if has_view else 0)


class AppearanceNetwork(spy.InstanceList):
    """gamma(x)[, v] -> [ReLU hidden] x2 -> rgb, all on the GPU. `has_view`
    picks 6.2's albedo field (False) or 6.3's radiance field (True) -- the
    entire diff between them is three extra input columns."""

    N_LAYERS = 3

    def __init__(self, has_view, L=6, sdf_L=6, hidden=HIDDEN, sdf_hidden=None, seed=0):
        module = _module(L, has_view, sdf_L, hidden, sdf_hidden)
        super().__init__(module["AppearanceNetwork"])
        self.L, self.has_view, self.sdf_L, self.hidden = L, bool(has_view), sdf_L, hidden
        self.sdf_hidden = sdf_hidden if sdf_hidden is not None else G.HIDDEN
        dims = [(pe_dim(L, has_view), hidden), (hidden, hidden), (hidden, 3)]
        self.layer0, self.layer1, self.layer2 = [
            sc.NetworkParameters(module, _device, n_in, n_out, seed + i)
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

    def value(self, pts, views=None, chunk=65536):
        """RGB at each (point[, view]). pts: (N, 3) -> (N, 3)."""
        pts = np.asarray(pts, np.float32)
        views = np.zeros_like(pts) if views is None else np.asarray(views, np.float32)
        if (self._predict_buffers is None or self._predict_buffers[0] != chunk):
            self._predict_buffers = (
                chunk,
                spy.Tensor.empty(_device, (chunk,), "float3"),
                spy.Tensor.empty(_device, (chunk,), "float3"),
                spy.Tensor.empty(_device, (chunk,), "float3"))
        _, pts_t, views_t, result_t = self._predict_buffers
        module = _module(self.L, self.has_view, self.sdf_L, self.hidden, self.sdf_hidden)
        out = np.empty((len(pts), 3), np.float32)
        for i in range(0, len(pts), chunk):
            p_block, v_block = pts[i:i + chunk], views[i:i + chunk]
            n = len(p_block)
            if n < chunk:
                pp = np.zeros((chunk, 3), np.float32); pp[:n] = p_block; p_block = pp
                vv = np.zeros((chunk, 3), np.float32); vv[:n] = v_block; v_block = vv
            pts_t.copy_from_numpy(p_block)
            views_t.copy_from_numpy(v_block)
            module.predict_batch(idx=spy.grid((chunk,)), pts=pts_t, views=views_t,
                                 network=self, _result=result_t)
            out[i:i + n] = result_t.to_numpy()[:n]
        return out

    def save(self, path):
        d = {"L": self.L, "has_view": int(self.has_view), "hidden": self.hidden,
             "n_layers": self.N_LAYERS}
        for i, layer in enumerate(self._layers):
            d[f"W{i}"] = layer.weights.to_numpy().T
            d[f"b{i}"] = layer.biases.to_numpy()
        np.savez(path, **d)

    @classmethod
    def load(cls, path, sdf_L=6, sdf_hidden=None):
        d = np.load(path)
        # Read the width off the weights rather than trusting a stored key:
        # checkpoints written before `hidden` was recorded still load.
        hidden = int(d["hidden"]) if "hidden" in d.files else int(d["W0"].shape[1])
        m = cls(has_view=bool(d["has_view"]), L=int(d["L"]), sdf_L=sdf_L,
               hidden=hidden, sdf_hidden=sdf_hidden)
        for i, layer in enumerate(m._layers):
            w = np.array(d[f"W{i}"].T, np.float32, order="C")   # a real copy -- see neural_sdf.Network.load
            w[np.abs(w) < 1e-20] = 0.0
            layer.weights.copy_from_numpy(w)
            layer.biases.copy_from_numpy(np.ascontiguousarray(d[f"b{i}"], np.float32))
        return m


# ----------------------------------------------------------------------------
# 6.2 / 6.3 -- training. Sampling, labelling, forward, and backward are all
# one GPU dispatch per step (`calculate_albedo_grads` / `_radiance_grads`).
# ----------------------------------------------------------------------------

# Nothing in these loops reads a result back, so nothing forces the queued
# GPU work to drain: left alone, the submitted-but-unretired work grows for
# the whole run and the process eventually segfaults inside Vulkan. It never
# bit at the old 800/1200 step counts; at 4000 it dies reproducibly around
# step ~3800. SYNC_EVERY is the cheap fix -- one wait per 250 dispatched
# steps costs nothing measurable and bounds the backlog.
SYNC_EVERY = 250


def train_albedo(model, steps=800, batch=4096, lr=1e-3, seed=0, verbose=True):
    module = _module(model.L, False, model.sdf_L, model.hidden, model.sdf_hidden)
    grid = spy.grid((batch,))
    model.value(np.zeros((1, 3), np.float32))    # warm up predict_batch
    t0 = time.time()
    for step in range(1, steps + 1):
        module.calculate_albedo_grads(idx=grid, seed=seed * 1_000_003 + step,
                                      n=batch, network=model)
        model.optimize(lr, step)
        if step % SYNC_EVERY == 0:
            _device.wait_for_idle()              # see SYNC_EVERY above
        if verbose and step % 200 == 0:
            print(f"    [albedo]   step {step:4d}  ({time.time() - t0:.0f}s)")
    return {"seconds": time.time() - t0}


def train_radiance(model, steps=1200, batch=4096, lr=1e-3, seed=0, verbose=True):
    module = _module(model.L, True, model.sdf_L, model.hidden, model.sdf_hidden)
    grid = spy.grid((batch,))
    model.value(np.zeros((1, 3), np.float32))    # warm up predict_batch
    t0 = time.time()
    for step in range(1, steps + 1):
        module.calculate_radiance_grads(idx=grid, seed=seed * 1_000_003 + step,
                                        n=batch, network=model)
        model.optimize(lr, step)
        if step % SYNC_EVERY == 0:
            _device.wait_for_idle()              # see SYNC_EVERY above
        if verbose and step % 300 == 0:
            print(f"    [radiance] step {step:4d}  ({time.time() - t0:.0f}s)")
    return {"seconds": time.time() - t0}


# ----------------------------------------------------------------------------
# 6.2 / 6.3 -- rendering: sphere-trace the FROZEN geometry field, colour
# each hit with this (trained) appearance field. Not a geometry weight is
# touched.
# ----------------------------------------------------------------------------

def render_albedo(geometry, appearance, eye=(1.9, -1.9, 1.5), width=300, height=225,
                  bg_color=(1.0, 1.0, 1.0)):
    module = _module(appearance.L, False, geometry.L, appearance.hidden, geometry.hidden)
    eye_v, fwd, right, up, half_h = G._camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_albedo(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, geometry=geometry, appearance=appearance,
        bg_color=list(bg_color), _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


def render_radiance(geometry, appearance, eye=(1.9, -1.9, 1.5), width=300, height=225,
                    bg_color=(1.0, 1.0, 1.0)):
    module = _module(appearance.L, True, geometry.L, appearance.hidden, geometry.hidden)
    eye_v, fwd, right, up, half_h = G._camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_radiance(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, geometry=geometry, appearance=appearance,
        bg_color=list(bg_color), _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


def render_albedo_exact(geometry, eye=(1.9, -1.9, 1.5), width=300, height=225,
                        bg_color=(1.0, 1.0, 1.0), L=6):
    """Ground truth: `texture_albedo` straight from the shader, no network --
    for Figure 6.1's comparison panel and PSNR."""
    module = _module(L, False, geometry.L, HIDDEN, geometry.hidden)
    eye_v, fwd, right, up, half_h = G._camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_albedo_exact(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, geometry=geometry, bg_color=list(bg_color), _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)


def render_radiance_exact(geometry, eye=(1.9, -1.9, 1.5), width=300, height=225,
                          bg_color=(1.0, 1.0, 1.0), L=6):
    """Ground truth: `radiance_gt` straight from the shader, no network --
    for Figure 6.2's comparison rows and PSNR."""
    module = _module(L, True, geometry.L, HIDDEN, geometry.hidden)
    eye_v, fwd, right, up, half_h = G._camera_basis(eye)
    n_px = width * height
    img_t = spy.Tensor.empty(_device, (n_px,), "float3")
    module.render_radiance_exact(
        idx=spy.grid((n_px,)), width=width, height=height,
        eye=eye_v.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(),
        half_h=half_h, geometry=geometry, bg_color=list(bg_color), _result=img_t)
    return np.clip(img_t.to_numpy().reshape(height, width, 3), 0, 1)
