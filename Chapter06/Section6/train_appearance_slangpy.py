"""
train_appearance_slangpy.py -- SlangPy port of app6.py's train_albedo_field()
and train_radiance_field() (6.2, 6.3)
============================================================================
Trains the same two MLPs as app6.py -- c(x) -> rgb and c(x,v) -> rgb -- on
the GPU via SlangPy (appearance_train.slang) instead of app6.py's hand-rolled
NumPy backprop, and saves a checkpoint in exactly the format
make_figures6.py's load_appearance() expects (aW0..aW2, ab0..ab2, rW0..rW2,
rb0..rb2). Run this, then run make_figures6.py unchanged -- it will pick up
the new checkpoint and render Figure 6.1/6.2 using the existing, untouched
NumPy sphere-tracing/rendering code. Figure 6.3 (the pure-vs-hybrid speed
race) is a deliberate NumPy wall-clock comparison and is not affected by
this script at all.

Run:  python train_appearance_slangpy.py [steps=800] [radiance_steps=1200]
"""

import sys
import time
import numpy as np
import slangpy as spy
from pathlib import Path

import app6 as A  # torus_surface, torus_normal, texture_albedo, radiance_gt, LIGHT

ALBEDO_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
RADIANCE_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
L = 6
HIDDEN, BATCH, LR = (64, 64), 4096, 1e-3

device = spy.create_device(spy.DeviceType.vulkan, enable_debug_layers=False,
                            include_paths=[Path(__file__).parent])
module_albedo = spy.Module.load_from_file(
    device, "appearance_train.slang", options={"defines": {"APP_L": str(L), "HAS_VIEW": "0"}})
module_radiance = spy.Module.load_from_file(
    device, "appearance_train.slang", options={"defines": {"APP_L": str(L), "HAS_VIEW": "1"}})


class NetworkParameters(spy.InstanceList):
    def __init__(self, module, inputs: int, outputs: int, seed: int = 0):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        rng = np.random.default_rng(seed)
        self.biases = spy.Tensor.from_numpy(device, np.zeros(outputs, np.float32))
        self.weights = spy.Tensor.from_numpy(
            device, (rng.standard_normal((outputs, inputs)) * np.sqrt(2 / inputs)).astype(np.float32))
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, module, lr, i):
        module.optimizer_step(self.biases, self.biases_grad, self.m_biases, self.v_biases, lr, i)
        module.optimizer_step(self.weights, self.weights_grad, self.m_weights, self.v_weights, lr, i)


class Network(spy.InstanceList):
    def __init__(self, module, n_in, hidden=HIDDEN, seed=0):
        super().__init__(module["Network"])
        self.module = module
        self.layer0 = NetworkParameters(module, n_in, hidden[0], seed)
        self.layer1 = NetworkParameters(module, hidden[0], hidden[1], seed + 1)
        self.layer2 = NetworkParameters(module, hidden[1], 3, seed + 2)

    def optimize(self, lr, i):
        self.layer0.optimize(self.module, lr, i)
        self.layer1.optimize(self.module, lr, i)
        self.layer2.optimize(self.module, lr, i)

    def predict(self, pts, views, chunk=65536):
        out = np.empty((len(pts), 3), np.float32)
        if not hasattr(self, "_pred_pts_t") or self._pred_chunk != chunk:
            self._pred_pts_t = spy.Tensor.empty(device, (chunk,), "float3")
            self._pred_views_t = spy.Tensor.empty(device, (chunk,), "float3")
            self._pred_result_t = spy.Tensor.empty(device, (chunk,), "float3")
            self._pred_chunk = chunk
        pts_t, views_t, result_t = self._pred_pts_t, self._pred_views_t, self._pred_result_t
        for i in range(0, len(pts), chunk):
            p_block = pts[i:i + chunk].astype(np.float32)
            v_block = views[i:i + chunk].astype(np.float32)
            n = len(p_block)
            if n < chunk:
                pp = np.zeros((chunk, 3), np.float32); pp[:n] = p_block; p_block = pp
                vv = np.zeros((chunk, 3), np.float32); vv[:n] = v_block; v_block = vv
            pts_t.copy_from_numpy(p_block)
            views_t.copy_from_numpy(v_block)
            self.module.predict_batch(idx=spy.grid((chunk,)), pts=pts_t, views=views_t,
                                      network=self, _result=result_t)
            out[i:i + chunk] = result_t.to_numpy()[:n]
        return out

    def weights_biases(self):
        return ([self.layer0.weights.to_numpy().T, self.layer1.weights.to_numpy().T,
                 self.layer2.weights.to_numpy().T],
                [self.layer0.biases.to_numpy(), self.layer1.biases.to_numpy(),
                 self.layer2.biases.to_numpy()])


def train_albedo(net, steps, pts_t, zero_views_t, dpred_t, batch_grid):
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        p = A.torus_surface(BATCH, rng)
        pts_t.copy_from_numpy(p)
        pred = net.predict(p, np.zeros((BATCH, 3), np.float32))
        gt = A.texture_albedo(p)
        mse = ((pred - gt) ** 2).mean()
        dpred_t.copy_from_numpy((2 * (pred - gt) / pred.size).astype(np.float32))
        module_albedo.calculate_grads_from_dpred(idx=batch_grid, pts=pts_t, views=zero_views_t,
                                                  dpred=dpred_t, network=net)
        net.optimize(LR, step)
        if step % 200 == 0 or step == steps:
            print(f"  [albedo]   step {step:4d}  MSE {mse:.5f}  ({time.time()-t0:.0f}s)")
    return net


def train_radiance(net, steps, pts_t, views_t, dpred_t, batch_grid):
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        p = A.torus_surface(BATCH, rng)
        v = rng.standard_normal((BATCH, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[(v * A.torus_normal(p)).sum(1) < 0] *= -1  # outward hemisphere
        pts_t.copy_from_numpy(p)
        views_t.copy_from_numpy(v)
        pred = net.predict(p, v)
        gt = A.radiance_gt(p, v)
        mse = ((pred - gt) ** 2).mean()
        dpred_t.copy_from_numpy((2 * (pred - gt) / pred.size).astype(np.float32))
        module_radiance.calculate_grads_from_dpred(idx=batch_grid, pts=pts_t, views=views_t,
                                                    dpred=dpred_t, network=net)
        net.optimize(LR, step)
        if step % 300 == 0 or step == steps:
            print(f"  [radiance] step {step:4d}  MSE {mse:.5f}  ({time.time()-t0:.0f}s)")
    return net


if __name__ == "__main__":
    print(f"appearance fields (SlangPy) | L = {L}, hidden = {HIDDEN}")
    # Construct both networks, warm up both modules' kernels, and allocate
    # every GPU buffer either training loop will use, ALL before either loop
    # actually runs. Any single new allocation made only after hundreds of
    # prior dispatches had already gone out intermittently failed to create
    # GPU resources (createBuffer / SLANG_FAIL) on this device/driver -- see
    # Section4/README.md. Front-loading every allocation sidesteps it.
    alb = Network(module_albedo, 3 + 6 * L)
    rad = Network(module_radiance, 3 + 6 * L + 3)
    alb.predict(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
    rad.predict(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))

    alb_pts_t = spy.Tensor.empty(device, (BATCH,), "float3")
    alb_zero_views_t = spy.Tensor.empty(device, (BATCH,), "float3")
    alb_zero_views_t.copy_from_numpy(np.zeros((BATCH, 3), np.float32))
    alb_dpred_t = spy.Tensor.empty(device, (BATCH,), "float3")
    rad_pts_t = spy.Tensor.empty(device, (BATCH,), "float3")
    rad_views_t = spy.Tensor.empty(device, (BATCH,), "float3")
    rad_dpred_t = spy.Tensor.empty(device, (BATCH,), "float3")
    batch_grid = spy.grid((BATCH,))

    print("training albedo field c(x) -> rgb ...")
    alb = train_albedo(alb, ALBEDO_STEPS, alb_pts_t, alb_zero_views_t, alb_dpred_t, batch_grid)
    print("training radiance field c(x, v) -> rgb ...")
    rad = train_radiance(rad, RADIANCE_STEPS, rad_pts_t, rad_views_t, rad_dpred_t, batch_grid)

    aW, ab = alb.weights_biases()
    rW, rb = rad.weights_biases()
    out = "ckpt_appearance.npz"
    np.savez(out,
             aW0=aW[0], aW1=aW[1], aW2=aW[2], ab0=ab[0], ab1=ab[1], ab2=ab[2],
             rW0=rW[0], rW1=rW[1], rW2=rW[2], rb0=rb[0], rb1=rb[1], rb2=rb[2])
    print(f"  saved {out} -- run make_figures6.py to render Figure 6.1/6.2 from it")
