"""
train_sdf_slangpy.py -- SlangPy port of sdf5.py's train_supervised() (5.2)
============================================================================
Trains the same coordinate MLP as sdf5.SDFNet, on the GPU via SlangPy
(sdf_train.slang) instead of sdf5.py's hand-rolled NumPy backprop, and saves
a checkpoint in exactly the format make_figures5.py's load_sdfnet() expects
(L, W0..W2, b0..b2). Run this, then run make_figures5.py unchanged -- it
will pick up the new checkpoint and render Figure 5.1/5.2 using the
existing, untouched NumPy sphere-tracing/rendering code.

Not ported: train_eikonal() (5.5), which trains from a point cloud alone
using a finite-difference second-order gradient (six shifted forward passes
per step, each individually backpropagated). That's a materially bigger
Slang kernel than this one for a section that's already deviated from the
book (see Section4/README.md's list of compile-time workarounds this
toolchain needed) -- ckpt_sdf_eikonal.npz is left as the NumPy-trained
checkpoint that ships with the section.

Run:  python train_sdf_slangpy.py [steps=1500] [L=6]
"""

import sys
import time
import numpy as np
import slangpy as spy
from pathlib import Path

import sdf5 as S  # torus_sdf, sample geometry -- reused unchanged

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6
HIDDEN, BATCH, LR = (64, 64), 4096, 1e-3
BETA_FAR = 0.2
DELTA = 0.10
BOUND = S.BOUND

device = spy.create_device(spy.DeviceType.vulkan, enable_debug_layers=False,
                            include_paths=[Path(__file__).parent])
module = spy.Module.load_from_file(device, "sdf_train.slang",
                                   options={"defines": {"SDF_L": str(L)}})


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int, seed: int = 0, out_scale: float = 1.0, out_bias: float = 0.0):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        rng = np.random.default_rng(seed)
        biases = np.full(outputs, out_bias, np.float32)
        weights = (rng.standard_normal((outputs, inputs)) * np.sqrt(2 / inputs) * out_scale).astype(np.float32)
        self.biases = spy.Tensor.from_numpy(device, biases)
        self.weights = spy.Tensor.from_numpy(device, weights)
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, lr, i):
        module.optimizer_step(self.biases, self.biases_grad, self.m_biases, self.v_biases, lr, i)
        module.optimizer_step(self.weights, self.weights_grad, self.m_weights, self.v_weights, lr, i)

    def to_numpy(self):
        return self.weights.to_numpy(), self.biases.to_numpy()


class Network(spy.InstanceList):
    def __init__(self, L, hidden=HIDDEN, seed=0):
        super().__init__(module["Network"])
        n_in = 3 + 6 * L
        # DeepSDF-style init (sdf5.SDFNet.__init__): shrink the last layer so
        # predictions start inside the clamp band, or the clamped-L1 gradient
        # is zero everywhere and training never moves (see section5.md).
        self.layer0 = NetworkParameters(n_in, hidden[0], seed)
        self.layer1 = NetworkParameters(hidden[0], hidden[1], seed + 1)
        self.layer2 = NetworkParameters(hidden[1], 1, seed + 2, out_scale=0.1)

    def optimize(self, lr, i):
        self.layer0.optimize(lr, i)
        self.layer1.optimize(lr, i)
        self.layer2.optimize(lr, i)

    def predict(self, pts, chunk=65536):
        out = np.empty(len(pts), np.float32)
        if not hasattr(self, "_pred_pts_t") or self._pred_chunk != chunk:
            self._pred_pts_t = spy.Tensor.empty(device, (chunk,), "float3")
            self._pred_result_t = spy.Tensor.empty(device, (chunk,), "float")
            self._pred_chunk = chunk
        pts_t, result_t = self._pred_pts_t, self._pred_result_t
        for i in range(0, len(pts), chunk):
            block = pts[i:i + chunk].astype(np.float32)
            n = len(block)
            if n < chunk:
                padded = np.zeros((chunk, 3), np.float32)
                padded[:n] = block
                block = padded
            pts_t.copy_from_numpy(block)
            module.predict_batch(idx=spy.grid((chunk,)), pts=pts_t,
                                 network=self, _result=result_t)
            out[i:i + chunk] = result_t.to_numpy()[:n]
        return out


def sample_sdf_batch(n, rng, delta_band=0.08):
    half = n // 2
    u = rng.uniform(0, 2 * np.pi, half)
    v = rng.uniform(0, 2 * np.pi, half)
    surf = np.stack([(0.65 + 0.28 * np.cos(v)) * np.cos(u),
                     (0.65 + 0.28 * np.cos(v)) * np.sin(u),
                     0.28 * np.sin(v)], 1)
    pts = np.vstack([rng.uniform(-BOUND, BOUND, (n - half, 3)),
                     surf + delta_band * rng.standard_normal((half, 3))])
    return pts.astype(np.float32), S.torus_sdf(pts).astype(np.float32)


def clamped_l1(pred, gt, delta=DELTA):
    cp, cg = np.clip(pred, -delta, delta), np.clip(gt, -delta, delta)
    diff = cp - cg
    grad = np.sign(diff) * (np.abs(pred) < delta)
    return np.abs(diff).mean(), grad.astype(np.float32) / len(pred)


def train():
    net = Network(L)
    rng = np.random.default_rng(0)
    pts_t = spy.Tensor.empty(device, (BATCH,), "float3")
    dpred_t = spy.Tensor.empty(device, (BATCH,), "float")
    batch_grid = spy.grid((BATCH,))
    net.predict(np.zeros((1, 3), np.float32))  # warm up predict_batch's kernel first
    t0 = time.time()
    for step in range(1, STEPS + 1):
        pts, gt = sample_sdf_batch(BATCH, rng)
        pts_t.copy_from_numpy(pts)
        pred = net.predict(pts)
        loss_c, d_c = clamped_l1(pred, gt)
        d_far = np.sign(pred - gt).astype(np.float32) / len(pred)
        dpred_t.copy_from_numpy((d_c + BETA_FAR * d_far).astype(np.float32))
        module.calculate_grads_from_dpred(idx=batch_grid, pts=pts_t, dpred=dpred_t, network=net)
        net.optimize(LR, step)
        if step % 500 == 0 or step == STEPS:
            print(f"  step {step:5d}  clamped-L1 {loss_c:.4f} "
                  f" plain-L1 {np.abs(pred - gt).mean():.4f} "
                  f" ({time.time()-t0:.0f}s)")
    return net


if __name__ == "__main__":
    print(f"SDF regression (SlangPy) | L = {L}, hidden = {HIDDEN}, steps = {STEPS}")
    net = train()

    rng = np.random.default_rng(1)
    P = rng.uniform(-BOUND, BOUND, (200_000, 3)).astype(np.float32)
    pred = net.predict(P)
    gt = S.torus_sdf(P)
    l1 = np.abs(pred - gt).mean()
    iou = np.logical_and(pred < 0, gt < 0).sum() / np.logical_or(pred < 0, gt < 0).sum()
    print(f"  IoU vs ground truth: {iou:.3f}   mean |L1| error: {l1:.4f}")

    # sdf5.SDFNet stores W[layer] as (inputs, outputs) -- so that forward()
    # can do `a @ W + b` -- while the Slang side stores it as (outputs,
    # inputs) to match get_weight(neuron, input)'s indexing (same convention
    # Chapter 4 uses). Transpose on the way out.
    weights = [net.layer0.weights.to_numpy().T, net.layer1.weights.to_numpy().T, net.layer2.weights.to_numpy().T]
    biases = [net.layer0.biases.to_numpy(), net.layer1.biases.to_numpy(), net.layer2.biases.to_numpy()]
    out = "ckpt_sdf_supervised.npz"
    np.savez(out, L=L, W0=weights[0], W1=weights[1], W2=weights[2],
             b0=biases[0], b1=biases[1], b2=biases[2])
    print(f"  saved {out} -- run make_figures5.py to render Figure 5.1/5.2 from it")
