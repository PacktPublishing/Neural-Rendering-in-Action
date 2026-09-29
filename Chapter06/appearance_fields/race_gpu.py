"""
race_gpu.py -- Figure 6.3, the race: pure MLP vs feature-grid hybrid
============================================================================
Both racers train on the knurled-torus occupancy task, entirely on the GPU
(`hybrid_race_train.slang`): sampling, labelling, forward and backward are
all one dispatch per step, same as every other trainer in this chapter.
Each side trains for a fixed WALL-CLOCK budget (not fixed steps), recording
a (time, batch-accuracy) curve so the two architectures race fairly
regardless of how many steps each completes.

Run:  python race_gpu.py [budget_seconds=40] [L=6]   # writes checkpoints/race_results.npz
"""

import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import slangpy_common as sc

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6
HIDDEN, BATCH, LR = 64, 4096, 1e-3
GRID_RES, GRID_FEAT, HEAD_HIDDEN = 32, 4, 32
BOUND = 1.10
NEAR_FRAC, JITTER = 0.5, 0.05
CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

device = sc.create_device(__file__)
module = spy.Module.load_from_file(device, "hybrid_race_train.slang",
                                   options={"defines": {"APP_L": str(L)}})


class PureNet(spy.InstanceList):
    def __init__(self, seed=0):
        super().__init__(module["PureNetwork"])
        n_in = 3 + 6 * L
        self.layer0 = sc.NetworkParameters(module, device, n_in, HIDDEN, seed)
        self.layer1 = sc.NetworkParameters(module, device, HIDDEN, HIDDEN, seed + 1)
        self.layer2 = sc.NetworkParameters(module, device, HIDDEN, 1, seed + 2)

    def optimize(self, lr, i):
        self.layer0.optimize(lr, i); self.layer1.optimize(lr, i); self.layer2.optimize(lr, i)

    def n_params(self):
        return self.layer0.n_params() + self.layer1.n_params() + self.layer2.n_params()

    def predict(self, pts_t, result_t, chunk):
        module.pure_predict_batch(idx=spy.grid((chunk,)), pts=pts_t, network=self, _result=result_t)


class FeatureGrid(spy.InstanceList):
    def __init__(self, seed=0):
        super().__init__(module["FeatureGrid"])
        rng = np.random.default_rng(seed)
        cells = (0.1 * rng.standard_normal(GRID_RES ** 3 * GRID_FEAT)).astype(np.float32)
        self.cells = spy.Tensor.from_numpy(device, cells)
        self.cells_grad = spy.Tensor.zeros_like(self.cells)
        self.m_cells = spy.Tensor.zeros_like(self.cells)
        self.v_cells = spy.Tensor.zeros_like(self.cells)

    def optimize(self, lr, i):
        module.optimizer_step(self.cells, self.cells_grad, self.m_cells, self.v_cells, lr, i)


class HybridNet(spy.InstanceList):
    def __init__(self, seed=0):
        super().__init__(module["HybridNetwork"])
        self.grid = FeatureGrid(seed)
        self.head0 = sc.NetworkParameters(module, device, GRID_FEAT + 3, HEAD_HIDDEN, seed + 1)
        self.head1 = sc.NetworkParameters(module, device, HEAD_HIDDEN, HEAD_HIDDEN, seed + 2)
        self.head2 = sc.NetworkParameters(module, device, HEAD_HIDDEN, 1, seed + 3)

    def optimize(self, lr, i):
        self.grid.optimize(lr, i)
        self.head0.optimize(lr, i); self.head1.optimize(lr, i); self.head2.optimize(lr, i)

    def n_params(self):
        return (GRID_RES ** 3 * GRID_FEAT + self.head0.n_params()
               + self.head1.n_params() + self.head2.n_params())

    def predict(self, pts_t, result_t, chunk):
        module.hybrid_predict_batch(idx=spy.grid((chunk,)), pts=pts_t, network=self,
                                    bound=BOUND, _result=result_t)


def race_one(kind, net, seconds, seed=0, tag=""):
    pts_t = spy.Tensor.empty(device, (BATCH,), "float3")
    y_t = spy.Tensor.empty(device, (BATCH,), "float")
    result_t = spy.Tensor.empty(device, (BATCH,), "float")
    grid = spy.grid((BATCH,))
    n_uniform = BATCH - int(BATCH * NEAR_FRAC)

    net.predict(pts_t, result_t, BATCH)   # warm up predict's kernel first

    # Warm up calculate_grads too, with one real dispatch OUTSIDE the timed
    # window: the first call to a differentiable kernel pays a one-time
    # compile/specialization cost (seconds, not milliseconds), which would
    # otherwise swallow most of a short race budget and badly skew
    # steps/sec for the model warmed up first.
    if kind == "pure":
        module.pure_calculate_grads(idx=grid, seed=seed, n_uniform=n_uniform, jitter=JITTER, network=net)
    else:
        module.hybrid_calculate_grads(idx=grid, seed=seed, n_uniform=n_uniform, jitter=JITTER,
                                      network=net, bound=BOUND)
    net.optimize(LR, 1)

    curve, t0, step = [], time.time(), 1
    while time.time() - t0 < seconds:
        step += 1
        rng_seed = seed * 1_000_003 + step
        if kind == "pure":
            module.pure_calculate_grads(idx=grid, seed=rng_seed, n_uniform=n_uniform,
                                        jitter=JITTER, network=net)
        else:
            module.hybrid_calculate_grads(idx=grid, seed=rng_seed, n_uniform=n_uniform,
                                          jitter=JITTER, network=net, bound=BOUND)
        net.optimize(LR, step)
        if step % 20 == 0:
            module.sample_occ_point(idx=grid, seed=rng_seed, n_uniform=n_uniform,
                                    jitter=JITTER, _result=pts_t)
            module.detailed_torus_inside_batch(idx=grid, pts=pts_t, _result=y_t)
            net.predict(pts_t, result_t, BATCH)
            p, y = result_t.to_numpy(), y_t.to_numpy()
            acc = float(((p > 0.5) == (y > 0.5)).mean())
            curve.append((time.time() - t0, acc))
    print(f"  {tag}: {step} steps in {seconds:.0f}s ({1000*seconds/step:.2f} ms/step)")
    return np.array(curve), step


if __name__ == "__main__":
    print(f"GPU race (Figure 6.3) | L = {L}, hidden = {HIDDEN}, budget = {BUDGET}s/model")
    pure = PureNet(seed=0)
    hyb = HybridNet(seed=0)
    print(f"  params: pure {pure.n_params():,} | hybrid {hyb.n_params():,}")

    c_pure, s_pure = race_one("pure", pure, BUDGET, tag="pure MLP  ")
    c_hyb, s_hyb = race_one("hybrid", hyb, BUDGET, tag="hybrid    ")

    P = np.random.default_rng(9).uniform(-BOUND, BOUND, (200_000, 3)).astype(np.float32)
    P_t = spy.Tensor.empty(device, (65536,), "float3")
    gt_t = spy.Tensor.empty(device, (65536,), "float")
    pred_t = spy.Tensor.empty(device, (65536,), "float")

    def score(net):
        inside_gt = np.empty(len(P), bool)
        inside_pred = np.empty(len(P), bool)
        for i in range(0, len(P), 65536):
            block = P[i:i + 65536]
            n = len(block)
            chunk = np.zeros((65536, 3), np.float32); chunk[:n] = block
            P_t.copy_from_numpy(chunk)
            module.detailed_torus_inside_batch(idx=spy.grid((65536,)), pts=P_t, _result=gt_t)
            net.predict(P_t, pred_t, 65536)
            inside_gt[i:i + n] = gt_t.to_numpy()[:n] > 0.5
            inside_pred[i:i + n] = pred_t.to_numpy()[:n] > 0.5
        return float((inside_gt & inside_pred).sum() / (inside_gt | inside_pred).sum())

    iou_pure, iou_hyb = score(pure), score(hyb)
    print(f"  IoU vs ground truth: pure {iou_pure:.3f} | hybrid {iou_hyb:.3f}")

    out = CHECKPOINTS / "race_results.npz"
    np.savez(out, c_pure=c_pure, c_hyb=c_hyb, iou_pure=iou_pure, iou_hyb=iou_hyb,
             s_pure=s_pure, s_hyb=s_hyb, budget=BUDGET,
             params_pure=pure.n_params(), params_hyb=hyb.n_params())
    print(f"  saved {out} -- run run_figures.py to render Figure 6.3 from it")
