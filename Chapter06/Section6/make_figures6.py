"""
make_figures6.py -- reproduces Figures 6.1-6.3 from the shipped artifacts.
==========================================================================
Run:

    py make_figures6.py                    # figures from checkpoints/results
    py make_figures6.py --rerun-race 40    # re-run the Part 6.4 race yourself
                                           # (40 s per model, overwrites
                                           #  race_results.npz)

Needs in this folder: app6.py, sdf5.py, ckpt_appearance.npz,
ckpt_sdf_supervised.npz, and (unless re-running) race_results.npz.
Writes: fig6_1_albedo.png, fig6_2_viewdep.png, fig6_3_hybrid.png

Note on reproducibility: Figures 6.1 and 6.2 are exact re-renders from the
trained weights. Figure 5.3 reports a wall-clock RACE, so its numbers are
machine-dependent by nature; the shipped race_results.npz reproduces the
book's figure exactly, while --rerun-race measures YOUR machine (the ratio
between the two models is what should replicate, not the absolute times).
"""

import os
import sys
import time
import numpy as np
import matplotlib
if os.name == "posix" and sys.platform != "darwin" \
        and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fail helpfully if the folder is incomplete (before local imports) ----
_REQUIRED = ["app6.py", "sdf5.py",
             "ckpt_appearance.npz", "ckpt_sdf_supervised.npz"]
_here = os.path.dirname(os.path.abspath(__file__))
_missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(_here, f))]
if _missing:
    sys.exit(
        "make_figures6.py: this folder is missing required file(s):\n"
        + "".join(f"    {f}\n" for f in _missing)
        + "All Section 6 files must sit in the SAME folder. Copy the full "
        "section package (see README / Section6.zip) and re-run.\n"
        "(race_results.npz is optional: if absent, the race is re-run, "
        "~40 s per model.)")

import app6 as A
import sdf5 as S


# --------------------------------------------------------------- loaders
def _local(name):
    """Resolve a data file relative to this script, not the CWD."""
    return os.path.join(_here, name)


def load_sdfnet(path="ckpt_sdf_supervised.npz"):
    d = np.load(_local(path))
    net = S.SDFNet(L=int(d["L"]))
    net.W = [np.ascontiguousarray(d[f"W{i}"]) for i in range(len(net.W))]
    net.b = [np.ascontiguousarray(d[f"b{i}"]) for i in range(len(net.b))]
    for w in net.W:
        w[np.abs(w) < 1e-20] = 0            # flush denormals (see Section 4)
    return net


def load_appearance(path="ckpt_appearance.npz"):
    ck = np.load(_local(path))
    alb = A.MLP([39, 128, 128, 3])           # gamma_L6(x) -> rgb
    alb.W = [np.ascontiguousarray(ck[f"aW{i}"]) for i in range(3)]
    alb.b = [np.ascontiguousarray(ck[f"ab{i}"]) for i in range(3)]
    rad = A.MLP([42, 128, 128, 3])           # gamma_L6(x), v -> rgb
    rad.W = [np.ascontiguousarray(ck[f"rW{i}"]) for i in range(3)]
    rad.b = [np.ascontiguousarray(ck[f"rb{i}"]) for i in range(3)]
    return alb, rad


# ------------------------------------------------------ shared rendering
EYE1, EYE2 = (1.9, -1.9, 1.5), (-0.5, -2.4, 1.2)


def trace(sdfnet, eye, W=300, H=225):
    """Sphere-trace the frozen Section 5 geometry; appearance networks only
    ever see the resulting hit points."""
    o, dirs, shape = S.make_camera(eye, width=W, height=H)
    t, hit = S.sphere_trace(lambda p: sdfnet.value(p), o, dirs,
                            step_scale=0.8)
    return o[hit] + t[hit, None] * dirs[hit], hit, shape, dirs


def compose(shape, hit, cols):
    """Scatter per-hit colors into a white image."""
    img = np.ones((*shape, 3), np.float32)
    flat = img.reshape(-1, 3)
    flat[hit] = np.clip(cols, 0, 1)
    return flat.reshape(*shape, 3)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-10))


def show(fig, name):
    fig.savefig(name, dpi=130, bbox_inches="tight")
    print(f"  saved {name}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6.1 -- the albedo field, Lambert-shaded through the frozen SDF
# ---------------------------------------------------------------------------
def figure_6_1(sdfnet, alb):
    p, hit, shape, _ = trace(sdfnet, EYE1)
    n = A.torus_normal(p)
    shade = (0.2 + 0.8 * np.clip((n * A.LIGHT).sum(1), 0, 1))[:, None]
    gt = np.clip(A.texture_albedo(p) * shade, 0, 1)
    nf = np.clip(alb.forward(A.encode(p)) * shade, 0, 1)
    ps = psnr(gt, nf)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax[0].imshow(compose(shape, hit, gt)); ax[0].set_axis_off()
    ax[0].set_title("Procedural texture (ground truth)", fontsize=10)
    ax[1].imshow(compose(shape, hit, nf)); ax[1].set_axis_off()
    ax[1].set_title(f"Neural albedo field $c_\\theta(x)$ — "
                    f"PSNR {ps:.1f} dB", fontsize=10)
    fig.suptitle("Figure 6.1 — appearance as a second field over the same "
                 "coordinates", y=1.02)
    fig.tight_layout()
    show(fig, "fig6_1_albedo.png")
    print(f"  albedo render PSNR: {ps:.1f} dB")


# ---------------------------------------------------------------------------
# Figure 6.2 -- view dependence: two cameras, GT above, neural field below
# ---------------------------------------------------------------------------
def figure_6_2(sdfnet, rad):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    for col, eye in enumerate([EYE1, EYE2]):
        p, hit, shape, dirs = trace(sdfnet, eye)
        v = -dirs[hit]                       # surface -> eye (unit)
        gt = A.radiance_gt(p, v)
        pred = np.clip(rad.forward(np.concatenate([A.encode(p), v], 1)), 0, 1)
        print(f"  radiance PSNR, camera {col + 1}: {psnr(gt, pred):.1f} dB")
        axes[0, col].imshow(compose(shape, hit, gt))
        axes[0, col].set_axis_off()
        axes[0, col].set_title(f"GT radiance, camera {col + 1}", fontsize=10)
        axes[1, col].imshow(compose(shape, hit, pred))
        axes[1, col].set_axis_off()
        axes[1, col].set_title(f"Neural $c_\\theta(x, v)$, camera {col + 1}",
                               fontsize=10)
    fig.suptitle("Figure 6.2 — view dependence: the highlight moves with the "
                 "camera,\nand the field (bottom row) reproduces it", y=0.99)
    fig.tight_layout()
    show(fig, "fig6_2_viewdep.png")


# ---------------------------------------------------------------------------
# Figure 6.3 -- the race: pure MLP vs feature-grid hybrid
# ---------------------------------------------------------------------------
class PureField:
    """Section 4's occupancy MLP wrapped to the race interface."""

    def __init__(self, L=6, seed=0):
        self.L = L
        self.mlp = A.MLP([3 + 6 * L, 256, 256, 1], seed=seed)

    def forward(self, pts):
        return self.mlp.forward(A.encode(pts, self.L)).ravel()

    def backward(self, dlogit):
        gW, gb = self.mlp.backward(dlogit[:, None])
        return self.mlp.tensors(), self.mlp.grads_as_list(gW, gb)

    def value(self, pts, chunk=131072):
        out = np.empty(len(pts), np.float32)
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = self.forward(pts[i:i + chunk])
        return out

    def n_params(self):
        return sum(t.size for t in self.mlp.tensors())


def run_race(budget=40.0, seed=0):
    """Re-run Part 6.4 on THIS machine and overwrite race_results.npz."""
    from sdf5 import knurl_inside, BOUND
    pure = PureField(seed=seed)
    hyb = A.HybridField(res=32, feat=4, hidden=32, seed=seed)
    print(f"  params: pure {pure.n_params():,} | hybrid {hyb.n_params():,}")
    c_pure, s_pure = A.train_occupancy(pure, pure.mlp.tensors(),
                                       pure.backward, seconds=budget,
                                       seed=seed, tag="pure MLP  ")
    c_hyb, s_hyb = A.train_occupancy(hyb, [hyb.grid.F] + hyb.head.tensors(),
                                     hyb.backward, seconds=budget,
                                     seed=seed, tag="hybrid    ")
    rng = np.random.default_rng(9)
    P = rng.uniform(-BOUND, BOUND, (200000, 3)).astype(np.float32)
    gt = knurl_inside(P)
    iou, thr = {}, {}
    Q = rng.uniform(-BOUND, BOUND, (1_000_000, 3)).astype(np.float32)
    for name, m in [("pure", pure), ("hyb", hyb)]:
        B = m.value(P) > 0
        iou[name] = float((gt & B).sum() / (gt | B).sum())
        t0 = time.time(); m.value(Q)
        thr[name] = 1e-6 * len(Q) / (time.time() - t0)
    np.savez(_local("race_results.npz"), c_pure=c_pure, c_hyb=c_hyb,
             iou_pure=iou["pure"], iou_hyb=iou["hyb"],
             thr_pure=thr["pure"], thr_hyb=thr["hyb"],
             s_pure=s_pure, s_hyb=s_hyb, budget=budget,
             params_pure=pure.n_params(), params_hyb=hyb.n_params())
    print("  race_results.npz overwritten with this machine's numbers")


def figure_6_3():
    r = np.load(_local("race_results.npz"))
    c_pure, c_hyb = r["c_pure"], r["c_hyb"]
    budget = float(r["budget"])
    s_pure, s_hyb = int(r["s_pure"]), int(r["s_hyb"])

    def smooth(c, k=25):
        y = np.convolve(c[:, 1], np.ones(k) / k, mode="valid")
        return c[k - 1:, 0], y

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(*smooth(c_pure), label=f"pure MLP ({s_pure} steps)", lw=2)
    ax.plot(*smooth(c_hyb),
            label=f"feature grid + tiny MLP ({s_hyb} steps)", lw=2)
    ax.set_xlabel("wall-clock seconds (same budget, same machine)")
    ax.set_ylabel("batch accuracy (smoothed)")
    ax.set_title("The race: equal time, unequal steps")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.01)

    ax = axes[1]
    x = np.arange(2)
    ms = [1000 * budget / s_pure, 1000 * budget / s_hyb]
    b1 = ax.bar(x - 0.2, ms, width=0.4, label="ms / training step")
    ax2 = ax.twinx()
    thr = [float(r["thr_pure"]), float(r["thr_hyb"])]
    b2 = ax2.bar(x + 0.2, thr, width=0.4, color="C1",
                 label="inference M queries/s")
    ax.set_xticks(x); ax.set_xticklabels(["pure MLP", "hybrid"])
    ax.set_ylabel("ms per training step")
    ax2.set_ylabel("inference (M queries / s)")
    for b in b1:
        ax.text(b.get_x() + 0.2, b.get_height() * 1.02,
                f"{b.get_height():.0f}", ha="center", fontsize=9)
    for b in b2:
        ax2.text(b.get_x() + 0.2, b.get_height() * 1.02,
                 f"{b.get_height():.1f}", ha="center", fontsize=9)
    ax.set_title(f"Cost per step and per query\n(IoU after {budget:.0f}s: "
                 f"{float(r['iou_pure']):.3f} vs {float(r['iou_hyb']):.3f})")
    fig.suptitle("Figure 6.3 — moving the shape out of the MLP and into a "
                 "feature grid", y=1.03)
    fig.tight_layout()
    show(fig, "fig6_3_hybrid.png")


if __name__ == "__main__":
    if "--rerun-race" in sys.argv:
        i = sys.argv.index("--rerun-race")
        budget = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 40.0
        print(f"re-running the Part 6.4 race ({budget:.0f}s per model):")
        run_race(budget=budget)
    elif not os.path.exists(_local("race_results.npz")):
        print("race_results.npz not found -> running the race once (40s/model)")
        run_race()

    print("loading checkpoints ...")
    sdfnet = load_sdfnet()
    alb, rad = load_appearance()
    print("Figure 6.1 (albedo field):");    figure_6_1(sdfnet, alb)
    print("Figure 6.2 (view dependence):"); figure_6_2(sdfnet, rad)
    print("Figure 6.3 (the race):");        figure_6_3()
    print("done.")
