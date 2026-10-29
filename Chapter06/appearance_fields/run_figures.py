"""Render Figures 6.5-6.7 from saved checkpoints (no retraining/no re-race).
Needs checkpoints/ckpt_sdf_supervised.npz, ckpt_appearance_albedo.npz,
ckpt_appearance_radiance.npz, and race_results.npz -- run train_one.py /
race_gpu.py (or run_section.py) first if any are missing.

    py run_figures.py                  # figures from checkpoints/results
    py run_figures.py --rerun-race 40  # re-run the race first (40s/model)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neural_sdf as G
import appearance as A
import race_gpu

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(HERE, "checkpoints")
os.makedirs("figures", exist_ok=True)
EYE1, EYE2 = (1.9, -1.9, 1.5), (-0.5, -2.4, 1.2)

if "--rerun-race" in sys.argv:
    i = sys.argv.index("--rerun-race")
    budget = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 40.0
    print(f"re-running the race ({budget:.0f}s per model):")
    pure = race_gpu.PureNet(seed=0); hyb = race_gpu.HybridNet(seed=0)
    c_pure, s_pure = race_gpu.race_one("pure", pure, budget, tag="pure MLP  ")
    c_hyb, s_hyb = race_gpu.race_one("hybrid", hyb, budget, tag="hybrid    ")
    np.savez(os.path.join(CHECKPOINTS, "race_results.npz"),
             c_pure=c_pure, c_hyb=c_hyb, s_pure=s_pure, s_hyb=s_hyb, budget=budget)

_REQUIRED = ["ckpt_sdf_supervised.npz", "ckpt_appearance_albedo.npz",
            "ckpt_appearance_radiance.npz"]
missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(CHECKPOINTS, f))]
if missing:
    sys.exit(f"\nMissing checkpoint(s) in ./checkpoints/: {missing}\n"
             f"Train first, e.g.:\n    py train_one.py albedo 800\n"
             f"    py train_one.py radiance 1200\n"
             f"(or run 'py run_section.py' to train + render in one go).\n"
             f"(race_results.npz is optional: if absent, the race is run once, "
             f"~40s per model.)")
if not os.path.exists(os.path.join(CHECKPOINTS, "race_results.npz")):
    print("race_results.npz not found -> running the race once (40s/model)")
    pure = race_gpu.PureNet(seed=0); hyb = race_gpu.HybridNet(seed=0)
    c_pure, s_pure = race_gpu.race_one("pure", pure, 40.0, tag="pure MLP  ")
    c_hyb, s_hyb = race_gpu.race_one("hybrid", hyb, 40.0, tag="hybrid    ")
    np.savez(os.path.join(CHECKPOINTS, "race_results.npz"),
             c_pure=c_pure, c_hyb=c_hyb, s_pure=s_pure, s_hyb=s_hyb, budget=40.0)

geo = G.Network.load(os.path.join(CHECKPOINTS, "ckpt_sdf_supervised.npz"))
alb = A.AppearanceNetwork.load(os.path.join(CHECKPOINTS, "ckpt_appearance_albedo.npz"), sdf_L=geo.L)
rad = A.AppearanceNetwork.load(os.path.join(CHECKPOINTS, "ckpt_appearance_radiance.npz"), sdf_L=geo.L)


def show(fig, name):
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-10))


# --- Figure 6.5 ---
print("Figure 6.5 (albedo field):")
img_gt = A.render_albedo_exact(geo, eye=EYE1, L=alb.L)
img_nf = A.render_albedo(geo, alb, eye=EYE1)
ps = psnr(img_gt, img_nf)
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
ax[0].imshow(img_gt); ax[0].set_axis_off()
ax[0].set_title("Procedural texture (ground truth)", fontsize=10)
ax[1].imshow(img_nf); ax[1].set_axis_off()
ax[1].set_title(f"Neural albedo field $c_\\theta(x)$ -- PSNR {ps:.1f} dB", fontsize=10)
fig.tight_layout(); show(fig, "fig6_5_albedo.png")
print(f"  albedo render PSNR: {ps:.1f} dB")

# --- Figure 6.6 ---
print("Figure 6.6 (view dependence):")
fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
for col, eye in enumerate([EYE1, EYE2]):
    gt = A.render_radiance_exact(geo, eye=eye, L=rad.L)
    pred = A.render_radiance(geo, rad, eye=eye)
    print(f"  radiance PSNR, camera {col + 1}: {psnr(gt, pred):.1f} dB")
    axes[0, col].imshow(gt); axes[0, col].set_axis_off()
    axes[0, col].set_title(f"GT radiance, camera {col + 1}", fontsize=10)
    axes[1, col].imshow(pred); axes[1, col].set_axis_off()
    axes[1, col].set_title(f"Neural $c_\\theta(x, v)$, camera {col + 1}", fontsize=10)
fig.tight_layout(); show(fig, "fig6_6_viewdep.png")

# --- Figure 6.7 ---
print("Figure 6.7 (the race):")
r = np.load(os.path.join(CHECKPOINTS, "race_results.npz"))
c_pure, c_hyb = r["c_pure"], r["c_hyb"]
s_pure, s_hyb, budget = int(r["s_pure"]), int(r["s_hyb"]), float(r["budget"])


def smooth(c, k=25):
    k = min(k, max(1, len(c) - 1))
    y = np.convolve(c[:, 1], np.ones(k) / k, mode="valid")
    return c[k - 1:, 0], y


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
ax.plot(*smooth(c_pure), label=f"pure MLP ({s_pure} steps)", lw=2)
ax.plot(*smooth(c_hyb), label=f"feature grid + tiny MLP ({s_hyb} steps)", lw=2)
ax.set_xlabel("wall-clock seconds (same budget, same GPU)")
ax.set_ylabel("batch accuracy (smoothed)")
ax.set_title("The race: equal time, unequal steps")
ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(0.5, 1.01)

ax = axes[1]
x = np.arange(2)
ms = [1000 * budget / s_pure, 1000 * budget / s_hyb]
ax.bar(x - 0.2, ms, width=0.4, label="ms / training step")
ax.set_xticks(x); ax.set_xticklabels(["pure MLP", "hybrid"])
ax.set_ylabel("ms per training step")
ax.set_title("Cost per step")
fig.tight_layout(); show(fig, "fig6_7_hybrid.png")

print("done.")
