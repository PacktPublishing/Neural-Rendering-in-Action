"""
Section 6 driver -- trains both appearance fields, runs the race, and
produces every figure.
========================================================================
Run:  python3 run_section.py     (runs on a headless server: figures saved only)

Needs neural_signed_distance's trained SDF checkpoint (this folder's own
checkpoints/ckpt_sdf_supervised.npz, copied from there -- see README.md):
geometry is frozen throughout this section.

Figures (./figures):
  fig6_1_albedo.png     the albedo field, Lambert-shaded through the frozen SDF (6.2)
  fig6_2_viewdep.png    view dependence: two cameras, GT above, neural field below (6.3)
  fig6_3_hybrid.png     the race: pure MLP vs feature-grid hybrid              (6.4)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neural_sdf as G
import appearance as A
# NOTE: race_gpu is deliberately NOT imported here -- it is imported in Part
# 6.4, where it is first used. It creates a Vulkan device and compiles its
# module AT IMPORT TIME, and neural_sdf/appearance each create a device of
# their own, so importing it up here left three live devices during training
# and segfaulted partway through (reproducibly, ~3800 of 4000 albedo steps;
# the same training completes fine at 4000 steps without it). Invisible at
# the old 800/1200 step counts, which is why it went unnoticed.

os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

L = 6
# 800/1200 (the book's figures) leave both fields badly under-trained: this
# texture is high frequency and the loss is still falling steeply there.
# Measured against exact surface points, the albedo field reaches only
# ~20.5 dB at 800 steps vs ~31 dB at 3200 -- at 800 the render is coloured
# static with no visible checker. See README.md.
ALBEDO_STEPS = 4000
RADIANCE_STEPS = 4000
RACE_BUDGET = 40.0
GEO_CKPT = os.path.join("checkpoints", "ckpt_sdf_supervised.npz")
EYE1, EYE2 = (1.9, -1.9, 1.5), (-0.5, -2.4, 1.2)


def banner(n, title):
    print("\n" + "=" * 70 + f"\nPart {n}: {title}\n" + "=" * 70)


def show(fig, name):
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [saved {path}]")
    plt.close(fig)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-10))


if not os.path.exists(GEO_CKPT):
    sys.exit(f"\nMissing {GEO_CKPT}.\nCopy it from "
             f"../neural_signed_distance/checkpoints/ckpt_sdf_supervised.npz "
             f"(train it there first: py train_one.py supervised 1500).\n")

banner(6.1, "loading the frozen geometry field (Section 5's trained SDF)")
geo = G.Network.load(GEO_CKPT)
print(f"  geometry: {geo.n_params():,} params, {geo.memory_bytes()/1024:.0f} KiB, L={geo.L}")


# ============================================================================
# Part 6.2 -- the albedo field
# ============================================================================
banner(6.2, f"albedo field c(x): a procedural texture ({ALBEDO_STEPS} steps)")
alb = A.AppearanceNetwork(has_view=False, L=L, sdf_L=geo.L, seed=0)
h1 = A.train_albedo(alb, steps=ALBEDO_STEPS)
print(f"  {alb.n_params():,} params, {h1['seconds']:.0f}s")

img_gt = A.render_albedo_exact(geo, eye=EYE1, L=L)
img_nf = A.render_albedo(geo, alb, eye=EYE1)
ps = psnr(img_gt, img_nf)
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
ax[0].imshow(img_gt); ax[0].set_axis_off()
ax[0].set_title("Procedural texture (ground truth)", fontsize=10)
ax[1].imshow(img_nf); ax[1].set_axis_off()
ax[1].set_title(f"Neural albedo field $c_\\theta(x)$ -- PSNR {ps:.1f} dB", fontsize=10)
fig.suptitle("Figure 6.1 -- appearance as a second field over the same "
             "coordinates", y=1.02)
fig.tight_layout()
show(fig, "fig6_1_albedo.png")
print(f"  albedo render PSNR: {ps:.1f} dB")


# ============================================================================
# Part 6.3 -- the radiance field (view dependence)
# ============================================================================
banner(6.3, f"radiance field c(x, v): a view-dependent highlight ({RADIANCE_STEPS} steps)")
rad = A.AppearanceNetwork(has_view=True, L=L, sdf_L=geo.L, seed=1)
h2 = A.train_radiance(rad, steps=RADIANCE_STEPS)
print(f"  {rad.n_params():,} params, {h2['seconds']:.0f}s")

fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
for col, eye in enumerate([EYE1, EYE2]):
    gt = A.render_radiance_exact(geo, eye=eye, L=L)
    pred = A.render_radiance(geo, rad, eye=eye)
    print(f"  radiance PSNR, camera {col + 1}: {psnr(gt, pred):.1f} dB")
    axes[0, col].imshow(gt); axes[0, col].set_axis_off()
    axes[0, col].set_title(f"GT radiance, camera {col + 1}", fontsize=10)
    axes[1, col].imshow(pred); axes[1, col].set_axis_off()
    axes[1, col].set_title(f"Neural $c_\\theta(x, v)$, camera {col + 1}", fontsize=10)
fig.suptitle("Figure 6.2 -- view dependence: the highlight moves with the "
             "camera,\nand the field (bottom row) reproduces it", y=0.99)
fig.tight_layout()
show(fig, "fig6_2_viewdep.png")

alb.save(os.path.join("checkpoints", "ckpt_appearance_albedo.npz"))
rad.save(os.path.join("checkpoints", "ckpt_appearance_radiance.npz"))


# ============================================================================
# Part 6.4 -- the race: pure MLP vs feature-grid hybrid
# ============================================================================
banner(6.4, f"the race: pure MLP vs feature grid + tiny MLP ({RACE_BUDGET:.0f}s/model)")
import race_gpu          # deferred on purpose -- see the note by the imports
pure = race_gpu.PureNet(seed=0)
hyb = race_gpu.HybridNet(seed=0)
print(f"  params: pure {pure.n_params():,} | hybrid {hyb.n_params():,}")
c_pure, s_pure = race_gpu.race_one("pure", pure, RACE_BUDGET, tag="pure MLP  ")
c_hyb, s_hyb = race_gpu.race_one("hybrid", hyb, RACE_BUDGET, tag="hybrid    ")


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
ms = [1000 * RACE_BUDGET / s_pure, 1000 * RACE_BUDGET / s_hyb]
ax.bar(x - 0.2, ms, width=0.4, label="ms / training step")
ax.set_xticks(x); ax.set_xticklabels(["pure MLP", "hybrid"])
ax.set_ylabel("ms per training step")
ax.set_title("Cost per step")
fig.suptitle("Figure 6.3 -- moving the shape out of the MLP and into a "
             "feature grid", y=1.03)
fig.tight_layout()
show(fig, "fig6_3_hybrid.png")

print("\nAll Section 6 parts complete.")
