"""
Section 5 driver -- trains both SDF recipes and produces every figure.
========================================================================
Run:  python3 run_section.py     (runs on a headless server: figures saved only)

Figures (./figures):
  fig5_1_spheretrace.png   analytic vs neural sphere-traced renders, plus
                           the neural field's own normals              (5.1/5.3/5.4)
  fig5_2_csg.png           smooth_union(trained field, sphere), 3 fillets (5.6)
  fig5_3_eikonal.png       an SDF learned from surface points alone     (5.5)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import neural_sdf as F

os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

L = 6
SUPERVISED_STEPS = 1500
EIKONAL_STEPS = 2700
EIKONAL_CHUNK = 900
PANEL_RES = 96

# The eikonal recipe (points alone, no exact distances) needs more capacity
# than the supervised one to recover the torus's hole and ripple detail --
# F.HIDDEN (64) is plenty for 5.2's supervised regression (IoU 0.984) but
# 5.5's eikonal SDF stays a hole-less blob at that width. The book's own
# 256 fixes the topology; see README.md's "Differences from the book" for
# the cost (~14 min here vs ~35s) and the still-open sign-calibration
# question it surfaces.
EIKONAL_HIDDEN = 256


def show(fig, name):
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [saved {path}]")
    plt.close(fig)


def trisurf(ax, verts, faces, color):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.33, -0.25, 0.91]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    base = np.array(matplotlib.colors.to_rgb(color))
    fc = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor="none"))


def fit_box(*point_sets, pad=0.12):
    """Shared, padded bounding box across all given point sets, with a
    per-axis box_aspect (not a forced cube) -- the knurled torus is a thin
    flat ring (z barely reaches +-0.36) and a cube box wastes most of the
    panel on empty space above/below it."""
    allpts = np.concatenate(point_sets, axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad_amt = np.maximum(hi - lo, 1e-6) * pad
    return lo - pad_amt, hi + pad_amt


def apply_box(ax, lo, hi, zoom=1.75):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=zoom)  # zoom > 1: fill more of the panel
    ax.set_axis_off(); ax.view_init(elev=50, azim=-60)


def banner(n, title):
    print("\n" + "=" * 70 + f"\nPart {n}: {title}\n" + "=" * 70)


# ============================================================================
# Part 5.2 -- train the supervised SDF (exact distances)
# ============================================================================
banner("5.2", f"regression SDF: exact smooth-torus distances ({SUPERVISED_STEPS} steps)")
net_sup = F.Network(L=L, seed=0)
h_sup = F.train_supervised(net_sup, steps=SUPERVISED_STEPS)
print(f"  {net_sup.n_params():,} params, {net_sup.memory_bytes()/1024:.0f} KiB, "
      f"{h_sup['seconds']:.0f}s")

rng = np.random.default_rng(1)
P = rng.uniform(-F.BOUND, F.BOUND, (200_000, 3)).astype(np.float32)
pred, gt = net_sup.value(P), np.sqrt(
    (np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - 0.65) ** 2 + P[:, 2] ** 2) - 0.28
l1 = np.abs(pred - gt).mean()
iou = np.logical_and(pred < 0, gt < 0).sum() / np.logical_or(pred < 0, gt < 0).sum()
print(f"  IoU vs ground truth: {iou:.3f}   mean |L1| error: {l1:.4f}")


# ============================================================================
# Part 5.1/5.3/5.4 -- sphere-traced renders: analytic vs neural, plus normals
# ============================================================================
banner("5.1/5.4", "sphere-tracing the trained field directly -- no mesh anywhere")
img_gt = F.render_reference(L=L)
img_nf = F.render(net_sup)
img_n = F.render_normals(net_sup)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
for ax, im, title in zip(axes, [img_gt, img_nf, img_n], [
        "Analytic SDF, sphere-traced\n(reference)",
        "NEURAL SDF, sphere-traced\n(no mesh anywhere in this image)",
        "Neural normals (central differences,\nGPU-side, in the render kernel)"]):
    ax.imshow(im); ax.set_axis_off(); ax.set_title(title, fontsize=10)
fig.suptitle("Figure 5.1 -- rendering the neural SDF directly by sphere "
             "tracing", y=1.03)
fig.tight_layout()
show(fig, "fig5_1_spheretrace.png")


# ============================================================================
# Part 5.6 -- CSG: smooth union of the neural torus and an analytic sphere
# ============================================================================
banner("5.6", "shapes as arithmetic: smooth_union(neural field, sphere)")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
for ax, k in zip(axes, [0.02, 0.10, 0.25]):
    im = F.render_smooth_union(net_sup, k=k)
    ax.imshow(im); ax.set_axis_off()
    ax.set_title(f"smooth_union, k={k}", fontsize=10)
fig.suptitle("Figure 5.2 -- CSG on distance fields: neural torus union "
             "analytic sphere, growing fillet radius", y=1.03)
fig.tight_layout()
show(fig, "fig5_2_csg.png")


# ============================================================================
# Part 5.5 -- an SDF from a surface point cloud alone (IGR-style)
# ============================================================================
banner("5.5", f"eikonal SDF: 30,000 knurled-torus surface points, no distances ({EIKONAL_STEPS} steps)")
module = F._module(L, EIKONAL_HIDDEN)
cloud_t = F.make_surface_cloud(module, 30_000, seed=0)
net_eik = F.Network(L=L, hidden=EIKONAL_HIDDEN, seed=1)


def report_iou(step, model):
    A, _, _ = F.knurl_grid(64)
    B, _, _ = F.sdf_grid(model, 64)
    inside_a, inside_b = A > 0.5, B < 0
    iou = np.logical_and(inside_a, inside_b).sum() / np.logical_or(inside_a, inside_b).sum()
    print(f"    [checkpoint @ step {step}] IoU vs knurled torus: {iou:.3f}")


h_eik = F.train_eikonal(net_eik, cloud_t, steps=EIKONAL_STEPS,
                        checkpoint_every=EIKONAL_CHUNK, on_checkpoint=report_iou)
print(f"  eikonal training: {h_eik['seconds']:.0f}s total")

gt_vol, og, sg = F.knurl_grid(PANEL_RES)
gv, gf, _, _ = measure.marching_cubes(gt_vol, level=0.5, spacing=tuple(sg))
gv += og

nf_vol, on_, sn = F.sdf_grid(net_eik, PANEL_RES)
nv, nf, _, _ = measure.marching_cubes(nf_vol, level=0.0, spacing=tuple(sn))
nv += on_

surf_pts = cloud_t.to_numpy()

lo, hi = fit_box(surf_pts, nv, gv)

fig = plt.figure(figsize=(13.5, 4.2))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2], s=1.5,
          c=surf_pts[:, 2], cmap="coolwarm")
apply_box(ax, lo, hi)
ax.set_title("The ONLY input:\nsurface points (no distances)", fontsize=10)
ax = fig.add_subplot(1, 3, 2, projection="3d"); trisurf(ax, nv, nf, "#cc8855")
apply_box(ax, lo, hi)
ax.set_title("Learned SDF, zero level set", fontsize=10)
ax = fig.add_subplot(1, 3, 3, projection="3d"); trisurf(ax, gv, gf, "#9c7bb8")
apply_box(ax, lo, hi)
ax.set_title("Reference shape\n(for comparison only)", fontsize=10)
fig.suptitle("Figure 5.3 -- an SDF from points alone: surface + eikonal + "
             "repulsion losses", y=1.03)
fig.tight_layout()
show(fig, "fig5_3_eikonal.png")

net_sup.save(os.path.join("checkpoints", "ckpt_sdf_supervised.npz"))
net_eik.save(os.path.join("checkpoints", "ckpt_sdf_eikonal.npz"))
print("\nAll Section 5 parts complete.")
