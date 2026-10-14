"""Render Figures 5.1-5.3 from saved checkpoints (no retraining). Needs
checkpoints/ckpt_sdf_supervised.npz and checkpoints/ckpt_sdf_eikonal.npz --
run train_one.py (or run_section.py) first if they're missing."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import neural_sdf as F

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(HERE, "checkpoints")
os.makedirs("figures", exist_ok=True)
PANEL_RES = 96

ckpt_sup = os.path.join(CHECKPOINTS, "ckpt_sdf_supervised.npz")
ckpt_eik = os.path.join(CHECKPOINTS, "ckpt_sdf_eikonal.npz")
missing = [p for p in (ckpt_sup, ckpt_eik) if not os.path.exists(p)]
if missing:
    sys.exit(f"\nMissing checkpoint(s): {missing}\nTrain first, e.g.:\n"
             f"    py train_one.py supervised 1500\n    py train_one.py eikonal 2700\n"
             f"(or run 'py run_section.py' to train + render in one go).\n")

net_sup = F.Network.load(ckpt_sup)
net_eik = F.Network.load(ckpt_eik)


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


# --- Figure 5.1 ---
print("Figure 5.1 (sphere tracing):")
img_gt = F.render_reference(L=net_sup.L)
img_nf = F.render(net_sup)
img_n = F.render_normals(net_sup)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
for ax, im, title in zip(axes, [img_gt, img_nf, img_n], [
        "Analytic SDF, sphere-traced\n(reference)",
        "NEURAL SDF, sphere-traced\n(no mesh anywhere in this image)",
        "Neural normals (central differences,\nGPU-side, in the render kernel)"]):
    ax.imshow(im); ax.set_axis_off(); ax.set_title(title, fontsize=10)
fig.tight_layout(); show(fig, "fig5_1_spheretrace.png")

# --- Figure 5.2 ---
print("Figure 5.2 (CSG):")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
for ax, k in zip(axes, [0.02, 0.10, 0.25]):
    im = F.render_smooth_union(net_sup, k=k)
    ax.imshow(im); ax.set_axis_off()
    ax.set_title(f"smooth_union, k={k}", fontsize=10)
fig.tight_layout(); show(fig, "fig5_2_csg.png")

# --- Figure 5.3 ---
print("Figure 5.3 (eikonal):")
module = F._module(net_eik.L, net_eik.hidden)
cloud_t = F.make_surface_cloud(module, 30_000, seed=0)  # deterministic: same seed
surf_pts = cloud_t.to_numpy()

gt_vol, og, sg = F.knurl_grid(PANEL_RES)
gv, gf, _, _ = measure.marching_cubes(gt_vol, level=0.5, spacing=tuple(sg))
gv += og

nf_vol, on_, sn = F.sdf_grid(net_eik, PANEL_RES)
nv, nf, _, _ = measure.marching_cubes(nf_vol, level=0.0, spacing=tuple(sn))
nv += on_

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

fig.tight_layout(); show(fig, "fig5_3_eikonal.png")

print("done.")
