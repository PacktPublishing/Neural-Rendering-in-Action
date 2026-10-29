"""Optional: train + render the Stanford bunny (the occupancy section's mesh path).

Usage:  python3 train_bunny.py [L] [steps]
Defaults: L=6, steps=3000.

Needs bunny.ply in this folder (already included) and `pip install
trimesh embreex` -- see README.md, "Optional: Stanford bunny mesh
support". The first run pays a one-time cost building+caching the
occupancy grid (trimesh.contains() over a 128^3 lattice, which needs
embreex to be fast/memory-safe); later runs reuse the cache.

Writes checkpoints/ckpt_bunny_L{L}.npz and figures/extra_bunny_occupancy.png.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import neural_occupancy as F
import mesh_shape as MS
from geometry import extract_mesh, sample_mesh_surface, trilinear_interpolate
from metrics import volumetric_iou, chamfer_distance

HERE = os.path.dirname(os.path.abspath(__file__))
PLY = os.path.join(HERE, os.pardir, "bunny.ply")      # mesh_shape.py/bunny.ply
                                                       # live at the chapter root
CHECKPOINTS = os.path.join(HERE, "checkpoints")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
HIDDEN = F.HIDDEN
BATCH = 4096
GRID_RES = 128
PANEL_RES = 128
ELEV, AZIM = 50, -60


def trisurf(ax, verts, faces, color):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    base = np.array(matplotlib.colors.to_rgb(color))
    fc = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor="none"))
    lo, hi = verts.min(), verts.max()
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=ELEV, azim=AZIM)


if not os.path.exists(PLY):
    raise SystemExit(f"missing {PLY} -- see README.md")

print(f"Part 1: load + cache the bunny (grid {GRID_RES}^3)")
occ, origin, spacing, cloud, V, Ftri, mesh = MS.load_and_cache(
    PLY, grid_res=GRID_RES, cache_dir=CHECKPOINTS)
print(f"  {len(V):,} verts, {len(Ftri):,} faces, {cloud.shape[0]:,}-pt surface cloud")

print(f"\nPart 2: train (L={L}, hidden={HIDDEN}, {STEPS} steps)")
model = F.CoordinateMLP(L=L, hidden=HIDDEN, seed=0)
hist = F.train_mesh(model, occ, origin, spacing, GRID_RES, cloud, steps=STEPS,
                    batch=BATCH)
print(f"  {model.n_params():,} params, {model.memory_bytes() / 1024:.0f} KiB, "
      f"{hist['seconds']:.0f}s, final batch acc {hist['acc'][-1]:.3f}")
ckpt = os.path.join(CHECKPOINTS, f"ckpt_bunny_L{L}.npz")
model.save(ckpt)
print(f"  [saved {ckpt}]")

print(f"\nPart 3: mesh the trained field + score against the cached grid")
occ_pred, o, sp = F.field_to_grid(model, PANEL_RES)
v_pred, f_pred, _ = extract_mesh(occ_pred, o, sp)
v_gt, f_gt, _ = extract_mesh(occ, origin, spacing)
gt_surf, _ = sample_mesh_surface(v_gt, f_gt, 30000)
pred_surf, _ = sample_mesh_surface(v_pred, f_pred, 30000)

# Ground truth for IoU is the same cached grid training labelled against
# (trilinearly interpolated, matching calculate_grads_mesh/mesh_inside_batch
# in neural_occupancy.slang) -- NOT mesh.contains() on scattered points,
# which is what trimesh's slow ray/triangle backend blows up on (see
# mesh_shape.py). This also means IoU measures "did training match its own
# labels", the same relationship the torus path already has.
iou = volumetric_iou(
    lambda p: trilinear_interpolate(occ, origin, spacing, p) > 0.5,
    lambda p: model.occupancy(p) > 0.5, n=200_000, bound=F.BOUND)
cham = chamfer_distance(gt_surf, pred_surf) * 1e3
print(f"  IoU (vs cached grid) {iou:.3f}   Chamfer {cham:.2f}e-3   "
      f"{len(v_pred):,} verts")

fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
trisurf(ax, v_gt, f_gt, "#9c7bb8")
ax.set_title("Ground truth\n(bunny, cached grid @ "
             f"{GRID_RES}^3, marching cubes)")
ax = fig.add_subplot(1, 2, 2, projection="3d")
trisurf(ax, v_pred, f_pred, "#6699cc")
ax.set_title(f"Neural occupancy field (L={L})\nIoU {iou:.3f}  Chamfer {cham:.2f}e-3")
fig.suptitle("Stanford bunny -- neural occupancy field vs. ground truth", y=1.00)
fig.tight_layout()
path = os.path.join(FIGURES, "extra_bunny_occupancy.png")
fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  [saved {path}]")
