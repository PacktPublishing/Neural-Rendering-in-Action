"""Optional: the Stanford bunny across three neural representations.

Usage:  python3 make_bunny_comparison.py [L]
Default L=6. Reads already-trained checkpoints -- run these first:

    (from neural_occupancy_fields/)  python3 train_bunny.py       [L]
    (from neural_signed_distance/)   python3 train_bunny_sdf.py   [L]

Renders a 5-panel comparison to figures/fig6_8_bunny.png:
  1. Ground truth        (mesh_shape's cached occupancy grid, marching cubes)
  2. Neural occupancy    (occupancy section: P(inside), regressed from labelled points)
  3. Neural SDF, supervised (supervised SDF: exact-ish distance, regressed
                              from a precomputed signed-distance grid)
  4. Points alone        (the ONLY input to panel 5 -- a surface-point
                           scatter, no distances, no inside/outside labels)
  5. Neural SDF, eikonal (points-alone SDF: trained on panel 4's points alone,
                           IGR-style -- no distance labels of any kind)

The three networks are trained completely independently (different
sections, different recipes, different loss functions) but all reconstruct
the same held-out shape -- this figure is the payoff for that: it's one
mesh, told to three different kinds of network in three different ways.
Panel 4 makes explicit just how little panel 5 was given to work with.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "neural_occupancy_fields"))
sys.path.insert(0, os.path.join(HERE, "neural_signed_distance"))
import neural_occupancy as OCC
import neural_sdf as SDF
import mesh_shape as MS

PLY = os.path.join(HERE, "bunny.ply")
OCC_CKPT_DIR = os.path.join(HERE, "neural_occupancy_fields", "checkpoints")
SDF_CKPT_DIR = os.path.join(HERE, "neural_signed_distance", "checkpoints")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GRID_RES = 128
ELEV, AZIM, ROLL = 80, -60, 40   # from above, ears up, base level -- see
                                 # the angle-comparison grids used to pick
                                 # these (elev=50 showed the open
                                 # base/underside; roll=0 left the bunny
                                 # sitting at a diagonal tilt)


def extract_mesh(occ, origin, spacing, level=0.5):
    verts, faces, normals, _ = measure.marching_cubes(occ, level=level,
                                                       spacing=tuple(spacing))
    return verts + origin, faces


def _setup_axes(ax, lim, zoom=1.4):
    lo, hi = lim
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    # zoom > 1 shrinks the margin matplotlib otherwise reserves around a 3D
    # axes box, so the content fills more of its panel.
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off(); ax.view_init(elev=ELEV, azim=AZIM, roll=ROLL)


def trisurf(ax, verts, faces, color, lim, zoom=1.4):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    base = np.array(matplotlib.colors.to_rgb(color))
    fc = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolor="none"))
    _setup_axes(ax, lim, zoom)


def scatter_points(ax, pts, lim, zoom=1.4):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 2],
              cmap="coolwarm", linewidths=0)
    _setup_axes(ax, lim, zoom)


if not os.path.exists(PLY):
    raise SystemExit(f"missing {PLY} -- see neural_occupancy_fields/README.md")

t0 = time.time()
print("Ground truth")
occ_gt, origin, spacing, cloud, V, Ftri, _ = MS.load_and_cache(
    PLY, grid_res=GRID_RES, cache_dir=OCC_CKPT_DIR)
v_gt, f_gt = extract_mesh(occ_gt, origin, spacing, level=0.5)
lim = (v_gt.min(), v_gt.max())
print(f"  {time.time() - t0:.2f}s (mesh_shape.load_and_cache + marching cubes; "
      f"reuses the cached grid/cloud if present)")

# panels: (kind, title, payload, color). kind "mesh" -> payload (verts, faces);
# kind "points" -> payload an (N, 3) point array, color unused (depth-colored).
panels = [("mesh", "Ground truth\n(bunny mesh)", (v_gt, f_gt), "#9c7bb8")]

occ_ckpt = os.path.join(OCC_CKPT_DIR, f"ckpt_bunny_L{L}.npz")
if os.path.exists(occ_ckpt):
    t0 = time.time()
    print("Neural occupancy field")
    m_occ = OCC.CoordinateMLP.load(occ_ckpt)
    occ_pred, o, sp = OCC.field_to_grid(m_occ, GRID_RES)
    v, f = extract_mesh(occ_pred, o, sp, level=0.5)
    panels.append(("mesh", f"Neural occupancy field\n(L={m_occ.L})", (v, f), "#6699cc"))
    print(f"  {time.time() - t0:.2f}s (load checkpoint + evaluate {GRID_RES}^3 "
          f"grid + marching cubes -- no training here, that already happened)")
else:
    print(f"  [skipped -- run train_bunny.py {L} in neural_occupancy_fields/ first]")

sup_ckpt = os.path.join(SDF_CKPT_DIR, f"ckpt_bunny_sdf_sup_L{L}.npz")
if os.path.exists(sup_ckpt):
    t0 = time.time()
    print("Neural SDF, supervised")
    m_sup = SDF.Network.load(sup_ckpt)
    val, o, sp = SDF.sdf_grid(m_sup, GRID_RES)
    v, f = extract_mesh(val, o, sp, level=0.0)
    panels.append(("mesh", f"Neural SDF, supervised\n(L={m_sup.L})", (v, f), "#cc7755"))
    print(f"  {time.time() - t0:.2f}s")
else:
    print(f"  [skipped -- run train_bunny_sdf.py {L} in neural_signed_distance/ first]")

eik_ckpt = os.path.join(SDF_CKPT_DIR, f"ckpt_bunny_sdf_eik_L{L}.npz")
if os.path.exists(eik_ckpt):
    t0 = time.time()
    print("Points alone (eikonal's only input)")
    # Same seed as train_bunny_sdf.py's own load_and_cache call (both
    # default to seed=0), so this is the SAME 200k-point cloud the
    # checkpoint below was actually trained on -- not just a similar one.
    # Subsampled for display only (matches the book's own Figure 6.3, which
    # shows 30,000 of these); the checkpoint trained on the full cloud.
    disp_n = min(30_000, cloud.shape[0])
    disp_idx = np.random.default_rng(0).choice(cloud.shape[0], disp_n, replace=False)
    panels.append(("points", f"Points alone\n({cloud.shape[0]:,} pts, no distances)",
                   cloud[disp_idx], None))
    print(f"  {time.time() - t0:.2f}s (subsample only -- cloud already in memory)")

    t0 = time.time()
    print("Neural SDF, eikonal (points alone)")
    m_eik = SDF.Network.load(eik_ckpt)
    val, o, sp = SDF.sdf_grid(m_eik, GRID_RES)
    v, f = extract_mesh(val, o, sp, level=0.0)
    panels.append(("mesh", f"Neural SDF, eikonal\n(points alone, L={m_eik.L})", (v, f), "#c0392b"))
    print(f"  {time.time() - t0:.2f}s (hidden=256, so slower to evaluate than "
          f"the hidden=64 networks above)")
else:
    print(f"  [skipped -- run train_bunny_sdf.py {L} in neural_signed_distance/ first]")

t0 = time.time()
n = len(panels)
fig = plt.figure(figsize=(3.4 * n, 3.6))
for i, (kind, title, payload, color) in enumerate(panels):
    ax = fig.add_subplot(1, n, i + 1, projection="3d")
    if kind == "mesh":
        v, f = payload
        trisurf(ax, v, f, color, lim)
    else:
        scatter_points(ax, payload, lim)
    ax.set_title(title, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.03, wspace=0.02)
path = os.path.join(FIGURES, "fig6_8_bunny.png")
fig.savefig(path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  {time.time() - t0:.2f}s (matplotlib: build the {n}-panel figure + save PNG)")
print(f"\n[saved {path}]")
