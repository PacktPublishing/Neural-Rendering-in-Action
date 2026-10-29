"""Render Parts 3-5 from saved checkpoints. (Parts 1-2 are produced by
run_section.py and don't depend on trained models.) Runs on a headless
server: save only."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import neural_occupancy as F
from geometry import extract_mesh, sample_mesh_surface, grid_memory_bytes
from metrics import volumetric_iou, chamfer_distance

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(HERE, "checkpoints")
os.makedirs("figures", exist_ok=True)
L_VALUES = [0, 4, 10]
PANEL_RES = 96
ELEV, AZIM = 50, -60

ckpt_path = lambda L: os.path.join(CHECKPOINTS, f"ckpt_L{L}.npz")
hist_path = lambda L: os.path.join(CHECKPOINTS, f"hist_L{L}.npz")
missing = [L for L in L_VALUES if not os.path.exists(ckpt_path(L))]
if missing:
    sys.exit(f"\nNo checkpoints for L={missing}. Train them first, e.g.:\n"
             f"    py train_one.py 0 1500\n    py train_one.py 4 1500\n"
             f"    py train_one.py 10 1500\n"
             f"(or run 'py run_section.py' to train + render in one go).\n"
             f"(checkpoints are read from ./checkpoints/)")

models = {L: F.CoordinateMLP.load(ckpt_path(L)) for L in L_VALUES}
hists = {L: np.load(hist_path(L)) for L in L_VALUES}


def save(fig, name):
    p = os.path.join("figures", name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [saved {p}]")
    plt.close(fig)


def trisurf(ax, v, f, color="#6699cc"):
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.3, 0.85]); light /= np.linalg.norm(light)
    sh = 0.35 + 0.65 * np.abs(n @ light)
    base = np.array(matplotlib.colors.to_rgb(color))
    ax.add_collection3d(Poly3DCollection(
        tri, facecolors=np.clip(sh[:, None] * base[None, :], 0, 1), edgecolor="none"))
    lo, hi = v.min(), v.max()
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=ELEV, azim=AZIM)


def smooth(a, k=40):
    return np.convolve(a, np.ones(k) / k, mode="valid")


# --- Part 3: training curves ---
print("Part 3: training curves")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for L in L_VALUES:
    axes[0].plot(smooth(hists[L]["loss"]), label=f"L = {L}")
    axes[1].plot(smooth(hists[L]["acc"]), label=f"L = {L}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("BCE loss (smoothed)")
axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("step"); axes[1].set_ylabel("batch accuracy (smoothed)")
axes[1].set_title("Batch accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

fig.tight_layout(); save(fig, "extra_training_curves.png")

# --- ground truth mesh (shared by Parts 4) ---
occ_gt, og, sg = F.occupancy_grid_gpu(PANEL_RES)
v_gt, f_gt, _ = extract_mesh(occ_gt, og, sg)
gt_surf, _ = sample_mesh_surface(v_gt, f_gt, 30000)

# --- Part 4: spectral-bias panels + metrics ---
print("Part 4: surface extraction + metrics")
panels = [("Ground truth", v_gt, f_gt, "#9c7bb8", None)]
rows = [("ground truth", "-", 1.000, 0.0)]
for L in L_VALUES:
    occ, o, sp = F.field_to_grid(models[L], PANEL_RES)
    v, f, _ = extract_mesh(occ, o, sp)
    iou = volumetric_iou(F.detailed_torus_inside_batch,
                         lambda p, mm=models[L]: mm.occupancy(p) > 0.5, n=200000)
    rec, _ = sample_mesh_surface(v, f, 30000)
    cham = chamfer_distance(gt_surf, rec) * 1e3
    label = "No encoding (L=0)" if L == 0 else f"L = {L}"
    panels.append((label, v, f, "#6699cc" if L == 10 else "#cc7755", iou))
    rows.append((f"MLP L={L}", f"{models[L].memory_bytes()/1024:.0f} KiB", iou, cham))
    print(f"  L={L:2d}: IoU {iou:.3f}, Chamfer {cham:.2f}e-3, {len(f)} faces")

fig = plt.figure(figsize=(16, 4.8))
for i, (title, v, f, color, iou) in enumerate(panels):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    trisurf(ax, v, f, color)
    ax.set_title(title + ("" if iou is None else f"\nIoU {iou:.3f}"))

fig.tight_layout(); save(fig, "fig6_1_encoding_bands.png")

print("\n  representation        memory      IoU    Chamfer(x1e-3)")
print("  " + "-" * 52)
for name, mem, iou, cham in rows:
    print(f"  {name:18s} {str(mem):>9s}   {iou:5.3f}   {cham:8.2f}")

# --- Part 5: resolution independence ---
print("\nPart 5: resolution independence")
best = models[10]
fig = plt.figure(figsize=(13, 4.6))
for i, R in enumerate([32, 64, 128]):
    occ, o, sp = F.field_to_grid(best, R)
    v, f, _ = extract_mesh(occ, o, sp)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    trisurf(ax, v, f, "#6699cc")
    ax.set_title(f"query grid {R}$^3$\n{len(v)} verts, "
                 f"net still {best.memory_bytes()/1024:.0f} KiB")
    print(f"  L=10 meshed at {R:3d}^3: {len(v):6d} verts "
          f"(net memory fixed at {best.memory_bytes()/1024:.0f} KiB)")
fig.tight_layout(); save(fig, "extra_resolution.png")

print(f"\n  memory: L=10 field = {best.memory_bytes()/1024:.0f} KiB (resolution-free)")
for R in (64, 128, 256):
    print(f"          {R:3d}^3 occupancy grid = {grid_memory_bytes(R)/1024:.0f} KiB")
print("\nFigures 4.3-4.5 complete.")
