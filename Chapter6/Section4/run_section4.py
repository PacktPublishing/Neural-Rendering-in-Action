"""
Section 4 driver -- runs every part and produces every figure.
==============================================================
Run:  python3 run_section4.py            (a window opens per figure)
      python3 run_section4.py --headless (figures saved only)

Figures (./figures):
  fig4_1_shape_and_data.png   the shape + a labelled point sample (Part 1)
  fig4_2_encoding.png         what positional encoding does       (Part 2)
  fig4_3_training.png         training curves, L = 0 / 4 / 10      (Part 3)
  fig4_4_spectral_bias.png    the hero: GT vs L=0 vs L=4 vs L=10   (Part 4)
  fig4_5_resolution.png       one network, three query grids       (Part 5)
"""

import os
import sys
import numpy as np
import matplotlib
HEADLESS = ("--headless" in sys.argv) or (
    os.name == "posix" and sys.platform != "darwin"
    and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import shapes4 as s
import field4 as F
from geometry import extract_mesh, sample_mesh_surface, grid_memory_bytes
from metrics import volumetric_iou, chamfer_distance

os.makedirs("figures", exist_ok=True)

# Experiment configuration -- one place to change everything.
HIDDEN = (256, 256, 256)
STEPS = 1500
BATCH = 4096
L_VALUES = [0, 4, 10]
PANEL_RES = 96            # marching-cubes resolution for the comparison panels
ELEV, AZIM = 50, -60


def show(fig, name):
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [saved {path}]")
    if not HEADLESS:
        plt.show()
    plt.close(fig)


def trisurf(ax, verts, faces, color="#6699cc"):
    """Lambertian-shaded mesh (same normals-from-cross-products as Listing 3.1)."""
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
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)


def banner(n, title):
    print("\n" + "=" * 70 + f"\nPart {n}: {title}\n" + "=" * 70)


# ============================================================================
# Part 1 -- the shape and the training data
# ============================================================================
banner(1, "the shape to memorize, and the labelled points we learn it from")
rng = np.random.default_rng(0)
pts, y = s.sample_training_data(6000, rng)
print(f"  drew {len(pts)} labelled points; inside fraction {y.mean():.3f}")

inside_fn, surface_fn, shape_name = s.get_shape()
print(f'  shape: {shape_name}')
occ_gt, og, sg = s.occupancy_grid(PANEL_RES, inside_fn=inside_fn)
v_gt, f_gt, _ = extract_mesh(occ_gt, og, sg)
gt_surf, _ = sample_mesh_surface(v_gt, f_gt, 30000)
print(f"  ground-truth mesh @ {PANEL_RES}^3: {len(v_gt)} verts, {len(f_gt)} faces")

fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
trisurf(ax, v_gt, f_gt, "#9c7bb8")
ax.set_title("The target shape\n(rippled torus: smooth ring + high-freq bumps)")
ax = fig.add_subplot(1, 2, 2, projection="3d")
inside = y > 0.5
ax.scatter(pts[inside, 0], pts[inside, 1], pts[inside, 2], s=4,
           c="#c0392b", label="inside (1)", alpha=0.6)
ax.scatter(pts[~inside, 0], pts[~inside, 1], pts[~inside, 2], s=2,
           c="#95a5a6", label="outside (0)", alpha=0.12)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=ELEV, azim=AZIM)
ax.set_title("Training data\n(points labelled inside / outside)")
ax.legend(loc="upper right", markerscale=2, framealpha=0.9)
fig.suptitle("Part 1 -- a neural occupancy field learns one shape from "
             "labelled 3D points", y=1.00)
fig.tight_layout()
show(fig, "fig4_1_shape_and_data.png")


# ============================================================================
# Part 2 -- what positional encoding does
# ============================================================================
banner(2, "positional encoding lifts a coordinate into many frequencies")
xs = np.linspace(-1, 1, 600)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
for k in range(4):
    ax.plot(xs, np.sin(2.0 ** k * np.pi * xs), label=f"sin($2^{k}\\pi x$)")
ax.plot(xs, xs, "k--", lw=1, label="raw $x$")
ax.set_xlabel("coordinate $x$"); ax.set_ylabel("feature value")
ax.set_title("Encoding bands: geometrically rising frequencies")
ax.legend(fontsize=8, loc="lower right")
ax = axes[1]
enc = F.positional_encoding(np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1), L=6)
im = ax.imshow(enc.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
               extent=[-1, 1, enc.shape[1], 0])
ax.set_xlabel("coordinate $x$"); ax.set_ylabel("encoding channel $\\gamma(x)$")
ax.set_title(f"Full encoding $\\gamma(x)$, $L=6$  ($\\to$ {F.pe_dim(6)} dims for 3D)")
fig.colorbar(im, ax=ax, fraction=0.046, label="value")
fig.suptitle("Part 2 -- positional encoding gives the MLP high-frequency raw "
             "material", y=1.00)
fig.tight_layout()
show(fig, "fig4_2_encoding.png")


# ============================================================================
# Part 3 -- train the three networks
# ============================================================================
banner(3, f"training the coordinate MLP for L in {L_VALUES} ({STEPS} steps)")
sampler = lambda n, r: s.sample_training_data(n, r, inside_fn=inside_fn, surface_fn=surface_fn)
models, hists = {}, {}
for L in L_VALUES:
    print(f"  L = {L:2d}:")
    m = F.CoordinateMLP(L=L, hidden=HIDDEN, seed=0)
    hists[L] = F.train(m, sampler, steps=STEPS, batch=BATCH, seed=0)
    models[L] = m
    print(f"    {m.n_params():,} params, {m.memory_bytes() / 1024:.0f} KiB, "
          f"{hists[L]['seconds']:.0f}s")


def smooth(a, k=50):
    return np.convolve(a, np.ones(k) / k, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for L in L_VALUES:
    axes[0].plot(smooth(hists[L]["loss"]), label=f"L = {L}")
    axes[1].plot(smooth(hists[L]["acc"]), label=f"L = {L}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("BCE loss (smoothed)")
axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("step"); axes[1].set_ylabel("batch accuracy (smoothed)")
axes[1].set_title("Batch accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("Part 3 -- more frequency bands -> lower loss, faster (spectral "
             "bias overcome)", y=1.00)
fig.tight_layout()
show(fig, "fig4_3_training.png")


# ============================================================================
# Part 4 -- the spectral-bias comparison (the hero figure)
# ============================================================================
banner(4, "extract each network's surface and compare to ground truth")
panels = [("Ground truth", v_gt, f_gt, "#9c7bb8", None)]
metrics_rows = [("ground truth", "-", 1.000, 0.0)]
for L in L_VALUES:
    occ, o, sp = F.field_to_grid(models[L], PANEL_RES)
    v, f, _ = extract_mesh(occ, o, sp)
    iou = volumetric_iou(inside_fn,
                         lambda p, mm=models[L]: mm.occupancy(p) > 0.5, n=200000)
    rec, _ = sample_mesh_surface(v, f, 30000)
    cham = chamfer_distance(gt_surf, rec) * 1e3
    label = "no encoding (L=0)" if L == 0 else f"L = {L}"
    panels.append((label, v, f, "#6699cc" if L == 10 else "#cc7755", iou))
    metrics_rows.append((f"MLP L={L}", f"{models[L].memory_bytes()/1024:.0f} KiB",
                         iou, cham))
    print(f"  L={L:2d}: IoU {iou:.3f}, Chamfer {cham:.2f}e-3, {len(f)} faces")

fig = plt.figure(figsize=(16, 4.8))
for i, (title, v, f, color, iou) in enumerate(panels):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    trisurf(ax, v, f, color)
    sub = "" if iou is None else f"\nIoU {iou:.3f}"
    ax.set_title(title + sub)
fig.suptitle("Part 4 -- spectral bias: without positional encoding the MLP "
             "smooths the bumps away; more bands recover them", y=1.02)
fig.tight_layout()
show(fig, "fig4_4_spectral_bias.png")

print("\n  representation        memory      IoU    Chamfer(x1e-3)")
print("  " + "-" * 52)
for name, mem, iou, cham in metrics_rows:
    print(f"  {name:18s} {str(mem):>9s}   {iou:5.3f}   {cham:8.2f}")


# ============================================================================
# Part 5 -- resolution independence
# ============================================================================
banner(5, "one trained network, meshed at several query resolutions")
best = models[10]
fig = plt.figure(figsize=(13, 4.6))
for i, R in enumerate([32, 64, 128]):
    occ, o, sp = F.field_to_grid(best, R)
    v, f, _ = extract_mesh(occ, o, sp)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    trisurf(ax, v, f, "#6699cc")
    ax.set_title(f"query grid {R}$^3$\n{len(v)} verts, "
                 f"net still {best.memory_bytes() / 1024:.0f} KiB")
    print(f"  meshed L=10 net at {R:3d}^3: {len(v):6d} verts "
          f"(network memory unchanged: {best.memory_bytes()/1024:.0f} KiB)")
fig.suptitle("Part 5 -- resolution is a property of the QUERY, not of the "
             "representation", y=1.00)
fig.tight_layout()
show(fig, "fig4_5_resolution.png")

# Memory comparison: the field vs voxel grids of various resolutions.
print("\n  memory: L=10 field = "
      f"{best.memory_bytes() / 1024:.0f} KiB (resolution-free)")
for R in (64, 128, 256):
    print(f"          {R:3d}^3 occupancy grid = "
          f"{grid_memory_bytes(R) / 1024:.0f} KiB (1 byte/voxel, fixed res)")

print("\nAll Section 4 parts complete.")
