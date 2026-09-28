"""
Section 3 driver: six experiments, each opening a display window.
=================================================================
Run:  python3 run_section3.py            (windows open one per experiment)
      python3 run_section3.py --headless (no windows; figures only)

Every experiment also writes its figure to ./figures and prints timing,
memory, and accuracy statistics so the COST of each representation is as
visible as its output. Experiment 6 is the centerpiece: classical and
neural representations of the same shape, side by side, with a metrics
table.

Chapter mapping: Listing 3.1/3.2 = geometry.py, 3.3 = pointnet.py,
3.4 = neuralfield.py, 3.5 = metrics.py, 3.6 = this driver.
"""

import os
import sys
import time
import numpy as np
import matplotlib

# --- display handling --------------------------------------------------------
# On a desktop this script opens an interactive window per experiment
# (rotate the 3D views with the mouse!). On a headless machine -- a server,
# CI, a container -- there is no display, so we fall back to the Agg backend
# and only save figures. The chapter text can simply say "a window appears".
HEADLESS = ("--headless" in sys.argv) or (
    os.name == "posix" and sys.platform != "darwin"
    and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry import (make_torus_mesh, voxelize_mesh, trilinear_interpolate,
                      trilinear_gradient, extract_mesh, sample_mesh_surface,
                      grid_memory_bytes, mesh_memory_bytes, mlp_memory_bytes)
import pointnet as pn
import neuralfield as nf
from metrics import volumetric_iou, chamfer_distance

os.makedirs("figures", exist_ok=True)
np.random.seed(0)


def show(fig, name):
    """Save the figure, then (if a display exists) open it in a window."""
    path = os.path.join("figures", name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [figure saved to {path}]")
    if not HEADLESS:
        plt.show()          # blocks until the reader closes the window
    plt.close(fig)


def trisurf(ax, verts, faces, color="#6699cc", elev=55, azim=-60):
    """Render a mesh with simple Lambertian shading: each face's brightness
    is proportional to the angle between its normal and a fixed light
    direction -- the same normals-from-cross-products computation used by
    `sample_mesh_surface` in Listing 3.1."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.3, 0.85]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)            # two-sided shading
    base = np.array(matplotlib.colors.to_rgb(color))
    facecolors = np.clip(shade[:, None] * base[None, :], 0, 1)
    mesh = Poly3DCollection(tri, facecolors=facecolors, edgecolor="none")
    ax.add_collection3d(mesh)
    lo, hi = verts.min(), verts.max()
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


def banner(n, title):
    print("\n" + "=" * 70 + f"\nExperiment {n}: {title}\n" + "=" * 70)


# ============================================================================
# Experiment 1 -- mesh -> occupancy grid -> mesh round trip
# ============================================================================

def experiment_1_voxelization(verts, faces):
    banner(1, "mesh -> occupancy grid -> mesh round trip")
    print(f"input mesh: {len(verts)} vertices, {len(faces)} triangles, "
          f"{mesh_memory_bytes(verts, faces) / 1024:.1f} KiB")

    resolutions = [8, 16, 32, 64]
    grids, vox_times = {}, {}
    for R in resolutions:
        t0 = time.time()
        occ, origin, spacing = voxelize_mesh(verts, faces, resolution=R)
        vox_times[R] = time.time() - t0
        grids[R] = (occ, origin, spacing)
        print(f"  R={R:3d}: voxelized in {vox_times[R]:5.2f} s | "
              f"{int(occ.sum()):7d}/{R ** 3:7d} voxels occupied | "
              f"{grid_memory_bytes(R) / 1024:7.1f} KiB at 1 B/voxel")

    fig = plt.figure(figsize=(16, 7))
    for col, R in enumerate(resolutions):
        occ, origin, spacing = grids[R]
        ax = fig.add_subplot(2, 4, col + 1, projection="3d")
        ax.voxels(occ.astype(bool), facecolors="#cc7755",
                  edgecolor="k", linewidth=0.1)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=55, azim=-60)
        ax.set_title(f"occupancy, $R={R}$\n"
                     f"({grid_memory_bytes(R) / 1024:.1f} KiB, "
                     f"{vox_times[R]:.2f} s)")
        ax = fig.add_subplot(2, 4, col + 5, projection="3d")
        t0 = time.time()
        mv, mf, _ = extract_mesh(occ, origin, spacing)
        trisurf(ax, mv, mf)
        ax.set_title(f"marching cubes ({time.time() - t0:.2f} s)\n"
                     f"{len(mv)} verts, {len(mf)} faces")
    fig.suptitle("A torus, voxelized: resolution buys fidelity at cubic cost "
                 "(top: voxel centers, bottom: marching-cubes round trip)")
    fig.tight_layout()
    show(fig, "fig1_voxelization.png")

    # Topology check: Euler characteristic chi = V - E + F = 2 - 2g; a torus
    # (genus 1) has chi = 0. Does the hole survive the round trip?
    for R in (16, 64):
        occ, origin, spacing = grids[R]
        mv, mf, _ = extract_mesh(occ, origin, spacing)
        edges = np.vstack([mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]])
        n_edges = len(np.unique(np.sort(edges, axis=1), axis=0))
        chi = len(mv) - n_edges + len(mf)
        print(f"  R={R}: Euler characteristic = {chi} "
              f"(torus expects 0 -> genus preserved: {chi == 0})")
    return grids, vox_times


# ============================================================================
# Experiment 2 -- the cubic memory curse
# ============================================================================

def experiment_2_memory(verts, faces):
    banner(2, "memory vs. resolution")
    Rs = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    occ_b = np.array([grid_memory_bytes(R, 1) for R in Rs])
    sdf_b = np.array([grid_memory_bytes(R, 4) for R in Rs])
    mesh_b = mesh_memory_bytes(verts, faces)
    # the exact network trained in Experiment 6: PE L=6, two hidden 128 layers
    field_b = mlp_memory_bytes(d_in=nf.pe_dim(6), hidden=128, n_hidden=2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(Rs, occ_b / 2 ** 20, "o-", label="occupancy grid (1 B/voxel)")
    ax.loglog(Rs, sdf_b / 2 ** 20, "s-", label="float32 value grid")
    ax.axhline(mesh_b / 2 ** 20, color="green", ls="--",
               label=f"torus mesh ({mesh_b / 1024:.0f} KiB)")
    ax.axhline(field_b / 2 ** 20, color="purple", ls=":",
               label=f"neural occupancy field, Exp. 6 ({field_b / 1024:.0f} KiB)")
    ax.set_xlabel("grid resolution $R$ (voxels per axis)")
    ax.set_ylabel("memory (MiB)")
    ax.set_title("The cubic memory curse: grids pay $O(R^3)$ for detail;\n"
                 "meshes and neural fields pay only for the surface/function")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    print(f"  mesh = {mesh_b / 1024:.0f} KiB | neural field = "
          f"{field_b / 1024:.0f} KiB | 512^3 float grid = "
          f"{grid_memory_bytes(512, 4) / 2 ** 30:.2f} GiB")
    show(fig, "fig2_memory.png")


# ============================================================================
# Experiment 3 -- continuous, differentiable queries of a discrete grid
# ============================================================================

def experiment_3_trilinear(grids):
    banner(3, "trilinear interpolation: a continuous field from a grid")
    occ, origin, spacing = grids[32]
    lo3 = origin - 0.5 * spacing
    hi3 = origin + (np.array(occ.shape) - 0.5) * spacing

    n_fine = 256
    xs = np.linspace(lo3[0], hi3[0], n_fine)
    ys = np.linspace(lo3[1], hi3[1], n_fine)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    q = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)
    t0 = time.time()
    vals_tri = trilinear_interpolate(occ, origin, spacing, q)
    t_query = time.time() - t0
    print(f"  {len(q):,} continuous queries of the 32^3 grid in "
          f"{t_query * 1e3:.0f} ms "
          f"({len(q) / t_query / 1e6:.1f} M queries/s)")
    vals_tri = vals_tri.reshape(n_fine, n_fine)
    gi = np.clip(np.round((q - origin) / spacing).astype(int), 0,
                 np.array(occ.shape) - 1)
    vals_nn = occ[gi[:, 0], gi[:, 1], gi[:, 2]].reshape(n_fine, n_fine)

    t = np.linspace(-1.15, 1.15, 600)
    line = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    prof_tri = trilinear_interpolate(occ, origin, spacing, line)
    gi = np.clip(np.round((line - origin) / spacing).astype(int), 0,
                 np.array(occ.shape) - 1)
    prof_nn = occ[gi[:, 0], gi[:, 1], gi[:, 2]]

    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (200, 3))
    g_an = trilinear_gradient(occ, origin, spacing, pts)
    eps, g_fd = 1e-5, np.zeros_like(g_an)
    for d in range(3):
        dp = np.zeros(3); dp[d] = eps
        g_fd[:, d] = (trilinear_interpolate(occ, origin, spacing, pts + dp)
                      - trilinear_interpolate(occ, origin, spacing, pts - dp)
                      ) / (2 * eps)
    err = np.abs(g_an - g_fd).max()
    print(f"  gradient check: max |analytic - finite difference| = {err:.2e}")

    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(vals_nn.T, origin="lower",
              extent=[lo3[0], hi3[0], lo3[1], hi3[1]], cmap="viridis")
    ax.set_title("nearest-neighbour lookup\n(blocky, zero gradient a.e.)")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(vals_tri.T, origin="lower",
              extent=[lo3[0], hi3[0], lo3[1], hi3[1]], cmap="viridis")
    ax.set_title("trilinear interpolation\n(continuous, differentiable)")
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, prof_nn, label="nearest", lw=1, drawstyle="steps-mid")
    ax.plot(t, prof_tri, label="trilinear", lw=2)
    ax.set_xlabel("x (probe along y = z = 0)"); ax.set_ylabel("occupancy")
    ax.set_title(f"1D probe through the tube\ngradient check err = {err:.1e}")
    ax.legend()
    fig.suptitle("From a $32^3$ grid to a continuous field: interpolation is "
                 "the bridge to neural fields", y=1.02)
    fig.tight_layout()
    show(fig, "fig3_trilinear.png")


# ============================================================================
# Experiment 4 -- point clouds from the mesh surface
# ============================================================================

def experiment_4_pointclouds(verts, faces):
    banner(4, "area-weighted surface sampling")
    fig = plt.figure(figsize=(13, 4.5))
    for col, n in enumerate([256, 1024, 4096]):
        t0 = time.time()
        pts, _ = sample_mesh_surface(verts, faces, n)
        dt = time.time() - t0
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2.5,
                   c=pts[:, 2], cmap="coolwarm")
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=55, azim=-60)
        ax.set_title(f"$n = {n}$ points\n"
                     f"({n * 3 * 4 / 1024:.0f} KiB, {dt * 1e3:.1f} ms)")
        print(f"  sampled {n:5d} points in {dt * 1e3:6.1f} ms "
              f"({n * 3 * 4 / 1024:5.1f} KiB as float32)")
    fig.suptitle("Point clouds: memory follows the surface, not the volume "
                 "-- but the data is unstructured and orderless")
    fig.tight_layout()
    show(fig, "fig4_pointclouds.png")


# ============================================================================
# Experiment 5 -- permutation invariance is an architectural property
# ============================================================================

def experiment_5_pointnet():
    banner(5, "mini-PointNet vs. flattened MLP under point shuffling")
    N_POINTS = 128
    Xtr, ytr = pn.make_dataset(n_per_class=200, n_points=N_POINTS, seed=0)
    Xte, yte = pn.make_dataset(n_per_class=60, n_points=N_POINTS, seed=1)
    print(f"dataset: {len(Xtr)} train / {len(Xte)} test clouds, "
          f"{N_POINTS} points each, classes = {pn.CLASS_NAMES}")

    print("\ntraining TinyPointNet (shared MLP + max-pool):")
    t0 = time.time()
    pnet = pn.TinyPointNet(seed=0)
    hist_pn = pn.train(pnet, Xtr, ytr, Xte, yte, epochs=40, seed=0)
    t_pn = time.time() - t0
    print("\ntraining FlatMLP (flatten to 3N vector):")
    t0 = time.time()
    flat = pn.FlatMLP(n_points=N_POINTS, seed=0)
    hist_fl = pn.train(flat, Xtr, ytr, Xte, yte, epochs=40, seed=0)
    t_fl = time.time() - t0
    n_pn = sum(v.size for v in pnet.p.values())
    n_fl = sum(v.size for v in flat.p.values())
    print(f"\n  PointNet: {n_pn:,} params, trained in {t_pn:.1f} s | "
          f"FlatMLP: {n_fl:,} params, trained in {t_fl:.1f} s")

    Xte_shuf = pn.shuffle_points(Xte, seed=2)
    acc = {"PointNet\nordered":  pn.accuracy(pnet, Xte, yte),
           "PointNet\nshuffled": pn.accuracy(pnet, Xte_shuf, yte),
           "FlatMLP\nordered":   pn.accuracy(flat, Xte, yte),
           "FlatMLP\nshuffled":  pn.accuracy(flat, Xte_shuf, yte)}
    d_pn = np.abs(pnet.forward(Xte) - pnet.forward(Xte_shuf)).max()
    d_fl = np.abs(flat.forward(Xte) - flat.forward(Xte_shuf)).max()
    for k, v in acc.items():
        print(f"  {k.replace(chr(10), ' / '):22s}: {v:.3f}")
    print(f"  max |logit change| under shuffling: "
          f"PointNet {d_pn:.2e} | FlatMLP {d_fl:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(hist_pn["test_acc"], label="TinyPointNet (test)", lw=2)
    ax.plot(hist_fl["test_acc"], label="FlatMLP (test)", lw=2)
    ax.plot(hist_pn["train_acc"], "--", c="C0", alpha=0.5, label="PointNet (train)")
    ax.plot(hist_fl["train_acc"], "--", c="C1", alpha=0.5, label="FlatMLP (train)")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title("Both models learn the ordered data...")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    bars = ax.bar(range(4), list(acc.values()),
                  color=["C0", "C0", "C1", "C1"])
    for i, b in enumerate(bars):
        b.set_alpha(1.0 if i % 2 == 0 else 0.55)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{list(acc.values())[i]:.2f}", ha="center")
    ax.axhline(1 / 3, color="gray", ls=":", label="chance (3 classes)")
    ax.set_xticks(range(4)); ax.set_xticklabels(list(acc.keys()))
    ax.set_ylim(0, 1.12); ax.set_ylabel("test accuracy")
    ax.set_title("...but only the symmetric architecture\n"
                 "survives point shuffling")
    ax.legend()
    fig.suptitle("Permutation invariance must be built into the architecture, "
                 "not hoped for")
    fig.tight_layout()
    show(fig, "fig5_pointnet.png")


# ============================================================================
# Experiment 6 -- THE PAYOFF: classical vs. neural, side by side
# ============================================================================

def experiment_6_side_by_side(verts, faces, grids, vox_times):
    banner(6, "classical vs. neural representations of the same shape")

    # --- build the neural representation ------------------------------------
    print("training the neural occupancy field "
          "(MLP memorizes the torus from labeled random points):")
    field, hist, t_train = nf.train_field(steps=4000, batch=2048, seed=0)
    print(f"  {field.n_params():,} parameters "
          f"({field.memory_bytes() / 1024:.1f} KiB), "
          f"trained in {t_train:.1f} s")

    # --- mesh every representation, timing the extraction -------------------
    R_grid = 32                                  # the classical grid to show
    R_field = 128                                # the SAME network, meshed fine
    occ_g, og, sg = grids[R_grid]
    t0 = time.time(); mv_g, mf_g, _ = extract_mesh(occ_g, og, sg)
    t_ext_g = time.time() - t0
    t0 = time.time()
    occ_f, of, sf = nf.field_to_grid(field, R_field)
    mv_f, mf_f, _ = extract_mesh(occ_f, of, sf)
    t_ext_f = time.time() - t0

    # --- metrics against the analytic ground truth --------------------------
    gt_pts, _ = sample_mesh_surface(verts, faces, 20000)
    rows = []
    for name, build_t, mem_b, ext_t, mverts, mfaces, inside_fn in [
        ("ground-truth mesh", 0.0, mesh_memory_bytes(verts, faces),
         0.0, verts, faces, nf.torus_inside),
        (f"voxel grid R={R_grid}", vox_times[R_grid],
         grid_memory_bytes(R_grid), t_ext_g, mv_g, mf_g,
         lambda p: trilinear_interpolate(occ_g, og, sg, p) > 0.5),
        (f"neural field @{R_field}^3", t_train, field.memory_bytes(),
         t_ext_f, mv_f, mf_f, lambda p: field.occupancy(p) > 0.5),
    ]:
        iou = volumetric_iou(nf.torus_inside, inside_fn)
        rec_pts, _ = sample_mesh_surface(mverts, mfaces, 20000)
        cham = chamfer_distance(gt_pts, rec_pts)
        rows.append((name, build_t, mem_b / 1024, ext_t,
                     len(mverts), len(mfaces), iou, cham * 1e3))

    print("\n  representation       build(s)  mem(KiB)  mesh(s)   verts"
          "   faces    IoU   Chamfer(x1e-3)")
    print("  " + "-" * 95)
    for r in rows:
        print(f"  {r[0]:20s} {r[1]:8.1f} {r[2]:9.1f} {r[3]:8.2f} "
              f"{r[4]:7d} {r[5]:7d} {r[6]:6.3f} {r[7]:10.2f}")
    print("\n  (build = voxelization or network training; "
          "mesh = marching-cubes extraction)")

    # --- the window ----------------------------------------------------------
    fig = plt.figure(figsize=(15, 5.4))
    panels = [("Ground truth: triangle mesh", verts, faces, "#88aa66", rows[0]),
              (f"Classical: voxel grid, $R={R_grid}$\n+ marching cubes",
               mv_g, mf_g, "#cc7755", rows[1]),
              (f"Neural: occupancy MLP\nmeshed at ${R_field}^3$",
               mv_f, mf_f, "#6699cc", rows[2])]
    for i, (title, v, f, color, r) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        trisurf(ax, v, f, color=color, elev=35, azim=-65)
        ax.set_title(f"{title}\nmem {r[2]:.0f} KiB | IoU {r[6]:.3f} | "
                     f"Chamfer {r[7]:.1f}e-3", fontsize=10)
    fig.suptitle("The same torus, three ways: explicit surface, discrete "
                 "volume, continuous neural function", y=0.99)
    fig.subplots_adjust(top=0.80)
    show(fig, "fig6_side_by_side.png")

    # bonus statistic: resolution independence in one line
    for Rq in (32, 64, 128):
        t0 = time.time()
        occ_q, oq, sq = nf.field_to_grid(field, Rq)
        mvq, mfq, _ = extract_mesh(occ_q, oq, sq)
        print(f"  same network meshed at {Rq:3d}^3: {len(mvq):6d} verts in "
              f"{time.time() - t0:5.1f} s (memory unchanged: "
              f"{field.memory_bytes() / 1024:.1f} KiB)")


if __name__ == "__main__":
    print(f"display mode: {'headless (figures only)' if HEADLESS else 'interactive windows'}")
    verts, faces = make_torus_mesh()
    grids, vox_times = experiment_1_voxelization(verts, faces)
    experiment_2_memory(verts, faces)
    experiment_3_trilinear(grids)
    experiment_4_pointclouds(verts, faces)
    experiment_5_pointnet()
    experiment_6_side_by_side(verts, faces, grids, vox_times)
    print("\nAll Section 3 experiments complete.")
