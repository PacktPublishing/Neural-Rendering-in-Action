"""
make_figures5.py -- reproduces Figures 5.1-5.3 from the shipped checkpoints.
============================================================================
This is where sdf5.render() is actually called. Run AFTER the checkpoints
exist (they ship with the section; to retrain from scratch see section5.md):

    py make_figures5.py

Needs: sdf5.py, ckpt_sdf_supervised.npz, ckpt_sdf_eikonal.npz in this folder.
Writes: fig5_1_spheretrace.png, fig5_2_csg.png, fig5_3_eikonal.png
"""

import os
import sys
import numpy as np
import matplotlib
if os.name == "posix" and sys.platform != "darwin" \
        and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import sdf5 as S


def load_sdfnet(path):
    d = np.load(path)
    net = S.SDFNet(L=int(d["L"]))
    net.W = [np.ascontiguousarray(d[f"W{i}"]) for i in range(len(net.W))]
    net.b = [np.ascontiguousarray(d[f"b{i}"]) for i in range(len(net.b))]
    for w in net.W:                       # flush denormals (see Section 4)
        w[np.abs(w) < 1e-20] = 0
    return net


def torus_normals(q):
    """Analytic normals of the smooth torus, for the reference render."""
    rr = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) + 1e-12
    ring = rr - 0.65
    g = np.stack([ring * q[:, 0] / rr, ring * q[:, 1] / rr, q[:, 2]], 1)
    return (g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
            ).astype(np.float32)


def show(fig, name):
    fig.savefig(name, dpi=130, bbox_inches="tight")
    print(f"  saved {name}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5.1 -- sphere-traced renders: analytic vs neural, plus neural normals
# ---------------------------------------------------------------------------
def figure_5_1(net, W=300, H=225):
    # render() call site #1: the analytic reference
    img_gt, _ = S.render(S.torus_sdf, torus_normals, width=W, height=H)
    # render() call site #2: the NEURAL field, shaded by its own gradient
    img_nf, hit = S.render(lambda p: net.value(p), net.normals,
                           width=W, height=H)
    # third panel: the normals themselves, visualized as RGB
    origins, dirs, shape = S.make_camera((1.9, -1.9, 1.5), width=W, height=H)
    t, h = S.sphere_trace(lambda p: net.value(p), origins, dirs,
                          step_scale=0.8)
    img_n = np.ones((*shape, 3), np.float32)
    p = origins[h] + t[h, None] * dirs[h]
    flat = img_n.reshape(-1, 3)
    flat[h] = 0.5 * net.normals(p) + 0.5
    img_n = flat.reshape(*shape, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, im, title in zip(axes, [img_gt, img_nf, img_n], [
            "Analytic SDF, sphere-traced\n(reference)",
            "NEURAL SDF, sphere-traced\n(no mesh anywhere in this image)",
            "Neural normals  $\\nabla f_\\theta/\\|\\nabla f_\\theta\\|$"
            "\n(analytic input gradient)"]):
        ax.imshow(im); ax.set_axis_off(); ax.set_title(title, fontsize=10)
    fig.suptitle("Figure 5.1 — rendering the neural SDF directly by "
                 "sphere tracing", y=1.03)
    fig.tight_layout()
    show(fig, "fig5_1_spheretrace.png")


# ---------------------------------------------------------------------------
# Figure 5.2 -- CSG: smooth union of the neural torus and an analytic sphere
# ---------------------------------------------------------------------------
def figure_5_2(net, W=300, H=225):
    def comp(k):
        return lambda p: S.smooth_union(net.value(p), S.sphere_sdf(p), k)

    def comp_normals(k, h=2e-3):
        # The blended field has no single network to backprop through, so we
        # take its normals by central differences of the composite.
        def nf(p):
            g = np.zeros((len(p), 3), np.float32)
            f = comp(k)
            for d in range(3):
                e = np.zeros(3, np.float32); e[d] = h
                g[:, d] = (f(p + e) - f(p - e)) / (2 * h)
            return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        return nf

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, k in zip(axes, [0.02, 0.10, 0.25]):
        # render() call sites #3-5: the composite field
        im, _ = S.render(comp(k), comp_normals(k), width=W, height=H,
                         base=(0.78, 0.55, 0.38))
        ax.imshow(im); ax.set_axis_off()
        ax.set_title(f"smooth_union, $k={k}$", fontsize=10)
    fig.suptitle("Figure 5.2 — CSG on distance fields: neural torus $\\cup$ "
                 "analytic sphere, growing fillet radius", y=1.03)
    fig.tight_layout()
    show(fig, "fig5_2_csg.png")


# ---------------------------------------------------------------------------
# Figure 5.3 -- the eikonal (from points alone) result, via marching cubes
# ---------------------------------------------------------------------------
def figure_5_3(net_eik, res=96):
    def grid(fn):
        xs = np.linspace(-1.1, 1.1, res)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        v = fn(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1
                        ).astype(np.float32))
        return v.reshape(res, res, res).astype(np.float32), xs[1] - xs[0]

    def mesh(vol, sp, level):
        v, f, _, _ = measure.marching_cubes(vol, level=level,
                                            spacing=(sp,) * 3)
        return v - 1.1, f

    def draw(ax, v, f, color):
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        lit = 0.35 + 0.65 * np.abs(n @ np.array([0.33, -0.25, 0.91]))
        base = np.array(matplotlib.colors.to_rgb(color))
        ax.add_collection3d(Poly3DCollection(
            tri, facecolors=np.clip(lit[:, None] * base, 0, 1),
            edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(50, -60)

    gt_vol, sp = grid(lambda p: S.knurl_inside(p).astype(np.float32))
    gv, gf = mesh(gt_vol, sp, 0.5)
    nf_vol, _ = grid(net_eik.value)
    nv, nf = mesh(nf_vol, sp, 0.0)          # SDF: surface at level ZERO
    rng = np.random.default_rng(3)
    pts = S.knurl_surface(4000, rng)

    fig = plt.figure(figsize=(13.5, 4.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 2],
               cmap="coolwarm")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(50, -60)
    ax.set_title("The ONLY input:\nsurface points (no distances)", fontsize=10)
    ax = fig.add_subplot(1, 3, 2, projection="3d"); draw(ax, nv, nf, "#cc8855")
    ax.set_title("Learned SDF, zero level set", fontsize=10)
    ax = fig.add_subplot(1, 3, 3, projection="3d"); draw(ax, gv, gf, "#9c7bb8")
    ax.set_title("Reference shape\n(for comparison only)", fontsize=10)
    fig.suptitle("Figure 5.3 — an SDF from points alone: surface + eikonal + "
                 "repulsion losses", y=1.03)
    fig.tight_layout()
    show(fig, "fig5_3_eikonal.png")


if __name__ == "__main__":
    print("loading checkpoints ...")
    net_sup = load_sdfnet("ckpt_sdf_supervised.npz")
    net_eik = load_sdfnet("ckpt_sdf_eikonal.npz")
    print("Figure 5.1 (sphere tracing):"); figure_5_1(net_sup)
    print("Figure 5.2 (CSG):");            figure_5_2(net_sup)
    print("Figure 5.3 (eikonal):");        figure_5_3(net_eik)
    print("done.")
