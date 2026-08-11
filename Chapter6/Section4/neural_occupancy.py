"""
Listing 4.0 -- neural_occupancy.py: the whole of Section 4 in one file
======================================================================
An MLP memorizes a 3D shape.

    f_theta(x, y, z)  ->  P(the point is inside the shape)

Recipe (this file is the direct implementation of it):
  1. pick a shape with an exact inside/outside test (a knurled torus:
     a torus whose tube is carved with high-frequency ripples);
  2. sample training points -- half uniform in the bounding box, half
     jittered around the surface -- and label them inside/outside;
  3. lift coordinates with Fourier positional encoding (set L = 0 to
     watch the network fail: spectral bias);
  4. train a small ReLU MLP with binary cross-entropy;
  5. evaluate the trained network on a dense grid and run marching cubes
     to get a mesh out; render it beside the ground truth.

Self-contained on purpose: only numpy, scikit-image, matplotlib. The rest
of Section 4 (shapes4.py, field4.py, ...) re-implements these pieces with
more instrumentation; read this file first.

Run:  python neural_occupancy.py [steps=1500] [L=6]
      e.g.  python neural_occupancy.py 3000 0     <- see spectral bias
"""

import os
import sys
import time
import numpy as np
from skimage import measure
import matplotlib
if os.name == "posix" and sys.platform != "darwin" \
        and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")                       # headless machines: save only
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6
HIDDEN, BATCH, LR = (256, 256), 4096, 1e-3
BOUND = 1.10                                    # shape fits in [-BOUND, BOUND]^3
GRID = 96                                       # marching-cubes resolution


# --------------------------------------------------------------- 1. the shape
def inside(p, R=0.65, r=0.28, amp=0.30, fu=8, fv=8):
    """Exact inside test for the knurled torus. p: (N, 3) -> bool (N,)."""
    ring = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R      # distance to ring axis
    d = np.sqrt(ring ** 2 + p[:, 2] ** 2)                # distance to centre circle
    u, v = np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 2], ring)
    return d < r * (1 + amp * np.sin(fu * u) * np.sin(fv * v))


def on_surface(n, rng, R=0.65, r=0.28, amp=0.30, fu=8, fv=8):
    """n exact surface points, via the parametric form of the same torus."""
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    er = r * (1 + amp * np.sin(fu * u) * np.sin(fv * v))
    rad = R + er * np.cos(v)
    return np.stack([rad * np.cos(u), rad * np.sin(u), er * np.sin(v)], axis=1)


# ------------------------------------------------------- 2. the training data
def sample_batch(n, rng):
    """Half uniform in the box, half jittered near the surface; exact labels."""
    half = n // 2
    pts = np.vstack([rng.uniform(-BOUND, BOUND, (n - half, 3)),
                     on_surface(half, rng) + 0.05 * rng.standard_normal((half, 3))])
    return pts.astype(np.float32), inside(pts).astype(np.float32)


# ------------------------------------------------ 3. the positional encoding
def encode(p):
    """gamma(p): (N, 3) -> (N, 3 + 6L). L = 0 returns raw coordinates."""
    out = [p] + [f(2.0 ** k * np.pi * p) for k in range(L)
                 for f in (np.sin, np.cos)]
    return np.concatenate(out, axis=1).astype(np.float32)


# ----------------------------------------------------------- 4. model + loss
class MLP:
    """encode(x) -> ReLU hidden layers -> occupancy logit. Manual backprop."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # layer widths: encoded input (3 + 6L) -> hidden layers -> 1 logit
        dims = [3 + 6 * L, *HIDDEN, 1]
        # He init: std = sqrt(2/fan_in), the right scale for ReLU layers
        he = lambda i, o: (rng.standard_normal((i, o)) * np.sqrt(2 / i)
                           ).astype(np.float32)
        self.W = [he(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(d, np.float32) for d in dims[1:]]

    def forward(self, pts):
        # cache inputs/pre-activations of every layer so backward() can reuse them
        self.h = [encode(pts)]                       # h[i]: input to layer i
        self.z = []                                  # z[i]: pre-activation (hidden only)
        a = self.h[0]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b                            # affine step
            if i < len(self.W) - 1:                  # hidden: ReLU
                self.z.append(z)
                a = np.maximum(z, 0)                 # ReLU nonlinearity
                self.h.append(a)                     # becomes next layer's input
            else:                                    # output: linear logit
                return z.ravel()                     # (N, 1) -> (N,)

    def backward(self, dlogit):
        # delta = dLoss/dz for the current layer; starts as dLoss/dlogit (output)
        gW, gb, delta = [], [], dlogit[:, None]
        for i in reversed(range(len(self.W))):       # walk layers output -> input
            gW.insert(0, self.h[i].T @ delta)        # weight grad: inputᵀ · delta
            gb.insert(0, delta.sum(0))               # bias grad: sum over the batch
            if i:                                    # not the input layer -> propagate
                # pull delta back through Wᵢ, then mask by the ReLU derivative (z>0)
                delta = (delta @ self.W[i].T) * (self.z[i - 1] > 0)
        return gW, gb                                # per-layer grads, input -> output

    def prob(self, pts, chunk=65536):                # inference, batched
        out = np.empty(len(pts), np.float32)
        for i in range(0, len(pts), chunk):          # chunk to bound peak memory
            z = self.forward(pts[i:i + chunk].astype(np.float32))
            # sigmoid(logit) -> occupancy probability; clip guards exp overflow
            out[i:i + chunk] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return out


def bce(logit, y):
    """Stable BCE-with-logits. Returns (mean loss, dloss/dlogit)."""
    loss = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    return loss.mean(), (p - y) / len(y)


# ---------------------------------------------------------------- 5. training
def train():
    rng = np.random.default_rng(0)
    net = MLP()
    mW = [np.zeros_like(w) for w in net.W]; vW = [np.zeros_like(w) for w in net.W]
    mb = [np.zeros_like(b) for b in net.b]; vb = [np.zeros_like(b) for b in net.b]
    b1, b2, eps = 0.9, 0.999, 1e-8                   # Adam
    t0 = time.time()
    for step in range(1, STEPS + 1):
        pts, y = sample_batch(BATCH, rng)
        loss, dlogit = bce(net.forward(pts), y)
        gW, gb = net.backward(dlogit)
        for P, G, M, V in ((net.W, gW, mW, vW), (net.b, gb, mb, vb)):
            for i in range(len(P)):
                M[i] = b1 * M[i] + (1 - b1) * G[i]
                V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                P[i] -= LR * (M[i] / (1 - b1 ** step)) \
                        / (np.sqrt(V[i] / (1 - b2 ** step)) + eps)
        if step % 500 == 0 or step == STEPS:
            print(f"  step {step:5d}   loss {loss:.4f}   ({time.time()-t0:.0f}s)")
    n_par = sum(w.size for w in net.W) + sum(b.size for b in net.b)
    print(f"  {n_par:,} parameters = {n_par * 4 / 1024:.0f} KiB, "
          f"trained {time.time()-t0:.0f}s")
    return net


# --------------------------------------- 6. mesh extraction, metrics, display
def grid_eval(fn, res):
    xs = np.linspace(-BOUND, BOUND, res)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    vals = fn(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
    return vals.reshape(res, res, res).astype(np.float32), xs[1] - xs[0]


def to_mesh(vol, spacing):
    v, f, _, _ = measure.marching_cubes(vol, level=0.5,
                                        spacing=(spacing,) * 3)
    return v - BOUND, f


def draw(ax, verts, faces, color):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lit = 0.35 + 0.65 * np.abs(n @ np.array([0.33, -0.25, 0.91]))
    base = np.array(matplotlib.colors.to_rgb(color))
    ax.add_collection3d(Poly3DCollection(
        tri, facecolors=np.clip(lit[:, None] * base, 0, 1), edgecolor="none"))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(50, -60)


if __name__ == "__main__":
    print(f"neural occupancy field | L = {L}, hidden = {HIDDEN}, "
          f"steps = {STEPS}")
    net = train()

    print("evaluating on the grid + marching cubes ...")
    gt_vol, sp = grid_eval(lambda p: inside(p).astype(np.float32), GRID)
    nf_vol, _ = grid_eval(net.prob, GRID)
    gv, gf = to_mesh(gt_vol, sp)
    nv, nf_ = to_mesh(nf_vol, sp)

    # Monte-Carlo volumetric IoU against the exact shape
    rng = np.random.default_rng(1)
    P = rng.uniform(-BOUND, BOUND, (200_000, 3))
    A, B = inside(P), net.prob(P) > 0.5
    iou = np.logical_and(A, B).sum() / np.logical_or(A, B).sum()
    print(f"  IoU vs ground truth: {iou:.3f}")

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw(ax, gv, gf, "#9c7bb8"); ax.set_title("Ground truth (exact test)")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    draw(ax, nv, nf_, "#6699cc")
    ax.set_title(f"Neural field, L = {L}\nIoU {iou:.3f}")
    fig.suptitle("A small MLP memorizes the knurled torus "
                 f"({STEPS} steps, BCE on labelled points)")
    fig.tight_layout()
    out = f"neural_occupancy_L{L}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
