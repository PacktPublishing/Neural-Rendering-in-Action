"""
app6.py -- companion source for Section 6: Appearance Fields & Hybrids
======================================================================
Fragments of this file appear throughout the section text. Depends on
sdf5.py (Section 5) for the sphere tracer and the trained torus SDF.

Contents (section part in brackets):
  texture_albedo, radiance_gt     the appearance to memorize        [6.1]
  MLP (multi-output)              shared trainable block            [6.1]
  train_albedo_field              c(x)    -> rgb                    [6.2]
  train_radiance_field            c(x, v) -> rgb (view-dependent)   [6.3]
  FeatureGrid, HybridField        explicit grid + tiny MLP          [6.4]
  train_hybrid / train_pure       the speed race                    [6.4]
"""

import time
import numpy as np

from sdf5 import torus_sdf, knurl_inside, BOUND

RNG = np.random.default_rng


# ------------------------------------------------------------------ [6.1]
def torus_uv(p, R=0.65):
    """Surface coordinates on the torus: u around the hole, v around the tube."""
    ring = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R
    return np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 2], ring)


def torus_normal(p, R=0.65):
    rr = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) + 1e-12
    ring = rr - R
    g = np.stack([ring * p[:, 0] / rr, ring * p[:, 1] / rr, p[:, 2]], 1)
    return (g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
            ).astype(np.float32)


def torus_surface(n, rng, R=0.65, r=0.28):
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    return np.stack([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u),
                     r * np.sin(v)], 1).astype(np.float32)


def texture_albedo(p):
    """Procedural surface color: smooth checkers in (u, v). Smooth on purpose:
    hard edges are 'infinite frequency' and would confound the experiment."""
    u, v = torus_uv(p)
    s = 0.5 + 0.5 * np.sin(8 * u) * np.sin(6 * v)          # in [0, 1]
    c_a = np.array([0.85, 0.45, 0.25], np.float32)          # terracotta
    c_b = np.array([0.20, 0.45, 0.75], np.float32)          # blue
    return (c_a[None] * s[:, None] + c_b[None] * (1 - s[:, None])
            ).astype(np.float32)


LIGHT = np.array([0.5, -0.3, 0.85], np.float32)
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def radiance_gt(p, view):
    """Outgoing radiance at surface point p toward unit direction `view`
    (pointing AWAY from the surface, toward the eye): Lambert diffuse on the
    albedo texture plus a white Blinn-Phong highlight. This is the ground
    truth the view-dependent field will memorize -- shading baked into the
    function, exactly as NeRF's color head does."""
    n = torus_normal(p)
    diff = np.clip((n * LIGHT).sum(1), 0, 1)[:, None]
    h = LIGHT[None] + view                                  # Blinn half-vector
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
    spec = np.clip((n * h).sum(1), 0, 1)[:, None] ** 60
    col = texture_albedo(p) * (0.20 + 0.80 * diff) + 0.9 * spec
    return np.clip(col, 0, 1).astype(np.float32)


# ---------------------------------------------------- shared blocks [6.1]
def encode(p, L=6):
    out = [p] + [f(2.0 ** k * np.pi * p) for k in range(L)
                 for f in (np.sin, np.cos)]
    return np.concatenate(out, axis=1).astype(np.float32)


class MLP:
    """Generic ReLU MLP with (N, C_out) output and manual backprop --
    Section 4's network, generalized from 1 output to C."""

    def __init__(self, dims, seed=0, out_scale=1.0):
        rng = RNG(seed)
        he = lambda i, o: (rng.standard_normal((i, o)) * np.sqrt(2 / i)
                           ).astype(np.float32)
        self.W = [he(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(d, np.float32) for d in dims[1:]]
        self.W[-1] *= out_scale

    def forward(self, x):
        self.h, self.z = [x.astype(np.float32)], []
        a = self.h[0]
        for i in range(len(self.W)):
            zz = a @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                self.z.append(zz)
                a = np.maximum(zz, 0)
                self.h.append(a)
            else:
                return zz

    def backward(self, dout):
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        delta = dout.astype(np.float32)
        for i in reversed(range(len(self.W))):
            gW[i] = self.h[i].T @ delta
            gb[i] = delta.sum(0)
            if i:
                delta = (delta @ self.W[i].T) * (self.z[i - 1] > 0)
        self.d_input = delta @ self.W[0].T          # for upstream modules
        return gW, gb

    def tensors(self):
        return self.W + self.b

    def grads_as_list(self, gW, gb):
        return gW + gb


class Adam:
    """Adam over an arbitrary list of tensors (weights, biases, OR grids)."""

    def __init__(self, tensors, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.T, self.lr, self.b1, self.b2, self.eps = tensors, lr, b1, b2, eps
        self.m = [np.zeros_like(t) for t in tensors]
        self.v = [np.zeros_like(t) for t in tensors]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            self.T[i] -= self.lr * (self.m[i] / (1 - self.b1 ** self.t)) / \
                (np.sqrt(self.v[i] / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------ [6.2]
def train_albedo_field(steps=800, batch=4096, L=6, seed=0, verbose=True):
    """c_theta(x) -> rgb, trained on surface points with MSE against the
    procedural texture. The geometry network never changes -- appearance is
    a SECOND field over the same coordinates."""
    net = MLP([3 + 6 * L, 128, 128, 3], seed=seed)
    opt = Adam(net.tensors())
    rng = RNG(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        p = torus_surface(batch, rng)
        pred = net.forward(encode(p, L))
        gt = texture_albedo(p)
        mse = ((pred - gt) ** 2).mean()
        opt.step(net.grads_as_list(*net.backward(2 * (pred - gt) / pred.size)))
        if verbose and step % 200 == 0:
            print(f"  step {step:4d}  MSE {mse:.5f}  ({time.time()-t0:.0f}s)")
    return net


# ------------------------------------------------------------------ [6.3]
def train_radiance_field(steps=1200, batch=4096, L=6, seed=0, verbose=True):
    """c_theta(x, v) -> rgb. Identical recipe; the ONLY change is three more
    input columns -- the view direction -- and suddenly the field can
    represent specular highlights that move with the camera."""
    net = MLP([3 + 6 * L + 3, 128, 128, 3], seed=seed)
    opt = Adam(net.tensors())
    rng = RNG(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        p = torus_surface(batch, rng)
        v = rng.standard_normal((batch, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[(v * torus_normal(p)).sum(1) < 0] *= -1          # outward hemisphere
        x = np.concatenate([encode(p, L), v], axis=1)
        pred = net.forward(x)
        gt = radiance_gt(p, v)
        mse = ((pred - gt) ** 2).mean()
        opt.step(net.grads_as_list(*net.backward(2 * (pred - gt) / pred.size)))
        if verbose and step % 300 == 0:
            print(f"  step {step:4d}  MSE {mse:.5f}  ({time.time()-t0:.0f}s)")
    return net


# ------------------------------------------------------------------ [6.4]
class FeatureGrid:
    """A dense G^3 grid of learnable C-dim feature vectors, queried by
    trilinear interpolation. The grid IS a parameter tensor: gradients
    scatter back into the 8 corners of each queried cell with the same
    trilinear weights used in the forward pass."""

    def __init__(self, res=32, feat=4, seed=0, scale=0.1):
        self.res, self.feat = res, feat
        self.F = (scale * RNG(seed).standard_normal((res, res, res, feat))
                  ).astype(np.float32)

    def lookup(self, pts):
        g = (pts + BOUND) / (2 * BOUND) * (self.res - 1)     # -> grid coords
        g = np.clip(g, 0, self.res - 1 - 1e-4)
        i0 = g.astype(np.int64)
        t = (g - i0).astype(np.float32)
        self._cache = (i0, t)
        out = np.zeros((len(pts), self.feat), np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.abs(1 - dx - t[:, 0]) * np.abs(1 - dy - t[:, 1])
                         * np.abs(1 - dz - t[:, 2]))[:, None]
                    out += w * self.F[i0[:, 0] + dx, i0[:, 1] + dy,
                                      i0[:, 2] + dz]
        return out

    def backward(self, dfeat):
        i0, t = self._cache
        gF = np.zeros_like(self.F)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.abs(1 - dx - t[:, 0]) * np.abs(1 - dy - t[:, 1])
                         * np.abs(1 - dz - t[:, 2]))[:, None]
                    np.add.at(gF, (i0[:, 0] + dx, i0[:, 1] + dy,
                                   i0[:, 2] + dz), w * dfeat)
        return gF


class HybridField:
    """Feature grid -> tiny MLP -> occupancy logit. The grid stores WHERE
    things are; the small network only has to DECODE features, so it can be
    an order of magnitude smaller (and faster) than Section 4's MLP."""

    def __init__(self, res=32, feat=4, hidden=32, seed=0):
        self.grid = FeatureGrid(res, feat, seed)
        self.head = MLP([feat + 3, hidden, hidden, 1], seed=seed + 1)

    def forward(self, pts):
        f = self.grid.lookup(pts)
        self._pts = pts.astype(np.float32)
        x = np.concatenate([f, self._pts], axis=1)   # raw xyz helps the head
        return self.head.forward(x).ravel()

    def backward(self, dlogit):
        gW, gb = self.head.backward(dlogit[:, None])
        gF = self.grid.backward(self.head.d_input[:, :self.grid.feat])
        return [self.grid.F] + self.head.tensors(), \
               [gF] + self.head.grads_as_list(gW, gb)

    def value(self, pts, chunk=131072):
        out = np.empty(len(pts), np.float32)
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = self.forward(pts[i:i + chunk])
        return out

    def n_params(self):
        return self.grid.F.size + sum(t.size for t in self.head.tensors())


def _bce(logit, y):
    loss = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    return loss.mean(), ((p - y) / len(y)).astype(np.float32)


def sample_occ(n, rng):
    half = n // 2
    from sdf5 import knurl_surface
    pts = np.vstack([rng.uniform(-BOUND, BOUND, (n - half, 3)),
                     knurl_surface(half, rng)
                     + 0.05 * rng.standard_normal((half, 3))]).astype(np.float32)
    return pts, knurl_inside(pts).astype(np.float32)


def train_occupancy(model, params, backward, seconds=30.0, batch=4096,
                    seed=0, tag=""):
    """Train ANY occupancy model for a fixed wall-clock budget; returns a
    (time, batch-accuracy) curve so architectures race fairly."""
    opt = Adam(params)
    rng = RNG(seed)
    curve, t0, step = [], time.time(), 0
    while time.time() - t0 < seconds:
        step += 1
        pts, y = sample_occ(batch, rng)
        logit = model.forward(pts)
        loss, dlogit = _bce(logit, y)
        tensors, grads = backward(dlogit)
        opt.T = tensors
        opt.step(grads)
        curve.append((time.time() - t0,
                      float(((logit > 0) == (y > 0.5)).mean())))
    print(f"  {tag}: {step} steps in {seconds:.0f}s "
          f"({1000*seconds/step:.1f} ms/step)")
    return np.array(curve), step
