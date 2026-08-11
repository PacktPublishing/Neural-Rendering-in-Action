"""
sdf5.py -- companion source for Section 5: Neural Signed Distance Functions
===========================================================================
Fragments of this file appear throughout the section text. Self-contained:
numpy + scikit-image + matplotlib only.

Contents (section part in brackets):
  torus_sdf, knurl_surface      exact SDF / surface sampler          [5.1]
  SDFNet                        coordinate MLP, linear output        [5.2]
  clamped_l1, train_supervised  DeepSDF-style regression             [5.2]
  SDFNet.input_gradient         analytic normals d f / d x           [5.3]
  sphere_trace, render          direct rendering, no mesh            [5.4]
  train_eikonal                 SDF from surface points alone        [5.5]
  smooth_union                  CSG on distance fields               [5.6]
"""

import time
import numpy as np

BOUND = 1.10


# ------------------------------------------------------------------ [5.1]
def torus_sdf(p, R=0.65, r=0.28):
    """EXACT signed distance to a smooth torus: negative inside."""
    ring = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R
    return np.sqrt(ring ** 2 + p[:, 2] ** 2) - r


def knurl_surface(n, rng, R=0.65, r=0.28, amp=0.30, fu=8, fv=8):
    """Exact surface samples of the KNURLED torus (whose true SDF has no
    closed form -- the point of Part 5.5)."""
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    er = r * (1 + amp * np.sin(fu * u) * np.sin(fv * v))
    rad = R + er * np.cos(v)
    return np.stack([rad * np.cos(u), rad * np.sin(u), er * np.sin(v)], 1)


def knurl_inside(p, R=0.65, r=0.28, amp=0.30, fu=8, fv=8):
    """Exact inside test of the knurled torus (for scoring only)."""
    ring = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R
    d = np.sqrt(ring ** 2 + p[:, 2] ** 2)
    u, v = np.arctan2(p[:, 1], p[:, 0]), np.arctan2(p[:, 2], ring)
    return d < r * (1 + amp * np.sin(fu * u) * np.sin(fv * v))


# ------------------------------------------------------------------ [5.2]
class SDFNet:
    """gamma(x) -> ReLU hidden layers -> signed distance (linear output).

    Identical body to Section 4's occupancy network; only the output's
    meaning changed: an unbounded distance instead of an inside logit.
    """

    def __init__(self, L=6, hidden=(256, 256), seed=0, out_bias=0.0):
        self.L = L
        rng = np.random.default_rng(seed)
        dims = [3 + 6 * L, *hidden, 1]
        he = lambda i, o: (rng.standard_normal((i, o)) * np.sqrt(2 / i)
                           ).astype(np.float32)
        self.W = [he(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(d, np.float32) for d in dims[1:]]
        self.W[-1] *= 0.1                 # start predictions near 0: inside the
                                          # clamp band, so gradients can flow
        self.b[-1][:] = out_bias          # geometric-style init: start "outside"

    # -- encoding ---------------------------------------------------------
    def encode(self, p):
        out = [p] + [f(2.0 ** k * np.pi * p) for k in range(self.L)
                     for f in (np.sin, np.cos)]
        return np.concatenate(out, axis=1).astype(np.float32)

    # -- forward / backward (same pattern as Listing 4.2) ------------------
    def forward(self, pts):
        self.h, self.z = [self.encode(pts)], []
        a = self.h[0]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            zz = a @ W + b
            if i < len(self.W) - 1:
                self.z.append(zz)
                a = np.maximum(zz, 0)
                self.h.append(a)
            else:
                return zz.ravel()

    def snapshot(self):                     # caches, for multi-pass losses
        return (list(self.h), list(self.z))

    def backward(self, dout, cache=None):
        h, z = cache if cache is not None else (self.h, self.z)
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        delta = dout[:, None].astype(np.float32)
        for i in reversed(range(len(self.W))):
            gW[i] = h[i].T @ delta
            gb[i] = delta.sum(0)
            if i:
                delta = (delta @ self.W[i].T) * (z[i - 1] > 0)
        return gW, gb

    def value(self, pts, chunk=65536):       # batched inference
        out = np.empty(len(pts), np.float32)
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = self.forward(
                pts[i:i + chunk].astype(np.float32))
        return out

    # ------------------------------------------------------------- [5.3]
    def input_gradient(self, pts):
        """Analytic spatial gradient  d f / d(x, y, z)  -- the surface normal
        direction, for free. Two chained steps:
          (a) backpropagate d f / d gamma through the MLP layers;
          (b) multiply by the encoding's own Jacobian d gamma / d x
              (sin and cos have closed-form derivatives).
        This is exactly what autograd would assemble for us in PyTorch.
        """
        pts = pts.astype(np.float32)
        self.forward(pts)
        # (a) df/d(encoding): ordinary backprop with dout = 1
        delta = np.ones((len(pts), 1), np.float32)
        for i in reversed(range(1, len(self.W))):
            delta = (delta @ self.W[i].T) * (self.z[i - 1] > 0)
        d_enc = delta @ self.W[0].T                       # (N, 3 + 6L)
        # (b) chain through gamma: raw block + each sin/cos band
        grad = d_enc[:, 0:3].copy()
        for k in range(self.L):
            w = np.float32(2.0 ** k * np.pi)
            s, c = 3 + 6 * k, 3 + 6 * k + 3               # sin / cos blocks
            grad += w * (d_enc[:, s:s + 3] * np.cos(w * pts)
                         - d_enc[:, c:c + 3] * np.sin(w * pts))
        return grad

    def normals(self, pts, chunk=65536):
        out = np.empty((len(pts), 3), np.float32)
        for i in range(0, len(pts), chunk):
            g = self.input_gradient(pts[i:i + chunk])
            out[i:i + chunk] = g / (np.linalg.norm(g, axis=1, keepdims=True)
                                    + 1e-12)
        return out


# -- Adam over the (W, b) lists ------------------------------------------
class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.mW = [np.zeros_like(w) for w in net.W]
        self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]
        self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb):
        self.t += 1
        for P, G, M, V in ((self.net.W, gW, self.mW, self.vW),
                           (self.net.b, gb, self.mb, self.vb)):
            for i in range(len(P)):
                M[i] = self.b1 * M[i] + (1 - self.b1) * G[i]
                V[i] = self.b2 * V[i] + (1 - self.b2) * G[i] ** 2
                P[i] -= self.lr * (M[i] / (1 - self.b1 ** self.t)) / \
                        (np.sqrt(V[i] / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------ [5.2]
def sample_sdf_batch(n, rng, delta_band=0.08):
    """Points + exact signed distances for the smooth torus: half uniform in
    the box, half hugging the surface where precision matters most."""
    half = n // 2
    u = rng.uniform(0, 2 * np.pi, half)
    v = rng.uniform(0, 2 * np.pi, half)
    surf = np.stack([(0.65 + 0.28 * np.cos(v)) * np.cos(u),
                     (0.65 + 0.28 * np.cos(v)) * np.sin(u),
                     0.28 * np.sin(v)], 1)
    pts = np.vstack([rng.uniform(-BOUND, BOUND, (n - half, 3)),
                     surf + delta_band * rng.standard_normal((half, 3))])
    return pts.astype(np.float32), torus_sdf(pts).astype(np.float32)


def clamped_l1(pred, gt, delta=0.10):
    """DeepSDF's loss: L1 between distances CLAMPED to [-delta, +delta].
    Errors far from the surface are capped, so the network spends its
    capacity where the level set actually lives. Returns (loss, dpred)."""
    cp, cg = np.clip(pred, -delta, delta), np.clip(gt, -delta, delta)
    diff = cp - cg
    grad = np.sign(diff) * (np.abs(pred) < delta)     # clamp kills gradient
    return np.abs(diff).mean(), grad.astype(np.float32) / len(pred)


def train_supervised(steps=1500, batch=4096, L=6, seed=0, beta_far=0.2,
                     verbose=True):
    """Loss = clamped_l1  +  beta_far * plain L1.

    The clamp focuses precision near the surface (DeepSDF); the small plain-L1
    term does two jobs: it keeps gradients alive even when a prediction sits
    outside the clamp band, and it keeps FAR-FIELD distances roughly correct,
    which sphere tracing (Part 5.4) will rely on for large safe steps."""
    net = SDFNet(L=L, seed=seed)
    opt = Adam(net)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        pts, gt = sample_sdf_batch(batch, rng)
        pred = net.forward(pts)
        loss_c, d_c = clamped_l1(pred, gt)
        d_far = np.sign(pred - gt).astype(np.float32) / len(pred)
        opt.step(*net.backward(d_c + beta_far * d_far))
        if verbose and step % 500 == 0:
            print(f"  step {step:5d}  clamped-L1 {loss_c:.4f} "
                  f" plain-L1 {np.abs(pred - gt).mean():.4f} "
                  f" ({time.time()-t0:.0f}s)")
    return net


# ------------------------------------------------------------------ [5.4]
def sphere_trace(sdf, origins, dirs, t_start=0.0, eps=1e-3, t_max=5.0,
                 max_steps=96, step_scale=0.9):
    """March each ray by the SDF value: the distance is a guaranteed-safe
    step. Returns (t, hit_mask). step_scale < 1 adds a safety margin for
    fields that are only approximately unit-gradient (e.g., ours)."""
    t = np.full(len(dirs), t_start, np.float32)
    hit = np.zeros(len(dirs), bool)
    alive = np.ones(len(dirs), bool)
    for _ in range(max_steps):
        if not alive.any():
            break
        p = origins[alive] + t[alive, None] * dirs[alive]
        d = sdf(p)
        newly_hit = d < eps
        idx = np.where(alive)[0]
        hit[idx[newly_hit]] = True
        t[alive] += step_scale * np.maximum(d, 0.5 * eps)
        alive[idx[newly_hit]] = False
        alive[t > t_max] = False
    return t, hit


def make_camera(eye, target=(0, 0, 0), up=(0, 0, 1), width=240, height=180,
                fov=40.0):
    eye = np.asarray(eye, np.float32); target = np.asarray(target, np.float32)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    a = np.tan(np.radians(fov) / 2)
    xs = np.linspace(-a, a, width) * width / height
    ys = np.linspace(a, -a, height)
    X, Y = np.meshgrid(xs, ys)
    dirs = (X[..., None] * right + Y[..., None] * upv + fwd).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(eye, dirs.shape).astype(np.float32)
    return origins, dirs.astype(np.float32), (height, width)


def render(sdf, normals_fn, eye=(1.9, -1.9, 1.5), width=240, height=180,
           light=(0.5, -0.3, 0.85), base=(0.42, 0.6, 0.8)):
    """Sphere-trace a full image and shade hits with Lambertian lighting
    from the field's own gradient. No mesh anywhere in this pipeline."""
    origins, dirs, shape = make_camera(eye, width=width, height=height)
    t, hit = sphere_trace(sdf, origins, dirs)
    img = np.ones((*shape, 3), np.float32)              # white background
    if hit.any():
        p = origins[hit] + t[hit, None] * dirs[hit]
        n = normals_fn(p)
        l = np.asarray(light, np.float32); l /= np.linalg.norm(l)
        lam = np.clip(n @ l, 0, 1)[:, None]
        col = np.asarray(base, np.float32) * (0.25 + 0.75 * lam)
        flat = img.reshape(-1, 3); flat[hit] = col
        img = flat.reshape(*shape, 3)
    return np.clip(img, 0, 1), hit.reshape(shape)


# ------------------------------------------------------------------ [5.5]
def train_eikonal(surface_pts, steps=1200, batch_s=1024, batch_u=1024,
                  L=6, lam_eik=0.1, lam_rep=0.02, alpha=50.0, fd_h=5e-3,
                  warmup=200, seed=0, verbose=True, net=None):
    """Learn an SDF from SURFACE POINTS ALONE (IGR-style):

        loss =  mean |f(surface)|                    (surface sticks to 0)
              + lam_eik * mean (||grad f|| - 1)^2    (eikonal: be a distance)
              + lam_rep * mean exp(-alpha |f(space)|)  (don't be 0 everywhere)

    The eikonal term needs d loss / d grad f -- a second-order derivative.
    Autograd gives it in one line; here we approximate grad f by central
    finite differences and backpropagate through all six shifted forward
    passes, which is exactly the same computation spelled out by hand.
    """
    fresh = net is None
    if fresh:
        net = SDFNet(L=L, seed=seed, out_bias=0.3)   # start "outside"
    opt = Adam(net)
    rng = np.random.default_rng(seed)

    # -- geometric warm-up: pull the field toward a sphere SDF first ---------
    # A randomly initialized field has arbitrary sign structure; the eikonal
    # loss can then lock in spurious interior pockets. 200 cheap supervised
    # steps toward ||x|| - r0 give the field one clean inside region to
    # deform, which is the practical essence of IGR's "geometric init".
    for _ in range(warmup if fresh else 0):
        p = rng.uniform(-BOUND, BOUND, (2048, 3)).astype(np.float32)
        gt = (np.linalg.norm(p, axis=1) - 0.75).astype(np.float32)
        pred = net.forward(p)
        opt.step(*net.backward(np.sign(pred - gt).astype(np.float32) / len(p)))

    shifts = np.concatenate([np.eye(3, dtype=np.float32),
                             -np.eye(3, dtype=np.float32)])   # +-x, +-y, +-z
    t0 = time.time()
    for step in range(1, steps + 1):
        # -- surface term ---------------------------------------------------
        sp = surface_pts[rng.integers(0, len(surface_pts), batch_s)]
        sp = (sp + 0.002 * rng.standard_normal(sp.shape)).astype(np.float32)
        f_s = net.forward(sp)
        loss_s = np.abs(f_s).mean()
        gW, gb = net.backward(np.sign(f_s).astype(np.float32) / len(f_s))

        # -- space points: eikonal + repulsion -------------------------------
        up = np.vstack([rng.uniform(-BOUND, BOUND, (batch_u // 2, 3)),
                        sp[:batch_u // 2] +
                        0.1 * rng.standard_normal((batch_u // 2, 3))
                        ]).astype(np.float32)
        f_u = net.forward(up)
        cache_u = net.snapshot()
        vals, caches = [], []
        for s in shifts:                       # six shifted forward passes
            vals.append(net.forward(up + fd_h * s))
            caches.append(net.snapshot())
        vals = np.stack(vals)                                  # (6, N)
        g = (vals[:3] - vals[3:]) / (2 * fd_h)                 # (3, N)
        gnorm = np.sqrt((g ** 2).sum(0)) + 1e-9
        res = gnorm - 1.0
        loss_e = (res ** 2).mean()
        # d loss_e / d f(x + h e_i) = +- lam * 2 res * g_i / gnorm / (N 2h)
        coef = lam_eik * 2 * res / gnorm / len(up) / (2 * fd_h)
        for i, s in enumerate(shifts):
            sign = 1.0 if i < 3 else -1.0
            dv = (sign * coef * g[i % 3]).astype(np.float32)
            gWi, gbi = net.backward(dv, cache=caches[i])
            for j in range(len(gW)):
                gW[j] += gWi[j]; gb[j] += gbi[j]
        # repulsion (uses the unshifted pass)
        loss_r = np.exp(-alpha * np.abs(f_u)).mean()
        d_rep = (lam_rep * (-alpha) * np.sign(f_u)
                 * np.exp(-alpha * np.abs(f_u)) / len(f_u)).astype(np.float32)
        gWr, gbr = net.backward(d_rep, cache=cache_u)
        for j in range(len(gW)):
            gW[j] += gWr[j]; gb[j] += gbr[j]

        opt.step(gW, gb)
        if verbose and step % 300 == 0:
            print(f"  step {step:5d}  surf {loss_s:.4f}  eik {loss_e:.4f} "
                  f" rep {loss_r:.4f}  ({time.time()-t0:.0f}s)")
    return net


# ------------------------------------------------------------------ [5.6]
def smooth_union(d1, d2, k=0.08):
    """Polynomial smooth minimum (Quilez): blends two distance fields into
    one shape with a fillet of radius ~k where they meet."""
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def sphere_sdf(p, c=(0.55, 0.0, 0.42), r=0.34):
    return np.linalg.norm(p - np.asarray(c, np.float32), axis=1) - r
