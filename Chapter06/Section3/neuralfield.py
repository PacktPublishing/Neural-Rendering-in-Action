"""
Listing 3.4 -- A neural occupancy field, from scratch
=====================================================
This module closes Section 3 by jumping ahead: instead of STORING occupancy
in a grid, we make a small MLP COMPUTE it,

    f_theta(x, y, z)  ->  probability that the point is inside the shape.

The network is the representation. Memory no longer scales with resolution
(there is no resolution), queries are continuous and differentiable by
construction, and the same trained network can be meshed at any grid size.

Two ingredients, both implemented below in plain NumPy:

  1. Fourier positional encoding gamma(x): raw (x, y, z) inputs suffer from
     "spectral bias" -- MLPs learn low frequencies first and may never fit
     fine detail. Lifting coordinates to [x, sin(2^k pi x), cos(2^k pi x)]
     for k = 0..L-1 fixes this (Tancik et al., 2020; Mildenhall et al., 2020).

  2. A 3-layer MLP trained with binary cross-entropy on randomly sampled
     points labeled inside/outside. Backprop is written out by hand, in the
     same style as Listing 3.3 (pointnet.py), so nothing is hidden.
"""

import time
import numpy as np

from pointnet import Adam, _init     # deliberate reuse of Listing 3.3 pieces


# ----------------------------------------------------------------------------
# 1. Ground truth: the analytic torus
# ----------------------------------------------------------------------------
# For training labels we use the torus's exact inside/outside test. (We could
# equally use the ray-stabbing test of Listing 3.1 on the mesh; the analytic
# form just makes the training data generator two lines long.)

def torus_inside(p, R=0.7, r=0.3):
    """True for points inside the torus with major radius R, minor radius r."""
    ring = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R
    return (ring ** 2 + p[:, 2] ** 2) < r ** 2


def sample_training_points(n, rng, R=0.7, r=0.3, bound=1.1, near_frac=0.4):
    """Mixed sampling: mostly uniform in the bounding box, plus a fraction
    concentrated near the surface, where the decision boundary lives."""
    n_near = int(n * near_frac)
    uniform = rng.uniform(-bound, bound, (n - n_near, 3))
    u = rng.uniform(0, 2 * np.pi, n_near)
    v = rng.uniform(0, 2 * np.pi, n_near)
    surface = np.stack([(R + r * np.cos(v)) * np.cos(u),
                        (R + r * np.cos(v)) * np.sin(u),
                        r * np.sin(v)], axis=1)
    near = surface + 0.08 * rng.standard_normal(surface.shape)
    pts = np.vstack([uniform, near])
    return pts, torus_inside(pts, R, r).astype(np.float64)


# ----------------------------------------------------------------------------
# 2. Fourier positional encoding
# ----------------------------------------------------------------------------

def positional_encoding(p, L=6):
    """gamma(p): (N, 3) -> (N, 3 + 6L). Includes the raw coordinates."""
    out = [p]
    for k in range(L):
        out.append(np.sin(2.0 ** k * np.pi * p))
        out.append(np.cos(2.0 ** k * np.pi * p))
    return np.concatenate(out, axis=1)


def pe_dim(L):
    return 3 + 6 * L


# ----------------------------------------------------------------------------
# 3. The coordinate MLP
# ----------------------------------------------------------------------------

class NeuralOccupancyField:
    """gamma(x, y, z) -> hidden -> hidden -> occupancy logit."""

    def __init__(self, L=6, hidden=128, seed=0):
        self.L = L
        d = pe_dim(L)
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": _init((d, hidden), rng),      "b1": np.zeros(hidden),
            "W2": _init((hidden, hidden), rng), "b2": np.zeros(hidden),
            "W3": _init((hidden, 1), rng),      "b3": np.zeros(1),
        }

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def memory_bytes(self):
        return self.n_params() * 4                       # float32 on disk

    def forward(self, pts):
        p, c = self.p, {}
        c["x"] = positional_encoding(pts, self.L)
        c["a1"] = c["x"] @ p["W1"] + p["b1"]
        c["h1"] = np.maximum(c["a1"], 0)
        c["a2"] = c["h1"] @ p["W2"] + p["b2"]
        c["h2"] = np.maximum(c["a2"], 0)
        logit = (c["h2"] @ p["W3"] + p["b3"]).ravel()
        self.cache = c
        return logit

    def backward(self, dlogit):
        p, c = self.p, self.cache
        g = {}
        d = dlogit[:, None]
        g["W3"] = c["h2"].T @ d
        g["b3"] = d.sum(0)
        dh2 = d @ p["W3"].T
        da2 = dh2 * (c["a2"] > 0)
        g["W2"] = c["h1"].T @ da2
        g["b2"] = da2.sum(0)
        dh1 = da2 @ p["W2"].T
        da1 = dh1 * (c["a1"] > 0)
        g["W1"] = c["x"].T @ da1
        g["b1"] = da1.sum(0)
        return g

    def occupancy(self, pts, batch=65536):
        """Probability of being inside, evaluated in batches (inference)."""
        out = np.empty(len(pts))
        for i in range(0, len(pts), batch):
            out[i:i + batch] = _sigmoid(self.forward(pts[i:i + batch]))
        return out


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def bce_with_logits(logit, y):
    """Numerically stable BCE. Returns (mean loss, dlogit)."""
    loss = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    return loss.mean(), (_sigmoid(logit) - y) / len(y)


# ----------------------------------------------------------------------------
# 4. Training: the MLP "memorizes" the shape
# ----------------------------------------------------------------------------

def train_field(field=None, steps=4000, batch=2048, lr=1e-3, seed=0,
                verbose=True):
    """Returns (trained field, history dict, wall-clock seconds)."""
    field = NeuralOccupancyField(seed=seed) if field is None else field
    rng = np.random.default_rng(seed)
    opt = Adam(field.p, lr=lr)
    history = {"loss": [], "acc": []}
    t0 = time.time()
    for step in range(steps):
        pts, y = sample_training_points(batch, rng)
        logit = field.forward(pts)
        loss, dlogit = bce_with_logits(logit, y)
        opt.step(field.backward(dlogit))
        history["loss"].append(loss)
        history["acc"].append(((logit > 0) == (y > 0.5)).mean())
        if verbose and (step + 1) % 1000 == 0:
            print(f"  step {step + 1:5d}  loss {loss:.4f}  "
                  f"batch acc {history['acc'][-1]:.3f}  "
                  f"({time.time() - t0:.1f} s)")
    return field, history, time.time() - t0


def field_to_grid(field, resolution, bound=1.1):
    """Evaluate the field on a dense lattice, ready for marching cubes.

    This is where resolution independence becomes tangible: the SAME network
    can be sampled at 32^3 or 256^3 -- resolution is now a property of the
    QUERY, not of the REPRESENTATION."""
    xs = np.linspace(-bound, bound, resolution)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    occ = field.occupancy(pts).reshape(resolution, resolution, resolution)
    origin = np.array([-bound] * 3)
    spacing = np.array([xs[1] - xs[0]] * 3)
    return occ.astype(np.float32), origin, spacing
