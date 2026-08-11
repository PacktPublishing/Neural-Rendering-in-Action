"""
Listing 3.3 -- A minimal PointNet, from scratch
=======================================================
Point clouds are SETS: the same shape can be listed in any order, so a
network that consumes them must be PERMUTATION INVARIANT. PointNet's recipe:

    per-point shared MLP  ->  symmetric pooling (max)  ->  classifier MLP

Every per-point feature is computed independently with SHARED weights, and
max-pooling doesn't care about order -- so the whole pipeline is invariant
by construction.

To make the point unmissable, we also implement the "obvious" baseline that
flattens the (N, 3) cloud into a single 3N-vector and feeds it to an MLP.
It trains fine -- and then collapses the moment you shuffle the points.

Everything here is NumPy with hand-written backprop, so the reader can see
exactly where gradients flow (including through the max-pool argmax).
"""

import numpy as np


# ----------------------------------------------------------------------------
# 1. A procedural point-cloud dataset: spheres vs. tori vs. boxes
# ----------------------------------------------------------------------------

def _random_rotation(rng):
    """Uniform random rotation matrix via QR decomposition."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


# NOTE: each generator emits its points in a DETERMINISTIC parametric order
# (point i always means "the same place on the shape"). This consistent
# ordering is what gives the flattened-MLP baseline a fighting chance on
# unshuffled data -- making its collapse under shuffling the fair, honest
# comparison. Real scanned point clouds, of course, come in no such order.

_PHI = (1 + 5 ** 0.5) / 2          # golden ratio, for low-discrepancy lattices


def _sphere(n, rng):
    """Fibonacci sphere lattice: deterministic, near-uniform ordering."""
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    theta = 2 * np.pi * i / _PHI
    rad = np.sqrt(1 - z ** 2)
    p = np.stack([rad * np.cos(theta), rad * np.sin(theta), z], axis=1)
    return p * rng.uniform(0.7, 1.0)


def _torus(n, rng):
    """Golden-angle lattice on the (u, v) parameter torus."""
    R, r = rng.uniform(0.55, 0.75), rng.uniform(0.15, 0.3)
    i = np.arange(n)
    u = 2 * np.pi * (i + 0.5) / n
    v = 2 * np.pi * ((i / _PHI) % 1.0)
    return np.stack([(R + r * np.cos(v)) * np.cos(u),
                     (R + r * np.cos(v)) * np.sin(u),
                     r * np.sin(v)], axis=1)


def _box(n, rng):
    """Round-robin over the 6 faces, golden-ratio lattice within each face."""
    half = rng.uniform(0.5, 0.9, size=3)
    i = np.arange(n)
    face = i % 6
    axis, side = face // 2, face % 2
    k = i // 6
    s = 2 * ((k / _PHI) % 1.0) - 1                     # in [-1, 1]
    t = 2 * ((k / _PHI ** 2) % 1.0) - 1
    p = np.zeros((n, 3))
    other = np.array([[1, 2], [0, 2], [0, 1]])[axis]   # the two free axes
    p[np.arange(n), other[:, 0]] = s * half[other[:, 0]]
    p[np.arange(n), other[:, 1]] = t * half[other[:, 1]]
    p[np.arange(n), axis] = np.where(side == 0, -half[axis], half[axis])
    return p


GENERATORS = [_sphere, _torus, _box]
CLASS_NAMES = ["sphere", "torus", "box"]


def make_dataset(n_per_class=200, n_points=128, noise=0.02, seed=0):
    """Random rotations + jitter force the nets to learn shape, not pose."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, gen in enumerate(GENERATORS):
        for _ in range(n_per_class):
            p = gen(n_points, rng) @ _random_rotation(rng).T
            p += noise * rng.standard_normal(p.shape)
            X.append(p)
            y.append(label)
    X, y = np.stack(X).astype(np.float64), np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ----------------------------------------------------------------------------
# 2. Tiny building blocks: parameters, Adam, softmax cross-entropy
# ----------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def softmax_xent(logits, y):
    """Returns (loss, dlogits). Stable softmax + cross-entropy."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    dlogits = p
    dlogits[np.arange(n), y] -= 1.0
    return loss, dlogits / n


def _init(shape, rng):
    """He initialization, suitable for ReLU layers."""
    fan_in = shape[0]
    return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)


# ----------------------------------------------------------------------------
# 3. The mini-PointNet
# ----------------------------------------------------------------------------

class TinyPointNet:
    """point cloud (B, N, 3) -> logits (B, 3)

    Architecture (a faithful miniature of Qi et al., CVPR 2017):
        shared MLP 3 -> 64 -> 128   (applied to every point, same weights)
        max-pool over the N points  (the symmetric function)
        head MLP 128 -> 64 -> 3
    """

    def __init__(self, n_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": _init((3, 64), rng),    "b1": np.zeros(64),
            "W2": _init((64, 128), rng),  "b2": np.zeros(128),
            "W3": _init((128, 64), rng),  "b3": np.zeros(64),
            "W4": _init((64, n_classes), rng), "b4": np.zeros(n_classes),
        }

    def forward(self, X):
        p, c = self.p, {}
        c["X"] = X
        c["a1"] = X @ p["W1"] + p["b1"]            # (B, N, 64)
        c["h1"] = np.maximum(c["a1"], 0)
        c["a2"] = c["h1"] @ p["W2"] + p["b2"]      # (B, N, 128)
        c["h2"] = np.maximum(c["a2"], 0)
        c["amax"] = c["h2"].argmax(axis=1)         # (B, 128) winning point ids
        c["g"] = c["h2"].max(axis=1)               # (B, 128) global feature
        c["a3"] = c["g"] @ p["W3"] + p["b3"]
        c["h3"] = np.maximum(c["a3"], 0)
        logits = c["h3"] @ p["W4"] + p["b4"]
        self.cache = c
        return logits

    def backward(self, dlogits):
        p, c = self.p, self.cache
        g = {}
        g["W4"] = c["h3"].T @ dlogits
        g["b4"] = dlogits.sum(0)
        dh3 = dlogits @ p["W4"].T
        da3 = dh3 * (c["a3"] > 0)
        g["W3"] = c["g"].T @ da3
        g["b3"] = da3.sum(0)
        dg = da3 @ p["W3"].T                       # (B, 128)

        # Max-pool backward: each channel's gradient is routed entirely to
        # the single point that achieved the max -- everyone else gets zero.
        B, N, C = c["h2"].shape
        dh2 = np.zeros_like(c["h2"])
        bi = np.repeat(np.arange(B), C)
        ci = np.tile(np.arange(C), B)
        dh2[bi, c["amax"].ravel(), ci] = dg.ravel()

        da2 = dh2 * (c["a2"] > 0)
        g["W2"] = np.einsum("bnc,bnd->cd", c["h1"], da2)
        g["b2"] = da2.sum((0, 1))
        dh1 = da2 @ p["W2"].T
        da1 = dh1 * (c["a1"] > 0)
        g["W1"] = np.einsum("bnc,bnd->cd", c["X"], da1)
        g["b1"] = da1.sum((0, 1))
        return g


class FlatMLP:
    """The strawman: flatten (N, 3) -> 3N and use an ordinary MLP.
    Works on ordered data; falls apart under permutation."""

    def __init__(self, n_points=128, n_classes=3, seed=0):
        rng = np.random.default_rng(seed)
        d = n_points * 3
        self.p = {
            "W1": _init((d, 128), rng),  "b1": np.zeros(128),
            "W2": _init((128, 64), rng), "b2": np.zeros(64),
            "W3": _init((64, n_classes), rng), "b3": np.zeros(n_classes),
        }

    def forward(self, X):
        p, c = self.p, {}
        c["x"] = X.reshape(len(X), -1)
        c["a1"] = c["x"] @ p["W1"] + p["b1"]
        c["h1"] = np.maximum(c["a1"], 0)
        c["a2"] = c["h1"] @ p["W2"] + p["b2"]
        c["h2"] = np.maximum(c["a2"], 0)
        logits = c["h2"] @ p["W3"] + p["b3"]
        self.cache = c
        return logits

    def backward(self, dlogits):
        p, c = self.p, self.cache
        g = {}
        g["W3"] = c["h2"].T @ dlogits
        g["b3"] = dlogits.sum(0)
        dh2 = dlogits @ p["W3"].T
        da2 = dh2 * (c["a2"] > 0)
        g["W2"] = c["h1"].T @ da2
        g["b2"] = da2.sum(0)
        dh1 = da2 @ p["W2"].T
        da1 = dh1 * (c["a1"] > 0)
        g["W1"] = c["x"].T @ da1
        g["b1"] = da1.sum(0)
        return g


# ----------------------------------------------------------------------------
# 4. Training loop and the permutation-invariance experiment
# ----------------------------------------------------------------------------

def accuracy(model, X, y, batch=256):
    correct = 0
    for i in range(0, len(X), batch):
        logits = model.forward(X[i:i + batch])
        correct += (logits.argmax(1) == y[i:i + batch]).sum()
    return correct / len(X)


def train(model, Xtr, ytr, Xte, yte, epochs=40, batch=32, lr=1e-3, seed=0,
          verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        losses = []
        for i in range(0, len(Xtr), batch):
            idx = order[i:i + batch]
            logits = model.forward(Xtr[idx])
            loss, dlogits = softmax_xent(logits, ytr[idx])
            opt.step(model.backward(dlogits))
            losses.append(loss)
        history["loss"].append(np.mean(losses))
        history["train_acc"].append(accuracy(model, Xtr, ytr))
        history["test_acc"].append(accuracy(model, Xte, yte))
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1:3d}  loss {history['loss'][-1]:.4f}  "
                  f"train {history['train_acc'][-1]:.3f}  "
                  f"test {history['test_acc'][-1]:.3f}")
    return history


def shuffle_points(X, seed=0):
    """Independently permute the point ORDER of every cloud (geometry
    untouched). A permutation-invariant model must be unaffected."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = X[i][rng.permutation(X.shape[1])]
    return out
