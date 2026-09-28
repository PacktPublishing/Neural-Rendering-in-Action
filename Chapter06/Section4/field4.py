"""
Listings 4.2 & 4.3 -- The coordinate network, with and without encoding
=======================================================================
This is the model that "memorizes" the shape: a multilayer perceptron

    f_theta : (x, y, z)  ->  logit, then sigmoid -> P(inside).

Two listings live here:

  3.2  positional encoding + the MLP architecture
  3.3  the training loop (binary cross-entropy on sampled points)

The single most important experimental knob is L, the number of Fourier
frequency bands in the positional encoding. L = 0 means "raw coordinates,
no encoding" -- the baseline that, in Part 4, fails to reproduce the
ripples. Everything else about the network is held fixed as L changes, so
any difference in the result is attributable to the encoding alone.

We reuse the Adam optimizer from Listing 3.3 to underline that nothing here
is special-purpose -- it is an ordinary little neural network.
"""

import time
import numpy as np

# Adam and _init are inlined from Listing 3.3 so that Section 4 runs on its own
# without importing from the Section 3 folder.


class Adam:
    """Adam optimizer over a flat dict of parameter arrays."""
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


def _init(shape, rng):
    """He initialization, suitable for ReLU layers."""
    return rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])


# ----------------------------------------------------------------------------
# Listing 4.2a -- Fourier positional encoding
# ----------------------------------------------------------------------------
# A ReLU MLP fed raw coordinates exhibits "spectral bias": it learns low
# frequencies quickly and high frequencies extremely slowly, if at all
# (Rahaman et al. 2019; Tancik et al. 2020). The fix is to lift each
# coordinate into a bank of sinusoids at geometrically spaced frequencies,
#
#   gamma(p) = [ p,
#                sin(2^0 pi p), cos(2^0 pi p),
#                sin(2^1 pi p), cos(2^1 pi p),
#                ...
#                sin(2^{L-1} pi p), cos(2^{L-1} pi p) ].
#
# High-frequency sinusoids give the network high-frequency "raw material" it
# can combine linearly, so it no longer has to manufacture detail out of
# piecewise-linear ReLU ramps.

def positional_encoding(p, L):
    """gamma(p): (N, 3) -> (N, 3 + 6L). L = 0 returns the raw coordinates."""
    out = [p]
    for k in range(L):
        out.append(np.sin(2.0 ** k * np.pi * p))
        out.append(np.cos(2.0 ** k * np.pi * p))
    return np.concatenate(out, axis=1)


def pe_dim(L):
    return 3 + 6 * L


# ----------------------------------------------------------------------------
# Listing 4.2b -- a generic-depth coordinate MLP with manual backprop
# ----------------------------------------------------------------------------

class CoordinateMLP:
    """gamma(x) -> [hidden, ReLU] x n -> logit.

    Weights live in a flat dict so the Section 3 Adam optimizer can step them
    directly. Backprop is written out so the chapter can trace every gradient;
    it works for any number of hidden layers.
    """

    def __init__(self, L=10, hidden=(256, 256, 256), seed=0):
        self.L = L
        rng = np.random.default_rng(seed)
        dims = [pe_dim(L)] + list(hidden) + [1]
        self.p, self.n_layers = {}, len(dims) - 1
        for i in range(self.n_layers):
            self.p[f"W{i}"] = _init((dims[i], dims[i + 1]), rng).astype(np.float32)
            self.p[f"b{i}"] = np.zeros(dims[i + 1], dtype=np.float32)

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def memory_bytes(self):
        return self.n_params() * 4                    # float32

    def forward(self, pts):
        a = positional_encoding(pts, self.L).astype(np.float32)
        self.acts = [a]                               # input to each layer
        self.pre = []                                 # pre-activations (hidden)
        for i in range(self.n_layers):
            z = a @ self.p[f"W{i}"] + self.p[f"b{i}"]
            if i < self.n_layers - 1:                 # hidden layer: ReLU
                self.pre.append(z)
                a = np.maximum(z, 0.0)
                self.acts.append(a)
            else:                                     # output layer: linear
                logit = z.ravel()
        return logit

    def backward(self, dlogit):
        g = {}
        delta = dlogit[:, None]                       # grad wrt output logit
        for i in reversed(range(self.n_layers)):
            a_in = self.acts[i]
            g[f"W{i}"] = a_in.T @ delta
            g[f"b{i}"] = delta.sum(0)
            if i > 0:                                 # propagate through ReLU
                delta = (delta @ self.p[f"W{i}"].T) * (self.pre[i - 1] > 0)
        return g

    def occupancy(self, pts, batch=65536):
        out = np.empty(len(pts))
        for i in range(0, len(pts), batch):
            out[i:i + batch] = _sigmoid(self.forward(pts[i:i + batch]))
        return out

    def save(self, path):
        np.savez(path, L=self.L, n_layers=self.n_layers, **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls.__new__(cls)
        m.L = int(d["L"]); m.n_layers = int(d["n_layers"])
        m.p = {}
        for k in d.files:
            if k.startswith(("W", "b")):
                w = np.ascontiguousarray(d[k], dtype=np.float32)
                w[np.abs(w) < 1e-20] = 0.0        # flush denormals (slow on CPU)
                m.p[k] = w
        return m


def _sigmoid(z):
    with np.errstate(over="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))


def bce_with_logits(logit, y):
    """Numerically stable binary cross-entropy. Returns (mean loss, dlogit)."""
    loss = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    return loss.mean(), (_sigmoid(logit) - y) / len(y)


# ----------------------------------------------------------------------------
# Listing 4.3 -- the training loop
# ----------------------------------------------------------------------------

def train(model, data_sampler, steps=4000, batch=4096, lr=1e-3, seed=0,
          verbose=True):
    """Fit `model` to a shape.

    data_sampler(n, rng) -> (points (n,3), labels (n,)) is the only thing that
    knows what shape we are learning, so the same loop trains on the rippled
    torus, the bunny, or anything else.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    history = {"loss": [], "acc": []}
    t0 = time.time()
    for step in range(steps):
        pts, y = data_sampler(batch, rng)
        pts = pts.astype(np.float32); y = y.astype(np.float32)
        logit = model.forward(pts)
        loss, dlogit = bce_with_logits(logit, y)
        opt.step(model.backward(dlogit))
        history["loss"].append(loss)
        history["acc"].append(float(((logit > 0) == (y > 0.5)).mean()))
        if verbose and (step + 1) % 1000 == 0:
            print(f"    step {step + 1:5d}  loss {loss:.4f}  "
                  f"batch acc {history['acc'][-1]:.3f}  "
                  f"({time.time() - t0:.1f}s)")
    history["seconds"] = time.time() - t0
    return history


def field_to_grid(model, resolution, bound=1.10):
    """Evaluate the trained field on a dense lattice, ready for marching cubes.

    The SAME network can be sampled at any resolution -- this is where the
    'continuous, resolution-free' claim becomes concrete (Part 5)."""
    xs = np.linspace(-bound, bound, resolution)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    occ = model.occupancy(pts).reshape([resolution] * 3).astype(np.float32)
    origin = np.array([-bound] * 3)
    spacing = np.array([xs[1] - xs[0]] * 3)
    return occ, origin, spacing
