# Section 5 — Neural Signed Distance Functions: Smooth Geometry with Implicit Surfaces

Section 4 taught a network to answer a yes/no question: *is this point inside
the shape?* That answer is all-or-nothing — a point one millimeter outside the
surface and a point one meter outside produce the same label, so the field is
informative only where it changes. In this section we upgrade the question to
*how far is this point from the surface, and on which side?* The answer is the
**signed distance function** (SDF): negative inside, positive outside, zero
exactly on the surface. Every point in space now carries useful information,
and three practical superpowers follow directly: the gradient of the field *is*
the surface normal; rays can be marched through the field with guaranteed-safe
step sizes (sphere tracing), so we can render the shape without ever building
a mesh; and shapes combine by simple arithmetic on their distances.

All code fragments in this section come from the companion file `sdf5.py`,
which — like Listing 4.0 — depends only on NumPy, scikit-image, and matplotlib.

## 5.1 A distance field you can hold in your hand

Before training anything, it pays to look at one exact SDF. The smooth torus
has a closed form: collapse the point onto the plane of the ring, measure how
far it sits from the ring's center circle, and subtract the tube radius.

```python
def torus_sdf(p, R=0.65, r=0.28):
    """EXACT signed distance to a smooth torus: negative inside."""
    ring = np.sqrt(p[:, 0]**2 + p[:, 1]**2) - R
    return np.sqrt(ring**2 + p[:, 2]**2) - r
```

Two things make this function special, and both are properties we will demand
of its neural replacement. First, its zero level set is the surface, so
*shape* is encoded as *where a scalar field crosses zero*. Second, its
gradient has unit length everywhere (`‖∇f‖ = 1`, the **eikonal property**),
which is just the statement that walking one meter changes your distance to
the surface by at most one meter. Keep the eikonal property in mind: it is the
mathematical fingerprint of a true distance function, and in Part 5.5 it will
become a training objective.

Note what we *cannot* write down: the exact SDF of Section 4's knurled torus.
Its inside/outside test was easy, but the distance to a rippled surface has no
closed form. That gap is deliberate — it is exactly the gap Part 5.5 closes.

## 5.2 Recipe: a neural SDF by regression

The network is Section 4's coordinate MLP with one change that costs zero
lines of code and changes everything about the output's meaning: the final
layer stays **linear**. An occupancy network squashed its output through a
sigmoid because it predicted a probability; a distance can be any real number.

Training data follows the Section 4 pattern — half the batch uniform in the
bounding box, half hugging the surface — except each point now carries an
exact distance instead of a binary label:

```python
pts = np.vstack([rng.uniform(-BOUND, BOUND, (n - half, 3)),
                 surf + delta_band * rng.standard_normal((half, 3))])
return pts.astype(np.float32), torus_sdf(pts).astype(np.float32)
```

For the loss, DeepSDF's recipe is an L1 distance between predictions and
targets that have both been **clamped** to a band `[-δ, +δ]` around the
surface:

```python
def clamped_l1(pred, gt, delta=0.10):
    cp, cg = np.clip(pred, -delta, delta), np.clip(gt, -delta, delta)
    diff = cp - cg
    grad = np.sign(diff) * (np.abs(pred) < delta)   # clamp kills gradient
    return np.abs(diff).mean(), grad.astype(np.float32) / len(pred)
```

The intuition: we do not care whether a point far from the shape reads 0.8 or
0.9 meters — we care intensely whether a point near the surface reads +2 mm or
−2 mm, because that is where the level set (and therefore the geometry) lives.
The clamp caps far-field errors so the network spends its capacity in the band
that matters.

**A bug worth its own paragraph.** Our first training run flat-lined: the loss
froze at 0.056 and the level-set IoU was exactly zero. The commented line
above is the culprit — the clamp's gradient is zero whenever `|pred| > δ`, and
with standard He initialization the network's initial outputs land mostly
*outside* the ±0.1 band. The network was born dead: no gradient could reach
it. The fix is two-fold, and both halves are standard practice worth knowing.
Initialize the last layer small, so predictions start near zero, inside the
band:

```python
self.W[-1] *= 0.1     # start predictions near 0: inside the clamp band
```

and blend in a lightly-weighted *unclamped* L1 term, which keeps gradients
alive everywhere and — a second job that pays off in Part 5.4 — keeps
far-field distances roughly correct so a ray marcher can take large steps:

```python
loss_c, d_c = clamped_l1(pred, gt)
d_far = np.sign(pred - gt).astype(np.float32) / len(pred)
opt.step(*net.backward(d_c + 0.2 * d_far))
```

With the fix, the same 76K-parameter network (positional encoding L = 6, two
hidden layers of 256) trains in about 50 seconds of pure NumPy: clamped-L1
falls from 0.056 to **0.0014**, the level set matches the true torus at **IoU
0.985**, and the mean absolute distance error over the whole box is 5.5 mm on
a shape roughly two meters across. One diagnostic is new compared to Section
4, and it is the SDF-specific one: near the surface, `‖∇f‖` averages within
0.2 of unit length — the network approximately learned the eikonal property
without ever being told about it, simply because the ground-truth distances it
regressed satisfy it.

## 5.3 Normals for free: differentiating with respect to the *input*

Everything we trained in Sections 4–5 backpropagated with respect to the
*weights*. An SDF makes a different derivative interesting: the gradient with
respect to the **input coordinates**, because at the surface `∇f / ‖∇f‖` is
the outward surface normal. In PyTorch this is one `autograd.grad` call; in
our from-scratch NumPy code it is two chained steps that show exactly what
autograd assembles for us. Step (a) is an ordinary backward pass, except we
stop at the encoding instead of continuing into the first weight matrix; step
(b) multiplies by the encoding's own Jacobian, which is closed-form because
sines and cosines have textbook derivatives:

```python
# (a) df/d(encoding): backprop with dout = 1, stop at the input
delta = np.ones((len(pts), 1), np.float32)
for i in reversed(range(1, len(self.W))):
    delta = (delta @ self.W[i].T) * (self.z[i - 1] > 0)
d_enc = delta @ self.W[0].T                       # (N, 3 + 6L)

# (b) chain through gamma: d sin(wx)/dx = w cos(wx), etc.
grad = d_enc[:, 0:3].copy()
for k in range(self.L):
    w = np.float32(2.0**k * np.pi)
    s, c = 3 + 6*k, 3 + 6*k + 3                   # sin / cos blocks
    grad += w * (d_enc[:, s:s+3] * np.cos(w * pts)
                 - d_enc[:, c:c+3] * np.sin(w * pts))
```

We verified this against a float64 twin of the network with tiny central
differences: median disagreement 2 × 10⁻⁵. The verification itself carries two
lessons. In float32, finite differences of a network are *noisy* — the
subtraction cancels most significant bits, and dividing by 2h amplifies what
remains, so a naive float32 check "fails" with errors around 0.05 even though
the analytic gradient is exact. And at ReLU kinks the two answers genuinely
differ, because the analytic gradient is one-sided while the stencil straddles
the kink. Both effects argue for the analytic gradient in production: it is
exact, deterministic, and costs one backward pass instead of six forward
passes.

## 5.4 Recipe: rendering without a mesh — sphere tracing

Here is the payoff that no occupancy field can offer. To find where a ray hits
the surface, march along it — and at every step, the SDF value *is* a
certificate that the nearest surface is at least that far away, so it is a
guaranteed-safe step size. Big values in empty space mean big steps; the march
automatically decelerates as it approaches the surface:

```python
for _ in range(max_steps):
    p = origins[alive] + t[alive, None] * dirs[alive]
    d = sdf(p)
    hit[idx[d < eps]] = True                    # close enough: a hit
    t[alive] += step_scale * np.maximum(d, 0.5 * eps)
```

The one engineering nuance is `step_scale = 0.9`: our *learned* field is only
approximately eikonal (‖∇f‖ runs up to ~1.2 in places), so it can slightly
overestimate the safe distance; scaling steps down by 10–20% restores the
safety guarantee at the cost of a few extra iterations. Shading then falls out
of Part 5.3 — the hit points' normals are one `input_gradient` call, and
Lambert's cosine law does the rest:

```python
n = normals_fn(p_hit)
lam = np.clip(n @ light_dir, 0, 1)[:, None]
color = base * (0.25 + 0.75 * lam)
```

Figure 5.1 shows the result: the analytic torus and the neural torus rendered
by the *same* tracer, side by side, plus the neural normals visualized as RGB.
There is no mesh anywhere in that image — no marching cubes, no triangles, no
rasterizer. The network is queried about 67,500 rays × ~20 steps ≈ 1.4 million
times, which takes 5 seconds in NumPy (the analytic SDF takes 0.1 s; the gap
is purely the cost of evaluating a network, and it is the gap that Section 6's
hybrid representations attack).

## 5.5 Recipe: an SDF from points alone — the eikonal loss

Now the section's hardest and most modern trick. Suppose you have only a point
cloud — a laser scan, say — with no distances at all. Regression is off the
table. But recall the two fingerprints of an SDF: it is zero *at* the surface,
and its gradient has unit norm *everywhere*. Both are checkable without ground
truth, so both can be losses (this is Implicit Geometric Regularization, Gropp
et al. 2020), plus one guard term that vetoes the trivial all-zero solution:

```python
loss =  mean |f(surface_points)|              # the surface sticks to zero
      + lam_eik * mean (‖∇f‖ - 1)**2          # eikonal: BE a distance field
      + lam_rep * mean exp(-alpha * |f|)      # don't be zero everywhere
```

The eikonal term hides a subtlety: its gradient with respect to the weights is
a derivative *of a derivative* — second-order backpropagation. PyTorch grants
this with `create_graph=True`; our NumPy code makes the mechanics visible by
approximating ∇f with central differences and backpropagating through all six
shifted forward passes:

```python
for s in shifts:                       # ±x, ±y, ±z: six forward passes
    vals.append(net.forward(up + fd_h * s)); caches.append(net.snapshot())
g = (vals[:3] - vals[3:]) / (2 * fd_h)          # finite-difference ∇f
res = np.sqrt((g**2).sum(0)) - 1.0              # eikonal residual
for i, s in enumerate(shifts):                  # backprop through each pass
    dv = (±1) * lam_eik * 2*res * g[i%3] / ‖g‖ / (N * 2*fd_h)
    accumulate(net.backward(dv, cache=caches[i]))
```

Training from points alone is genuinely harder than regression, and we hit —
and will show you — its two classic failure modes. First run: the field
learned to be *nearly flat*, hugging zero everywhere (value range −0.03 to
+0.07 across the whole box). It had satisfied the surface term perfectly and
simply declined to become a distance field; the cure was raising the eikonal
and repulsion weights and enlarging the finite-difference step (small steps
drown in float32 noise). Second: with random initialization the sign structure
of the field is arbitrary, and the optimization can lock in spurious interior
pockets. The standard cure is **geometric initialization** — start the field
as a sphere's SDF so there is one clean inside to deform — which we implement
as a 200-step supervised warm-up toward `‖x‖ − 0.75` before the eikonal
training begins.

We aimed this machinery at the shape regression could not touch: the knurled
torus, supervised by nothing but 30,000 surface samples. The trajectory tells
the honest story — level-set IoU against the reference shape climbed 0.38 →
0.60 → **0.69** across three 900-step chunks (about a minute each) and was
still rising when we stopped. Figure 5.3 shows the state at 2,700 steps: the
torus, its hole, and the emerging knurls are all recovered from coordinates
alone, alongside some spurious floating sheets that longer optimization
erodes. Published IGR-class results run one to two orders of magnitude more
optimization on GPUs with exact autograd gradients; the recipe is the same,
only the budget differs.

## 5.6 Shapes as arithmetic: CSG on distance fields

A final gift of the distance representation: solid modeling becomes
arithmetic. The union of two shapes is `min(d₁, d₂)`, intersection is `max`,
subtraction is `max(d₁, −d₂)` — and replacing the hard `min` with a polynomial
**smooth minimum** (Quilez) welds the shapes together with a fillet of radius
~k:

```python
def smooth_union(d1, d2, k=0.08):
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)
```

Nothing in that function knows or cares that `d1` comes from a trained neural
network and `d2` from a four-line analytic sphere. Figure 5.2 sphere-traces
`smooth_union(neural_torus, sphere)` at k = 0.02, 0.10, 0.25: the same two
shapes, welded with a growing fillet, edited *after* training without touching
a single weight. Try doing that to a triangle mesh — boolean operations on
meshes are a notoriously fragile corner of geometry processing, and here they
are one line. The same blending idea, applied between two *learned* fields
with an interpolation weight, is shape morphing, and it is the doorway to
DeepSDF's latent shape spaces — one network, many shapes, a code per shape —
which we leave as this section's capstone exercise.

## 5.7 What it costs, and what comes next

| experiment | supervision | steps | time (NumPy) | result |
|---|---|---|---|---|
| regression SDF (smooth torus) | exact distances | 1,500 | ~50 s | IoU 0.985, L1 5.5 mm |
| sphere-traced render, 300×225 | — | ~20/ray | ~5 s | Fig. 5.1, no mesh |
| eikonal SDF (knurled torus) | 30K surface points only | 2,700 | ~3 min | IoU 0.69, improving |

The pattern of the whole chapter repeats one level up: richer supervision
buys faster convergence (distances ≫ points), and every query is a network
evaluation, which is why rendering costs seconds instead of milliseconds. Both
pressures — training cost and query cost — point at the same door: Section 6's
hybrid representations, which keep the continuous, differentiable,
CSG-friendly interface of this section while storing most of the shape in
fast, explicit feature grids.

**Exercises.** (1) Retrain the regression SDF with δ = 0.02 and δ = 0.5 and
sphere-trace both: too tight a clamp corrupts the far field the tracer needs;
too loose wastes capacity off-surface. (2) Log ‖∇f‖ statistics during
*regression* training and watch the eikonal property emerge without an eikonal
loss. (3) Replace the finite-difference ∇f in `train_eikonal` with the
analytic `input_gradient` (you will need one more backward pass through Part
5.3's chain — derive it) and measure the speedup. (4) Train a second SDF on
the sphere and morph: render `(1−t)·f_torus + t·f_sphere` for t ∈ [0, 1]; then
read the DeepSDF paper and explain how a latent code turns this trick into a
shape *space*.
