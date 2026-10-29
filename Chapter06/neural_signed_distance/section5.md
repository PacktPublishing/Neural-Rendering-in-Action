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

The companion code is `neural_sdf.py` (the Python driver) and `neural_sdf.slang`
(the network, every training kernel, and the rendering hookups), together with
`render.slang` (the generic sphere-tracing renderer). All of it runs on the
GPU through SlangPy — there is no NumPy mirror of the weights anywhere, no
batch sampled on the CPU, and no per-pixel renderer loop in Python. The only
NumPy left in this section is checkpoint I/O and handing a finished image to
matplotlib.

## 5.1 A distance field you can hold in your hand

Before training anything, it pays to look at one exact SDF. The smooth torus
has a closed form: collapse the point onto the plane of the ring, measure how
far it sits from the ring's center circle, and subtract the tube radius. In
`common_shapes.slang`, shared by every section that needs this shape:

```slang
public float torus_sdf(float3 p)
{
    float ring = length(p.xy) - TORUS_R;
    return length(float2(ring, p.z)) - TORUS_r;
}
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

The network (`neural_sdf.slang`'s `Network`) is Section 4's Fourier-encoded
coordinate MLP with one change that costs zero lines and changes everything
about the output's meaning: the final layer stays **linear**. An occupancy
network squashed its output through a sigmoid because it predicted a
probability; a distance can be any real number. This recipe runs the network
at hidden width 64 (76,289-parameter widths are overkill for exact
supervision — Part 5.5's harder recipe is where the extra capacity earns its
keep).

Training data follows the Section 4 pattern — half the batch uniform in the
bounding box, half hugging the surface — except each point now carries an
exact distance instead of a binary label. For the loss, DeepSDF's recipe is an
L1 distance between predictions and targets that have both been **clamped**
to a band `[-δ, +δ]` around the surface, blended with a lightly-weighted
*unclamped* L1 term that keeps gradients alive everywhere:

```slang
[Differentiable]
float clamped_l1_loss(no_diff float3 p, no_diff float gt, no_diff float delta,
                      no_diff float beta_far, Network network)
{
    float pred = network.eval(p);
    float cp = clamp(pred, -delta, delta);
    float cg = clamp(gt, -delta, delta);
    return abs(cp - cg) + beta_far * abs(pred - gt);
}
```

The intuition is the same as the original NumPy version of this recipe: we do
not care whether a point far from the shape reads 0.8 or 0.9 — we care
intensely whether a point near the surface reads +2 mm or −2 mm, because that
is where the level set (and therefore the geometry) lives. What changed is who
computes the "clamp kills gradient outside the band" mask: in the original
code this was a hand-written NumPy mask; here it is just `clamp`'s own
zero-derivative region, and Slang's autodiff (`bwd_diff`) gets it for free.
One GPU dispatch per step — `calculate_supervised_grads` — draws the batch,
labels it with `torus_sdf`, and runs the whole forward-and-backward pass:

```slang
void calculate_supervised_grads(int idx, uint seed, int n_uniform, float jitter,
                                float delta, float beta_far, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p;
    if (idx < n_uniform) p = rng.next_float3(-BOUND, BOUND);
    else { /* jittered sample on the exact torus surface */ }
    float gt = torus_sdf(p);
    bwd_diff(clamped_l1_loss)(p, gt, delta, beta_far, network, 1.0f);
}
```

**A bug worth its own paragraph, carried over from the original recipe.** A
network trained from a standard init lands mostly *outside* the ±0.1 clamp
band, where the clamp's gradient is zero — the network is born dead, with no
gradient able to reach it. The fix is the same two-fold one the original
NumPy code used: initialize the last layer small (`out_scale=0.1` on
`Network`'s final `NetworkParameters`, in the Python driver), so predictions
start near zero inside the band, and keep the unclamped far-field term
(`beta_far=0.2` by default) blended in so gradients never fully vanish.

With the fix, this 6,785-parameter network trains in well under a minute on a
modern GPU: 1,500 steps reach IoU **0.985** against the exact inside test,
with a mean absolute distance error of **0.009** over the whole bounding box
— matching the quality the original, much slower NumPy recipe reached.

## 5.3 What happened to analytic normals

The original recipe derived the surface normal — the gradient of the field
with respect to its *input coordinates*, not its weights — by hand-chaining
backprop through the positional encoding's own closed-form Jacobian. The
natural GPU-side equivalent would be a `bwd_diff` call with the input point
marked differentiable and the network's weights held `no_diff`. We tried
exactly that: it compiled, but returned `NaN` gradients on this toolchain.
Rather than debug Slang's autodiff machinery further, `render.slang` takes
the simpler, standard production route instead — **central finite
differences** of the field itself, six extra ordinary forward evaluations,
no differentiation machinery involved at all:

```slang
public float3 normal_fd<S : Sdf>(S sdf, float3 p, float h = 2e-3f)
{
    float3 g = float3(
        sdf.distance(p + float3(h, 0, 0)) - sdf.distance(p - float3(h, 0, 0)),
        sdf.distance(p + float3(0, h, 0)) - sdf.distance(p - float3(0, h, 0)),
        sdf.distance(p + float3(0, 0, h)) - sdf.distance(p - float3(0, 0, h)));
    float n = length(g);
    return n > 1e-12f ? g / n : float3(0, 0, 1);
}
```

This is visually indistinguishable from the analytic gradient for a smooth
field, which is exactly why production sphere tracers use it too. The
eikonal property from Part 5.1 is still the reason this works at all: a true
distance field's gradient has unit length everywhere, so a cheap
finite-difference estimate of it is well-conditioned near the surface.

## 5.4 Recipe: rendering without a mesh — sphere tracing

Here is the payoff that no occupancy field can offer. To find where a ray hits
the surface, march along it — and at every step, the SDF value *is* a
certificate that the nearest surface is at least that far away, so it is a
guaranteed-safe step size:

```slang
public bool sphere_trace<S : Sdf>(S sdf, float3 origin, float3 dir, out float t_hit,
                                  float t_max = 5.0f, float eps = 1e-3f,
                                  float step_scale = 0.9f)
{
    float t = 0.0f;
    for (int i = 0; i < SPHERE_TRACE_MAX_STEPS; ++i)
    {
        float3 p = origin + t * dir;
        float d = sdf.distance(p);
        if (d < eps) { t_hit = t; return true; }
        t += step_scale * max(d, 0.5f * eps);
        if (t > t_max) break;
    }
    t_hit = t;
    return false;
}
```

`step_scale = 0.9` is the one engineering nuance: a *learned* field is only
approximately eikonal, so it can slightly overestimate the safe distance;
scaling steps down by 10% restores the safety guarantee at the cost of a few
extra iterations. `render.slang`'s `Sdf` interface keeps this march, the
normal from Part 5.3, and Lambert shading generic over *what* is being
traced — the exact analytic torus (`AnalyticSdf`, the reference render) or a
trained network (`NeuralSdf`, defined in `neural_sdf.slang`), through the
identical `render_pixel` code. One thread per pixel does the whole
march-then-shade-then-composite pipeline; there is no mesh anywhere in the
resulting image, no marching cubes, no rasterizer.

## 5.5 Recipe: an SDF from points alone — the eikonal loss

Now the section's hardest and most modern trick. Suppose you have only a point
cloud — a laser scan, say — with no distances at all. Regression is off the
table. But recall the two fingerprints of an SDF: it is zero *at* the surface,
and its gradient has unit norm *everywhere*. Both are checkable without ground
truth, so both can be losses (Implicit Geometric Regularization, Gropp et al.
2020), plus a repulsion term that vetoes the trivial all-zero solution. Three
GPU kernels implement the three terms. The surface term pushes sampled cloud
points to zero:

```slang
[Differentiable]
float surface_loss(no_diff float3 p, Network network) { return abs(network.eval(p)); }
```

The eikonal + repulsion term runs over "space" points, and needs the field's
gradient — which this recipe gets the same way Part 5.3 settled on for
rendering, central finite differences, except now *through* `bwd_diff` rather
than as a one-off evaluation. That needs seven network evaluations per point
(one unshifted, six shifted by `±fd_h` along each axis), and
`calculate_eikonal_grads` dispatches **seven GPU threads per point** to get
them: every thread redundantly recomputes all seven forward passes (cheap, no
autodiff) to build the shared finite-difference gradient, then backprops
exactly one of the seven terms. The alternatives — one thread computing all
seven `bwd_diff` calls, or one `bwd_diff` over all seven evaluations at once —
give identical gradients but run 2.5–4× slower, or get rejected outright by
the GPU driver at hidden width 256; seven threads per point is simply the
fastest form that compiles at every width this recipe uses.

A 200-step geometric warm-up toward a sphere SDF (`‖x‖ − 0.75`) runs first,
giving the field one clean inside region to deform before the eikonal loss —
otherwise a random init can lock in spurious interior pockets. One detail
matters as much as the math: every one of these losses is a **batch mean**,
not a sum, because the surface, eikonal, and repulsion gradients all land in
the same Adam step, so their relative scale *is* their relative weight. A
summed surface term outweighs the other two roughly 1,000× and the field
collapses to zero everywhere.

This recipe trains at hidden width 256 (76,289 parameters — the same number
the original NumPy version used), because width 64 isn't enough capacity to
recover the knurled torus's hole; it reliably comes out as a hole-less blob at
that width. At 256, one run (seed=1, the default `run_section.py` uses) traced
IoU **0.41 → 0.49 → 0.59** after 900/1,800/2,700 steps — a real fix for the
hole, in 14–29 minutes depending on GPU load. That number is honestly short of
the 0.69 the original recipe's slower, smaller-scale experiments reached, and
quality here is genuinely **seed-sensitive** in a way the supervised recipe
is not: a different seed (0) reached something worse than width 64 ever
does — IoU 0.03, with the field's sign and magnitude badly miscalibrated even
though the zero level set still traced a recognizable torus. If a retrain
ever looks like a rough, "woven" blob rather than a clean ring, that is this
failure mode, and retrying with a different seed is the current workaround.

## 5.6 Shapes as arithmetic: CSG on distance fields

A final gift of the distance representation: solid modeling becomes
arithmetic. The union of two shapes is `min(d1, d2)`, intersection is `max`,
subtraction is `max(d1, -d2)` — and replacing the hard `min` with a polynomial
**smooth minimum** (Quílez) welds the shapes together with a fillet of radius
~k:

```slang
public float smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return lerp(d2, d1, h) - k * h * (1.0f - h);
}
```

Nothing in that function knows or cares that `d1` comes from a trained neural
network and `d2` from a four-line analytic sphere — `UnionSdf` in
`neural_sdf.slang` composes exactly that, sphere-traced through the same
generic `render_pixel` pipeline as every other image in this section. Figure
5.2 sphere-traces `smooth_union(neural_torus, sphere)` at k = 0.02, 0.10, 0.25:
the same two shapes, welded with a growing fillet, edited *after* training
without touching a single weight. Try doing that to a triangle mesh — boolean
operations on meshes are a notoriously fragile corner of geometry processing,
and here they are one line.

## 5.7 What it costs, and what comes next

| experiment | supervision | width | steps | result |
|---|---|---|---|---|
| regression SDF (smooth torus) | exact distances | 64 | 1,500 | IoU 0.985, mean abs. error 0.009, well under a minute |
| sphere-traced render | — | 64 | up to 96/ray | Fig. 5.1, no mesh |
| eikonal SDF (knurled torus), seed 1 | 30K surface points only | 256 | 2,700 | IoU 0.59, 14–29 min |

The pattern of the whole chapter repeats one level up: richer supervision
buys faster, more reliable convergence (distances ≫ points), and every query
is a network evaluation — rendering a single frame still means millions of
GPU threads each running a small MLP. Both pressures — training cost and
query cost — point at the same door: Section 6's hybrid representations,
which keep the continuous, differentiable, CSG-friendly interface of this
section while storing most of the shape in fast, explicit feature grids.

**Exercises.** (1) Retrain the regression SDF with δ = 0.02 and δ = 0.5 and
sphere-trace both: too tight a clamp corrupts the far field the tracer needs;
too loose wastes capacity off-surface. (2) Try writing the `bwd_diff`
input-gradient kernel described in Part 5.3 yourself and reproduce the `NaN` —
then see if you can find a formulation that survives this toolchain. (3) Rerun
the eikonal recipe with a few different seeds at hidden=256 and plot the IoU
trajectory for each; characterize how often you hit the seed-0-style failure.
(4) Train a second SDF on the sphere and morph: render
`smooth_union`-style interpolation between the two fields for several k, then
read the DeepSDF paper and explain how a latent code turns this trick into a
shape *space*.
