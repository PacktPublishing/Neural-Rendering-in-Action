# Section 6 — Appearance Fields and Hybrid Representations

Sections 4 and 5 answered geometric questions: *is this point inside?* and
*how far is the surface?* But a shape you can only measure is a gray shape.
This section makes two final moves that complete the chapter's arc. First we
show that **appearance is just another field over the same coordinates** — a
network that answers *what color is here?*, and then *what color is here, seen
from there?* — which places us one step from NeRF. Second, we confront the
cost that every experiment so far has quietly paid: each query is a full
network evaluation. The fix, **hybrid representations**, moves most of the
shape out of the MLP and into an explicit grid of learned features, and we
will measure exactly what that buys.

The companion code is `appearance.py` (the Python driver) and
`appearance_train.slang` (both appearance networks, their ground truth,
training, and rendering kernels), plus `race_gpu.py`/`hybrid_race_train.slang`
for Part 6.4's race. Everything here runs on the GPU through SlangPy — there
is no NumPy anywhere in this section's computation. Section 5's trained SDF
(`neural_sdf.py`) is imported frozen: geometry and appearance are two
independently trained fields over the same coordinates, exactly as the text
frames it, and that separation is itself the first lesson.

## 6.1 The appearance to memorize

As in Section 4, we choose ground truth we can evaluate exactly. For color, a
procedural texture in the torus's own surface coordinates — smooth checkers,
because hard edges are "infinite frequency" and would tangle this experiment
with Section 4's spectral-bias story:

```slang
float3 texture_albedo(float3 p)
{
    float2 uv = torus_uv(p);
    float s = 0.5f + 0.5f * sin(8.0f * uv.x) * sin(6.0f * uv.y);
    float3 terracotta = float3(0.85f, 0.45f, 0.25f);
    float3 blue = float3(0.20f, 0.45f, 0.75f);
    return terracotta * s + blue * (1.0f - s);
}
```

For the view-dependent experiment we need ground truth where the answer
legitimately depends on where you stand. Physically that is *specularity*: a
highlight is a mirror-like reflection of the light source, so it lives at
different surface points for different cameras. We bake a Blinn–Phong
highlight into the target — diffuse texture plus a white specular lobe:

```slang
float3 radiance_gt(float3 p, float3 view)      // view: surface -> eye, unit
{
    float3 n = torus_normal(p);
    float diff = clamp(dot(n, LIGHT), 0.0f, 1.0f);
    float3 h = normalize(LIGHT + view);        // Blinn half-vector
    float spec = pow(clamp(dot(n, h), 0.0f, 1.0f), 60.0f);
    float3 col = texture_albedo(p) * (0.20f + 0.80f * diff) + 0.9f * spec;
    return clamp(col, 0.0f, 1.0f);
}
```

Note what this function *is*: outgoing radiance as a function of position and
direction, shading baked in. That is precisely the quantity NeRF's color head
predicts, and memorizing it is this section's dress rehearsal for the next
chapter.

## 6.2 Recipe: an albedo field

The network (`AppearanceNetwork` in `appearance_train.slang`) is Section 4's
coordinate MLP with a three-channel output — `γ(x) → 128 → 128 → rgb` —
trained by MSE on surface samples. Sampling, labelling, the forward pass, and
the backward pass are one fused GPU dispatch, same pattern as every other
trainer in this chapter:

```slang
void calculate_albedo_grads(int idx, uint seed, int n, AppearanceNetwork network)
{
    Rng rng = Rng(seed, uint(idx));
    float u = rng.next_float(0.0f, 6.28318530718f);
    float v = rng.next_float(0.0f, 6.28318530718f);
    float3 p = torus_surface(float2(u, v));
    float3 gt = texture_albedo(p);
    bwd_diff(mse_loss)(p, float3(0, 0, 0), gt, network, 1.0f / float(n));
}
```

Nothing about the *geometry* changed. To render, we sphere-trace Section 5's
frozen SDF to find the hit points, then ask the appearance network for their
colors. Shading uses the *analytic* torus normal rather than the learned
field's own gradient (`torus_normal(p)`, not Part 5.3's `normal_fd`) — this
section is about appearance with known, frozen geometry, and shading with the
learned SDF's gradient instead makes every ripple in its only-approximately-
unit-gradient field show up as surface roughness, burying the texture the
figure is actually about:

```slang
float3 n = torus_normal(p);        // analytic normal, not the learned field's own gradient
float shade = 0.2f + 0.8f * clamp(dot(n, LIGHT), 0.0f, 1.0f);
return clamp(appearance.eval(p, float3(0, 0, 0)) * shade, 0.0f, 1.0f);
```

Figure 6.1 shows the result at hidden width 128, 4,000 steps: the neural
texture against the procedural ground truth at **26.9 dB PSNR** — the checker
pattern, its colors, and its alignment with the geometry all recovered from
point samples. (The network needs real training time to get there: evaluated
directly against exact surface points with no sphere tracing involved, PSNR
climbs 20.5 dB → 25.1 dB → 31.2 dB at 800/1,600/3,200 steps and is still
rising — at the lower step counts the field has essentially only learned the
texture's mean color, which renders as colored static rather than a visible
checker.) The architectural lesson is the modularity: geometry and appearance
are *two fields over the same coordinates*, trainable separately, swappable
independently.

## 6.3 Recipe: view dependence in three input columns

Here is the entire diff between an albedo field and a view-dependent radiance
field — three extra input columns, raw and unencoded:

```slang
#if HAS_VIEW
    enc[3 + 6 * APP_ENC_L + 0] = v.x;
    enc[3 + 6 * APP_ENC_L + 1] = v.y;
    enc[3 + 6 * APP_ENC_L + 2] = v.z;
#endif
```

The view direction is appended *raw*, without positional encoding — radiance
varies smoothly and at low frequency with direction (a Phong lobe, not a
checkerboard), so it needs no high-frequency lift. NeRF makes exactly the same
asymmetric choice (L = 10 for position, L = 4 for direction), and it is worth
pausing on why: positional encoding is a budget you spend where the signal has
detail. Training samples pairs of (surface point, random outward view
direction) labeled by `radiance_gt`:

```slang
void calculate_radiance_grads(int idx, uint seed, int n, AppearanceNetwork network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p = torus_surface(...);
    float3 nrm = torus_normal(p);
    float3 view = rng.next_unit_vector();
    if (dot(view, nrm) < 0.0f) view = -view;      // outward hemisphere
    float3 gt = radiance_gt(p, view);
    bwd_diff(mse_loss)(p, view, gt, network, 1.0f / float(n));
}
```

Figure 6.2 is the payoff, read as a 2×2 grid: two cameras (columns), ground
truth above, neural field below. The diffuse checker is clearly reproduced in
both columns (19.4 / 19.9 dB PSNR after 4,000 steps) — but **the specular
highlight itself does not train, and more steps do not fix it.** `radiance_gt`'s
highlight is `pow(dot(n, h), 60)`, a very sharp lobe covering a tiny fraction
of the (point, direction) domain that training samples uniformly; under plain
MSE the highlight contributes almost nothing to the average loss, so the
network is right, by its own objective, to ignore it almost entirely. Fixing
this needs a change of recipe — importance-sampling view directions near the
mirror direction, a lower exponent, or a loss that does not let a rare sharp
feature be averaged away — not a larger step budget. A pure albedo field
could not represent this image pair at all, even in principle: it must commit
to one color per point. Those three extra input columns are the entire
distance between a texture and a light field, even when this particular
training recipe under-delivers on the highlight itself.

## 6.4 Recipe: the hybrid — a feature grid with a tiny decoder

Every render in this chapter has cost real time because every query runs a
full MLP. The insight behind hybrid representations is that the MLP is doing
two jobs we can separate: *storing* the shape and *decoding* queries. Storage
can move back into an explicit structure — the voxel grid of Section 3,
resurrected, but holding **learned feature vectors** instead of occupancy
bits. `hybrid_race_train.slang`'s `FeatureGrid` is a dense grid of GPU-resident
parameters, read through a `[Differentiable]` trilinear accessor:

```slang
[Differentiable]
float[GRID_FEAT] lookup(no_diff float3 p, no_diff float bound)
{
    no_diff float3 g = (p + bound) / (2.0f * bound) * float(GRID_RES - 1);
    // ...8-corner trilinear blend of get_cell(...)...
}

[BackwardDerivativeOf(get_cell)]
void get_cell_bwd(int ix, int iy, int iz, int c, float grad)
{
    int idx = ((ix * GRID_RES + iy) * GRID_RES + iz) * GRID_FEAT + c;
    cells_grad.add(idx, grad);          // atomic scatter into the 8 touched corners
}
```

Section 3 proved trilinear interpolation is differentiable in the query point;
the new observation is that it is also — trivially — differentiable in the
*stored values*, because the output is linear in them. `bwd_diff` scatters
each query's gradient into the same 8 corners automatically, through an
`AtomicTensor` — no manual `np.add.at`-style bookkeeping needed, because many
queries landing on the same corner in one batch is exactly what the atomic
add is for. The decoder that follows is tiny, because the grid has already
done the spatial work:

```slang
struct HybridNetwork
{
    FeatureGrid grid;                                  // 131,072 params: storage
    NetworkParameters<GRID_FEAT + 3, HEAD_HIDDEN> head0;
    NetworkParameters<HEAD_HIDDEN, HEAD_HIDDEN> head1;
    NetworkParameters<HEAD_HIDDEN, 1> head2;            // ~1,345 params: decoding
}
```

Look at that parameter split: almost all of the parameters are in the grid,
where a query touches only 4 × 8 = 32 of them; the MLP that every query must
fully traverse shrinks to a couple of 32-wide layers. Positional encoding
disappears entirely for this network — the grid's spatial addressing replaces
it.

**The race.** `race_gpu.py` trains this hybrid and a pure coordinate MLP on
the same knurled-torus occupancy task, same GPU, same 40-second wall-clock
budget per model — not a fixed step count, so the two architectures race
fairly regardless of how many steps each completes:

| | pure MLP | feature grid + tiny MLP |
|---|---|---|
| parameters | 6,785 | 132,417 (99% in the grid) |
| training steps in 40 s | 7,560 (5.3 ms/step) | 16,160 (**2.5 ms/step**) |
| IoU after 40 s | 0.954 | **0.985** |

The hybrid takes about 2.1× more optimization steps per second and reaches
*higher* quality in the same wall-clock time. The production-scale descendants
of this idea attack the one weakness a dense grid still has — it pays for
empty space just as Section 3's voxels did: Instant-NGP replaces the dense
grid with a multi-resolution *hash table* (collisions are resolved by the
decoder, which learns to disambiguate), triplanes factor the 3D grid into
three 2D ones, and 3D Gaussians abandon the grid for a sparse set of
primitives. Different storage, same recipe: explicit, spatially addressed
features + a small neural decoder. As of this writing, essentially every
real-time neural rendering system is some version of this sentence.

## 6.5 Choosing a representation

The chapter closes where its Section 1 table began, now with evidence attached
to every row. As a working decision rule: if you need **compatibility with
existing pipelines** (game engines, DCC tools), extract a mesh — everything in
this chapter can produce one via marching cubes. If you need **editing and
composition**, SDFs turn solid modeling into arithmetic (Section 5.6). If you
need **gradient-based reconstruction from data** — the common case in this
book — you want a field, and the remaining choice is speed vs. memory: a pure
MLP when the model must be tiny or the shape space must generalize (DeepSDF's
latent codes live here), a hybrid when training or query speed matters, which
in practice is almost always. And if your data is images rather than points or
distances, you need one more ingredient — a differentiable way to compare a
field against a photograph — which is volume rendering, and it is the subject
of the next chapter. Every piece is now on the table: a radiance field
`c(x, v)` (Part 6.3), fast spatial features (Part 6.4), and positional
encoding (Section 4). NeRF is these three ideas plus an integral.

**Exercises.** (1) Fix Part 6.3's highlight: try importance-sampling view
directions near the mirror direction, or a loss that weights rare, sharp
features more heavily, and see how much PSNR (and visible highlight) it
recovers. (2) Sweep the hybrid's grid resolution over {8, 16, 32, 64} at fixed
training budget and plot IoU and memory; find the knee. (3) Starve the
hybrid: set `GRID_FEAT = 1` and `HEAD_HIDDEN = 8` and see how little decoding
capacity the grid actually needs. (4) Build an appearance *hybrid*: store rgb
features in a `FeatureGrid` and decode with a small head; race it against
Part 6.2's MLP. (5) Implement a two-level grid (coarse 8³ + fine 64³, features
concatenated) — you have just reinvented the "multi-resolution" half of
Instant-NGP; read the paper's Section 3 and identify what the hash adds.
