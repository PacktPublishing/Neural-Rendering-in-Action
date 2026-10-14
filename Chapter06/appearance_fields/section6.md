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

All fragments come from the companion `app6.py`, which reuses Section 5's
sphere tracer and its trained torus SDF (`sdf5.py` + checkpoint): the geometry
network is frozen throughout this section, and everything new happens in
separate appearance networks. That separation is itself the first lesson.

## 6.1 The appearance to memorize

As in Section 4, we choose ground truth we can evaluate exactly. For color, a
procedural texture in the torus's own surface coordinates — smooth checkers,
because hard edges are "infinite frequency" and would tangle this experiment
with Section 4's spectral-bias story:

```python
def texture_albedo(p):
    u, v = torus_uv(p)                       # angles around hole and tube
    s = 0.5 + 0.5 * np.sin(8 * u) * np.sin(6 * v)
    return c_terracotta * s[:, None] + c_blue * (1 - s[:, None])
```

For the view-dependent experiment we need ground truth where the answer
legitimately depends on where you stand. Physically that is *specularity*: a
highlight is a mirror-like reflection of the light source, so it lives at
different surface points for different cameras. We bake a Blinn–Phong
highlight into the target — diffuse texture plus a white specular lobe:

```python
def radiance_gt(p, view):                    # view: surface -> eye, unit
    n = torus_normal(p)
    diff = clip(n @ LIGHT, 0, 1)
    h = normalize(LIGHT + view)              # Blinn half-vector
    spec = clip((n * h).sum(1), 0, 1) ** 60
    return texture_albedo(p) * (0.2 + 0.8*diff) + 0.9 * spec
```

Note what this function *is*: outgoing radiance as a function of position and
direction, shading baked in. That is precisely the quantity NeRF's color head
predicts, and memorizing it is this section's dress rehearsal for the next
chapter.

## 6.2 Recipe: an albedo field

The network is Section 4's coordinate MLP with a three-channel output —
`γ(x) → 128 → 128 → rgb` — trained by MSE on surface samples. The training
loop should look almost boringly familiar by now, and that is the point:

```python
p    = torus_surface(batch, rng)
pred = net.forward(encode(p, L))
gt   = texture_albedo(p)
opt.step(net.backward(2 * (pred - gt) / pred.size))
```

Nothing about the *geometry* changed. To render, we sphere-trace Section 5's
frozen SDF to find the hit points, then ask the new network for their colors
and apply Lambert shading from the analytic normals:

```python
p_hit = origins + t_hit * dirs               # from Section 5's tracer
color = albedo_net(encode(p_hit)) * (0.2 + 0.8 * lambert)
```

Figure 6.1 shows the result after 3,000 steps (about 45 s in NumPy): the
neural texture against the procedural ground truth at **23.7 dB PSNR** — the
checker pattern, its colors, and its alignment with the geometry all recovered
from point samples. The architectural lesson is the modularity: geometry and
appearance are *two fields over the same coordinates*, trainable separately,
swappable independently. Re-texturing the torus means retraining a 128-wide
color net for 45 seconds while the SDF sits untouched.

## 6.3 Recipe: view dependence in three input columns

Here is the entire diff between an albedo field and a view-dependent radiance
field:

```python
net = MLP([3 + 6*L + 3, 128, 128, 3])        # +3 inputs: the view direction
...
x = np.concatenate([encode(p, L), v], axis=1)
```

The view direction is appended *raw*, without positional encoding — radiance
varies smoothly and at low frequency with direction (a Phong lobe, not a
checkerboard), so it needs no high-frequency lift. NeRF makes exactly the same
asymmetric choice (L = 10 for position, L = 4 for direction), and it is worth
pausing on why: positional encoding is a budget you spend where the signal has
detail.

Training samples pairs of (surface point, random outward direction) labeled by
`radiance_gt`. Rendering changes by one line — the query direction is the ray
direction, flipped to point at the eye:

```python
v = -dirs[hit]                               # surface -> eye
color = radiance_net(concat(encode(p_hit), v))
```

Figure 6.2 is the payoff, and it must be read as a 2×2 grid: two cameras
(columns), ground truth above, neural field below. The white highlight sits on
a *different part of the torus in each column* — and the learned field moves
it correctly (~18.4 dB after 4,000 steps), because the highlight is not
painted on the surface; it is a property of the (position, direction) pair.
A pure albedo field cannot represent this image pair at all: it must commit to
one color per point. Those three extra input columns are the entire distance
between a texture and a light field.

## 6.4 Recipe: the hybrid — a feature grid with a tiny decoder

Every render in this chapter has cost seconds because every query runs a full
MLP: two 256-wide matrix multiplies, ~76K parameters exercised per point,
millions of points per image. The insight behind hybrid representations is
that the MLP is doing two jobs we can separate: *storing* the shape and
*decoding* queries. Storage can move back into an explicit structure — the
voxel grid of Section 3, resurrected, but holding **learned feature vectors**
instead of occupancy bits:

```python
class FeatureGrid:
    def __init__(self, res=32, feat=4):
        self.F = 0.1 * randn(res, res, res, feat)     # a PARAMETER tensor
    def lookup(self, pts):                            # trilinear, as in Sec. 3
        ...8-corner gather, blend by trilinear weights...
```

Section 3 proved trilinear interpolation is differentiable in the query point;
the new observation is that it is also — trivially — differentiable in the
*stored values*, because the output is linear in them. The backward pass is
therefore the forward pass mirrored: each query scatters its gradient into the
same 8 corners, weighted by the same trilinear weights:

```python
def backward(self, dfeat):
    for each corner (dx, dy, dz):
        np.add.at(gF, (i0+dx, j0+dy, k0+dz), w_corner * dfeat)
```

(We verified this scatter against numeric differentiation: agreement to
3 × 10⁻⁶. `np.add.at` rather than `+=` matters — many queries can hit the same
corner in one batch, and plain fancy-indexing silently drops the collisions.)

The decoder that follows can now be tiny, because the grid has already done
the spatial work — the network only has to translate a local feature vector
into a value:

```python
class HybridField:                    # grid -> tiny MLP -> occupancy logit
    self.grid = FeatureGrid(res=32, feat=4)         # 131,072 params: storage
    self.head = MLP([4 + 3, 32, 32, 1])             #   1,345 params: decoding
```

Look at that parameter split: 99% of the parameters are in the grid, where a
query touches only 8 × 4 = 32 of them; the MLP that every query must fully
traverse shrank from 76K parameters to 1.3K. Positional encoding disappears
entirely — the grid's spatial addressing replaces it.

**The race.** We trained this hybrid and Section 4's pure MLP on the same
knurled-torus occupancy task, same batches, same machine, same 40-second
wall-clock budget (Figure 6.3):

| | pure MLP | feature grid + tiny MLP |
|---|---|---|
| parameters | 76,289 | 132,417 (99% in the grid) |
| training steps in 40 s | 1,160 (34.5 ms/step) | 4,940 (**8.1 ms/step**) |
| IoU after 40 s | 0.947 | **0.975** |
| inference throughput | 0.22 M queries/s | **1.80 M queries/s** |

The hybrid takes 4.3× more optimization steps per second, reaches *higher*
quality in the same time, and answers queries 8× faster afterward — which
converts Section 5's 5-second sphere-traced render into a sub-second one. The
one column where the pure MLP wins is memory: 298 KiB vs 517 KiB, because a
dense grid pays for empty space just as Section 3's voxels did. That residual
weakness is precisely what the production-scale hybrids attack: Instant-NGP
replaces the dense grid with a multi-resolution *hash table* (collisions are
resolved by the decoder, which learns to disambiguate), triplanes factor the
3D grid into three 2D ones, and 3D Gaussians abandon the grid for a sparse
set of primitives. Different storage, same recipe: explicit, spatially
addressed features + a small neural decoder. As of this writing, essentially
every real-time neural rendering system is some version of this sentence.

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

**Exercises.** (1) Retrain the radiance field with the view direction
positionally encoded at L = 6 and compare convergence — you will find it
slightly *worse*; explain why using the frequency argument of Part 6.3.
(2) Sweep the hybrid's grid resolution over {8, 16, 32, 64} at fixed budget
and plot IoU and memory; find the knee. (3) Starve the hybrid: set `feat = 1`
and `hidden = 8` and see how little decoding capacity the grid actually needs.
(4) Build an appearance *hybrid*: store rgb features in a `FeatureGrid` and
decode with a 2-layer head; race it against Part 6.2's MLP. (5) Implement a
two-level grid (coarse 8³ + fine 64³, features concatenated) — you have just
reinvented the "multi-resolution" half of Instant-NGP; read the paper's
Section 3 and identify what the hash adds.
