# Appearance Fields and Hybrid Representations

Appearance is a second field over the same coordinates as the SDF section's
geometry -- trained, sampled, and rendered entirely through SlangPy. There
is no NumPy anywhere in this section's computation: ground truth
(`texture_albedo`, `radiance_gt`), sampling, training, sphere tracing,
shading, and the occupancy race are all GPU kernels. The only NumPy left is
checkpoint I/O and handing a finished array to matplotlib.

## Files and what depends on what

Two independently trained networks live here (an albedo field and a
radiance field), plus the occupancy race. Neither touches the SDF section's
geometry weights -- it's loaded once, frozen, and only ever sphere-traced.

```
  run_section.py     all-in-one: train both fields, run the race, render every figure
  run_figures.py     render Figures 6.1-6.3 from existing checkpoints/results
  train_one.py       train + checkpoint ONE field, albedo or radiance (resumable)
  race_gpu.py         Figure 6.7: pure MLP vs feature-grid hybrid, GPU-vs-GPU
       |
       |  run_section.py / run_figures.py / train_one.py import:
       +--> neural_sdf.py .......... The SDF section's FROZEN geometry field,
       |       |                     copied here (each folder is
       |       |                     self-contained -- Python only
       |       |                     searches its own script's directory)
       |       +--> neural_sdf.slang, render.slang  (also copied)
       +--> appearance.py .......... THE appearance networks, Python side:
       |       |                     build, train, checkpoint, evaluate, render
       |       +--> appearance_train.slang ... THE appearance networks, GPU
       |                 |                     side: encoding, ground truth,
       |                 |                     sampling, losses, and the
       |                 |                     rendering kernels that compose
       |                 |                     an appearance field with the
       |                 |                     frozen geometry field
       |                 +--> ../common_network.slang  NetworkParameters
       |                 +--> ../common_shapes.slang   torus_uv/normal/surface
       |                 +--> ../common_rng.slang       GPU-side sampling
       |                 +--> neural_sdf.slang  (imported for rendering)
       +--> race_gpu.py
               +--> hybrid_race_train.slang ... the race's two architectures:
                         |                      a pure coordinate MLP and a
                         |                      FeatureGrid + tiny-MLP hybrid
                         +--> ../common_network.slang, common_shapes.slang,
                                                  common_rng.slang
```

The `../` entries live in the chapter root and are shared with the occupancy
and SDF sections (`.slang` files are genuinely shared via Slang's `import`; the
Python `neural_sdf.py`/`.slang`/`render.slang` copies are Python-only, so
they need copying).

## Quickstart

Train the SDF section's geometry first if you haven't (the appearance section renders through
its frozen SDF):

    cd ../neural_signed_distance && py train_one.py supervised 1500
    cp checkpoints/ckpt_sdf_supervised.npz ../appearance_fields/checkpoints/
    cd ../appearance_fields
    py train_one.py albedo 800
    py train_one.py radiance 1200
    py race_gpu.py 40
    py run_figures.py

Or train + race + render everything in one command:

    py run_section.py

## What each part trains

- **`train_one.py albedo`** — `c(x) -> rgb`, MSE against
  `texture_albedo` (a procedural checker in the torus's own surface
  coordinates). One fused GPU dispatch per step samples a surface point,
  labels it, and backprops the MSE (`calculate_albedo_grads`).
- **`train_one.py radiance`** — `c(x, v) -> rgb`, MSE against
  `radiance_gt` (Lambert diffuse + a Blinn-Phong highlight, so the target
  genuinely depends on view direction). The only architectural diff from
  albedo is three extra input columns for the (un-encoded) view direction;
  `calculate_radiance_grads` also samples a random outward view direction
  per point.
- **`race_gpu.py`** — a pure coordinate MLP vs a FeatureGrid + tiny-MLP
  hybrid, both trained on the knurled-torus occupancy task for the same
  wall-clock budget. This chapter's earlier NumPy-vs-NumPy CPU race
  (Figure 6.7's original form) is gone -- with the book text being rewritten
  there's no reason to keep two races side by side, and the GPU-vs-GPU
  comparison already told the same story (hybrid trains faster per step
  *and* reaches higher accuracy in the same wall-clock time) without any
  hand-rolled NumPy backprop.

Rendering (`render_albedo`/`render_radiance` in `appearance.py`) sphere-traces
the *frozen* geometry field, then colours each hit with the trained
appearance field -- one kernel dispatch per image, no geometry weight ever
touched. `render_albedo_exact`/`render_radiance_exact` render the same hit
points straight from `texture_albedo`/`radiance_gt` (no network at all),
for the ground-truth comparison panels and their PSNR numbers.

## Differences from the book

Same toolchain workarounds as the occupancy and SDF sections (see
`neural_occupancy_fields/README.md` and `neural_signed_distance/README.md`
for the full detail): `L`/`HAS_VIEW`/`SDF_L` as preprocessor `#define`s
rather than Slang generics, and every GPU buffer pre-allocated before
training starts.

**Hidden width is 128, matching the book for this section.** It was 64 for
a while, forced by a compile-time blow-up that has since been fixed (see
"Two Slang autodiff pitfalls" in `neural_occupancy_fields/README.md`). 128
compiles and trains fine here now, and is worth roughly +1.3 dB on the
albedo render over 64 at the same step count.

**Step counts are 4000, not the book's 800/1200.** At the book's numbers
both fields are still far from converged -- see the note on "coloured
static" below, which is what that under-training actually looks like.

**Current checkpoints and figures** were regenerated together by one
`run_section.py` run at hidden=128 / 4000 steps:

  part                          this run          previously (64 wide, 800/1200)
  albedo, render PSNR           26.9 dB           22.0 dB
  radiance, cameras 1 / 2       19.4 / 19.9 dB    19.5 / 19.4 dB
  hybrid race, pure / hybrid    7,460 / 15,560 steps in 40 s each
                                (5.36 / 2.57 ms per step)

The race numbers are hardware-specific and not comparable to the RTX 4090
figures an earlier version of this note quoted; the hybrid still wins on
both speed and step count.

**The view-dependence highlight does not train, and more steps do not help.**
Note the radiance row above: 4000 steps scores the same as 1200 did. The
field learns the diffuse/albedo part well (the checker is clearly
reproduced in Figure 6.6's bottom row) but the specular lobe is missing
almost entirely, so the figure does not currently demonstrate the thing its
caption claims. `radiance_gt`'s highlight is `pow(dot(n, h), 60)` -- a very
sharp lobe covering a tiny fraction of the (surface point, view direction)
domain that `calculate_radiance_grads` samples uniformly. Under a plain MSE
the highlight contributes almost nothing to the average loss, so the
network is right, by its own objective, to ignore it. Fixing it needs a
change of recipe -- importance-sampling view directions near the mirror
direction, a lower exponent, or a loss that does not let a rare sharp
feature be averaged away -- not a larger step count.

**The "coloured static" look was under-training, and more steps DO fix
it.** Two earlier versions of this note got this wrong in opposite
directions; this one is measured. Evaluating the trained albedo field
directly on EXACT torus surface points -- no sphere tracing, no renderer
involved -- shows the loss is still falling steeply at the book's step
count:

  steps    PSNR vs texture_albedo, on exact surface points
    800    20.5 dB   (no visible checker at all; pure colour noise)
   1600    25.1 dB
   3200    31.2 dB   (still improving)

At 800 steps the field has essentially only learned the texture's MEAN
colour, which is why the render looked like static. The fix is the step
count, which is why `run_section.py` now trains for 4000.

**Figure 6.5 shades with the analytic torus normal, not the learned
field's gradient.** `render_albedo`/`render_albedo_exact` take `n` from
`common_shapes.torus_normal(p)`; the neural SDF is used only to find WHERE
each ray hits. Shading with `normal_fd(sdf, p)` instead -- central
differences of the learned field, which is what this port did originally
-- makes every ripple in the (only approximately unit-gradient) SDF show
up as surface roughness, and the torus renders visibly knurled even in the
ground-truth panel where the colour is exact. `render.slang`'s remark that
finite differences are "visually indistinguishable from the analytic
gradient" holds for a smooth field; this learned one is not smooth enough.
The tradeoff is that the
Lambert term now uses analytic knowledge of the shape rather than being
purely neural -- reasonable for a section about APPEARANCE with frozen,
known geometry, but worth stating in the text.

**Two struct types share field names (`layer0`/`layer1`/`layer2`) in the
same kernel call.** `render_albedo`/`render_radiance` take both a `Network`
(geometry) and an `AppearanceNetwork` (appearance) parameter; both structs
happen to name their layers the same way. This compiled and ran correctly
in testing, but it's the one place in this chapter where two
differently-typed network parameters with identical field names are passed
to one dispatch -- worth knowing about if a future SlangPy version's
parameter binding ever gets this wrong.

## Toolchain

```
pip install slangpy==0.42.0
```

Pin that version -- see `neural_occupancy_fields/README.md`'s Toolchain
note on the 0.43.x gradient bug, which likely also affects
`hybrid_race_train.slang`'s `FeatureGrid` accessor.

Needs a Vulkan-capable GPU, plus matplotlib for the plots (no scikit-image
dependency in this section -- there's no marching cubes here). On a machine
with more than one GPU, see `neural_occupancy_fields/README.md`'s "Picking
a GPU" for the `CHAPTER6_GPU` environment variable -- shared by every
script in this chapter via `../slangpy_common.py`.
