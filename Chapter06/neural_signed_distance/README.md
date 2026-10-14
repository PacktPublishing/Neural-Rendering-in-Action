# Section 5 — Neural Signed Distance Functions: Smooth Geometry with Implicit Surfaces

A neural SDF stores a shape as a function `f(x,y,z) -> signed distance`,
trained and evaluated entirely through SlangPy: training, sampling, ground
truth, sphere tracing, and shading all run as GPU kernels. There is no
NumPy anywhere in this section's computation -- the only NumPy left is I/O
(checkpoint `.npz` files) and handing a finished array to matplotlib.

## Files and what depends on what

There is exactly ONE network in this section. `neural_sdf.slang` defines it
and `neural_sdf.py` drives it; nothing else here defines a network.

```
  run_section.py     all-in-one: train both recipes, render every figure
  run_figures.py     render Figures 5.1-5.3 from existing checkpoints
  train_one.py       train + checkpoint ONE recipe (resumable)
       |
       |  all three import:
       +--> neural_sdf.py .......... THE network, Python side: build,
       |       |                     train, checkpoint, evaluate, render
       |       +--> neural_sdf.slang ... THE network, GPU side: encoding,
       |       |         |               2 ReLU layers, both training
       |       |         |               recipes' losses and samplers
       |       |         +--> ../common_network.slang  NetworkParameters
       |       |         |                              + the Adam step
       |       |         +--> ../common_shapes.slang   the smooth and
       |       |         |     (ported from neural_occupancy_fields/       knurled tori
       |       |         |      shapes.py -- one canonical implementation,
       |       |         |      not three near-duplicates across sections)
       |       |         +--> ../common_rng.slang       GPU-side sampling,
       |       |                                        no CPU RNG anywhere
       |       +--> render.slang ......... the sphere-tracing renderer:
       |       |         |                 march, shade, composite -- one
       |       |         |                 thread per pixel, generic over
       |       |         |                 any Sdf (analytic or neural)
       |       |         +--> ../common_shapes.slang
       |       +--> ../slangpy_common.py .... device creation, the Python
       |                                      NetworkParameters, chunked
       |                                      batched inference
```

The `../` entries live in the chapter root and are shared with Sections 4
and 6. Everything else must sit together in this folder: Python searches
only the script's own directory (Slang's `import` does reach the chapter
root, so `.slang` files are genuinely shared, not copied).

## Quickstart

    py train_one.py supervised 1500
    py train_one.py eikonal 2700
    py run_figures.py

Or train + render everything in one command:

    py run_section.py

## What each recipe trains

- **`train_one.py supervised`** (5.2) — exact smooth-torus distances,
  clamped-L1 loss. Sampling (half uniform in the box, half near-surface +
  jitter), the exact `torus_sdf` label, and the clamped-L1 + far-field loss
  are all one differentiable Slang expression (`neural_sdf.slang`'s
  `clamped_l1_loss`): `clamp`'s zero-derivative outside its band is exactly
  the "clamp kills gradient" mask the book computes by hand in NumPy, for
  free from autodiff. One GPU
  dispatch per step (`calculate_supervised_grads`) does sampling, labelling,
  the forward pass, and the backward pass together.

- **`train_one.py eikonal`** (5.5) — an SDF from a 30,000-point surface
  cloud alone (no distances anywhere), via the IGR-style recipe: a surface
  term (sticks to zero), an eikonal term (be a unit-gradient distance
  field), and a repulsion term (don't collapse to zero everywhere), plus a
  200-step geometric warm-up toward a sphere SDF. Two dispatches per step:
  `calculate_surface_grads` (one thread per surface sample) and
  `calculate_eikonal_grads` (**seven** threads per "space" point -- see
  the note below for why).

## Why the eikonal kernel runs seven threads per point

The eikonal loss needs seven network evaluations per point (one unshifted,
six shifted by `+-fd_h` along each axis) to build its finite-difference
gradient. `calculate_eikonal_grads` dispatches **seven GPU threads per
point**: every thread redundantly recomputes all seven forward evaluations
(cheap, no autodiff) to get the shared finite-difference coefficient, then
calls `bwd_diff` once, on its own one of the seven terms.

This used to be forced. An earlier version of this README said a kernel
could call `bwd_diff` at most once, because more took over 9 minutes to
compile. That was measured before the fixes in
`neural_occupancy_fields/README.md` ("Two Slang autodiff pitfalls") and no
longer holds. Re-measured on an RTX 4090 against this kernel, on identical
points and weights:

  one thread per point, instead:      hidden 64               hidden 256
  seven bwd_diff calls in one kernel  4 s compile, 2.7x slower   43 s compile, 3.9x slower
  one bwd_diff over all seven evals   7 s compile, 2.5x slower   driver rejects it (error -13)

Both alternatives produce the same gradients (to about 1e-6) wherever they
run. Seven threads per point stays because it's the fastest (7x the
threads keeps the GPU busier), it compiles in a few seconds, and it's the
only form that works at every width.

## Differences from the book

Same toolchain workarounds as Sections 4 and 6 (see
`neural_occupancy_fields/README.md` for the full detail on each): `L` and
`HIDDEN` as preprocessor `#define`s rather than Slang generics, and every
GPU buffer pre-allocated and a kernel warm-up dispatch issued before
training starts.

**Both L and HIDDEN used to be silently ignored.** An earlier version of
`_module()` passed them via SlangPy's `options={"defines": ...}` on
`Module.load_from_file`, which never reaches the Slang preprocessor (same
pitfall Section 4 documents) -- the shader always fell back to its own
`SDF_L 6` / `HIDDEN 64`, no matter what Python asked for. Harmless for `L`
only because every caller always asked for 6 anyway; it's why `HIDDEN` in
particular looked like a dead knob until now. Fixed by compiling with
`Module.load_from_source` and both `#define`s prepended to the source
text directly, cached by `(L, hidden)` instead of `L` alone -- the same
mechanism Section 4 already used.

**5.2's supervised recipe stays at hidden=64** (IoU 0.985, matching the
book) -- plenty of capacity for exact-distance supervision, and 64 trains
in seconds. **5.5's eikonal recipe now defaults to the book's hidden=256**
(`EIKONAL_HIDDEN` in `run_section.py` / `train_one.py`), because 64 isn't
enough capacity for this harder, points-only recipe to recover the torus's
hole: it reliably comes out as a hole-less blob at that width (IoU 0.32 ->
0.38 -> 0.43 after 900/1,800/2,700 steps, one seed's trajectory). 256 now
compiles in seconds too (see "Two Slang autodiff pitfalls" in
`neural_occupancy_fields/README.md`), so the only real cost left is
training time: on this project's dev GPUs, 2,700 steps took 14-29 minutes
at hidden=256 (the seven-thread-per-point eikonal kernel is heavy, and the
two measurements were taken under very different concurrent GPU load --
treat this as "tens of minutes," not a fixed number), against well under a
minute at 64.

**Quality at hidden=256 is seed-sensitive in a way 64 is not.** One run
(seed=1, matching what `run_section.py` actually uses) reached IoU 0.41 ->
0.49 -> 0.59 after 900/1,800/2,700 steps -- a real fix for the hole, closer
to the book's own 0.38 -> 0.60 -> 0.69 (also measured at width 256, on an
RTX 4090). A different seed (0) reliably reached something worse than
hidden=64 ever does: IoU 0.03, with 90.7% of the evaluation volume reading
"inside" against a true ~9% -- the zero level set still traced a
recognizable torus with the correct hole, but the field's sign/magnitude
was badly miscalibrated (values crushed into roughly [-0.18, 0.05] instead
of spanning the domain like a real distance field). This looks like a
real, open instability specific to wide eikonal networks -- not something
this width change fixes on its own, just something it makes visible. If a
retrain ever looks like a rough, "woven"-textured blob rather than a clean
ring, that's this failure mode; retrying with a different seed is the only
current workaround.

**5.5's losses are batch means.** The surface, eikonal and repulsion
gradients are added into one Adam step, so their relative scale is their
relative weight. An earlier version of this port summed the surface term
and the warm-up over their batches instead of averaging them as the
original NumPy recipe does. That made the surface term ~1,000x too strong:
with correct gradients the field collapses to f = 0 everywhere (IoU below
0.1). Wrong gradients happened to hide this when the shipped checkpoint was
trained. `calculate_surface_grads` and `calculate_warmup_grads` now divide
by their batch size. The warm-up matters too, because it shares Adam's
moment buffers with the training that follows.

**Rendering normals are finite differences, not the analytic input
gradient.** Section 5.3's `input_gradient` derives the analytic gradient by
hand-chaining backprop through the encoding's Jacobian -- elegant in NumPy,
but not needed for a renderer: `render.slang`'s `normal_fd` takes six extra
forward evaluations of the SDF around the hit point instead. This is the
standard choice in production sphere tracers (visually indistinguishable
for a smooth field) and needs no `[Differentiable]` machinery at all, so it
carries none of the compile-time risk above. An attempt at a true
autodiff-with-respect-to-the-input kernel (`bwd_diff` with the point
differentiable and the network held `no_diff`) compiled but returned `NaN`
gradients on this toolchain; finite differences were used instead rather
than debugging that further.

**Expected quality.** Scoring the sign of f against the exact inside test
at 200,000 random points in the box:

  recipe                           result
  5.2 supervised, 1500 steps       IoU 0.985, mean abs. error 0.009
  5.5 eikonal, hidden=64, one      IoU 0.32 / 0.38 / 0.43 after
  2,700-step run                     900 / 1,800 / 2,700 steps
  5.5 eikonal, hidden=256, seed=1  IoU 0.41 / 0.49 / 0.59 after
  (as run_section.py trains it)      900 / 1,800 / 2,700 steps

Stray zero-level sheets away from the surface are part of this recipe, as
the book notes. The checkpoints in `checkpoints/` were retrained after the
gradient, batch-mean and HIDDEN-wiring fixes above -- current as shipped.

## Toolchain

```
pip install slangpy==0.42.0
```

Pin that version -- see `neural_occupancy_fields/README.md`'s Toolchain
note on the 0.43.x gradient bug.

Needs a Vulkan-capable GPU, plus scikit-image and matplotlib for the
marching-cubes comparison panel and the plots. On a machine with more than
one, see `neural_occupancy_fields/README.md`'s "Picking a GPU" for the
`CHAPTER6_GPU` environment variable -- shared by every script in this
chapter via `../slangpy_common.py`.
