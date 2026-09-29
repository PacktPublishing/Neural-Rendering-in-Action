# Section 4 — Neural Occupancy Fields: An MLP That Memorizes a Shape

A neural occupancy field stores a shape as a function `f(x,y,z) -> P(inside)`,
trained and evaluated entirely through SlangPy: training-batch sampling,
ground truth, training, and inference all run as GPU kernels. The only
NumPy left is checkpoint I/O, a NumPy mirror of the positional encoding kept
solely to draw one figure, and things with no Slang equivalent
(`skimage.measure.marching_cubes`, `scipy.spatial.cKDTree`).

## Files and what depends on what

There is exactly ONE network in this section. `neural_occupancy.slang` defines
it and `neural_occupancy.py` drives it; nothing else here defines a network.

```
  run_section.py     all-in-one: train every model, render every figure
  run_figures.py     render Parts 3-5 from existing checkpoints
  train_one.py       train + checkpoint ONE model (resumable)
  train_bunny.py     optional: train + render the Stanford bunny instead
       |                        of the torus (see "Optional: Stanford
       |                        bunny mesh support" below)
       |
       |  all four import:
       +--> neural_occupancy.py .... THE network, Python side: build, train,
       |       |                     checkpoint, evaluate, mesh -- plus the
       |       |                     GPU-dispatched ground truth (sampling,
       |       |                     lattice/batch evaluation) every figure
       |       |                     and metric needs
       |       +--> neural_occupancy.slang ... THE network, GPU side:
       |       |         |                     encoding, 3 ReLU layers,
       |       |         |                     BCE loss, gradients, and the
       |       |         |                     fused sampler that feeds them
       |       |         +--> ../common_network.slang  NetworkParameters
       |       |         |                              + the Adam step
       |       |         +--> ../common_shapes.slang   the rippled torus
       |       |         |     (ported from THIS section's original
       |       |         |      NumPy formula -- now the one canonical
       |       |         |      implementation, shared with Sections 5/6)
       |       |         +--> ../common_rng.slang       GPU-side sampling,
       |       |                                        no CPU RNG anywhere
       |       +--> ../slangpy_common.py .... device creation, the Python
       |                                      NetworkParameters, chunked
       |                                      batched inference
       +--> geometry.py ............ marching cubes, surface sampling (Sec. 3)
       +--> metrics.py ............. volumetric IoU, Chamfer distance (Sec. 3)

  train_bunny.py also imports mesh_shape.py .... loads bunny.ply (trimesh),
                                                  builds + caches its
                                                  occupancy grid and surface
                                                  cloud, both consumed by
                                                  new calculate_grads_mesh /
                                                  sample_point_mesh /
                                                  mesh_inside_batch kernels
                                                  in neural_occupancy.slang
                                                  (which call the SAME
                                                  Network + bce_loss as the
                                                  torus path) via
                                                  grid_trilerp in
                                                  ../common_shapes.slang
```

The `../` entries live in the chapter root and are shared with Sections 5
and 6 (the `.slang` files are genuinely shared via Slang's `import`;
Python only searches its own script's directory). Everything else must sit
together in this folder, or a half-copied folder is the usual cause of
ModuleNotFoundError.

## Quickstart

    py train_one.py 0 1500
    py train_one.py 4 1500
    py train_one.py 10 1500
    py run_figures.py

Or train + render everything in one command: `py run_section.py`

Optional, needs `trimesh` + `embreex` (see Toolchain below):

    py train_bunny.py 6 3000

## What moved to the GPU, and why

Sections 5 and 6 were consolidated to run their ground truth, sampling,
training, and rendering entirely as Slang/SlangPy kernels. This section
predates that rule (it was already reorganized into today's file layout
earlier, before "solely Slang" was adopted) — this pass brings its
*computation* up to the same standard:

- **Shape math**: this section's own `_effective_r`/`detailed_torus_inside`/
  `detailed_torus_surface` NumPy formulas are gone. `common_shapes.slang`'s
  `effective_tube_radius`/`detailed_torus_inside`/`detailed_torus_surface`
  — ported from exactly this code while consolidating Section 5, so
  Sections 5/6 wouldn't have to reinvent it — are now the one canonical
  implementation, used here too.
- **Training-batch sampling**: `neural_occupancy.train()` no longer
  uploads a batch a Python `data_sampler` computed. `calculate_grads`
  samples the point itself (half uniform in the box, half near-surface +
  jitter, via `common_rng.slang`), labels it with `detailed_torus_inside`,
  and backprops it -- one GPU dispatch per step, no `pts`/`labels` tensor
  ever uploaded from Python.
- **Ground truth for figures and scoring**: Part 1's labelled-point
  scatter, the marching-cubes ground-truth grid (Parts 1 and 4), and the
  IoU ground truth (Part 4) are all GPU dispatches now
  (`sample_labelled_points`, `occupancy_grid_gpu`,
  `detailed_torus_inside_batch` in `neural_occupancy.py`) instead of NumPy
  lattice/point evaluation.

## Optional: Stanford bunny mesh support

An earlier version of this README said the bunny path was gone for good:
there's no closed-form point-in-mesh test to port to Slang the way
`detailed_torus_inside` is, and reimplementing a *live* GPU mesh test felt
disproportionate for an optional, off-by-default path. That's still
true -- but it turned out not to be necessary. The original NumPy code
never did a live test either: it paid the expensive point-in-mesh check
**once** (a cached occupancy grid), then answered every training-time
query by trilinear interpolation of that grid. Interpolating a
precomputed grid is trivial to port to Slang, and this section already
has the exact precedent for "precompute a big buffer once, upload it,
index into it from a training kernel" -- `neural_signed_distance`'s
surface-point `cloud` Tensor.

So the bunny path is back, as a parallel, opt-in path alongside the torus
one, driven by `train_bunny.py`:

- `mesh_shape.load_and_cache` loads `bunny.ply` via `trimesh`, closes its
  open base (`mesh.fill_holes()` -- the raw scan's underside is a hole),
  normalizes it into `[-BOUND, BOUND]^3`, and builds a `128^3` occupancy
  grid via `mesh.contains()` over the lattice -- cached to
  `checkpoints/bunny_grid_R128.npz` so this only runs once. A large
  surface-point cloud (`geometry.sample_mesh_surface`, already used
  elsewhere in this file) is resampled cheaply every run.
- **`mesh.contains()` needs `embreex` (or `pyembree`) installed.**
  Without it, trimesh falls back to a pure-Python ray/triangle
  intersector that tests every query point against every triangle at
  once and runs out of memory well before a `128^3` lattice finishes
  (measured: >5 GiB for a single 100k-point chunk against this mesh's
  69,451 faces). With `embreex`, the whole lattice labels in under a
  second.
- `neural_occupancy.slang` gets three new functions --
  `calculate_grads_mesh`, `sample_point_mesh`, `mesh_inside_batch` --
  structurally identical to the torus path's `calculate_grads`/
  `sample_point`/`detailed_torus_inside_batch`, but reading the cached
  grid (via `grid_trilerp`, `../common_shapes.slang`, a new generic
  trilinear-lookup function) and cloud instead of the analytic torus
  functions. They call the exact same `Network` and `bce_loss` the torus
  path does -- only sampling and labeling differ.
- **The grid, not `mesh.contains()`, is training's ground truth.** Once
  the grid is built, nothing in the training loop touches the mesh
  again -- `calculate_grads_mesh` labels every point by
  `grid_trilerp(...) > 0.5`. `train_bunny.py` scores IoU the same way
  (`trilinear_interpolate`, not `mesh.contains()` on scattered points),
  so IoU measures "did training match its own labels" -- the same
  relationship the torus path already has.
- A from-scratch alternative (`geometry.py`'s own ray-stabbing
  `voxelize_mesh`/`_column_crossings`, already in this file for the
  torus's Listing 3.2, avoiding the trimesh dependency entirely) was
  tried first and rejected: `bun_zipper.ply` isn't fully watertight even
  after `fill_holes()` (the raw scan has a few small residual gaps
  beyond the base), which broke that approach's even/odd ray-crossing
  parity and produced visibly wrong ground truth -- thin spurious
  comb-like fins through the mesh. Caught by looking at the rendered
  figure, not by IoU, which doesn't distinguish a genuinely bad ground
  truth from a bad reconstruction. `mesh.contains()` is not immune to
  non-watertightness in principle, but does not show this artifact on
  this mesh in practice.

Measured (L=6, hidden=64, 3000 steps, one seed): IoU 0.965, Chamfer
9.65e-3 against the cached grid -- comparable to the torus path's own L=4
number (0.94 / 10.8e-3) in this README's table below. A few isolated
floating specks appear in the marching-cubes extraction, the same
under-fit-high-frequency-noise artifact already documented for the SDF
section's Figure 6.2.

## What stays NumPy/SciPy (no sensible Slang equivalent)

- `geometry.extract_mesh` (`skimage.measure.marching_cubes`) and
  `metrics.chamfer_distance` (`scipy.spatial.cKDTree`), plus
  `geometry.sample_mesh_surface` (mesh -> point cloud for Chamfer) --
  unavoidable; these are the exception every other section's README
  points back to.
- `positional_encoding()` -- a NumPy mirror of the encoding the shader
  computes, kept exactly as before. It exists only to draw Part 2's figure
  and document the column order; the network never calls it.
- `train()`'s reporting-only loss/accuracy readback (drives the Part 3
  curves) -- kept exactly as before. This is, in fact, the *original*
  precedent Sections 5/6's own READMEs point back to when justifying their
  own reporting-only NumPy.
- Checkpoint I/O (`CoordinateMLP.save`/`.load`) -- unaffected by this
  refactor; the network weights and file format don't change.

Retraining with the new GPU sampler produces a different (not worse, just
different-seeded) training trajectory than any previously-shipped
checkpoint -- the same caveat Sections 5/6 note for their own fused
samplers.

## Five parts (torus reference numbers)

  representation            memory     IoU    Chamfer x1e-3
  ground truth                 -      1.000      0.0
  MLP, no encoding (L=0)    519 KiB   0.779     28.5
  MLP, L=4                  543 KiB   0.971     10.8
  MLP, L=10                 579 KiB   0.954     11.7

These are the book's numbers, for its 256-wide network (hence ~520 KiB).
At this code's default width of 64, 1500 steps reach IoU 0.79 (L=0), 0.94
(L=4) and 0.87 (L=10). Those were measured on an RTX 4090 against the exact
inside test at 200,000 uniform random points in the box, so treat them as a
rough comparison, not like-for-like with the table.

Without positional encoding the network smooths high-frequency detail away
(spectral bias); more bands recover it, though bands beyond the shape's own
frequency content add off-surface noise, so there is an optimal L. Part 5 meshes
one trained network at 32/64/128^3 from the same weights: resolution is a
property of the query, not the representation.

## Caveats

- Denormal floats: trained weights accumulate subnormal values that hit a slow
  CPU path; `CoordinateMLP.load` flushes them.
- The images in `figures/` were rendered before the gradient fixes below;
  rerun `run_section.py` to regenerate them.

## Differences from the book

**Hidden width defaults to 64, not the book's 256.** That cap used to be
forced: at 128 and 256 the training kernel never finished compiling (over
an hour), and on NVIDIA and Intel GPUs even 24 wide failed outright. An
earlier version of this README blamed the GPU driver's shader compiler.
The real cause was this code, and it's fixed (see "Two Slang autodiff
pitfalls" below). On an RTX 4090 the 256-wide kernel now compiles in about
10 s, and `python diag_hidden_width.py 256 4 1500` trains the book's width
end to end in about 75 s (final loss 0.026, batch accuracy 0.991). 64
stays the default because it trains about 12x faster (roughly 5 s for 1500
steps). Change `HIDDEN` in `neural_occupancy.py` for the book's network.

**Resolved: the "training collapse at hidden=64".** An earlier version of
this README reported that L=4 and L=10 at width 64 collapsed to a constant
predictor (P(inside) = 0 everywhere, IoU 0), blamed it on class imbalance
in the unweighted BCE loss, and proposed a weighted BCE. The real cause was
wrong gradients (pitfall 1 below): the 64 -> 1 output layer passed a wrong
gradient back, which corrupted every layer before it. With correct
gradients, L=4 and L=10 at width 64 train normally with the plain BCE and
the default `near_frac=0.5` (IoU 0.94 and 0.87, above). No loss weighting
is needed.

## Two Slang autodiff pitfalls (and how this code avoids them)

Sections 5 and 6 point here. All three sections share
`../common_network.slang` and the same ReLU pattern, so both fixes apply
to every network in this chapter.

1. **Wrong gradients through nested `[MaxIters]` loops.**
   `NetworkParameters.forward` used to be a plain `[Differentiable]`
   function with two nested `[MaxIters]` loops, leaving the backward pass
   to Slang. The gradient Slang generates with respect to the layer's
   *input* is wrong whenever the layer's input and output counts differ by
   more than 2. That was measured on every slangpy release from 0.38.1 to
   0.43.1 (Slang 2025.21 to 2026.12). The layer's own weight gradients are
   right, so in a network only the last layer trained correctly and every
   earlier layer got garbage, often with the wrong sign. Nothing errors;
   on the CUDA backend the same kernel faults with an illegal memory
   access. `[ForceUnroll]` on those loops gives correct gradients but
   compiles for minutes even at width 32. So `common_network.slang` writes
   the dense layer's backward by hand (`forward_bwd`: dW += dy x^T,
   db += dy, dx = W^T dy). It is checked against NumPy backprop and finite
   differences.

2. **In-place ReLU loops under `[MaxIters]`.** `h[i] = max(h[i], 0)` inside
   a `[MaxIters]` loop makes Slang's backward pass save the whole
   activation array on every iteration, which grows as width squared.
   Around 24 wide, the NVIDIA and Intel drivers reject the kernel. The
   symptom is `m_rhi_command_encoder->finish(...) failed with error: -13`
   on the first training dispatch (`VK_ERROR_UNKNOWN` from pipeline
   creation). Other GPUs just compile for a very long time. The ReLU loops
   are `[ForceUnroll]`, which keeps the saved state linear in width.

## Toolchain

```
pip install slangpy==0.42.0
```

Keep that pin. slangpy 0.43.x adds a second, separate gradient bug:
gradients routed through a `[BackwardDerivativeOf]` accessor come back
wrong. In testing on 0.43.1, every weight gradient of the old
accessor-based `NetworkParameters` was exactly zero; training still runs,
so this fails silently. `common_network.slang` no longer uses accessors,
but `hybrid_race_train.slang`'s `FeatureGrid.get_cell` still does, so
assume it's affected. Upgrading doesn't fix pitfall 1 above either: that
bug is in every release tested.

Needs a Vulkan-capable GPU, plus scikit-image and matplotlib for the meshing
and the plots. `train_bunny.py` additionally needs:

```
pip install trimesh embreex
```

(`embreex` only for `mesh.contains()`'s speed/memory -- see "Optional:
Stanford bunny mesh support" above. Neither is needed for the torus path.)

### Picking a GPU

On a machine with more than one Vulkan-capable GPU, every script in this
chapter goes through `../slangpy_common.py`'s `create_device`, which picks
Vulkan's own default adapter unless you steer it. List what's available
and select one with the `CHAPTER6_GPU` environment variable:

```
python -c "import slangpy_common as sc; sc.list_adapters()"

(bash)       CHAPTER6_GPU=nvidia python train_one.py 4 1500
(PowerShell) $env:CHAPTER6_GPU="nvidia"; python train_one.py 4 1500
```

`CHAPTER6_GPU` accepts a vendor shorthand (`amd`, `nvidia`, `intel`) or any
substring of the adapter's name (e.g. `3050`, `radeon`) -- useful with two
GPUs from the same vendor, or an unfamiliar vendor string. An unset or
empty value leaves Vulkan's default alone; a value that matches nothing
raises, listing what the machine actually has.

### Passing L and the hidden width to the shader

`neural_occupancy.py` compiles the shader by prepending `#define` lines to its
source and calling `Module.load_from_source`. Do not switch this to SlangPy's
`options={"defines": ...}`: that argument never reaches the Slang preprocessor
(`Module.load_from_file` passes the path to `device.load_module` and keeps
`options` for call configuration), so the shader silently keeps its own defaults
while Python sizes the input layer for the L you asked for. Nothing errors; the
network just trains wired to the wrong inputs.
