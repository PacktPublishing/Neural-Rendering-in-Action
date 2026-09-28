# Section 4 — Neural Occupancy Fields: An MLP That Memorizes a Shape

Pure NumPy + SciPy + scikit-image + matplotlib (no PyTorch). A neural occupancy
field stores a shape as a function `f(x,y,z) -> P(inside)` learned by a small MLP.

## Files (all six .py must sit in the SAME folder)

- `shapes4.py`   — the shape and training data (rippled torus, or a mesh/bunny)
- `field4.py`    — Fourier encoding + coordinate MLP (self-contained)
- `geometry.py`  — marching cubes, surface sampling, trilinear interp (from Sec. 3)
- `metrics.py`   — volumetric IoU and Chamfer distance (from Sec. 3)
- `train_one.py`    — train + checkpoint one model (resumable)
- `run_figures.py`  — render Parts 3-5 from checkpoints
- `run_section4.py` — all-in-one: train + render every figure

Python only searches the script's own directory, so a half-copied folder is the
usual cause of ModuleNotFoundError.

## Quickstart A — the rippled torus (chapter default)

    py train_one.py 0 1500
    py train_one.py 4 1500
    py train_one.py 10 1500
    py run_figures.py

## Quickstart B — the Stanford bunny

The bunny needs THREE things; skipping the middle one is why figures may still
show the torus:

1. PROVIDE THE MESH. `pip install trimesh`, put `bunny.ply` in this folder.
   `shapes4.py` already sets `BUNNY_PLY = "bunny.ply"`, so it is picked up
   automatically. (Set `BUNNY_PLY = None` to go back to the torus.)

2. RETRAIN THE CHECKPOINTS ON THE BUNNY. `run_figures.py` renders whatever the
   `ckpt_L*.npz` were trained on; torus checkpoints render as a torus no matter
   what you evaluate against. You MUST retrain:

       del ckpt_L*.npz hist_L*.npz        (remove any torus checkpoints first)
       py train_one.py 0 1500
       py train_one.py 4 1500
       py train_one.py 10 1500

3. RENDER. `py run_figures.py` — it prints `evaluating against shape: bunny.ply`
   at the top. If it prints `rippled_torus`, the mesh file was not found.

Or do everything in one command (trains on whatever shape is active, then
renders every figure):  `py run_section4.py`

Safeguards: `run_figures.py` stops with a clear message if checkpoints are
missing, or if they were trained on a different shape than you're evaluating
against (each checkpoint records its training shape).

### Getting bunny.ply
- Open3D:  py -c "import open3d as o3d, shutil; shutil.copy(o3d.data.BunnyMesh().path,'bunny.ply')"
- Or download bun_zipper.ply from the Stanford 3D Scanning Repository and rename.

### Notes on the bunny path
- The first bunny run labels a 128^3 grid with trimesh.contains (~2M tests).
  Slow the first time, then CACHED to occ_bunny.ply_128.npz (instant after).
  `pip install pyembree` speeds it up a lot; lower grid_res for more speed.
- We call mesh.fill_holes() because the classic bunny is open at the base.
- Different spectral-bias story than the torus: the bunny's hard parts are the
  thin ears / fine detail (lost without encoding); the smooth body looks fine
  even at L=0. Raise steps/grid_res if the contrast is subtle.

## Five parts (torus reference numbers)

  representation            memory     IoU    Chamfer x1e-3
  ground truth                 -      1.000      0.0
  MLP, no encoding (L=0)    519 KiB   0.779     28.5
  MLP, L=4                  543 KiB   0.971     10.8
  MLP, L=10                 579 KiB   0.954     11.7

Without positional encoding the network smooths high-frequency detail away
(spectral bias); more bands recover it, though bands beyond the shape's own
frequency content add off-surface noise, so there is an optimal L. Part 5 meshes
one trained network at 32/64/128^3 from the same weights: resolution is a
property of the query, not the representation.

## Caveats
- Speed: ~1-2 min/model for 1500 steps in NumPy; seconds in PyTorch on GPU.
- Denormal floats: trained weights accumulate subnormal values that hit a slow
  CPU path; CoordinateMLP.load flushes them.

## `neural_occupancy.py` — the SlangPy port

`neural_occupancy.py` is a separate, self-contained file ("the whole of
Section 4 in one file" per its own docstring) that the chapter text quotes
directly. Its MLP forward/backward pass and Adam optimizer now run on the
GPU via SlangPy (`neural_occupancy.slang`), using the same
`NetworkParameters` pattern as Chapter 4, instead of the hand-rolled NumPy
backprop the section originally used; the geometry, sampling, marching
cubes, and plotting stay in NumPy/scikit-image/matplotlib. This does *not*
touch `field4.py` / `train_one.py` / `run_figures.py` / `run_section4.py`
above, which are a separate, more instrumented pipeline (checkpoints, the
IoU/Chamfer table, bunny support) and still use their own NumPy backprop.

```
pip install slangpy scikit-image matplotlib
python neural_occupancy.py [steps=1500] [L=6]
```

Tested with slangpy 0.43.1 on the Vulkan backend, Python 3.11. Notable
differences from the NumPy version, and why -- worth reading if you touch
this file:

- **Hidden width is 64, not 256.** Differentiating the per-element weight
  loop at 256-wide did not finish compiling in this Slang toolchain even
  after 15+ minutes (32-wide: ~1.5s, 64-wide: ~9-25s depending on loop
  strategy -- clearly superlinear, and 256-wide falls off that curve
  entirely). 64 is the widest that compiled reliably in single-digit
  seconds. Going wider would need the network rewritten around Slang's
  native `matrix<T,M,N>` ops instead of a scalar per-element loop, which is
  a larger change than this port covers. Expect a correspondingly lower IoU
  than the table above (this port reaches ~0.56 at L=6, 1500 steps).
- **`L` is a preprocessor `#define`, not a Slang generic.** A generic
  `Network<let L : int>` struct did not survive `bwd_diff` in testing -- the
  compiler either hung indefinitely or crashed differentiating through a
  struct whose field types depend on a generic int parameter. Plain structs
  sized from a `#define` compile and train correctly, so the Python driver
  passes `L` in via the SlangPy `defines` compiler option.
- **The `predict_batch` kernel is "warmed up" with a dummy call before
  training starts.** Calling it for the first time only *after* hundreds of
  `calculate_grads` dispatches intermittently failed to allocate its GPU
  buffers (`createBuffer` / `SLANG_FAIL`); triggering that one-time pipeline
  creation up front avoids it. Reusing persistent GPU buffers (via
  `copy_from_numpy`) instead of allocating a fresh one every training step
  was also necessary -- allocating thousands of short-lived buffers back to
  back triggered the same failure.

These read as real limitations/bugs in this specific Slang/SlangPy version,
not deliberate design choices -- a newer toolchain release may not need them.
