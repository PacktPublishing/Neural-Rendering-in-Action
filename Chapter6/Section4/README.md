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
