# Section 3 — Voxel Grids and Point Clouds: Discrete Neural-Compatible Representations

Reference implementation for the book chapter, written from scratch in
**pure NumPy + SciPy + scikit-image + matplotlib** so that every algorithm the
text describes — including the voxelizer and both neural networks' backward
passes — is visible in the code. Each file is a numbered chapter listing.

## Running it

```
python3 run_section3.py              # opens an interactive window per experiment
python3 run_section3.py --headless   # no display: figures saved to ./figures only
```

The script auto-detects the absence of a display (servers, CI) and falls back
to headless mode. Total runtime ≈ 2–3 minutes on a laptop CPU; Experiment 6
trains a neural network (~50 s in NumPy).

## Chapter listings

| listing | file | contents |
|---|---|---|
| 3.1–3.2 | `geometry.py` | procedural torus mesh; ray-stabbing voxelizer; trilinear interpolation + analytic gradient; marching cubes; area-weighted surface sampling; memory accounting |
| 3.3 | `pointnet.py` | sphere/torus/box point-cloud dataset; mini-PointNet and flattened-MLP baseline with hand-written backprop; Adam; permutation stress test |
| 3.4 | `neuralfield.py` | neural occupancy field: Fourier positional encoding + 3-layer coordinate MLP, manual backprop, BCE training, grid evaluation for meshing |
| 3.5 | `metrics.py` | volumetric IoU (Monte Carlo) and symmetric Chamfer distance |
| 3.6 | `run_section3.py` | the six experiments below |

## The six experiments

1. **Voxelization round trip** (`fig1`). A genus-1 torus mesh → occupancy
   grids at R = 8/16/32/64 (z-axis ray stabbing) → meshes (marching cubes).
   Prints voxelization/extraction *times* per resolution and verifies via
   Euler characteristic that the hole — the topology — survives.

2. **The cubic memory curse** (`fig2`). Grid memory grows O(R³): a 512³ float
   grid costs 0.5 GiB; the mesh is 162 KiB; the Experiment-6 neural field is
   85 KiB.

3. **Trilinear interpolation = a continuous field** (`fig3`). The discrete
   grid becomes a continuous, piecewise-differentiable function. Prints query
   throughput and a gradient check (analytic vs. finite differences, ~3e-10).
   Conceptual bridge: a neural field replaces "fetch 8 corners and blend"
   with "evaluate an MLP".

4. **Point clouds** (`fig4`). Area-weighted surface sampling with the
   sqrt-barycentric trick; prints sampling time and memory per density.

5. **Permutation invariance** (`fig5`). Mini-PointNet vs. flattened MLP: both
   reach ~99–100% on consistently ordered clouds; shuffle the point order and
   the flat MLP collapses to ~43% (chance 33%) while PointNet's logits change
   by exactly 0.0. Invariance is architectural, not learned.

6. **Classical vs. neural, side by side** (`fig6`, the centerpiece window).
   Ground-truth mesh | voxel grid + marching cubes | neural occupancy field,
   rendered with Lambertian shading and annotated with memory/IoU/Chamfer.
   A printed table compares build time, memory, extraction time, mesh
   complexity, IoU, and Chamfer distance. Representative run:

   | representation | build (s) | mem (KiB) | mesh (s) | verts | IoU | Chamfer ×1e-3 |
   |---|---|---|---|---|---|---|
   | ground-truth mesh | 0.0 | 162 | — | 4,608 | 1.000 | 0.00 |
   | voxel grid R=32 | 0.3 | 32 | 0.00 | 4,992 | 0.953 | 13.4 |
   | neural field @128³ | 50.5 | 85 | 8.1 | 39,910 | 0.987 | 10.7 |

   The 85 KiB network beats the voxel grid on both fidelity metrics, and the
   same weights can be meshed at 32³, 64³, or 128³ — resolution is a property
   of the *query*, not of the representation. The tradeoff is also honest:
   "build" costs 50 s of optimization and meshing costs millions of network
   evaluations.

## Suggested reader exercises

- Bit-pack the occupancy grid (`np.packbits`) and redraw the memory figure.
- Replace ray stabbing with a winding-number test; compare on a mesh with a
  hole punched in it (non-watertight).
- Retrain the neural field with `L = 0` (no positional encoding) and watch
  the IoU drop — you have just discovered spectral bias, the opening problem
  of Section 4.
- Make the grid VALUES trainable: backpropagate through
  `trilinear_interpolate` to fit a grid to point samples — the "explicit"
  half of the hybrid representations in Section 6.
