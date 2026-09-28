# Chapter 9 — 3D Gaussian Splatting

A hand-held video goes in; a scene you can fly around in real time comes out. Part 1 is Python:
structure from motion, then gradient descent on the Gaussians. Part 2 is C++ and Vulkan, on
[LightweightVK](https://github.com/corporateshark/lightweightvk). Cameras are OpenCV in both:
x right, y **down**, z forward.

| File | Listing | |
|---|---|---|
| `splatting.py` | 9.1 | All of Part 1 in one module, thirteen sections in chapter order |
| `01_reconstruct.py` | 9.3 | The driver: video in, `splats.ply` and `cameras.json` out |
| `viewer/main.cpp` | 9.2 | The real-time viewer; five non-obvious things marked GOTCHA |
| `02_compare_renderers.py` | 9.4 | Renders one camera both ways and differences the images |
| `deploy_deps.py`, `figures/build.py` | — | Third-party bootstrap; Graphviz figure sources |

## Part 1 — video to splats

SIFT is in the main OpenCV package, so `opencv-python` is enough.

```
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install opencv-python scipy numpy
python 01_reconstruct.py ../video.mp4
```

Every stage writes into `work/` and can be skipped if its result is already there — structure from
motion takes minutes, splat optimization tens of minutes.

```
python 01_reconstruct.py ../video.mp4 --stop-after frames   # keyframes only
python 01_reconstruct.py --skip frames sfm --iterations 15000
```

Defaults are the chapter's: 350 keyframes, 7,000 iterations, 450,000 Gaussians, focal prior 1.05
image widths. `--optimize-focal` is off deliberately; see the note on `splatting.SfMConfig`.
Training holds out every eighth view, and that held-out PSNR is the only number that means
anything. Part 2 needs both output files: a PLY carries no camera information, and a viewer left to
guess where to stand shows a forward-facing capture as shattered glass.

## Part 2 — the viewer

One CMake project; it bootstraps `Chapter09/deps/` itself. Needs CMake 3.22+, a C++20 compiler, a Vulkan GPU,
and Python on `PATH`. Run from the repository root:

```
cmake -S Chapter09/viewer -B .build
cmake --build .build --config Release
.build/Release/splat_viewer Chapter09/work/splats.ply Chapter09/work/cameras.json
```

Both paths are optional — with neither, the viewer offers what it finds nearby. `W/S/A/D` and
`1`/`2` move, `Shift` is fast, left mouse looks, the wheel dollies, `Space` resets, `[` and `]`
step the spherical-harmonic degree, `Esc` quits. `--screenshot out.png` renders the camera from
`cameras.json` and exits.

## Do the two renderers agree?

```
python 02_compare_renderers.py --unit   # synthetic, non-overlapping splats
python 02_compare_renderers.py          # the real reconstruction
```

About 0.45/255 mean error, 1/255 at the 99th percentile, and `work/compare_renderers.jpg` showing
PyTorch, C++ and the 8x difference. Nothing overlaps in `--unit`, so blend order cannot matter
there: a clean `--unit` with a dirty real scene points at ordering, not at the per-splat
mathematics. That split caught the viewer's 16-bit depth sort — 12.6/255 on the real scene, clean
on `--unit`.

## Figures

`python figures/build.py` renders every `*.dot` to a 300 DPI PNG. The PNGs are build products: edit
the text sources, never the images.
