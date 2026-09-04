# Chapter 2 code

For OpenCV in Python, install it using

``pip install opencv-python``

Bundle adjustment additionally requires SciPy:

``pip install scipy``

Tested with OpenCV 4.12, NumPy 2.2, SciPy 1.16, and Python 3.11.

## StereoBM script

File: ``stereobm.py``

Creates a GUI to tune numDisparities, blockSize, uniquenessRatio, and
speckleWindowSize, and displays the resulting disparity map for the bundled
stereo pair (`im2.png` / `im6.png`, the Middlebury 2003 "Cones" pair). Press
`q` to close the window.

The `BASELINE_M` / `FOCAL_PX` constants in the script are illustrative only
and are not this pair's calibration -- the depth values it prints are not
metric. See the comments at the top of the script for guidance on choosing
the numDisparities search range, and for the exact slider values used to
produce the book's Figure 2.7.

## Bundle adjustment script

File: ``bundleAdjustment.py``

Runs a minimal structure-from-motion pipeline on the bundled stereo pair:
SIFT feature detection and matching, essential-matrix pose estimation,
manual triangulation, and a SciPy `least_squares` bundle adjustment over
camera poses and 3D points. It prints progress and a final optimisation
summary; no window is opened.

Run it with:

```
python bundleAdjustment.py
```

Expected output (values such as keypoint counts, matches, and the final
cost will vary slightly by OpenCV version and are not reproduced exactly):

```
Loaded 2 images successfully.
Camera intrinsics: focal length 800, principal point (225.0, 187.5)

Processing pair: image 0 and image 1
Image 0: 1242 keypoints, descriptors shape: (1242, 128)
Image 1: 1199 keypoints, descriptors shape: (1199, 128)
Total matches found: 1242
Good matches after ratio test: 534
Essential matrix:
...
Fundamental matrix:
...
Rank of fundamental matrix: 2
Inliers after essential matrix estimation: 518
Successfully triangulated 518 points between image 0 and 1.

Total 3D points triangulated across all pairs: 518

Starting bundle adjustment...
Initial parameters shape: (1560,)
`ftol` termination condition is satisfied.
Function evaluations 10, initial cost 5.2223e+00, final cost 4.6043e+00, first-order optimality 7.04e-01.
Bundle Adjustment Optimized!
Final cost: 4.604310538013865, number of function evaluations: 10
```

If "Inliers after essential matrix estimation" prints 0, bundle adjustment
is skipped entirely -- this previously happened because the script filtered
`recoverPose()`'s inlier mask with `mask == 255`, but not all OpenCV builds
mark inliers with exactly 255 (this build uses 1). The script now filters
with a nonzero test (`mask != 0`) instead.

## COLMAP reconstruction pipeline

The chapter's COLMAP walkthrough (frame extraction through sparse
reconstruction) isn't a standalone script here, but the commands are
version-sensitive enough to be worth pinning down.

Tested with **COLMAP 4.2.0** (commit `be5e291`, built with CUDA). Install
from https://github.com/colmap/colmap/releases; the Windows installer link
in the chapter is Windows-only -- other platforms should build from source
or use the instructions at https://github.com/colmap/colmap#download.

```
mkdir images
ffmpeg -i input_video.mp4 -vf fps=2 -qscale:v 2 images/frame_%06d.jpg

colmap feature_extractor --database_path database.db --image_path images ^
    --ImageReader.single_camera 1 ^
    --FeatureExtraction.use_gpu 1 --FeatureExtraction.max_image_size 8192

colmap sequential_matcher --database_path database.db ^
    --SequentialMatching.overlap 15

mkdir sparse
colmap mapper --database_path database.db --image_path images ^
    --output_path sparse --Mapper.num_threads 8
```

(`^` is the Windows line-continuation character; use `\` on Linux/macOS.)
`ffmpeg` must create `images/` itself -- it won't, so `mkdir images` first.
Likewise `mapper` needs `sparse/` to already exist. The bundled `im2.png`
and `im6.png` are a wide-baseline stereo pair, not a video sequence, so
running `mapper` on just those two will correctly fail with "No good
initial image pair found" -- that step needs a real frame sequence
(roughly 150-300 frames per the extraction guidance above), not the
StereoBM/bundle-adjustment sample images.

**`--FeatureExtraction`/`--SiftExtraction` flag namespace by version** --
this has changed twice and picking the wrong one for your install will
fail with `unrecognised option`:

| COLMAP version | `use_gpu` | `max_image_size` |
|---|---|---|
| 3.12.x and earlier | `--SiftExtraction.use_gpu` | `--SiftExtraction.max_image_size` |
| 3.13 | `--FeatureExtraction.use_gpu` | `--SiftExtraction.max_image_size` |
| 4.2.0 (tested here) | `--FeatureExtraction.use_gpu` | `--FeatureExtraction.max_image_size` |

If your `colmap feature_extractor --help` output doesn't match the
command above, check the `SiftExtraction`/`FeatureExtraction` prefix in
its listed options and adjust accordingly. All other flags used here
(`--ImageReader.single_camera`, `--SequentialMatching.overlap`,
`--Mapper.num_threads`) are unchanged across 3.12.6, 3.13, and 4.2.0.
