# Chapter 2: 3D Scene Understanding

These examples accompany the chapter on cameras, projective geometry, and
multi-view reconstruction. Both use the bundled `im2.png` and `im6.png`
images from the Middlebury Cones stereo pair.

## Setup

Use Python with NumPy, OpenCV, and SciPy. Run commands from `Chapter02`
so the scripts can find the images:

```sh
python -m pip install numpy opencv-python scipy
```

## Examples

- `python stereobm.py` opens a window with sliders for tuning stereo block
  matching. Adjust them to see how the disparity map changes; press `q`
  to close the window.
- `python bundleAdjustment.py` matches SIFT features, estimates camera
  pose, triangulates points, and refines the reconstruction with SciPy.
  It prints the optimization results without opening a window.

The camera intrinsics and stereo baseline in the scripts are illustrative.
The reported depth values should not be treated as calibrated measurements.

## COLMAP walkthrough

The chapter also covers extracting video frames with FFmpeg and reconstructing
them with COLMAP. That walkthrough needs your own image sequence and separate
FFmpeg and COLMAP installations; it is not a script in this folder. Use a
sequence with overlapping views, rather than the two bundled stereo images.
Check `colmap feature_extractor --help` against the chapter commands, since
the `SiftExtraction` and `FeatureExtraction` option names vary by version.
