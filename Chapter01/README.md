# Chapter 1: Introduction to Inverse Rendering

The chapter introduces inverse rendering and the role of image features in
3D reconstruction. The example here walks through Harris corner detection,
from image gradients to the final corner locations. It runs on the CPU.

## Run the example

With Python installed, run these commands from `Chapter01`:

```sh
python -m pip install -r requirements.txt
python HarrisCornerDetection.py
```

At the prompt, press Enter to use the generated checkerboard, enter
`testImg.jpg` for the bundled photo, or give the path to your own image.

A Matplotlib window shows the input, horizontal and vertical gradients,
gradient strength, Harris response, and detected corners. To experiment,
change `k`, `threshold`, or `neighborhood_size` in the call to
`harris_corner_detector()` near the end of the script.
