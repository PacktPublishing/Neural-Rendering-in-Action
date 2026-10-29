# Chapter 8: Building and Training a NeRF Model

Train a NeRF on the Tiny NeRF dataset, then render the scene from new
camera positions. The implementation includes positional encoding,
volume rendering, and coarse/fine sampling in PyTorch. It does not need
SlangPy or outputs from earlier chapters.

## Setup

Run commands from `Chapter08` using Python 3.12. The recorded dependencies
use the CUDA 12.9 build of PyTorch:

```sh
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
python tiny_nerf_quality_long.py --download-only
```

The dataset is downloaded to `data/tiny_nerf_data.npz` on the first run.
An NVIDIA GPU makes training much faster; `--device auto` selects CUDA
when available and otherwise uses the CPU. For a CPU-only installation,
install `torch==2.8.0 numpy pillow certifi` instead of the CUDA-pinned
requirements.

## Train and view the result

Start with a short run to check setup and output generation:

```sh
python tiny_nerf_quality_long.py --iters 2 --n-rays 32 --outdir runs/tiny_smoke --frames 3 --scale 0.25
```

Two steps only check that the pipeline runs. For the chapter's full run:

```sh
python tiny_nerf_quality_long.py --device cuda --outdir runs/chapter8_reproduction --iters 12000 --n-rays 2048 --frames 36 --scale 2 --phi -30
```

Open `runs/chapter8_reproduction/turntable/viewer.html` in a browser to
browse RGB, depth, and opacity views. The run also saves a checkpoint,
a target/render comparison, and a turntable GIF. Training uses 100 × 100
images; `--scale 2` changes the output renders to 200 × 200.

An existing `checkpoint.pt` in the output folder is reused for rendering.
Use a new `--outdir` or add `--force-train` to train from scratch.
The network and sampling settings are collected in
`TINY_QUALITY_LONG_CONFIG` near the top of the script.

## Camera plot

To draw the training and held-out camera positions:

```sh
python -m pip install matplotlib
python plot_tiny_nerf_cameras.py
```

The plot is saved to `runs/tiny_quality_long/camera_rig.png`.
