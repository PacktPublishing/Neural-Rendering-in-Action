# Chapter 8: Tiny NeRF

Run commands from this `Chapter8` folder. The chapter's coarse and fine networks use the same architecture with **separate learned weights**; `TINY_QUALITY_LONG_CONFIG` controls both.

| Setting | Chapter configuration |
| --- | --- |
| Trunk | Eight fully connected layers, width 256, ReLU after each |
| Skip | `skip=4` is zero-based: concatenate the encoded position after layer 5, into layer 6 |
| Position encoding | 10 frequencies, raw input included, 63 numbers |
| Direction encoding | 4 frequencies, raw input included, 27 numbers |
| Frequency convention | `use_pi=False`: frequencies `2**k`; the paper's equation uses `pi * 2**k` |
| Density head | 256 to 1, no direction input; the renderer applies ReLU |
| Color branch | Feature 256 to 256; concatenate direction to 283; 283 to 128 with ReLU; 128 to 3 with sigmoid |
| Parameters | 595,844 per network; 1,191,688 total |
| Sampling | 64 coarse depths, 64 additional fine depths; fine evaluates the merged 128 |
| Training | Seed 7, 2,048 rays per step, 12,000 steps, Adam learning rate 0.0005 decaying by a factor of 10 |
| Other settings | Central image crop for the first 500 steps; density noise 0; black background |
| Data split | Every eighth image held out: 92 training views, 14 test views; reported example uses test index 0 |

Set `use_pi=True` to experiment with the alternative encoding. Changing that setting changes the model's inputs and requires a new training run. Checkpoint rendering uses the saved configuration.

## Environment and first run

Use the Python/PyTorch environment from Chapter 3, or install the recorded dependencies:

```text
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python tiny_nerf_quality_long.py --download-only
```

The manifest records the package versions used for the checks below. For GPU training, use the PyTorch CUDA build appropriate for your machine; the tested environment had Python 3.12, PyTorch 2.8.0+cu129, NumPy 2.5.2, Pillow 12.3.0, and certifi 2026.7.22 on Windows. CPU execution works but is slower. Linux and macOS have not been tested in this review.

First verify the complete workflow with a short run in its own directory:

```text
python tiny_nerf_quality_long.py --device cuda --iters 2 --n-rays 32 --outdir runs/tiny_smoke --frames 3 --scale 0.25
```

This checks training, checkpoint saving, held-out rendering, PNG/GIF output, and HTML viewer generation. Two steps do not produce a trained reconstruction. Use `--device cpu` for a CPU check. The default `--device auto` chooses CUDA when available; an unavailable explicit `--device cuda` also falls back to CPU and prints a message.

## Reproduce the chapter configuration

Train a new run using the chapter settings:

```text
python tiny_nerf_quality_long.py --device cuda --outdir runs/chapter8_reproduction --iters 12000 --n-rays 2048 --frames 36 --scale 2 --phi -30
```

The remaining settings come from the table above. Training images are 100 by 100 pixels. `--scale 2` renders the turntable at 200 by 200; it does not change the training image resolution. The orbit uses 36 azimuths from -180 degrees up to, excluding, +180 degrees, elevation -30 degrees, and a radius equal to the mean training-camera distance from the origin (approximately 4.0311 for the provided dataset).

Open `runs/chapter8_reproduction/turntable/viewer.html` in a browser. The slider's `8 / 36` position selects zero-based frame 7. The viewer displays RGB, expected depth, and opacity, with depth mapped to the near/far interval and opacity mapped to [0, 1]. `turntable/render_config.json` records the actual camera orbit and output dimensions of newly generated viewers.

Outputs include `checkpoint.pt`, `config.json` for training, `chapter_config.json` for the configuration actually rendered, `target_000.png`, `render_final.png`, `comparison_large.png`, depth and opacity images, and the turntable viewer/GIF. Checkpoints store network weights, configuration, and the completed step for rendering. They are saved at step 1, every `eval_every` steps (4,000 by default), and the final step. 
