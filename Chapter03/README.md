# Chapter 3: Neural Rendering Toolkit

Start here for the Slang, SlangPy, and PyTorch examples: automatic
differentiation, fitting an image, training a digit classifier, and
optimizing a sparse voxel scene. No code or output from earlier chapters
is needed.

## Setup

The chapter uses Python 3.12, an NVIDIA GPU, and CUDA Toolkit 12.9 with a
compatible driver. Run the following from `Chapter03`, preferably in a
virtual environment:

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install slangpy==0.42.0 numpy matplotlib pillow
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

SlangPy includes the Slang compiler used by the Python examples. Keep the
`.slang` files beside the scripts. The MNIST example explicitly uses the
CUDA backend.

## Examples

| File | What it does |
| --- | --- |
| `01_autodiff_basics_forward.py` | Computes a derivative with forward-mode autodiff. |
| `01_autodiff_basics_backward.py` | Computes parameter gradients with reverse-mode autodiff. |
| `02_match_target_image.py` | Fits a circle to a generated target image and saves results in `results/`. |
| `03_train_mnist_slang.py` | Trains an MLP in Slang; downloads MNIST to `data/` on the first run. |
| `04_spc_neural_optimization_tutorial.ipynb` | Uses Kaolin to optimize voxel colors and densities against a target image. |

Run a script directly, for example:

```sh
python 01_autodiff_basics_forward.py
python 02_match_target_image.py
```

## Voxel notebook

The notebook also needs Kaolin, JupyterLab, and plotting packages. These
commands use the same PyTorch/CUDA pairing as above:

```sh
python -m pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html
python -m pip install jupyterlab ipykernel==6.29.5 jupyter-client==7.4.9 plotly tqdm ipywidgets
python -m jupyterlab 04_spc_neural_optimization_tutorial.ipynb
```

Select the notebook kernel for this environment and run the cells in order.
Kaolin's sparse voxel rendering operations require CUDA.
