# Chapter 10: Generative Models for Graphics

The [diffusion primer notebook](diffusion_primer.ipynb) moves from small
GANs and VAEs to diffusion models, Stable Diffusion, and ControlNet.
It runs independently of the other chapters.

## Run in Colab

[Open the notebook in Colab](https://colab.research.google.com/github/PacktPublishing/Neural-Rendering-in-Action/blob/main/Chapter10/diffusion_primer.ipynb),
select a GPU under **Runtime > Change runtime type**, then run the cells
from top to bottom. Later sections reuse models and helpers defined earlier.
The setup cell installs the extra packages needed in Colab.

## Run locally

Use Python 3.12, preferably in a virtual environment. For GPU execution,
install a matching PyTorch/torchvision CUDA build using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/)
first. From `Chapter10`:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab diffusion_primer.ipynb
```

Select the kernel for this environment. You can skip the notebook's first
`%pip install` cell when the dependencies are already installed; that cell
upgrades the diffusion libraries.

Parts 1–6 can run on the CPU, though image-model training takes longer.
Use an NVIDIA CUDA-capable GPU for Parts 7–10. The notebook selects CUDA
when available and otherwise falls back to the CPU.

## What's in the notebook

| Parts | Examples |
| --- | --- |
| 1–2 | A GAN on 2D points and a VAE trained on MNIST. |
| 3–6 | Noise schedules, diffusion on 2D points, a Fashion-MNIST DDPM, and a larger U-Net. |
| 7–8 | Stable Diffusion's denoising loop and a classifier-free guidance comparison. |
| 9 | ControlNet generation conditioned on a synthetic sphere depth map. |
| 10 | Texture tiling with circular padding. |
| 11 | A conceptual mesh-texturing loop requiring your own renderer and UV-mapped mesh. |

The first run downloads MNIST and Fashion-MNIST into `data/`, followed by
several gigabytes of pretrained weights for
[Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
and [depth ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-depth).
Plots and generated images appear in the notebook.

Some examples use shorter runs than the book: Fashion-MNIST training defaults
to five epochs, and Stable Diffusion uses 30 denoising steps. The notebook
marks these settings where they are used. It uses seed 42; results can still
vary across hardware and library versions.

