# Chapter 10 — Generative Models for Graphics: A Diffusion Primer

Companion code for the *Neural Rendering in Action* chapter: GANs, VAEs, diffusion from first
principles, the U-Net denoiser, Stable Diffusion driven by hand, classifier-free guidance,
ControlNet, and seamless tileable textures.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erprashu/neural-rendering-ch10-diffusion/blob/main/ch10_diffusion_primer.ipynb)

## Run it

**Colab** — click the badge, set **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run
all**. Package installs, dataset downloads, and model downloads all happen inside the notebook.

**Locally:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook diffusion_primer.ipynb
```

The notebook follows the chapter section by section and runs top to bottom. Parts 1–6 are fine on
CPU; Parts 7–10 want a GPU. Those parts use
[`stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and
[`sd-controlnet-depth`](https://huggingface.co/lllyasviel/sd-controlnet-depth) — both public and
ungated, so **no HuggingFace account or token is needed**.

A few training runs are scaled down from the book's numbers so the whole notebook finishes quickly
on free Colab hardware; each reduction is called out in place, with the chapter's number alongside
it. Everything stochastic is seeded (`SEED = 42`), so runs reproduce on the same hardware and
library versions.

