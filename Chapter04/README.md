# Chapter 4: Implementing Texture Compression Using Neural Networks

Five examples build up a neural texture compressor, starting with a small
coordinate MLP and ending with a shared decoder for PBR material textures.
Each example can be run on its own.

## Setup and first run

Use Python 3.11 or 3.12 and a Vulkan-capable GPU. From `Chapter04`:

```sh
python -m pip install slangpy==0.42.0 numpy pillow
python 01_ReLU_activation.py
```

Keep SlangPy at 0.42.0: versions 0.43.0 and 0.43.1 produced incorrect
gradient accumulation in these samples.

## Examples

| Script | What changes |
| --- | --- |
| `01_ReLU_activation.py` | Baseline coordinate MLP with Leaky ReLU activations. |
| `02_siren_activation.py` | Sine activations and SIREN weight initialization. |
| `03_frequency_encoding.py` | Sine/cosine encoding of the input coordinates. |
| `04_latent_texture.py` | A learnable latent texture feeding a small decoder. |
| `05_neural_block_compression.py` | A shared latent texture and decoder for albedo, normals, and roughness. |

Run any script with `python <script-name>`. Each opens a window showing
training progress and prints the loss. Press Escape to close it or F2 to
save a screenshot. The input textures are included; the PBR example
downloads replacements if its source textures are missing.

The corresponding `.slang` files contain the networks and training kernels;
`app.py` supplies the shared window and Vulkan setup. The optional
`make_technique_comparison.py` figure script also needs Matplotlib and uses
Direct3D 12, so run it on Windows.
