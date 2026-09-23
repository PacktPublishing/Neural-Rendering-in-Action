# Chapter 4 code

These samples build on the Slang and SlangPy environment set up in Chapter 3.

```
pip install slangpy==0.42.0
```

Tested with slangpy 0.42.0 on the **Vulkan** backend, Python 3.11, on an NVIDIA GPU
(RTX 3050 Laptop). `App` defaults to `spy.DeviceType.vulkan`; pass a different
`device_type` to `App(...)` to try another backend.

**Do not upgrade past 0.42.0.** slangpy 0.43.0 (and 0.43.1) ship a gradient-tensor
bug in their rewritten native `Tensor`/`AtomicTensor` implementation
([slangpy PR #1000, "Native tensor"](https://github.com/shader-slang/slangpy/pull/1000))
that silently corrupts gradient accumulation for exactly the
`NetworkParameters`/`AtomicTensor.add()` pattern these samples use. Training
still runs and the loss still drops for the first few hundred steps, but it
then plateaus roughly an order of magnitude above the loss these samples
should reach (confirmed by bisection: 0.33.1 through 0.42.0 all train
correctly; 0.43.0 and 0.43.1 do not). There is no known application-level
workaround -- constructing gradient tensors independently instead of via
`zeros_like()` does not avoid it, so the bug appears to be in slangpy's
internal differentiable-dispatch wiring rather than in tensor allocation.

Each sample opens a window and trains a small MLP live, printing the step and
loss to the console. Press `Escape` to close the window, `F1` to preview the
output texture in [tev](https://github.com/Tom94/tev), or `F2` to save a
screenshot.

## Samples

| File | Description |
|---|---|
| `01_ReLU_activation.py` / `.slang` | Baseline: a 3-layer MLP (2→32→32→3) with Leaky ReLU activations, coordinate input only. |
| `02_siren_activation.py` / `.slang` | Replaces Leaky ReLU with a SIREN sine activation, sin(ω₀·x) with ω₀ = 30, and the matching ±√(6/n)/ω₀ weight initialization (Sitzmann et al. 2020). |
| `03_frequency_encoding.py` / `.slang` | Encodes the UV input as 16 sine/cosine features (`NumOctaves = 4`, `scale = 2π·2^octave`) before the same MLP shape, with Leaky ReLU activations in the hidden layers. |
| `04_latent_texture.py` / `.slang` | Replaces the raw coordinate input with a bilinearly-sampled 32×32×4 learnable latent texture. |
| `05_neural_block_compression.py` / `.slang` | Compresses a small PBR material set (albedo, normal, roughness) sharing one latent texture and MLP decoder. |

Run any sample directly, e.g.:

```
python 01_ReLU_activation.py
```

Expected (loss decreases steadily over the first few hundred steps; exact
values vary by run):

```
Compiling shaders... this may take a while
Step     20 | Loss: 0.43106
Step     40 | Loss: 0.27354
...
Step    360 | Loss: 0.06648
```

## Experimental `neural.slang` module

SlangPy ships an experimental, undocumented `neural` module
(`slang-standard-module/slang/neural.slang` in the installed package) that
abstracts common MLP primitives -- layers, activations, and parameter
storage/address management -- behind a higher-level API. It is explicitly
marked `[ExperimentalModule]` with no stable public documentation or
reference usage in the SlangPy package itself, so these samples continue to
use the explicit `NetworkParameters` struct shown in the chapter instead.
