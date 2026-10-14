# Chapter 5 code

The first two recipes (`main_section1.py`, differentiable material recovery, and `main_section2.py`, SDF shape optimization) build on the Slang and SlangPy environment set up in Chapter 3.

```
pip install slangpy==0.42.0
```

**Do not upgrade past 0.42.0.** slangpy 0.43.0 and 0.43.1 ship a gradient-tensor bug in their rewritten native `Tensor`/`AtomicTensor` implementation that silently corrupts gradient accumulation for the `NetworkParameters`/`AtomicTensor.add()` pattern both of these recipes use. Training still runs and the loss still drops at first, but then plateaus well above where it should. See `Chapter04/README.md` for the full writeup of this bug.

The third recipe (the spherical-harmonic lighting notebook) additionally needs PyTorch3D on top of the same environment:

```
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+d9839a9pt2.8.0cu129
```

Tested with Python 3.12.6, torch 2.8.0+cu129, torchvision 0.23.0+cu129, pytorch3d 0.7.9+d9839a9pt2.8.0cu129, slangpy 0.42.0, numpy 2.5.2, matplotlib 3.11.1, and jupyterlab 4.6.3, on Windows 11 with a CUDA-capable GPU.

The fourth recipe builds the differentiable fork of Autodesk's Aurora path tracer in `Aurora/`. See `Aurora/README.md` for its own build prerequisites (Visual Studio 2022, CMake 3.29.3+, Python 3.11+ with PySide6/PyOpenGL/Jinja2, NASM 2.16, Vulkan SDK 1.3.296.0+, and a GPU with hardware ray tracing support).
