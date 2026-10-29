# Chapter 5: Implementing a Differentiable Renderer

Four examples recover material, shape, or lighting parameters by comparing
renders with target images. The first two use SlangPy, the lighting notebook
uses PyTorch3D, and the final example is a differentiable fork of Aurora.

## Material and shape examples

Use Python 3.12 and a GPU supported by SlangPy. Run commands from `Chapter05`:

```sh
python -m pip install slangpy==0.42.0 numpy
python main_section1.py
```

`main_section1.py` recovers material parameters with differentiable shading.
Run `python main_section2.py` for SDF shape fitting. Both create their own
scenes and open an interactive window. Space pauses or resumes optimization,
R resets it, and Escape closes the window.

Keep the SlangPy 0.42.0 pin; versions 0.43.0 and 0.43.1 produced gradient
accumulation problems in these examples.

## Lighting notebook

`sh_lighting_tutorial_section3.ipynb` fits spherical-harmonic lighting using
PyTorch3D. The chapter's Windows setup uses Python 3.12 and an NVIDIA
CUDA-capable GPU, with this matching PyTorch/PyTorch3D pair:

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install numpy matplotlib jupyterlab
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+d9839a9pt2.8.0cu129
python -m jupyterlab sh_lighting_tutorial_section3.ipynb
```

Select the kernel for this environment and run the cells in order. The
PyTorch3D wheel above comes from a third-party wheel index and must match
the installed PyTorch and CUDA versions.

## Aurora path tracer

The fourth example lives in [Aurora/](Aurora/README.md). Its differentiable
path was developed on Windows with a DirectX Raytracing-capable GPU.
Follow its [build instructions](Aurora/Doc/Build.md) for the C++ toolchain,
SDKs, and dependencies, then use the chapter's walkthrough to explore the
differentiable renderer. Aurora has its own build and does not need the
Python examples to be run first.

The differentiable renderer is based on Autodesk's [Aurora](https://github.com/Autodesk/Aurora). Commit [98d8d20](https://github.com/PacktPublishing/Neural-Rendering-in-Action/commit/98d8d20b27b126c7d40d4dfed717b957fb7ae8d8) adds the differentiable rendering changes.
