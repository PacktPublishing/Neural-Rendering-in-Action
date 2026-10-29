<h1 align="center">Neural Rendering in Action</h1>

<h3 align="center">
Build Modern Rendering Systems with Differentiable Rendering, NeRFs, and Gaussian Splatting
</h3>

<p align="center">
Preetish Kakkar · Mauricio Maurer · Sergey Kosarevsky · Prashant Kumar
</p>

<p align="center">
Companion code for <a href="https://www.amazon.com/Neural-Rendering-Action-Differentiable-Splatting/dp/1806700654">Neural Rendering in Action</a>, published by Packt.
</p>

<p align="center">
<a href="https://www.amazon.com/Neural-Rendering-Action-Differentiable-Splatting/dp/1806700654">Find the book on Amazon</a> ·
<a href="#chapters">Browse the chapters</a> ·
<a href="#getting-started">Run your first example</a>
</p>

<details open>
<summary><h2>About the book</h2></summary>

<a href="https://www.amazon.com/Neural-Rendering-Action-Differentiable-Splatting/dp/1806700654">
<img src="https://m.media-amazon.com/images/I/81EqFmqC75L._SL1500_.jpg" alt="Book cover" height="256" align="right">
</a>

How do you recover a scene from photographs, train a renderer, or replace part
of a graphics pipeline with a neural network? This book works through those
questions with Python, PyTorch, Slang, and C++ examples.

The chapters begin with image features, camera geometry, and reconstruction.
From there, you write differentiable shaders, compress textures with neural
networks, and fit geometry, materials, and lighting to target images. Later
examples cover neural radiance caching, NeRF training, a 3D Gaussian Splatting
reconstruction and viewer, and diffusion models for image and texture generation.

This repository contains the code and notebooks for all ten chapters. Most
chapters can be used independently; each folder has its own README with setup
instructions, entry points, and notes about the examples.

<br clear="right">

</details>

<details open>
<summary><h2>What you will learn</h2></summary>

- Recover camera poses and 3D structure from multiple images.
- Use Slang and SlangPy to compute gradients and train networks on the GPU.
- Compare coordinate networks, frequency encodings, and latent textures for compression.
- Optimize materials, shape, and lighting with differentiable rendering.
- Represent geometry and appearance with occupancy fields, SDFs, and neural fields.
- Train a neural radiance cache inside a Vulkan path tracer.
- Build a NeRF and reconstruct scenes with 3D Gaussian Splatting.
- Explore diffusion, classifier-free guidance, ControlNet, and texture tiling.

</details>

<a id="chapters"></a>

<details open>
<summary><h2>Chapters</h2></summary>

Follow a chapter link for its code, dependencies, and run instructions.

| Chapter | Topic |
| --- | --- |
| [01](Chapter01/README.md) | Introduction to Inverse Rendering |
| [02](Chapter02/README.md) | 3D Scene Understanding |
| [03](Chapter03/README.md) | Neural Rendering Toolkit |
| [04](Chapter04/README.md) | Implementing Texture Compression Using Neural Networks |
| [05](Chapter05/README.md) | Implementing a Differentiable Renderer |
| [06](Chapter06/README.md) | Representing Geometry and Appearance with Neural Networks |
| [07](Chapter07/README.md) | Implementing a Simple Neural Radiance Cache |
| [08](Chapter08/README.md) | Building and Training a NeRF Model |
| [09](Chapter09/README.md) | Real-Time Rendering with 3D Gaussian Splatting |
| [10](Chapter10/README.md) | Learning the Fundamentals of Diffusion Models |

</details>

<a id="getting-started"></a>

<details open>
<summary><h2>Getting started</h2></summary>

Clone the repository:

```sh
git clone https://github.com/PacktPublishing/Neural-Rendering-in-Action.git
cd Neural-Rendering-in-Action
```

Choose a chapter and follow its README. Run commands from the directory it
specifies, since some scripts load images, shaders, and checkpoints using
relative paths. Use a separate Python virtual environment for a chapter when
its package versions differ from those you already have installed.

For a small CPU example, open `Chapter01` in your Python environment:

```sh
cd Chapter01
python -m pip install -r requirements.txt
python HarrisCornerDetection.py
```

Press Enter at the prompt to use the generated checkerboard, or enter
`testImg.jpg` to use the bundled photo. A window shows the detection process.

</details>

<details open>
<summary><h2>Requirements for this book</h2></summary>

The examples assume familiarity with vectors, matrices, basic graphics
programming, and the fundamentals of neural networks. Python 3.12 is a useful
starting point for the Python chapters.

| Chapters | Main tools and hardware |
| --- | --- |
| 01–02 | Python, NumPy, SciPy, Matplotlib/Pillow or OpenCV; CPU examples. |
| 03 | SlangPy and PyTorch; an NVIDIA CUDA-capable GPU for MNIST and Kaolin. |
| 04, 06 | SlangPy and a Vulkan-capable GPU; Chapter 06 also includes CPU examples. |
| 05 | SlangPy for material and shape fitting, PyTorch3D for lighting, and a separate C++/ray-tracing build for Aurora. |
| 07 | CMake 3.22+, C++20, Vulkan, and NVIDIA cooperative-vector training support; pinned nvpro_core2 and RTXNS dependencies. |
| 08 | PyTorch; CUDA makes NeRF training faster, with a CPU fallback. |
| 09 | PyTorch, OpenCV, NumPy, and SciPy for reconstruction; CMake 3.22+, C++20, and Vulkan for the viewer. |
| 10 | PyTorch, Diffusers, Transformers, and JupyterLab; a CUDA-capable GPU for the larger generation examples, or Colab. |

Check the chapter README before installing packages or building a project.
Several SlangPy examples require version **0.42.0**, and the PyTorch3D and
Kaolin installations must match their PyTorch/CUDA builds. The C++ examples
have their own CMake projects and SDK requirements.

Windows is the main environment described in the book. Platform support and
GPU requirements vary by example. Dataset and pretrained-model downloads
require an internet connection and additional disk space.

</details>

<details>
<summary><h2>Get to know the authors</h2></summary>

**Preetish Kakkar** is a senior graphics engineer with over 15 years of
experience in graphics and AI. His work brings NeRFs, Gaussian Splatting,
and differentiable rendering into production tools and engines.

**Mauricio Maurer** has more than 20 years of rendering experience. He
co-authored *The Modern Vulkan Cookbook* and has worked on AR rendering at
Meta and rendering research at Qualcomm.

**Sergey Kosarevsky** has led rendering at Ubisoft RedLynx and Vulkan
development at Meta. His experience spans mobile, AR, and embedded graphics,
and he co-authored the *3D Graphics Rendering Cookbook* and
*Vulkan 3D Graphics Rendering Cookbook*.

**Prashant Kumar** is a researcher in computer vision and multimodal AI with
a Ph.D. from IIT Roorkee. His background includes applied research and
software development across Samsung Research, Scale AI, Turing, and AFINIT.


</details>

## License

The repository's code is provided under the [MIT License](LICENSE).
Bundled projects retain their own licenses, including
[Aurora](Chapter05/Aurora/LICENSE.md) and the
[Vulkan glTF renderer](Chapter07/LICENSE).
