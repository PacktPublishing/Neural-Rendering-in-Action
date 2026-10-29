# Chapter 7: Building a Simple Neural Radiance Cache

This chapter adds a neural radiance cache (NRC) to the Vulkan glTF path
tracer in this folder. A small network trains while the renderer runs and
predicts indirect lighting. The project builds independently of the other
chapters.

## Requirements

- Windows or Linux, Git, CMake 3.22+, and a C++20 compiler
  (Visual Studio 2022 on Windows).
- A Vulkan SDK with cooperative-vector headers, and an NVIDIA GPU/driver
  supporting `VK_NV_cooperative_vector` training with FP32 accumulation.
- The pinned `nvpro_core2` and RTX Neural Shading (RTXNS) checkouts below.

CMake selects Slang 2026.5 for the NRC shaders and downloads additional
build dependencies. CUDA Toolkit is needed if you want the optional OptiX
denoiser.

## Build and run

Run these commands from `Chapter07`. Keep the dependency revisions pinned;
the renderer was written against these versions.

```sh
git clone https://github.com/nvpro-samples/nvpro_core2.git
git -C nvpro_core2 checkout 334ed20e4bde4623ce507ca01b9f2d3b7cd6a846
git clone https://github.com/NVIDIA-RTX/RTXNS.git
git -C RTXNS checkout 1a2513120190a43f61c23590811a8b80ec914bd7
git -C RTXNS submodule update --init --recursive
```

On Windows, use a terminal with the Visual Studio C++ tools available:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\_bin\Release\Chapter07.exe
```

On Linux, with Ninja installed:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./_bin/Release/Chapter07
```

The executable takes its name from the source folder, so it is `Chapter07`
in this checkout. NRC is enabled by default. In the path tracer settings,
use `NRC Left` and `NRC Right` to compare it with the regular path tracer;
turn off DLSS when comparing NRC.

If RTXNS is installed elsewhere, pass `-DRTXNS_ROOT=<path>` when configuring.
Optional denoisers can be disabled with `-DUSE_DLSS=OFF` and
`-DUSE_OPTIX_DENOISER=OFF`.

## Code and further reading

Commit [bf2ace0](https://github.com/PacktPublishing/Neural-Rendering-in-Action/commit/bf2ace0217267f5f8b8df11f94fbf91a83dd4483)
added NRC to `vk_gltf_renderer`. Its diff shows the cache implementation and
the path tracer changes covered in this chapter.

Start with [src/nrc/](src/nrc/README.md) for the cache implementation and
`src/renderer_pathtracer.cpp` for its integration into the renderer.
The [user guide](docs/user-guide.md) covers scene loading and renderer
controls; the [developer guide](docs/developer.md) covers the wider codebase.

This renderer is based on NVIDIA's Vulkan glTF renderer
