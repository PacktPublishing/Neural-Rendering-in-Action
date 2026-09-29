# Chapter 6: Representing Geometry and Appearance with Neural Networks

This chapter compares voxel grids and point clouds with learned occupancy,
signed-distance, and appearance fields. The examples share helpers in this
folder, but do not need code or outputs from another chapter.

## Setup

Use Python 3.12. From `Chapter06`, install the packages for the CPU examples:

```sh
python -m pip install numpy scipy scikit-image matplotlib
```

The occupancy, SDF, and appearance examples also need a Vulkan-capable GPU
and SlangPy:

```sh
python -m pip install slangpy==0.42.0
```

## Pick an example

Run each command **inside the folder listed**, so figures and checkpoints
go to the right place. The linked READMEs explain the individual experiments.

| Folder | Run | Covers |
| --- | --- | --- |
| [3d_representations](3d_representations/README.md) | `python run_section3.py` | Voxelization, point clouds, and NumPy neural networks; CPU only. |
| [neural_occupancy_fields](neural_occupancy_fields/README.md) | `python run_section.py` | Learning inside/outside tests and extracting a mesh. |
| [neural_signed_distance](neural_signed_distance/README.md) | `python run_section.py` | Learning SDFs, sphere tracing, and constructive solid geometry. |
| [appearance_fields](appearance_fields/README.md) | `python run_section.py` | Albedo, view-dependent color, and a feature-grid/MLP comparison. |

For example, starting from `Chapter06`:

```sh
cd 3d_representations
python run_section3.py
```

These drivers save plots in their section's `figures/` folder. The appearance
example needs `appearance_fields/checkpoints/ckpt_sdf_supervised.npz`, which
is included. If it is missing, run `python train_one.py supervised 1500`
inside `neural_signed_distance`, then copy the resulting checkpoint into
`appearance_fields/checkpoints/`.

The optional bunny examples use the bundled `bunny.ply` and additionally
need `trimesh` and `embreex`. On machines with multiple GPUs, set
`CHAPTER6_GPU` to a vendor or adapter-name substring, such as `nvidia`;
see the occupancy section's README for details.
