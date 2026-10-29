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

## Book figures

Files named `fig6_N_*.png` match the book's Figure 6.N. Files named
`extra_*.png` are supplementary plots that do not appear in the book.

| Book figure | Output file | Produced by |
| --- | --- | --- |
| 6.1 Positional encoding bands | `neural_occupancy_fields/figures/fig6_1_encoding_bands.png` | `run_section.py` or `run_figures.py` |
| 6.2 Sphere tracing the neural SDF | `neural_signed_distance/figures/fig6_2_spheretrace.png` | `run_section.py` or `run_figures.py` |
| 6.3 An SDF from surface points alone | `neural_signed_distance/figures/fig6_3_points_alone.png` | `run_section.py` or `run_figures.py` |
| 6.4 CSG on distance fields | `neural_signed_distance/figures/fig6_4_csg.png` | `run_section.py` or `run_figures.py` |
| 6.5 Albedo field | `appearance_fields/figures/fig6_5_albedo.png` | `run_section.py` or `run_figures.py` |
| 6.6 View dependence | `appearance_fields/figures/fig6_6_viewdep.png` | `run_section.py` or `run_figures.py` |
| 6.7 Pure MLP vs feature grid | `appearance_fields/figures/fig6_7_hybrid.png` | `run_section.py` or `run_figures.py` |
| 6.8 The Stanford bunny | `figures/fig6_8_bunny.png` | `make_bunny_comparison.py` |

The optional bunny examples use the bundled `bunny.ply` and additionally
need `trimesh` and `embreex`. On machines with multiple GPUs, set
`CHAPTER6_GPU` to a vendor or adapter-name substring, such as `nvidia`;
see the occupancy section's README for details.
