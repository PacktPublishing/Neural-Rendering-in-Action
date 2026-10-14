"""Optional: train the Stanford bunny as a neural SDF, two ways.

Usage:  python3 train_bunny_sdf.py [L] [sup_steps] [eik_steps]
Defaults: L=6, sup_steps=1500, eik_steps=2700.

  1. Supervised (5.2's recipe): regress a precomputed signed-distance grid
     (mesh_shape.build_or_load_sdf_grid) -- needs exact-ish labels, same
     shape as train_supervised's torus recipe.
  2. Eikonal (5.5's recipe): train_eikonal() AS-IS, fed the bunny's surface
     cloud instead of the knurled torus's -- this recipe never needed
     labels in the first place, so no mesh-specific Slang code was needed
     for it at all.

Needs bunny.ply at the chapter root (../bunny.ply) and `pip install
trimesh embreex` -- see ../neural_occupancy_fields/README.md, "Optional:
Stanford bunny mesh support".

Writes checkpoints/ckpt_bunny_sdf_sup_L{L}.npz and
checkpoints/ckpt_bunny_sdf_eik_L{L}.npz. See ../make_bunny_comparison.py
for the combined figure across this and Section 4's occupancy field.
"""

import os
import sys

import numpy as np
import slangpy as spy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neural_sdf as F                 # sets up chapter root on sys.path
import mesh_shape as MS

HERE = os.path.dirname(os.path.abspath(__file__))
PLY = os.path.join(HERE, os.pardir, "bunny.ply")
CHECKPOINTS = os.path.join(HERE, "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SUP_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
EIK_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2700
GRID_RES = 128

if not os.path.exists(PLY):
    raise SystemExit(f"missing {PLY} -- see ../neural_occupancy_fields/README.md")

print(f"Part 1: load + cache the bunny (grid {GRID_RES}^3)")
occ, origin, spacing, cloud, V, Ftri, mesh = MS.load_and_cache(
    PLY, grid_res=GRID_RES, cache_dir=CHECKPOINTS)
print(f"  {len(V):,} verts, {len(Ftri):,} faces, {cloud.shape[0]:,}-pt surface cloud")

sdf_grid, sdf_origin, sdf_spacing = MS.build_or_load_sdf_grid(
    occ, origin, spacing, V, Ftri, GRID_RES, cache_dir=CHECKPOINTS)

print(f"\nPart 2: supervised SDF (L={L}, {SUP_STEPS} steps)")
model_sup = F.Network(L=L, hidden=F.HIDDEN, seed=0)
h_sup = F.train_supervised_mesh(model_sup, sdf_grid, sdf_origin, sdf_spacing,
                                GRID_RES, cloud, steps=SUP_STEPS)
print(f"  {h_sup['seconds']:.0f}s")
ckpt_sup = os.path.join(CHECKPOINTS, f"ckpt_bunny_sdf_sup_L{L}.npz")
model_sup.save(ckpt_sup)
print(f"  [saved {ckpt_sup}]")

print(f"\nPart 3: eikonal SDF from points alone (L={L}, {EIK_STEPS} steps)")
# train_eikonal() is completely generic over its cloud_t -- no bunny-specific
# Slang code was needed for this recipe at all, only this upload.
# hidden=256, not neural_sdf.HIDDEN's default 64: this recipe needs more
# capacity than the supervised one to recover fine detail from points
# alone -- already measured for the torus (see this file's own git log,
# "bump eikonal to hidden=256": hidden=64 there converges to a hole-less
# blob, same failure shape as an undertrained bunny here).
model_eik = F.Network(L=L, hidden=256, seed=1)
cloud_t = spy.Tensor.empty(F._device, (cloud.shape[0],), "float3")
cloud_t.copy_from_numpy(np.ascontiguousarray(cloud, np.float32))
h_eik = F.train_eikonal(model_eik, cloud_t, steps=EIK_STEPS, seed=1)
print(f"  {h_eik['seconds']:.0f}s")
ckpt_eik = os.path.join(CHECKPOINTS, f"ckpt_bunny_sdf_eik_L{L}.npz")
model_eik.save(ckpt_eik)
print(f"  [saved {ckpt_eik}]")
