"""Train Section 5's supervised SDF + Section 6's albedo/radiance fields
FROM SCRATCH, using the ORIGINAL NumPy code's own default hyperparameters
(sdf5.train_supervised: hidden=(256,256), 1500 steps; app6's MLP([...,128,
128,...]), steps=800/1200) -- not the shipped checkpoints, which may have
been trained differently (more steps, a hand-picked seed, or touched up for
print) than what these functions' defaults alone would produce.

Writes checkpoints/*_fresh.npz; run make_figures6.py --suffix _fresh to
render and compare against the shipped originals.
"""
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import app6 as A
import sdf5 as S

os.makedirs(os.path.join(_here, "checkpoints"), exist_ok=True)

print("training Section 5 supervised SDF (hidden=(256,256), 1500 steps)...")
sdfnet = S.train_supervised(steps=1500, seed=0)
np.savez(os.path.join(_here, "checkpoints", "ckpt_sdf_supervised_fresh.npz"),
         L=sdfnet.L, **{f"W{i}": w for i, w in enumerate(sdfnet.W)},
         **{f"b{i}": b for i, b in enumerate(sdfnet.b)})
print("  saved checkpoints/ckpt_sdf_supervised_fresh.npz")

print("training Section 6 albedo field (hidden=128, 800 steps)...")
alb = A.train_albedo_field(steps=800, seed=0)
print("training Section 6 radiance field (hidden=128, 1200 steps)...")
rad = A.train_radiance_field(steps=1200, seed=0)
np.savez(os.path.join(_here, "checkpoints", "ckpt_appearance_fresh.npz"),
         **{f"aW{i}": w for i, w in enumerate(alb.W)},
         **{f"ab{i}": b for i, b in enumerate(alb.b)},
         **{f"rW{i}": w for i, w in enumerate(rad.W)},
         **{f"rb{i}": b for i, b in enumerate(rad.b)})
print("  saved checkpoints/ckpt_appearance_fresh.npz")
print("done. Run: py make_figures6.py --suffix _fresh")
