"""Resumable training for one of Section 6's two appearance fields.
Usage: python3 train_one.py <albedo|radiance> [chunk_steps]
Loads ckpt_appearance_<which>.npz if present and trains chunk_steps more,
accumulating history in hist_appearance_<which>.npz. Call repeatedly to
reach the target step count -- same pattern as
neural_signed_distance/train_one.py. Needs neural_signed_distance's
ckpt_sdf_supervised.npz (or this folder's own copy of it) for the frozen
geometry -- see README.md."""
import os
import sys

import numpy as np

import neural_sdf as G
import appearance as A

WHICH = sys.argv[1]
assert WHICH in ("albedo", "radiance"), "usage: train_one.py <albedo|radiance> [chunk_steps]"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 4000   # see run_section.py
L = 6
CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)
geo_ckpt = os.path.join(CHECKPOINTS, "ckpt_sdf_supervised.npz")
ckpt = os.path.join(CHECKPOINTS, f"ckpt_appearance_{WHICH}.npz")
histf = os.path.join(CHECKPOINTS, f"hist_appearance_{WHICH}.npz")

if not os.path.exists(geo_ckpt):
    sys.exit(f"\nMissing {geo_ckpt}.\nCopy it from neural_signed_distance/checkpoints/ "
             f"(train it there first with train_one.py supervised) -- Section 6's "
             f"appearance fields render through that frozen geometry.\n")
geo = G.Network.load(geo_ckpt)

if os.path.exists(ckpt):
    m = A.AppearanceNetwork.load(ckpt, sdf_L=geo.L)
    if os.path.exists(histf):
        prev = np.load(histf)
        prev_secs, done = float(prev["seconds"]), int(prev["done"])
    else:
        prev_secs, done = 0.0, 0
    seed = 1000 + done
else:
    m = A.AppearanceNetwork(has_view=(WHICH == "radiance"), L=L, sdf_L=geo.L, seed=0)
    prev_secs, done, seed = 0.0, 0, 0

train_fn = A.train_albedo if WHICH == "albedo" else A.train_radiance
h = train_fn(m, steps=CHUNK, seed=seed, verbose=False)

m.save(ckpt)
total = done + CHUNK
np.savez(histf, seconds=prev_secs + h["seconds"], done=total)
print(f"{WHICH}: +{CHUNK} steps -> {total} total, {m.memory_bytes()/1024:.0f} KiB, "
      f"chunk {h['seconds']:.0f}s, cumulative {prev_secs + h['seconds']:.0f}s")
