"""Resumable training for one of Section 5's two SDF recipes.
Usage: python3 train_one.py <supervised|eikonal> [chunk_steps]
Loads ckpt_sdf_<which>.npz if present and trains chunk_steps more,
accumulating history in hist_sdf_<which>.npz. Call repeatedly to reach the
target step count -- same pattern as neural_occupancy_fields/train_one.py."""
import os
import sys

import numpy as np

import neural_sdf as F

WHICH = sys.argv[1]
assert WHICH in ("supervised", "eikonal"), "usage: train_one.py <supervised|eikonal> [chunk_steps]"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else (1500 if WHICH == "supervised" else 900)
L = 6
# Eikonal needs the book's width to recover the torus's hole; supervised
# stays at the fast default -- see run_section.py's EIKONAL_HIDDEN note.
EIKONAL_HIDDEN = 256
CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)
ckpt = os.path.join(CHECKPOINTS, f"ckpt_sdf_{WHICH}.npz")
histf = os.path.join(CHECKPOINTS, f"hist_sdf_{WHICH}.npz")

if os.path.exists(ckpt):
    m = F.Network.load(ckpt)
    if os.path.exists(histf):
        prev = np.load(histf)
        prev_secs, done = float(prev["seconds"]), int(prev["done"])
    else:
        # A checkpoint from before this driver tracked history (or written
        # by the pre-consolidation scripts): resume training on it, just
        # without a step count to add to.
        prev_secs, done = 0.0, 0
    seed = 1000 + done
else:
    hidden = EIKONAL_HIDDEN if WHICH == "eikonal" else F.HIDDEN
    m = F.Network(L=L, hidden=hidden, seed=0)
    prev_secs, done, seed = 0.0, 0, 0

if WHICH == "supervised":
    h = F.train_supervised(m, steps=CHUNK, seed=seed, verbose=False)
else:
    module = F._module(L, m.hidden)
    cloud_t = F.make_surface_cloud(module, 30_000, seed=0)
    h = F.train_eikonal(m, cloud_t, steps=CHUNK, warmup=(0 if done else 200),
                        seed=seed, verbose=False)

m.save(ckpt)
total = done + CHUNK
np.savez(histf, seconds=prev_secs + h["seconds"], done=total)
print(f"{WHICH}: +{CHUNK} steps -> {total} total, {m.memory_bytes()/1024:.0f} KiB, "
      f"chunk {h['seconds']:.0f}s, cumulative {prev_secs + h['seconds']:.0f}s")
