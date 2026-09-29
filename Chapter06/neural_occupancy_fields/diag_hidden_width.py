"""Diagnostic, not part of the regular pipeline: how long does the
differentiable kernel take to compile and train at a given hidden width?

Usage: python diag_hidden_width.py [hidden] [L] [steps]
  python diag_hidden_width.py 256 4 1500      # the book's own width

Why this exists: wide networks used to fail to compile (a Slang autodiff
blow-up, now fixed -- see README.md's "Two Slang autodiff pitfalls"). This
script is the quick check that a given width compiles AND trains: it runs
real training steps, not just one dispatch, so it also shows whether the
loss curve is sane. On an RTX 4090, `256 4 1500` takes about 75 s end to
end.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_occupancy as F

HIDDEN = int(sys.argv[1]) if len(sys.argv) > 1 else 256
L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

print(f"hidden={HIDDEN}, L={L}, steps={STEPS}, batch=4096, lr=1e-3", flush=True)
t0 = time.time()
m = F.CoordinateMLP(L=L, hidden=HIDDEN, seed=0)
h = F.train(m, steps=STEPS, batch=4096, lr=1e-3, seed=0, verbose=True)
elapsed = time.time() - t0
loss, acc = np.array(h["loss"]), np.array(h["acc"])
print(f"\ntotal (compile + {STEPS} steps): {elapsed:.0f}s")
print(f"loss: start {loss[0]:.3f}, min {loss.min():.3f} @ step {loss.argmin()}, "
      f"max {loss.max():.3f} @ step {loss.argmax()}, end {loss[-1]:.3f}")
print(f"acc:  start {acc[0]:.3f}, max {acc.max():.3f} @ step {acc.argmax()}, "
      f"end {acc[-1]:.3f}")
