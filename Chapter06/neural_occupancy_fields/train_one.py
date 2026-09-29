"""Resumable training. Usage: python3 train_one.py L [chunk_steps]
Loads ckpt_L{L}.npz if present and trains chunk_steps more, accumulating
history in hist_L{L}.npz. Call repeatedly to reach the target step count."""
import os, sys
import numpy as np
import neural_occupancy as F

L = int(sys.argv[1])
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
HIDDEN = F.HIDDEN
BATCH = 4096
CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)
ckpt = os.path.join(CHECKPOINTS, f"ckpt_L{L}.npz")
histf = os.path.join(CHECKPOINTS, f"hist_L{L}.npz")

if os.path.exists(ckpt):
    m = F.CoordinateMLP.load(ckpt)
    prev = np.load(histf)
    loss_hist, acc_hist = list(prev["loss"]), list(prev["acc"])
    prev_secs, done = float(prev["seconds"]), len(prev["loss"])
    seed = 1000 + done                       # vary stream across resumes
else:
    m = F.CoordinateMLP(L=L, hidden=HIDDEN, seed=0)
    loss_hist, acc_hist, prev_secs, done, seed = [], [], 0.0, 0, 0

h = F.train(m, steps=CHUNK, batch=BATCH, seed=seed, verbose=False)
m.save(ckpt)
loss_hist += h["loss"]; acc_hist += h["acc"]
np.savez(histf, loss=np.array(loss_hist), acc=np.array(acc_hist),
         seconds=prev_secs + h["seconds"])
total = done + CHUNK
print(f"L={L}: +{CHUNK} steps -> {total} total, {m.memory_bytes()/1024:.0f} KiB, "
      f"chunk {h['seconds']:.0f}s, cumulative {prev_secs + h['seconds']:.0f}s, "
      f"last batch acc {h['acc'][-1]:.3f}")
