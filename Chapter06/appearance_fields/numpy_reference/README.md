# NumPy reference implementation (Section 6, pre-SlangPy)

This folder holds the **original, unmodified, pure-NumPy** Section 6 code,
recovered from git history (`08f7eee^`, the commit right before the
SlangPy migration) — `app6.py`, `sdf5.py`, `make_figures6.py` — plus the
original shipped checkpoints, also recovered from history. It exists
purely as a **reference point** for the current Slang/SlangPy port one
level up (`../appearance.py`, `../neural_sdf.py`, `../*.slang`): a way to
directly run the pre-port code and compare, rather than reason about it
from memory or old screenshots.

Nothing here is meant to be maintained going forward, and nothing here
depends on Slang/SlangPy/Vulkan at all — it's plain `numpy` +
`matplotlib` + `scikit-image`.

## Files

- `app6.py`, `sdf5.py` — the original library code, byte-for-byte as
  recovered (only path handling in `make_figures6.py` was adjusted, not
  the math/training/rendering itself).
- `make_figures6.py` — the original figure-generation script, adapted
  only to use `checkpoints/`/`figures/` subfolders (this repo's current
  convention) instead of flat files.
- `checkpoints/ckpt_appearance.npz`, `checkpoints/ckpt_sdf_supervised.npz`
  — the ORIGINAL shipped checkpoints, recovered from the same commit.
  These are the exact weights that (approximately — see below) produced
  the book-era figures.
- `train_fresh.py` — trains both fields from scratch using the original
  code's own default hyperparameters, for comparison against the shipped
  checkpoints above.

## Run it

```
py make_figures6.py                    # renders from the shipped originals
py make_figures6.py --rerun-race 40    # also re-runs the 6.4 race (slow)
py train_fresh.py                      # trains fresh checkpoints (*_fresh.npz)
py make_figures6.py --suffix _fresh    # renders from the fresh ones instead
```

## What this settled

The question this folder was built to answer: is the "noisy, colored-static"
look in the current Slang/SlangPy port's `fig6_1_albedo.png` /
`fig6_2_viewdep.png` a port-introduced regression, or something inherent to
this recipe? Measured PSNR (full sphere-traced render vs. exact ground
truth):

| | albedo | radiance (cam 1 / cam 2) |
|---|---|---|
| Fresh NumPy retrain (this code, its own defaults) | 20.9 dB | 16.1 / 16.8 dB |
| Shipped original checkpoint (recovered from git) | 23.7 dB | 18.2 / 18.6 dB |
| Current Slang/SlangPy port | 22.0 dB | 19.5 / 19.4 dB |

**A fresh retrain of the untouched original NumPy code, with its own
default hyperparameters, reproduces the same noisy look** — no GPU, no
Slang, no port involved. The Slang/SlangPy port's numbers sit squarely in
the same range as the original code's own numbers (ahead on radiance,
slightly behind on albedo). This is not a port regression.

The shipped checkpoint scores a bit better than a bare fresh retrain,
meaning it was likely produced with something beyond the plain function
defaults (more steps, a favorable seed, or manual tuning) that isn't
captured by `train_albedo_field`/`train_radiance_field`'s defaults alone
— but even that checkpoint, rendered with the unmodified original code,
does not reproduce a dramatically cleaner, sharp-highlight result. If a
much cleaner-looking figure exists (e.g. in the printed book), it most
likely came from something not preserved in this repository's history —
extended training, a specific cherry-picked run, or manual print-figure
polish — not from any checkpoint+code that can actually be run today, in
either implementation.

**Practical conclusion**: this recipe (a plain coordinate MLP + Fourier
encoding, memorizing a deliberately high-frequency procedural checker
texture, in a few hundred to ~1200 steps) is genuinely close to its
practical quality ceiling here, independent of NumPy vs. Slang/GPU. See
`../README.md`'s note on this for the current port; don't chase this
further as if it were a Slang-specific bug.
