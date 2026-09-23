# Section 6's appearance fields, trained via SlangPy

`train_appearance_slangpy.py` / `appearance_train.slang` are a GPU/SlangPy
port of `app6.py`'s `train_albedo_field()` (6.2) and `train_radiance_field()`
(6.3) -- same pattern as Chapter 4 and Sections 4-5, instead of `app6.py`'s
hand-rolled NumPy backprop.

Not ported: `train_occupancy()` / `FeatureGrid` / `HybridField` (6.4, Figure
6.3's "race"). That figure is a deliberate wall-clock CPU speed comparison
between two NumPy implementations -- `make_figures6.py` says outright "its
numbers are machine-dependent by nature" -- so moving either side to the GPU
would just measure something else. `figure_6_3()` and `race_results.npz` are
untouched.

```
pip install slangpy
python train_appearance_slangpy.py [albedo_steps=800] [radiance_steps=1200]
# writes ckpt_appearance.npz
python make_figures6.py
# renders Figure 6.1/6.2 from it (Figure 6.3 unaffected, see above)
```

Tested with slangpy 0.43.1 on the Vulkan backend, Python 3.11. Reaches PSNR
~20 dB (albedo) / ~16-17 dB (radiance) at these step counts, hidden width 64
-- lower than the book's 128-wide network, see "Differences" below.

Same three toolchain workarounds as Section 4/5 (`Section4/README.md` has
the detail on each): hidden width capped at 64 (128, the book's own width
for this section, wasn't attempted -- see below), `L` as a preprocessor
define rather than a Slang generic, and every GPU buffer pre-allocated
before training starts rather than allocated on the fly, because a single
new allocation issued only after many prior dispatches intermittently failed
here (`createBuffer` / `SLANG_FAIL`) -- this section needed the fix applied
more aggressively than Section 4/5, since it trains *two* separate networks
(and so uses two separately-specialized Slang modules) back to back; even
warming up both kernels up front wasn't enough on its own, so every batch
buffer for *both* training loops is now allocated before either loop runs.

**The positional-encoding column order matches `app6.py`'s `encode()`**
exactly (per octave `[sin(x),sin(y),sin(z), cos(x),cos(y),cos(z)]`, not
interleaved per axis) -- see Section 5's README for why this specifically
matters whenever trained weights get saved into a checkpoint that a
different (NumPy) implementation reads back with its own encoding.

## Why hidden width 64, not the book's 128

Section 4/5 already established that this Slang toolchain can't
differentiate a 256-wide per-element weight loop in practical time. 128 is
smaller than that but bigger than the proven-fast 64, and untested here --
given the same superlinear scaling documented in `Section4/README.md`
(32-wide: ~1.5s, 64-wide: ~9-25s), 128 was a real risk of a many-minutes (or
worse) compile with no guarantee of finishing, so this port stayed with the
already-verified width rather than gambling the time to find out. If you
want to try 128, `HIDDEN` in `train_appearance_slangpy.py` and `appearance_
train.slang` is the only place it's set.
