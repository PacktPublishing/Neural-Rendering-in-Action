# Section 5's supervised SDF, trained via SlangPy

`train_sdf_slangpy.py` / `sdf_train.slang` are a GPU/SlangPy port of `sdf5.py`'s
`train_supervised()` (5.2) -- the same `NetworkParameters` pattern as Chapter 4
and Section 4, instead of `sdf5.py`'s hand-rolled NumPy backprop.

It does **not** touch rendering: sphere tracing, normals (`input_gradient`),
CSG, or marching cubes. It only trains, then saves a checkpoint in exactly the
format `make_figures5.py`'s `load_sdfnet()` expects (`L`, `W0..W2`, `b0..b2`),
so the existing, unmodified NumPy rendering pipeline picks it up automatically.

```
pip install slangpy
python train_sdf_slangpy.py [steps=1500] [L=6]   # writes ckpt_sdf_supervised.npz
python make_figures5.py                           # renders Figure 5.1/5.2 from it
```

Tested with slangpy 0.43.1 on the Vulkan backend, Python 3.11. Reaches IoU
~0.92 at 1500 steps (book's NumPy version, 256-wide: IoU 0.985) -- see
"Differences" below for why.

**The positional-encoding column order must match `SDFNet.encode()` exactly.**
Trained weights get saved into a checkpoint that `SDFNet.forward()` reads back
with its own encoding, so if the two don't agree column-for-column the loaded
network is not the one that was trained -- every weight ends up wired to the
wrong input feature. `SDFNet.encode()` (and `app6.py`'s `encode()`, same
pattern) order each octave as `[sin(x),sin(y),sin(z), cos(x),cos(y),cos(z)]`;
this is *not* the more natural per-axis interleaving `[sin(x),cos(x),
sin(y),cos(y),sin(z),cos(z)]`. Getting this wrong doesn't error -- it just
silently trains a self-consistent network that produces garbled shapes with
noticeable but scrambled torus-like structure once loaded into `SDFNet`
(the network only talks to itself during GPU training, so nothing catches
the mismatch until a NumPy consumer reads the checkpoint back).

## Differences from the book's NumPy version

Same three issues as Section 4's port (see `Section4/README.md` for the full
detail on each):

- **Hidden width is 64, not 256** -- differentiating the per-element weight
  loop at 256-wide didn't finish compiling in this Slang toolchain.
- The **positional-encoding order `L`** is a preprocessor `#define`
  (`SDF_L`), not a Slang generic -- a generic `Network<let L : int>` struct
  did not survive `bwd_diff`.
- **`predict_batch` is warmed up** with a dummy call before training starts,
  and training reuses persistent GPU buffers instead of allocating fresh
  ones every step -- both needed to avoid intermittent `createBuffer`
  failures after many dispatches.

## Not ported: `train_eikonal()` (5.5)

Learning an SDF from a point cloud alone needs a second-order gradient: the
eikonal loss depends on `||grad f||`, which `sdf5.py` computes via central
finite differences -- six shifted forward passes per training step, each
individually backpropagated and accumulated. That's a materially larger and
riskier Slang kernel than the regression case above (which just backprops
one externally supplied per-point gradient), so it wasn't attempted here.
`ckpt_sdf_eikonal.npz` is left as the original NumPy-trained checkpoint that
ships with the section; Figure 5.3 still renders from it unchanged.
