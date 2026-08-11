"""
Listing 4.1 -- The shape to memorize, and the data to memorize it from
======================================================================
A neural occupancy field learns a single shape by being SHOWN that shape:
we feed it 3D points labelled inside (1) or outside (0), and it learns the
function that reproduces those labels everywhere.

So Section 4 needs two things:
  (a) a shape with detail across several spatial scales, so that the
      effect of positional encoding (Part 2) is dramatic and not subtle;
  (b) a way to generate millions of labelled (point, inside?) pairs.

For (a) we keep the chapter's running example -- the torus of Section 3 --
but carve high-frequency ripples into its tube, like the thread of a screw.
The smooth global structure (a ring with a hole) is easy for any network;
the ripples are HIGH FREQUENCY and, as Part 3 shows, a plain MLP cannot
represent them. That contrast is the whole point of the section.

For (b) the rippled torus is *star-shaped in each tube cross-section*, so we
have both an exact inside/outside test and an exact parametric sampler of
its surface -- no meshing required to make training data.

To use the real Stanford bunny instead, see `load_mesh_shape` at the bottom.
"""

import numpy as np

# Shape parameters. R: ring radius, r: tube radius, amp: ripple depth,
# f_theta: ripples around the hole, f_phi: ripples around the tube.
PARAMS = dict(R=0.65, r=0.28, amp=0.30, f_theta=8, f_phi=8)
BOUND = 1.10                       # the shape fits inside [-BOUND, BOUND]^3


def _effective_r(theta, phi, p=PARAMS):
    """The (angle-dependent) tube radius -- this is where the ripples live."""
    return p["r"] * (1.0 + p["amp"] * np.sin(p["f_theta"] * theta)
                                    * np.sin(p["f_phi"] * phi))


def detailed_torus_inside(pts, p=PARAMS):
    """Exact inside/outside test. pts: (N, 3) -> bool (N,).

    ring : signed distance from the z-axis to the ring centre line
    d    : distance from the tube's centre circle
    The point is inside when d is less than the (rippled) tube radius.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ring = np.sqrt(x ** 2 + y ** 2) - p["R"]
    d = np.sqrt(ring ** 2 + z ** 2)
    theta = np.arctan2(y, x)        # angle around the hole
    phi = np.arctan2(z, ring)       # angle around the tube
    return d < _effective_r(theta, phi, p)


def detailed_torus_surface(n, rng, p=PARAMS):
    """Sample n points exactly ON the surface (used for near-surface data)."""
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    er = _effective_r(theta, phi, p)
    rad = p["R"] + er * np.cos(phi)
    return np.stack([rad * np.cos(theta),
                     rad * np.sin(theta),
                     er * np.sin(phi)], axis=1)


def sample_training_data(n, rng, near_frac=0.5, jitter=0.05, inside_fn=None,
                         surface_fn=None):
    """Generate n labelled training points.

    Half are uniform in the bounding cube (so the network learns the empty
    space too); half are clustered near the surface, where the decision
    boundary lives and detail must be resolved. Labels come from the exact
    inside test, so even the jittered near-surface points are correctly
    labelled inside or outside.
    """
    inside_fn = detailed_torus_inside if inside_fn is None else inside_fn
    surface_fn = detailed_torus_surface if surface_fn is None else surface_fn

    n_near = int(n * near_frac)
    uniform = rng.uniform(-BOUND, BOUND, (n - n_near, 3))
    near = surface_fn(n_near, rng) + jitter * rng.standard_normal((n_near, 3))
    pts = np.vstack([uniform, near])
    labels = inside_fn(pts).astype(np.float64)
    return pts, labels


def occupancy_grid(resolution, inside_fn=None, bound=BOUND):
    """Evaluate the exact inside test on a dense lattice -> occupancy grid.

    Used only to mesh the GROUND TRUTH for figures (via marching cubes) and
    to score the trained networks. The networks themselves never see this
    grid -- they train on sampled points.
    """
    inside_fn = detailed_torus_inside if inside_fn is None else inside_fn
    xs = np.linspace(-bound, bound, resolution)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    occ = inside_fn(pts).astype(np.float32).reshape([resolution] * 3)
    origin = np.array([-bound] * 3)
    spacing = np.array([xs[1] - xs[0]] * 3)
    return occ, origin, spacing


# ----------------------------------------------------------------------------
# Using a real mesh instead of the torus (the Stanford bunny)
# ----------------------------------------------------------------------------
# Set BUNNY_PLY to a mesh path (or set the environment variable of the same
# name) and BOTH training and evaluation switch to that mesh -- no other edits
# needed. Leave it None to use the procedural rippled torus.
import os

# Set to a mesh path to train/evaluate on that mesh; set to None to force the
# procedural rippled torus (the chapter's default figures). If the path is set
# but the file is missing, we fall back to the torus with a warning. It ships
# as "bunny.ply" so that dropping the Stanford bunny into this folder Just Works.
BUNNY_PLY = None


def load_mesh_shape(path, grid_res=128, pad=0.12):
    """Return (inside_fn, surface_fn) for an arbitrary mesh (e.g. the bunny).

    The inside test is the expensive part, so we pay it ONCE: normalize the
    mesh into the same box the torus used, sample a dense occupancy grid with
    trimesh's point-in-mesh test (batched, with progress), cache it to disk,
    and thereafter answer every per-step query by fast trilinear lookup of that
    grid (Listing 3.2). Surface points come from area-weighted triangle
    sampling (Listing 3.1). Only the one-time grid build needs trimesh.
    """
    import trimesh                                   # reader-side dependency
    from geometry import trilinear_interpolate, sample_mesh_surface

    mesh = trimesh.load(path, force="mesh")
    mesh.fill_holes()                                # the bunny's base is open
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices /= np.abs(mesh.vertices).max() / (1.0 - pad)
    V = np.asarray(mesh.vertices)
    Ftri = np.asarray(mesh.faces)

    cache = f"occ_{os.path.basename(path)}_{grid_res}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        occ, origin, spacing = d["occ"], d["origin"], d["spacing"]
    else:
        xs = np.linspace(-BOUND, BOUND, grid_res)
        XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
        P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
        print(f"  [one-time] labelling {P.shape[0]:,} grid points with "
              f"trimesh.contains -> {cache}")
        occ = np.empty(len(P), dtype=np.float32)
        CH = 100_000
        for i in range(0, len(P), CH):
            occ[i:i + CH] = mesh.contains(P[i:i + CH])
            print(f"    {min(i + CH, len(P)):>9,}/{len(P):,}", end="\r")
        print()
        occ = occ.reshape([grid_res] * 3)
        origin = np.array([-BOUND] * 3)
        spacing = np.array([xs[1] - xs[0]] * 3)
        np.savez(cache, occ=occ, origin=origin, spacing=spacing)

    def inside_fn(pts):
        return trilinear_interpolate(occ, origin, spacing, pts) > 0.5

    def surface_fn(n, rng):
        pts, _ = sample_mesh_surface(V, Ftri, n, rng=rng)
        return pts

    return inside_fn, surface_fn


def get_shape():
    """The single switch every script calls. Returns (inside_fn, surface_fn,
    name). Uses BUNNY_PLY (constant or environment variable) if that file
    exists; otherwise the rippled torus."""
    path = BUNNY_PLY or os.environ.get("BUNNY_PLY")
    if path and os.path.exists(path):
        inside_fn, surface_fn = load_mesh_shape(path)
        return inside_fn, surface_fn, os.path.basename(path)
    if path:
        print(f"  warning: mesh '{path}' not found -> using rippled torus")
    return detailed_torus_inside, detailed_torus_surface, "rippled_torus"
