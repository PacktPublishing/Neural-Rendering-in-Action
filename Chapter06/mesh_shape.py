"""
Optional: Stanford bunny mesh support, shared by Sections 4 and 5.
======================================================================
Both neural_occupancy_fields and neural_signed_distance train against
shapes whose inside test / surface parametrization / distance are
closed-form (the rippled and smooth tori, common_shapes.slang). An
arbitrary mesh (the bunny) has none of those, so this module pays the
expensive part -- point-in-mesh testing and nearest-surface distance --
ONCE on the CPU, caches it, and hands GPU-ready arrays back to each
section's own train_bunny*.py. At training time, neither section's
mesh-specific kernels (calculate_grads_mesh in neural_occupancy.slang,
calculate_supervised_grads_mesh in neural_signed_distance/neural_sdf.slang)
touch the mesh itself: they only read the cached grid(s) (trilinearly
interpolated by grid_trilerp, common_shapes.slang) and the surface cloud.

Lives at the chapter root (not inside one section's folder) because both
sections need it -- same reasoning as common_shapes.slang/
slangpy_common.py. Each section's own *_slangpy driver already inserts
the chapter root onto sys.path (see e.g. neural_occupancy.py's own
`sys.path.insert(0, ...parent.parent)`), so `import mesh_shape` resolves
here from either section without further setup.

trimesh (plus its `embreex` backend) is the one dependency this optional
path adds on top of the chapter's otherwise from-scratch NumPy/Slang
code -- see neural_occupancy_fields/README.md. `mesh.contains()` needs
`embreex` (or `pyembree`) installed: trimesh's fallback pure-Python
ray/triangle intersector tests every query point against every triangle
at once and runs out of memory well before a 128^3 lattice finishes
(measured: >5 GiB for a single 100k-point chunk against this mesh's
69,451 faces). With `embreex`, the same lattice labels in well under a
second.

`bun_zipper.ply` is NOT watertight even after `fill_holes()` closes its
open base -- the raw scan has a few smaller residual gaps too. An earlier
version of this module used a from-scratch ray-stabbing voxelizer instead
of trimesh, to stay dependency-light; it produced visibly wrong ground
truth (thin spurious comb-like fins where the mesh isn't watertight,
since even/odd ray-crossing parity breaks down at real gaps). Caught by
actually looking at the rendered ground-truth panel, not by IoU, which
doesn't distinguish a bad ground truth from a bad reconstruction.
`mesh.contains()` isn't immune to non-watertightness in principle, but
does not show the artifact on this mesh in practice.
"""

import os

import numpy as np

BOUND = 1.10          # matches common_shapes.slang / neural_occupancy.BOUND


def sample_mesh_surface(vertices, faces, n_points, rng=None):
    """Sample points uniformly on a triangle mesh's surface: pick a
    triangle with probability proportional to its area, then a uniform
    point inside it via the sqrt barycentric trick. Ported from
    neural_occupancy_fields/geometry.py (Listing 3.1) -- inlined here
    (rather than imported) so this module has no cross-section Python
    dependency, only trimesh + numpy. Returns (points (n,3), normals (n,3))."""
    rng = np.random.default_rng(0) if rng is None else rng
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    probs = area / area.sum()

    idx = rng.choice(len(faces), size=n_points, p=probs)
    u, v = rng.random(n_points), rng.random(n_points)
    su = np.sqrt(u)
    b0, b1, b2 = 1.0 - su, su * (1.0 - v), su * v

    p = (b0[:, None] * tri[idx, 0] + b1[:, None] * tri[idx, 1]
        + b2[:, None] * tri[idx, 2])
    nrm = cross[idx] / np.linalg.norm(cross[idx], axis=1, keepdims=True)
    return p, nrm


def _load_normalized_mesh(ply_path, pad=0.12):
    """Load the mesh, close its open base, and fit it into [-BOUND, BOUND]^3
    with `pad` fraction of headroom -- same normalization the original
    (pre-consolidation) load_mesh_shape used."""
    import trimesh

    mesh = trimesh.load(ply_path, force="mesh")
    mesh.fill_holes()                                # the bunny's base is open
    V = np.asarray(mesh.vertices, dtype=np.float64)
    V = V - V.mean(axis=0)
    V = V / (np.abs(V).max() / (1.0 - pad))
    mesh.vertices = V
    return mesh


def _lattice(grid_res):
    """Corner-aligned linspace(-BOUND, BOUND, R), matching
    neural_occupancy.field_to_grid/occupancy_grid_gpu's convention, so
    grids built here compose the same way with extract_mesh/
    trilinear_interpolate elsewhere in the chapter."""
    xs = np.linspace(-BOUND, BOUND, grid_res)
    origin = np.array([xs[0]] * 3, dtype=np.float32)
    spacing = np.array([xs[1] - xs[0]] * 3, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    return P, origin, spacing


def load_and_cache(ply_path, grid_res=128, n_cloud=200_000, cache_dir="checkpoints",
                   seed=0):
    """Return (occ, origin, spacing, cloud, V, F, mesh):

      occ, origin, spacing  -- a grid_res^3 occupancy grid covering
                                [-BOUND, BOUND]^3, float32, ready to upload
                                as a flat GPU Tensor for grid_trilerp.
      cloud                 -- (n_cloud, 3) float32 points sampled on the
                                mesh surface (area-weighted), ready to
                                upload as a training kernel's cloud Tensor
                                (both sections' calculate_*_grads_mesh
                                kernels, and neural_sdf.train_eikonal
                                as-is for the points-alone recipe).
      V, F, mesh             -- the normalized vertices/faces (as used to
                                build occ) and the trimesh object itself.
                                Score against `occ`
                                (geometry.trilinear_interpolate), not
                                mesh.contains() on scattered points -- see
                                the module docstring for why.

    The grid is the expensive step (one mesh.contains() call per lattice
    point), so it's cached to
    <cache_dir>/bunny_grid_R{grid_res}.npz across runs. The cloud is cheap
    (pure NumPy triangle sampling) and is redrawn every call.
    """
    mesh = _load_normalized_mesh(ply_path)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"bunny_grid_R{grid_res}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        occ, origin, spacing = d["occ"], d["origin"], d["spacing"]
    else:
        P, origin, spacing = _lattice(grid_res)
        print(f"  [one-time] labelling {P.shape[0]:,} grid points with "
              f"trimesh.contains (needs embreex/pyembree) -> {cache}")
        occ = np.empty(len(P), dtype=np.float32)
        CH = 200_000
        for i in range(0, len(P), CH):
            occ[i:i + CH] = mesh.contains(P[i:i + CH])
            print(f"    {min(i + CH, len(P)):>9,}/{len(P):,}", end="\r")
        print()
        occ = occ.reshape([grid_res] * 3)
        np.savez(cache, occ=occ, origin=origin, spacing=spacing)

    rng = np.random.default_rng(seed)
    cloud, _ = sample_mesh_surface(V, F, n_cloud, rng=rng)

    return (occ.astype(np.float32), np.asarray(origin, np.float32),
           np.asarray(spacing, np.float32), cloud.astype(np.float32), V, F, mesh)


def build_or_load_sdf_grid(occ, origin, spacing, V, F, grid_res, n_surface=300_000,
                           cache_dir="checkpoints", seed=1):
    """A signed-distance grid for supervised SDF training (Section 5.2's
    recipe, which needs an exact-ish label at every sampled point -- unlike
    Section 4's occupancy, there's no torus_sdf() closed form for a mesh).

    Not an exact closest-point-on-mesh distance (that needs a proper
    triangle-distance query); instead: UNSIGNED distance to the nearest
    point in a dense surface sample (scipy.spatial.cKDTree -- the same
    tool neural_occupancy_fields/metrics.chamfer_distance already uses in
    this chapter), SIGNED by `occ` (this module's own occupancy ground
    truth, already verified visually to be clean -- see load_and_cache's
    docstring). Good enough to supervise a network: at grid_res=128 /
    n_surface=300k the nearest-sample-point distance and the true
    nearest-surface distance agree to well under one voxel almost
    everywhere.

    occ/origin/spacing must be load_and_cache's own return values (same
    grid_res, same [-BOUND, BOUND]^3 lattice) so the sign lines up with
    the distance lattice point for point.
    """
    from scipy.spatial import cKDTree

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"bunny_sdf_R{grid_res}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["sdf"], d["origin"], d["spacing"]

    P, o2, s2 = _lattice(grid_res)
    assert np.allclose(o2, origin) and np.allclose(s2, spacing), (
        "build_or_load_sdf_grid's grid_res must match the occ grid it signs with")

    rng = np.random.default_rng(seed)
    surf, _ = sample_mesh_surface(V, F, n_surface, rng=rng)
    print(f"  [one-time] signed distance for {P.shape[0]:,} grid points "
          f"(cKDTree vs {n_surface:,}-pt surface sample) -> {cache}")
    dist, _ = cKDTree(surf).query(P, k=1)
    sign = np.where(occ.ravel() > 0.5, -1.0, 1.0)         # inside: negative
    sdf = (dist * sign).astype(np.float32).reshape([grid_res] * 3)
    np.savez(cache, sdf=sdf, origin=origin, spacing=spacing)
    return sdf, origin, spacing
