"""
Listings 3.1 & 3.2 -- Voxel Grids: meshes, voxelization, and continuous queries
======================================================================================
Core geometry utilities, implemented from scratch in NumPy:

  * make_torus_mesh        -- a procedural watertight test mesh (genus 1!)
  * voxelize_mesh          -- mesh -> occupancy grid via z-axis ray stabbing
  * trilinear_interpolate  -- continuous, differentiable queries of a grid
  * trilinear_gradient     -- the analytic spatial gradient of those queries
  * extract_mesh           -- occupancy grid -> mesh via marching cubes
  * sample_mesh_surface    -- mesh -> point cloud (area-weighted sampling)

Only dependencies: numpy, scikit-image (for marching cubes).
"""

import numpy as np
from skimage import measure


# ----------------------------------------------------------------------------
# 1. A procedural test mesh
# ----------------------------------------------------------------------------

def make_torus_mesh(R=0.7, r=0.3, n_major=96, n_minor=48):
    """Build a watertight triangulated torus.

    The torus is a deliberately chosen test shape: it has genus 1 (a hole),
    so any representation that survives a round trip mesh -> voxels -> mesh
    must preserve nontrivial topology.

    Returns (vertices (V,3) float64, faces (F,3) int64).
    """
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)   # around the hole
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)   # around the tube
    uu, vv = np.meshgrid(u, v, indexing="ij")

    x = (R + r * np.cos(vv)) * np.cos(uu)
    y = (R + r * np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Two triangles per parametric quad, with wraparound indexing.
    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            faces.append([a, b, c])
            faces.append([a, c, d])
    return vertices, np.asarray(faces, dtype=np.int64)


# ----------------------------------------------------------------------------
# 2. Voxelization by ray stabbing
# ----------------------------------------------------------------------------
# The idea: shoot a ray along +z through every (x, y) column of the grid and
# record where it crosses the surface. A point is INSIDE the shape iff the
# number of crossings below it is odd (Jordan curve theorem in 1D).
#
# Because every ray points along +z, the usual Moller-Trumbore ray/triangle
# test collapses to a 2D point-in-triangle test in the xy plane, followed by
# barycentric interpolation of the crossing depth z. Triangles seen edge-on
# (zero projected area) contribute nothing to the parity, which is exactly
# what we want for a watertight mesh.

def _column_crossings(px, py, tri_xy, tri_z, eps=1e-12):
    """Z-depths where the vertical ray through (px, py) crosses the mesh.

    tri_xy : (F, 3, 2) xy coordinates of triangle vertices
    tri_z  : (F, 3)    z  coordinates of triangle vertices
    """
    a, b, c = tri_xy[:, 0], tri_xy[:, 1], tri_xy[:, 2]
    # Signed area denominators (twice the projected triangle area).
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) \
        + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > eps                       # discard edge-on triangles

    w0 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0])
          + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / np.where(ok, den, 1.0)
    w1 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0])
          + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / np.where(ok, den, 1.0)
    w2 = 1.0 - w0 - w1

    inside = ok & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    z_hit = w0 * tri_z[:, 0] + w1 * tri_z[:, 1] + w2 * tri_z[:, 2]
    return np.sort(z_hit[inside])


def voxelize_mesh(vertices, faces, resolution=32, padding=0.05, jitter=1e-6,
                  rng=None):
    """Convert a watertight triangle mesh into a dense occupancy grid.

    Returns
    -------
    occ     : (R, R, R) float32 array of {0, 1}; occ[i, j, k] is the
              inside/outside state of the voxel CENTER at grid index (i, j, k)
    origin  : (3,) world coordinates of voxel center (0, 0, 0)
    spacing : (3,) world-space distance between adjacent voxel centers
    """
    rng = np.random.default_rng(0) if rng is None else rng
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    pad = (hi - lo).max() * padding
    lo, hi = lo - pad, hi + pad

    R = resolution
    spacing = (hi - lo) / R
    origin = lo + 0.5 * spacing                      # centers, not corners
    xs = origin[0] + np.arange(R) * spacing[0]
    ys = origin[1] + np.arange(R) * spacing[1]
    zs = origin[2] + np.arange(R) * spacing[2]

    tri = vertices[faces]                            # (F, 3, 3)
    tri_xy, tri_z = tri[..., :2], tri[..., 2]

    occ = np.zeros((R, R, R), dtype=np.float32)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # A microscopic jitter avoids rays passing exactly through
            # shared triangle edges (which would be double-counted).
            crossings = _column_crossings(
                x + jitter * rng.standard_normal(),
                y + jitter * rng.standard_normal(),
                tri_xy, tri_z)
            # Parity: voxel center is inside iff an odd number of surface
            # crossings lies below it.
            n_below = np.searchsorted(crossings, zs)
            occ[i, j, :] = (n_below % 2).astype(np.float32)
    return occ, origin, spacing


# ----------------------------------------------------------------------------
# 3. Continuous, differentiable grid queries
# ----------------------------------------------------------------------------
# A discrete grid only stores values at lattice points -- but trilinear
# interpolation turns it into a CONTINUOUS, PIECEWISE-DIFFERENTIABLE function
# f(x, y, z) defined everywhere. This is the conceptual bridge to neural
# fields: a neural field simply replaces "look up 8 corners and blend" with
# "evaluate an MLP".

def _grid_coords(grid, origin, spacing, queries):
    g = (np.atleast_2d(queries) - origin) / spacing          # (N, 3)
    g = np.clip(g, 0.0, np.array(grid.shape) - 1.0 - 1e-9)
    i0 = np.floor(g).astype(np.int64)
    i0 = np.minimum(i0, np.array(grid.shape) - 2)
    return g - i0, i0


def trilinear_interpolate(grid, origin, spacing, queries):
    """Evaluate the grid at arbitrary world-space points. queries: (N, 3)."""
    t, i0 = _grid_coords(grid, origin, spacing, queries)
    x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    c000 = grid[x,     y,     z    ]
    c100 = grid[x + 1, y,     z    ]
    c010 = grid[x,     y + 1, z    ]
    c110 = grid[x + 1, y + 1, z    ]
    c001 = grid[x,     y,     z + 1]
    c101 = grid[x + 1, y,     z + 1]
    c011 = grid[x,     y + 1, z + 1]
    c111 = grid[x + 1, y + 1, z + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


def trilinear_gradient(grid, origin, spacing, queries):
    """Analytic spatial gradient d f / d (x, y, z) of the interpolated field.

    The existence of this closed-form gradient is the whole point: it is what
    makes a grid 'neural-network compatible'. Gradients can flow through grid
    queries to upstream parameters (or to the grid values themselves).
    """
    t, i0 = _grid_coords(grid, origin, spacing, queries)
    x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    c = {(a, b, d): grid[x + a, y + b, z + d]
         for a in (0, 1) for b in (0, 1) for d in (0, 1)}

    def blend(w):  # w maps corner index -> weight product, for one partial
        return sum(c[k] * w[k] for k in c)

    wx = {(a, b, d): (1 if a else -1)
          * ((1 - ty) if b == 0 else ty) * ((1 - tz) if d == 0 else tz)
          for (a, b, d) in c}
    wy = {(a, b, d): ((1 - tx) if a == 0 else tx)
          * (1 if b else -1) * ((1 - tz) if d == 0 else tz)
          for (a, b, d) in c}
    wz = {(a, b, d): ((1 - tx) if a == 0 else tx)
          * ((1 - ty) if b == 0 else ty) * (1 if d else -1)
          for (a, b, d) in c}

    grad = np.stack([blend(wx), blend(wy), blend(wz)], axis=-1)
    return grad / spacing                                     # chain rule


# ----------------------------------------------------------------------------
# 4. Back to a mesh: marching cubes
# ----------------------------------------------------------------------------

def extract_mesh(occ, origin, spacing, level=0.5):
    """Occupancy grid -> triangle mesh at the given iso-level."""
    verts, faces, normals, _ = measure.marching_cubes(occ, level=level,
                                                      spacing=tuple(spacing))
    return verts + origin, faces, normals


# ----------------------------------------------------------------------------
# 5. Point clouds: sampling a mesh surface
# ----------------------------------------------------------------------------

def sample_mesh_surface(vertices, faces, n_points, rng=None):
    """Sample points uniformly on the surface of a triangle mesh.

    Two-step recipe (this is THE standard algorithm):
      1. pick a triangle with probability proportional to its area;
      2. pick a uniform point inside it with random barycentric coordinates,
         using the sqrt trick to avoid clustering near one vertex.
    Returns (points (n, 3), normals (n, 3)).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    tri = vertices[faces]                                     # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    probs = area / area.sum()

    idx = rng.choice(len(faces), size=n_points, p=probs)
    u, v = rng.random(n_points), rng.random(n_points)
    su = np.sqrt(u)
    b0, b1, b2 = 1.0 - su, su * (1.0 - v), su * v             # barycentric

    p = (b0[:, None] * tri[idx, 0]
         + b1[:, None] * tri[idx, 1]
         + b2[:, None] * tri[idx, 2])
    nrm = cross[idx] / np.linalg.norm(cross[idx], axis=1, keepdims=True)
    return p, nrm


# ----------------------------------------------------------------------------
# 6. Memory accounting helpers (for the resolution/memory tradeoff figure)
# ----------------------------------------------------------------------------

def grid_memory_bytes(resolution, bytes_per_voxel=1):
    return resolution ** 3 * bytes_per_voxel


def mesh_memory_bytes(vertices, faces):
    # float32 positions + int32 indices, the common GPU layout
    return vertices.shape[0] * 3 * 4 + faces.shape[0] * 3 * 4


def mlp_memory_bytes(d_in=39, hidden=256, n_hidden=4, d_out=1):
    """Parameter memory of the coordinate MLP used later in the chapter
    (positional encoding with L=6 -> 3 + 2*3*6 = 39 inputs)."""
    n = d_in * hidden + hidden
    for _ in range(n_hidden - 1):
        n += hidden * hidden + hidden
    n += hidden * d_out + d_out
    return n * 4
