// SPDX-License-Identifier: Apache-2.0
//
// The two tori used across Sections 4-6, ported once from
// neural_occupancy_fields/shapes.py rather than re-derived per section.
//
// Section 4's occupancy target, Section 5's eikonal target, and Section 6's
// race target are all the SAME shape -- the "knurled" torus, a smooth ring
// whose tube radius is modulated by a product of two sinusoids -- under two
// different names (`shapes.py`'s `detailed_torus_inside`/`_effective_r` and
// `sdf5.py`'s `knurl_inside`/`knurl_surface`). This module keeps
// `shapes.py`'s names and parameters (R=0.65, r=0.28, amp=0.30, f_theta=8,
// f_phi=8) as the one canonical implementation. It also carries the SMOOTH
// torus (Section 5's exact-SDF regression target and Section 6's appearance
// surface), which shares the same R/r.
//
// Genuinely shared, not copied: every section's own .slang file reaches
// this by `__exported import common_shapes;` -- Slang's compiler include
// path already covers the chapter root (see slangpy_common.create_device),
// unlike Python's per-folder-only imports.

import slangpy;

public static const float TORUS_R = 0.65f;
public static const float TORUS_r = 0.28f;
public static const float KNURL_AMP = 0.30f;
public static const float KNURL_F_THETA = 8.0f;
public static const float KNURL_F_PHI = 8.0f;
public static const float BOUND = 1.10f;          // the shape fits inside [-BOUND, BOUND]^3

// -------------------------------------------------------------------------
// The smooth torus [Section 5.1 / 6.1] -- has a closed-form SDF, unlike the
// knurled one below.
// -------------------------------------------------------------------------

public float torus_sdf(float3 p)
{
    float ring = length(p.xy) - TORUS_R;
    return length(float2(ring, p.z)) - TORUS_r;
}

// Surface coordinates: u around the hole, v around the tube.
public float2 torus_uv(float3 p)
{
    float ring = length(p.xy) - TORUS_R;
    return float2(atan2(p.y, p.x), atan2(p.z, ring));
}

public float3 torus_normal(float3 p)
{
    float rr = length(p.xy) + 1e-12f;
    float ring = rr - TORUS_R;
    float3 g = float3(ring * p.x / rr, ring * p.y / rr, p.z);
    return normalize(g);
}

// Exact parametric surface point at (u, v) in [0, 2*pi)^2.
public float3 torus_surface(float2 uv)
{
    float rad = TORUS_R + TORUS_r * cos(uv.y);
    return float3(rad * cos(uv.x), rad * sin(uv.x), TORUS_r * sin(uv.y));
}

// -------------------------------------------------------------------------
// The knurled torus [Section 4 occupancy / Section 5 eikonal / Section 6
// race] -- star-shaped in each tube cross-section, so an exact
// inside/outside test and an exact surface sampler exist even though the
// exact SIGNED DISTANCE does not (that gap is Section 5.5's whole point).
// -------------------------------------------------------------------------

// The (angle-dependent) tube radius -- this is where the knurling lives.
public float effective_tube_radius(float theta, float phi)
{
    return TORUS_r * (1.0f + KNURL_AMP * sin(KNURL_F_THETA * theta)
                                        * sin(KNURL_F_PHI * phi));
}

public bool detailed_torus_inside(float3 p)
{
    float ring = length(p.xy) - TORUS_R;
    float d = length(float2(ring, p.z));
    float theta = atan2(p.y, p.x);
    float phi = atan2(p.z, ring);
    return d < effective_tube_radius(theta, phi);
}

// Lattice evaluation of the knurled-torus inside test, for marching-cubes
// ground-truth comparisons (Section 4 Figure 4.4, Section 5 Figure 5.3):
// one thread per voxel, laid out row-major (ix*res+iy)*res+iz.
public float detailed_torus_grid_value(int idx, float3 origin, float3 spacing, int res)
{
    int iz = idx % res;
    int iy = (idx / res) % res;
    int ix = idx / (res * res);
    float3 p = origin + spacing * float3(ix, iy, iz);
    return detailed_torus_inside(p) ? 1.0f : 0.0f;
}

// Same inside test, but at an arbitrary batch of points rather than a
// lattice -- for scoring (IoU) against points already used for a
// prediction, e.g. Section 6's hybrid-vs-pure race.
public float detailed_torus_inside_batch(int idx, Tensor<float3, 1> pts)
{
    return detailed_torus_inside(pts.load(idx)) ? 1.0f : 0.0f;
}

// Exact parametric surface point at (theta, phi) in [0, 2*pi)^2.
public float3 detailed_torus_surface(float2 tp)
{
    float er = effective_tube_radius(tp.x, tp.y);
    float rad = TORUS_R + er * cos(tp.y);
    return float3(rad * cos(tp.x), rad * sin(tp.x), er * sin(tp.y));
}

// -------------------------------------------------------------------------
// Cached-grid lookup [Section 4 optional: Stanford bunny] -- trilinear
// interpolation of a precomputed grid, for shapes with no closed-form
// inside test (an arbitrary mesh). The grid itself is built once on the
// CPU (mesh_shape.py, via trimesh) and uploaded as a flat buffer; this is
// the Slang port of geometry.py's trilinear_interpolate. Row-major
// indexed exactly like detailed_torus_grid_value above:
// idx = (ix*res + iy)*res + iz.
// -------------------------------------------------------------------------
public float grid_trilerp(Tensor<float, 1> grid, float3 origin, float3 spacing, int res, float3 p)
{
    float3 g = (p - origin) / spacing;
    g = clamp(g, float3(0.0f), float3(float(res) - 1.0f - 1e-4f));
    int3 i0 = int3(floor(g));
    i0 = min(i0, int3(res - 2));
    float3 t = g - float3(i0);

    float c000 = grid.load((i0.x       * res + i0.y    ) * res + i0.z    );
    float c100 = grid.load(((i0.x + 1) * res + i0.y    ) * res + i0.z    );
    float c010 = grid.load((i0.x       * res + i0.y + 1) * res + i0.z    );
    float c110 = grid.load(((i0.x + 1) * res + i0.y + 1) * res + i0.z    );
    float c001 = grid.load((i0.x       * res + i0.y    ) * res + i0.z + 1);
    float c101 = grid.load(((i0.x + 1) * res + i0.y    ) * res + i0.z + 1);
    float c011 = grid.load((i0.x       * res + i0.y + 1) * res + i0.z + 1);
    float c111 = grid.load(((i0.x + 1) * res + i0.y + 1) * res + i0.z + 1);

    float c00 = lerp(c000, c100, t.x);
    float c10 = lerp(c010, c110, t.x);
    float c01 = lerp(c001, c101, t.x);
    float c11 = lerp(c011, c111, t.x);
    float c0 = lerp(c00, c10, t.y);
    float c1 = lerp(c01, c11, t.y);
    return lerp(c0, c1, t.z);
}

// -------------------------------------------------------------------------
// CSG on distance fields [Section 5.6] -- arithmetic, not geometry.
// -------------------------------------------------------------------------

// Polynomial smooth minimum (Quilez): welds two distance fields with a
// fillet of radius ~k where they meet.
public float smooth_union(float d1, float d2, float k)
{
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return lerp(d2, d1, h) - k * h * (1.0f - h);
}

public float sphere_sdf(float3 p, float3 c, float r)
{
    return length(p - c) - r;
}
