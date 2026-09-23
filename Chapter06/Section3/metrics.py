"""
Listing 3.5 -- Metrics: how do we compare representations of the same shape?
============================================================================
Two complementary measures, both standard in the 3D learning literature:

  * Volumetric IoU -- agreement of the inside/outside decision, estimated by
    Monte Carlo: sample many random points, ask both representations
    "inside?", and report |intersection| / |union|. Sensitive to volume
    errors; blind to small surface wobble.

  * Chamfer distance -- average nearest-neighbour distance between point
    samples of the two SURFACES, symmetrized. Sensitive to surface wobble;
    blind to thin volumetric errors. (We report the L2 mean, scaled by 1e3
    for readability.)

Reporting both keeps us honest: a representation can win one and lose the
other.
"""

import numpy as np
from scipy.spatial import cKDTree


def volumetric_iou(inside_a, inside_b, n=200_000, bound=1.1, seed=0):
    """inside_a / inside_b: callables (N, 3) -> bool array."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-bound, bound, (n, 3))
    A, B = inside_a(pts), inside_b(pts)
    union = np.logical_or(A, B).sum()
    if union == 0:
        return 1.0
    return np.logical_and(A, B).sum() / union


def chamfer_distance(points_a, points_b):
    """Symmetric mean nearest-neighbour distance between two point sets."""
    d_ab = cKDTree(points_b).query(points_a, k=1)[0].mean()
    d_ba = cKDTree(points_a).query(points_b, k=1)[0].mean()
    return 0.5 * (d_ab + d_ba)
