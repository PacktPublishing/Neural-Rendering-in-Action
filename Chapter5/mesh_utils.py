"""
Procedural mesh generation utilities
Generates icosphere meshes
"""

import numpy as np
from typing import Tuple


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms


def make_icosphere(subdivisions: int = 1, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an icosphere mesh by subdividing an icosahedron.

    Args:
        subdivisions: Number of subdivision iterations (0=icosahedron with 20 faces,
                      1=80 faces, 2=320 faces).
        radius: Radius of the sphere.

    Returns:
        vertices: (N, 3) float32 array of vertex positions.
        faces: (F, 3) int32 array of triangle indices.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []

        def get_midpoint(i0, i1):
            key = (min(i0, i1), max(i0, i1))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (verts[i0] + verts[i1]) * 0.5
            idx = len(verts_list)
            verts_list.append(mid)
            edge_midpoints[key] = idx
            return idx

        verts_list = list(verts)
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])

        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int32)

    verts = _normalize(verts) * radius
    return verts.astype(np.float32), faces.astype(np.int32)


def faces_to_flat_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Expand indexed mesh into flat vertex array: each triangle gets its own 3 vertices.
    Returns (numTris*3, 3) float32 array suitable for direct GPU upload.
    """
    flat = vertices[faces.flatten()]
    return flat.astype(np.float32)

