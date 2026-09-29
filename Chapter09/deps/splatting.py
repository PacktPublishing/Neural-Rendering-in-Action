"""
Listing 9.1 -- The whole of Part 1: video in, Gaussian splats out
================================================================
Written by Claude (Anthropic).

One module, thirteen sections, in the order the chapter explains them:

    1-2   the vocabulary: one lens, one posed image set, and the keyframes
    3-9   structure from motion: features, matches, tracks, triangulation,
          bundle adjustment, and the incremental loop that drives them
    10-12 the splat optimizer: what a Gaussian is, how it is drawn, and how
          the drawing becomes a gradient
    13    the two files Part 2 reads

Cameras are OpenCV throughout: x right, y down, z forward, and the rows of R are the camera axes in world space.

Nothing here renders for a human -- that is Part 2's job, in C++. The rasterizer in section 11 exists because training needs gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import json
import time

from scipy.spatial import cKDTree
from torch.utils.checkpoint import checkpoint
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. The camera model and the reconstruction container
# ------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Intrinsics:
    """One lens, shared by every frame of the video.

    `focal` is in pixels and pixels are assumed square (fx = fy). `k1` and `k2` are the two-term radial polynomial: a point at normalized
    radius r is displaced to r (1 + k1 r^2 + k2 r^4), negative k1 being barrel.
    """
    focal: float
    principal: np.ndarray       # (2,) held fixed at the image center: letting
                                # it float mostly trades against translation
    k1: float = 0.0
    k2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal, 0.0, self.principal[0]], [0.0, self.focal, self.principal[1]], [0.0, 0.0, 1.0]])

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV's (k1, k2, p1, p2) vector -- no tangential terms."""
        return np.array([self.k1, self.k2, 0.0, 0.0])


@dataclass
class Reconstruction:
    """A posed image set plus a sparse point cloud."""

    image_names: list[str]
    width: int
    height: int
    focal: float                # shared, in pixels (fx = fy: square pixels)
    principal: np.ndarray       # (2,) principal point in pixels
    R: np.ndarray               # (N, 3, 3) world -> camera rotations
    t: np.ndarray               # (N, 3)    world -> camera translations
    points: np.ndarray          # (M, 3)    sparse cloud
    colors: np.ndarray          # (M, 3)    linear RGB in [0, 1]
    k1: float = 0.0             # shared radial distortion, r -> r(1 + k1 r^2
    k2: float = 0.0             #                                 + k2 r^4)

    @property
    def n_images(self) -> int:
        return len(self.image_names)

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics(focal=self.focal, principal=self.principal, k1=self.k1, k2=self.k2)

    @property
    def fov_x_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(0.5 * self.width / self.focal)))

    @property
    def fov_y_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(0.5 * self.height / self.focal)))

    @property
    def camera_centers(self) -> np.ndarray:
        """(N, 3) camera positions in world space: C = -R^T t."""
        return -np.einsum("nij,ni->nj", self.R, self.t)

    def prune_outliers(self, camera_radius: float = 3.0, point_radius: float = 5.0, max_camera_fraction: float = 0.1) -> tuple[int, int]:
        """Drop cameras and points that cannot be where they claim to be; returns how many of each went.

        A single bad pose is expensive out of all proportion to its number. Every scene-relative setting below -- the learning rates, the
        size threshold that decides which Gaussians are split, the extent printed by `summary` -- is derived from how far the cameras
        spread, so one camera parked twenty radii outside the capture inflates all of them at once and the whole scene trains as a blur.
        Points behave the same way: a track triangulated from a near-degenerate pair can land thousands of units out.

        Both rules are percentiles of the thing being measured, which makes them scale free, so this runs before `normalize`. A camera
        farther than `camera_radius` times the 95th percentile camera distance was never part of the orbit, and a point beyond
        `point_radius` times the 99th percentile radius of the cloud is not in the room. Using percentiles rather than the deviation from
        the median matters: the last camera of a trajectory legitimately sits several deviations out, and a rule that trims it would keep
        nibbling at the capture every time it ran. If more than `max_camera_fraction` of the cameras look like strays then they are not
        strays, the reconstruction has a real problem, and nothing is dropped.
        """
        centers  = self.camera_centers
        middle   = np.median(centers, axis=0)
        distance = np.linalg.norm(centers - middle, axis=1)
        keep_cam = distance <= camera_radius * float(np.percentile(distance, 95.0))
        if keep_cam.sum() < len(keep_cam) * (1.0 - max_camera_fraction):
            print(f"[scene] {(~keep_cam).sum()} of {len(keep_cam)} cameras look like strays, which is too many to be strays; "
                  f"keeping them all")
            keep_cam = np.ones(len(keep_cam), dtype=bool)

        spread     = np.linalg.norm(self.points - middle, axis=1)
        keep_point = spread <= point_radius * float(np.percentile(spread, 99.0))

        self.image_names = [name for name, keep in zip(self.image_names, keep_cam) if keep]
        self.R           = self.R[keep_cam]
        self.t           = self.t[keep_cam]
        self.points      = self.points[keep_point]
        self.colors      = self.colors[keep_point]
        dropped = (int((~keep_cam).sum()), int((~keep_point).sum()))
        if any(dropped):
            print(f"[scene] dropped {dropped[0]} camera(s) and {dropped[1]} point(s) from outside the capture")
        return dropped

    def normalize(self, target_radius: float = 1.0) -> float:
        """Recenter and rescale the scene in place; returns the scale.

        A similarity changes nothing observable -- every image reprojects to the same pixels -- but it makes scene-relative hyperparameters
        mean the same thing on every capture. The MEDIAN camera distance, not the mean: one stray pose would otherwise set the scale for the
        scene.
        """
        centers = self.camera_centers
        origin  = centers.mean(axis=0)
        radius  = float(np.median(np.linalg.norm(centers - origin, axis=1)))
        scale   = target_radius / max(radius, 1e-9)

        # x' = s (x - o) leaves each x_cam scaled by s, so R is untouched and t absorbs both the shift and the scale.
        self.t      = (self.t + np.einsum("nij,j->ni", self.R, origin)) * scale
        self.points = (self.points - origin) * scale
        return scale

    def level(self, max_roll_deg: float = 15.0) -> float:
        """Rotate the scene so that up is -Y in place; returns the tilt removed, in degrees.

        Scale and origin are not the whole gauge: the world frame also inherits the orientation of the first camera of the
        initial pair, so the reconstruction comes out tilted by however that camera was held. A rotation changes nothing
        observable either, and it makes the scene open upright in any viewer instead of leaning by ten degrees.

        Gravity is estimated from the cameras. A phone held in landscape keeps its right axis horizontal whatever the pitch,
        so the up direction is the one most nearly orthogonal to every right axis: the smallest eigenvector of sum(r r^T).
        Averaging the cameras' own up axes does not work, because a camera aimed at the floor has a horizontal up axis. If the
        camera was rolled more than `max_roll_deg` on average the assumption has failed, and the scene is left as it is.
        """
        rights = self.R[:, 0, :]                        # camera right axes, in world space
        _, vectors = np.linalg.eigh(rights.T @ rights)
        up = vectors[:, 0]                              # smallest eigenvalue: the direction no right axis points along
        if up @ -self.R[:, 1, :].mean(axis=0) < 0.0:    # point it the same way as the cameras, not into the floor
            up = -up

        roll = float(np.degrees(np.arcsin(np.clip(np.abs(rights @ up), 0.0, 1.0))).mean())
        if roll > max_roll_deg:
            print(f"[scene] up is unreliable (mean camera roll {roll:.0f} deg); leaving the orientation alone")
            return 0.0

        target = np.array([0.0, -1.0, 0.0])
        axis   = np.cross(up, target)
        length = float(np.linalg.norm(axis))
        tilt   = float(np.degrees(np.arctan2(length, up @ target)))
        if length < 1e-9:
            return tilt

        # x' = Rw x, so a camera that read x_cam = R x + t now reads x_cam = (R Rw^T) x' + t.
        world = cv2.Rodrigues(axis / length * np.radians(tilt))[0]
        self.R      = self.R @ world.T
        self.points = self.points @ world.T
        return tilt

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 image_names=np.array(self.image_names),
                 width=self.width, height=self.height, focal=self.focal,
                 principal=self.principal, k1=self.k1, k2=self.k2,
                 R=self.R, t=self.t,
                 points=self.points, colors=self.colors)
        print(f"[scene] saved reconstruction to {path}")

    @staticmethod
    def load(path: str | Path) -> "Reconstruction":
        data = np.load(Path(path), allow_pickle=False)
        return Reconstruction(
            image_names=[str(n) for n in data["image_names"]],
            width=int(data["width"]), height=int(data["height"]),
            focal=float(data["focal"]), principal=data["principal"],
            R=data["R"], t=data["t"],
            points=data["points"], colors=data["colors"],
            k1=float(data["k1"]), k2=float(data["k2"]))

    def summary(self) -> str:
        centers = self.camera_centers
        extent  = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
        return (f"{self.n_images} cameras, {len(self.points)} points\n"
                f"  image      {self.width} x {self.height}\n"
                f"  focal      {self.focal:.1f} px "
                f"({self.fov_x_deg:.1f} x {self.fov_y_deg:.1f} deg FOV)\n"
                f"  distortion k1 {self.k1:+.4f}, k2 {self.k2:+.4f}\n"
                f"  extent     {extent:.3f} (camera radius, normalized)")


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. From video to a usable set of training images
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# Everything downstream is bounded by the image set, and all 649 frames of a 21-second clip is the wrong set: consecutive frames triangulate
# depth from a few millimeters of baseline, and hand-held video is full of motion blur that ends up baked into the poses. So bucket the
# video into evenly spaced windows and keep the sharpest frame of each.
#
# The JPEGs in the output directory *are* the result -- there is no sidecar record of the selection, so reusing them later is a directory
# listing.


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high = sharp, low = blurred.

    Blur is a low-pass filter and destroys exactly the frequencies the Laplacian measures. The score is comparable only between frames of
    the same scene, which is how we use it: to rank frames inside one window.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    num_frames: int = 150,
    max_width: int = 1280,
    jpeg_quality: int = 95,
) -> list[str]:
    """Decode `video_path`, keep the sharpest frame per window, write JPEGs.

    `num_frames` of 100-200 is the sweet spot for a single-object orbit: enough parallax for SfM, few enough that the optimizer sees every
    image many times per minute. `max_width` caps the written resolution -- SIFT does not need 1080p and the rasterizer is linear in pixel
    count.

    Returns the file names it wrote, in video order.
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        raise RuntimeError("video reports zero frames")

    num_frames = min(num_frames, total)
    # Evenly spaced buckets over the video. Sampling uniformly in TIME rather than in camera motion assumes a roughly constant orbit speed.
    edges = np.linspace(0, total, num_frames + 1).astype(int)

    print(f"[frames] {video_path.name}: {total} frames @ {fps:.2f} fps ({total / fps:.1f} s) -> {num_frames} keyframes")

    # Decode sequentially and never seek: seeking in an H.264 stream decodes back to the previous keyframe every time, and we must touch
    # every frame anyway to rank sharpness.
    best: list[tuple[int, float, np.ndarray] | None] = [None] * num_frames
    window = 0
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        while window + 1 < num_frames and idx >= edges[window + 1]:
            window += 1

        small = _downscale(frame, max_width)
        score = sharpness(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        if best[window] is None or score > best[window][1]:
            best[window] = (idx, score, small)

        if idx % 100 == 0:
            print(f"  scanned {idx}/{total} frames", end="\r")
    cap.release()
    print(" " * 40, end="\r")

    names: list[str]    = []
    scores: list[float] = []
    height = width = 0
    for slot in best:
        if slot is None:
            continue                      # empty window (short/truncated video)
        _, score, image = slot
        name = f"frame_{len(names):04d}.jpg"
        cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        names.append(name)
        scores.append(score)
        height, width = image.shape[:2]

    print(f"[frames] wrote {len(names)} images at {width}x{height} to {out_dir}")
    print(f"[frames] sharpness: min {min(scores):.0f}  median {np.median(scores):.0f}  max {max(scores):.0f}")
    return names


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    """Resize so the width is at most `max_width`, preserving aspect ratio.

    INTER_AREA is the right filter for minification: a box prefilter, so it does not alias. Aliasing looks exactly like fine texture to a
    detector.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. Features
# ------------------------------------------------------------------------------------------------------------------------------------------

def detect_features(image_paths: list[Path], max_features: int = 6000):
    """RootSIFT keypoints, descriptors and per-keypoint colors per image.

    RootSIFT (Arandjelovic & Zisserman 2012) is a one-line upgrade to SIFT: L1-normalize, then take the element-wise square root. Euclidean
    distance between two such vectors is the Hellinger distance between the original histograms, which suits SIFT's long-tailed bins far
    better.
    """
    sift = cv2.SIFT_create(nfeatures=max_features,
                           contrastThreshold=0.005,   # video is soft: keep
                           edgeThreshold=12)          # weaker responses too
    keypoints, descriptors, colors, size = [], [], [], None
    for n, path in enumerate(image_paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {path}")
        size    = (bgr.shape[1], bgr.shape[0])
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        xy  = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
        des = np.zeros((0, 128), np.float32) if des is None else des
        # RootSIFT
        des = des / (des.sum(axis=1, keepdims=True) + 1e-7)
        des = np.sqrt(des).astype(np.float32)

        px  = np.clip(np.round(xy).astype(int), 0, [size[0] - 1, size[1] - 1])
        rgb = bgr[px[:, 1], px[:, 0]][:, ::-1].astype(np.float32) / 255.0

        keypoints.append(xy)
        descriptors.append(des)
        colors.append(rgb)
        print(f"  [{n + 1}/{len(image_paths)}] {path.name}: {len(xy)} features", end="\r")
    counts = [len(k) for k in keypoints]
    print(f"[sfm] detected features in {len(image_paths)} images "
          f"(min {min(counts)}, median {int(np.median(counts))}, "
          f"max {max(counts)})")
    return keypoints, descriptors, colors, size


# ------------------------------------------------------------------------------------------------------------------------------------------
# 4. Matching -- a matrix product on the GPU
# ------------------------------------------------------------------------------------------------------------------------------------------

def match_pair(d1: torch.Tensor, d2: torch.Tensor,
               ratio: float = 0.85) -> np.ndarray:
    """Mutual-nearest-neighbor matching with Lowe's ratio test.

    RootSIFT descriptors are unit vectors, so squared distance is 2 - 2 (d1 . d2) and the whole N x M table is one matrix product -- a
    few milliseconds on a GPU against most of a second on a CPU.

    Both filters matter: the ratio test throws away features on repeated texture (our wooden floor has hundreds), and mutual consistency
    demands that i's best match be j and j's best be i.

    Returns an (M, 2) int array of feature-index pairs.
    """
    if d1.shape[0] < 2 or d2.shape[0] < 2:
        return np.zeros((0, 2), np.int64)

    sim = d1 @ d2.T                                  # cosine similarity
    d12 = (2.0 - 2.0 * sim).clamp_min_(0.0)          # squared L2 distance

    # Two nearest going forward, because the ratio test needs the runner-up; only the nearest coming back, because mutual consistency
    # does not. It is worth the asymmetry: reducing along dim 0 crosses the matrix against its layout, and on a 12000 x 12000 table
    # `topk` costs 30 ms there against 2.6 ms for `argmin`.
    best2_fwd = torch.topk(d12, 2, dim=1, largest=False)
    nn_bwd    = d12.argmin(dim=0)

    nn_fwd = best2_fwd.indices[:, 0]
    ok     = best2_fwd.values[:, 0] <= (ratio ** 2) * best2_fwd.values[:, 1]
    ok &= nn_bwd[nn_fwd] == torch.arange(d1.shape[0], device=d1.device)
    idx1 = torch.nonzero(ok, as_tuple=True)[0]
    return torch.stack([idx1, nn_fwd[idx1]], dim=1).cpu().numpy()


def candidate_pairs(n_images: int, window: int = 12, subset_stride: int = 6) -> list[tuple[int, int]]:
    """Which image pairs are worth matching at all.

    Temporal order is a strong prior on overlap, so instead of all O(n^2) pairs we take every pair inside a sliding `window` -- guaranteed
    overlap, and most of the constraint -- plus every pair from a strided subset, which is the cheap insurance that lets the reconstruction
    close the loop when the operator walks back round to the starting side. Without those, a reconstruction drifts into a spiral instead of
    a circle.
    """
    pairs   = {(i, j) for i in range(n_images) for j in range(i + 1, min(i + window + 1, n_images))}
    anchors = list(range(0, n_images, subset_stride))
    pairs.update((i, j) for i, j in itertools.combinations(anchors, 2))
    return sorted(pairs)


def verify_pair(xy1: np.ndarray, xy2: np.ndarray, matches: np.ndarray, K: np.ndarray, threshold_px: float = 1.5):
    """RANSAC essential matrix + the planar-degeneracy check.

    Returns (inlier_matches, homography_ratio) or (None, None) if the pair fails. The ratio is (homography inliers) / (essential inliers);
    near 1 means a plane or a pure rotation explains the pair just as well, the essential matrix is degenerate and `recoverPose` returns
    confident nonsense. We use it to keep such pairs out of the initialization, where a bad choice is fatal.
    """
    if len(matches) < 30:
        return None, None
    p1 = xy1[matches[:, 0]]
    p2 = xy2[matches[:, 1]]

    E, mask_e = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.9999, threshold=threshold_px)
    if E is None or E.shape != (3, 3) or mask_e is None:
        return None, None
    mask_e = mask_e.ravel().astype(bool)
    if int(mask_e.sum()) < 25:
        return None, None

    # Both models must be scored at the same inlier threshold, or the ratio measures our thresholds rather than the geometry.
    _, mask_h = cv2.findHomography(p1, p2, cv2.RANSAC, threshold_px)
    n_h       = 0 if mask_h is None else int(mask_h.ravel().sum())
    return matches[mask_e], n_h / max(int(mask_e.sum()), 1)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 5. Tracks -- chaining pairwise matches into multi-view observations
# ------------------------------------------------------------------------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.size   = np.ones(n, dtype=np.int64)

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:          # path compression
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def build_tracks(n_features: list[int], pair_matches: dict[tuple[int, int], np.ndarray], min_length: int = 3):
    """Merge pairwise matches into tracks: one 3D point, many 2D observations.

    "Same world point" is an equivalence relation, so the tracks are the connected components of the match graph and union-find finds them
    in near-linear time.

    A track may hold at most ONE feature per image. A track that violates that has absorbed a bad match somewhere in the chain and we cannot
    tell which link, so we drop the whole track -- far cheaper than letting it poison bundle adjustment.

    Returns (tracks, track_of) where `tracks[t]` is a list of (image, feature) and `track_of[i][f]` is the track index of feature f in image
    i, or -1.
    """
    offsets = np.cumsum([0] + n_features)
    uf      = UnionFind(int(offsets[-1]))
    for (i, j), m in pair_matches.items():
        for a, b in m:
            uf.union(int(offsets[i] + a), int(offsets[j] + b))

    roots      = np.array([uf.find(i) for i in range(int(offsets[-1]))])
    order      = np.argsort(roots, kind="stable")
    boundaries = np.flatnonzero(np.diff(roots[order])) + 1

    tracks: list[list[tuple[int, int]]] = []
    track_of = [np.full(n, -1, np.int64) for n in n_features]
    n_conflict = 0
    for group in np.split(order, boundaries):
        if group.size < min_length:
            continue
        images = np.searchsorted(offsets, group, side="right") - 1
        if np.unique(images).size != images.size:       # two features, one image
            n_conflict += 1
            continue
        t   = len(tracks)
        obs = []
        for node, img in zip(group, images):
            feat = int(node - offsets[img])
            track_of[img][feat] = t
            obs.append((int(img), feat))
        tracks.append(obs)

    lengths = np.array([len(t) for t in tracks])
    print(f"[sfm] {len(tracks)} tracks (>= {min_length} views), "
          f"{n_conflict} discarded as inconsistent; "
          f"track length: median {int(np.median(lengths))}, "
          f"max {lengths.max()}, mean {lengths.mean():.1f}")
    return tracks, track_of


# ------------------------------------------------------------------------------------------------------------------------------------------
# 6. Triangulation
# ------------------------------------------------------------------------------------------------------------------------------------------

def triangulate_dlt(P: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Batched multi-view triangulation by the Direct Linear Transform.

    `P` is (B, V, 3, 4) projection matrices and `xy` is (B, V, 2) *normalized* image coordinates. "Projects to (x, y)" means the ray and the
    homogeneous point are parallel, so their cross product vanishes; two of the three resulting equations are independent:

        x * P[2] - P[0] = 0        y * P[2] - P[1] = 0

    Stacked over V views that is A X = 0 with A of shape (2V, 4), whose least-squares solution under |X| = 1 is the right singular vector of
    the smallest singular value -- one batched SVD for the whole set. Normalized coordinates keep every entry of A within an order of
    magnitude of every other, which is the conditioning Hartley's normalized DLT is about.
    """
    A          = np.empty((P.shape[0], 2 * P.shape[1], 4))
    A[:, 0::2] = xy[..., 0:1] * P[:, :, 2] - P[:, :, 0]
    A[:, 1::2] = xy[..., 1:2] * P[:, :, 2] - P[:, :, 1]
    _, _, Vh   = np.linalg.svd(A)
    X          = Vh[:, -1]
    return X[:, :3] / np.where(np.abs(X[:, 3:]) < 1e-12, 1e-12, X[:, 3:])


def _pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.empty((3, 4))
    P[:, :3], P[:, 3] = R, t
    return P


# ------------------------------------------------------------------------------------------------------------------------------------------
# 7. Bundle adjustment: the joint refinement that makes splats sharp
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# The incremental loop in section 8 is greedy -- every pose is estimated from points that came from earlier poses -- so errors accumulate
# and a 120-image orbit closes with a visible seam. Bundle adjustment takes all cameras, all points and the shared lens at once and
# minimizes the only thing we observe, the reprojection error in pixels:
#
#     E = sum over observations (i, j) of  rho( || pi(C_i, X_j; lens) - u_ij || )
#
# It matters here because splatting has no mechanism to correct a bad pose: pose error is absorbed as geometry error, turning a crisp edge
# into a smear of semi-transparent blobs.
#
# Four ideas carry the implementation:
#
#   * a LOCAL rotation parameterization -- solve for a small left perturbation
#     R <- exp([d]x) R, whose derivative at d = 0 is just -[p]x;
#   * LEVENBERG-MARQUARDT damping, which also regularizes a system that is
#     singular by construction (the scene is fixed only up to a similarity);
#   * the SCHUR COMPLEMENT: the point-point block is block-diagonal, so the
#     points can be eliminated analytically, leaving 6 * n_cameras + n_lens
#     unknowns instead of a 120,000-square factorization;
#   * SHARED LENS parameters -- one focal length and two radial coefficients
#     for the whole video. Distortion is not optional: a pinhole-only model
#     compensates by inflating the focal length, on this capture from a
#     sensible 60-degree field of view to an absurd 19-degree one.
#
# Accumulation runs in PyTorch so the one large operation -- scattering millions of small blocks into the reduced matrix -- happens on the
# GPU.


# -- the local rotation parameterization ------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrices: [v]x u = v x u."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1)], dim=-2)


def _so3_exp(d: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angle increments -> (N, 3, 3) rotation matrices.

    The Taylor branch matters: LM increments are tiny, so sin(t)/t is evaluated at t ~ 1e-8 on almost every iteration and the naive form
    loses all its significant digits.
    """
    theta = d.norm(dim=1, keepdim=True)
    small = (theta < 1e-8)
    t     = theta.clamp_min(1e-12)
    a     = torch.where(small, torch.ones_like(t), torch.sin(t) / t)
    b     = torch.where(small, 0.5 * torch.ones_like(t), (1.0 - torch.cos(t)) / t ** 2)
    K     = _skew(d)
    eye   = torch.eye(3, dtype=d.dtype, device=d.device).expand(d.shape[0], 3, 3)
    return eye + a.unsqueeze(2) * K + b.unsqueeze(2) * (K @ K)


# -- Levenberg-Marquardt with the Schur complement -------------------------

def _point_pair_index(pt_idx: torch.Tensor, n_points: int):
    """Every (observation a, observation b) pair that shares a 3D point.

    These pairs are exactly the nonzero blocks of W V^-1 W^T: two cameras are coupled in the reduced system precisely when they see a common
    point. A track of length n contributes n^2 pairs, so long tracks dominate the cost. Built once and reused -- the sparsity structure
    never changes.
    """
    order  = torch.argsort(pt_idx)
    counts = torch.bincount(pt_idx, minlength=n_points)
    starts = torch.cumsum(counts, 0) - counts

    n_pairs    = counts * counts
    pair_pt    = torch.repeat_interleave(torch.arange(n_points, device=pt_idx.device), n_pairs)
    pair_start = torch.cumsum(n_pairs, 0) - n_pairs
    local      = (torch.arange(int(n_pairs.sum()), device=pt_idx.device) - torch.repeat_interleave(pair_start, n_pairs))
    n_of       = counts[pair_pt]
    return (order[starts[pair_pt] + local // n_of], order[starts[pair_pt] + local % n_of])


def bundle_adjust(
    R: np.ndarray,              # (n_cams, 3, 3) world->camera rotations
    t: np.ndarray,              # (n_cams, 3)    world->camera translations
    points: np.ndarray,         # (n_pts, 3)
    cam_idx: np.ndarray,        # (n_obs,) which camera made each observation
    pt_idx: np.ndarray,         # (n_obs,) which point it observed
    observed: np.ndarray,       # (n_obs, 2) measured pixel coordinates
    intr: Intrinsics,
    optimize_intrinsics: bool = True,
    huber_pixels: float = 2.0,
    iterations: int = 25,
    device: str | None = None,
    verbose: bool = True,
):
    """Refine cameras, points and (optionally) the lens jointly.

    Returns (R, t, points, Intrinsics, median_reprojection_error_px).

    The robust loss is Huber at `huber_pixels`, applied by iteratively reweighted least squares: a residual above the threshold is
    down-weighted by delta/|r|, which is exactly the weight that turns a quadratic step into a Huber step. Under a squared loss one
    200-pixel mismatch would outvote a thousand good measurements and drag the camera with it.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Float64 throughout. Bundle adjustment differences nearly equal quantities (a projection and a measurement of the same point); in
    # float32 the normal equations lose most of their significant digits.
    f64 = torch.float64

    Rc   = torch.as_tensor(R, dtype=f64, device=dev).clone()
    tc   = torch.as_tensor(t, dtype=f64, device=dev).clone()
    X    = torch.as_tensor(points, dtype=f64, device=dev).clone()
    obs  = torch.as_tensor(observed, dtype=f64, device=dev)
    ci   = torch.as_tensor(cam_idx, dtype=torch.long, device=dev)
    pi   = torch.as_tensor(pt_idx, dtype=torch.long, device=dev)
    c0   = torch.as_tensor(intr.principal, dtype=f64, device=dev)
    lens = torch.tensor([intr.focal, intr.k1, intr.k2], dtype=f64, device=dev)

    n_cams, n_pts, n_obs = Rc.shape[0], X.shape[0], ci.shape[0]
    # The shared lens parameters join the cameras on the "few unknowns" side of the Schur complement, so the reduced system is 6 per camera
    # plus these.
    n_lens = 3 if optimize_intrinsics else 0
    block  = 6 + n_lens                      # unknowns touched by one observation
    dim    = 6 * n_cams + n_lens             # size of the reduced system
    delta  = float(huber_pixels)

    def forward(Rc_, tc_, X_, lens_):
        """Camera-space points, depths, normalized coords and residuals."""
        p    = torch.einsum("kij,kj->ki", Rc_[ci], X_[pi]) + tc_[ci]
        z    = p[:, 2].clamp_min(1e-6)
        xy   = p[:, :2] / z.unsqueeze(1)
        r2   = (xy ** 2).sum(dim=1)
        d    = 1.0 + lens_[1] * r2 + lens_[2] * r2 ** 2
        proj = lens_[0] * xy * d.unsqueeze(1) + c0
        return p, z, xy, r2, d, proj - obs

    def huber_cost(res: torch.Tensor) -> float:
        n    = res.norm(dim=1)
        quad = torch.clamp(n, max=delta)
        return float((0.5 * quad ** 2 + delta * (n - quad)).sum())

    a_idx, b_idx = _point_pair_index(pi, n_pts)
    p, z, xy, r2, dist, res = forward(Rc, tc, X, lens)
    cost = huber_cost(res)
    err0 = res.norm(dim=1)
    if verbose:
        print(f"[ba] {n_cams} cameras, {n_pts} points, {n_obs} observations; "
              f"start RMS {float((err0 ** 2).mean().sqrt()):.3f} px "
              f"(median {float(err0.median()):.3f} px, "
              f"{int((err0 > delta).sum())} above {delta:g} px)")

    # Column indices touched by each observation: its camera's 6, then the shared lens block at the end. One array drives every scatter
    # below.
    cols        = torch.empty(n_obs, block, dtype=torch.long, device=dev)
    cols[:, :6] = (ci * 6).unsqueeze(1) + torch.arange(6, device=dev)
    if n_lens:
        cols[:, 6:] = 6 * n_cams + torch.arange(n_lens, device=dev)

    lam, iteration = 1e-3, 0
    for iteration in range(iterations):
        # ---- Jacobians of one observation ---------------------------------
        # Chain rule: pixels <- distorted normalized <- normalized <- camera.
        x, y = xy[:, 0], xy[:, 1]
        s    = lens[1] + 2.0 * lens[2] * r2         # d(dist)/d(r2)
        # M = d(u,v)/d(x,y), including the distortion's own dependence on x, y
        M          = torch.empty(n_obs, 2, 2, dtype=f64, device=dev)
        M[:, 0, 0] = lens[0] * (dist + 2.0 * x * x * s)
        M[:, 0, 1] = M[:, 1, 0] = lens[0] * 2.0 * x * y * s
        M[:, 1, 1] = lens[0] * (dist + 2.0 * y * y * s)
        # Bp = d(x,y)/dp = (1/z) [[1, 0, -x], [0, 1, -y]]
        Bp          = torch.zeros(n_obs, 2, 3, dtype=f64, device=dev)
        inv_z       = 1.0 / z
        Bp[:, 0, 0] = inv_z
        Bp[:, 1, 1] = inv_z
        Bp[:, 0, 2] = -x * inv_z
        Bp[:, 1, 2] = -y * inv_z
        A           = M @ Bp                         # d(u,v)/dp   (K, 2, 3)

        # IRLS: fold sqrt(w) into residual and Jacobian, then solve plain weighted least squares.
        err_now = res.norm(dim=1).clamp_min(1e-12)
        weight  = torch.where(err_now <= delta, torch.ones_like(err_now), delta / err_now)
        sw      = torch.sqrt(weight).unsqueeze(1)

        Jc = torch.empty(n_obs, 2, block, dtype=f64, device=dev)
        Jc[:, :, :3] = -A @ _skew(p)                 # rotation increment
        Jc[:, :, 3:6] = A                            # translation increment
        if n_lens:
            Jc[:, :, 6] = xy * dist.unsqueeze(1)                  # d/d focal
            Jc[:, :, 7] = f * xy * r2.unsqueeze(1)                # d/d k1
            Jc[:, :, 8] = f * xy * (r2 ** 2).unsqueeze(1)         # d/d k2
        Jc *= sw.unsqueeze(2)
        Jp = (A @ Rc[ci]) * sw.unsqueeze(2)                       # (K, 2, 3)
        rw = res * sw

        # ---- normal equations ---------------------------------------------
        V = torch.zeros(n_pts, 3, 3, dtype=f64, device=dev)
        V.index_add_(0, pi, Jp.transpose(1, 2) @ Jp)
        W  = Jc.transpose(1, 2) @ Jp                              # (K, B, 3)
        bp = torch.zeros(n_pts, 3, dtype=f64, device=dev)
        bp.index_add_(0, pi, -torch.einsum("kij,ki->kj", Jp, rw))

        S      = torch.zeros(dim, dim, dtype=f64, device=dev)
        S_flat = S.view(-1)
        S_flat.index_add_(0, (cols.unsqueeze(2) * dim + cols.unsqueeze(1)).reshape(-1), (Jc.transpose(1, 2) @ Jc).reshape(-1))

        g = torch.zeros(dim, dtype=f64, device=dev)
        g.index_add_(0, cols.reshape(-1), -torch.einsum("kij,ki->kj", Jc, rw).reshape(-1))

        # ---- damping (Marquardt's scale-aware variant) ---------------------
        # Adding lambda * diag(H) rather than lambda * I makes the damping invariant to the units of each unknown -- radians, scene units
        # and pixels differ by orders of magnitude.
        diag = torch.diagonal(S).clone().clamp_min(1e-12)
        S += torch.diag(lam * diag + 1e-12)
        Vd = V + torch.diag_embed(lam * torch.diagonal(V, dim1=1, dim2=2).clamp_min(1e-12) + 1e-12)

        # ---- eliminate the points (Schur complement) -----------------------
        Vinv = torch.linalg.inv(Vd)
        Y    = W @ Vinv[pi]                                       # (K, B, 3)
        g.index_add_(0, cols.reshape(-1), -torch.einsum("kij,kj->ki", Y, bp[pi]).reshape(-1))

        # S -= W V^-1 W^T, one chunk of shared-point observation pairs at a time (the full index array would be tens of gigabytes).
        chunk = max(1, (1 << 22) // (block * block))
        for lo in range(0, a_idx.numel(), chunk):
            a, b   = a_idx[lo:lo + chunk], b_idx[lo:lo + chunk]
            ca, cb = cols[a], cols[b]
            S_flat.index_add_(0, (ca.unsqueeze(2) * dim + cb.unsqueeze(1)).reshape(-1), -(Y[a] @ W[b].transpose(1, 2)).reshape(-1))

        # ---- solve, then back-substitute for the points --------------------
        try:
            dc = torch.linalg.solve(S, g)
        except Exception:                       # singular even with damping
            lam *= 10.0
            continue
        if not torch.isfinite(dc).all():
            lam *= 10.0
            continue

        coupled = torch.zeros(n_pts, 3, dtype=f64, device=dev)
        coupled.index_add_(0, pi, torch.einsum("kij,ki->kj", W, dc[cols]))
        dX = torch.einsum("nij,nj->ni", Vinv, bp - coupled)

        # ---- accept or reject ----------------------------------------------
        dcam     = dc[:6 * n_cams].view(n_cams, 6)
        dR       = _so3_exp(dcam[:, :3])
        R_new    = dR @ Rc
        t_new    = torch.einsum("nij,nj->ni", dR, tc) + dcam[:, 3:6]
        X_new    = X + dX
        lens_new = lens.clone()
        if n_lens:
            lens_new = lens + dc[6 * n_cams:]

        trial    = forward(R_new, t_new, X_new, lens_new)
        cost_new = huber_cost(trial[-1])
        if cost_new < cost:
            improvement = (cost - cost_new) / max(cost, 1e-18)
            Rc, tc, X, lens = R_new, t_new, X_new, lens_new
            p, z, xy, r2, dist, res = trial
            cost = cost_new
            lam = max(lam * 0.3, 1e-10)
            if improvement < 1e-5:              # converged
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                break

    err = res.norm(dim=1)
    out = Intrinsics(focal=float(lens[0]), principal=intr.principal.copy(), k1=float(lens[1]), k2=float(lens[2]))
    if verbose:
        print(f"[ba] {iteration + 1} LM iterations; "
              f"RMS {float((err0 ** 2).mean().sqrt()):.3f} -> "
              f"{float((err ** 2).mean().sqrt()):.3f} px, "
              f"median {float(err0.median()):.3f} -> {float(err.median()):.3f} px"
              + (f"; focal {intr.focal:.1f} -> {out.focal:.1f}, "
                 f"k1 {out.k1:+.4f}, k2 {out.k2:+.4f}"
                 if optimize_intrinsics else ""))
    return (Rc.cpu().numpy(), tc.cpu().numpy(), X.cpu().numpy(), out, float(err.median()))


# ------------------------------------------------------------------------------------------------------------------------------------------
# 8. Incremental reconstruction
# ------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class SfMConfig:
    # How many following frames each keyframe is matched against. What matters is the arc of camera motion the window spans, not the number
    # of frames, so it scales with how finely the video was sampled: 20 frames of a capture sampled at four keyframes per second is about
    # five seconds of motion, which is a wide enough baseline to triangulate and a narrow enough one to still match.
    match_window: int = 20
    subset_stride: int = 6
    ratio: float = 0.80
    ransac_px: float = 1.5
    min_track_length: int = 3
    min_triangulation_angle_deg: float = 1.5

    # These four defaults are the difference between a scene and a blur, and the capture in this chapter is what tuned them. At 6000
    # features a pair of neighboring keyframes shared only 168 inliers, which left 62 points per camera; at 12000 it is 154 points per
    # camera. Thin correspondences also let one bad pose through, and bundle adjustment then diverged -- a start RMS of 4e8 px -- so the
    # RANSAC gates are tighter and the adjuster runs more often, catching errors while they are still small.
    max_features: int = 12000
    max_reprojection_px: float = 2.0
    pnp_ransac_px: float = 2.5
    ba_growth: float = 1.07          # re-run BA when the model grows 7%

    # What it takes to accept a camera. A pose supported by 20 of 300 correspondences is a coin toss that lands inside the capture volume,
    # where nothing downstream can see that it is wrong: `prune_outliers` looks for cameras that are far away, and the splat optimizer has
    # no way to move a camera at all, so it bends the geometry until the wrong view is explained. On this capture three cameras came in
    # that way, and the views next to them trained to 13 dB while the rest of the scene reached 24.
    min_pnp_inliers: int = 40
    min_pnp_inlier_ratio: float = 0.25
    min_camera_observations: float = 0.25   # of the median camera's, after the final adjustment
    max_pnp_attempts: int = 3

    # f = 1.05 * image width is about a 51-degree horizontal field of view, which is where a phone main camera sits in 16:9 video mode.
    focal_prior: float = 1.05

    # Refining the focal length is normally free accuracy, and on a capture that covers many angles it is: this chapter's video moves
    # it 1344 -> 1451 px, an 8% correction in a sensible direction. It is off by default because whether the refinement can be trusted
    # depends on the capture, and the capture cannot tell you. On an earlier single-elevation orbit of the same objects it ran to about
    # 3550 px -- an 18-degree field of view, impossible for a phone held 40 cm away -- and halved the reprojection error while flattening
    # the scene into a bas-relief: as the focal grows the model tends toward orthographic, whose extra gauge freedom absorbs the rolling
    # shutter and stabilization warps a pinhole model cannot express, and no view from above exists to contradict it. Scan fixed priors
    # first (the chapter shows how); if the error has a minimum at the physical focal length rather than falling all the way, turn this on.
    optimize_focal: bool = False


class IncrementalSfM:
    """Grows a reconstruction one camera at a time.

    State:
      poses[i]     -- (R, t) world->camera for every registered image
      points[t]    -- the 3D position of track t, once triangulated
    """

    def __init__(self, keypoints, descriptors, colors, image_size, tracks, track_of, pair_matches, cfg: SfMConfig):
        self.kp, self.desc, self.col = keypoints, descriptors, colors
        self.width, self.height = image_size
        self.tracks, self.track_of = tracks, track_of
        self.pair_matches = pair_matches
        self.cfg = cfg
        self.n_images = len(keypoints)

        self.intr = Intrinsics(focal=cfg.focal_prior * self.width, principal=np.array([self.width / 2.0, self.height / 2.0]))
        self.poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.points: dict[int, np.ndarray] = {}
        self.failed: set[int] = set()      # images PnP could not place
        # Images whose support was too thin, with the attempt count and the map size at the time. They are offered again once the map has
        # grown, because the reason is usually that there was not yet enough geometry to aim at.
        self.deferred: dict[int, tuple[int, int]] = {}
        # Mirror of the point dictionary, so "how many of this image's features already have a 3D point?" is one vectorized lookup.
        self.has_point     = np.zeros(len(tracks), dtype=bool)
        self._last_ba_size = 0

        self._normalized = [None] * self.n_images
        self._refresh_normalized()

    @property
    def K(self) -> np.ndarray:
        return self.intr.K

    def _refresh_normalized(self) -> None:
        """Undistorted, K-free image coordinates for every feature.

        Every geometric routine in this file works in this (u - c) / f space, so the lens model appears in exactly one place.
        `undistortPoints` inverts the radial polynomial numerically -- it has no closed-form inverse -- and divides out K in the same pass.
        """
        for i in range(self.n_images):
            self._normalized[i] = cv2.undistortPoints(
                self.kp[i].reshape(-1, 1, 2), self.intr.K,
                self.intr.dist_coeffs).reshape(-1, 2).astype(np.float64)

    # -- initialization ------------------------------------------------------

    def choose_initial_pair(self, homography_ratios,
                            n_candidates: int = 60) -> tuple[int, int] | None:
        """Pick the two images that will define the world frame.

        Every camera and point that follows is measured against this pair and a bad seed cannot be repaired later. Three properties compete:
        many verified matches; a WIDE baseline, since depth error scales as 1/tan(angle) and neighboring frames triangulate a fog of noise;
        and non-degenerate geometry, measured by the homography ratio. Our floor keeps that ratio high for every pair, so it only ranks --
        the triangulation angle does the real filtering.

        Hence the score `points * min(angle, 15 deg)`: a product refuses to trade away either property, and the cap stops a pair with an
        enormous baseline and a handful of surviving matches from winning.
        """
        candidates = sorted(((len(m), i, j) for (i, j), m in self.pair_matches.items() if j - i >= 3), reverse=True)[:n_candidates]

        best = None
        for _, i, j in candidates:
            pose = self.two_view_pose(i, j)
            if pose is None:
                continue
            R, t, angle, n_ok = pose
            if angle < 4.0 or n_ok < 100:
                continue
            score = n_ok * min(angle, 15.0) / (1.0 + homography_ratios[(i, j)])
            if best is None or score > best[0]:
                best = (score, i, j, R, t, angle, n_ok)

        if best is None:
            return None
        _, i, j, R, t, angle, n_ok = best
        print(f"[sfm] initial pair ({i}, {j}): {n_ok} points triangulated, "
              f"median angle {angle:.1f} deg, "
              f"H-ratio {homography_ratios[(i, j)]:.2f}")
        self.poses[i] = (np.eye(3), np.zeros(3))
        self.poses[j] = (R, t)
        return i, j

    def two_view_pose(self, i: int, j: int):
        """Relative pose from the essential matrix, plus a quality report."""
        m       = self.pair_matches[(i, j)]
        p1, p2  = self.kp[i][m[:, 0]], self.kp[j][m[:, 1]]
        E, mask = cv2.findEssentialMat(p1, p2, self.K, method=cv2.RANSAC, prob=0.9999, threshold=self.cfg.ransac_px)
        if E is None or E.shape != (3, 3):
            return None
        # recoverPose resolves the four-fold ambiguity of E with the cheirality test: keep the decomposition that puts most points in front
        # of both cameras.
        n_ok, R, t, mask_pose = cv2.recoverPose(E, p1, p2, self.K, mask=mask)
        if n_ok < 50:
            return None
        keep = mask_pose.ravel() > 0
        n1   = self._normalized[i][m[keep, 0]]
        n2   = self._normalized[j][m[keep, 1]]
        R, t = R.astype(np.float64), t.ravel().astype(np.float64)

        P = np.stack([np.tile(_pose_matrix(np.eye(3), np.zeros(3)),
                              (n1.shape[0], 1, 1)),
                      np.tile(_pose_matrix(R, t), (n1.shape[0], 1, 1))], axis=1)
        X      = triangulate_dlt(P, np.stack([n1, n2], axis=1))
        angles = self._triangulation_angles(X, [np.zeros(3), -R.T @ t])
        return R, t, float(np.median(angles)), int(keep.sum())

    @staticmethod
    def _triangulation_angles(X: np.ndarray, centers) -> np.ndarray:
        """Angle (degrees) subtended at each point by two camera centers.

        The conditioning of a depth estimate: near 0 degrees the rays are parallel and depth is unconstrained, which is why frames
        millimeters apart cannot triangulate anything on their own.
        """
        r1 = X - centers[0]
        r2 = X - centers[1]
        r1 /= np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12
        r2 /= np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12
        return np.degrees(np.arccos(np.clip(np.sum(r1 * r2, axis=1), -1, 1)))

    # -- growing the model ---------------------------------------------------

    def register_next(self) -> int | None:
        """Register the unregistered image with the most 3D correspondences.

        Once points exist, adding a camera is no longer a two-view problem: the pose comes from Perspective-n-Point inside RANSAC, which
        anchors the new camera to the existing map rather than to its neighbor -- that is what stops drift from compounding.
        """
        best, best_count = None, 0
        for i in range(self.n_images):
            if i in self.poses or i in self.failed:
                continue
            attempts, size = self.deferred.get(i, (0, 0))
            if attempts and len(self.points) < 1.15 * size:
                continue                       # wait for the map to grow before trying this one again
            tid   = self.track_of[i]
            count = int(np.count_nonzero(self.has_point[tid[tid >= 0]]))
            if count > best_count:
                best, best_count = i, count
        if best is None or best_count < 25:
            return None

        tid   = self.track_of[best]
        feats = np.flatnonzero((tid >= 0) & self.has_point[np.maximum(tid, 0)])
        obj   = np.array([self.points[tid[f]] for f in feats])
        img   = self.kp[best][feats]

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.intr.K, self.intr.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=self.cfg.pnp_ransac_px,
            iterationsCount=2000, confidence=0.9999)
        support = 0 if (not ok or inliers is None) else len(inliers)
        if support < max(self.cfg.min_pnp_inliers, self.cfg.min_pnp_inlier_ratio * best_count):
            self._defer(best, support, best_count)
            return -1
        inliers = inliers.ravel()
        # Refine on the inliers only: RANSAC gives a hypothesis supported by a minimal sample, LM turns it into the least-squares optimum.
        rvec, tvec = cv2.solvePnPRefineLM(obj[inliers], img[inliers],
                                          self.intr.K, self.intr.dist_coeffs,
                                          rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        self.poses[best] = (R, tvec.ravel())
        print(f"  + image {best:3d}: PnP {len(inliers)}/{best_count} inliers, {len(self.poses)}/{self.n_images} registered", end="\r")
        return best

    def triangulate_new_points(self) -> int:
        """Triangulate every track that is now seen by >= 2 registered views.

        Accepted only if the point survives three tests: positive depth in every view, a sufficient triangulation angle, and a small
        reprojection error. A failing track is left untriangulated and gets another chance as more cameras arrive.
        """
        groups: dict[int, list[int]] = {}
        for t, obs in enumerate(self.tracks):
            if t in self.points:
                continue
            seen = [(i, f) for i, f in obs if i in self.poses]
            if len(seen) >= 2:
                groups.setdefault(len(seen), []).append(t)

        added = 0
        for n_views, track_ids in groups.items():
            P       = np.empty((len(track_ids), n_views, 3, 4))
            xy      = np.empty((len(track_ids), n_views, 2))
            centers = np.empty((len(track_ids), n_views, 3))
            for row, t in enumerate(track_ids):
                seen = [(i, f) for i, f in self.tracks[t] if i in self.poses]
                for v, (i, f) in enumerate(seen):
                    R, tr           = self.poses[i]
                    P[row, v]       = _pose_matrix(R, tr)
                    xy[row, v]      = self._normalized[i][f]
                    centers[row, v] = -R.T @ tr
            X = triangulate_dlt(P, xy)

            # depth in every view = third row of (R X + t)
            depth = np.einsum("bvij,bj->bvi", P[..., :3], X) + P[..., 3]
            good  = np.all(depth[..., 2] > 1e-4, axis=1)

            # widest baseline available for this point
            rays = X[:, None, :] - centers
            rays /= np.linalg.norm(rays, axis=2, keepdims=True) + 1e-12
            cos   = np.einsum("bvi,bwi->bvw", rays, rays)
            angle = np.degrees(np.arccos(np.clip(cos.min(axis=(1, 2)), -1, 1)))
            good &= angle >= self.cfg.min_triangulation_angle_deg

            proj = depth[..., :2] / np.maximum(depth[..., 2:], 1e-6)
            err  = np.linalg.norm(proj - xy, axis=2).max(axis=1) * self.intr.focal
            good &= err <= self.cfg.max_reprojection_px

            for row, t in enumerate(track_ids):
                if good[row]:
                    self.points[t]    = X[row]
                    self.has_point[t] = True
                    added += 1
        return added

    # -- refinement ----------------------------------------------------------

    def _gather_observations(self, track_ids: list[int]):
        image_ids = sorted(self.poses)
        slot = {i: n for n, i in enumerate(image_ids)}
        cam_idx, pt_idx, uv = [], [], []
        for row, t in enumerate(track_ids):
            for i, f in self.tracks[t]:
                if i in slot:
                    cam_idx.append(slot[i])
                    pt_idx.append(row)
                    uv.append(self.kp[i][f])
        return (image_ids, np.array(cam_idx, np.int64), np.array(pt_idx, np.int64), np.array(uv, np.float64))

    def run_bundle_adjustment(self, max_points: int = 40000, iterations: int = 25, optimize_focal: bool = True) -> None:
        """Global refinement of every registered camera and point."""
        track_ids = sorted(self.points)
        if len(track_ids) > max_points:
            # Cameras are over-determined long before every point is used, so intermediate calls subsample. The final call does not.
            rng       = np.random.default_rng(0)
            track_ids = sorted(rng.choice(track_ids, max_points, replace=False).tolist())

        image_ids, cam_idx, pt_idx, uv = self._gather_observations(track_ids)
        R = np.stack([self.poses[i][0] for i in image_ids])
        t = np.stack([self.poses[i][1] for i in image_ids])
        pts = np.array([self.points[k] for k in track_ids])

        R, t, pts, intr, _ = bundle_adjust(
            R, t, pts, cam_idx, pt_idx, uv, self.intr,
            optimize_intrinsics=optimize_focal, iterations=iterations)

        for row, i in enumerate(image_ids):
            self.poses[i] = (R[row], t[row])
        for row, k in enumerate(track_ids):
            self.points[k] = pts[row]
        if optimize_focal:
            self.intr = intr
            self._refresh_normalized()
        self._last_ba_size = len(self.poses)

    def _defer(self, image: int, support: int, candidates: int) -> None:
        """Put a weakly supported image back in the queue instead of failing it for good."""
        attempts = self.deferred.get(image, (0, 0))[0] + 1
        if attempts >= self.cfg.max_pnp_attempts:
            self.failed.add(image)
            self.deferred.pop(image, None)
            print(f"  ! image {image:3d}: PnP support stayed thin ({support}/{candidates} inliers); leaving it out")
            return
        self.deferred[image] = (attempts, len(self.points))
        print(f"  - image {image:3d}: PnP support too thin ({support}/{candidates} inliers); deferring", end="\r")

    def drop_weak_cameras(self) -> int:
        """Retire cameras that kept far fewer observations than the rest; returns how many went.

        This is the last line of defence against a wrong pose. Registration can succeed on a consensus of mismatches, and the camera then
        lands somewhere plausible inside the capture. What gives it away is what happens next: observation filtering removes nearly
        everything it claimed to see, and it ends the reconstruction with a few dozen observations where its neighbors have several hundred.
        """
        counts = {i: 0 for i in self.poses}
        for t in self.points:
            for i, _ in self.tracks[t]:
                if i in counts:
                    counts[i] += 1
        if len(counts) < 8:
            return 0
        floor = self.cfg.min_camera_observations * float(np.median(list(counts.values())))
        weak  = sorted(i for i, n in counts.items() if n < floor)
        for i in weak:
            del self.poses[i]
        if weak:
            print(f"[sfm] retired {len(weak)} camera(s) with under {floor:.0f} observations: "
                  + ", ".join(f"{i} ({counts[i]})" for i in weak))
        return len(weak)

    def filter_observations(self) -> int:
        """Drop individual observations that no longer fit, not whole points.

        A track can be 95% correct: nine images agree and the tenth latched onto a different knot in the wood grain. Dropping the track
        throws away nine good measurements; keeping it drags the point, and every camera that sees it, toward the bad one. So we prune per
        observation, as COLMAP does, and retire the point only when too few views remain.
        """
        removed = 0
        for t in list(self.points):
            X    = self.points[t]
            keep = []
            for i, f in self.tracks[t]:
                if i not in self.poses:
                    keep.append((i, f))            # not yet registered: hold on
                    continue
                p   = self.poses[i][0] @ X + self.poses[i][1]
                err = (np.linalg.norm(p[:2] / p[2] - self._normalized[i][f]) * self.intr.focal) if p[2] > 1e-6 else np.inf
                if err <= self.cfg.max_reprojection_px:
                    keep.append((i, f))
                else:
                    self.track_of[i][f] = -1
                    removed += 1
            if sum(1 for i, _ in keep if i in self.poses) < 2:
                del self.points[t]
                self.has_point[t] = False
            self.tracks[t] = keep
        return removed

    def filter_points(self) -> int:
        """Drop points that no longer reproject consistently.

        Bundle adjustment moves cameras, so a point that was fine under the old poses may now be an outlier. Alternating refinement and
        filtering keeps the cloud clean enough to initialize Gaussians with.
        """
        removed = []
        for t, X in self.points.items():
            errs, depths = [], []
            for i, f in self.tracks[t]:
                if i not in self.poses:
                    continue
                R, tr = self.poses[i]
                p     = R @ X + tr
                depths.append(p[2])
                if p[2] > 1e-6:
                    errs.append(np.linalg.norm(p[:2] / p[2] - self._normalized[i][f]) * self.intr.focal)
            if (not errs or min(depths) <= 1e-6 or np.mean(errs) > self.cfg.max_reprojection_px):
                removed.append(t)
        for t in removed:
            del self.points[t]
            self.has_point[t] = False
        return len(removed)

    # -- the driver ----------------------------------------------------------

    def run(self, homography_ratios) -> None:
        if self.choose_initial_pair(homography_ratios) is None:
            raise RuntimeError("no suitable initial image pair: the scene may be planar or the matches too weak")
        print(f"[sfm] triangulated {self.triangulate_new_points()} seed points")
        self.run_bundle_adjustment(iterations=30, optimize_focal=self.cfg.optimize_focal)

        while True:
            image = self.register_next()
            if image is None:
                break
            if image < 0:
                continue
            self.triangulate_new_points()
            if len(self.poses) >= self.cfg.ba_growth * self._last_ba_size:
                print()
                self.run_bundle_adjustment(iterations=20, optimize_focal=self.cfg.optimize_focal)
                self.filter_observations()
                self.filter_points()
                self.triangulate_new_points()
        print()

        print("[sfm] final global bundle adjustment")
        self.run_bundle_adjustment(max_points=1 << 30, iterations=60, optimize_focal=self.cfg.optimize_focal)
        print(f"[sfm] pruned {self.filter_observations()} stray observations, {self.filter_points()} points")
        if self.drop_weak_cameras():
            self.filter_observations()
            self.filter_points()
        self.run_bundle_adjustment(max_points=1 << 30, iterations=60, optimize_focal=self.cfg.optimize_focal)

    # -- output --------------------------------------------------------------

    def to_reconstruction(self, image_names: list[str]) -> Reconstruction:
        track_ids = sorted(self.points)
        xyz       = np.array([self.points[t] for t in track_ids])
        # Mean color over the observations: averaging across views cancels most of the shading and exposure variation.
        rgb = np.array([
            np.mean([self.col[i][f] for i, f in self.tracks[t]], axis=0)
            for t in track_ids])

        image_ids = sorted(self.poses)
        return Reconstruction(
            image_names=[image_names[i] for i in image_ids],
            width=self.width, height=self.height,
            focal=self.intr.focal, principal=self.intr.principal.copy(),
            k1=self.intr.k1, k2=self.intr.k2,
            R=np.stack([self.poses[i][0] for i in image_ids]),
            t=np.stack([self.poses[i][1] for i in image_ids]),
            points=xyz, colors=np.clip(rgb, 0.0, 1.0))


# ------------------------------------------------------------------------------------------------------------------------------------------
# 9. Video in, posed cameras out
# ------------------------------------------------------------------------------------------------------------------------------------------

def reconstruct(frames_dir: str | Path, image_names: list[str],
                cfg: SfMConfig | None = None,
                device: str = "cuda") -> Reconstruction:
    """Run the whole SfM pipeline over a directory of keyframes."""
    cfg        = cfg or SfMConfig()
    frames_dir = Path(frames_dir)
    paths      = [frames_dir / n for n in image_names]

    kp, desc, col, size = detect_features(paths, cfg.max_features)

    device   = device if torch.cuda.is_available() else "cpu"
    gpu_desc = [torch.from_numpy(d).to(device) for d in desc]

    pairs = candidate_pairs(len(paths), cfg.match_window, cfg.subset_stride)
    print(f"[sfm] matching {len(pairs)} image pairs on {device}")
    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    homography_ratios: dict[tuple[int, int], float] = {}
    K_guess = np.array([[cfg.focal_prior * size[0], 0, size[0] / 2], [0, cfg.focal_prior * size[0], size[1] / 2], [0, 0, 1.0]])
    for n, (i, j) in enumerate(pairs):
        m = match_pair(gpu_desc[i], gpu_desc[j], cfg.ratio)
        inliers, h_ratio = verify_pair(kp[i], kp[j], m, K_guess, cfg.ransac_px)
        if inliers is not None:
            pair_matches[(i, j)]      = inliers
            homography_ratios[(i, j)] = h_ratio
        if n % 50 == 0:
            print(f"  {n}/{len(pairs)} pairs, {len(pair_matches)} verified", end="\r")
    counts = np.array([len(m) for m in pair_matches.values()])
    print(f"[sfm] verified {len(pair_matches)}/{len(pairs)} pairs "
          f"(median {int(np.median(counts))} inliers per pair)")

    tracks, track_of = build_tracks([len(k) for k in kp], pair_matches, cfg.min_track_length)

    sfm = IncrementalSfM(kp, desc, col, size, tracks, track_of, pair_matches, cfg)
    sfm.run(homography_ratios)
    rec = sfm.to_reconstruction(image_names)
    print(f"[sfm] reconstruction: {len(rec.image_names)}/{len(paths)} images, "
          f"{len(rec.points)} points, focal {rec.focal:.1f} px "
          f"({rec.fov_x_deg:.1f} deg horizontal FOV), "
          f"k1 {rec.k1:+.4f}, k2 {rec.k2:+.4f}")
    return rec


# ------------------------------------------------------------------------------------------------------------------------------------------
# 10. The Gaussian model: parameters, activations, adaptive density
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# A splat scene is a list of primitives, each carrying:
#
#     position    mu      (3)   where it is
#     scale       s       (3)   how big it is along its own three axes
#     rotation    q       (4)   which way those axes point (a quaternion)
#     opacity     alpha   (1)   how much light it blocks
#     color      SH     (16x3) view-dependent color, spherical harmonics
#
# 59 free numbers per Gaussian. There is no network: the scene *is* the parameters, and training is gradient descent on them. Two mechanisms
# follow from that.
#
# ACTIVATIONS. A scale must be positive, an opacity must lie in [0, 1], a quaternion must be unit length. Rather than project after each
# step we store an unconstrained parameter and apply exp, sigmoid or normalization on use, which leaves gradient descent no boundaries to
# bump into and makes the step size multiplicative -- a Gaussian a hundredth the size of another moves proportionally rather than
# absolutely.
#
# ADAPTIVE DENSITY CONTROL. The initialization is a sparse SfM cloud: far too few points, placed where SIFT found texture rather than where
# the scene needs them, and gradient descent cannot fix that because a Gaussian cannot split itself. So every few hundred iterations the
# population is edited. A Gaussian whose screen-space position keeps receiving a large gradient is one the optimizer keeps trying to move to
# fix an error that moving cannot fix; if it is small it is CLONED, if it is large it is SPLIT. Transparent and runaway Gaussians are
# deleted. All opacities are also pushed back toward zero periodically, which makes every splat re-earn its place -- otherwise floaters
# near the cameras accumulate and never leave.


# Spherical harmonics band 0: a constant. Converting an RGB color to the DC coefficient (and back) is a division by this.
SH_C0 = 0.28209479177387814

SH_C1 = 0.4886025119029199
SH_C2 = np.array([1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396])
SH_C3 = np.array([-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                  0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                  -0.5900435899266435])


def eval_sh(degree: int, coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics. `coeffs` (N, K, 3), `dirs` (N, 3) unit.

    Band 0 alone is a flat Lambertian color; each further band adds an angular lobe, which is what lets a highlight move with the camera.
    Degree 3 (16 coefficients) is the standard choice.
    """
    result = SH_C0 * coeffs[:, 0]
    if degree == 0:
        return result
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result  = (result - SH_C1 * y * coeffs[:, 1] + SH_C1 * z * coeffs[:, 2] - SH_C1 * x * coeffs[:, 3])
    if degree == 1:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (result
              + SH_C2[0] * xy * coeffs[:, 4]
              + SH_C2[1] * yz * coeffs[:, 5]
              + SH_C2[2] * (2.0 * zz - xx - yy) * coeffs[:, 6]
              + SH_C2[3] * xz * coeffs[:, 7]
              + SH_C2[4] * (xx - yy) * coeffs[:, 8])
    if degree == 2:
        return result
    return (result
            + SH_C3[0] * y * (3.0 * xx - yy) * coeffs[:, 9]
            + SH_C3[1] * xy * z * coeffs[:, 10]
            + SH_C3[2] * y * (4.0 * zz - xx - yy) * coeffs[:, 11]
            + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[:, 12]
            + SH_C3[4] * x * (4.0 * zz - xx - yy) * coeffs[:, 13]
            + SH_C3[5] * z * (xx - yy) * coeffs[:, 14]
            + SH_C3[6] * x * (xx - 3.0 * yy) * coeffs[:, 15])


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


class GaussianModel(nn.Module):
    """The scene. Every attribute is a directly optimized tensor."""

    def __init__(self, max_sh_degree: int = 3):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        # Bands are introduced one at a time (see `raise_sh_degree`): fit all 48 coefficients from iteration 1 and the model explains
        # geometry errors as view-dependent color, permanently.
        self.active_sh_degree = 0

        self.xyz           = nn.Parameter(torch.zeros(0, 3))
        self.scaling       = nn.Parameter(torch.zeros(0, 3))      # log scale
        self.rotation      = nn.Parameter(torch.zeros(0, 4))      # quaternion
        self.opacity       = nn.Parameter(torch.zeros(0, 1))      # logit
        self.features_dc   = nn.Parameter(torch.zeros(0, 1, 3))
        self.features_rest = nn.Parameter(torch.zeros(0, 15, 3))

        # Densification statistics, accumulated between edits.
        self.register_buffer("grad_accum", torch.zeros(0))
        self.register_buffer("grad_denom", torch.zeros(0))
        self.register_buffer("max_radius", torch.zeros(0))
        self.spatial_scale = 1.0

    # -- derived quantities --------------------------------------------------

    @property
    def n(self) -> int:
        return self.xyz.shape[0]

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)

    def get_features(self) -> torch.Tensor:
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def colors(self, camera_center: torch.Tensor) -> torch.Tensor:
        """View-dependent RGB for the current camera. Returns (N, 3).

        Clamped at zero, not at one: a Gaussian is allowed to be brighter than white so that a stack of semi-transparent splats can still
        composite to a saturated highlight.
        """
        dirs = self.xyz.detach() - camera_center
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rgb  = eval_sh(self.active_sh_degree, self.get_features(), dirs) + 0.5
        return rgb.clamp_min(0.0)

    # -- initialization ------------------------------------------------------

    def initialize(self, points: np.ndarray, colors: np.ndarray,
                   spatial_scale: float, random_points: int = 0,
                   max_initial_size: float = 0.01, device: str = "cuda",
                   seed: int = 0) -> None:
        """Seed the model from the SfM point cloud.

        `spatial_scale` is the radius of the camera rig; it sets the units for the position learning rate and the size thresholds below, so
        that the same hyperparameters work on a tabletop object and on a building.

        `random_points` scatters extra Gaussians uniformly inside the scene, because SfM returns points only where it found texture and
        densification can only subdivide what exists. Inside, not on a shell around it: distant Gaussians project to enormous splats that
        blanket every image in fog, and the fog absorbs the gradients real geometry needs.

        `max_initial_size` caps the starting radius at a fraction of the spatial scale -- in a cloud this uneven, nearest-neighbor spacing
        alone would start the isolated points out meters wide.
        """
        rng = np.random.default_rng(seed)
        self.spatial_scale = float(spatial_scale)

        xyz = np.asarray(points, np.float32)
        rgb = np.asarray(colors, np.float32)
        if random_points > 0:
            lo, hi = np.percentile(xyz, [1.0, 99.0], axis=0)
            extra  = rng.uniform(lo, hi, (random_points, 3)).astype(np.float32)
            xyz    = np.vstack([xyz, extra])
            rgb    = np.vstack([rgb, np.full((random_points, 3), 0.5, np.float32)])

        # A Gaussian should start about as big as the gap it has to cover: much smaller is a cloud of invisible dots with no gradient, much
        # larger is a fog that takes thousands of iterations to sharpen.
        dist2  = np.maximum(mean_neighbor_distance_squared(xyz), 1e-8)
        sigma  = np.minimum(np.sqrt(dist2), max_initial_size * self.spatial_scale)
        scales = np.log(sigma)[:, None].repeat(3, axis=1)

        quats       = np.zeros((len(xyz), 4), np.float32)
        quats[:, 0] = 1.0                                  # identity rotation

        features       = np.zeros((len(xyz), 16, 3), np.float32)
        features[:, 0] = (rgb - 0.5) / SH_C0

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        self.xyz = nn.Parameter(t(xyz))
        self.scaling = nn.Parameter(t(scales.astype(np.float32)))
        self.rotation = nn.Parameter(t(quats))
        # Start nearly transparent: opacity is the one parameter that can remove a Gaussian, so every splat kept has been justified.
        self.opacity = nn.Parameter(inverse_sigmoid(
            0.1 * torch.ones(len(xyz), 1, device=device)))
        self.features_dc   = nn.Parameter(t(features[:, :1]))
        self.features_rest = nn.Parameter(t(features[:, 1:]))
        self._reset_stats(device)
        print(f"[gaussians] initialized {self.n} Gaussians "
              f"({len(points)} from SfM + {random_points} random); "
              f"spatial scale {self.spatial_scale:.3f}, "
              f"median size {np.exp(np.median(scales)):.4f}")

    def _reset_stats(self, device=None) -> None:
        device = device or self.xyz.device
        z = lambda: torch.zeros(self.n, device=device)
        self.grad_accum, self.grad_denom, self.max_radius = z(), z(), z()

    def raise_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # -- optimizer -----------------------------------------------------------

    def make_optimizer(self, lr_scale: float = 1.0) -> torch.optim.Adam:
        """One Adam, six parameter groups, six very different learning rates.

        The rates are not arbitrary. Positions are in world units, so their rate is scaled by the size of the scene. Opacity moves through a
        sigmoid and needs a large rate to cross it. The higher SH bands get a rate 20x smaller than the DC term, which keeps view-dependent
        color from racing ahead of the base color and papering over geometry.
        """
        s = self.spatial_scale * lr_scale
        return torch.optim.Adam([
            {"params": [self.xyz], "lr": 1.6e-4 * s, "name": "xyz"},
            {"params": [self.features_dc], "lr": 2.5e-3, "name": "f_dc"},
            {"params": [self.features_rest], "lr": 2.5e-3 / 20.0,
             "name": "f_rest"},
            {"params": [self.opacity], "lr": 0.05, "name": "opacity"},
            {"params": [self.scaling], "lr": 5e-3, "name": "scaling"},
            {"params": [self.rotation], "lr": 1e-3, "name": "rotation"},
        ], eps=1e-15)

    def position_lr(self, step: int, total: int) -> float:
        """Exponentially decayed position learning rate.

        Positions start free to move and end nearly frozen: late in training only appearance should still be refining, and positions that
        keep moving at the initial rate turn sharp edges back into mush.
        """
        f = min(max(step / max(total, 1), 0.0), 1.0)
        return 1.6e-4 * self.spatial_scale * (0.01 ** f)

    # -- densification statistics --------------------------------------------

    def record_gradients(self, mean2d: torch.Tensor, visible: torch.Tensor,
                         radius: torch.Tensor) -> None:
        """Accumulate the screen-space position gradient for visible splats.

        |dL/d(screen position)| answers "how much would the loss improve if
        this Gaussian moved a pixel?". A large sustained value means it wants to be in two places at once -- which more primitives would
        resolve.
        """
        if mean2d.grad is None:
            return
        with torch.no_grad():
            g = mean2d.grad[visible].norm(dim=1)
            self.grad_accum[visible] += g
            self.grad_denom[visible] += 1.0
            self.max_radius[visible] = torch.maximum(self.max_radius[visible], radius[visible])

    # -- editing the population ----------------------------------------------

    def _replace_tensors(self, optimizer: torch.optim.Adam,
                         new: dict[str, torch.Tensor]) -> None:
        """Swap parameters and carry Adam's state across the change.

        Adam keeps two running moments per parameter element, and adding or removing Gaussians leaves those buffers misaligned. Rebuilding
        the optimizer would discard every survivor's momentum and show as a hitch after each densification, so we edit the state in place:
        keep the moments of survivors, zero them for newcomers.
        """
        for group in optimizer.param_groups:
            name = group["name"]
            if name not in new:
                continue
            old         = group["params"][0]
            state       = optimizer.state.get(old, None)
            replacement = nn.Parameter(new[name].contiguous().requires_grad_(True))
            if state is not None:
                del optimizer.state[old]
                optimizer.state[replacement] = state
            group["params"][0] = replacement
            setattr(self, {"f_dc": "features_dc", "f_rest": "features_rest"}.get(name, name), replacement)

    def _prune(self, optimizer, keep: torch.Tensor) -> None:
        for group in optimizer.param_groups:
            old   = group["params"][0]
            state = optimizer.state.get(old, None)
            if state is not None:
                state["exp_avg"]    = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
        self._replace_tensors(optimizer, {
            "xyz": self.xyz.data[keep], "f_dc": self.features_dc.data[keep],
            "f_rest": self.features_rest.data[keep],
            "opacity": self.opacity.data[keep],
            "scaling": self.scaling.data[keep],
            "rotation": self.rotation.data[keep]})
        self.grad_accum = self.grad_accum[keep]
        self.grad_denom = self.grad_denom[keep]
        self.max_radius = self.max_radius[keep]

    def _append(self, optimizer, new: dict[str, torch.Tensor]) -> None:
        for group in optimizer.param_groups:
            old   = group["params"][0]
            extra = new[group["name"]]
            state = optimizer.state.get(old, None)
            if state is not None:
                z = torch.zeros_like(extra)
                state["exp_avg"] = torch.cat([state["exp_avg"], z], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(extra)], dim=0)
        self._replace_tensors(optimizer, {
            k: torch.cat([getattr(self, {"f_dc": "features_dc",
                                         "f_rest": "features_rest"}
                                  .get(k, k)).data, v], dim=0)
            for k, v in new.items()})
        self._reset_stats()

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold: float,
                          min_opacity: float, max_screen_radius: float | None,
                          percent_dense: float = 0.01) -> dict[str, int]:
        """One round of adaptive density control. Returns a small report."""
        grads = self.grad_accum / self.grad_denom.clamp_min(1.0)
        grads[self.grad_denom == 0] = 0.0
        scales = self.get_scaling()
        big = scales.max(dim=1).values > percent_dense * self.spatial_scale
        wants_help = grads >= grad_threshold

        n_before = self.n
        n_clone  = int((wants_help & ~big).sum())
        n_split  = int((wants_help & big).sum())

        # --- clone: a small, over-worked Gaussian gets a twin ---------------
        # The copy is identical, position included, but the two are no longer tied: the next step pushes them apart and the region ends up
        # with twice the capacity.
        sel = torch.nonzero(wants_help & ~big, as_tuple=True)[0]
        clones = {"xyz": self.xyz.data[sel],
                  "f_dc": self.features_dc.data[sel],
                  "f_rest": self.features_rest.data[sel],
                  "opacity": self.opacity.data[sel],
                  "scaling": self.scaling.data[sel],
                  "rotation": self.rotation.data[sel]}

        # --- split: a large, over-worked Gaussian becomes two smaller ones ---
        # Children are sampled from the parent's own distribution and shrunk by 1.6 (the paper's factor), which covers the parent's
        # footprint without leaving a seam.
        sel_s = torch.nonzero(wants_help & big, as_tuple=True)[0]
        k     = 2
        if sel_s.numel() > 0:
            std     = scales[sel_s].repeat(k, 1)
            offset  = torch.normal(torch.zeros_like(std), std)
            rot     = quaternion_to_matrix(self.rotation.data[sel_s].repeat(k, 1))
            new_xyz = (torch.bmm(rot, offset.unsqueeze(2)).squeeze(2) + self.xyz.data[sel_s].repeat(k, 1))
            splits = {"xyz": new_xyz,
                      "f_dc": self.features_dc.data[sel_s].repeat(k, 1, 1),
                      "f_rest": self.features_rest.data[sel_s].repeat(k, 1, 1),
                      "opacity": self.opacity.data[sel_s].repeat(k, 1),
                      "scaling": torch.log(scales[sel_s].repeat(k, 1) / (0.8 * k)),
                      "rotation": self.rotation.data[sel_s].repeat(k, 1)}
        else:
            splits = {key: value[:0] for key, value in clones.items()}

        # `_append` resets the statistics, so keep the screen radii we need for pruning; the newcomers have not been rendered yet and get 0.
        old_radius = self.max_radius.clone()
        self._append(optimizer, {key: torch.cat([clones[key], splits[key]]) for key in clones})
        radius = torch.zeros(self.n, device=old_radius.device)
        radius[:old_radius.numel()] = old_radius

        # A split parent is redundant once its children exist.
        keep = torch.ones(self.n, dtype=torch.bool, device=self.xyz.device)
        if sel_s.numel() > 0:
            keep[sel_s] = False

        keep &= (self.get_opacity().squeeze(1) > min_opacity)
        keep &= (self.get_scaling().max(dim=1).values
                 < 0.5 * self.spatial_scale)      # runaway blobs
        if max_screen_radius is not None:
            keep &= radius < max_screen_radius
        self._prune(optimizer, keep)

        return {"before": n_before, "cloned": n_clone, "split": n_split, "pruned": int((~keep).sum()) - n_split, "after": self.n}

    @torch.no_grad()
    def reset_opacity(self, optimizer, value: float = 0.01) -> None:
        """Push every opacity down; the useful splats will climb back.

        Floaters are the characteristic artifact: a blob near one camera that explains a few of its pixels and is invisible to every other
        view, so nothing penalizes it. Capping all opacities makes every Gaussian re-earn its own; a floater, backed by one view, falls
        below the pruning threshold and disappears.
        """
        new = torch.min(self.get_opacity(), torch.full_like(self.opacity, value))
        for group in optimizer.param_groups:
            if group["name"] != "opacity":
                continue
            state = optimizer.state.get(group["params"][0], None)
            if state is not None:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
        self._replace_tensors(optimizer, {"opacity": inverse_sigmoid(new.clamp(1e-6, 1 - 1e-6))})


def mean_neighbor_distance_squared(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean squared distance to the k nearest other points, per point."""
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)        # column 0 is the point itself
    return (d[:, 1:] ** 2).mean(axis=1)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 11. A differentiable tile rasterizer
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# Given N anisotropic Gaussians and a camera, produce an image -- and be differentiable with respect to every Gaussian parameter. Every
# gradient that trains the scene flows back through here. Following Kerbl et al. (2023):
#
#   1. PROJECT. The perspective divide is nonlinear, so the 3D covariance
#      cannot simply be transformed; the EWA Jacobian J of the projection,
#      evaluated at the center, linearizes it: Sigma_2D = J R Sigma R^T J^T.
#      Each Gaussian lands as an ellipse rather than a head-on point sprite.
#   2. BIN into the 16x16 pixel tiles its 3-sigma ellipse touches, so a pixel
#      only consults Gaussians from its own tile and shares them with its 255
#      neighbors.
#   3. SORT by depth once per tile, not per pixel. That is the approximation
#      that buys splatting its speed; it is visible only where Gaussians in
#      one tile intersect each other.
#   4. BLEND front to back with transmittance T_i = prod_{j<i} (1 - alpha_j).
#
# Step 4 is where pure PyTorch has to be clever, because the reference CUDA kernel walks the sorted list per pixel. A prefix product is a
# prefix sum in log space:
#
#     T_i = exp( sum_{j<i} log(1 - alpha_j) )
#
# and `cumsum` is one differentiable call. That substitution is the reason this is 200 lines of PyTorch instead of 1000 lines of CUDA. The
# price is memory -- one entry per (Gaussian, pixel) pair, tens of millions of them -- paid with tile chunking and gradient checkpointing.


TILE         = 16             # pixels per tile edge; 256 pixels per tile
SIGMA_CUTOFF = 3.0            # ellipse radius in standard deviations
DILATION     = 0.3            # low-pass filter, in pixels^2 (see below)


# -- Camera ----------------------------------------------------------------

@dataclass
class Camera:
    """A pinhole view. Same convention as the reconstruction: x_cam = R x + t."""
    R: torch.Tensor             # (3, 3) world -> camera
    t: torch.Tensor             # (3,)
    focal: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.01

    @property
    def center(self) -> torch.Tensor:
        """Camera position in world space, C = -R^T t."""
        return -self.R.transpose(0, 1) @ self.t

    def to(self, device) -> "Camera":
        return Camera(self.R.to(device), self.t.to(device), self.focal, self.cx, self.cy, self.width, self.height, self.near)


# -- Geometry: 3D covariance, projection, screen-space ellipse -------------

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices.

    Normalized on use rather than in the optimizer, so gradient steps never have to be projected back onto the unit sphere.
    """
    q          = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def covariance_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Sigma = R S S^T R^T for each Gaussian. Returns (N, 3, 3).

    Factoring as rotation-times-scale keeps the optimization well posed: a covariance must stay symmetric positive semi-definite and descent
    on its six entries would wander out of that set, whereas any (R, S) pair gives a valid covariance.
    """
    M = quaternion_to_matrix(quats) * scales.unsqueeze(1)   # R @ diag(s)
    return M @ M.transpose(1, 2)


def project_gaussians(means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, cam: Camera):
    """World-space Gaussians -> screen-space ellipses.

    Returns (mean2d, conic, depth, radius, visible) where `conic` is the inverse 2D covariance -- the quadratic form actually evaluated per
    pixel -- and `radius` is the 3-sigma extent in pixels.
    """
    p       = means @ cam.R.transpose(0, 1) + cam.t    # camera space
    depth   = p[:, 2]
    visible = depth > cam.near

    z      = depth.clamp_min(cam.near)
    mean2d = torch.stack([cam.focal * p[:, 0] / z + cam.cx, cam.focal * p[:, 1] / z + cam.cy], dim=1)

    # EWA splatting: linearize the perspective divide at the Gaussian center. J = d(u, v)/d(x, y, z) evaluated there; combined with the
    # world->camera rotation it maps the 3D covariance to the image plane.
    #
    # The clamp is not optional: for a Gaussian far off to the side the Jacobian's off-diagonal terms grow without bound, the projected
    # covariance explodes, and the splat smears over the image as a bright haze. Clamping x/z and y/z to just outside the frustum (1.3, as
    # in the reference) evaluates the Jacobian where it still means something and leaves everything inside the frustum untouched.
    f     = cam.focal
    lim_x = 1.3 * 0.5 * cam.width / f
    lim_y = 1.3 * 0.5 * cam.height / f
    tx    = torch.clamp(p[:, 0] / z, -lim_x, lim_x) * z
    ty    = torch.clamp(p[:, 1] / z, -lim_y, lim_y) * z

    J          = torch.zeros(means.shape[0], 2, 3, dtype=means.dtype, device=means.device)
    J[:, 0, 0] = f / z
    J[:, 1, 1] = f / z
    J[:, 0, 2] = -f * tx / z ** 2
    J[:, 1, 2] = -f * ty / z ** 2
    T          = J @ cam.R                              # (N, 2, 3)
    cov2d      = T @ covariance_3d(scales, quats) @ T.transpose(1, 2)

    # Dilation: add a fraction of a pixel of variance to both axes. Without it a Gaussian smaller than a pixel becomes a spike that lands
    # between the sample points -- it flickers as the camera moves and its gradient is almost always zero. This is a low-pass prefilter,
    # exactly as in EWA.
    a = cov2d[:, 0, 0] + DILATION
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + DILATION

    det   = (a * c - b * b).clamp_min(1e-9)
    conic = torch.stack([c / det, -b / det, a / det], dim=1)   # (N, 3): xx,xy,yy

    # 3 sigma along the major axis of the ellipse: the larger eigenvalue of a symmetric 2x2 matrix has a closed form.
    mid    = 0.5 * (a + c)
    disc   = (mid * mid - det).clamp_min(0.0).sqrt()
    radius = SIGMA_CUTOFF * (mid + disc).sqrt()

    visible = visible & (det > 1e-9) & (radius > 0.5)
    # cheap frustum reject: does the ellipse's bounding box touch the image?
    visible &= ((mean2d[:, 0] + radius > 0) & (mean2d[:, 0] - radius < cam.width)
                & (mean2d[:, 1] + radius > 0)
                & (mean2d[:, 1] - radius < cam.height))
    return mean2d, conic, depth, radius, visible


# -- Binning: which Gaussians touch which tiles, sorted by depth -----------

def bin_to_tiles(mean2d: torch.Tensor, radius: torch.Tensor,
                 depth: torch.Tensor, visible: torch.Tensor,
                 grid_w: int, grid_h: int):
    """Build the depth-sorted (tile, Gaussian) work list.

    Returns (gaussian_ids, tile_offsets, tile_counts). `gaussian_ids` is one flat array in which every tile's Gaussians are contiguous and
    ordered front to back.

    Pure index arithmetic under `no_grad`: which Gaussians to evaluate is a discrete decision, and gradients flow through the values
    evaluated later, not through the choice.
    """
    idx = torch.nonzero(visible, as_tuple=True)[0]
    if idx.numel() == 0:
        empty  = torch.zeros(0, dtype=torch.long, device=mean2d.device)
        counts = torch.zeros(grid_w * grid_h, dtype=torch.long, device=mean2d.device)
        return empty, counts, counts

    m, r = mean2d[idx], radius[idx]
    x0   = ((m[:, 0] - r) / TILE).floor().clamp(0, grid_w - 1).long()
    x1   = ((m[:, 0] + r) / TILE).floor().clamp(0, grid_w - 1).long()
    y0   = ((m[:, 1] - r) / TILE).floor().clamp(0, grid_h - 1).long()
    y1   = ((m[:, 1] + r) / TILE).floor().clamp(0, grid_h - 1).long()

    nx, ny    = x1 - x0 + 1, y1 - y0 + 1
    per_gauss = nx * ny
    total     = int(per_gauss.sum())

    # Expand each Gaussian into one entry per tile it covers, without a loop: repeat_interleave gives the Gaussian index, and a local
    # counter turned into (dx, dy) gives the tile.
    gid    = torch.repeat_interleave(idx, per_gauss)
    starts = torch.cumsum(per_gauss, 0) - per_gauss
    local  = (torch.arange(total, device=mean2d.device) - torch.repeat_interleave(starts, per_gauss))
    nx_rep = torch.repeat_interleave(nx, per_gauss)
    tx     = torch.repeat_interleave(x0, per_gauss) + local % nx_rep
    ty     = torch.repeat_interleave(y0, per_gauss) + local // nx_rep
    tile   = ty * grid_w + tx

    # Sort by depth, then stably by tile: each tile's slice ends up ordered front to back, which is what the blend below assumes.
    by_depth = torch.argsort(depth[gid])
    order    = by_depth[torch.argsort(tile[by_depth], stable=True)]

    counts  = torch.bincount(tile, minlength=grid_w * grid_h)
    offsets = torch.cumsum(counts, 0) - counts
    return gid[order], offsets, counts


# -- Blending --------------------------------------------------------------

def _blend_chunk(mean2d, conic, opacity, colors, gauss_ids, valid, pixels, tally=None):
    """Alpha-composite one batch of tiles.

    Shapes: `gauss_ids` and `valid` are (T, G) -- T tiles, padded to G Gaussians each -- and `pixels` is (T, P, 2) with P = TILE^2. The
    result is (T, P, 3) color and (T, P) remaining transmittance.

    `tally`, if given, accumulates per Gaussian the weight it contributed to this image, which is what `prune_unsupported` measures.
    Only valid under `no_grad`: with checkpointing the chunk runs twice and would be counted twice.
    """
    mu     = mean2d[gauss_ids]                               # (T, G, 2)
    d      = pixels.unsqueeze(1) - mu.unsqueeze(2)           # (T, G, P, 2)
    cx, cy = d[..., 0], d[..., 1]
    q      = conic[gauss_ids]                                # (T, G, 3)
    power  = -0.5 * (q[..., 0:1] * cx * cx + 2.0 * q[..., 1:2] * cx * cy + q[..., 2:3] * cy * cy)

    alpha = (opacity.reshape(-1)[gauss_ids].unsqueeze(2) * power.exp()).clamp(max=0.99)
    alpha = alpha * valid.unsqueeze(2)

    # Front-to-back transmittance as an exclusive prefix product, evaluated in log space so that `cumsum` can do the work. The 1e-7 keeps
    # the log finite for a fully opaque Gaussian.
    log_one_minus = torch.log((1.0 - alpha).clamp_min(1e-7))
    trans         = torch.exp(torch.cumsum(log_one_minus, dim=1) - log_one_minus)

    weight    = alpha * trans                                # (T, G, P)
    if tally is not None:
        tally.index_add_(0, gauss_ids.reshape(-1), weight.sum(dim=2).reshape(-1))
    color     = torch.einsum("tgp,tgc->tpc", weight, colors[gauss_ids])
    remaining = torch.exp(log_one_minus.sum(dim=1))
    return color, remaining


def rasterize(means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor,
              opacity: torch.Tensor, colors: torch.Tensor, cam: Camera,
              background: torch.Tensor | None = None,
              max_elements: int = 1 << 23,
              use_checkpoint: bool = True,
              tally: torch.Tensor | None = None):
    """Render Gaussians into an image. Returns (image, aux).

    `image` is (H, W, 3). `aux` carries what the densification heuristics in section 10 need: the screen-space means (with gradients
    retained), the per-Gaussian pixel radius, and the visibility mask.
    """
    device = means.device
    grid_w = (cam.width + TILE - 1) // TILE
    grid_h = (cam.height + TILE - 1) // TILE

    mean2d, conic, depth, radius, visible = project_gaussians(means, scales, quats, cam)
    # The densification signal is dL/d(screen position); retaining it here costs no extra backward pass.
    if mean2d.requires_grad:
        mean2d.retain_grad()

    with torch.no_grad():
        gauss_ids, offsets, counts = bin_to_tiles(mean2d.detach(), radius.detach(), depth.detach(), visible, grid_w, grid_h)

    n_tiles  = grid_w * grid_h
    n_pixels = TILE * TILE
    bg       = (background if background is not None else torch.zeros(3, dtype=means.dtype, device=device))

    aux = {"mean2d": mean2d, "radius": radius, "visible": visible, "n_rendered": int(gauss_ids.numel())}

    def assemble(tile_colors: torch.Tensor) -> torch.Tensor:
        """(n_tiles, P, 3) tiles -> (H, W, 3) image."""
        return (tile_colors.view(grid_h, grid_w, TILE, TILE, 3)
                .permute(0, 2, 1, 3, 4)
                .reshape(grid_h * TILE, grid_w * TILE, 3)
                [:cam.height, :cam.width])

    empty = bg.expand(n_tiles, n_pixels, 3).contiguous()
    if gauss_ids.numel() == 0:
        return assemble(empty), aux

    # Pixel coordinates inside a tile, shared by every tile (we add the tile's own offset per chunk). Sample at pixel CENTERS.
    ar = torch.arange(TILE, device=device, dtype=means.dtype)
    local_y, local_x = torch.meshgrid(ar, ar, indexing="ij")
    local = torch.stack([local_x.reshape(-1), local_y.reshape(-1)],
                        dim=1) + 0.5                          # (P, 2)

    # Process tiles in chunks, grouped by how many Gaussians they contain. Every chunk is padded to its busiest tile, so sorting first means
    # a nearly-empty tile is never padded out to match a crowded one -- on a typical view this halves the work.
    busy         = torch.nonzero(counts > 0, as_tuple=True)[0]
    busy         = busy[torch.argsort(counts[busy])]
    order_counts = counts[busy]

    def blend(ids: torch.Tensor, valid: torch.Tensor, pixels: torch.Tensor):
        args = (mean2d, conic, opacity, colors, ids, valid, pixels)
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(_blend_chunk, *args, use_reentrant=False)
        return _blend_chunk(*args, tally=tally)

    def tile_pixels(sel: torch.Tensor) -> torch.Tensor:
        tile_x = (sel % grid_w).to(means.dtype) * TILE
        tile_y = (sel // grid_w).to(means.dtype) * TILE
        return local.unsqueeze(0) + torch.stack([tile_x, tile_y], dim=1).unsqueeze(1)

    # The most Gaussians one tile can hold within the budget. A tile busier than this is blended in slices below, because a chunk is padded
    # to its busiest tile and a single crowded tile would otherwise blow the budget by any factor it liked: 293,000 entries in one tile of a
    # close-up view asks for 75 M elements against the 8.4 M this bounds it to.
    slice_limit = max(1, max_elements // n_pixels)

    rendered_tiles: list[torch.Tensor] = []
    rendered_ids: list[torch.Tensor] = []
    start = 0
    while start < busy.numel():
        if int(order_counts[start]) > slice_limit:
            # One tile, front to back, `slice_limit` Gaussians at a time. Each slice is attenuated by the transmittance the slices in front
            # of it left behind, and the transmittances multiply -- which is all that front-to-back compositing ever does.
            sel    = busy[start:start + 1]
            pixels = tile_pixels(sel)
            count  = int(order_counts[start])
            base   = int(offsets[sel])
            total  = torch.zeros(1, n_pixels, 3, dtype=means.dtype, device=device)
            trans  = torch.ones(1, n_pixels, dtype=means.dtype, device=device)
            for first in range(0, count, slice_limit):
                last  = base + min(first + slice_limit, count)
                ids   = gauss_ids[base + first:last].unsqueeze(0)
                valid = torch.ones_like(ids, dtype=means.dtype)
                color, remaining = blend(ids, valid, pixels)
                total = total + trans.unsqueeze(2) * color
                trans = trans * remaining
            rendered_tiles.append(total + trans.unsqueeze(2) * bg)
            rendered_ids.append(sel)
            start += 1
            continue

        # Grow the chunk until the padded tensor would exceed max_elements.
        n = 1
        while start + n < busy.numel():
            g = int(order_counts[start + n])
            if g > slice_limit or (n + 1) * g * TILE * TILE > max_elements:
                break
            n += 1
        sel   = busy[start:start + n]
        g_max = int(order_counts[start + n - 1])
        start += n

        cnt   = counts[sel]
        off   = offsets[sel]
        ar_g  = torch.arange(g_max, device=device)
        valid = (ar_g.unsqueeze(0) < cnt.unsqueeze(1))
        flat  = (off.unsqueeze(1) + ar_g.unsqueeze(0)).clamp_max(gauss_ids.numel() - 1)
        ids   = gauss_ids[flat]                              # (T, G)

        color, remaining = blend(ids, valid.to(means.dtype), tile_pixels(sel))
        rendered_tiles.append(color + remaining.unsqueeze(2) * bg)
        rendered_ids.append(sel)

    # One scatter for the whole image: tiles nobody touched keep the background, and autograd sees a single index_copy instead of a chain.
    tile_colors = empty.index_copy(0, torch.cat(rendered_ids),
                                   torch.cat(rendered_tiles, dim=0))
    return assemble(tile_colors), aux


# ------------------------------------------------------------------------------------------------------------------------------------------
# 12. Training: fitting Gaussians to the posed images
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# Pick a training view, render it, compare with the photograph, backpropagate, step, repeat a few thousand times. There is no network: the
# scene parameters are the model. Three details do the real work.
#
# THE LOSS IS NOT JUST L1. A pixel loss is happy with a slightly blurry image. D-SSIM compares local means, variances and covariance over an
# 11x11 window, so it notices an edge that has become a gradient. The paper's 0.8 / 0.2 mix keeps L1's well-conditioned gradients while
# making blur expensive.
#
# SH BANDS ARE UNLOCKED GRADUALLY. Handed all 48 color coefficients at once, the optimizer explains a misplaced Gaussian as view-dependent
# color -- red from here, gray from there -- and bakes the geometry error into appearance.
#
# DENSITY IS EDITED ON A SCHEDULE, between iteration 500 and the halfway mark, after which the population is frozen and only the parameters
# refine. Growing the scene to the last moment leaves thousands of unconverged blobs.
#
# Evaluation holds out every eighth view. With enough primitives splatting can memorize its training images, so held-out PSNR is the only
# number that says a scene was reconstructed rather than a set of pictures.


# -- Views: posed images on the GPU ----------------------------------------

@dataclass
class View:
    camera: Camera
    image: torch.Tensor         # (H, W, 3) float32 in [0, 1]
    name: str


def load_views(rec: Reconstruction, frames_dir: str | Path, scale: float = 0.5, device: str = "cuda") -> list[View]:
    """Load every reconstructed frame at `scale` of its stored resolution.

    Resolution is the biggest lever on training time -- cost is linear in pixels, so half resolution is four times faster -- and it
    prefilters soft, noisy phone video that would otherwise be fitted noise and all.

    Images are undistorted here when the reconstruction carries distortion, so that the rasterizer's pure pinhole model is exactly right.
    """
    frames_dir = Path(frames_dir)
    intr       = rec.intrinsics
    distorted  = abs(rec.k1) > 1e-9 or abs(rec.k2) > 1e-9

    views: list[View] = []
    for i, name in enumerate(rec.image_names):
        bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"missing training image {name}")
        if distorted:
            # Keep the same K, so focal and principal point stay valid; the corners simply go black where the lens saw nothing.
            bgr = cv2.undistort(bgr, intr.K, intr.dist_coeffs, None, intr.K)
        h, w   = bgr.shape[:2]
        tw, th = int(round(w * scale)), int(round(h * scale))
        if (tw, th) != (w, h):
            bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
        cam = Camera(
            R=torch.from_numpy(rec.R[i]).float().to(device),
            t=torch.from_numpy(rec.t[i]).float().to(device),
            focal=rec.focal * scale,
            cx=float(rec.principal[0]) * scale,
            cy=float(rec.principal[1]) * scale,
            width=tw, height=th)
        views.append(View(camera=cam, image=rgb, name=name))
    print(f"[train] loaded {len(views)} views at {views[0].camera.width}"
          f"x{views[0].camera.height}"
          + (" (undistorted)" if distorted else ""))
    return views


# -- Losses ----------------------------------------------------------------

def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Structural similarity between two (H, W, 3) images, averaged.

    Local brightness, contrast and correlation over a sliding window. A blurred image keeps the first and loses the other two, which is
    exactly what a pixel-wise loss cannot see.
    """
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    pad = window.shape[-1] // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=3)
    mu_b = F.conv2d(b, window, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, window, padding=pad, groups=3) - mu_a2
    sb = F.conv2d(b * b, window, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, window, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))).mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.clamp(0, 1) - b) ** 2)
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


# -- Configuration ---------------------------------------------------------

@dataclass
class TrainConfig:
    iterations: int         = 7000
    resolution_scale: float = 0.5
    random_points: int      = 30000

    # Coarse to fine. Geometry converges perfectly well on half-resolution images, and at a quarter of the cost per iteration, but fine
    # texture cannot: a wood grain that a 640x360 target only hints at is a wood grain the Gaussians never learn. So the last stretch of
    # training reloads the photographs at `fine_resolution_scale` and sharpens what the cheap iterations built. The switch happens after
    # densification has stopped, so no threshold has to be reinterpreted at the new scale.
    fine_from_fraction: float    = 0.6
    fine_resolution_scale: float = 1.0

    lambda_dssim: float    = 0.2
    sh_increase_every: int = 1000

    densify_from: int = 500
    densify_until_fraction: float = 0.5     # of total iterations
    densify_every: int = 100
    opacity_reset_every: int = 3000
    # Threshold on the mean screen-space position gradient, expressed in the normalized device coordinates the reference implementation uses
    # so that the familiar 0.0002 is meaningful; converted to pixels below.
    grad_threshold_ndc: float = 2.0e-4
    min_opacity: float = 0.005
    percent_dense: float = 0.01
    # Of the image width. The reference implementation uses 0.05, which on this capture doubled the Gaussian count for the same PSNR: a
    # close-up view legitimately wants splats that wide, and deleting them only makes densification replace them with many small ones.
    max_screen_radius_fraction: float = 0.5

    holdout_every: int = 8
    eval_every: int    = 1000
    log_every: int     = 100
    # A long run must not lose everything to a crash in its last hour, so the model is written out at this cadence. Nothing reads these
    # files; they are insurance, and the final `splats.ply` is still written by the driver.
    checkpoint_every: int = 2500

    # What survives the last step of training: the share of contributed weight worth keeping, and how much of a Gaussian's contribution may
    # come from one view before it is treated as a floater. See `prune_unsupported`.
    keep_weight: float           = 0.999
    max_single_view_share: float = 0.9
    # A budget, and this capture spends all of it: densification is still adding Gaussians when the cap stops it, so the cap, not the
    # scene, decides the final count. Raising it buys detail at a roughly linear cost in time and memory.
    max_gaussians: int = 450_000
    seed: int          = 0


# -- The training loop -----------------------------------------------------

def train(rec: Reconstruction, frames_dir: str | Path, out_dir: str | Path,
          cfg: TrainConfig | None = None, device: str = "cuda") -> GaussianModel:
    cfg     = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    views = load_views(rec, frames_dir, cfg.resolution_scale, device)
    # Held-out views are never rendered for the loss, so their PSNR measures generalization to viewpoints the optimizer never saw.
    holdout = set(range(0, len(views), cfg.holdout_every)) \
        if cfg.holdout_every > 0 else set()
    train_ids = [i for i in range(len(views)) if i not in holdout]
    test_ids  = sorted(holdout)
    # The training column of the report is measured on as many views as the held-out column, for a fair comparison, but they have to be
    # spread over the capture: taking the first N instead means reporting whichever part of the video happens to come first, and on this
    # capture that is the hardest part of it -- several dB pessimistic, for no reason anyone reading the number could guess.
    train_sample = train_ids[::max(1, len(train_ids) // max(len(test_ids), 1))][:len(test_ids)]
    print(f"[train] {len(train_ids)} training views, {len(test_ids)} held out")

    # The 95th percentile, not the maximum: one mis-registered camera parked far outside the capture would otherwise set the
    # scale for every learning rate and size threshold below, and the whole scene would train as a blur.
    centers       = rec.camera_centers
    spatial_scale = float(np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 95.0))

    model = GaussianModel(max_sh_degree=3).to(device)
    model.initialize(rec.points, rec.colors, spatial_scale, random_points=cfg.random_points, device=device, seed=cfg.seed)
    optimizer = model.make_optimizer()

    window = _gaussian_window(11, 1.5, device)
    background = torch.zeros(3, device=device)
    width = views[0].camera.width
    grad_threshold = cfg.grad_threshold_ndc * 2.0 / width
    max_screen_radius = cfg.max_screen_radius_fraction * width
    densify_until = int(cfg.densify_until_fraction * cfg.iterations)

    rng = np.random.default_rng(cfg.seed)
    order: list[int] = []
    history: list[dict] = []
    t0 = time.time()
    running_loss = 0.0

    fine_from = int(cfg.fine_from_fraction * cfg.iterations)

    for step in range(1, cfg.iterations + 1):
        if cfg.fine_resolution_scale > cfg.resolution_scale and step == fine_from + 1:
            views = load_views(rec, frames_dir, cfg.fine_resolution_scale, device)
            width = views[0].camera.width
            grad_threshold    = cfg.grad_threshold_ndc * 2.0 / width
            max_screen_radius = cfg.max_screen_radius_fraction * width
            order.clear()

        # Every view once per epoch, in a fresh order. Sampling with replacement would leave some views unseen for long stretches.
        if not order:
            order = list(rng.permutation(train_ids))
        view = views[order.pop()]

        for group in optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = model.position_lr(step, cfg.iterations)
        if step % cfg.sh_increase_every == 0:
            model.raise_sh_degree()

        image, aux = rasterize(
            model.xyz, model.get_scaling(), model.rotation,
            model.get_opacity(), model.colors(view.camera.center),
            view.camera, background=background)

        l1   = torch.abs(image - view.image).mean()
        loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1.0 - ssim(image, view.image, window))

        loss.backward()
        running_loss += float(l1.detach())

        with torch.no_grad():
            if step <= densify_until:
                model.record_gradients(aux["mean2d"], aux["visible"], aux["radius"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (cfg.densify_from < step <= densify_until and step % cfg.densify_every == 0 and model.n < cfg.max_gaussians):
                report = model.densify_and_prune(
                    optimizer, grad_threshold, cfg.min_opacity,
                    max_screen_radius if step > cfg.opacity_reset_every else None,
                    cfg.percent_dense)
                print(f"  [{step:5d}] density: {report['before']} "
                      f"+{report['cloned']} cloned +{report['split']} split "
                      f"-{report['pruned']} pruned -> {report['after']}")
            if step % cfg.opacity_reset_every == 0 and step <= densify_until:
                model.reset_opacity(optimizer)
                print(f"  [{step:5d}] opacity reset")

        if step % cfg.log_every == 0:
            print(f"  [{step:5d}/{cfg.iterations}] L1 "
                  f"{running_loss / cfg.log_every:.4f}  "
                  f"{model.n} Gaussians  "
                  f"{step / (time.time() - t0):.1f} it/s", end="\r")
            running_loss = 0.0

        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0 and step != cfg.iterations:
            print()
            save_gaussian_ply(out_dir / f"splats_{step:06d}.ply", model)

        if step % cfg.eval_every == 0 or step == cfg.iterations:
            print()
            stats = evaluate(model, views, test_ids, train_sample, background)
            stats.update(step=step, n=model.n, width=views[0].camera.width, height=views[0].camera.height)
            history.append(stats)
            print(f"  [{step:5d}] PSNR train {stats['train_psnr']:.2f} dB, "
                  f"held-out {stats['test_psnr']:.2f} dB, "
                  f"{model.n} Gaussians, {views[0].camera.width}x{views[0].camera.height}, "
                  f"{time.time() - t0:.0f} s")

    print(f"[train] finished {cfg.iterations} iterations in {time.time() - t0:.0f} s ({model.n} Gaussians)")
    if cfg.keep_weight < 1.0:
        prune_unsupported(model, views, cfg.keep_weight, cfg.max_single_view_share)
    np.save(out_dir / "train_history.npy", np.array(history, dtype=object), allow_pickle=True)
    return model


@torch.no_grad()
def prune_unsupported(model: GaussianModel, views: list[View],
                      keep_weight: float = 0.999,
                      max_single_view_share: float = 0.9) -> dict:
    """Delete the Gaussians the photographs cannot justify. Returns a report.

    Densification is deliberately generous, and a scene finishes with a large minority of Gaussians that no image needs. Two different
    failures hide in there, and one measurement separates them: for every view, the weight `alpha * T` each Gaussian contributes to each
    pixel, which is exactly how much of the final image it is responsible for.

      * Gaussians whose total, over every view, is negligible. Hidden behind a surface, outside every frustum, or too faint to matter.
        Dropping the tail that holds the last `1 - keep_weight` of all contributed weight costs nothing measurable.
      * FLOATERS, which are a different animal: a blob in empty space that lines up with something in the one photograph that can see it.
        The signature is concentration, not size or opacity -- almost all of its contribution comes from a single view, where a surface
        splat is seen by twenty. Anything above `max_single_view_share` goes.

    What this cannot fix is the scene looking like a pile of shards from far outside the capture. Those flakes are the surfaces: they carry
    99% of the contributed weight and are 5.7 times longer than they are thin, so they tile a surface seen from where the camera was and
    stop looking like one from anywhere else. That is the representation, not garbage, and deleting any of it makes the images worse.
    """
    device  = model.xyz.device
    total   = torch.zeros(model.n, device=device, dtype=torch.float64)
    largest = torch.zeros(model.n, device=device)
    single  = torch.zeros(model.n, device=device)

    for view in views:
        single.zero_()
        rasterize(model.xyz, model.get_scaling(), model.rotation, model.get_opacity(),
                  model.colors(view.camera.center), view.camera, use_checkpoint=False, tally=single)
        total  += single.double()
        largest = torch.maximum(largest, single)

    order = torch.argsort(total, descending=True)
    share = torch.cumsum(total[order], 0) / total.sum().clamp_min(1e-30)
    n_keep = int(torch.searchsorted(share, torch.tensor(keep_weight, dtype=share.dtype, device=device))) + 1

    carries = torch.zeros(model.n, dtype=torch.bool, device=device)
    carries[order[:n_keep]] = True
    lonely  = (largest / total.clamp_min(1e-30).float()) > max_single_view_share
    keep    = carries & ~lonely

    report = {"before": model.n, "faint": int((~carries).sum()),
              "floaters": int((carries & lonely).sum()), "after": int(keep.sum())}
    for name in ("xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation"):
        setattr(model, name, torch.nn.Parameter(getattr(model, name).detach()[keep]))
    print(f"[prune] {report['before']} -> {report['after']} Gaussians "
          f"({report['faint']} contributed almost nothing, {report['floaters']} mattered to a single view)")
    return report


@torch.no_grad()
def evaluate(model: GaussianModel, views: list[View], test_ids: list[int], train_ids: list[int], background: torch.Tensor) -> dict:
    """PSNR on held-out views, and on the same number of training views."""
    def mean_psnr(ids):
        if not ids:
            return float("nan")
        total = 0.0
        for i in ids:
            v = views[i]
            image, _ = rasterize(model.xyz, model.get_scaling(), model.rotation,
                                 model.get_opacity(),
                                 model.colors(v.camera.center), v.camera,
                                 background=background, use_checkpoint=False)
            total += psnr(image, v.image)
        return total / len(ids)
    return {"test_psnr": mean_psnr(test_ids), "train_psnr": mean_psnr(train_ids)}


# ------------------------------------------------------------------------------------------------------------------------------------------
# 13. splats.ply and cameras.json: the handoff to Part 2
# ------------------------------------------------------------------------------------------------------------------------------------------
# -- splats.ply, a binary PLY written by hand ------------------------------

# The format is a short ASCII header naming the per-vertex properties followed by a raw dump of the records, which is little enough that a
# library would cost more than it saves.


def _gaussian_properties(n_rest: int) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(n_rest)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]
    return names


def save_gaussian_ply(path: str | Path, model) -> None:
    """Write a GaussianModel to the standard 3D Gaussian splatting PLY.

    Two conventions in that format cause most interoperability bugs.

    VALUES ARE PRE-ACTIVATION: `opacity` goes into a sigmoid and `scale_i` into an exp. A viewer that forgets is not subtly wrong -- every
    splat is either invisible or the size of the room.

    THE HIGHER SH BANDS ARE TRANSPOSED: coefficient-major (N, 15, 3) in the model, channel-major in the file -- 15 reds, then 15 greens,
    then 15 blues, as the reference implementation writes them.
    """
    def np_(t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    xyz     = np_(model.xyz)
    normals = np.zeros_like(xyz)
    f_dc    = np_(model.features_dc).reshape(len(xyz), -1)
    # (N, 15, 3) -> (N, 3, 15) -> flat: all reds, then greens, then blues.
    f_rest   = np_(model.features_rest).transpose(0, 2, 1).reshape(len(xyz), -1)
    opacity  = np_(model.opacity).reshape(len(xyz), 1)
    scale    = np_(model.scaling)
    rotation = np_(model.rotation)

    data  = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rotation], axis=1).astype(np.float32)
    names = _gaussian_properties(f_rest.shape[1])
    assert data.shape[1] == len(names), (data.shape, len(names))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {name}" for name in names]
    header += ["end_header", ""]
    blob = "\n".join(header).encode("ascii")
    with open(path, "wb") as f:
        f.write(blob)
        f.write(data.tobytes())
    mib = (len(blob) + data.nbytes) / (1 << 20)
    print(f"[scene] wrote {len(data)} Gaussians to {path} ({mib:.1f} MiB)")


def load_gaussian_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read a Gaussian PLY into raw arrays (still pre-activation).

    Returns a dict with `xyz`, `features_dc`, `features_rest`, `opacity`, `scaling` and `rotation`.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = f.readline().split()
        if fmt[1] != b"binary_little_endian":
            raise ValueError(f"unsupported PLY format: {fmt[1]!r}")

        count, names = 0, []
        while True:
            line = f.readline().split()
            if not line:
                continue
            if line[0] == b"element" and line[1] == b"vertex":
                count = int(line[2])
            elif line[0] == b"property":
                if line[1] != b"float":
                    raise ValueError(f"unsupported property type: {line[1]!r}")
                names.append(line[2].decode())
            elif line[0] == b"end_header":
                break
        raw = np.frombuffer(f.read(count * len(names) * 4), dtype="<f4").reshape(count, len(names))

    column = {name: raw[:, i] for i, name in enumerate(names)}
    n_rest = sum(1 for name in names if name.startswith("f_rest_"))

    def stack(prefix: str, k: int) -> np.ndarray:
        return np.stack([column[f"{prefix}{i}"] for i in range(k)], axis=1)

    # Undo the channel-major layout: (N, 3, 15) -> (N, 15, 3).
    rest = stack("f_rest_", n_rest).reshape(count, 3, n_rest // 3)
    return {
        "xyz": np.stack([column["x"], column["y"], column["z"]], axis=1),
        "features_dc": stack("f_dc_", 3).reshape(count, 1, 3),
        "features_rest": rest.transpose(0, 2, 1).copy(),
        "opacity": column["opacity"].reshape(count, 1),
        "scaling": stack("scale_", 3),
        "rotation": stack("rot_", 4),
    }


def model_from_ply(path: str | Path, device: str = "cuda"):
    """Rebuild a GaussianModel from a PLY, activations and all.

    The inverse of `save_gaussian_ply`, and it sits beside it so that the two directions of the format cannot drift apart.
    """
    data = load_gaussian_ply(path)
    model = GaussianModel(max_sh_degree=3).to(device)
    model.active_sh_degree = 3
    t = lambda a: torch.nn.Parameter(torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device))
    model.xyz = t(data["xyz"])
    model.features_dc = t(data["features_dc"])
    model.features_rest = t(data["features_rest"])
    model.opacity = t(data["opacity"])
    model.scaling = t(data["scaling"])
    model.rotation = t(data["rotation"])
    return model


# -- cameras.json, the other half of the handoff ----------------------------

# A PLY carries no camera information at all, so without this a viewer has to guess where to stand and which way is up -- and for a
# forward-facing capture the guess lands outside the volume the cameras occupied, where the scene looks like shattered glass.
#
# We write the reference implementation's format rather than one of our own, so this file opens our capture in somebody else's viewer too.
# One object per camera:
#
#   position   the camera center in world space, C = -R^T t
#   rotation   camera-to-world, R^T, row-major -- so its COLUMNS are the
#              camera's right, down and forward axes in world space
#   fx, fy     focal length in pixels; width, height the image size
#
# Reading and writing live together: a convention implemented in two places is a convention that drifts.


def write_cameras_json(rec: Reconstruction, path: str | Path) -> None:
    """Write every reconstructed pose in the reference trainer's format.

    A viewer opens at the median camera, reproducing its orientation exactly: a better first impression than any pose we could compute,
    because it is one the capture actually had.
    """
    cameras = [{
        "id": i, "img_name": Path(name).stem,
        "width": rec.width, "height": rec.height,
        "position": [float(v) for v in -rec.R[i].T @ rec.t[i]],
        "rotation": [[float(v) for v in row] for row in rec.R[i].T],
        "fx": float(rec.focal), "fy": float(rec.focal),
    } for i, name in enumerate(rec.image_names)]
    Path(path).write_text(json.dumps(cameras, indent=1), encoding="utf-8")
    print(f"[scene] wrote {len(cameras)} cameras to {path}")


def read_cameras(path: str | Path) -> dict:
    """The camera a viewer opens at: the median entry of a cameras.json."""
    cameras  = json.loads(Path(path).read_text(encoding="utf-8"))
    c        = cameras[len(cameras) // 2]
    rotation = np.array(c["rotation"], dtype=float)
    eye      = np.array(c["position"], dtype=float)
    return {"eye": eye,
            "up": -rotation[:, 1],
            "center": eye + rotation[:, 2] * 3.0,
            "fovy_deg": float(np.degrees(2.0 * np.arctan(c["height"] / (2.0 * c["fy"]))))}


def write_cameras(path: str | Path, eye, target, up, fov_y_deg: float, width: int = 1920, height: int = 1080) -> None:
    """A one-camera cameras.json, for a synthetic scene."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    forward         = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up / np.linalg.norm(up))
    right /= np.linalg.norm(right)
    down     = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=1)      # columns are the axes
    focal    = 0.5 * height / np.tan(np.radians(fov_y_deg) * 0.5)
    Path(path).write_text(json.dumps([{
        "id": 0, "img_name": "synthetic", "width": width, "height": height,
        "position": eye.tolist(),
        "rotation": [[float(v) for v in row] for row in rotation],
        "fx": focal, "fy": focal,
    }], indent=1), encoding="utf-8")


def look_at_view(target: np.ndarray, up: np.ndarray, eye: np.ndarray):
    """The world-to-camera (R, t) a viewer ends up with at that pose.

    The chapter's convention throughout: x right, y down, z forward.
    """
    up      = up / np.linalg.norm(up)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R    = np.stack([right, down, forward])
    return R, -R @ eye
