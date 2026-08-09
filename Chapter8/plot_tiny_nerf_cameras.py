from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd().resolve()
TINY_NERF_URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def valid_tiny_nerf_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            required = {"images", "poses", "focal"}
            if not required.issubset(data.files):
                return False
            return data["images"].ndim == 4 and data["poses"].shape[-2:] == (4, 4)
    except Exception:
        return False


def make_ssl_context() -> ssl.SSLContext:
    # The dataset host redirects plain HTTP to HTTPS. Python on Windows does not use the
    # system certificate store, so verification fails unless we point it at a CA bundle.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_file(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    context = make_ssl_context()
    try:
        response = urllib.request.urlopen(url, context=context, timeout=60)
    except urllib.error.URLError as err:
        if isinstance(err.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                f"TLS certificate verification failed for {url}.\n"
                "Install a CA bundle with `pip install --upgrade certifi` and retry, or download "
                f"the file manually and place it at {path}."
            ) from err
        raise

    with response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(path)
    print(f"Saved {path}")


def ensure_tiny_nerf_data(data_root: Path) -> Path:
    data_path = data_root / "tiny_nerf_data.npz"
    if valid_tiny_nerf_file(data_path):
        print(f"Using Tiny NeRF data: {data_path}")
        return data_path
    print(f"Tiny NeRF data missing or incomplete: {data_path}")
    download_file(TINY_NERF_URL, data_path)
    if not valid_tiny_nerf_file(data_path):
        raise RuntimeError(f"Downloaded file is not a valid Tiny NeRF dataset: {data_path}")
    return data_path


def make_intrinsics(H: int, W: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_tiny_nerf(path: Path, holdout: int):
    with np.load(path) as data:
        # Only the image shape is needed here, so skip materialising the pixel data.
        image_shape = data["images"].shape
        poses = data["poses"].astype(np.float32)
        focal = float(data["focal"])
    H, W = image_shape[1:3]
    K = make_intrinsics(H, W, focal)
    test_idx = set(np.arange(0, poses.shape[0], holdout).tolist())
    return image_shape, poses, K, test_idx


def pixel_to_camera_point(u: float, v: float, K: np.ndarray, depth: float) -> np.ndarray:
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = -(v - K[1, 2]) / K[1, 1] * depth
    z = -depth
    return np.array([x, y, z], dtype=np.float32)


def transform_points(c2w: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (c2w @ points_h.T).T[:, :3]


def frustum_points(c2w: np.ndarray, K: np.ndarray, H: int, W: int, depth: float) -> np.ndarray:
    corners_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            pixel_to_camera_point(0.0, 0.0, K, depth),
            pixel_to_camera_point(float(W), 0.0, K, depth),
            pixel_to_camera_point(float(W), float(H), K, depth),
            pixel_to_camera_point(0.0, float(H), K, depth),
        ],
        dtype=np.float32,
    )
    return transform_points(c2w, corners_cam)


def draw_frustum(ax, c2w: np.ndarray, K: np.ndarray, H: int, W: int, depth: float, color: str, alpha: float):
    p = frustum_points(c2w, K, H, W, depth)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for a, b in edges:
        ax.plot(*zip(p[a], p[b]), color=color, alpha=alpha, linewidth=0.8)
    forward = c2w[:3, :3] @ np.array([0.0, 0.0, -depth * 0.9], dtype=np.float32)
    ax.quiver(*p[0], *forward, color="tab:red", length=1.0, normalize=False, linewidth=1.0, alpha=alpha)
    return p


def draw_sample_rays(ax, c2w: np.ndarray, K: np.ndarray, H: int, W: int, length: float) -> np.ndarray:
    samples = [
        (W * 0.5, H * 0.5),
        (W * 0.25, H * 0.5),
        (W * 0.75, H * 0.5),
        (W * 0.5, H * 0.25),
        (W * 0.5, H * 0.75),
    ]
    origin = c2w[:3, 3]
    ends = []
    for u, v in samples:
        point_cam = pixel_to_camera_point(u, v, K, 1.0)
        direction = c2w[:3, :3] @ point_cam
        direction = direction / max(np.linalg.norm(direction), 1e-8)
        end = origin + direction * length
        ax.plot(*zip(origin, end), color="tab:orange", linewidth=1.4)
        ends.append(end)
    return np.stack(ends, axis=0)


def set_equal_3d(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max())
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    # Equal ranges alone are not enough: the default 3D box aspect squashes z.
    ax.set_box_aspect((1.0, 1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Plot Tiny NeRF camera positions and view frustums.")
    parser.add_argument("--data-root", default=None, help="Defaults to ./data under the current directory.")
    parser.add_argument("--out", default=None, help="Defaults to ./runs/tiny_quality_long/camera_rig.png.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth camera is marked as validation/test.")
    parser.add_argument("--max-cameras", type=int, default=106)
    parser.add_argument("--frustum-depth", type=float, default=0.35)
    parser.add_argument("--draw-rays", action="store_true", help="Draw a few example rays from the first camera.")
    parser.add_argument("--ray-length", type=float, default=2.0)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    args = parser.parse_args()

    if args.holdout < 1:
        parser.error("--holdout must be at least 1.")
    if args.max_cameras < 1:
        parser.error("--max-cameras must be at least 1.")
    if args.frustum_depth <= 0.0:
        parser.error("--frustum-depth must be greater than 0.")
    if args.ray_length <= 0.0:
        parser.error("--ray-length must be greater than 0.")

    data_root = resolve_path(args.data_root, PROJECT_ROOT / "data")
    out = resolve_path(args.out, PROJECT_ROOT / "runs" / "tiny_quality_long" / "camera_rig.png")
    data_path = ensure_tiny_nerf_data(data_root)
    image_shape, poses, K, test_idx = load_tiny_nerf(data_path, args.holdout)

    H, W = image_shape[1:3]
    count = min(args.max_cameras, poses.shape[0])
    poses = poses[:count]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    train_centers = []
    test_centers = []
    all_points = [np.zeros((1, 3), dtype=np.float32)]
    for idx, pose in enumerate(poses):
        is_test = idx in test_idx
        color = "tab:green" if is_test else "tab:blue"
        alpha = 0.6 if is_test else 0.24
        all_points.append(draw_frustum(ax, pose, K, H, W, args.frustum_depth, color, alpha))
        if is_test:
            test_centers.append(pose[:3, 3])
        else:
            train_centers.append(pose[:3, 3])

    if train_centers:
        c = np.stack(train_centers, axis=0)
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=12, color="tab:blue", label="train cameras")
        all_points.append(c)
    if test_centers:
        c = np.stack(test_centers, axis=0)
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=24, color="tab:green", label="held-out cameras")
        all_points.append(c)

    ax.scatter([0], [0], [0], s=54, color="black", label="scene origin")

    if args.draw_rays:
        all_points.append(draw_sample_rays(ax, poses[0], K, H, W, args.ray_length))

    set_equal_3d(ax, np.concatenate(all_points, axis=0))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Tiny NeRF camera rig ({count} poses, {H}x{W} images)")
    ax.legend(loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Dataset: {data_path}")
    print(f"Images:  {image_shape[0]} images, {H}x{W}, focal={K[0, 0]:.4f}")
    print(f"Wrote:   {out}")


if __name__ == "__main__":
    main()
