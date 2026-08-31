from __future__ import annotations

import argparse
import json
import math
import shutil
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path.cwd().resolve()


TINY_NERF_URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"


TINY_QUALITY_LONG_CONFIG = {
    "dataset_type": "tiny",
    "seed": 7,
    "width": 256,
    "depth": 8,
    "skip": 4,
    "pos_freqs": 10,
    "dir_freqs": 4,
    "include_input": True,
    "use_pi": False,
    "n_coarse": 64,
    "n_fine": 64,
    "n_rays": 2048,
    "iters": 12000,
    "lr": 5e-4,
    "lr_decay": True,
    "precrop_iters": 500,
    "raw_noise_std": 0.0,
    "white_bkgd": False,
    "chunk": 32768,
    "eval_rays": 4096,
    "log_every": 500,
    "eval_every": 4000,
    "holdout": 8,
    "test_index": 0,
}


@dataclass
class NerfData:
    images: torch.Tensor
    poses: torch.Tensor
    K: torch.Tensor
    near: float
    far: float
    render_poses: torch.Tensor | None = None

    @property
    def H(self) -> int:
        return int(self.images.shape[1])

    @property
    def W(self) -> int:
        return int(self.images.shape[2])


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True, use_pi: bool = True):
        super().__init__()
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if use_pi:
            freqs = freqs * torch.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim_per_input_dim(self) -> int:
        return (1 if self.include_input else 0) + 2 * int(self.freqs.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x[..., None] * self.freqs
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1).flatten(-2)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc


class NeRF(nn.Module):
    def __init__(
        self,
        pos_dim: int = 63,
        dir_dim: int = 27,
        width: int = 256,
        depth: int = 8,
        skip: int = 4,
    ):
        super().__init__()
        self.skip = skip

        layers = []
        in_dim = pos_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            in_dim = width + (pos_dim if i == skip else 0)

        self.trunk = nn.ModuleList(layers)
        self.sigma_head = nn.Linear(width, 1)
        self.feature_head = nn.Linear(width, width)
        self.color_body = nn.Linear(width + dir_dim, width // 2)
        self.color_head = nn.Linear(width // 2, 3)

    def forward(self, x_enc: torch.Tensor, d_enc: torch.Tensor):
        h = x_enc
        for i, layer in enumerate(self.trunk):
            h = torch.relu(layer(h))
            if i == self.skip:
                h = torch.cat([h, x_enc], dim=-1)

        sigma_raw = self.sigma_head(h)
        feature = self.feature_head(h)
        h_color = torch.relu(self.color_body(torch.cat([feature, d_enc], dim=-1)))
        rgb = torch.sigmoid(self.color_head(h_color))
        return rgb, sigma_raw


def make_intrinsics(H: int, W: int, focal: float, device=None, dtype=torch.float32):
    return torch.tensor(
        [[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def get_rays(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor, pixel_center: float = 0.5):
    device = c2w.device
    dtype = c2w.dtype
    j, i = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij",
    )

    dirs = torch.stack(
        [
            ((i + pixel_center) - K[0, 2]) / K[0, 0],
            (-(j + pixel_center) + K[1, 2]) / K[1, 1],
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    rays_d = torch.einsum("hwc,rc->hwr", dirs, c2w[:3, :3])
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def load_tiny_nerf(path: str | Path, holdout: int = 8) -> tuple[NerfData, NerfData]:
    data = np.load(path)
    images = data["images"].astype(np.float32)
    poses = data["poses"].astype(np.float32)
    focal = float(data["focal"])
    H, W = images.shape[1:3]
    K = make_intrinsics(H, W, focal)
    render_poses = None
    if "render_poses" in data.files:
        render_poses = torch.from_numpy(data["render_poses"].astype(np.float32))

    test_idx = np.arange(0, images.shape[0], holdout)
    train_idx = np.array([i for i in range(images.shape[0]) if i not in set(test_idx)])

    train = NerfData(
        images=torch.from_numpy(images[train_idx]),
        poses=torch.from_numpy(poses[train_idx]),
        K=K,
        near=2.0,
        far=6.0,
        render_poses=render_poses,
    )
    test = NerfData(
        images=torch.from_numpy(images[test_idx]),
        poses=torch.from_numpy(poses[test_idx]),
        K=K,
        near=2.0,
        far=6.0,
        render_poses=render_poses,
    )
    return train, test


def load_dataset(config: dict) -> tuple[NerfData, NerfData]:
    if config.get("dataset_type", "tiny") != "tiny":
        raise ValueError("This standalone chapter script only supports the Tiny NeRF dataset.")
    return load_tiny_nerf(config["datadir"], holdout=int(config.get("holdout", 8)))


def raw2outputs(
    raw_rgb: torch.Tensor,
    raw_sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = True,
):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = dists * torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    sigma = raw_sigma
    if raw_noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * raw_noise_std

    alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)
    trans_inclusive = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    trans = torch.cat(
        [torch.ones_like(trans_inclusive[..., :1]), trans_inclusive[..., :-1]],
        dim=-1,
    )

    weights = alpha * trans
    rgb_map = (weights[..., None] * raw_rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return {
        "rgb": rgb_map,
        "depth": depth_map,
        "acc": acc_map,
        "weights": weights,
        "alpha": alpha,
    }


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_samples: int, det: bool = False):
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device, dtype=bins.dtype)
        u = u.expand(*cdf.shape[:-1], n_samples).contiguous()
    else:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=bins.device, dtype=bins.dtype)

    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    return bins_below + t * (bins_above - bins_below)


def run_network_on_samples(
    net_fn,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    chunk: int,
    raw_noise_std: float,
    white_bkgd: bool,
):
    R, S = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    dirs = F.normalize(rays_d, dim=-1)[:, None, :].expand_as(pts)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    rgbs = []
    sigmas = []
    for start in range(0, pts_flat.shape[0], chunk):
        rgb, sigma = net_fn(pts_flat[start : start + chunk], dirs_flat[start : start + chunk])
        rgbs.append(rgb)
        sigmas.append(sigma)

    raw_rgb = torch.cat(rgbs, dim=0).reshape(R, S, 3)
    raw_sigma = torch.cat(sigmas, dim=0).reshape(R, S)
    return raw2outputs(raw_rgb, raw_sigma, z_vals, rays_d, raw_noise_std, white_bkgd)


def render_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    coarse_fn,
    fine_fn=None,
    n_coarse: int = 64,
    n_fine: int = 64,
    perturb: bool = True,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
    chunk: int = 32768,
):
    R = rays_o.shape[0]
    t = torch.linspace(0.0, 1.0, n_coarse, device=rays_o.device, dtype=rays_o.dtype)
    z_vals = near * (1.0 - t) + far * t
    z_vals = z_vals.expand(R, n_coarse)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    coarse = run_network_on_samples(
        coarse_fn, rays_o, rays_d, z_vals, chunk, raw_noise_std, white_bkgd
    )

    if n_fine <= 0 or fine_fn is None:
        return coarse, None

    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_fine = sample_pdf(
        mids,
        coarse["weights"][..., 1:-1].detach(),
        n_fine,
        det=not perturb,
    )
    z_all, _ = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)
    fine = run_network_on_samples(
        fine_fn, rays_o, rays_d, z_all, chunk, raw_noise_std, white_bkgd
    )
    return coarse, fine


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def save_image(path: str | Path, image: torch.Tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = image.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray(path: str | Path, image: torch.Tensor, min_value=None, max_value=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = image.detach().cpu()
    if min_value is None:
        min_value = float(x.min())
    if max_value is None:
        max_value = float(x.max())
    x = (x - min_value) / max(max_value - min_value, 1e-8)
    arr = (x.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def build_model(config: dict, device: torch.device):
    pos_freqs = int(config.get("pos_freqs", 10))
    dir_freqs = int(config.get("dir_freqs", 4))
    include_input = bool(config.get("include_input", True))
    use_pi = bool(config.get("use_pi", True))
    encode_pos = PositionalEncoding(pos_freqs, include_input, use_pi).to(device)
    encode_dir = PositionalEncoding(dir_freqs, include_input, use_pi).to(device)

    pos_dim = 3 * encode_pos.out_dim_per_input_dim
    dir_dim = 3 * encode_dir.out_dim_per_input_dim
    width = int(config.get("width", 256))
    depth = int(config.get("depth", 8))
    skip = int(config.get("skip", 4))
    coarse = NeRF(pos_dim, dir_dim, width=width, depth=depth, skip=skip).to(device)
    fine = None
    if int(config.get("n_fine", 0)) > 0:
        fine = NeRF(pos_dim, dir_dim, width=width, depth=depth, skip=skip).to(device)

    def wrap(net):
        def query(points, dirs):
            return net(encode_pos(points), encode_dir(dirs))

        return query

    return coarse, fine, wrap


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def precompute_rays(data, device):
    K = data.K.to(device)
    rays_o = []
    rays_d = []
    for pose in data.poses:
        ro, rd = get_rays(data.H, data.W, K, pose.to(device))
        rays_o.append(ro.cpu())
        rays_d.append(rd.cpu())
    return torch.stack(rays_o, dim=0), torch.stack(rays_d, dim=0)


def sample_batch(images, all_rays_o, all_rays_d, n_rays, step, precrop_iters):
    n_images, H, W = images.shape[:3]
    if step <= precrop_iters:
        y0, y1 = H // 4, 3 * H // 4
        x0, x1 = W // 4, 3 * W // 4
        ys = torch.randint(y0, y1, (n_rays,))
        xs = torch.randint(x0, x1, (n_rays,))
    else:
        ys = torch.randint(0, H, (n_rays,))
        xs = torch.randint(0, W, (n_rays,))
    ns = torch.randint(0, n_images, (n_rays,))
    return all_rays_o[ns, ys, xs], all_rays_d[ns, ys, xs], images[ns, ys, xs]


def render_image(data, pose, coarse, fine, wrap, config, device):
    K = data.K.to(device)
    ro, rd = get_rays(data.H, data.W, K, pose.to(device))
    ro = ro.reshape(-1, 3)
    rd = rd.reshape(-1, 3)
    out_chunks = []
    depth_chunks = []
    acc_chunks = []
    eval_rays = int(config.get("eval_rays", 4096))
    with torch.no_grad():
        for start in range(0, ro.shape[0], eval_rays):
            out_c, out_f = render_rays(
                ro[start : start + eval_rays],
                rd[start : start + eval_rays],
                data.near,
                data.far,
                wrap(coarse),
                wrap(fine) if fine is not None else None,
                n_coarse=int(config.get("n_coarse", 64)),
                n_fine=int(config.get("n_fine", 0)),
                perturb=False,
                raw_noise_std=0.0,
                white_bkgd=bool(config.get("white_bkgd", True)),
                chunk=int(config.get("chunk", 32768)),
            )
            out = out_f if out_f is not None else out_c
            out_chunks.append(out["rgb"])
            depth_chunks.append(out["depth"])
            acc_chunks.append(out["acc"])
    return {
        "rgb": torch.cat(out_chunks, dim=0).reshape(data.H, data.W, 3),
        "depth": torch.cat(depth_chunks, dim=0).reshape(data.H, data.W),
        "acc": torch.cat(acc_chunks, dim=0).reshape(data.H, data.W),
    }


def train_nerf(config: dict):
    set_seed(int(config.get("seed", 0)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(config.get("outdir", "runs/nerf"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_data, test_data = load_dataset(config)
    print(f"Loaded {config.get('dataset_type')} train images: {tuple(train_data.images.shape)}")
    print(f"Device: {device}")

    coarse, fine, wrap = build_model(config, device)
    total_params = count_parameters(coarse) + (count_parameters(fine) if fine is not None else 0)
    print(f"Trainable parameters: {total_params:,}")

    params = list(coarse.parameters())
    if fine is not None:
        params += list(fine.parameters())
    lr = float(config.get("lr", 5e-4))
    optimizer = torch.optim.Adam(params, lr=lr)

    all_rays_o, all_rays_d = precompute_rays(train_data, device)
    train_images = train_data.images.float()

    n_iters = int(config.get("iters", 1000))
    n_rays = int(config.get("n_rays", 1024))
    eval_every = int(config.get("eval_every", 250))
    precrop_iters = int(config.get("precrop_iters", 0))
    raw_noise_std = float(config.get("raw_noise_std", 0.0))

    for step in range(1, n_iters + 1):
        rays_o, rays_d, target = sample_batch(
            train_images, all_rays_o, all_rays_d, n_rays, step, precrop_iters
        )
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target = target.to(device)

        out_c, out_f = render_rays(
            rays_o,
            rays_d,
            train_data.near,
            train_data.far,
            wrap(coarse),
            wrap(fine) if fine is not None else None,
            n_coarse=int(config.get("n_coarse", 64)),
            n_fine=int(config.get("n_fine", 0)),
            perturb=True,
            raw_noise_std=raw_noise_std,
            white_bkgd=bool(config.get("white_bkgd", True)),
            chunk=int(config.get("chunk", 32768)),
        )

        loss = F.mse_loss(out_c["rgb"], target)
        if out_f is not None:
            loss = loss + F.mse_loss(out_f["rgb"], target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if bool(config.get("lr_decay", True)):
            decay = 0.1 ** (step / max(n_iters, 1))
            for group in optimizer.param_groups:
                group["lr"] = lr * decay

        if step == 1 or step % int(config.get("log_every", 50)) == 0:
            print(f"[{step:06d}/{n_iters}] loss={loss.item():.6f} psnr={psnr_from_mse(loss).item():.2f}")

        if step == 1 or step % eval_every == 0 or step == n_iters:
            idx = int(config.get("test_index", 0)) % test_data.images.shape[0]
            render = render_image(test_data, test_data.poses[idx], coarse, fine, wrap, config, device)
            gt = test_data.images[idx].to(device)
            mse = F.mse_loss(render["rgb"], gt)
            psnr = psnr_from_mse(mse)
            print(f"  eval[{idx}] mse={mse.item():.6f} psnr={psnr.item():.2f}")
            save_image(outdir / f"render_{step:06d}.png", render["rgb"])
            save_image(outdir / f"target_{idx:03d}.png", gt)
            save_gray(outdir / f"depth_{step:06d}.png", render["depth"], train_data.near, train_data.far)
            save_gray(outdir / f"opacity_{step:06d}.png", render["acc"], 0.0, 1.0)
            torch.save(
                {
                    "step": step,
                    "coarse": coarse.state_dict(),
                    "fine": fine.state_dict() if fine is not None else None,
                    "config": config,
                },
                outdir / "checkpoint.pt",
            )


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return "cpu"
    return name


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


def make_config(data_path: Path, outdir: Path, device: str, iters: int, n_rays: int) -> dict:
    config = dict(TINY_QUALITY_LONG_CONFIG)
    config["datadir"] = str(data_path)
    config["outdir"] = str(outdir)
    config["device"] = device
    config["iters"] = int(iters)
    config["n_rays"] = int(n_rays)
    return config


def load_checkpoint_models(checkpoint_path: Path, data_path: Path, outdir: Path, device_name: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["datadir"] = str(data_path)
    config["outdir"] = str(outdir)
    config["device"] = device_name

    device = torch.device(device_name)
    coarse, fine, wrap = build_model(config, device)
    coarse.load_state_dict(checkpoint["coarse"])
    if fine is not None and checkpoint.get("fine") is not None:
        fine.load_state_dict(checkpoint["fine"])

    coarse.eval()
    if fine is not None:
        fine.eval()
    return checkpoint, config, device, coarse, fine, wrap


def render_eval_image(checkpoint_path: Path, data_path: Path, outdir: Path, device_name: str) -> dict:
    checkpoint, config, device, coarse, fine, wrap = load_checkpoint_models(
        checkpoint_path, data_path, outdir, device_name
    )
    _, test_data = load_dataset(config)
    idx = int(config.get("test_index", 0)) % test_data.images.shape[0]

    render = render_image(test_data, test_data.poses[idx], coarse, fine, wrap, config, device)
    target = test_data.images[idx].to(device)
    mse = F.mse_loss(render["rgb"], target)
    psnr = psnr_from_mse(mse)

    step = int(checkpoint.get("step", config.get("iters", 0)))
    step_tag = f"{step:06d}" if step > 0 else "final"

    paths = {
        "target": outdir / f"target_{idx:03d}.png",
        "render": outdir / f"render_{step_tag}.png",
        "render_final": outdir / "render_final.png",
        "depth": outdir / f"depth_{step_tag}.png",
        "opacity": outdir / f"opacity_{step_tag}.png",
    }

    save_image(paths["target"], target)
    save_image(paths["render"], render["rgb"])
    save_image(paths["render_final"], render["rgb"])
    save_gray(paths["depth"], render["depth"], test_data.near, test_data.far)
    save_gray(paths["opacity"], render["acc"], 0.0, 1.0)

    print(f"Evaluation MSE: {mse.item():.6f}")
    print(f"Evaluation PSNR: {psnr.item():.2f} dB")
    return paths


def make_contact_sheet(items: list[tuple[str, Path]], out_path: Path, scale: float = 2.0):
    parsed = []
    for label, path in items:
        img = Image.open(path).convert("RGB")
        if scale != 1.0:
            size = (int(round(img.width * scale)), int(round(img.height * scale)))
            img = img.resize(size, Image.Resampling.NEAREST)
        parsed.append((label, img))

    tile_w = max(img.width for _, img in parsed)
    tile_h = max(img.height for _, img in parsed)
    label_h = 28
    pad = 10
    sheet = Image.new("RGB", (pad + len(parsed) * (tile_w + pad), tile_h + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(parsed):
        x = pad + idx * (tile_w + pad)
        y = pad + label_h
        sheet.paste(img, (x, y))
        draw.text((x, pad), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Wrote {out_path}")


def pose_spherical(theta_degrees: float, phi_degrees: float, radius: float) -> torch.Tensor:
    theta = math.radians(theta_degrees)
    phi = math.radians(phi_degrees)

    trans = np.eye(4, dtype=np.float32)
    trans[2, 3] = radius

    rot_phi = np.eye(4, dtype=np.float32)
    rot_phi[1, 1] = math.cos(phi)
    rot_phi[1, 2] = -math.sin(phi)
    rot_phi[2, 1] = math.sin(phi)
    rot_phi[2, 2] = math.cos(phi)

    rot_theta = np.eye(4, dtype=np.float32)
    rot_theta[0, 0] = math.cos(theta)
    rot_theta[0, 2] = -math.sin(theta)
    rot_theta[2, 0] = math.sin(theta)
    rot_theta[2, 2] = math.cos(theta)

    c2w = rot_theta @ rot_phi @ trans
    coord_fix = np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    return torch.from_numpy(coord_fix @ c2w)


def scaled_data_like(data, scale: float):
    H = int(round(data.H * scale))
    W = int(round(data.W * scale))
    focal = float(data.K[0, 0]) * scale
    return SimpleNamespace(H=H, W=W, near=data.near, far=data.far, images=data.images, K=make_intrinsics(H, W, focal))


def write_viewer(out_dir: Path, frame_names: list[str], depth_names: list[str], opacity_names: list[str]):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiny NeRF Turntable Viewer</title>
<style>
body {{
  margin: 0;
  font-family: Segoe UI, Arial, sans-serif;
  background: #11161d;
  color: #e9eef4;
}}
.wrap {{
  max-width: 980px;
  margin: 0 auto;
  padding: 24px;
}}
.stage {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 14px;
  align-items: start;
}}
figure {{
  margin: 0;
  background: #1c2530;
  border: 1px solid #2e3a48;
  border-radius: 8px;
  padding: 10px;
}}
img {{
  width: 100%;
  image-rendering: auto;
  background: #000;
  display: block;
}}
figcaption {{
  color: #9fb0c4;
  font-size: 13px;
  margin-top: 8px;
}}
.controls {{
  display: flex;
  gap: 12px;
  align-items: center;
  margin: 18px 0 8px;
}}
input[type=range] {{ flex: 1; }}
button {{
  background: #2b6cb0;
  color: white;
  border: 0;
  border-radius: 6px;
  padding: 9px 12px;
  cursor: pointer;
}}
.meta {{ color: #9fb0c4; font-size: 14px; }}
@media (max-width: 760px) {{ .stage {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Tiny NeRF Turntable Viewer</h1>
  <p class="meta">Drag the slider, use left/right arrow keys, or press Play.</p>
  <div class="controls">
    <button id="play">Play</button>
    <input id="slider" type="range" min="0" max="{len(frame_names) - 1}" value="0">
    <span id="counter">1 / {len(frame_names)}</span>
  </div>
  <div class="stage">
    <figure><img id="rgb" src="{frame_names[0]}"><figcaption>RGB render</figcaption></figure>
    <figure><img id="depth" src="{depth_names[0]}"><figcaption>Expected depth</figcaption></figure>
    <figure><img id="opacity" src="{opacity_names[0]}"><figcaption>Opacity</figcaption></figure>
  </div>
</div>
<script>
const frames = {json.dumps(frame_names)};
const depths = {json.dumps(depth_names)};
const opacities = {json.dumps(opacity_names)};
let idx = 0;
let playing = false;
let timer = null;
const slider = document.getElementById('slider');
const counter = document.getElementById('counter');
const rgb = document.getElementById('rgb');
const depth = document.getElementById('depth');
const opacity = document.getElementById('opacity');
const play = document.getElementById('play');
function show(i) {{
  idx = (i + frames.length) % frames.length;
  slider.value = idx;
  counter.textContent = `${{idx + 1}} / ${{frames.length}}`;
  rgb.src = frames[idx];
  depth.src = depths[idx];
  opacity.src = opacities[idx];
}}
slider.addEventListener('input', e => show(Number(e.target.value)));
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') show(idx + 1);
  if (e.key === 'ArrowLeft') show(idx - 1);
}});
play.addEventListener('click', () => {{
  playing = !playing;
  play.textContent = playing ? 'Pause' : 'Play';
  if (playing) {{
    timer = setInterval(() => show(idx + 1), 120);
  }} else {{
    clearInterval(timer);
  }}
}});
</script>
</body>
</html>
"""
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")
    print(f"Wrote {out_dir / 'viewer.html'}")


def render_turntable(
    checkpoint_path: Path,
    data_path: Path,
    out_dir: Path,
    device_name: str,
    frames: int,
    scale: float,
    phi: float,
    radius: float | None,
    write_gif: bool,
):
    _, config, device, coarse, fine, wrap = load_checkpoint_models(checkpoint_path, data_path, out_dir.parent, device_name)
    train_data, test_data = load_dataset(config)
    data = scaled_data_like(test_data, scale)
    out_dir.mkdir(parents=True, exist_ok=True)

    centers = train_data.poses[:, :3, 3]
    orbit_radius = radius if radius is not None else float(torch.linalg.norm(centers, dim=-1).mean())

    frame_names = []
    depth_names = []
    opacity_names = []
    pil_frames = []

    for i, theta in enumerate(np.linspace(-180.0, 180.0, frames, endpoint=False)):
        pose = pose_spherical(float(theta), phi, orbit_radius)
        render = render_image(data, pose, coarse, fine, wrap, config, device)

        rgb_name = f"rgb_{i:03d}.png"
        depth_name = f"depth_{i:03d}.png"
        opacity_name = f"opacity_{i:03d}.png"

        save_image(out_dir / rgb_name, render["rgb"])
        save_gray(out_dir / depth_name, render["depth"], data.near, data.far)
        save_gray(out_dir / opacity_name, render["acc"], 0.0, 1.0)

        frame_names.append(rgb_name)
        depth_names.append(depth_name)
        opacity_names.append(opacity_name)
        palette = Image.Palette.ADAPTIVE if hasattr(Image, "Palette") else Image.ADAPTIVE
        pil_frames.append(Image.open(out_dir / rgb_name).convert("P", palette=palette))
        print(f"Turntable frame {i + 1:03d}/{frames}: theta={theta:.1f}")

    if write_gif and pil_frames:
        gif_path = out_dir / "turntable.gif"
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=90, loop=0)
        print(f"Wrote {gif_path}")

    write_viewer(out_dir, frame_names, depth_names, opacity_names)
    print(f"Turntable radius: {orbit_radius:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Train and render the chapter's Tiny NeRF long-run example.")
    parser.add_argument("--data-root", default=None, help="Defaults to ./data under the current directory.")
    parser.add_argument("--outdir", default=None, help="Defaults to ./runs/tiny_quality_long under the current directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--iters", type=int, default=12000)
    parser.add_argument("--n-rays", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--phi", type=float, default=-30.0)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--checkpoint", default=None, help="Use an existing checkpoint instead of training.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if outdir/checkpoint.pt exists.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-turntable", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    if args.checkpoint and args.force_train:
        parser.error("--checkpoint and --force-train cannot be used together.")
    if args.iters < 1:
        parser.error("--iters must be at least 1.")
    if args.n_rays < 1:
        parser.error("--n-rays must be at least 1.")
    if args.frames < 1:
        parser.error("--frames must be at least 1.")
    if args.scale <= 0.0:
        parser.error("--scale must be greater than 0.")

    data_root = resolve_path(args.data_root, PROJECT_ROOT / "data")
    outdir = resolve_path(args.outdir, PROJECT_ROOT / "runs" / "tiny_quality_long")
    device_name = resolve_device(args.device)

    data_path = ensure_tiny_nerf_data(data_root)
    if args.download_only:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    config = make_config(data_path, outdir, device_name, args.iters, args.n_rays)
    (outdir / "chapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    checkpoint_path = resolve_path(args.checkpoint, outdir / "checkpoint.pt") if args.checkpoint else outdir / "checkpoint.pt"
    if args.checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        print(f"Using checkpoint: {checkpoint_path}")
    elif checkpoint_path.exists() and not args.force_train:
        print(f"Reusing existing checkpoint: {checkpoint_path}")
        print("Pass --force-train to retrain from scratch.")
    else:
        train_nerf(config)
        if not checkpoint_path.exists():
            raise RuntimeError(f"Training finished but checkpoint was not written: {checkpoint_path}")

    eval_paths = render_eval_image(checkpoint_path, data_path, outdir, device_name)
    make_contact_sheet(
        [("target", eval_paths["target"]), ("tiny_quality_long", eval_paths["render_final"])],
        outdir / "comparison_large.png",
        scale=args.scale,
    )

    if not args.skip_turntable:
        render_turntable(
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            out_dir=outdir / "turntable",
            device_name=device_name,
            frames=args.frames,
            scale=args.scale,
            phi=args.phi,
            radius=args.radius,
            write_gif=not args.no_gif,
        )

    print("\nDone.")
    print(f"Comparison: {outdir / 'comparison_large.png'}")
    if not args.skip_turntable:
        print(f"Viewer:     {outdir / 'turntable' / 'viewer.html'}")
        if not args.no_gif:
            print(f"GIF:        {outdir / 'turntable' / 'turntable.gif'}")


if __name__ == "__main__":
    main()
