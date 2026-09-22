"""
Listing 9.4 -- Do the two renderers agree?
===========================================
The chapter ships the same algorithm twice: in PyTorch, where it has to be differentiable so the scene can be optimized (Listing 9.1), and
in GLSL, where it has to be fast so the scene can be flown around (Listing 9.2). Two implementations of one model is two chances to get a
convention wrong -- a transposed rotation, a spherical-harmonic sign, a forgotten activation -- and each of those bugs produces an image
that looks plausible. The scene still appears; it is just quietly wrong.

So we check: render one camera through the Python rasterizer, ask the C++ viewer for the same camera through its `--screenshot` flag, and
difference the two images.

    python 02_compare_renderers.py --unit   # synthetic, non-overlapping
    python 02_compare_renderers.py          # the real reconstruction

They should agree to within 8-bit rounding, and they do: about 0.45/255 mean error, 1/255 at the 99th percentile. Bit-identical they are not
-- the GPU discards fragments outside the 3-sigma quad and quantizes once at the end -- but nothing structural should survive.

`--unit` separates "different blend order" from "different mathematics": no two splats overlap there, so ordering cannot matter and any
disagreement is in the per-splat math. A clean `--unit` with a dirty real scene points at ordering or accumulation instead.

That distinction is not hypothetical. This script caught three real bugs in the viewer, only the first of them visible to the eye: the scene
rendered upside down, because LightweightVK binds a negative-height viewport (`--unit`, 30/255); blending into the 8-bit swapchain instead
of a float target, which re-quantizes after every one of the hundreds of faint splats a pixel integrates; and a depth sort quantized to 16
bits, so splats sharing a bucket blended in array order (12.6/255 on the real scene, `--unit` clean throughout
-- exactly the split this script is for).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from splatting import (Camera, look_at_view, model_from_ply, rasterize, read_cameras, write_cameras)


def unit_test(args) -> None:
    """Compare the renderers on a scene where sorting cannot matter.

    Splats on a sphere shell, small enough not to overlap on screen, so there is no ordering freedom left for a bug to hide in.
    """
    from splatting import GaussianModel, inverse_sigmoid, save_gaussian_ply

    work = Path(args.work) / "unit"
    work.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    n   = 1500
    d   = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    points = (d * rng.uniform(1.0, 1.4, (n, 1))).astype(np.float32)
    colors = rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32)

    model = GaussianModel()
    model.active_sh_degree = 3
    model.initialize(points, colors, 1.0, random_points=0, device=args.device)
    with torch.no_grad():
        model.scaling.fill_(float(np.log(0.012)))
        model.scaling += torch.randn_like(model.scaling) * 0.2
        model.opacity.fill_(float(inverse_sigmoid(torch.tensor(0.9))))
        model.rotation.copy_(torch.nn.functional.normalize(torch.randn_like(model.rotation), dim=1))
        model.features_rest.copy_(torch.randn_like(model.features_rest) * 0.15)
    save_gaussian_ply(work / "splats.ply", model)
    write_cameras(work / "cameras.json", eye=(0, 0, -4), target=(0, 0, 0), up=(0, -1, 0), fov_y_deg=30.0)

    args.ply  = str(work / "splats.ply")
    args.work = str(work)
    compare(args, label="non-overlapping splats")


def find_viewer(explicit: str | None) -> Path | None:
    """Locate the Part 2 executable in whichever build tree exists.

    `Chapter09/viewer/CMakeLists.txt` is the book's only C++ project, so the usual build tree is `.build` at the repository root; a tree
    beside the sources and single-config generators are covered too.
    """
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / ".build/Release/splat_viewer.exe",
        here.parent / ".build/splat_viewer",
        here / "viewer/build/Release/splat_viewer.exe",
        here / "viewer/build/splat_viewer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compare(args, label: str) -> None:
    work       = Path(args.work)
    ply        = Path(args.ply) if args.ply else work / "splats.ply"
    scene_path = work / "cameras.json"
    scene      = read_cameras(scene_path)

    # -- the C++ renderer ----------------------------------------------------
    shot   = work / "compare_cpp.png"
    viewer = find_viewer(args.viewer)
    if viewer is not None:
        print(f"[compare] rendering with {viewer}")
        subprocess.run([str(viewer), str(ply), str(scene_path), "--screenshot", str(shot)], check=True, stdout=subprocess.DEVNULL)
    else:
        print("[compare] splat_viewer not built (see the chapter text); " f"comparing against the existing {shot} if present")
    cpp           = cv2.cvtColor(cv2.imread(str(shot), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    height, width = cpp.shape[:2]

    # -- the Python renderer, same camera ------------------------------------
    R, t  = look_at_view(scene["center"], scene["up"], scene["eye"])
    focal = 0.5 * height / np.tan(np.radians(float(scene["fovy_deg"])) * 0.5)
    cam = Camera(R=torch.tensor(R, dtype=torch.float32, device=args.device),
                 t=torch.tensor(t, dtype=torch.float32, device=args.device),
                 focal=float(focal), cx=width / 2.0, cy=height / 2.0,
                 width=width, height=height)

    model = model_from_ply(ply, args.device)
    print(f"[compare] {label}: {model.n} Gaussians, {width}x{height}, focal {focal:.1f} px")
    with torch.no_grad():
        # The viewer clears to this color, so the trainer must too.
        background = torch.tensor([0.02, 0.02, 0.03], device=args.device)
        image, _ = rasterize(model.xyz, model.get_scaling(), model.rotation,
                             model.get_opacity(), model.colors(cam.center), cam,
                             background=background, use_checkpoint=False)
    py = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # -- the verdict ---------------------------------------------------------
    diff = np.abs(py.astype(np.int16) - cpp.astype(np.int16))
    print(f"[compare] mean |difference| {diff.mean():.3f} / 255 "
          f"({diff.mean() / 2.55:.2f}%)")
    print(f"[compare] 99th percentile   {np.percentile(diff, 99):.0f} / 255")
    print(f"[compare] pixels above 4    "
          f"{100.0 * (diff.max(axis=2) > 4).mean():.3f}%")

    panel = np.hstack([py, cpp, np.clip(diff * 8, 0, 255).astype(np.uint8)])
    out   = work / "compare_renderers.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[compare] wrote {out}  (PyTorch | C++ | 8x difference)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="work")
    parser.add_argument("--ply", default=None)
    parser.add_argument("--viewer", default=None, help="path to splat_viewer; found automatically in the usual build trees if omitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unit", action="store_true", help="run the ordering-independent sanity check instead")
    args = parser.parse_args()

    if args.unit:
        unit_test(args)
    else:
        compare(args, label="reconstructed scene")


if __name__ == "__main__":
    main()
