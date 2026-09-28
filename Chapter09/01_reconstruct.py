"""
Listing 9.3 -- The driver: video in, Gaussian splats out
========================================================
Part 1 of the chapter's demo, end to end:

    video.mp4
      -> sharp keyframes                    (Listing 9.1, section 2)
      -> features, matches, tracks          (Listing 9.1, sections 3-5)
      -> camera poses + sparse point cloud  (Listing 9.1, sections 6-9)
      -> optimized 3D Gaussians             (Listing 9.1, sections 10-12)
      -> splats.ply + cameras.json          (Listing 9.1, section 13)

Every stage writes its result to the working directory and every stage can be skipped if that result already exists, because the stages have
very different costs -- structure from motion takes minutes, splat optimization takes tens of minutes -- and you will want to re-run the
last one many times.

    python 01_reconstruct.py ../video.mp4
    python 01_reconstruct.py --skip frames sfm --iterations 15000
    python 01_reconstruct.py ../video.mp4 --stop-after frames

The two output files are the handoff to Part 2:

  splats.ply    the scene itself, in the standard 3D Gaussian splatting
                format
  cameras.json  the poses, as the reference trainer writes them. A PLY carries
                no camera information, and a viewer left to guess where to
                stand shows a forward-facing capture as shattered glass.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


from splatting import (Reconstruction, SfMConfig, TrainConfig, extract_keyframes, reconstruct, save_gaussian_ply, train, write_cameras_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="video -> 3D Gaussian splats")
    parser.add_argument("video", nargs="?", default="../video.mp4")
    parser.add_argument("--work", default="work", help="working directory for all intermediates")
    parser.add_argument("--skip", nargs="*", default=[], choices=["frames", "sfm", "train"], help="reuse existing results for these stages")
    parser.add_argument("--stop-after", choices=["frames", "sfm"],
                        help="run the pipeline only this far, which is how a "
                             "single stage is run on its own")
    parser.add_argument("--num-frames", type=int, default=350,
                        help="keyframes to select; about four per second of video suits a hand-held orbit")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--max-gaussians", type=int, default=450_000,
                        help="budget, not a target; the capture in the chapter reaches it")
    parser.add_argument("--resolution-scale", type=float, default=0.5)
    parser.add_argument("--focal-prior", type=float, default=1.05,
                        help="initial focal length as a multiple of the image "
                             "width (1.05 ~ 51 degrees horizontal FOV)")
    parser.add_argument("--optimize-focal", action="store_true",
                        help="let bundle adjustment refine the intrinsics "
                             "(see the warning on splatting.SfMConfig)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    work       = Path(args.work)
    frames_dir = work / "frames"
    rec_path   = work / "reconstruction.npz"
    t_start    = time.time()

    # -- 1. video -> keyframes ----------------------------------------------
    if "frames" in args.skip and frames_dir.is_dir():
        names = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
        print(f"[run] reusing {len(names)} keyframes in {frames_dir}")
    else:
        names = extract_keyframes(args.video, frames_dir, args.num_frames, args.max_width)

    if args.stop_after == "frames":
        return

    # -- 2. keyframes -> posed cameras + sparse points -----------------------
    if "sfm" in args.skip and rec_path.exists():
        rec = Reconstruction.load(rec_path)
        print(f"[run] reusing reconstruction with {rec.n_images} cameras")
    else:
        cfg = SfMConfig(focal_prior=args.focal_prior, optimize_focal=args.optimize_focal)
        rec = reconstruct(frames_dir, names, cfg, device=args.device)

    # A canonical frame before anything downstream sees it, so every learning rate and size threshold means the same thing on every capture.
    # Pruning comes first, because one camera far outside the capture would set the scale for all of them; the rest is a similarity
    # transform, so no image changes.
    rec.prune_outliers()
    scale = rec.normalize()
    tilt  = rec.level()
    print(f"[run] normalized the scene by {scale:.4f}, removed {tilt:.1f} deg of tilt")
    print(rec.summary())
    # Saved only now, after the canonical frame is fixed. A reconstruction on disk in a different frame from the `cameras.json` and
    # `splats.ply` beside it is a trap: everything that reads the three together -- a viewer, a measurement, a second training run --
    # silently places the cameras somewhere the model never was.
    rec.save(rec_path)
    write_cameras_json(rec, work / "cameras.json")

    if args.stop_after == "sfm":
        return

    # -- 3. posed images -> Gaussians ---------------------------------------
    ply_path = work / "splats.ply"
    if "train" in args.skip and ply_path.exists():
        print(f"[run] reusing {ply_path}")
        return

    cfg   = TrainConfig(iterations=args.iterations, resolution_scale=args.resolution_scale,
                        max_gaussians=args.max_gaussians)
    model = train(rec, frames_dir, work, cfg, device=args.device)
    save_gaussian_ply(ply_path, model)

    print(f"\n[run] done in {(time.time() - t_start) / 60.0:.1f} minutes")
    print(f"[run] Part 2 needs two files:\n        {ply_path}\n        {work / 'cameras.json'}")


if __name__ == "__main__":
    main()
