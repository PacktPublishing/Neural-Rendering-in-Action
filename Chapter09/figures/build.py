#!/usr/bin/env python3
"""Render every diagram source in this folder to a PNG.

    python figures/build.py

Each `*.dot` goes through Graphviz. The PNGs are build products, stamped with
their print resolution: edit the text sources, never the images.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DPI = 300                      # print resolution; also stamped into the PNG


def stamp_dpi(png: Path, dpi: int = DPI) -> None:
    """Record the physical resolution, so Word sizes the figure correctly."""
    from PIL import Image

    with Image.open(png) as im:
        im.save(png, dpi=(dpi, dpi))


def main() -> None:
    for src in sorted(HERE.glob("*.dot")):
        out = src.with_suffix(".png")
        subprocess.run(["dot", "-Tpng", f"-Gdpi={DPI}", str(src), "-o", str(out)],
                       check=True)
        stamp_dpi(out)
        print(f"[figures] {src.name} -> {out.name}")


if __name__ == "__main__":
    sys.exit(main())
