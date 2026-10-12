#!/usr/bin/env python
"""Make an animated GIF of subframes to spot clouds, satellites, bad tracking, dew.

PixInsight equivalent: Blink.

  python scripts/blink.py "lights/*.fit" -o blink.gif --bin 4
  python scripts/blink.py lights/*.fit -o blink.gif --per-frame --fps 4

By default all frames share the STF of the first frame, so changes in sky brightness (clouds, dawn,
moon) are visible; --per-frame auto-stretches each one to compare structure only.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def binning(img, b):
    if b <= 1:
        return img
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    v = img[:h, :w]
    return v.reshape(h // b, b, w // b, b, *v.shape[2:]).mean((1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+"); ap.add_argument("-o", "--out", default="blink.gif")
    ap.add_argument("--bin", type=int, default=4, help="downsample factor")
    ap.add_argument("--fps", type=float, default=3)
    ap.add_argument("--per-frame", action="store_true")
    a = ap.parse_args()
    from PIL import Image, ImageDraw

    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    frames, ref = [], None
    for f in files:
        x = binning(at.to_luminance(at.load_image(f)), a.bin)
        if ref is None or a.per_frame:
            lo, hi = float(np.min(x)), float(np.max(x))
            ref = (lo, hi) + at.stf_params(at.normalize(x, lo, hi))
        lo, hi, c0, m = ref
        y = at.mtf(m, np.clip((at.normalize(x, lo, hi) - c0) / (1 - c0), 0, 1))
        im = Image.fromarray((y * 255).astype(np.uint8)).convert("P")
        ImageDraw.Draw(im).text((8, 8), os.path.basename(f), fill=255)
        frames.append(im)
        print(" ", os.path.basename(f), f"median={np.median(x):.1f}")
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f"saved {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
