#!/usr/bin/env python
"""Shrink stars (morphological erosion through a star mask) and/or export the star mask.

PixInsight equivalent: StarMask + MorphologicalTransformation (Erosion / Morphological Selection).

  python scripts/star_reduce.py M27_LRGB.tif -o M27_small_stars.tif --size 3 --amount 0.7
  python scripts/star_reduce.py M27_LRGB.tif --mask-only -o starmask.png

Works best on STRETCHED images. The mask comes from small-scale structure (image minus a blurred copy)
above k*noise, so nebula detail is protected while stars are eroded with a size x size minimum filter.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--out")
    ap.add_argument("--size", type=int, default=3, help="erosion kernel (odd)")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--amount", type=float, default=0.7, help="0..1")
    ap.add_argument("--k", type=float, default=4.0, help="star detection threshold (noise sigmas)")
    ap.add_argument("--grow", type=int, default=2, help="mask dilation px")
    ap.add_argument("--mask-only", action="store_true")
    a = ap.parse_args()

    img = at.load_image(a.input)
    mask = at.star_mask(img, k=a.k, grow=a.grow)
    print(f"star mask covers {(mask > 0.5).mean() * 100:.1f}% of the image")
    if a.mask_only:
        dst = a.out or at.out_name(a.input, "_starmask", ".png")
        at.save_image(dst, mask)
        return print("saved", dst)
    er = img
    for _ in range(a.iters):
        er = np.dstack([at.min_filter(er[..., c], a.size) for c in range(3)]) if img.ndim == 3 \
            else at.min_filter(er, a.size)
        er = at.gaussian_blur(er, 0.6)   # soften the square kernel footprint
    m = (mask * a.amount)[..., None] if img.ndim == 3 else mask * a.amount
    out = m * np.minimum(er, img) + (1 - m) * img
    dst = a.out or at.out_name(a.input, "_starreduced")
    at.save_image(dst, out)
    print("saved", dst)


if __name__ == "__main__":
    main()
