#!/usr/bin/env python
"""Linear -> non-linear stretch: PixInsight AutoSTF (+HistogramTransformation), arcsinh, or manual MTF.

  python scripts/stretch.py master_L.fits -o L.jpg                      # AutoSTF
  python scripts/stretch.py rgb.fits -o rgb.tif --unlinked              # per-channel STF (colour cast fix)
  python scripts/stretch.py rgb.fits -o rgb.jpg --method asinh --beta 300   # keeps star colour
  python scripts/stretch.py L.fits -o L.png --method mtf --black 0.02 --mid 0.08
  python scripts/stretch.py L.fits -o L.png --target 0.15 --shadows -2  # darker background

AutoSTF:  c0 = median + shadows*MADN ;  m = MTF(target, median - c0)
          out = MTF(m, (x - c0)/(1 - c0))   with   MTF(m,x) = (m-1)x / ((2m-1)x - m)
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--out")
    ap.add_argument("--method", choices=["stf", "asinh", "mtf"], default="stf")
    ap.add_argument("--target", type=float, default=0.25, help="STF target background")
    ap.add_argument("--shadows", type=float, default=-2.8, help="STF shadows clipping in MADN")
    ap.add_argument("--unlinked", action="store_true", help="stretch RGB channels independently")
    ap.add_argument("--beta", type=float, default=500, help="asinh softening")
    ap.add_argument("--black", type=float, default=0.0); ap.add_argument("--mid", type=float, default=0.5)
    ap.add_argument("--hist", help="save before/after histogram PNG (matplotlib)")
    a = ap.parse_args()

    img = at.load_image(a.input)
    if a.method == "stf":
        out = at.stf(img, a.shadows, a.target, linked=not a.unlinked)
    elif a.method == "asinh":
        out = at.asinh_stretch(img, a.beta)
    else:
        x = at.normalize(img)
        out = at.mtf(a.mid, np.clip((x - a.black) / (1 - a.black), 0, 1))
    dst = a.out or at.out_name(a.input, "_stretched", ".jpg")
    at.save_image(dst, out, history=f"stretch {a.method}")
    print(f"saved {dst}  (output median {np.median(out):.3f})")
    if a.hist:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for axi, d, t in ((ax[0], at.normalize(img), "linear"), (ax[1], out, a.method)):
            axi.hist(d.ravel()[::17], 512, range=(0, 1), color="0.3", log=True)
            axi.set_title(t)
        fig.tight_layout(); fig.savefig(a.hist, dpi=100)
        print("saved", a.hist)


if __name__ == "__main__":
    main()
