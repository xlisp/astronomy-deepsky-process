#!/usr/bin/env python
"""Background neutralisation + star-based white balance for a linear RGB image.

PixInsight equivalent: BackgroundNeutralization + ColorCalibration (a simplified, catalogue-free
version of SPCC: the median star is assumed to be white, close to a G2V average).

  python scripts/color_calibration.py rgb.fits -o rgb_cc.fits
  python scripts/color_calibration.py -r master_R.fits -g master_G.fits -b master_B.fits -o rgb_cc.fits

Steps (all on LINEAR data, before any stretch):
  1. background of each channel = sigma-clipped median of the darkest pixels -> subtract, so the
     sky becomes neutral grey (R=G=B)
  2. detect unsaturated stars on the luminance, measure aperture flux in R, G, B
  3. scale R and B so median(R/G) = median(B/G) = 1
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def background_level(ch, k=2.0):
    v = ch[np.isfinite(ch)][::7]
    v = v[v < np.percentile(v, 50)]          # sky-dominated half
    for _ in range(5):
        m, s = np.median(v), at.mad_std(v)
        v = v[np.abs(v - m) < k * s]
    return float(np.median(v))


def aperture_flux(ch, x, y, r=5, r_in=8, r_out=12):
    out = []
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.hypot(yy, xx)
    h, w = ch.shape
    for xi, yi in zip(np.round(x).astype(int), np.round(y).astype(int)):
        if not (r_out <= yi < h - r_out and r_out <= xi < w - r_out):
            out.append(np.nan); continue
        st = ch[yi - r_out:yi + r_out + 1, xi - r_out:xi + r_out + 1]
        sky = np.median(st[(rr >= r_in) & (rr <= r_out)])
        out.append(float((st[rr <= r] - sky).sum()))
    return np.array(out)


def calibrate(rgb, max_stars=300):
    rgb = rgb.astype(np.float32).copy()
    bg = [background_level(rgb[..., c]) for c in range(3)]
    rgb -= np.array(bg, np.float32)
    st = at.detect_stars(rgb.mean(-1), k=10, max_stars=max_stars)
    f = np.stack([aperture_flux(rgb[..., c], st["x"], st["y"]) for c in range(3)], 1)
    ok = np.all(np.isfinite(f) & (f > 0), 1)
    if ok.sum() < 5:
        raise SystemExit(f"only {ok.sum()} usable stars for white balance")
    rg, bgr = np.median(f[ok, 0] / f[ok, 1]), np.median(f[ok, 2] / f[ok, 1])
    scale = np.array([1 / rg, 1.0, 1 / bgr], np.float32)
    print(f"background R,G,B = {bg[0]:.4g}, {bg[1]:.4g}, {bg[2]:.4g}")
    print(f"{ok.sum()} stars: R/G={rg:.3f} B/G={bgr:.3f} -> channel scale {[round(float(v), 3) for v in scale]}")
    return rgb * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="RGB FITS/TIFF (linear)")
    ap.add_argument("-r"); ap.add_argument("-g"); ap.add_argument("-b")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.input:
        rgb = at.load_image(a.input)
    else:
        rgb = np.dstack([at.load_image(p) for p in (a.r, a.g, a.b)])
    out = calibrate(rgb)
    if not at.is_fits(a.out):
        out = at.stf(out)       # non-FITS output: show it stretched
    at.save_image(a.out, out, history="background neutralisation + star white balance")
    print("saved", a.out)


if __name__ == "__main__":
    main()
