#!/usr/bin/env python
"""PixelMath / ChannelCombination / LinearFit in one tool: combine mono masters with expressions.

  # HOO bicolour, OIII linear-fitted to Ha first, stretched preview as JPG
  python scripts/pixelmath.py -i H=master_H.fits -i O=master_O.fits --linearfit H \\
         --r "H" --g "0.35*H+0.65*O" --b "O" -o hoo.fits --preview hoo.jpg

  # SHO (Hubble palette)
  python scripts/pixelmath.py -i S=master_S.fits -i H=master_H.fits -i O=master_O.fits \\
         --linearfit H --r "0.55*S+0.45*H" --g "0.6*H+0.4*O" --b O -o sho.fits

  # mono expression (e.g. synthetic luminance, star-subtraction, masks)
  python scripts/pixelmath.py -i L=master_L.fits -i R=R.fits -i G=G.fits -i B=B.fits \\
         --expr "0.5*L + 0.5*(R+G+B)/3" -o Lsyn.fits

Expressions are numpy: + - * / **, min/max/where/clip/sqrt/log/exp/abs, mean/median/mad(x).
LinearFit (PixInsight): fit each image as a*img+b to the reference so their background and signal
levels match - essential before mixing narrowband channels with very different strengths.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at

FUNCS = dict(min=np.minimum, max=np.maximum, where=np.where, clip=np.clip, sqrt=np.sqrt,
             log=np.log, exp=np.exp, abs=np.abs, mean=np.nanmean, median=np.nanmedian,
             mad=at.mad_std, mtf=at.mtf, pi=np.pi)


def linear_fit(img, ref, clip=(0.0, 0.92)):
    """Robust least-squares img -> a*img+b matching ref; uses pixels in a percentile window."""
    x, y = img[::3, ::3].ravel(), ref[::3, ::3].ravel()
    lo, hi = np.nanpercentile(y, [clip[0] * 100, clip[1] * 100])
    m = np.isfinite(x) & np.isfinite(y) & (y >= lo) & (y <= hi)
    a, b = np.polyfit(x[m], y[m], 1)
    for _ in range(3):  # reject outliers (stars that differ between filters)
        r = y - (a * x + b)
        s = at.mad_std(r[m])
        m &= np.abs(r) < 3 * s
        a, b = np.polyfit(x[m], y[m], 1)
    return a * img + b, a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--r"); ap.add_argument("--g"); ap.add_argument("--b")
    ap.add_argument("--expr", help="single (mono) expression")
    ap.add_argument("--linearfit", metavar="NAME", help="linear-fit all inputs to this one")
    ap.add_argument("--rescale", action="store_true", help="rescale result to [0,1]")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--preview", help="also save an STF-stretched JPG/PNG")
    a = ap.parse_args()

    env, hdr = {}, None
    for spec in a.input:
        name, path = spec.split("=", 1)
        if at.is_fits(path):
            env[name], h = at.load_fits(path, True)
            hdr = hdr if hdr is not None else h
        else:
            env[name] = at.load_image(path)
    if a.linearfit:
        ref = env[a.linearfit]
        for k in env:
            if k != a.linearfit:
                env[k], s, o = linear_fit(env[k], ref)
                print(f"linearfit {k} -> {a.linearfit}: {s:.4g}*{k} + {o:.4g}")

    def ev(expr):
        r = eval(expr, {"__builtins__": {}}, {**FUNCS, **env})
        return np.broadcast_to(np.asarray(r, np.float32), next(iter(env.values())).shape[:2])

    if a.expr:
        out = ev(a.expr)
    elif a.r and a.g and a.b:
        out = np.dstack([ev(a.r), ev(a.g), ev(a.b)])
    else:
        raise SystemExit("give --expr or all of --r --g --b")
    if a.rescale:
        out = at.normalize(out)
    at.save_image(a.out, out if at.is_fits(a.out) else at.stf(out), hdr,
                  f"pixelmath r={a.r} g={a.g} b={a.b} expr={a.expr}")
    if a.preview:
        at.save_image(a.preview, at.stf(out, linked=False))
    print("saved", a.out)


if __name__ == "__main__":
    main()
