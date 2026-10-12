#!/usr/bin/env python
"""Build master bias/dark/flat and calibrate light frames.

PixInsight equivalent: ImageIntegration (masters) + ImageCalibration (+ CosmeticCorrection).

  # 1. masters
  python scripts/calibrate.py master bias  -o master_bias.fits bias/*.fit
  python scripts/calibrate.py master dark  -o master_dark.fits darks/*.fit
  python scripts/calibrate.py master flat  -o master_flat_L.fits --bias master_bias.fits flats/L_*.fit
  # 2. lights:  cal = (light - dark') / (flat / mean(flat))
  python scripts/calibrate.py apply -o calibrated/ --bias master_bias.fits --dark master_dark.fits \\
         --flat master_flat_L.fits --cosmetic lights/L_*.fit

dark' = dark                                   if no bias given (dark must match exposure/temp)
dark' = bias + (dark - bias) * t_light/t_dark  if bias given     (dark scaling)
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def exptime(hdr):
    return float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 0)) or 0)


def cosmetic(img, k=8.0):
    """Replace hot/cold pixels that deviate > k sigma from their 3x3 median (no master dark needed)."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    med = np.median(swv(np.pad(img, 1, mode="edge"), (3, 3)), axis=(-2, -1))
    diff = img - med
    s = at.mad_std(diff)
    bad = np.abs(diff) > k * s
    out = img.copy()
    out[bad] = med[bad]
    return out, int(bad.sum())


def make_master(kind, files, out, bias=None, dark=None, k=3.0):
    frames, hdr0 = [], None
    for f in files:
        d, h = at.load_fits(f, with_header=True)
        hdr0 = hdr0 if hdr0 is not None else h
        if kind == "flat":
            if dark is not None:          # flat-dark already contains bias
                d = d - dark
            elif bias is not None:
                d = d - bias
            d = d / np.median(d)          # normalise each flat before combining
        frames.append(d)
        print(f"  {os.path.basename(f)} median={np.median(d):.4g}")
    master, rej = at.sigma_clip_mean(np.stack(frames), k, k)
    master = np.nan_to_num(master, nan=float(np.nanmedian(master)))
    at.save_fits(out, master, hdr0, f"master {kind} of {len(frames)} frames, sigma-clip {k}")
    print(f"== master {kind}: {len(frames)} frames, rejected {rej * 100:.2f}% -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("master")
    m.add_argument("kind", choices=["bias", "dark", "flat"])
    a2 = sub.add_parser("apply")
    a2.add_argument("--cosmetic", action="store_true", help="hot/cold pixel removal after calibration")
    a2.add_argument("--pedestal", type=float, default=0, help="add constant to avoid negative values")
    for p in (m, a2):
        p.add_argument("files", nargs="+")
        p.add_argument("-o", "--out", required=True, help="output file (master) or directory (apply)")
        p.add_argument("--bias"); p.add_argument("--dark"); p.add_argument("--flat")
        p.add_argument("--sigma", type=float, default=3.0)
    a = ap.parse_args()

    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    bias = at.load_fits(a.bias) if a.bias else None
    dark, dhdr = at.load_fits(a.dark, True) if a.dark else (None, None)
    if a.cmd == "master":
        return make_master(a.kind, files, a.out, bias, dark, a.sigma)

    flat = at.load_fits(a.flat) if a.flat else None
    if flat is not None:
        flat = flat / np.mean(flat[flat.shape[0] // 4: -flat.shape[0] // 4, flat.shape[1] // 4: -flat.shape[1] // 4])
        flat = np.where(flat > 0.05, flat, 1.0)
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        img, hdr = at.load_fits(f, with_header=True)
        steps = []
        if dark is not None:
            d = dark
            if bias is not None and exptime(dhdr) > 0 and exptime(hdr) > 0:
                d = bias + (dark - bias) * (exptime(hdr) / exptime(dhdr))
                steps.append(f"dark x{exptime(hdr) / exptime(dhdr):.3f}")
            else:
                steps.append("dark")
            img = img - d
        elif bias is not None:
            img = img - bias; steps.append("bias")
        if flat is not None:
            img = img / flat; steps.append("flat")
        if a.cosmetic:
            img, n = cosmetic(img); steps.append(f"cosmetic {n}px")
        img = img + a.pedestal
        out = at.out_name(f, "_c", ".fits", a.out)
        at.save_fits(out, img, hdr, "calibrated: " + ", ".join(steps))
        print(f"  {os.path.basename(f)} -> {out}  [{', '.join(steps)}]")


if __name__ == "__main__":
    main()
