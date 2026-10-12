#!/usr/bin/env python
"""Grade subframes (FWHM, eccentricity, star count, background, noise) and reject outliers.

PixInsight equivalent: SubframeSelector (+ the weighting step of WBPP).

  python scripts/subframe_selector.py lights/*.fit --csv grades.csv --accepted good.txt
  python scripts/subframe_selector.py lights/*.fit --max-fwhm 5 --max-ecc 0.6 --kappa 2.5

Weight (PSF-SNR like):  w = (stars_flux_median / noise)^2 / fwhm^2, normalised to best = 1.
A frame is rejected when it is a > kappa*sigma outlier (FWHM/ecc/noise high, star count low)
or breaks the absolute --max-* limits.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--kappa", type=float, default=3.0, help="outlier threshold in robust sigmas")
    ap.add_argument("--max-fwhm", type=float, help="absolute FWHM limit in px")
    ap.add_argument("--max-ecc", type=float, help="absolute eccentricity limit (0=round)")
    ap.add_argument("--min-stars", type=int, default=10)
    ap.add_argument("--csv"); ap.add_argument("--accepted", help="write accepted file list")
    a = ap.parse_args()

    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    rows = []
    for f in files:
        img = at.load_image(f)
        st = at.detect_stars(img, max_stars=300)
        n = len(st["x"])
        lum = at.to_luminance(img)
        r = dict(file=f, stars=n, bkg=float(np.median(lum)), noise=at.noise_sigma(lum),
                 fwhm=float(np.median(st["fwhm"])) if n else np.nan,
                 hfr=float(np.median(st["hfr"])) if n else np.nan,
                 ecc=float(np.median(st["ecc"])) if n else np.nan,
                 flux=float(np.median(st["flux"])) if n else 0.0)
        r["weight"] = (r["flux"] / r["noise"]) ** 2 / r["fwhm"] ** 2 if n else 0.0
        rows.append(r)
        print(f"  {os.path.basename(f)}: stars={n} fwhm={r['fwhm']:.2f} ecc={r['ecc']:.2f} "
              f"bkg={r['bkg']:.4g} noise={r['noise']:.4g}")

    wmax = max(r["weight"] for r in rows) or 1
    def robust(k):
        v = np.array([r[k] for r in rows], float)
        return np.nanmedian(v), max(at.mad_std(v), 1e-9)
    stats = {k: robust(k) for k in ("fwhm", "ecc", "noise", "stars")}
    for r in rows:
        r["weight"] /= wmax
        why = []
        for k in ("fwhm", "ecc", "noise"):
            med, s = stats[k]
            if r[k] > med + a.kappa * s:
                why.append(f"{k} outlier")
        med, s = stats["stars"]
        if r["stars"] < max(a.min_stars, med - a.kappa * s):
            why.append("few stars")
        if a.max_fwhm and r["fwhm"] > a.max_fwhm: why.append("fwhm>max")
        if a.max_ecc and r["ecc"] > a.max_ecc: why.append("ecc>max")
        r["status"] = "REJECT: " + ",".join(why) if why else "ok"

    print(f"\n{'file':<40} {'stars':>5} {'fwhm':>6} {'ecc':>5} {'noise':>8} {'weight':>6}  status")
    for r in sorted(rows, key=lambda r: -r["weight"]):
        print(f"{os.path.basename(r['file'])[-40:]:<40} {r['stars']:5d} {r['fwhm']:6.2f} {r['ecc']:5.2f} "
              f"{r['noise']:8.3g} {r['weight']:6.3f}  {r['status']}")
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\naccepted {len(ok)}/{len(rows)}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.accepted:
        with open(a.accepted, "w") as fh:
            fh.writelines(r["file"] + "\n" for r in ok)


if __name__ == "__main__":
    main()
