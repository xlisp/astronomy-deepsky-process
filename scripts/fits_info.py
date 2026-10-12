#!/usr/bin/env python
"""Inventory a folder of FITS subs: filter, exposure, gain, temperature, stats + total integration.

PixInsight equivalent: FITSHeader + Statistics + the file list of WBPP.

  python scripts/fits_info.py "/data/M27/*.fit"
  python scripts/fits_info.py lights/*.fit --stats --csv inventory.csv
"""
import argparse, glob, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at

KEYS = {"filter": ("FILTER",), "exp": ("EXPTIME", "EXPOSURE"), "gain": ("GAIN",),
        "temp": ("CCD-TEMP", "SET-TEMP"), "date": ("DATE-OBS",), "type": ("IMAGETYP", "FRAME"),
        "object": ("OBJECT",), "bin": ("XBINNING",)}


def first(hdr, keys, default=""):
    for k in keys:
        if k in hdr:
            return hdr[k]
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="FITS files or glob patterns")
    ap.add_argument("--stats", action="store_true", help="also read pixels: median, noise, star FWHM")
    ap.add_argument("--csv", help="write table to CSV")
    a = ap.parse_args()
    from astropy.io import fits

    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    rows, total = [], defaultdict(float)
    for f in files:
        hdr = fits.getheader(f)
        r = {"file": os.path.basename(f), "size": f"{hdr.get('NAXIS1')}x{hdr.get('NAXIS2')}"}
        r.update({k: first(hdr, v) for k, v in KEYS.items()})
        if "FOCALLEN" in hdr and "XPIXSZ" in hdr:
            r["scale"] = f"{206.265 * hdr['XPIXSZ'] / hdr['FOCALLEN']:.2f}\"/px"
        if a.stats:
            img = at.load_fits(f)
            st = at.detect_stars(img, max_stars=200)
            r.update(median=f"{float(at.np.median(img)):.1f}", noise=f"{at.noise_sigma(img):.2f}",
                     stars=len(st["x"]),
                     fwhm=f"{float(at.np.median(st['fwhm'])):.2f}" if len(st["x"]) else "-")
        rows.append(r)
        try:
            total[(str(r["filter"]) or "?", r["type"])] += float(r["exp"] or 0)
        except ValueError:
            pass

    cols = [c for c in rows[0] if any(str(r.get(c, "")) for r in rows)] if rows else []
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    print("\nIntegration per filter/type:")
    for (flt, typ), s in sorted(total.items()):
        n = sum(1 for r in rows if (str(r["filter"]) or "?", r["type"]) == (flt, typ))
        print(f"  {flt:<8} {typ:<12} {n:4d} subs  {s:8.0f} s = {s / 3600:.2f} h")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
