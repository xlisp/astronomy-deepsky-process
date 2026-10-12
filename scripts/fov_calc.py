#!/usr/bin/env python
"""Plate scale / field of view / sampling calculator (no dependencies).

  python scripts/fov_calc.py --focal 1197 --pixel 2.4 --sensor 5544x3694 --aperture 200 --seeing 3
  python scripts/fov_calc.py --focal 1197 --pixel 2.4 --sensor 5544x3694 --target 8x5.7

Rules of thumb:
  * plate scale ["/px] = 206.265 * pixel[um] / focal[mm]
  * well sampled when star FWHM spans ~2-3 px (Nyquist); < 2 px under-sampled (blocky stars),
    > 3.5 px over-sampled (bin 2x2 or drizzle less).
  * Dawes limit ["] = 116 / aperture[mm]
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--focal", type=float, required=True, help="focal length mm (incl. reducer)")
    ap.add_argument("--pixel", type=float, required=True, help="pixel size um")
    ap.add_argument("--sensor", default=None, help="WxH in pixels, e.g. 5544x3694")
    ap.add_argument("--aperture", type=float, help="aperture mm")
    ap.add_argument("--seeing", type=float, default=3.0, help="expected FWHM arcsec (default 3)")
    ap.add_argument("--target", help="target size in arcmin, e.g. 8x5.7 (M27)")
    a = ap.parse_args()

    for b in (1, 2, 3):
        scale = 206.265 * a.pixel * b / a.focal
        px = a.seeing / scale
        verdict = "under-sampled" if px < 2 else "over-sampled" if px > 3.5 else "good"
        line = f"bin{b}: {scale:5.2f}\"/px  seeing {a.seeing}\" = {px:4.1f} px FWHM ({verdict})"
        if a.sensor:
            w, h = (int(v) // b for v in a.sensor.lower().split("x"))
            fw, fh = w * scale / 60, h * scale / 60
            line += f"  FOV {fw:.1f}' x {fh:.1f}' ({fw / 60:.2f} x {fh / 60:.2f} deg)"
            if a.target:
                tw, th = (float(v) for v in a.target.lower().split("x"))
                line += f"  target fills {100 * tw / fw:.0f}% x {100 * th / fh:.0f}%"
        print(line)
    if a.aperture:
        print(f"f/{a.focal / a.aperture:.1f}  Dawes limit {116 / a.aperture:.2f}\"  "
              f"Airy disk (550nm) {2.44 * 550e-6 / a.aperture * 206265:.2f}\"")


if __name__ == "__main__":
    main()
