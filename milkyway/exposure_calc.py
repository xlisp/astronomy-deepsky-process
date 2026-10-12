#!/usr/bin/env python
"""Longest untracked exposure before stars trail: 500 rule and NPF rule (no dependencies).

  python milkyway/exposure_calc.py --focal 14 --fnum 2.8 --pixel 5.9 --crop 1.0
  python milkyway/exposure_calc.py --focal 24 --fnum 1.4 --pixel 3.76 --dec -30   # Galactic core from China

  500 rule : t = 500 / (focal * crop)                     (film era, ~2-3 px trails on modern sensors)
  NPF rule : t = (35 * N + 30 * p) / focal                (Frederic Michaud, simplified, pinpoint stars)
  both are divided by cos(declination): stars near the celestial pole move slower.
"""
import argparse
import math


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--focal", type=float, required=True, help="real focal length mm")
    ap.add_argument("--fnum", type=float, required=True, help="aperture f-number")
    ap.add_argument("--pixel", type=float, required=True, help="pixel pitch um")
    ap.add_argument("--crop", type=float, default=1.0, help="crop factor (APS-C 1.5, M43 2.0)")
    ap.add_argument("--dec", type=float, default=0.0, help="target declination deg (Galactic core ~ -29)")
    ap.add_argument("--frames", type=int, default=20, help="planned number of frames")
    a = ap.parse_args()

    k = 1 / max(math.cos(math.radians(a.dec)), 1e-3)
    t500 = 500 / (a.focal * a.crop) * k
    tnpf = (35 * a.fnum + 30 * a.pixel) / a.focal * k
    scale = 206.265 * a.pixel / a.focal
    drift = 15.04 * math.cos(math.radians(a.dec)) / scale       # px per second
    print(f"plate scale {scale:.1f}\"/px, star drift {drift:.2f} px/s at dec {a.dec:+.0f} deg")
    print(f"500 rule : {t500:5.1f} s  -> trail {t500 * drift:4.1f} px")
    print(f"NPF rule : {tnpf:5.1f} s  -> trail {tnpf * drift:4.1f} px   (recommended)")
    print(f"{a.frames} x {tnpf:.0f} s = {a.frames * tnpf / 60:.1f} min total, "
          f"stack SNR gain sqrt({a.frames}) = x{math.sqrt(a.frames):.1f}")
    print(f"sky rotation during the sequence: {a.frames * tnpf / 240:.2f} deg -> needs star alignment")


if __name__ == "__main__":
    main()
