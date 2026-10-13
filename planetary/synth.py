#!/usr/bin/env python
"""Simulate a planetary capture (SER) with known ground truth - test data for the whole pipeline.

  python planetary/synth.py -o demo/jup_a.ser --time 2026-10-13T18:30:00 --truth demo/truth_a.npy
  python planetary/synth.py -o demo/jup.ser --frames 400 --seeing 2.2 --bayer RGGB --bits 8

The planet is a textured oblate spheroid rendered with the same geometry WinJUPOS uses (orthographic
view, sub-earth latitude De, flattening, System II rotation), so captures at different times really
rotate and can be used to test derotation. Each frame then goes through a seeing model:

  tip/tilt jitter + slow drift      -> whole image moves (global alignment fixes it)
  smooth random displacement field  -> local warping ("boiling"; needs alignment points)
  blur that varies frame to frame   -> lucky imaging: only a few frames are sharp
  ... and across the field          -> different parts of the disc are sharp in different frames
  atmospheric dispersion            -> R and B shifted vertically (RGB align / ADC)
  Poisson + read noise, quantisation, optional Bayer mosaic
"""
import argparse, datetime as dt, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planetools as pt
at = pt.at


def band(lat, centre, width):
    return np.exp(-0.5 * ((lat - centre) / width) ** 2)


def oval(lat, lon, lat0, lon0, rlat, rlon):
    d = (lon - lon0 + 180) % 360 - 180
    return np.exp(-0.5 * ((lat - lat0) / rlat) ** 2 - 0.5 * (d / rlon) ** 2)


def jupiter_albedo(lat, lon):
    """Toy Jupiter: belts/zones with wavy edges, GRS, white ovals, festoons. lat, lon (System II) in deg."""
    rad = np.radians(lon)
    w = (1.4 * np.sin(3 * rad + 0.5) + 0.8 * np.sin(7 * rad + 2.0)) * band(lat, 15, 6) \
        + (1.2 * np.sin(4 * rad + 1.0) + 0.7 * np.sin(9 * rad)) * band(lat, -15, 6)
    la = lat + w
    belts = (0.50 * band(la, 13, 4.0) + 0.45 * band(la, -15, 4.5) + 0.25 * band(la, 27, 2.0)
             + 0.25 * band(la, -29, 2.2) + 0.12 * band(la, 38, 3) + 0.12 * band(la, -40, 3))
    fine = 0.05 * np.sin(np.radians(la * 23)) * np.cos(np.radians(lat * 5 + 2 * lon))
    polar = 0.45 / (1 + np.exp(-(np.abs(lat) - 52) / 5))
    fest = sum(0.35 * oval(lat, lon, 7, l0, 1.2, 4) for l0 in range(10, 360, 36))     # festoons on the NEB
    grs = oval(lat, lon, -22, 60, 4.5, 9)
    ring = oval(lat, lon, -22, 60, 6.5, 13) - grs
    whites = sum(oval(lat, lon, -33, l0, 1.6, 2.5) for l0 in (120, 165, 210, 300))
    v = 1 - belts - polar - fest + fine + 0.35 * ring + 0.45 * whites
    v = np.clip(v, 0.05, 1.3)
    red = 1.0 + 0.25 * belts + 0.20 * grs
    blue = 0.82 - 0.25 * belts - 0.30 * grs + 0.10 * whites
    return np.stack([v * red, v * (0.93 - 0.10 * grs), v * blue], -1) * (1 - 0.30 * grs[..., None])


def render(size, cx, cy, radius, cm_lon, flattening=0.06487, de=3.0, north=0.0, ss=2, limb_k=0.35):
    """Sharp RGB image of the planet; cm_lon = System II longitude of the central meridian (deg)."""
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n].astype(np.float64) + 0.5) / ss - 0.5
    u, v = pt.image_to_planet(x, y, cx, cy, radius, north)
    X, Y, Z, on = pt.ray_spheroid(u, v, flattening, de)
    lat, dlon = pt.surface_latlon(X, Y, Z, flattening)
    _, _, mu, _ = pt.project_spheroid(X, Y, Z, flattening, de)
    lon = (cm_lon - dlon) % 360                         # west longitude: CM - offset toward west
    img = jupiter_albedo(lat, lon) * (np.clip(mu, 0, 1) ** limb_k)[..., None]
    img[~on] = 0
    return img.reshape(size, ss, size, ss, 3).mean((1, 3)).astype(np.float32)


def smooth_field(rng, shape, cells, amp):
    return at.upsample_grid(rng.normal(0, amp, (cells, cells)).astype(np.float32), shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output .ser")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=float, default=50)
    ap.add_argument("--size", type=int, default=256, help="frame size (square)")
    ap.add_argument("--radius", type=float, default=70, help="equatorial radius, px")
    ap.add_argument("--time", default="2026-10-13T18:30:00", help="UTC start time")
    ap.add_argument("--cm0", type=float, default=40.0, help="System II CM longitude at --time (deg)")
    ap.add_argument("--de", type=float, default=3.0, help="sub-earth latitude (deg)")
    ap.add_argument("--north", type=float, default=0.0, help="pole angle in the image (deg ccw)")
    ap.add_argument("--seeing", type=float, default=1.8, help="median blur sigma, px")
    ap.add_argument("--jitter", type=float, default=4.0, help="tip/tilt rms, px")
    ap.add_argument("--drift", type=float, default=25.0, help="total tracking drift, px")
    ap.add_argument("--warp", type=float, default=0.9, help="local distortion rms, px")
    ap.add_argument("--dispersion", type=float, default=1.0, help="R/B vertical offset, px")
    ap.add_argument("--peak", type=float, default=1500, help="photo-electrons at the brightest pixel")
    ap.add_argument("--read-noise", type=float, default=2.5)
    ap.add_argument("--bits", type=int, default=8, choices=(8, 12, 16))
    ap.add_argument("--bayer", default="RGGB", help="RGGB/GRBG/... or 'none' for an RGB SER")
    ap.add_argument("--truth", help="save the sharp mid-capture image (centred) as .npy")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    t0 = pt.parse_time(a.time)
    rate = pt.rotation_rate("jupiter")
    times = [t0 + dt.timedelta(seconds=i / a.fps) for i in range(a.frames)]
    c = a.size / 2 - 0.5
    yy, xx = np.mgrid[0:a.size, 0:a.size].astype(np.float32)
    drift_dir = rng.normal(size=2)
    drift_dir /= np.linalg.norm(drift_dir)
    sharp, last_cm, frames = None, None, []
    for i, t in enumerate(times):
        cm = a.cm0 + rate * (t - t0).total_seconds() / 86400
        if sharp is None or abs(cm - last_cm) * np.pi / 180 * a.radius > 0.1:   # re-render when > 0.1 px
            sharp, last_cm = render(a.size, c, c, a.radius, cm, de=a.de, north=a.north), cm
        f = i / max(a.frames - 1, 1) - 0.5
        gy, gx = drift_dir * a.drift * f + rng.normal(0, a.jitter, 2)
        dy = gy + smooth_field(rng, sharp.shape[:2], 5, a.warp)
        dx = gx + smooth_field(rng, sharp.shape[:2], 5, a.warp)
        chans = []
        for ch, off in zip(range(3), (a.dispersion, 0.0, -a.dispersion)):   # red appears higher (toward zenith)
            chans.append(pt.bilinear(sharp[..., ch], yy - dy + off, xx - dx))
        img = np.dstack(chans)
        s1 = a.seeing * np.exp(rng.normal(0, 0.35))                 # frame-to-frame seeing
        s2 = s1 * np.exp(rng.normal(0, 0.35))
        mix = 1 / (1 + np.exp(-smooth_field(rng, img.shape[:2], 3, 2.0)))[..., None]   # patchy seeing
        img = mix * at.gaussian_blur(img, s1) + (1 - mix) * at.gaussian_blur(img, s2)
        e = rng.poisson(np.clip(img, 0, None) * a.peak / 1.25) + rng.normal(0, a.read_noise, img.shape)
        img = e / (a.peak / 1.25) * 0.8                             # brightest pixel at ~80% of full scale
        if a.bayer.lower() != "none":
            img = pt.mosaic(img, a.bayer)
        frames.append(pt.to_uint(img, a.bits))
    color = "RGB" if a.bayer.lower() == "none" else a.bayer.upper()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pt.write_ser(a.out, np.array(frames), color, a.bits, times, instrument="synth.py", telescope="simulated")
    mid = t0 + (times[-1] - t0) / 2
    print(f"{a.out}: {a.frames} frames {a.size}x{a.size} {color} {a.bits}-bit, "
          f"{times[0]:%H:%M:%S}..{times[-1]:%H:%M:%S} UTC")
    if a.truth:
        cm = a.cm0 + rate * (mid - t0).total_seconds() / 86400
        np.save(a.truth, render(a.size, c, c, a.radius, cm, de=a.de, north=a.north) * 0.8)
        print(f"truth at {mid:%H:%M:%S} (CM II = {cm:.2f} deg) -> {a.truth}")


if __name__ == "__main__":
    main()
