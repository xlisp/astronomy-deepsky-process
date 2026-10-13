#!/usr/bin/env python
"""WinJUPOS-style derotation: undo the planet's rotation so images taken minutes apart can be combined.

  # how long may one video be before rotation blurs it?
  python planetary/derotate.py limit --planet jupiter --diameter-arcsec 45 --aperture-mm 254
  # check the automatic outline / orientation (writes a PNG with limb + lat/lon grid)
  python planetary/derotate.py measure 2026-10-13-1830_0-j.tif --de 3 --grid grid.png
  # several stacks of one session -> one image at the mean time ("Image derotation" / "Stack")
  python planetary/derotate.py stack 2026-10-13-18*.tif -o derot.tif --de 3
  # mono R, G, B (and L) shot one after another -> colour image at the G time ("De-rotation of R/G/B")
  python planetary/derotate.py rgb --r R.tif --g G.tif --b B.tif -o rgb.tif --de 3
  # derotate every frame of a long video to its middle time ("De-rotation of video streams")
  python planetary/derotate.py video long.ser -o long_derot.ser --de 3
  # cylindrical map from one or more images
  python planetary/derotate.py map 2026-10-13-18*.tif -o map.png --cm0 40 --de 3

Times come from --times (ISO, comma separated), else from file names in the WinJUPOS / FireCapture
convention YYYY-MM-DD-HHMM_T (T = tenths of a minute), else from SER frame timestamps.

Geometry (what WinJUPOS calls the "outline frame"):
  centre + equatorial radius ... measured automatically from the disc (or --radius)
  pole angle in the image ...... --north (deg, counter-clockwise from up); auto: from the disc's
                                 shape (Jupiter/Saturn are visibly oblate), the solution closest to 'up'
  mirror ....................... --mirror for star-diagonal images (east and west swapped)
  sub-earth latitude De ........ --de (from an ephemeris; Jupiter within +-4 deg, Saturn up to +-27)
Derotation maps each output pixel onto the spheroid, rotates it in longitude by
omega * (t_out - t_in), and projects it back into the source image (orthographic view).
"""
import argparse, datetime as dt, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planetools as pt
at = pt.at


# ----------------------------------------------------------------------------- geometry
def measure(img, a):
    d = pt.find_disk(img, a.frac, a.flattening)
    if d is None:
        sys.exit("no planet found")
    north = a.north
    if north is None:
        north = d["pole"] if d["pole"] <= 90 else d["pole"] - 180          # closest to 'up'
        if a.flattening < 0.03:                                            # Mars: shape says nothing
            north = 0.0
    return dict(cx=d["cx"], cy=d["cy"], a=a.radius or d["a"], north=north, mirror=a.mirror)


def derotate_image(img, delta, gs, go, f, de, mu_power=1.0):
    """Re-project img (geometry gs) to the rotation state delta degrees later, on geometry go.

    Returns (image, weight). weight = mu_src**mu_power on the disc (0 where the feature was behind
    the limb at the source time), 1 off the disc (sky, simply translated)."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = pt.image_to_planet(x, y, go["cx"], go["cy"], go["a"], go["north"], go["mirror"])
    us, vs, mu, valid, on = pt.rotate_view(u, v, delta, f, de)
    xs, ys = pt.planet_to_image(us, vs, gs["cx"], gs["cy"], gs["a"], gs["north"], gs["mirror"])
    xs = np.where(on, xs, x - go["cx"] + gs["cx"])
    ys = np.where(on, ys, y - go["cy"] + gs["cy"])
    out = pt.bilinear(img, ys, xs)
    wgt = np.where(on, np.where(valid, np.clip(mu, 0, 1) ** mu_power, 0.0), 1.0).astype(np.float32)
    return out, wgt


def combine(imgs, wgts):
    num = sum(i * (w[..., None] if i.ndim == 3 else w) for i, w in zip(imgs, wgts))
    den = sum(wgts)
    den = den[..., None] if imgs[0].ndim == 3 else den
    fallback = sum(imgs) / len(imgs)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), fallback)


def get_times(files, a, n=None):
    if a.times:
        ts = [pt.parse_time(s) for s in a.times.split(",")]
    else:
        ts = [pt.parse_time(os.path.basename(f)) for f in files]
    return ts


def omega(a):
    return pt.rotation_rate(a.planet, a.system) / 86400.0           # deg per second


def fmt_t(t):
    return t.strftime("%H:%M:%S")


# ----------------------------------------------------------------------------- commands
def cmd_limit(a):
    w = np.radians(pt.rotation_rate(a.planet, a.system)) / 86400.0   # rad/s
    print(f"{a.planet} rotation {pt.rotation_rate(a.planet, a.system):.3f} deg/day "
          f"(period {360 / pt.rotation_rate(a.planet, a.system) * 24:.4f} h)")
    if a.diameter_arcsec and a.aperture_mm:
        res = np.degrees(1.22 * a.wavelength_nm * 1e-9 / (a.aperture_mm * 1e-3)) * 3600
        v = w * a.diameter_arcsec / 2                                  # arcsec/s at disc centre
        print(f"Rayleigh limit {res:.3f}\" ; centre of a {a.diameter_arcsec}\" disc moves {v * 60:.3f}\"/min")
        print(f"max video length (blur <= resolution limit): {res / v:.0f} s  "
              f"(half that for a strict criterion: {res / v / 2:.0f} s)")
    if a.radius_px:
        print(f"at {a.radius_px} px radius the centre moves {w * a.radius_px * 60:.3f} px/min; "
              f"{a.blur_px} px blur after {a.blur_px / (w * a.radius_px):.0f} s")


def cmd_measure(a):
    for f in a.inputs:
        img = pt.load(f)
        g = measure(img, a)
        print(f"{os.path.basename(f)}: centre ({g['cx']:.2f}, {g['cy']:.2f})  equatorial radius {g['a']:.2f} px  "
              f"polar {g['a'] * (1 - a.flattening):.2f} px  north {g['north']:+.1f} deg  mirror {g['mirror']}")
        if a.grid:
            h, w = img.shape[:2]
            y, x = np.mgrid[0:h, 0:w].astype(np.float64)
            u, v = pt.image_to_planet(x, y, g["cx"], g["cy"], g["a"], g["north"], g["mirror"])
            X, Y, Z, on = pt.ray_spheroid(u, v, a.flattening, a.de)
            lat, lon = pt.surface_latlon(X, Y, Z, a.flattening)
            step = 15.0
            line = lambda q: np.abs((q + step / 2) % step - step / 2) < 0.8
            rim = on & ~(np.roll(on, 1, 0) & np.roll(on, -1, 0) & np.roll(on, 1, 1) & np.roll(on, -1, 1))
            base = img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)
            out = at.normalize(base, 0, np.percentile(base, 99.9)).copy()
            out[on & (line(lat) | line(lon))] = 0.5 * out[on & (line(lat) | line(lon))] + (0, 0.35, 0)
            out[on & (np.abs(lon) < 0.8)] = (1, 0.2, 0.2)                   # central meridian
            out[on & (np.abs(lat) < 0.8)] = (0.2, 0.6, 1)                   # equator
            out[rim] = (1, 1, 0)                                            # fitted limb
            pt.save(a.grid, out)
            print("grid overlay ->", a.grid, "(red = CM, blue = equator, yellow = fitted limb)")


def cmd_stack(a):
    imgs = [pt.load(f) for f in a.inputs]
    times = get_times(a.inputs, a)
    geoms = [measure(i, a) for i in imgs]
    if a.radius is None:                      # all images of one session share the scale
        r = float(np.median([g["a"] for g in geoms]))
        for g in geoms:
            g["a"] = r
    t0 = pt.parse_time(a.t0) if a.t0 else times[0] + sum((t - times[0] for t in times), dt.timedelta()) / len(times)
    ref = geoms[int(np.argmin([abs((t - t0).total_seconds()) for t in times]))]
    outs, wgts = [], []
    for f, img, g, t in zip(a.inputs, imgs, geoms, times):
        delta = omega(a) * (t0 - t).total_seconds()
        o, w = derotate_image(img, delta, g, ref, a.flattening, a.de, a.mu_power)
        outs.append(o)
        wgts.append(w)
        print(f"{os.path.basename(f)} {fmt_t(t)}  rotate {delta:+.2f} deg  centre ({g['cx']:.1f},{g['cy']:.1f})")
    res = combine(outs, wgts)
    pt.save(a.out, res)
    print(f"{len(imgs)} images derotated to {t0.isoformat()} -> {a.out}")
    if a.naive:
        pt.save(a.naive, sum(outs_naive(imgs, geoms, ref)) / len(imgs))
        print("plain (only centred) average ->", a.naive)


def outs_naive(imgs, geoms, ref):
    """Centre-aligned but NOT derotated, for comparison."""
    res = []
    for img, g in zip(imgs, geoms):
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        res.append(pt.bilinear(img, y - ref["cy"] + g["cy"], x - ref["cx"] + g["cx"]))
    return res


def cmd_rgb(a):
    files = [a.r, a.g, a.b] + ([a.l] if a.l else [])
    imgs = [pt.luminance(pt.load(f)) for f in files]
    times = get_times(files, a)
    geoms = [measure(i, a) for i in imgs]
    r = a.radius or float(np.median([g["a"] for g in geoms]))
    for g in geoms:
        g["a"] = r
    t0 = pt.parse_time(a.t0) if a.t0 else times[1]                # WinJUPOS default: time of G (or L)
    ref = geoms[1] if not a.l else geoms[3]
    chans = []
    for f, img, g, t in zip(files, imgs, geoms, times):
        delta = omega(a) * (t0 - t).total_seconds()
        o, w = derotate_image(img, delta, g, ref, a.flattening, a.de, 0)
        o = np.where(w > 0, o, img)          # behind-limb gaps: keep the original pixel (limb only)
        chans.append(o)
        print(f"{os.path.basename(f)} {fmt_t(t)}  rotate {delta:+.2f} deg")
    rgb = np.dstack(chans[:3])
    if a.l:                                  # LRGB: keep colour ratios, take brightness from L
        rgb = rgb * (chans[3] / np.maximum(rgb.mean(-1), 1e-6))[..., None]
    pt.save(a.out, rgb)
    print("->", a.out)


def cmd_video(a):
    src = pt.SerReader(a.inputs[0])
    if not src.times:
        sys.exit("video derotation needs SER frame timestamps")
    t0 = pt.parse_time(a.t0) if a.t0 else src.times[0] + (src.times[-1] - src.times[0]) / 2
    frames = [pt.debayer(src[i], src.bayer) if src.bayer else src[i] for i in range(len(src))]
    ref_img = np.mean(frames[:: max(1, len(frames) // 50)], 0)
    gref = measure(ref_img, a)
    out = []
    for i, (img, t) in enumerate(zip(frames, src.times)):
        d = pt.find_disk(img, a.frac, a.flattening)
        g = dict(gref, cx=d["cx"], cy=d["cy"]) if d else gref
        delta = omega(a) * (t0 - t).total_seconds()
        o, w = derotate_image(img, delta, g, dict(g), a.flattening, a.de, 0)
        out.append(pt.to_uint(np.where((w > 0)[..., None] if img.ndim == 3 else w > 0, o, img), src.depth))
    span = (src.times[-1] - src.times[0]).total_seconds()
    pt.write_ser(a.out, np.array(out), "RGB" if out[0].ndim == 3 else "MONO", src.depth, src.times)
    print(f"{len(out)} frames over {span:.0f} s ({omega(a) * span:.2f} deg of rotation) derotated to "
          f"{t0.isoformat()} -> {a.out}")


def cmd_map(a):
    imgs = [pt.load(f) for f in a.inputs]
    times = get_times(a.inputs, a)
    t0 = pt.parse_time(a.t0) if a.t0 else times[0]
    H = a.height
    lat = np.linspace(90, -90, H)[:, None] * np.ones((1, 2 * H))
    lon = np.linspace(360, 0, 2 * H)[None, :] * np.ones((H, 1))     # west longitude increases to the left
    b = 1 - a.flattening
    beta = np.arctan(b * np.tan(np.radians(lat)))                     # planetographic -> parametric
    num = np.zeros(lat.shape + ((3,) if imgs[0].ndim == 3 else ()), np.float32)
    den = np.zeros(lat.shape, np.float32)
    for img, t in zip(imgs, times):
        g = measure(img, a)
        cm = a.cm0 + omega(a) * (t - t0).total_seconds()
        dl = np.radians(cm - lon)                                     # offset west of the CM
        X, Y, Z = np.cos(beta) * np.cos(dl), np.cos(beta) * np.sin(dl), b * np.sin(beta)
        u, v, mu, vis = pt.project_spheroid(X, Y, Z, a.flattening, a.de)
        xs, ys = pt.planet_to_image(u, v, g["cx"], g["cy"], g["a"], g["north"], g["mirror"])
        wgt = np.where(vis & (mu > 0.15), mu ** 2, 0).astype(np.float32)
        s = pt.bilinear(img, ys, xs)
        num += s * (wgt[..., None] if s.ndim == 3 else wgt)
        den += wgt
        print(f"{fmt_t(t)}  CM = {cm % 360:.1f} deg")
    m = num / np.maximum(den, 1e-6)[(...,) + ((None,) if num.ndim == 3 else ())]
    pt.save(a.out, m)
    print(f"map {2 * H}x{H} (lon 360..0 left->right, lat +90..-90) -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--planet", default="jupiter", choices=sorted(pt.PLANETS))
    common.add_argument("--system", help="rotation system (Jupiter I/II/III, Saturn III/I)")
    common.add_argument("--de", type=float, default=0.0, help="sub-earth latitude, deg")
    common.add_argument("--north", type=float, help="pole angle in the image, deg ccw from up (auto)")
    common.add_argument("--mirror", action="store_true", help="mirror-reversed image")
    common.add_argument("--radius", type=float, help="equatorial radius in px (auto)")
    common.add_argument("--flattening", type=float, help="override planet flattening")
    common.add_argument("--frac", type=float, default=0.15, help="limb threshold (fraction of peak)")
    common.add_argument("--times", help="comma separated ISO times, one per input")
    common.add_argument("--t0", help="output time (default: mean / G / middle)")

    p = sub.add_parser("limit", parents=[common], help="max capture duration")
    p.add_argument("--diameter-arcsec", type=float)
    p.add_argument("--aperture-mm", type=float)
    p.add_argument("--wavelength-nm", type=float, default=550)
    p.add_argument("--radius-px", type=float)
    p.add_argument("--blur-px", type=float, default=1.0)
    p = sub.add_parser("measure", parents=[common], help="fit outline, optional grid overlay")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--grid")
    p = sub.add_parser("stack", parents=[common], help="derotate images to one time and average")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--mu-power", type=float, default=1.0, help="weight = cos(emission)^p")
    p.add_argument("--naive", help="also save a centred-only average for comparison")
    p = sub.add_parser("rgb", parents=[common], help="combine R,G,B(,L) taken at different times")
    for c in "rgbl":
        p.add_argument(f"--{c}", required=c != "l")
    p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("video", parents=[common], help="derotate every frame of a SER")
    p.add_argument("inputs", nargs=1)
    p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("map", parents=[common], help="cylindrical map")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--cm0", type=float, default=0.0, help="CM longitude at the first image (deg)")
    p.add_argument("--height", type=int, default=360)
    a = ap.parse_args()
    if a.flattening is None:
        a.flattening = pt.PLANETS[a.planet]["flattening"]
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
