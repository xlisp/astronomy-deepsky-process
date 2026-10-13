#!/usr/bin/env python
"""Pre-process a planetary video like PIPP: debayer, detect, reject, centre, crop, sort by quality.

  python planetary/prep.py capture.ser -o capture_pipp.ser --crop 200
  python planetary/prep.py capture.ser -o best.ser --crop 240 --keep 50% --sort
  python planetary/prep.py "frames/*.png" -o centred.ser --crop 300 --no-debayer

Per frame:
  1. debayer (bilinear) if the SER header says it is a Bayer stream (or --bayer RGGB)
  2. find the planet: threshold between sky and planet brightness, centroid of the blob
  3. reject:  no planet / planet touching the frame edge / disc area or brightness far from the median
  4. crop a --crop window centred on the planet at an INTEGER pixel offset (no resampling, so no
     interpolation blur - the stacker does all sub-pixel work exactly once)
  5. quality score (gradient energy); optionally keep the best N or N% and sort best-first
Output keeps the per-frame timestamps (needed by WinJUPOS-style derotation) and writes a CSV log.
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planetools as pt


def crop_centered(img, cx, cy, size):
    """Integer crop (pads with zeros if the window leaves the frame)."""
    h, w = img.shape[:2]
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, h), min(x0 + size, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def keep_count(spec, n):
    if spec is None:
        return n
    return max(1, int(round(n * float(spec[:-1]) / 100))) if spec.endswith("%") else min(n, int(spec))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".ser / .avi / glob of images")
    ap.add_argument("-o", "--out", required=True, help="output .ser (or a directory for PNG/TIFF frames)")
    ap.add_argument("--crop", type=int, default=0, help="output size (square); 0 = 1.6 x planet diameter")
    ap.add_argument("--bayer", help="force Bayer pattern (RGGB, GRBG, GBRG, BGGR)")
    ap.add_argument("--no-debayer", action="store_true")
    ap.add_argument("--frac", type=float, default=0.2, help="detection threshold, fraction sky->peak")
    ap.add_argument("--min-diameter", type=float, default=10, help="reject if planet smaller (px)")
    ap.add_argument("--allow-edge", action="store_true", help="keep frames where the planet touches the edge")
    ap.add_argument("--max-dev", type=float, default=0.25,
                    help="reject if disc area or brightness deviates > this fraction from the median")
    ap.add_argument("--keep", help="keep the best N frames, or N%%")
    ap.add_argument("--sort", action="store_true", help="write frames best-first (PIPP 'quality order')")
    ap.add_argument("--sigma", type=float, default=1.2, help="noise-robust smoothing for the quality score")
    ap.add_argument("--bits", type=int, choices=(8, 16), help="output bit depth (default: as input)")
    a = ap.parse_args()

    src = pt.open_frames(a.input)
    bayer = None if a.no_debayer else (a.bayer or src.bayer)
    n = len(src)
    print(f"{a.input}: {n} frames, bayer={bayer}")

    info, keep = [], []
    for i in range(n):
        img = src[i]
        if bayer:
            img = pt.debayer(img, bayer)
        d = pt.find_disk(img, a.frac)
        row = dict(frame=i, status="ok", cx="", cy="", diameter="", mean="", quality="")
        if d is None:
            row["status"] = "no planet"
        else:
            row.update(cx=round(d["cx"], 2), cy=round(d["cy"], 2), diameter=round(2 * d["a"], 1))
            y0, y1, x0, x1 = d["bbox"]
            row["mean"] = float(pt.luminance(img)[y0:y1 + 1, x0:x1 + 1].mean())
            if 2 * d["a"] < a.min_diameter:
                row["status"] = "too small"
            elif d["edge"] and not a.allow_edge:
                row["status"] = "touches edge"
        info.append(row)
        keep.append((img, d) if row["status"] == "ok" else None)

    ok = [r for r in info if r["status"] == "ok"]
    if not ok:
        sys.exit("no usable frames")
    med_d = np.median([r["diameter"] for r in ok])
    med_b = np.median([r["mean"] for r in ok])
    size = a.crop or int(np.ceil(med_d * 1.6 / 8) * 8)
    for r in ok:          # partial frames, cloud, eclipse by a moon shadow ... show up as outliers
        if abs(r["diameter"] / med_d - 1) > a.max_dev or abs(r["mean"] / med_b - 1) > a.max_dev:
            r["status"] = "outlier"
            keep[r["frame"]] = None

    crops, idx = [], []
    for i, k in enumerate(keep):
        if k is None:
            continue
        img, d = k
        c = crop_centered(img, d["cx"], d["cy"], size)
        info[i]["quality"] = pt.frame_quality(c, a.sigma)
        crops.append(c)
        idx.append(i)
    q = np.array([info[i]["quality"] for i in idx])
    order = np.argsort(-q) if a.sort else np.arange(len(idx))
    nk = keep_count(a.keep, len(idx))
    if nk < len(idx):
        best = set(np.argsort(-q)[:nk].tolist())
        order = [j for j in order if j in best]
        for j in range(len(idx)):
            if j not in best:
                info[idx[j]]["status"] = "low quality"

    out = [crops[j] for j in order]
    times = [src.times[idx[j]] for j in order] if src.times and all(src.times) else None
    depth = a.bits or (8 if getattr(src, "depth", 8) <= 8 else 16)
    if a.out.lower().endswith(".ser"):
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        color = "RGB" if out[0].ndim == 3 else "MONO"
        pt.write_ser(a.out, np.array([pt.to_uint(o, depth) for o in out]), color, depth, times)
    else:
        os.makedirs(a.out, exist_ok=True)
        ext = ".png" if depth == 8 else ".tif"
        for k, o in enumerate(out):
            name = pt.winjupos_name(times[k], "j", f"_{k:05d}") if times else f"frame_{k:05d}"
            pt.at.save_image(os.path.join(a.out, name + ext), o)
    log = os.path.splitext(a.out.rstrip("/"))[0] + "_pipp.csv"
    with open(log, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(info[0]))
        wr.writeheader()
        wr.writerows(info)
    stats = {}
    for r in info:
        stats[r["status"]] = stats.get(r["status"], 0) + 1
    print(f"crop {size}x{size}, planet diameter ~{med_d:.0f} px; " + ", ".join(f"{k}: {v}" for k, v in stats.items()))
    print(f"wrote {len(out)} frames -> {a.out}  (log {log})")


if __name__ == "__main__":
    main()
