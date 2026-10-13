#!/usr/bin/env python
"""Lucky-imaging stacker in the style of AutoStakkert!4: quality ranking + multi-point alignment.

  python planetary/stack.py capture_pipp.ser -o stack.tif                      # 25% best, AP 32 px
  python planetary/stack.py capture_pipp.ser -o stack.tif --keep 10% --ap-size 24 --upscale 1.5
  python planetary/stack.py capture_pipp.ser -o global.tif --no-local          # single-point (Registax-like)

Algorithm (AS!4 names in brackets)
  1. quality of every frame = gradient energy after a small Gaussian blur          [Quality estimator,
     (the blur is the "noise robust" setting: it stops noise counting as detail)    Noise Robust]
  2. global alignment of every frame to the best one by phase correlation          [Surface / Planet COG]
  3. reference = average of the best --ref-frames frames, aligned again to it      [Reference frame]
  4. alignment points (APs): a grid of overlapping boxes (50% overlap) over the     [AP size / Min Bright]
     disc, skipped where the reference is fainter than --min-bright
  5. for EVERY AP separately:
       - local quality of each frame inside the AP; keep the best --keep frames    [Local quality, % to stack]
       - local shift of each kept frame by phase correlation against the reference
       - resample the ORIGINAL frames once (global + local shift), average
  6. blend AP patches with periodic Hann windows (they sum to 1 at 50% overlap),
     fill parts without APs from the globally aligned stack
  7. optional --upscale (AS!4 "Resampled 1.5x/3x": sample the frames on a finer grid)
Output is LINEAR (no sharpening): continue with wavelets.py (RegiStax) and derotate.py (WinJUPOS).
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planetools as pt


def pct(spec, n):
    if spec.endswith("%"):
        return max(1, int(round(n * float(spec[:-1]) / 100)))
    return max(1, min(n, int(spec)))


def load_frames(path, bayer=None, limit=None):
    src = pt.open_frames(path)
    bayer = bayer or src.bayer
    n = len(src) if limit is None else min(limit, len(src))
    F = []
    for i in range(n):
        img = src[i]
        F.append(pt.debayer(img, bayer) if (bayer and img.ndim == 2) else img)
    F = np.array(F, np.float32)
    times = src.times[:n] if src.times and all(src.times) else None
    return (F if F.ndim == 4 else F[..., None]), times


def shifts_to(ref, L, chunk=24, power=0.5):
    """Global shifts of every frame relative to ref (batched FFT to bound memory)."""
    dy, dx = np.zeros(len(L)), np.zeros(len(L))
    for s in range(0, len(L), chunk):
        dy[s:s + chunk], dx[s:s + chunk], _ = pt.phase_correlate(ref, L[s:s + chunk], power)
    return dy, dx


def align(L, dy, dx):
    h, w = L.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.array([pt.bilinear(L[k], yy + dy[k], xx + dx[k]) for k in range(len(L))], np.float32)


def box_sums(E, y0, y1, x0, x1):
    """Sum of E[:, y0:y1, x0:x1] for every frame using an integral image (E: (N,H,W) cumsum'ed)."""
    return E[:, y1, x1] - E[:, y0, x1] - E[:, y1, x0] + E[:, y0, x0]


def stack(F, keep=0.25, ap_size=32, min_bright=0.15, ref_frames=None, sigma=1.0, upscale=1.0,
          local=True, max_shift=None, power=0.5, verbose=True):
    """F: (N,H,W,C) float frames. Returns dict(image, global_image, quality, order, shifts, aps)."""
    n, h, w, nc = F.shape
    L = F.mean(-1)
    log = print if verbose else (lambda *a, **k: None)

    # 1. global quality
    q = np.array([pt.frame_quality(l, sigma) for l in L])
    order = np.argsort(-q)
    nk = max(1, int(round(keep * n)))
    nr = ref_frames or max(3, min(50, nk // 2))
    log(f"{n} frames {w}x{h}x{nc}; quality best/median/worst = "
        f"{q[order[0]] / np.median(q):.2f}/1.00/{q[order[-1]] / np.median(q):.2f} (relative)")

    # 2-3. global alignment and reference
    dy, dx = shifts_to(L[order[0]], L, power=power)
    ref = align(L[order[:nr]], dy[order[:nr]], dx[order[:nr]]).mean(0)
    dy, dx = shifts_to(ref, L, power=power)
    A = align(L, dy, dx)
    ref = A[order[:nr]].mean(0)
    log(f"global shifts: rms {np.sqrt(np.mean((dy - dy.mean()) ** 2 + (dx - dx.mean()) ** 2)):.2f} px, "
        f"range y {dy.min():+.1f}..{dy.max():+.1f}, x {dx.min():+.1f}..{dx.max():+.1f}; reference = best {nr}")

    sc = float(upscale)
    ho, wo = int(round(h * sc)), int(round(w * sc))
    oy, ox = ((np.arange(ho) + 0.5) / sc - 0.5).astype(np.float32), ((np.arange(wo) + 0.5) / sc - 0.5).astype(np.float32)

    # globally aligned stack of the best frames (also the fallback where there are no APs)
    gl = np.zeros((ho, wo, nc), np.float32)
    for k in order[:nk]:
        gl += pt.bilinear(F[k], oy[:, None] + dy[k], ox[None, :] + dx[k])
    gl /= nk
    result = dict(global_image=gl, quality=q, order=order, shifts=(dy, dx), aps=[], reference=ref)
    if not local:
        result["image"] = gl
        return result

    # 4. alignment-point grid (in output pixels; so = AP size on the output grid, must be even)
    so = int(round(ap_size * sc / 2)) * 2
    step = so // 2
    win = np.outer(pt.hann(so), pt.hann(so))[..., None]
    max_shift = ap_size / 4 if max_shift is None else max_shift
    E = np.zeros((n, h + 1, w + 1), np.float64)                # integral images of local contrast
    for k in range(n):
        E[k, 1:, 1:] = pt.gradient_energy(A[k], sigma).cumsum(0).cumsum(1)
    norm = (L.reshape(n, -1).mean(1) + 1e-9) ** 2              # transparency-independent quality
    num = np.zeros((ho + 2 * so, wo + 2 * so, nc), np.float32)
    wsum = np.zeros((ho + 2 * so, wo + 2 * so, 1), np.float32)
    thr = min_bright * float(ref.max())
    n_rej = 0
    for Y0 in range(-step, ho, step):
        for X0 in range(-step, wo, step):
            yi, xi = oy[0] + (Y0 + np.arange(so)) / sc, ox[0] + (X0 + np.arange(so)) / sc   # input coords
            iy0, iy1 = int(np.clip(np.floor(yi[0]), 0, h - 1)), int(np.clip(np.ceil(yi[-1]) + 1, 1, h))
            ix0, ix1 = int(np.clip(np.floor(xi[0]), 0, w - 1)), int(np.clip(np.ceil(xi[-1]) + 1, 1, w))
            if iy1 - iy0 < ap_size // 2 or ix1 - ix0 < ap_size // 2:
                continue
            if ref[iy0:iy1, ix0:ix1].mean() < thr:
                continue
            # 5a. local quality -> best frames for this AP
            lq = box_sums(E, iy0, iy1, ix0, ix1) / norm
            sel = np.argsort(-lq)[:nk]
            # 5b. local shift of the kept frames (on the globally aligned luminance, input grid)
            cy, cx = (iy0 + iy1) / 2, (ix0 + ix1) / 2
            gy = cy - ap_size / 2 + np.arange(ap_size, dtype=np.float32)
            gx = cx - ap_size / 2 + np.arange(ap_size, dtype=np.float32)
            refp = pt.bilinear(ref, gy[:, None], gx[None, :])
            pat = pt.bilinear(A, gy[None, :, None], gx[None, None, :], k=sel[:, None, None])
            ly, lx, _ = pt.phase_correlate(refp, pat, power)
            bad = np.hypot(ly, lx) > max_shift
            ly[bad], lx[bad] = 0, 0
            n_rej += int(bad.sum())
            # 5c. one resampling of the original frames: global + local shift
            sy = (dy[sel] + ly)[:, None, None]
            sx = (dx[sel] + lx)[:, None, None]
            patch = pt.bilinear(F, yi[None, :, None] + sy, xi[None, None, :] + sx, k=sel[:, None, None]).mean(0)
            num[Y0 + so:Y0 + 2 * so, X0 + so:X0 + 2 * so] += win * patch
            wsum[Y0 + so:Y0 + 2 * so, X0 + so:X0 + 2 * so] += win
            result["aps"].append((Y0 + step, X0 + step, float(np.hypot(ly, lx).mean())))
    num, wsum = num[so:so + ho, so:so + wo], wsum[so:so + ho, so:so + wo]
    # 6. blend: AP patches where covered, global stack elsewhere
    result["image"] = (num + np.maximum(1 - wsum, 0) * gl) / np.maximum(wsum, 1)
    result["coverage"] = wsum[..., 0]
    shifts = [s for *_, s in result["aps"]]
    log(f"{len(result['aps'])} APs of {ap_size} px, {nk} frames each ({keep:.0%}); "
        f"mean local shift {np.mean(shifts) if shifts else 0:.2f} px, {n_rej} outlier shifts reset")
    return result


def save_ap_map(path, ref, aps, ap_size, sc):
    """PNG of the reference with AP centres marked (like the AS!4 AP overlay)."""
    img = np.repeat(pt.at.normalize(ref)[..., None], 3, -1)
    big = np.kron(img, np.ones((int(np.ceil(sc)), int(np.ceil(sc)), 1)))[:int(ref.shape[0] * sc), :int(ref.shape[1] * sc)]
    for y, x, _ in aps:
        y, x = int(y), int(x)
        big[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = (0, 1, 0)
    pt.save(path, big)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".ser / .avi / glob (ideally PIPP-centred: prep.py)")
    ap.add_argument("-o", "--out", required=True, help=".tif (16-bit), .npy (float) or .png")
    ap.add_argument("--keep", default="25%", help="frames to stack per AP: N or N%%")
    ap.add_argument("--ap-size", type=int, default=32)
    ap.add_argument("--min-bright", type=float, default=0.15, help="AP only where ref > this x max")
    ap.add_argument("--ref-frames", type=int, help="frames averaged into the reference (default keep/2)")
    ap.add_argument("--sigma", type=float, default=1.0, help="noise-robust blur for quality (px)")
    ap.add_argument("--upscale", type=float, default=1.0, help="1.5 / 2 / 3 (needs many frames + dithering)")
    ap.add_argument("--no-local", action="store_true", help="global alignment only")
    ap.add_argument("--max-shift", type=float, help="reject local shifts larger than this (px)")
    ap.add_argument("--bayer", help="force debayer pattern")
    ap.add_argument("--limit", type=int, help="use only the first N frames")
    ap.add_argument("--normalize", action="store_true", help="scale so the 99.9th percentile = 0.95")
    ap.add_argument("--save-global", help="also save the global-alignment-only stack")
    ap.add_argument("--ap-map", help="PNG with AP positions")
    a = ap.parse_args()

    F, times = load_frames(a.input, a.bayer, a.limit)
    keep = pct(a.keep, len(F)) / len(F)
    r = stack(F, keep, a.ap_size, a.min_bright, a.ref_frames, a.sigma, a.upscale, not a.no_local, a.max_shift)
    img = r["image"][..., 0] if F.shape[-1] == 1 else r["image"]
    gimg = r["global_image"][..., 0] if F.shape[-1] == 1 else r["global_image"]
    if a.normalize:
        s = 0.95 / max(float(np.percentile(img, 99.9)), 1e-6)
        img, gimg = img * s, gimg * s
    pt.save(a.out, img)
    print("saved", a.out)
    if a.save_global:
        pt.save(a.save_global, gimg)
        print("saved", a.save_global)
    if a.ap_map:
        save_ap_map(a.ap_map, r["reference"], r["aps"], a.ap_size, a.upscale)
    with open(os.path.splitext(a.out)[0] + "_quality.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "quality", "rank", "dy", "dx", "time"])
        rank = np.empty(len(F), int)
        rank[r["order"]] = np.arange(len(F))
        for k in range(len(F)):
            wr.writerow([k, f"{r['quality'][k]:.6g}", rank[k], f"{r['shifts'][0][k]:.2f}",
                         f"{r['shifts'][1][k]:.2f}", times[k].isoformat() if times else ""])
    if times:   # stack time = mean time of the frames (what WinJUPOS needs)
        tm = times[0] + (times[-1] - times[0]) / 2
        print(f"capture mid-time {tm.isoformat()} UTC  (WinJUPOS name: {pt.winjupos_name(tm)})")


if __name__ == "__main__":
    main()
