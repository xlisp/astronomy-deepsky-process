#!/usr/bin/env python
"""Remove light-pollution gradients / vignetting with a robust 2D polynomial background model.

PixInsight equivalent: AutomaticBackgroundExtractor (ABE); DBE with automatic sample placement.

  python scripts/gradient_removal.py master_L.fits                    # -> master_L_bg.fits
  python scripts/gradient_removal.py rgb.fits --degree 3 --grid 32 --model bgmodel.fits
  python scripts/gradient_removal.py master_H.fits --divide           # multiplicative (vignetting)
  python scripts/gradient_removal.py master_L.fits --deband           # also kill row/column banding

How it works:
  1. split the image into grid x grid tiles and take each tile's median -> sample (x, y, value)
  2. least-squares fit  z = sum c_ab x^a y^b  (a+b <= degree)
  3. iteratively drop samples brighter than model + 2.5 sigma (stars, nebula) and refit
  4. subtract (additive gradient) or divide (vignetting) by the model, keep the original pedestal
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def poly_terms(x, y, deg):
    return np.stack([x ** a * y ** b for a in range(deg + 1) for b in range(deg + 1 - a)], -1)


def fit_background(img, grid=24, deg=2, k=2.5, iters=6):
    h, w = img.shape
    gy, gx = h // grid, w // grid
    t = img[: gy * grid, : gx * grid].reshape(grid, gy, grid, gx).swapaxes(1, 2).reshape(grid, grid, -1)
    vals = np.nanmedian(t, axis=-1).ravel()
    jj, ii = np.meshgrid(np.arange(grid), np.arange(grid))
    xs, ys = ((jj.ravel() + .5) * gx / w), ((ii.ravel() + .5) * gy / h)
    A = poly_terms(xs, ys, deg)
    keep = np.isfinite(vals)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        r = vals - A @ c
        s = at.mad_std(r[keep])
        new = np.isfinite(vals) & (r < k * s) & (r > -4 * k * s)
        if (new == keep).all():
            break
        keep = new
    # evaluate on a coarse grid then upsample: same result for low order, much less memory
    gh, gw = -(-h // 8), -(-w // 8)
    cy, cx = np.meshgrid((np.arange(gh) + .5) / gh, (np.arange(gw) + .5) / gw, indexing="ij")
    coarse = poly_terms(cx, cy, deg) @ c
    model = at.upsample_grid(coarse.astype(np.float32), (h, w))
    return model, keep.reshape(grid, grid)


def deband(img, k=3.0, smooth=101):
    """Remove row then column pattern noise: clipped median profile minus its smooth trend."""
    for axis in (1, 0):
        med, s = np.median(img), at.noise_sigma(img)
        prof = np.nanmedian(np.where(np.abs(img - med) < k * s, img, np.nan), axis=axis)
        trend = np.convolve(np.pad(prof, smooth // 2, mode="edge"), np.ones(smooth) / smooth, "valid")
        prof = prof - trend
        img = img - (prof[:, None] if axis == 1 else prof[None, :])
    return img


def process(ch, a):
    if a.deband:
        ch = deband(ch)
    model, keep = fit_background(ch, a.grid, a.degree, a.kappa)
    ped = float(np.median(model))
    out = ch / model * ped if a.divide else ch - model + ped
    return out, model, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--out")
    ap.add_argument("--degree", type=int, default=2, help="polynomial degree (1-4)")
    ap.add_argument("--grid", type=int, default=24, help="tiles per side")
    ap.add_argument("--kappa", type=float, default=2.5, help="sample rejection threshold")
    ap.add_argument("--divide", action="store_true", help="divide instead of subtract")
    ap.add_argument("--deband", action="store_true", help="remove row/column banding first")
    ap.add_argument("--model", help="save background model")
    a = ap.parse_args()

    img, hdr = (at.load_fits(a.input, True) if at.is_fits(a.input) else (at.load_image(a.input), None))
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    res = [process(c, a) for c in chans]
    out = res[0][0] if img.ndim == 2 else np.dstack([r[0] for r in res])
    for i, (_, m, keep) in enumerate(res):
        print(f"channel {i}: model range {m.min():.4g}..{m.max():.4g}, "
              f"samples used {keep.sum()}/{keep.size}")
    dst = a.out or at.out_name(a.input, "_bg")
    at.save_image(dst, out, hdr, f"gradient removal deg={a.degree} grid={a.grid}")
    if a.model:
        mdl = res[0][1] if img.ndim == 2 else np.dstack([r[1] for r in res])
        at.save_image(a.model, mdl if at.is_fits(a.model) else at.normalize(mdl))
    print("saved", dst)


if __name__ == "__main__":
    main()
