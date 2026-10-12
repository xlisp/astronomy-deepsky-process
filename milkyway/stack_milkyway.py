#!/usr/bin/env python
"""Milky Way / nightscape stacker: align the sky on the stars, stack the ground unaligned, blend.

Equivalent tools: Sequator, Starry Landscape Stacker, DeepSkyStacker + Photoshop masks.

  python milkyway/stack_milkyway.py "shots/*.ARW" -o out/                     # auto horizon
  python milkyway/stack_milkyway.py shots/*.tif -o out/ --mask sky_mask.png   # hand-made mask
  python milkyway/stack_milkyway.py shots/*.jpg -o out/ --horizon 0.62 --model homography
  python milkyway/stack_milkyway.py shots/*.NEF -o out/ --darks darks/*.NEF --bin 2 --gradient

Pipeline
  1. load (RAW via rawpy -> linear 16 bit; TIFF/PNG/JPG as is), optional dark subtraction / 2x2 binning
  2. sky mask: --mask file | --horizon fraction | auto (star density per tile) | --no-ground
  3. detect stars in the sky part of every frame
  4. match to the reference frame: predicted transform (previous frame) -> mutual nearest neighbours
     -> RANSAC homography -> iterative refit with the final model (poly2 handles wide-angle distortion)
  5. warp sky (bilinear) into a disk cache; keep ground pixels unaligned in a second cache
  6. robust sigma-clipped mean, strip by strip (planes, satellites, hot pixels rejected)
  7. feathered blend: sky_stack * mask + ground_stack * (1 - mask)
"""
import argparse, glob, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import astrotools as at  # noqa: E402

RAW_EXT = (".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw")


# ----------------------------------------------------------------------------- loading
def load_frame(path, binning=1):
    """Return (H,W,3) float32 and whether the data is linear."""
    if path.lower().endswith(RAW_EXT):
        import rawpy
        with rawpy.imread(path) as r:
            img = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                gamma=(1, 1)).astype(np.float32) / 65535
        linear = True
    else:
        img = at.load_image(path)
        linear = at.is_fits(path)
        if linear:                       # ADU -> [0,1] so the float16 cache keeps precision
            img = img / 65535
    if img.ndim == 2:
        img = img[..., None].repeat(3, -1)
    img = img[..., :3]
    if binning > 1:
        h, w = (img.shape[0] // binning) * binning, (img.shape[1] // binning) * binning
        img = img[:h, :w].reshape(h // binning, binning, w // binning, binning, 3).mean((1, 3))
    return img.astype(np.float32), linear


def remove_hot_pixels(img, k=8.0):
    """Fixed-pattern hot pixels in the (unaligned) ground stack: brighter than all 8 neighbours by k sigma."""
    out = img.copy()
    for c in range(img.shape[2]):
        ch = img[..., c]
        h, w = ch.shape
        p = np.pad(ch, 1, mode="edge")
        nb = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        mx, s = nb[0].copy(), nb[0].copy()
        for n in nb[1:]:
            np.maximum(mx, n, out=mx)
            s += n
        hot = ch > mx + k * at.noise_sigma(ch)
        out[..., c][hot] = s[hot] / 8
    return out


# ----------------------------------------------------------------------------- sky mask
def mask_from_horizon(shape, horizon_y):
    """horizon_y: scalar fraction or per-column array of pixel rows. 1 = sky."""
    h, w = shape
    hy = np.full(w, horizon_y * h) if np.isscalar(horizon_y) else horizon_y
    return (np.arange(h)[:, None] < hy[None, :]).astype(np.float32)


def auto_horizon(lum, per_band=15):
    """Sky = where the stars are. Split the frame into vertical bands; in each band walk down the sorted
    star y positions and stop at the first gap much larger than the typical spacing (the ground has no
    stars, isolated lights below the horizon are left out by the gap test)."""
    h, w = lum.shape
    st = at.detect_stars(lum, k=5, max_stars=4000)
    n = len(st["x"])
    cols = int(np.clip(n / per_band, 4, 64))
    hy = np.full(cols, np.nan)
    for j in range(cols):
        ys = np.sort(st["y"][(st["x"] >= j * w / cols) & (st["x"] < (j + 1) * w / cols)])
        if len(ys) < 3:
            continue
        d = np.diff(np.r_[0, ys])
        g = np.median(d)
        big = np.nonzero(d > max(6 * g, 0.1 * h))[0]
        last = (big[0] - 1) if len(big) and big[0] > 0 else len(ys) - 1
        hy[j] = min(h, ys[last] + g)
    if not np.isfinite(hy).any():
        return np.zeros(w), n
    dev = np.abs(hy - np.nanmedian(hy))
    hy[dev > 3 * at.mad_std(hy[np.isfinite(hy)]) + 0.05 * h] = np.nan   # bands fooled by a random gap
    ok = np.isfinite(hy)
    hy = np.interp(np.arange(cols), np.nonzero(ok)[0], hy[ok])
    p = np.pad(hy, 1, mode="edge")
    hy = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), 0)                  # light smoothing
    return np.interp(np.arange(w), (np.arange(cols) + .5) * w / cols, hy), n


def feather(mask, sigma):
    return np.clip(at.gaussian_blur(mask, sigma), 0, 1) if sigma > 0 else mask


# ----------------------------------------------------------------------------- geometry
class Transform:
    """Maps REFERENCE pixel coords -> SOURCE frame pixel coords (what inverse warping needs).
    model: 'homography' (3x3) or 'affine' / 'poly2' / 'poly3' (polynomials in normalised coords)."""
    DEG = {"affine": 1, "poly2": 2, "poly3": 3}

    def __init__(self, model, shape, coef=None):
        self.model, (self.h, self.w) = model, shape
        if coef is None:
            if model == "homography":
                coef = np.eye(3)
            else:
                coef = np.zeros((len(self._exps()), 2))
                coef[1, 0] = coef[2, 1] = 1.0
        self.coef = coef

    def _exps(self):
        d = self.DEG[self.model]
        return [(a, n - a) for n in range(d + 1) for a in range(n, -1, -1)]

    def _norm(self, p):
        s = self.w / 2
        return (p[:, 0] - self.w / 2) / s, (p[:, 1] - self.h / 2) / s

    def _terms(self, p):
        u, v = self._norm(p)
        return np.stack([u ** a * v ** b for a, b in self._exps()], 1)

    def apply(self, p):
        p = np.asarray(p, np.float64)
        if self.model == "homography":
            q = np.c_[p, np.ones(len(p))] @ self.coef.T
            return q[:, :2] / q[:, 2:3]
        uv = self._terms(p) @ self.coef
        s = self.w / 2
        return np.c_[uv[:, 0] * s + self.w / 2, uv[:, 1] * s + self.h / 2]

    def fit(self, ref, src):
        if self.model == "homography":
            return Transform(self.model, (self.h, self.w), fit_homography(ref, src))
        u, v = self._norm(src)
        coef, *_ = np.linalg.lstsq(self._terms(ref), np.c_[u, v], rcond=None)
        return Transform(self.model, (self.h, self.w), coef)

    def shifted(self, dx, dy):
        t = Transform(self.model, (self.h, self.w), self.coef.copy())
        if self.model == "homography":
            t.coef = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ t.coef
        else:
            t.coef[0] += (dx / (self.w / 2), dy / (self.w / 2))
        return t


def fit_homography(src, dst):
    """Normalised DLT: dst ~ H @ src."""
    def cond(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-9)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    Ts, Td = cond(src), cond(dst)
    ps = np.c_[src, np.ones(len(src))] @ Ts.T
    pd = np.c_[dst, np.ones(len(dst))] @ Td.T
    A = np.zeros((2 * len(src), 9))
    x, y, u, v = ps[:, 0], ps[:, 1], pd[:, 0], pd[:, 1]
    A[0::2, 0:3] = -ps; A[0::2, 6:9] = ps * u[:, None]
    A[1::2, 3:6] = -ps; A[1::2, 6:9] = ps * v[:, None]
    H = np.linalg.svd(A)[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def mutual_nn(pred, xy, radius):
    d2 = ((pred[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    i = np.arange(len(pred))
    ok = (d2[i, j] < radius ** 2) & (d2.argmin(0)[j] == i)
    return i[ok], j[ok]


def ransac_homography(ref, src, thr=3.0, iters=400, rng=np.random.default_rng(0)):
    best = np.zeros(len(ref), bool)
    t0 = Transform("homography", (1, 1))
    for _ in range(iters):
        s = rng.choice(len(ref), 4, replace=False)
        try:
            t = t0.fit(ref[s], src[s])
        except np.linalg.LinAlgError:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            inl = np.linalg.norm(t.apply(ref) - src, axis=1) < thr
        if inl.sum() > best.sum():
            best = inl
    return best


def phase_shift(a, b):
    """Translation (dx, dy) such that b(x) ~ a(x - d), via phase correlation."""
    F = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    r = np.fft.irfft2(F / np.maximum(np.abs(F), 1e-12), s=a.shape)
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    h, w = a.shape
    return (dx - w if dx > w // 2 else dx), (dy - h if dy > h // 2 else dy)


def highpass_bin(lum, mask, b=4):
    x = np.where(mask > 0.5, lum, np.median(lum[mask > 0.5]))
    x = x - at.gaussian_blur(x, 3)
    h, w = (x.shape[0] // b) * b, (x.shape[1] // b) * b
    return x[:h, :w].reshape(h // b, b, w // b, b).max((1, 3))


def register(ref_xy, xy, guess, model, min_inliers=12):
    """Return (Transform, n_inliers, rms) or None."""
    t = guess
    for r in (40, 12, 5):
        i, j = mutual_nn(t.apply(ref_xy), xy, r)
        if len(i) < min_inliers:
            return None
        inl = ransac_homography(ref_xy[i], xy[j], thr=max(3.0, r / 4))
        if inl.sum() < min_inliers:
            return None
        t = Transform("homography", (guess.h, guess.w)).fit(ref_xy[i][inl], xy[j][inl])
    final = Transform(model, (guess.h, guess.w))
    i, j = mutual_nn(t.apply(ref_xy), xy, 4)
    keep = np.ones(len(i), bool)
    for _ in range(5):                                   # grow the inlier set with the final model
        t = final.fit(ref_xy[i][keep], xy[j][keep])
        res = np.linalg.norm(t.apply(ref_xy[i]) - xy[j], axis=1)
        new = res < max(1.5, 3 * at.mad_std(res[keep]))
        if new.sum() < min_inliers or (new == keep).all():
            break
        keep = new
    return t, int(keep.sum()), float(np.sqrt(np.mean(res[keep] ** 2)))


def warp(img, t, out, rows=256):
    """Inverse bilinear warp of img (H,W,C) into out (H,W,C) through t (ref -> src); outside = NaN."""
    h, w = out.shape[:2]
    xs = np.arange(w, dtype=np.float64)
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        gx, gy = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float64))
        p = t.apply(np.c_[gx.ravel(), gy.ravel()])
        x, y = p[:, 0], p[:, 1]
        x0, y0f = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        ok = (x0 >= 0) & (y0f >= 0) & (x0 < img.shape[1] - 1) & (y0f < img.shape[0] - 1)
        x0c, y0c = np.clip(x0, 0, img.shape[1] - 2), np.clip(y0f, 0, img.shape[0] - 2)
        fx, fy = (x - x0c)[:, None], (y - y0c)[:, None]
        v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
             + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
        v[~ok] = np.nan
        out[y0:y1] = v.reshape(y1 - y0, w, -1)


# ----------------------------------------------------------------------------- stacking
def combine(cache, k, rows=64):
    n, h, w, c = cache.shape
    out = np.empty((h, w, c), np.float32)
    rej = 0.0
    for y0 in range(0, h, rows):
        cube = np.asarray(cache[:, y0:y0 + rows], np.float32)
        if n >= 3 and k > 0:
            m, r = at.sigma_clip_mean(cube, k, k)
            rej += r * cube.shape[1] / h
        else:
            m = np.nanmean(cube, 0)
        out[y0:y0 + rows] = m
    return out, rej


def stretch_preview(img, sky):
    """Neutralise the sky background per channel, then a linked AutoSTF with a darker target."""
    v = img.copy()
    ref = v[sky > 0.5] if (sky > 0.5).any() else v.reshape(-1, 3)
    bg = np.median(ref[::17], 0)
    v -= bg - bg.min()
    return at.stf(v, shadows=-2.8, target=0.18, linked=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="milkyway_out")
    ap.add_argument("--ref", type=int, help="reference frame index (default: middle)")
    ap.add_argument("--model", choices=["poly2", "poly3", "homography", "affine"], default="poly2")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mask", help="sky mask image, white = sky (same size as frames)")
    g.add_argument("--horizon", type=float, help="straight horizon at this fraction of height")
    g.add_argument("--no-ground", action="store_true", help="whole frame is sky")
    ap.add_argument("--feather", type=float, default=0.006, help="mask blur, fraction of width")
    ap.add_argument("--darks", nargs="+", help="dark frames (same ISO/exposure/temperature)")
    ap.add_argument("--bin", type=int, default=1, help="2 = half resolution, 4x faster, less noise")
    ap.add_argument("--sigma", type=float, default=3.0, help="rejection kappa (0 = plain mean)")
    ap.add_argument("--ground", choices=["mean", "median"], default="mean",
                    help="ground stack: sigma-clipped mean, or median (more robust to car lights)")
    ap.add_argument("--gradient", action="store_true", help="remove light-pollution gradient from sky")
    ap.add_argument("--keep-cache", action="store_true")
    a = ap.parse_args()

    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    if len(files) < 2:
        raise SystemExit("need at least 2 frames")
    os.makedirs(a.out, exist_ok=True)
    ref_i = len(files) // 2 if a.ref is None else a.ref
    dark = None
    if a.darks:
        dfiles = sorted({f for p in a.darks for f in (glob.glob(p) or [p])})
        dark = np.median(np.stack([load_frame(f, a.bin)[0] for f in dfiles]), 0)
        print(f"master dark from {len(dfiles)} frames")

    def load(i):
        img, lin = load_frame(files[i], a.bin)
        return (img - dark if dark is not None else img), lin

    ref, linear = load(ref_i)
    H, W = ref.shape[:2]
    lum = ref.mean(-1)
    if a.no_ground:
        mask = np.ones((H, W), np.float32)
    elif a.mask:
        m = at.load_image(a.mask)
        mask = (at.to_luminance(m) > 0.5).astype(np.float32)
        if mask.shape != (H, W):
            raise SystemExit(f"mask {mask.shape} != frame {(H, W)} (did you use --bin?)")
    elif a.horizon is not None:
        mask = mask_from_horizon((H, W), a.horizon)
    else:
        hy, n = auto_horizon(lum)
        mask = mask_from_horizon((H, W), hy)
        print(f"auto horizon from {n} stars: rows {hy.min():.0f}..{hy.max():.0f} of {H} "
              f"(check sky_mask.png; override with --mask / --horizon)")
    at.save_image(os.path.join(a.out, "sky_mask.png"), mask)
    if mask.max() < 0.5:
        raise SystemExit("sky mask is empty - pass --horizon 0.6 or --mask sky.png or --no-ground")
    has_ground = mask.min() < 0.5
    sky_rows = np.nonzero(mask.max(1) > 0.5)[0]
    gnd_rows = np.nonzero(mask.min(1) < 0.5)[0]

    def sky_stars(img):
        l = img.mean(-1)
        sky = l[mask > 0.5]
        fill = np.median(sky) + at.mad_std(sky - np.median(sky)) * rng.standard_normal(l.shape)
        l = np.where(mask > 0.5, l, fill.astype(np.float32))   # same noise level -> right threshold
        st = at.detect_stars(l, k=5, max_stars=800)
        return np.c_[st["x"], st["y"]]

    rng = np.random.default_rng(0)
    ref_xy = sky_stars(ref)
    ref_hp = highpass_bin(lum, mask)
    print(f"reference {os.path.basename(files[ref_i])}: {W}x{H}, {len(ref_xy)} stars, "
          f"{'linear' if linear else 'non-linear'} data, model={a.model}")

    n = len(files)
    sky_cache = np.lib.format.open_memmap(os.path.join(a.out, "_sky_cache.npy"), "w+", np.float16,
                                          (n, H, W, 3))
    g0, g1 = (gnd_rows.min(), gnd_rows.max() + 1) if has_ground else (0, 0)
    gnd_cache = np.lib.format.open_memmap(os.path.join(a.out, "_ground_cache.npy"), "w+", np.float16,
                                          (n, g1 - g0, W, 3)) if has_ground else None
    order = [ref_i] + list(range(ref_i + 1, n)) + list(range(ref_i - 1, -1, -1))
    tmp = np.empty((H, W, 3), np.float32)
    used = np.zeros(n, bool)
    prev = Transform(a.model, (H, W))
    t_start = time.time()
    for i in order:
        if i == ref_i - 1:
            prev = Transform(a.model, (H, W))          # restart the prediction chain from the reference
        img = ref if i == ref_i else load(i)[0]
        if img.shape[:2] != (H, W):
            print(f"  skip {files[i]}: size differs"); continue
        if gnd_cache is not None:
            gnd_cache[i] = img[g0:g1]
        if i == ref_i:
            t, ninl, rms = Transform(a.model, (H, W)), len(ref_xy), 0.0
        else:
            xy = sky_stars(img)
            res = register(ref_xy, xy, prev, a.model) if len(xy) >= 12 else None
            if res is None:                              # lost track: re-acquire by phase correlation
                dx, dy = phase_shift(ref_hp, highpass_bin(img.mean(-1), mask))
                res = register(ref_xy, xy, Transform(a.model, (H, W)).shifted(4 * dx, 4 * dy), a.model) \
                    if len(xy) >= 12 else None
            if res is None:
                print(f"  {os.path.basename(files[i])}: registration FAILED ({len(xy)} stars), skipped")
                continue
            t, ninl, rms = res
        prev = t
        warp(img, t, tmp)
        sky_cache[i] = tmp
        used[i] = True
        c = t.apply(np.array([[W / 2, H / 2]]))[0]
        print(f"  {os.path.basename(files[i])}: {ninl} stars, rms {rms:.2f}px, "
              f"centre shift ({c[0] - W / 2:+.1f},{c[1] - H / 2:+.1f})")
    print(f"registered {used.sum()}/{n} frames in {time.time() - t_start:.0f}s")
    idx = np.nonzero(used)[0]

    sky, rej = combine(sky_cache[idx], a.sigma)
    print(f"sky stacked, rejected {rej * 100:.2f}% pixels")
    sky = np.where(np.isfinite(sky), sky, ref)         # warped borders: fall back to the reference
    if a.gradient:
        from gradient_removal import fit_background
        for c in range(3):
            ch = np.where(mask > 0.5, sky[..., c], np.nan)
            model, keep = fit_background(ch, 24, 2)
            sky[..., c] = sky[..., c] - model + np.nanmedian(model[mask > 0.5])
        print("sky gradient removed")
    out = sky
    if has_ground:
        gidx = idx if len(idx) else np.arange(n)
        if a.ground == "median":
            gnd = np.median(np.asarray(gnd_cache[gidx], np.float32), 0)
        else:
            gnd, _ = combine(gnd_cache[gidx], a.sigma)
        if dark is None:
            gnd = remove_hot_pixels(gnd)
        full_g = ref.copy()
        full_g[g0:g1] = gnd
        m = feather(mask, a.feather * W)[..., None]
        out = sky * m + full_g * (1 - m)
        print("ground stacked unaligned and blended")

    noise_in = at.noise_sigma(ref.mean(-1)[sky_rows.min():sky_rows.max() + 1])
    noise_out = at.noise_sigma(out.mean(-1)[sky_rows.min():sky_rows.max() + 1])
    print(f"sky noise {noise_in:.4g} -> {noise_out:.4g}  (x{noise_in / max(noise_out, 1e-12):.2f}, "
          f"ideal sqrt(N) = {np.sqrt(len(idx)):.2f})")

    at.save_image(os.path.join(a.out, "milkyway_stack.tif"), np.clip(out / max(out.max(), 1e-9), 0, 1)
                  if linear else out)
    preview = stretch_preview(out, mask) if linear else np.clip(out, 0, 1)
    at.save_image(os.path.join(a.out, "milkyway_preview.jpg"), preview)
    print(f"saved {a.out}/milkyway_stack.tif ({'linear' if linear else 'as input'}), "
          f"milkyway_preview.jpg, sky_mask.png")
    del sky_cache, gnd_cache
    if not a.keep_cache:
        for f in ("_sky_cache.npy", "_ground_cache.npy"):
            p = os.path.join(a.out, f)
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
