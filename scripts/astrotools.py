"""Shared helpers for the deep-sky scripts.

The algorithms here only need numpy, so every script can run on a phone (Termux) or a server.
astropy is imported lazily for FITS I/O; tifffile / cv2 are optional for 16-bit RGB TIFF output.
"""
import os
import numpy as np

FITS_EXT = (".fit", ".fits", ".fts")


# ----------------------------------------------------------------------------- I/O
def is_fits(path):
    return path.lower().endswith(FITS_EXT)


def load_fits(path, with_header=False):
    """Return float32 image (H,W) or (H,W,3); colour FITS stored as (3,H,W) is moved to channels-last."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data, hdr = hdu.data.astype(np.float32), hdu.header.copy()
                break
        else:
            raise ValueError(f"{path}: no image data")
    if data.ndim == 3 and data.shape[0] in (3, 4):
        data = np.moveaxis(data[:3], 0, -1)
    return (data, hdr) if with_header else data


def save_fits(path, data, header=None, history=None):
    from astropy.io import fits
    d = np.asarray(data, np.float32)
    if d.ndim == 3:
        d = np.moveaxis(d, -1, 0)
    hdr = header.copy() if header is not None else fits.Header()
    for k in ("BZERO", "BSCALE", "NAXIS3"):
        hdr.remove(k, ignore_missing=True)
    if history:
        hdr.add_history(history)
    fits.writeto(path, d, hdr, overwrite=True)


def load_image(path):
    """FITS -> raw float32 values; TIFF/PNG/JPG -> float32 in [0,1]."""
    if is_fits(path):
        return load_fits(path)
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            a = tifffile.imread(path)
            return _to_unit(a)
    except ImportError:
        pass
    from PIL import Image
    return _to_unit(np.array(Image.open(path)))


def _to_unit(a):
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535
    return a.astype(np.float32)


def save_image(path, img, header=None, history=None):
    """Save by extension. Non-FITS outputs expect data in [0,1]: .tif -> 16 bit, .png/.jpg -> 8 bit."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if is_fits(path):
        return save_fits(path, img, header, history)
    img = np.clip(np.nan_to_num(img), 0, 1)
    if path.lower().endswith((".tif", ".tiff")):
        a16 = (img * 65535 + .5).astype(np.uint16)
        try:
            import tifffile
            return tifffile.imwrite(path, a16, photometric="rgb" if a16.ndim == 3 else "minisblack")
        except ImportError:
            pass
        try:
            import cv2
            return cv2.imwrite(path, a16[..., ::-1] if a16.ndim == 3 else a16)
        except ImportError:
            pass
        from PIL import Image
        if a16.ndim == 2:
            return Image.fromarray(a16).save(path)
        print(f"warning: no tifffile/cv2, {path} written as 8-bit")
    from PIL import Image
    kw = {"quality": 95} if path.lower().endswith((".jpg", ".jpeg")) else {}
    Image.fromarray((img * 255 + .5).astype(np.uint8)).save(path, **kw)


def out_name(path, suffix, ext=None, outdir=None):
    base, e = os.path.splitext(os.path.basename(path))
    return os.path.join(outdir or os.path.dirname(path) or ".", f"{base}{suffix}{ext or e}")


# ----------------------------------------------------------------------------- statistics
def mad_std(a):
    """Robust sigma (MAD * 1.4826); ignores NaN. Subsamples big arrays for speed."""
    a = np.asarray(a)
    if a.size > 4_000_000:
        a = a.ravel()[:: a.size // 2_000_000]
    med = np.nanmedian(a)
    return 1.4826 * np.nanmedian(np.abs(a - med))


def noise_sigma(img):
    """Pixel noise estimated from the finest wavelet-like scale (signal mostly cancels)."""
    d = img[1:-1, 1:-1] - 0.25 * (img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:])
    return mad_std(d) / np.sqrt(1 + 4 * 0.25 ** 2)


def sigma_clip_mean(cube, k_low=3.0, k_high=3.0, weights=None, iters=2):
    """Pixel-wise sigma-clipped (weighted) mean along axis 0 - the core of ImageIntegration."""
    cube = np.array(cube, np.float32, copy=True)
    for _ in range(iters):
        med = np.nanmedian(cube, axis=0)
        sd = 1.4826 * np.nanmedian(np.abs(cube - med), axis=0)
        sd = np.maximum(sd, 1e-6)
        bad = (cube < med - k_low * sd) | (cube > med + k_high * sd)
        cube[bad] = np.nan
    w = np.ones(len(cube), np.float32) if weights is None else np.asarray(weights, np.float32)
    w = w.reshape((-1,) + (1,) * (cube.ndim - 1))
    ww = np.where(np.isnan(cube), 0, w)
    s = ww.sum(0)
    out = np.nansum(cube * w, axis=0) / np.maximum(s, 1e-12)
    out[s == 0] = np.nan
    return out, float(np.isnan(cube).mean())


# ----------------------------------------------------------------------------- filters (pure numpy)
def _conv1d(img, k, axis, step=1):
    """Correlate along one axis with a (possibly 'holed') kernel, reflect borders."""
    r = (len(k) // 2) * step
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    n = img.shape[axis]
    out = np.zeros_like(img, dtype=np.float32)
    for i, w in enumerate(k):
        out += w * np.take(p, range(i * step, i * step + n), axis=axis)
    return out


def sep_filter(img, k, step=1):
    k = np.asarray(k, np.float32)
    if img.ndim == 3:
        return np.dstack([sep_filter(img[..., c], k, step) for c in range(img.shape[2])])
    return _conv1d(_conv1d(img.astype(np.float32), k, 0, step), k, 1, step)


def gaussian_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img, sigma):
    return img if sigma <= 0 else sep_filter(img, gaussian_kernel(sigma))


def _rank1d(img, size, axis, fn):
    r = size // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    return fn(np.lib.stride_tricks.sliding_window_view(p, size, axis=axis), axis=-1)


def max_filter(img, size):
    return _rank1d(_rank1d(img, size, 0, np.max), size, 1, np.max)


def min_filter(img, size):
    return _rank1d(_rank1d(img, size, 0, np.min), size, 1, np.min)


def upsample_grid(grid, shape):
    """Bilinear interpolation of a coarse tile grid (values at tile centres) to full resolution."""
    gh, gw = grid.shape
    h, w = shape

    def weights(n_out, n_in):
        c = (np.arange(n_out) + .5) / n_out * n_in - .5
        c = np.clip(c, 0, n_in - 1)
        i0 = np.minimum(np.floor(c).astype(int), max(n_in - 2, 0))
        f = (c - i0).astype(np.float32)
        return i0, np.minimum(i0 + 1, n_in - 1), f

    y0, y1, fy = weights(h, gh)
    x0, x1, fx = weights(w, gw)
    rows = grid[y0] * (1 - fy)[:, None] + grid[y1] * fy[:, None]
    return rows[:, x0] * (1 - fx) + rows[:, x1] * fx


def background_map(img, tile=64, k=2.5):
    """SExtractor-style background: sigma-clipped median per tile, then bilinear upsample."""
    h, w = img.shape
    gh, gw = max(1, h // tile), max(1, w // tile)
    t = img[: gh * tile, : gw * tile].reshape(gh, tile, gw, tile).swapaxes(1, 2).reshape(gh, gw, -1)
    med = np.median(t, axis=-1, keepdims=True)
    sd = 1.4826 * np.median(np.abs(t - med), axis=-1, keepdims=True)
    t = np.where(np.abs(t - med) < k * np.maximum(sd, 1e-6), t, np.nan)
    return upsample_grid(np.nanmedian(t, axis=-1), img.shape).astype(np.float32)


def to_luminance(img):
    return img if img.ndim == 2 else img[..., :3].mean(-1)


# ----------------------------------------------------------------------------- stars
def detect_stars(img, k=5.0, max_stars=500, radius=7, saturation=None):
    """Find stars and measure them with moments.

    Returns dict of arrays: x, y, flux, peak, fwhm, hfr, ecc, (sorted by flux, brightest first).
    fwhm uses 2nd moments (2.3548*sigma); hfr = half-flux radius (Gaussian: fwhm == 2*hfr).
    """
    img = to_luminance(img).astype(np.float32)
    bkg = background_map(img)
    sub = img - bkg
    sigma = mad_std(sub)
    sm = gaussian_blur(sub, 1.0)
    peaks = (sm == max_filter(sm, 5)) & (sm > k * sigma / 2.0)  # smoothing lowers noise ~2x
    h, w = img.shape
    ys, xs = np.nonzero(peaks)
    edge = (ys >= radius) & (ys < h - radius) & (xs >= radius) & (xs < w - radius)
    ys, xs = ys[edge], xs[edge]
    order = np.argsort(-sm[ys, xs])[: max_stars * 3]
    ys, xs = ys[order], xs[order]
    if saturation is None:
        saturation = 0.95 * float(img.max())

    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr = np.hypot(yy, xx)
    aper = rr <= radius
    rows = []
    for y, x in zip(ys, xs):
        st = sub[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if img[y, x] >= saturation:
            continue
        v = np.where(aper & (st > 0), st, 0)
        f = v.sum()
        if f <= 0:
            continue
        cx, cy = (v * xx).sum() / f, (v * yy).sum() / f
        dx, dy = xx - cx, yy - cy
        mxx, myy, mxy = (v * dx * dx).sum() / f, (v * dy * dy).sum() / f, (v * dx * dy).sum() / f
        tr, det = mxx + myy, mxx * myy - mxy ** 2
        disc = np.sqrt(max(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        if l2 <= 0.1:  # hot pixel / cosmic ray: too sharp
            continue
        rd = np.hypot(dx, dy).ravel()
        o = np.argsort(rd)
        cum = np.cumsum(v.ravel()[o])
        hfr = rd[o][np.searchsorted(cum, cum[-1] / 2)]
        rows.append((x + cx, y + cy, f, st.max(), 2.3548 * np.sqrt((l1 + l2) / 2), hfr,
                     np.sqrt(1 - l2 / l1)))
        if len(rows) >= max_stars:
            break
    names = ("x", "y", "flux", "peak", "fwhm", "hfr", "ecc")
    a = np.array(rows, np.float64).reshape(-1, len(names))
    return {n: a[:, i] for i, n in enumerate(names)}


def star_mask(img, k=4.0, grow=2, soften=2.0, scale=4):
    """Soft star mask (StarMask): small-scale structures above k*noise, dilated and blurred."""
    lum = to_luminance(img).astype(np.float32)
    small = lum - gaussian_blur(lum, scale)          # remove nebula / large scales
    m = (small > k * noise_sigma(lum)).astype(np.float32)
    m *= sep_filter(m, [1, 1, 1]) >= 4               # drop isolated noise pixels (<4 hits in 3x3)
    if grow:
        m = max_filter(m, 2 * grow + 1)
    return np.clip(gaussian_blur(m, soften), 0, 1)


# ----------------------------------------------------------------------------- stretch
def mtf(m, x):
    """Midtones transfer function: mtf(m, m) = 0.5, mtf(m,0)=0, mtf(m,1)=1. Solves mtf(mtf(t,b), b)=t."""
    x = np.asarray(x, np.float32)
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def normalize(img, lo=None, hi=None):
    lo = np.nanmin(img) if lo is None else lo
    hi = np.nanmax(img) if hi is None else hi
    return np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)


def stf_params(ch, shadows=-2.8, target=0.25):
    """PixInsight AutoSTF on a [0,1] channel -> (c0 black point, m midtone)."""
    med = float(np.nanmedian(ch))
    madn = mad_std(ch)
    c0 = float(np.clip(med + shadows * madn, 0, 1))
    b = (med - c0) / max(1 - c0, 1e-12)
    return c0, float(mtf(target, b))


def stf(img, shadows=-2.8, target=0.25, linked=True):
    """Non-linear 'screen transfer' stretch. Input any scale; output [0,1]."""
    x = normalize(img)
    if x.ndim == 2:
        c0, m = stf_params(x, shadows, target)
        return mtf(m, np.clip((x - c0) / (1 - c0), 0, 1))
    if linked:
        c0, m = stf_params(x.mean(-1), shadows, target)
        return mtf(m, np.clip((x - c0) / (1 - c0), 0, 1))
    return np.dstack([stf(x[..., c], shadows, target) for c in range(x.shape[2])])


def asinh_stretch(img, beta=500.0, black=None):
    """Arcsinh stretch (Lupton et al. 2004); colour-preserving when applied to luminance ratio."""
    x = normalize(img)
    if black is None:
        black = max(float(np.nanmedian(x)) - 2.8 * mad_std(x), 0)
    x = np.clip((x - black) / (1 - black), 0, 1)
    if x.ndim == 3:
        lum = np.maximum(x.mean(-1, keepdims=True), 1e-9)
        return np.clip(x * np.arcsinh(beta * lum) / np.arcsinh(beta) / lum, 0, 1)
    return np.arcsinh(beta * x) / np.arcsinh(beta)


# ----------------------------------------------------------------------------- wavelets
B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16


def starlet(img, levels):
    """Isotropic undecimated wavelet ('a trous', B3 spline) used by PixInsight MLT.
    Returns [w1..wn, residual]; sum(result) == img exactly."""
    c = img.astype(np.float32)
    layers = []
    for j in range(levels):
        nxt = sep_filter(c, B3, step=2 ** j)
        layers.append(c - nxt)
        c = nxt
    layers.append(c)
    return layers


# noise sigma of each starlet layer for unit gaussian white noise (Starck & Murtagh)
STARLET_NOISE = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005, 0.0025]
