"""Stage 2: gradient removal, stretch and colour composition (LRGB / SHO / HOO)."""
import os
import glob
import numpy as np
import cv2
from astropy.io import fits

D = os.path.dirname(os.path.abspath(__file__))
M = {k: fits.getdata(os.path.join(D, f"master_{k}.fits")).astype(np.float32) for k in "LRGBSHO"}

# common valid area (drift left NaN/zero borders)
valid = np.ones_like(M["L"], bool)
for v in M.values():
    valid &= v != 0
ys, xs = np.where(valid)
pad = 20
y0, y1, x0, x1 = ys.min() + pad, ys.max() - pad, xs.min() + pad, xs.max() - pad
M = {k: v[y0:y1, x0:x1] for k, v in M.items()}


def remove_gradient(img, grid=24, deg=2):
    """DBE-like: sample sigma-clipped medians on a grid (skipping nebula/stars), fit 2D polynomial."""
    h, w = img.shape
    pts, vals = [], []
    gy, gx = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            tile = img[i * gy:(i + 1) * gy, j * gx:(j + 1) * gx]
            pts.append(((j + .5) * gx / w, (i + .5) * gy / h))
            vals.append(np.median(tile))
    pts, vals = np.array(pts), np.array(vals)
    keep = np.ones(len(vals), bool)
    for _ in range(5):
        A = np.stack([pts[:, 0] ** a * pts[:, 1] ** b for a in range(deg + 1) for b in range(deg + 1 - a)], 1)
        c, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        r = vals - A @ c
        s = 1.4826 * np.median(np.abs(r[keep]))
        keep = r < 2.5 * s  # reject tiles brighter than model (nebula)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w; yy /= h
    model = sum(c[n] * xx ** a * yy ** b
                for n, (a, b) in enumerate((a, b) for a in range(deg + 1) for b in range(deg + 1 - a)))
    out = img - model
    return out - np.median(out)


def noise(img):
    return 1.4826 * np.median(np.abs(img[::4, ::4] - np.median(img[::4, ::4])))


def deband(img):
    """Remove row/column pattern noise (no bias/dark frames were taken)."""
    for axis in (1, 0):
        med = np.median(img); s = noise(img)
        clipped = np.where(np.abs(img - med) < 3 * s, img, np.nan)
        prof = np.nanmedian(clipped, axis=axis)
        prof -= \
            np.convolve(np.pad(prof, 50, mode="edge"), np.ones(101) / 101, "valid")
        img = img - (prof[:, None] if axis == 1 else prof[None, :])
    return img


for k in M:
    M[k] = remove_gradient(deband(M[k]))
    print(k, "noise", noise(M[k]))

# locate M27 via the plate solution of the reference frame (frame 0006 of filter 1)
from astropy.wcs import WCS
ref = glob.glob(os.path.join(os.path.dirname(D), "*Bin1_1_*_0006.fit"))[0]
px, py = WCS(fits.getheader(ref)).all_world2pix(299.9016, 22.7212, 0)
cx, cy = int(px) - x0, int(py) - y0
h_, w_ = M["H"].shape
half = 1000
cy = int(np.clip(cy, half, h_ - half)); cx = int(np.clip(cx, half, w_ - half))
print("M27 centre", cx, cy)


def crop(a):
    return a[cy - half:cy + half, cx - half:cx + half]


def mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def stretch(img, target_bg=0.12, black_sigma=-1.5, white=None):
    """PixInsight STF-like autostretch: black point at bg+k*sigma, midtone to put bg at target."""
    med = np.median(img); s = noise(img)
    lo = med + black_sigma * s
    hi = white if white is not None else np.percentile(img, 99.995)
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    b = (med - lo) / (hi - lo)
    m = mtf(target_bg, b)  # midtone balance that maps b -> target_bg
    return mtf(m, x)


def denoise(img, strength):
    """Mild luminance-masked smoothing: smooth faint areas more than bright ones."""
    blur = cv2.bilateralFilter(img.astype(np.float32), 0, strength, 3)
    w = np.clip(img / 0.5, 0, 1)
    return w * img + (1 - w) * blur


def scnr(rgb, ch=1, amount=1.0):
    o = [c for c in range(3) if c != ch]
    lim = (rgb[..., o[0]] + rgb[..., o[1]]) / 2
    rgb[..., ch] = (1 - amount) * rgb[..., ch] + amount * np.minimum(rgb[..., ch], lim)
    return rgb


def save(name, rgb):
    rgb = np.clip(rgb, 0, 1)
    out8 = (rgb[..., ::-1] * 255 + .5).astype(np.uint8) if rgb.ndim == 3 else (rgb * 255 + .5).astype(np.uint8)
    cv2.imwrite(os.path.join(D, name + ".jpg"), out8, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out16 = (rgb[..., ::-1] * 65535 + .5).astype(np.uint16) if rgb.ndim == 3 else (rgb * 65535 + .5).astype(np.uint16)
    cv2.imwrite(os.path.join(D, name + ".tif"), out16)
    print("saved", name)


def lrgb_combine(rgb, L, sat=1.3):
    """Replace luminance in Lab, boost chroma."""
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    lab[..., 0] = L * 100
    lab[..., 1:] *= sat
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def color_balance(chs):
    """Scale linear channels so background noise levels match (simple background neutralisation +
    white balance on average star colour via median of bright pixels)."""
    ref = chs[0]
    out = []
    for c in chs:
        out.append(c - np.median(c))
    # white balance: match fluxes of bright stars (top 0.1% excluding saturated)
    def starflux(c):
        t = np.percentile(c, 99.9)
        return np.median(c[(c > t) & (c < c.max() * .8)])
    f = [starflux(c) for c in out]
    return [c * f[0] / fi for c, fi in zip(out, f)]


C = {k: crop(v) for k, v in M.items()}

# ---------- LRGB ----------
R, G, B = color_balance([C["R"], C["G"], C["B"]])
white = np.percentile(np.stack([R, G, B]), 99.99)
rgb_s = np.dstack([stretch(c, 0.10, white=white) for c in (R, G, B)])
# synthetic luminance: real L plus RGB sum (more SNR)
Lsyn = C["L"] / noise(C["L"]) ** 2 + (R + G + B) / noise(R + G + B) ** 2
Ls = denoise(stretch(Lsyn, 0.10), 25)
rgb_s = cv2.GaussianBlur(rgb_s, (0, 0), 1.5)  # chroma noise reduction
lrgb = lrgb_combine(rgb_s, Ls, 1.6)
lrgb = scnr(lrgb, 1, 0.8)
save("M27_LRGB", lrgb)

# ---------- narrowband ----------
S, Hh, O = C["S"], C["H"], C["O"]
nb_white = lambda c: np.percentile(c, 99.98)
Ss, Hs, Os = (stretch(c, 0.08, white=nb_white(c)) for c in (S, Hh, O))
Ss, Hs, Os = (denoise(c, 30) for c in (Ss, Hs, Os))

# SHO (Hubble palette): R=SII, G=Ha, B=OIII, then remove green cast
sho = np.dstack([0.55 * Ss + 0.45 * Hs, 0.6 * Hs + 0.4 * Os, Os])
sho = scnr(sho, 1, 0.3)
Lnb = denoise(stretch(Hh / noise(Hh) ** 2 + O / noise(O) ** 2 + S / noise(S) ** 2, 0.10), 25)
sho = lrgb_combine(cv2.GaussianBlur(sho, (0, 0), 1.2), Lnb, 1.5)
save("M27_SHO", sho)

# HOO (natural-ish bicolour): R=Ha, G=0.5Ha+0.5OIII, B=OIII
hoo = np.dstack([Hs, 0.35 * Hs + 0.65 * Os, Os])
hoo = lrgb_combine(cv2.GaussianBlur(hoo, (0, 0), 1.2), Lnb, 1.4)
save("M27_HOO", hoo)

# LRGB + Ha/OIII enhanced (Ha into red, OIII into G/B)
Hn = np.clip(stretch(Hh, 0.06, white=nb_white(Hh)), 0, 1)
On = np.clip(stretch(O, 0.06, white=nb_white(O)), 0, 1)
enh = rgb_s.copy()
enh[..., 0] = np.maximum(enh[..., 0], Hn)
enh[..., 1] = np.maximum(enh[..., 1], 0.5 * On)
enh[..., 2] = np.maximum(enh[..., 2], On)
Lenh = np.maximum(Ls, 0.6 * Lnb)
enh = scnr(lrgb_combine(cv2.GaussianBlur(enh, (0, 0), 1.5), Lenh, 1.5), 1, 0.7)
save("M27_HaLRGB", enh)

# mono masters preview
save("M27_L_mono", Ls)
for k, v in zip("SHO", (Ss, Hs, Os)):
    save(f"M27_{k}_mono", v)
