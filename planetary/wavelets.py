#!/usr/bin/env python
"""RegiStax 6 style wavelet sharpening of a stacked planetary image (+ RGB align, balance, dering).

  python planetary/wavelets.py stack.tif -o sharp.tif --sharpen 0.5,0.5,0.3 --denoise 0.1,0.05
  python planetary/wavelets.py stack.tif -o sharp.png --scheme linear --initial 1 --step 1 --sharpen 0.6,0.5,0.3,0.1
  python planetary/wavelets.py stack.tif -o sharp.tif --rgb-align --balance auto --dering 0.8 --gamma 0.9
  python planetary/wavelets.py stack.tif --dump layers/                    # inspect the six layers

What RegiStax 6 does in its Wavelet tab, and what the options here map to:
  wavelet filter  "Gaussian" -> --filter gaussian : layer_j = G(s_(j-1)) - G(s_j)   (difference of Gaussians)
                  "Default"  -> --filter atrous   : B3-spline a-trous (starlet) layers, scales 1,2,4,8...
  wavelet scheme  "Dyadic"   -> --scheme dyadic   : s_j = initial * 2^(j-1)
                  "Linear"   -> --scheme linear   : s_j = initial + (j-1) * step
  per-layer slider "Sharpen"  -> --sharpen g1,g2.. : layer_j *= 1 + g_j       (0 = unchanged)
  per-layer       "Denoise"   -> --denoise d1,d2.. : soft threshold at d_j*10 x layer noise sigma
  "Deringing"                 -> --dering A        : clamp overshoot near the limb to the original's range
  "RGB align"                 -> --rgb-align       : phase-correlate R and B to G, sub-pixel shift
  "RGB balance"               -> --balance auto    : equal channel medians on the disc (or r,g,b gains)
  "Gamma"                     -> --gamma g
Because sum(layers) + residual == image, gains of 0 return the input exactly; the sharpening is just
"add back more of the fine scales", i.e. an unsharp mask applied separately at several radii.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planetools as pt
at = pt.at


def floats(s):
    return [float(v) for v in s.split(",")] if s else []


def layer_scales(n, scheme="dyadic", initial=1.0, step=1.0):
    if scheme == "dyadic":
        return [initial * 2 ** j for j in range(n)]
    return [initial + j * step for j in range(n)]


def decompose(ch, n=6, filt="gaussian", scheme="dyadic", initial=1.0, step=1.0):
    """Return ([layer_1..layer_n], residual, scales)."""
    if filt == "atrous":
        layers = at.starlet(ch, n)
        return layers[:-1], layers[-1], [2 ** j for j in range(n)]
    scales = layer_scales(n, scheme, initial, step)
    prev, layers = ch.astype(np.float32), []
    for s in scales:
        nxt = at.gaussian_blur(ch, s)
        layers.append(prev - nxt)
        prev = nxt
    return layers, prev, scales


def sharpen_channel(ch, gains, denoise, **kw):
    layers, resid, scales = decompose(ch, max(len(gains), len(denoise), 1), **kw)
    out = resid.copy()
    for j, w in enumerate(layers):
        d = denoise[j] if j < len(denoise) else 0
        if d > 0:
            t = d * 10 * at.mad_std(w)
            w = np.sign(w) * np.maximum(np.abs(w) - t, 0)
        g = gains[j] if j < len(gains) else 0
        out += w * (1 + g)
    return out, scales


def rgb_align(img, ref=1):
    """Shift R and B onto G (atmospheric dispersion / ADC residual). Returns image, shifts."""
    shifts = []
    out = img.copy()
    for c in range(3):
        if c == ref:
            shifts.append((0.0, 0.0))
            continue
        dy, dx, _ = pt.phase_correlate(img[..., ref], img[..., c])
        out[..., c] = pt.fourier_shift(img[..., c], -float(dy), -float(dx))
        shifts.append((float(dy), float(dx)))
    return out, shifts


def disc_mask(img, frac=0.25):
    lum = at.gaussian_blur(pt.luminance(img), 2)
    return lum > frac * np.percentile(lum, 99.5)


def balance(img, spec):
    if spec == "none":
        return img
    if spec == "auto":
        m = disc_mask(img)
        med = np.array([np.median(img[..., c][m]) for c in range(3)])
        g = med.mean() / np.maximum(med, 1e-9)
    else:
        g = np.array(floats(spec))
    return img * g


def dering(orig, out, amount, scale):
    """Limb deringing: inside a band around the limb, clamp to the original's local min/max."""
    r = max(1, int(round(2 * scale)))
    lo = at.min_filter(orig, 2 * r + 1)
    hi = at.max_filter(orig, 2 * r + 1)
    m = disc_mask(orig if orig.ndim == 3 else orig[..., None]).astype(np.float32)
    band = np.clip(1 - np.abs(at.gaussian_blur(m, scale) - 0.5) * 2, 0, 1) ** 0.5
    band = band[..., None] if out.ndim == 3 else band
    clamped = np.clip(out, lo, hi)
    return out + amount * band * (clamped - out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--out")
    ap.add_argument("--filter", choices=("gaussian", "atrous"), default="gaussian")
    ap.add_argument("--scheme", choices=("dyadic", "linear"), default="dyadic")
    ap.add_argument("--initial", type=float, default=1.0, help="scale of layer 1 (px)")
    ap.add_argument("--step", type=float, default=1.0, help="linear scheme increment")
    ap.add_argument("--sharpen", type=floats, default=[0.5, 0.5, 0.3, 0, 0, 0], help="per-layer gains")
    ap.add_argument("--denoise", type=floats, default=[0.1, 0.05], help="per-layer denoise (0..1)")
    ap.add_argument("--dering", type=float, default=0.0, help="0..1")
    ap.add_argument("--rgb-align", action="store_true")
    ap.add_argument("--balance", default="none", help="none | auto | r,g,b")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--stretch", action="store_true", help="map 0.02%%..99.98%% to 0..1 at the end")
    ap.add_argument("--dump", help="write the layers as PNG into this directory")
    a = ap.parse_args()

    img = pt.load(a.input).astype(np.float32)
    kw = dict(filt=a.filter, scheme=a.scheme, initial=a.initial, step=a.step)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        layers, resid, scales = decompose(pt.luminance(img), max(len(a.sharpen), 6), **kw)
        for j, (w, s) in enumerate(zip(layers, scales)):
            pt.save(os.path.join(a.dump, f"layer{j + 1}_{s:g}px.png"), at.normalize(w, *np.percentile(w, [0.5, 99.5])))
        pt.save(os.path.join(a.dump, "residual.png"), at.normalize(resid))
        print("layers ->", a.dump, "scales", scales)
        if not a.out:
            return
    if img.ndim == 3 and a.rgb_align:
        img, sh = rgb_align(img)
        print("RGB align (dy, dx) R=%+.2f,%+.2f  B=%+.2f,%+.2f px" % (*sh[0], *sh[2]))
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    res = [sharpen_channel(c, a.sharpen, a.denoise, **kw) for c in chans]
    out = res[0][0] if img.ndim == 2 else np.dstack([r[0] for r in res])
    print("layer scales (px):", ", ".join(f"{s:g}" for s in res[0][1]), "| gains:", a.sharpen, "| denoise:", a.denoise)
    if a.dering > 0:
        out = dering(img, out, a.dering, res[0][1][min(2, len(res[0][1]) - 1)])
    if img.ndim == 3:
        out = balance(out, a.balance)
    if a.stretch:
        lo, hi = np.percentile(out, [0.02, 99.98])
        out = (out - lo) / max(hi - lo, 1e-9)
    if a.gamma != 1:
        out = np.clip(out, 0, None) ** a.gamma
    dst = a.out or os.path.splitext(a.input)[0] + "_wavelet.tif"
    pt.save(dst, np.clip(out, 0, 1))
    print("saved", dst)


if __name__ == "__main__":
    main()
