#!/usr/bin/env python
"""Multiscale (starlet / a-trous wavelet) noise reduction and sharpening.

PixInsight equivalent: MultiscaleLinearTransform (MLT) / MultiscaleMedianTransform.

  # noise reduction on LINEAR data, 4 layers, thresholds in noise sigmas, protect bright areas
  python scripts/wavelet.py master_L.fits -o L_nr.fits --denoise 3,2,1,0.5 --amount 0.8 --mask
  # sharpening on STRETCHED data: boost layers 2 and 3 (scales ~2-8 px)
  python scripts/wavelet.py L.tif -o L_sharp.tif --bias 0,0.3,0.2 --mask
  # just look at the layers
  python scripts/wavelet.py master_L.fits --layers 6 --dump layers/

The starlet transform splits the image into detail layers w1..wn (scale 2^(j-1) px) plus a residual,
with img == w1+...+wn+residual exactly. White noise lives mostly in w1/w2, its sigma per layer is known
(0.889, 0.200, 0.086 ... x pixel sigma), so each layer is soft-thresholded at k_j * sigma_j.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def floats(s):
    return [float(v) for v in s.split(",")] if s else []


def process(ch, levels, k, amount, bias):
    sigma = at.noise_sigma(ch)
    layers = at.starlet(ch, levels)
    for j in range(levels):
        w = layers[j]
        if j < len(k) and k[j] > 0:
            t = k[j] * sigma * at.STARLET_NOISE[j]
            soft = np.sign(w) * np.maximum(np.abs(w) - t, 0)
            w = (1 - amount) * w + amount * soft
        if j < len(bias):
            w = w * (1 + bias[j])
        layers[j] = w
    return sum(layers), sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--out")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--denoise", type=floats, default=[], help="per-layer thresholds (sigma units)")
    ap.add_argument("--amount", type=float, default=1.0, help="0..1 noise reduction strength")
    ap.add_argument("--bias", type=floats, default=[], help="per-layer gain, e.g. 0,0.3,0.2")
    ap.add_argument("--mask", action="store_true",
                    help="luminance mask: denoise faint areas / sharpen bright areas more")
    ap.add_argument("--dump", help="directory to write each layer as stretched PNG")
    a = ap.parse_args()

    img, hdr = (at.load_fits(a.input, True) if at.is_fits(a.input) else (at.load_image(a.input), None))
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        for j, w in enumerate(at.starlet(at.to_luminance(img), a.layers)):
            name = f"residual" if j == a.layers else f"w{j + 1}_scale{2 ** j}px"
            at.save_image(os.path.join(a.dump, name + ".png"), at.normalize(w, *np.percentile(w, [0.5, 99.5])))
        print("layers written to", a.dump)
        if not (a.denoise or a.bias):
            return
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    res = [process(c, a.layers, a.denoise, a.amount, a.bias) for c in chans]
    out = res[0][0] if img.ndim == 2 else np.dstack([r[0] for r in res])
    print("noise sigma per channel:", [f"{r[1]:.4g}" for r in res],
          "->", [f"{at.noise_sigma(c):.4g}" for c in ([out] if out.ndim == 2 else np.moveaxis(out, -1, 0))])
    if a.mask:
        m = at.stf(at.to_luminance(img))                 # bright = 1
        m = m[..., None] if img.ndim == 3 else m
        if a.denoise and not a.bias:                     # NR: protect bright areas
            out = m * img + (1 - m) * out
        else:                                            # sharpen: protect background
            out = m * out + (1 - m) * img
    dst = a.out or at.out_name(a.input, "_mlt")
    at.save_image(dst, out, hdr, f"starlet denoise={a.denoise} amount={a.amount} bias={a.bias}")
    print("saved", dst)


if __name__ == "__main__":
    main()
