#!/usr/bin/env python
"""Richardson-Lucy deconvolution with a Gaussian PSF measured from the stars.

PixInsight equivalent: DynamicPSF + Deconvolution (BlurXTerminator is the AI successor).

  python scripts/deconvolve.py master_L.fits -o L_deconv.fits                 # PSF from stars
  python scripts/deconvolve.py master_L.fits --fwhm 4.2 --iters 30 --strength 0.7

Richardson-Lucy:  u_{k+1} = u_k * ( P^T conv ( d / (P conv u_k) ) )
The Gaussian PSF is applied in Fourier space: FT(gauss) = exp(-2 pi^2 sigma^2 (fx^2 + fy^2)).
RL amplifies noise and rings around stars, so the result is blended through
  * a luminance mask  (only high-SNR areas get deconvolved)
  * a deringing mask  (stars brighter than --dering x max keep the original profile)
Star size before/after is reported as 2*HFR (half-flux radius), which ringing does not bias.
Use on LINEAR data, before stretching.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astrotools as at


def richardson_lucy(img, fwhm, iters=20):
    sigma = fwhm / 2.3548
    pad = int(4 * sigma) + 8
    d = np.pad(img, pad, mode="reflect")
    fy = np.fft.fftfreq(d.shape[0])[:, None]
    fx = np.fft.rfftfreq(d.shape[1])[None, :]
    otf = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    conv = lambda x: np.fft.irfft2(np.fft.rfft2(x) * otf, s=x.shape)
    u = d.copy()
    for i in range(iters):
        u *= conv(d / np.maximum(conv(u), 1e-7))
    return u[pad:-pad, pad:-pad].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("-o", "--out")
    ap.add_argument("--fwhm", type=float, help="PSF FWHM in px (default: median of detected stars)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--strength", type=float, default=1.0, help="0..1 blend with original")
    ap.add_argument("--dering", type=float, default=0.5, help="protect stars above this fraction of max")
    ap.add_argument("--no-mask", action="store_true", help="skip luminance/deringing masks")
    a = ap.parse_args()

    img, hdr = (at.load_fits(a.input, True) if at.is_fits(a.input) else (at.load_image(a.input), None))
    lum = at.to_luminance(img)
    fwhm = a.fwhm
    if fwhm is None:
        st = at.detect_stars(lum, k=10, max_stars=200)
        fwhm = float(np.median(st["fwhm"]))
        print(f"PSF from {len(st['x'])} stars: FWHM={fwhm:.2f}px ecc={np.median(st['ecc']):.2f} "
              f"2*HFR={2 * np.median(st['hfr']):.2f}px")
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    x = (img - lo) / (hi - lo) + 1e-4                     # RL needs strictly positive data
    chans = [x] if x.ndim == 2 else [x[..., c] for c in range(x.shape[2])]
    dec = [richardson_lucy(c, fwhm, a.iters) for c in chans]
    dec = dec[0] if x.ndim == 2 else np.dstack(dec)
    w = np.full(lum.shape, a.strength, np.float32)
    if not a.no_mask:
        lmask = np.clip((at.stf(lum, target=0.15) - 0.2) / 0.3, 0, 1)   # background 0, signal 1
        bright = (at.normalize(lum) > a.dering).astype(np.float32)
        big = np.clip(at.gaussian_blur(at.max_filter(bright, 2 * int(2 * fwhm) + 1), fwhm / 2), 0, 1)
        w *= lmask * (1 - big)
    w = w[..., None] if x.ndim == 3 else w
    out = (w * dec + (1 - w) * x - 1e-4) * (hi - lo) + lo
    if a.fwhm is None:
        st2 = at.detect_stars(at.to_luminance(out), k=10, max_stars=200)
        print(f"after: 2*HFR={2 * np.median(st2['hfr']):.2f}px")
    dst = a.out or at.out_name(a.input, "_deconv")
    at.save_image(dst, out if at.is_fits(dst) else at.normalize(out), hdr,
                  f"RL deconvolution fwhm={fwhm:.2f} iters={a.iters}")
    print("saved", dst)


if __name__ == "__main__":
    main()
