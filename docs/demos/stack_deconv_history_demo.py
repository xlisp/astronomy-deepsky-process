#!/usr/bin/env python
"""Numerical experiments behind docs/stacking_deconvolution_history.md (numpy only).

  python docs/demos/stack_deconv_history_demo.py

Part 1  stacking: why averaging N frames gives sqrt(N), why median is ~1.25x noisier,
        why sigma clipping (Peirce 1852 / Chauvenet 1863 -> IRAF imcombine) beats both with outliers.
Part 2  deconvolution, in historical order on a 1-D "two close stars + nebula" signal:
        inverse filter -> van Cittert (1931) -> Wiener (1942/49) -> Tikhonov (1963)
        -> Richardson-Lucy (1972/74) -> CLEAN (Hogbom 1974).
"""
import numpy as np

rng = np.random.default_rng(42)


# ----------------------------------------------------------------------------- part 1
def part1():
    print("=== Part 1: stacking ===")
    sigma, S = 10.0, 100.0
    print(f"{'N':>4} {'mean noise':>11} {'sigma/sqrtN':>12} {'median noise':>13} {'median/mean':>12}")
    for n in (1, 4, 16, 64):
        frames = S + rng.normal(0, sigma, (n, 200_000))
        m, md = frames.mean(0), np.median(frames, 0)
        print(f"{n:4d} {m.std():11.3f} {sigma / np.sqrt(n):12.3f} {md.std():13.3f} {md.std() / m.std():12.3f}")
    print("  theory for large N: median/mean = sqrt(pi/2) = 1.2533  (median efficiency 2/pi = 64%)")

    # Poisson photon noise: signal S photons -> noise sqrt(S); stacking adds photons
    s = 50
    one = rng.poisson(s, 200_000)
    stack = rng.poisson(s, (16, 200_000)).sum(0)
    print(f"  Poisson: 1 frame SNR {s / one.std():.2f} (sqrt(S)={np.sqrt(s):.2f}); "
          f"16 frames SNR {16 * s / stack.std():.2f} (x{(16 * s / stack.std()) / (s / one.std()):.2f})")

    # outliers: cosmic rays / satellite in 1 of N frames
    n = 10
    frames = S + rng.normal(0, sigma, (n, 200_000))
    hit = rng.random(200_000) < 0.05                       # 5% of pixels get a hit in frame 0
    frames[0, hit] += 2000
    mean = frames.mean(0)
    med = np.median(frames, 0)
    c = frames.copy()
    for _ in range(2):
        mm = np.nanmedian(c, 0)
        sd = 1.4826 * np.nanmedian(np.abs(c - mm), 0)
        c[np.abs(c - mm) > 3 * sd] = np.nan
    clip = np.nanmean(c, 0)
    rms = lambda x: np.sqrt(np.mean((x - S) ** 2))
    print(f"  N={n}, 5% pixels hit by a 2000-ADU cosmic ray in one frame:")
    print(f"    mean  rms error {rms(mean):7.2f}   (outliers leak in: +2000/N)")
    print(f"    median rms error {rms(med):7.2f}")
    print(f"    sigma-clip mean {rms(clip):7.2f}   (ideal sigma/sqrtN = {sigma / np.sqrt(n):.2f})")


# ----------------------------------------------------------------------------- part 2
def gauss_kernel_ft(n, fwhm):
    s = fwhm / 2.3548
    f = np.fft.rfftfreq(n)
    return np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2)


def conv(x, H):
    return np.fft.irfft(np.fft.rfft(x) * H, n=len(x))


def part2():
    print("\n=== Part 2: deconvolution (1-D) ===")
    n, fwhm = 512, 10.0
    x = np.arange(n)
    truth = np.zeros(n)
    a_, b_ = 250, 258                                      # two stars 8 px apart (< FWHM 10: one blob)
    truth[a_] = 100.0
    truth[b_] = 70.0
    truth += 3.0 * np.exp(-0.5 * ((x - 380) / 25) ** 2)    # faint nebula
    truth += 1.0                                           # sky
    H = gauss_kernel_ft(n, fwhm)
    blurred = conv(truth, H)
    data = rng.poisson(np.clip(blurred, 0, None) * 2000) / 2000.0   # photon noise (high-SNR stack)

    def report(name, est):
        pk = min(est[a_ - 2:a_ + 3].max(), est[b_ - 2:b_ + 3].max())
        dip = est[a_ + 2:b_ - 1].min() / pk if pk > 0 else np.inf  # < 1: a valley between two peaks
        ok = np.isfinite(dip) and 0 <= dip < 0.8 and np.abs(est).max() < 1e4
        print(f"  {name:<28} rms {np.sqrt(np.mean((est - truth) ** 2)):10.3g}   "
              f"valley/peak {dip:9.3g}   min {est.min():9.3g}  {'RESOLVED' if ok else 'blended / broken'}")

    report("observed (blurred)", data)

    # inverse filter: D/H  -> noise at high frequency divided by ~0
    report("inverse filter 1/H", np.fft.irfft(np.fft.rfft(data) / np.maximum(H, 1e-12), n=n))

    # van Cittert 1931: u <- u + (d - h*u)
    u = data.copy()
    for it in range(1, 201):
        u = u + (data - conv(u, H))
        if it in (10, 200):
            report(f"van Cittert ({it} it)", u.copy())

    # Wiener 1942/49: H* / (|H|^2 + N/S)
    nsr = 1e-4
    report("Wiener (NSR=1e-4)", np.fft.irfft(np.fft.rfft(data) * H / (H ** 2 + nsr), n=n))

    # Tikhonov 1963 with a gradient (smoothness) penalty
    lam = 1e-4
    lap = (2 * np.sin(np.pi * np.fft.rfftfreq(n))) ** 2
    report("Tikhonov (gradient, 1e-4)", np.fft.irfft(np.fft.rfft(data) * H / (H ** 2 + lam * lap), n=n))

    # Richardson 1972 / Lucy 1974: u <- u * H^T( d / (H u) )
    u = np.full(n, data.mean())
    for it in range(1, 1001):
        u *= conv(data / np.maximum(conv(u, H), 1e-9), H)
        if it in (50, 1000):
            report(f"Richardson-Lucy ({it} it)", u.copy())

    # CLEAN, Hogbom 1974: repeatedly subtract a scaled PSF at the peak
    psf = conv(np.eye(1, n, n // 2)[0], H)
    psf /= psf.max()
    res = data - np.median(data)
    comps = np.zeros(n)
    for _ in range(3000):
        k = np.argmax(res)
        if res[k] < 0.05:
            break
        a = 0.1 * res[k]
        comps[k] += a
        res -= a * np.roll(psf, k - n // 2)
    beam = conv(np.eye(1, n, n // 2)[0], gauss_kernel_ft(n, 3.0))
    beam /= beam.max()
    clean = np.fft.irfft(np.fft.rfft(comps) * np.fft.rfft(np.roll(beam, -n // 2)), n=n) + res \
        + np.median(data)
    report("CLEAN (gain 0.1, beam 3px)", clean)
    print("  (CLEAN restores with a narrower 'clean beam', so its rms vs. the delta-function truth is"
          " not comparable; look at valley/peak)")


if __name__ == "__main__":
    part1()
    part2()
