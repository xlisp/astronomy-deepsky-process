#!/usr/bin/env python
"""End-to-end demo on simulated data with known ground truth.

  python planetary/demo.py                 # -> planetary/demo_out/ (montage.png, table in the terminal)
  python planetary/demo.py --frames 300 --out /tmp/pl

Three captures of Jupiter 5 minutes apart (18:30, 18:35, 18:40) go through the same chain a planetary
imager uses with the real programs:

  synth.py (FireCapture) -> prep.py (PIPP) -> stack.py (AutoStakkert!4) -> derotate.py (WinJUPOS)
                                                                         -> wavelets.py (RegiStax 6)
and every intermediate result is scored against the sharp truth image at 18:35:
  blur  = sigma (px) of the Gaussian that, applied to the truth, best matches the result (lower = sharper)
  resid = remaining rms difference after that match, in % of the disc's rms contrast (noise + artefacts)
  detail= error of the non-zonal detail (spots, festoons, GRS), the part that rotation smears
  sharp = rms error against the sharp truth itself (what sharpening tries to reduce)
All errors are in % of the truth's rms contrast on the disc.
"""
import argparse, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planetools as pt
at = pt.at


def run(*args):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, args[0])] + [str(a) for a in args[1:]],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    last = [l for l in r.stdout.strip().splitlines() if l][-1:] or [""]
    print(f"  {args[0]:<12} {time.time() - t:5.1f} s  {last[0][:110]}")
    return r.stdout


def match(img, truth):
    """Bring truth onto img's grid (centre crop + sub-pixel shift)."""
    I, T = pt.luminance(img), pt.luminance(truth)
    o = (T.shape[0] - I.shape[0]) // 2
    T = T[o:o + I.shape[0], o:o + I.shape[1]]
    dy, dx, _ = pt.phase_correlate(T, I)
    return I, pt.fourier_shift(T, float(dy), float(dx))


def score(img, truth):
    I, T = match(img, truth)
    disc = at.min_filter((at.gaussian_blur(T, 2) > 0.1 * T.max()).astype(np.float32), 9) > 0
    best = None
    for s in np.arange(0, 4.01, 0.1):
        Tb = at.gaussian_blur(T, s) if s > 0 else T
        A = np.vstack([Tb[disc], np.ones(disc.sum())]).T
        c, *_ = np.linalg.lstsq(A, I[disc], rcond=None)
        r = np.sqrt(np.mean((I[disc] - A @ c) ** 2)) / (abs(c[0]) * Tb[disc].std())
        if best is None or r < best[1]:
            best = (s, r, Tb, c)
    s, r, Tb, c = best
    k = at.gaussian_kernel(8)
    hp = lambda x: x - at._conv1d(x, k, 1)           # remove the zonal (banded) component
    det = np.sqrt(np.mean((hp((I - c[1]) / c[0]) - hp(Tb))[disc] ** 2)) / hp(Tb)[disc].std()
    A = np.vstack([T[disc], np.ones(disc.sum())]).T   # error against the SHARP truth (no blur allowed)
    c0, *_ = np.linalg.lstsq(A, I[disc], rcond=None)
    sharp = np.sqrt(np.mean((I[disc] - A @ c0) ** 2)) / (abs(c0[0]) * T[disc].std())
    return s, 100 * r, 100 * det, 100 * sharp


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    ImageDraw.Draw(im).text((4, 3), text, fill=(255, 255, 0))
    return np.asarray(im).astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "demo_out"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seeing", type=float, default=1.8)
    ap.add_argument("--reuse", action="store_true", help="skip steps 1-3 if their outputs exist")
    a = ap.parse_args()
    d = a.out
    os.makedirs(d, exist_ok=True)
    P = lambda n: os.path.join(d, n)
    caps = [("1830_0", "18:30:00"), ("1835_0", "18:35:00"), ("1840_0", "18:40:00")]
    sharpen = ["--sharpen", "3,1.5,0.6,0.2", "--denoise", "0.12,0.05"]

    if a.reuse and all(os.path.exists(P(f"2026-10-13-{t}-j.tif")) for t, _ in caps):
        print("1-3. reusing captures and stacks in", d)
    else:
        print("1. capture (synth.py ~ FireCapture/SharpCap)")
        for i, (tag, hhmmss) in enumerate(caps):
            run("synth.py", "-o", P(f"cap_{tag}.ser"), "--frames", a.frames, "--time", f"2026-10-13T{hhmmss}",
                "--cm0", 40 + pt.rotation_rate("jupiter") * 5 * i / 1440, "--seeing", a.seeing,
                "--seed", i + 1, "--truth", P(f"truth_{tag}.npy"))
        print("2. pre-process (prep.py ~ PIPP): debayer, centre, crop 200x200")
        for tag, _ in caps:
            run("prep.py", P(f"cap_{tag}.ser"), "-o", P(f"pipp_{tag}.ser"), "--crop", 200)
        print("3. stack (stack.py ~ AutoStakkert!4): best 25 %, AP 32 px")
        for tag, _ in caps:
            run("stack.py", P(f"pipp_{tag}.ser"), "-o", P(f"2026-10-13-{tag}-j.tif"),
                "--save-global", P(f"global_{tag}.tif"), "--keep", "25%", "--ap-size", 32)
    print("4. sharpen one stack (wavelets.py ~ RegiStax 6): RGB align + 4 layers")
    run("wavelets.py", P("2026-10-13-1835_0-j.tif"), "-o", P("wav_1835.tif"), "--rgb-align", "--dering", 0.6, *sharpen)
    print("5. derotate the 3 stacks to 18:35 (derotate.py ~ WinJUPOS), then sharpen")
    stacks = [P(f"2026-10-13-{tag}-j.tif") for tag, _ in caps]
    run("derotate.py", "stack", *stacks, "-o", P("derot.tif"), "--de", 3, "--t0", "2026-10-13T18:35:00",
        "--naive", P("naive.tif"))
    for n in ("derot", "naive"):
        run("wavelets.py", P(f"{n}.tif"), "-o", P(f"{n}_wav.tif"), "--rgb-align", "--dering", 0.6,
            "--sharpen", "4,2,0.7,0.2", "--denoise", "0.10,0.04")

    truth = np.load(P("truth_1835_0.npy"))
    ser = pt.SerReader(P("pipp_1835_0.ser"))
    q = [pt.frame_quality(ser[i]) for i in range(len(ser))]
    rows = [("best single frame", ser[int(np.argmax(q))]),
            ("median single frame", ser[int(np.argsort(q)[len(q) // 2])]),
            ("stack, global alignment only", pt.load(P("global_1835_0.tif"))),
            ("stack, alignment points (AS!4)", pt.load(P("2026-10-13-1835_0-j.tif"))),
            ("  + wavelets (RegiStax)", pt.load(P("wav_1835.tif"))),
            ("3 stacks averaged, no derotation", pt.load(P("naive.tif"))),
            ("  + wavelets", pt.load(P("naive_wav.tif"))),
            ("3 stacks derotated (WinJUPOS)", pt.load(P("derot.tif"))),
            ("  + wavelets", pt.load(P("derot_wav.tif")))]
    print(f"\n{'result (vs truth at 18:35)':<36}{'blur px':>8}{'resid %':>9}{'detail %':>10}{'sharp %':>9}")
    for name, img in rows:
        s, r, det, sh = score(img, truth)
        print(f"{name:<36}{s:8.1f}{r:9.1f}{det:10.1f}{sh:9.1f}")

    o = (truth.shape[0] - 200) // 2
    tiles = [label(rows[0][1] / 0.8, "best frame"), label(rows[3][1] / 0.8, "AP stack"),
             label(rows[4][1] / 0.8, "+ wavelets"), label(rows[6][1] / 0.8, "3x no derot"),
             label(rows[8][1] / 0.8, "3x derotated"), label(truth[o:o + 200, o:o + 200] / 0.8, "truth")]
    pt.save(P("montage.png"), np.hstack(tiles))
    print("\nmontage ->", P("montage.png"))


if __name__ == "__main__":
    main()
