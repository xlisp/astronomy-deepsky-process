"""Stage 1: calibrate (cosmetic), register and sigma-clip stack each filter.
Filters: 1=L 2=R 3=G 4=B 5=S(SII) 6=H(Ha) 7=O(OIII)
Output: process/master_<F>.fits (float32, aligned to a common reference)
"""
import glob, os, re, sys
import numpy as np
import cv2
import sep
import astroalign as aa
from astropy.io import fits

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SRC, "process")
NAMES = {1: "L", 2: "R", 3: "G", 4: "B", 5: "S", 6: "H", 7: "O"}


def load(path):
    d = fits.getdata(path).astype(np.float32)
    # cosmetic correction: kill hot pixels (no darks available)
    med = cv2.medianBlur(d, 3)
    diff = d - med
    s = 1.4826 * np.median(np.abs(diff[::4, ::4]))
    hot = diff > 8 * s
    d[hot] = med[hot]
    return d


def stars(img, n=150, thresh=5):
    bkg = sep.Background(img)
    sub = img - bkg
    objs = sep.extract(sub, thresh, err=bkg.globalrms, minarea=6)
    objs = objs[(objs["flag"] == 0) & (objs["peak"] < 60000)]
    objs = objs[np.argsort(-objs["flux"])][:n]
    return np.c_[objs["x"], objs["y"]], objs


def guided_transform(xy, ref_xy, T0, radius=30):
    """Match stars using a predicted transform T0 (2x3), refine with RANSAC similarity fit."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_xy)
    T = T0
    for r in (radius, 6, 3):
        p = xy @ T[:, :2].T + T[:, 2]
        d, idx = tree.query(p, distance_upper_bound=r)
        ok = np.isfinite(d)
        if ok.sum() < 4:
            raise RuntimeError(f"only {ok.sum()} matches")
        M, inl = cv2.estimateAffinePartial2D(xy[ok].astype(np.float32), ref_xy[idx[ok]].astype(np.float32),
                                             method=cv2.RANSAC, ransacReprojThreshold=2.0)
        T = M
    return T, int(inl.sum())


def fwhm_score(objs):
    a = np.median(objs["a"]); b = np.median(objs["b"])
    return a, b


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.fit")))
    groups = {}
    for f in files:
        m = re.search(r"Bin1_(\d)_", os.path.basename(f))
        groups.setdefault(int(m.group(1)), []).append(f)

    # reference: sharpest L frame
    best = None
    for f in groups[1]:
        img = load(f)
        _, o = stars(img)
        a, b = fwhm_score(o)
        print(f"  L {os.path.basename(f)[-12:]} a={a:.2f} b={b:.2f} n={len(o)}")
        if best is None or a < best[0]:
            best = (a, f)
    ref = load(best[1])
    ref_xy, _ = stars(ref, 200)
    ref_xy_dense, _ = stars(ref, 1500, thresh=3)
    prev_T = np.float32([[1, 0, 0], [0, 1, 0]])
    print("reference:", os.path.basename(best[1]))
    H, W = ref.shape

    only = [int(x) for x in sys.argv[1:]] or sorted(groups)
    for k in sorted(groups):
        name = NAMES[k]
        if k not in only:
            continue
        stack, weights = [], []
        for f in groups[k]:
            img = load(f)
            xy, o = stars(img, 200)
            try:
                if len(o) < 60:
                    raise RuntimeError("too few stars for triangles")
                T, _ = aa.find_transform(xy, ref_xy, max_control_points=60)
                M = T.params[:2].astype(np.float32)
                how = "tri"
            except Exception:
                # sparse narrowband frame: use previous transform as prediction
                xy, o = stars(img, 200, thresh=3)
                try:
                    M, ninl = guided_transform(xy, ref_xy_dense, prev_T)
                    how = f"guided({ninl})"
                except Exception as e:
                    print(f"  {name} skip {os.path.basename(f)}: {e}")
                    continue
            prev_T = M
            warped = cv2.warpAffine(img, M.astype(np.float32), (W, H),
                                    flags=cv2.INTER_LANCZOS4, borderValue=np.nan)
            a, _ = fwhm_score(o)
            bkg = np.nanmedian(warped[::8, ::8])
            stack.append(warped - bkg)
            weights.append(1.0 / a**2)
            print(f"  {name} {os.path.basename(f)[-12:]} shift=({M[0, 2]:.1f},{M[1, 2]:.1f}) "
                  f"rot={np.degrees(np.arctan2(M[1, 0], M[0, 0])):.3f} fwhm_a={a:.2f} {how}")
        cube = np.stack(stack)
        w = np.array(weights, np.float32)[:, None, None]
        # winsorized sigma clipping
        med = np.nanmedian(cube, axis=0)
        sd = 1.4826 * np.nanmedian(np.abs(cube - med), axis=0)
        mask = np.abs(cube - med) > 3.0 * np.maximum(sd, 1e-3)
        cube[mask] = np.nan
        ww = np.where(np.isnan(cube), 0, w)
        master = np.nansum(cube * w, axis=0) / np.maximum(ww.sum(0), 1e-9)
        master[ww.sum(0) == 0] = 0
        fits.writeto(os.path.join(OUT, f"master_{name}.fits"), master.astype(np.float32), overwrite=True)
        print(f"== {name}: {len(stack)} frames stacked, rejected {mask.mean()*100:.2f}%")
        del cube, stack


if __name__ == "__main__":
    main()
