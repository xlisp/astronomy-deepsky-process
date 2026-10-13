"""Shared helpers for the planetary (lucky imaging) scripts.

Only numpy is required (PIL for PNG/JPG, OpenCV optional for AVI/MP4). Generic filters, wavelets and
image I/O are reused from ../scripts/astrotools.py.

Contents
  SER video read/write ............ SerReader, write_ser, open_frames
  Bayer demosaic .................. debayer
  sub-pixel resampling ............ bilinear, fourier_shift
  registration .................... phase_correlate (batched, sub-pixel)
  frame quality ................... gradient_energy, frame_quality
  planet disc ..................... find_disk
  spheroid geometry (WinJUPOS) .... PLANETS, image_to_planet, planet_to_image, ray_spheroid, rotate_view
  times ........................... parse_time, ticks_to_dt, dt_to_ticks
"""
import datetime as dt
import glob, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import astrotools as at  # noqa: E402  (gaussian_blur, starlet, save_image, load_image, mad_std ...)


# ============================================================================= SER video
# SER v3 (Heiko Wilkens / Grischa Hahn): 178-byte header, raw frames, optional int64 timestamp trailer.
SER_COLOR = {0: "MONO", 8: "RGGB", 9: "GRBG", 10: "GBRG", 11: "BGGR", 100: "RGB", 101: "BGR"}
SER_ID = {v: k for k, v in SER_COLOR.items()}
_SER_HDR = struct.Struct("<14s7i40s40s40sqq")
_EPOCH = dt.datetime(1, 1, 1)          # SER time = 100 ns ticks since 0001-01-01 (UTC for DateTime_UTC)


def ticks_to_dt(t):
    return _EPOCH + dt.timedelta(microseconds=int(t) // 10)


def dt_to_ticks(d):
    return ((d - _EPOCH) // dt.timedelta(microseconds=1)) * 10


class SerReader:
    """Memory-mapped SER file. .frames[i] is the raw frame (H,W) or (H,W,3) in file dtype."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            hdr = f.read(_SER_HDR.size)
        (fid, self.lu_id, self.color_id, self.little_endian, self.width, self.height, self.depth,
         self.count, obs, ins, tel, self.date, self.date_utc) = _SER_HDR.unpack(hdr)
        if not fid.startswith(b"LUCAM-RECORDER"):
            raise ValueError(f"{path}: not a SER file")
        self.observer, self.instrument, self.telescope = (
            s.rstrip(b"\0 ").decode("latin-1") for s in (obs, ins, tel))
        self.planes = 3 if self.color_id in (100, 101) else 1
        # The LittleEndian flag is set inconsistently by capture programs; virtually all real files
        # are little-endian, which is what PIPP / AS!4 / FireCapture assume too.
        self.dtype = np.dtype("<u2") if self.depth > 8 else np.dtype("u1")
        shape = (self.count, self.height, self.width) + ((3,) if self.planes == 3 else ())
        self.frames = np.memmap(path, self.dtype, "r", _SER_HDR.size, shape)
        self.scale = 1.0 / (2 ** self.depth - 1)
        end = _SER_HDR.size + self.frames.nbytes
        self.times = None
        if os.path.getsize(path) >= end + 8 * self.count:
            ts = np.fromfile(path, "<i8", self.count, offset=end)
            if np.all(ts > 0):
                self.times = [ticks_to_dt(t) for t in ts]

    @property
    def bayer(self):
        p = SER_COLOR.get(self.color_id, "MONO")
        return p if len(p) == 4 else None

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        """Frame i as float32 in [0,1]; BGR files are returned as RGB."""
        a = np.asarray(self.frames[i], np.float32) * self.scale
        return a[..., ::-1] if self.color_id == 101 else a


def write_ser(path, frames, color="MONO", depth=None, times=None, observer="", instrument="",
              telescope=""):
    """frames: uint8/uint16 array (N,H,W) or (N,H,W,3) (color 'RGB'). times: list of datetime (UTC)."""
    frames = np.asarray(frames)
    if frames.dtype not in (np.uint8, np.uint16):
        raise TypeError("write_ser needs uint8 or uint16 frames")
    depth = depth or (8 if frames.dtype == np.uint8 else 16)
    n, h, w = frames.shape[:3]
    t0 = dt_to_ticks(times[0]) if times else 0
    hdr = _SER_HDR.pack(b"LUCAM-RECORDER", 0, SER_ID[color], 1, w, h, depth, n,
                        observer.encode()[:40], instrument.encode()[:40], telescope.encode()[:40], t0, t0)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(frames.astype(frames.dtype.newbyteorder("<"), copy=False).tobytes())
        if times:
            f.write(np.array([dt_to_ticks(t) for t in times], "<i8").tobytes())


def to_uint(img, depth):
    """float [0,1] -> uint8 (depth 8) or uint16 holding `depth` bits."""
    m = 2 ** depth - 1
    a = np.clip(np.nan_to_num(img), 0, 1) * m + 0.5
    return a.astype(np.uint8 if depth <= 8 else np.uint16)


class _ListSource:
    def __init__(self, frames, times=None, bayer=None, depth=8):
        self.frames, self.times, self.bayer, self.depth = frames, times, bayer, depth

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        return at.load_image(f) if isinstance(f, str) else f


def open_frames(spec):
    """SER file, AVI/MP4 (needs OpenCV), or a glob / list of image files (FITS, TIFF, PNG ...).

    Returns an object with len(), [i] -> float32 frame, .times (list|None), .bayer (str|None), .depth.
    """
    if isinstance(spec, str) and spec.lower().endswith(".ser"):
        return SerReader(spec)
    if isinstance(spec, str) and spec.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
        import cv2
        cap, frames = cv2.VideoCapture(spec), []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[..., ::-1].astype(np.float32) / 255 if fr.ndim == 3 else fr / 255.0)
        return _ListSource(frames)
    files = sorted(glob.glob(spec)) if isinstance(spec, str) else list(spec)
    if not files:
        raise FileNotFoundError(spec)
    return _ListSource(files, times=[parse_time(os.path.basename(f), strict=False) for f in files], depth=16)


# ============================================================================= Bayer
def conv3(img, k):
    """3x3 correlation with reflect borders (reflect keeps the Bayer parity at the edges)."""
    p = np.pad(img, 1, mode="reflect")
    h, w = img.shape
    return sum(k[i, j] * p[i:i + h, j:j + w] for i in range(3) for j in range(3) if k[i, j])


_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4


def debayer(raw, pattern="RGGB"):
    """Bilinear demosaic. pattern = colours of the 2x2 cell read row-wise, e.g. RGGB."""
    h, w = raw.shape
    out = []
    for c in "RGB":
        m = np.zeros((h, w), np.float32)
        for i, pc in enumerate(pattern.upper()):
            if pc == c:
                m[i // 2::2, i % 2::2] = 1
        out.append(conv3(raw * m, _K_G if c == "G" else _K_RB))
    return np.dstack(out)


def mosaic(rgb, pattern="RGGB"):
    """Inverse of debayer (used by the simulator)."""
    raw = np.zeros(rgb.shape[:2], rgb.dtype)
    for i, pc in enumerate(pattern.upper()):
        raw[i // 2::2, i % 2::2] = rgb[i // 2::2, i % 2::2, "RGB".index(pc)]
    return raw


def luminance(img):
    return img if img.ndim == 2 else img.mean(-1)


# ============================================================================= resampling
def bilinear(img, y, x, k=None):
    """Sample img at float coordinates (clamped to the border).

    img: (H,W) or (H,W,C); or, with k given, a stack (N,H,W[,C]) sampled at frame indices k.
    y, x, k broadcast together; result shape = broadcast shape (+ (C,)).
    """
    h, w = (img.shape[1:3] if k is not None else img.shape[:2])
    y = np.clip(y, 0, h - 1.0001)
    x = np.clip(x, 0, w - 1.0001)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    fy = (y - y0).astype(np.float32)
    fx = (x - x0).astype(np.float32)
    idx = (lambda yy, xx: img[k, yy, xx]) if k is not None else (lambda yy, xx: img[yy, xx])
    a, b, c, d = idx(y0, x0), idx(y0, x0 + 1), idx(y0 + 1, x0), idx(y0 + 1, x0 + 1)
    if a.ndim > fy.ndim:
        fy, fx = fy[..., None], fx[..., None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fourier_shift(img, dy, dx):
    """Shift by (dy, dx) px using the Fourier shift theorem (no interpolation blur, wraps around)."""
    if img.ndim == 3:
        return np.dstack([fourier_shift(img[..., c], dy, dx) for c in range(img.shape[2])])
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    ph = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.irfft2(np.fft.rfft2(img) * ph, s=(h, w)).astype(np.float32)


# ============================================================================= registration
def hann(n):
    """Periodic Hann window: hann(n)[i] + hann(n)[i + n/2] == 1 (partition of unity at 50% overlap)."""
    return (np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2).astype(np.float32)


def phase_correlate(ref, img, power=0.5, window=True):
    """Sub-pixel translation between images (also batched over leading axes).

    Returns (dy, dx, peak) with img(y, x) ~= ref(y - dy, x - dx), i.e. img is ref moved by +d.
    power: spectrum whitening exponent; 1 = pure phase correlation (sharp peak, noise sensitive),
           0 = plain cross-correlation (robust, broad peak). 0.5 is a good compromise for planets.
    """
    ref = np.asarray(ref, np.float32)
    img = np.asarray(img, np.float32)
    h, w = img.shape[-2:]
    ref = ref - ref.mean(axis=(-2, -1), keepdims=True)
    img = img - img.mean(axis=(-2, -1), keepdims=True)
    if window:
        win = np.outer(hann(h), hann(w))
        ref, img = ref * win, img * win
    cross = np.fft.fft2(img) * np.conj(np.fft.fft2(ref))
    if power:
        cross /= (np.abs(cross) + 1e-12) ** power
    c = np.fft.ifft2(cross).real.reshape(-1, h, w)
    flat = c.reshape(len(c), -1)
    i = flat.argmax(1)
    py, px = i // w, i % w
    b = np.arange(len(c))

    def parabola(lo, mid, hi):
        den = lo - 2 * mid + hi
        return np.where(np.abs(den) > 1e-12, 0.5 * (lo - hi) / np.where(den == 0, 1, den), 0.0)

    m = c[b, py, px]
    oy = parabola(c[b, (py - 1) % h, px], m, c[b, (py + 1) % h, px])
    ox = parabola(c[b, py, (px - 1) % w], m, c[b, py, (px + 1) % w])
    dy = np.where(py > h // 2, py - h, py) + np.clip(oy, -0.5, 0.5)
    dx = np.where(px > w // 2, px - w, px) + np.clip(ox, -0.5, 0.5)
    peak = m / (np.abs(flat).sum(1) + 1e-12)
    shp = np.broadcast_shapes(ref.shape[:-2], img.shape[:-2])
    return dy.reshape(shp), dx.reshape(shp), peak.reshape(shp)


# ============================================================================= quality
def gradient_energy(img, sigma=1.0):
    """Squared gradient magnitude of the Gaussian-smoothed image (smoothing = noise robustness)."""
    s = at.gaussian_blur(luminance(img).astype(np.float32), sigma)
    gy = np.zeros_like(s)
    gx = np.zeros_like(s)
    gy[1:-1] = 0.5 * (s[2:] - s[:-2])
    gx[:, 1:-1] = 0.5 * (s[:, 2:] - s[:, :-2])
    return gy * gy + gx * gx


def frame_quality(img, sigma=1.0, method="gradient"):
    """Sharpness score of one frame, normalised by brightness^2 so thin cloud does not bias it.

    gradient : mean squared gradient after smoothing (AutoStakkert-like "local contrast")
    laplace  : variance of the Laplacian of the smoothed image (PIPP's default style estimator)
    """
    lum = luminance(img).astype(np.float32)
    mean = float(lum.mean()) + 1e-9
    if method == "gradient":
        return float(gradient_energy(lum, sigma).mean()) / mean ** 2
    s = at.gaussian_blur(lum, sigma)
    lap = s[1:-1, 1:-1] * 4 - s[:-2, 1:-1] - s[2:, 1:-1] - s[1:-1, :-2] - s[1:-1, 2:]
    return float(lap.var()) / mean ** 2


# ============================================================================= planet disc
def find_disk(img, frac=0.2, flattening=0.0, smooth=1.5):
    """Locate the planet: threshold at bg + frac*(peak-bg), keep the blob around the main mass.

    Returns dict(cx, cy, a (equatorial radius), area, pole (deg, pole axis angle from image 'up',
    counter-clockwise, modulo 180), bbox (y0, y1, x0, x1), edge (touches frame border)) or None.
    """
    lum = at.gaussian_blur(luminance(img).astype(np.float32), smooth)
    h, w = lum.shape
    border = np.concatenate([lum[:3].ravel(), lum[-3:].ravel(), lum[:, :3].ravel(), lum[:, -3:].ravel()])
    bg = float(np.median(border))
    pk = float(np.percentile(lum, 99.7))
    if pk - bg <= 1e-6:
        return None
    mask = lum > bg + frac * (pk - bg)
    yy, xx = np.mgrid[0:h, 0:w]
    if mask.sum() < 20:
        return None
    sel = mask
    for _ in range(3):                      # drop moons / hot pixels far from the main body
        ys, xs = yy[sel], xx[sel]
        cy, cx = ys.mean(), xs.mean()
        r = np.sqrt(sel.sum() / np.pi)
        sel = mask & ((yy - cy) ** 2 + (xx - cx) ** 2 < (1.25 * r + 3) ** 2)
    ys, xs = yy[sel].astype(np.float64), xx[sel].astype(np.float64)
    cy, cx = ys.mean(), xs.mean()
    area = float(sel.sum())
    cov = np.cov(np.vstack([xs - cx, cy - ys]))           # (x right, y up)
    ev, vec = np.linalg.eigh(cov)
    minor = vec[:, 0]                                     # smallest spread = polar axis
    pole = float(np.degrees(np.arctan2(-minor[0], minor[1]))) % 180
    a = float(np.sqrt(area / (np.pi * (1 - flattening))))
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    return dict(cx=float(cx), cy=float(cy), a=a, area=area, pole=pole, bbox=(y0, y1, x0, x1),
                edge=y0 <= 1 or x0 <= 1 or y1 >= h - 2 or x1 >= w - 2)


# ============================================================================= spheroid geometry
# rotation rates in degrees/day; flattening f = (a-b)/a. Values as used by WinJUPOS / IAU.
PLANETS = {
    "jupiter": dict(flattening=0.06487, systems={"I": 877.900, "II": 870.270, "III": 870.536}, default="II"),
    "saturn": dict(flattening=0.09796, systems={"III": 810.7939024, "I": 844.300}, default="III"),
    "mars": dict(flattening=0.00589, systems={"": 350.891983}, default=""),
}


def rotation_rate(planet, system=None):
    p = PLANETS[planet.lower()]
    return p["systems"][p["default"] if system is None else system]


def image_to_planet(x, y, cx, cy, a, north=0.0, mirror=False):
    """Pixel -> planet-plane coords (u towards celestial west / preceding limb, v towards north),
    in units of the equatorial radius. north: pole direction, degrees counter-clockwise from 'up'.
    mirror: image is mirror-reversed (star diagonal, Newtonian with odd reflections)."""
    p = np.radians(north)
    dx, dy = (x - cx) / a, (cy - y) / a
    u = dx * np.cos(p) + dy * np.sin(p)
    v = -dx * np.sin(p) + dy * np.cos(p)
    return (-u if mirror else u), v


def planet_to_image(u, v, cx, cy, a, north=0.0, mirror=False):
    p = np.radians(north)
    u = -u if mirror else u
    dx = u * np.cos(p) - v * np.sin(p)
    dy = u * np.sin(p) + v * np.cos(p)
    return cx + dx * a, cy - dy * a


def ray_spheroid(u, v, flattening, de):
    """Orthographic view of a spheroid (a=1, b=1-f) from sub-observer latitude de (deg).

    Body frame: X towards the observer's central meridian (in the equator plane), Y west (= +u),
    Z north. Returns X, Y, Z of the visible surface point and valid mask (on the disc).
    """
    b = 1.0 - flattening
    s, c = np.sin(np.radians(de)), np.cos(np.radians(de))
    x0, z0 = -v * s, v * c                     # point in the image plane, then march along d=(c,0,s)
    A = c * c + s * s / b ** 2
    B = 2 * (x0 * c + z0 * s / b ** 2)
    C = x0 * x0 + u * u + z0 * z0 / b ** 2 - 1
    disc = B * B - 4 * A * C
    ok = disc >= 0
    t = (-B + np.sqrt(np.where(ok, disc, 0))) / (2 * A)   # far root = side facing the observer
    return x0 + t * c, u, z0 + t * s, ok


def project_spheroid(X, Y, Z, flattening, de):
    """Inverse of ray_spheroid: body coords -> (u, v, mu, visible); mu = cos(emission angle)."""
    b = 1.0 - flattening
    s, c = np.sin(np.radians(de)), np.cos(np.radians(de))
    u, v = Y, -s * X + c * Z
    nx, ny, nz = X, Y, Z / b ** 2
    mu = (nx * c + nz * s) / np.sqrt(nx * nx + ny * ny + nz * nz)
    return u, v, mu, mu > 0


def surface_latlon(X, Y, Z, flattening):
    """Planetographic latitude and longitude offset from the CM (deg, positive towards west)."""
    b = 1.0 - flattening
    lat = np.degrees(np.arctan2(Z / b ** 2, np.hypot(X, Y)))
    return lat, np.degrees(np.arctan2(Y, X))


def rotate_view(u, v, delta_deg, flattening, de):
    """Where was the feature now seen at (u, v) when the planet was rotated back by delta_deg?

    Positive delta = the output time is later than the source time. Returns u_src, v_src, mu_src,
    valid (on the disc AND visible at the source time), on_disc (output pixel lies on the disc).
    """
    X, Y, Z, on = ray_spheroid(u, v, flattening, de)
    d = np.radians(delta_deg)
    Xs = X * np.cos(d) + Y * np.sin(d)          # longitude-from-CM decreases by delta
    Ys = -X * np.sin(d) + Y * np.cos(d)
    us, vs, mu, vis = project_spheroid(Xs, Ys, Z, flattening, de)
    return us, vs, mu, on & vis, on


# ============================================================================= times
_FC_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(?:[_.](\d))?")


def parse_time(s, strict=True):
    """ISO time ('2026-10-13T18:34:30') or WinJUPOS/FireCapture file name ('2026-10-13-1834_5-...',
    _5 = tenths of a minute). Returns naive UTC datetime."""
    if isinstance(s, dt.datetime):
        return s
    try:
        return dt.datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        pass
    m = _FC_NAME.search(s)
    if m:
        y, mo, d, hh, mm, tenth = m.groups()
        return dt.datetime(int(y), int(mo), int(d), int(hh), int(mm)) + dt.timedelta(seconds=6 * int(tenth or 0))
    if strict:
        raise ValueError(f"cannot parse a time from {s!r}")
    return None


def winjupos_name(t, planet="j", suffix=""):
    """WinJUPOS file-name convention: YYYY-MM-DD-HHMM_T-<id>.ext (T = tenths of a minute)."""
    return f"{t:%Y-%m-%d-%H%M}_{t.second // 6}-{planet}{suffix}"


def to_unit(img):
    return np.clip(np.nan_to_num(img), 0, 1).astype(np.float32)


# ============================================================================= image I/O
# Minimal baseline TIFF (uncompressed, 8/16-bit, grey or RGB) so 16-bit colour works without
# tifffile / OpenCV. AutoStakkert!, RegiStax and WinJUPOS all read and write exactly this flavour.
_TIFF_T = {3: "H", 4: "I"}


def save_tiff(path, img, bits=16):
    a = to_uint(img, bits)
    h, w = a.shape[:2]
    spp = 1 if a.ndim == 2 else a.shape[2]
    data = a.astype(a.dtype.newbyteorder("<")).tobytes()
    extra_off = 8 + len(data)
    extra = struct.pack(f"<{spp}H", *([bits] * spp)) if spp > 2 else b""
    ifd_off = extra_off + len(extra) + (len(extra) & 1)
    tags = [(256, 4, 1, w), (257, 4, 1, h),
            (258, 3, spp, extra_off if spp > 2 else bits), (259, 3, 1, 1),
            (262, 3, 1, 2 if spp >= 3 else 1), (273, 4, 1, 8), (277, 3, 1, spp),
            (278, 4, 1, h), (279, 4, 1, len(data)), (284, 3, 1, 1)]
    ifd = struct.pack("<H", len(tags))
    for tag, typ, cnt, val in tags:
        v = struct.pack("<HH", val, 0) if (typ == 3 and cnt == 1) else struct.pack("<I", val)
        ifd += struct.pack("<HHI", tag, typ, cnt) + v
    ifd += struct.pack("<I", 0)
    with open(path, "wb") as f:
        f.write(b"II*\0" + struct.pack("<I", ifd_off) + data + extra + b"\0" * (len(extra) & 1) + ifd)


def load_tiff(path):
    """Read uncompressed baseline TIFF (8/16-bit, 1/3/4 samples) -> float32 [0,1]; None if unsupported."""
    with open(path, "rb") as f:
        buf = f.read()
    bo = {b"II": "<", b"MM": ">"}.get(buf[:2])
    if bo is None:
        return None
    (off,) = struct.unpack(bo + "I", buf[4:8])
    (n,) = struct.unpack(bo + "H", buf[off:off + 2])
    tags = {}
    for i in range(n):
        tag, typ, cnt = struct.unpack(bo + "HHI", buf[off + 2 + 12 * i: off + 10 + 12 * i])
        if typ not in _TIFF_T:
            continue
        size = cnt * (2 if typ == 3 else 4)
        p = off + 10 + 12 * i
        if size > 4:
            (p,) = struct.unpack(bo + "I", buf[p:p + 4])
        tags[tag] = struct.unpack(bo + _TIFF_T[typ] * cnt, buf[p:p + size])
    w, h = tags[256][0], tags[257][0]
    bits, spp = tags.get(258, (1,))[0], tags.get(277, (1,))[0]
    if tags.get(259, (1,))[0] != 1 or tags.get(284, (1,))[0] != 1 or bits not in (8, 16):
        return None
    dt_ = np.dtype(bo + ("u2" if bits == 16 else "u1"))
    raw = b"".join(buf[o:o + c] for o, c in zip(tags[273], tags[279]))
    a = np.frombuffer(raw, dt_, h * w * spp).reshape((h, w, spp) if spp > 1 else (h, w))
    a = a[..., :3] if spp == 4 else a
    return a.astype(np.float32) / (65535 if bits == 16 else 255)


def load(path):
    """.npy, .ser (mean of all frames), .tif (16-bit RGB ok), FITS / PNG / JPG -> float32."""
    lp = path.lower()
    if lp.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if lp.endswith((".tif", ".tiff")):
        a = load_tiff(path)
        if a is not None:
            return a
    return at.load_image(path)


def save(path, img):
    """.npy keeps float values; .tif -> 16-bit; .png/.jpg -> 8-bit; FITS via astropy."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lp = path.lower()
    if lp.endswith(".npy"):
        return np.save(path, np.asarray(img, np.float32))
    if lp.endswith((".tif", ".tiff")):
        return save_tiff(path, img)
    return at.save_image(path, img)
