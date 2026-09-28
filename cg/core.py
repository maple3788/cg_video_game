"""Core primitives: frame buffer, colours, paints, paths, easing and noise.

All drawing happens in a fixed *design space* of 1920x1080.  The actual
render resolution is ``W*RS x H*RS`` (RS = render scale), the skia canvas is
pre-scaled so scene code never has to care about it.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import skia

W, H = 1920, 1080
FPS = 24

_RS = 1.0


def set_render_scale(s: float) -> None:
    global _RS
    _RS = float(s)


def rs() -> float:
    return _RS


def px(n: float) -> float:
    """Design-space length -> render pixels (for numpy/cv2 post effects)."""
    return n * _RS


# --------------------------------------------------------------------------
# math / easing
# --------------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def lerp3(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def inv(a, b, x):
    """Clamped inverse lerp."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return clamp((x - a) / (b - a))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ease_in(x):
    x = clamp(x)
    return x * x * x


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_out_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x, s=1.70158):
    x = clamp(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def env(t, t0, t1, fi=0.5, fo=0.5):
    """Trapezoid envelope: 0 before t0, ramps up over fi, holds, ramps down over fo ending at t1."""
    if t <= t0 or t >= t1:
        return 0.0
    a = smooth((t - t0) / fi) if fi > 0 else 1.0
    b = smooth((t1 - t) / fo) if fo > 0 else 1.0
    return min(a, b)


def seg(t, t0, t1, fn=smooth):
    return fn(inv(t0, t1, t))


def mix_angle(a, b, t):
    return a + (b - a) * t


def rot(x, y, a):
    ca, sa = math.cos(a), math.sin(a)
    return x * ca - y * sa, x * sa + y * ca


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------

def _hash(i, seed=0):
    i = np.asarray(i, dtype=np.int64)
    h = (i * 374761393 + seed * 668265263 + 1442695040888963407) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


def hash1(i, seed=0) -> float:
    return float(_hash(int(i), seed))


def vnoise(x, seed=0):
    """Smooth 1D value noise in [0,1]; accepts scalars or arrays."""
    xa = np.asarray(x, dtype=np.float64)
    i = np.floor(xa)
    f = xa - i
    u = f * f * (3 - 2 * f)
    a = _hash(i.astype(np.int64), seed)
    b = _hash(i.astype(np.int64) + 1, seed)
    r = a + (b - a) * u
    return float(r) if np.ndim(r) == 0 else r


def fbm(x, octaves=4, seed=0, lac=2.0, gain=0.5):
    amp, freq, tot, norm = 1.0, 1.0, 0.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise(np.asarray(x) * freq, seed + o * 17)
        norm += amp
        amp *= gain
        freq *= lac
    r = tot / norm
    return float(r) if np.ndim(r) == 0 else r


def snoise(x, seed=0):
    """Signed value noise in [-1, 1]."""
    return vnoise(x, seed) * 2 - 1


def noise_tex(w, h, cell=64.0, octaves=4, seed=0, gain=0.5) -> np.ndarray:
    """Fractal value-noise texture (float32, ~[0,1]) built from upscaled random grids."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    c = cell
    for _ in range(octaves):
        gw, gh = max(2, int(w / c) + 3), max(2, int(h / c) + 3)
        grid = rng.random((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (int(gw * c), int(gh * c)), interpolation=cv2.INTER_CUBIC)
        ox, oy = int(c), int(c)
        out += amp * up[oy:oy + h, ox:ox + w]
        norm += amp
        amp *= gain
        c /= 2.0
        if c < 1.5:
            break
    out /= norm
    return out


# --------------------------------------------------------------------------
# colours & paints
# --------------------------------------------------------------------------

def hexc(s: str):
    s = s.lstrip('#')
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def c4(col, a=1.0) -> skia.Color4f:
    if isinstance(col, str):
        col = hexc(col)
    if len(col) == 4:
        return skia.Color4f(col[0], col[1], col[2], col[3] * a)
    return skia.Color4f(col[0], col[1], col[2], a)


def gray(v):
    return (v, v, v)


BLEND = {
    None: None,
    'plus': skia.BlendMode.kPlus,
    'screen': skia.BlendMode.kScreen,
    'multiply': skia.BlendMode.kMultiply,
    'overlay': skia.BlendMode.kOverlay,
    'soft': skia.BlendMode.kSoftLight,
    'dstout': skia.BlendMode.kDstOut,
    'dstin': skia.BlendMode.kDstIn,
    'srcatop': skia.BlendMode.kSrcATop,
    'lighten': skia.BlendMode.kLighten,
    'darken': skia.BlendMode.kDarken,
    'color': skia.BlendMode.kColor,
    'src': skia.BlendMode.kSrc,
}


def paint(col=(1, 1, 1), a=1.0, blur=0.0, stroke=0.0, cap='round', join='round',
          blend=None, shader=None, aa=True, outer=False) -> skia.Paint:
    p = skia.Paint(AntiAlias=aa)
    p.setColor4f(c4(col, a))
    if stroke and stroke > 0:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap({'round': skia.Paint.kRound_Cap, 'butt': skia.Paint.kButt_Cap,
                        'square': skia.Paint.kSquare_Cap}[cap])
        p.setStrokeJoin({'round': skia.Paint.kRound_Join, 'miter': skia.Paint.kMiter_Join,
                         'bevel': skia.Paint.kBevel_Join}[join])
    if blur and blur > 0.05:
        style = skia.kOuter_BlurStyle if outer else skia.kNormal_BlurStyle
        p.setMaskFilter(skia.MaskFilter.MakeBlur(style, blur))
    if blend:
        p.setBlendMode(BLEND[blend])
    if shader is not None:
        p.setShader(shader)
    return p


def grad_colors(stops):
    """stops: list of (rgb, alpha) or rgb tuples -> skia colors."""
    out = []
    for s in stops:
        if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], (tuple, str)):
            out.append(c4(s[0], s[1]).toColor())
        else:
            out.append(c4(s).toColor())
    return out


def linear(x0, y0, x1, y1, stops, pos=None):
    return skia.GradientShader.MakeLinear([skia.Point(x0, y0), skia.Point(x1, y1)],
                                          grad_colors(stops), pos)


def radial(cx, cy, r, stops, pos=None):
    return skia.GradientShader.MakeRadial(skia.Point(cx, cy), max(r, 0.01), grad_colors(stops), pos)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def poly(pts: Iterable, close=True) -> skia.Path:
    p = skia.Path()
    first = True
    for x, y in pts:
        if first:
            p.moveTo(float(x), float(y))
            first = False
        else:
            p.lineTo(float(x), float(y))
    if close:
        p.close()
    return p


def smooth_path(pts: Sequence, close=False, tension=1.0) -> skia.Path:
    """Catmull-Rom spline through points, as cubic beziers."""
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    p = skia.Path()
    if n < 2:
        return p
    p.moveTo(*pts[0])
    rng_ = range(n) if close else range(n - 1)
    for i in rng_:
        p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (close or i + 2 < n) else pts[(i + 1) % n]
        k = tension / 6.0
        c1 = (p1[0] + (p2[0] - p0[0]) * k, p1[1] + (p2[1] - p0[1]) * k)
        c2 = (p2[0] - (p3[0] - p1[0]) * k, p2[1] - (p3[1] - p1[1]) * k)
        p.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if close:
        p.close()
    return p


def ribbon(pts: Sequence, widths: Sequence) -> skia.Path:
    """Filled variable-width ribbon along a polyline."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n < 2:
        return skia.Path()
    d = np.gradient(pts, axis=0)
    ln = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = -d[:, 1] / ln, d[:, 0] / ln
    w = np.asarray(widths, np.float64) * 0.5
    left = np.stack([pts[:, 0] + nx * w, pts[:, 1] + ny * w], 1)
    right = np.stack([pts[:, 0] - nx * w, pts[:, 1] - ny * w], 1)
    outline = np.concatenate([left, right[::-1]], 0)
    return smooth_path(outline, close=True, tension=0.8)


# --------------------------------------------------------------------------
# frame & layers
# --------------------------------------------------------------------------

class Frame:
    """A render-resolution RGBA buffer with a skia canvas in design space."""

    def __init__(self):
        s = _RS
        self.w, self.h = int(round(W * s)), int(round(H * s))
        self.buf = np.zeros((self.h, self.w, 4), np.uint8)
        self.surf = skia.Surface(self.buf, colorType=skia.kRGBA_8888_ColorType,
                                 alphaType=skia.kPremul_AlphaType)
        self.c = self.surf.getCanvas()
        self.c.scale(s, s)

    def clear(self, col=(0, 0, 0)):
        self.c.clear(c4(col, 1.0))

    def fill(self, shader):
        p = skia.Paint(AntiAlias=False)
        p.setShader(shader)
        self.c.drawRect(skia.Rect.MakeWH(W, H), p)

    def to_float(self) -> np.ndarray:
        return self.buf[:, :, :3].astype(np.float32) * (1.0 / 255.0)

    def from_float(self, arr: np.ndarray):
        np.clip(arr, 0.0, 1.0, out=arr)
        self.buf[:, :, :3] = (arr * 255.0 + 0.5).astype(np.uint8)
        self.buf[:, :, 3] = 255

    def rgb_bytes(self) -> bytes:
        return np.ascontiguousarray(self.buf[:, :, :3]).tobytes()


def _image_from_rgba(arr: np.ndarray) -> skia.Image:
    return skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType,
                                alphaType=skia.kPremul_AlphaType)


class Layer:
    """Pre-rendered RGBA image sized in design units (optionally blurred)."""

    def __init__(self, w: float, h: float, draw: Callable | None = None, blur: float = 0.0,
                 arr: np.ndarray | None = None, post: Callable | None = None):
        s = _RS
        self.W, self.H = float(w), float(h)
        if arr is None:
            pw, ph = max(1, int(math.ceil(w * s))), max(1, int(math.ceil(h * s)))
            arr = np.zeros((ph, pw, 4), np.uint8)
            surf = skia.Surface(arr, colorType=skia.kRGBA_8888_ColorType,
                                alphaType=skia.kPremul_AlphaType)
            cv = surf.getCanvas()
            cv.scale(s, s)
            if draw is not None:
                draw(cv)
            del cv, surf
        if blur > 0.05:
            arr = cv2.GaussianBlur(arr, (0, 0), blur * s)
        if post is not None:
            arr = post(arr)
        self.arr = arr
        self.img = _image_from_rgba(arr)

    def draw(self, c: skia.Canvas, x: float, y: float, alpha: float = 1.0, blend=None,
             scale: float = 1.0, sx: float | None = None, sy: float | None = None):
        sx = scale if sx is None else sx
        sy = scale if sy is None else sy
        dst = skia.Rect.MakeXYWH(x, y, self.W * sx, self.H * sy)
        p = skia.Paint(AntiAlias=True)
        p.setAlphaf(clamp(alpha))
        if blend:
            p.setBlendMode(BLEND[blend])
        c.drawImageRect(self.img, skia.Rect.MakeWH(self.img.width(), self.img.height()), dst,
                        skia.SamplingOptions(skia.FilterMode.kLinear), p)

    def draw_centered(self, c, cx, cy, alpha=1.0, blend=None, scale=1.0):
        self.draw(c, cx - self.W * scale / 2, cy - self.H * scale / 2, alpha, blend, scale)


class VLayer:
    """Vector layer recorded once into an SkPicture; stays crisp at any zoom."""

    def __init__(self, w: float, h: float, draw: Callable):
        self.W, self.H = float(w), float(h)
        rec = skia.PictureRecorder()
        try:
            cv = rec.beginRecording(skia.Rect.MakeWH(w, h), skia.RTreeFactory())
        except Exception:  # pragma: no cover - older skia-python
            cv = rec.beginRecording(skia.Rect.MakeWH(w, h))
        draw(cv)
        self.pic = rec.finishRecordingAsPicture()

    def draw(self, c: skia.Canvas, x: float, y: float, alpha: float = 1.0, blend=None, scale: float = 1.0):
        c.save()
        c.translate(x, y)
        if scale != 1.0:
            c.scale(scale, scale)
        if alpha < 0.999 or blend:
            p = skia.Paint()
            p.setAlphaf(clamp(alpha))
            if blend:
                p.setBlendMode(BLEND[blend])
            c.drawPicture(self.pic, None, p)
        else:
            c.drawPicture(self.pic)
        c.restore()


def gray_layer_from_mask(mask: np.ndarray, col, alpha=1.0) -> Layer:
    """Build a Layer from a float mask (render-res, 0..1) and a flat colour."""
    col = hexc(col) if isinstance(col, str) else col
    h, w = mask.shape
    a = np.clip(mask * alpha, 0, 1)
    arr = np.zeros((h, w, 4), np.uint8)
    for i in range(3):
        arr[:, :, i] = (a * col[i] * 255 + 0.5).astype(np.uint8)
    arr[:, :, 3] = (a * 255 + 0.5).astype(np.uint8)
    return Layer(w / _RS, h / _RS, arr=arr)


class Rng:
    """Deterministic helper around numpy's Generator."""

    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def u(self, a=0.0, b=1.0):
        return float(self.g.uniform(a, b))

    def n(self, mu=0.0, s=1.0):
        return float(self.g.normal(mu, s))

    def i(self, a, b):
        return int(self.g.integers(a, b))

    def choice(self, seq):
        return seq[int(self.g.integers(0, len(seq)))]
