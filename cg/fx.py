"""Atmospheric effects: god rays, fog, particles, colour-reveal masks."""
from __future__ import annotations

import functools
import math

import cv2
import numpy as np
import skia

from .core import (W, H, Layer, Rng, c4, clamp, lerp, noise_tex, paint, poly, rs, smooth,
                   vnoise, snoise, radial, linear)


# ------------------------------------------------------------------ light

def god_rays(c, ox, oy, t, n=7, spread=0.9, length=1500, col=(1, 1, 1), a=0.18, seed=3,
             base_angle=math.pi / 2, width=(40, 160), blur=28.0, blend='plus'):
    r = Rng(seed)
    for i in range(n):
        ang = base_angle + (r.u(-0.5, 0.5)) * spread
        w0 = r.u(*width)
        flick = 0.55 + 0.45 * vnoise(t * 0.35 + i * 3.7, seed + i)
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        L = length * r.u(0.7, 1.1)
        ex, ey = ox + dx * L, oy + dy * L
        w1 = w0 * 2.4
        pts = [(ox + nx * w0 * 0.15, oy + ny * w0 * 0.15), (ox - nx * w0 * 0.15, oy - ny * w0 * 0.15),
               (ex - nx * w1, ey - ny * w1), (ex + nx * w1, ey + ny * w1)]
        sh = linear(ox, oy, ex, ey, [(col, a * flick), (col, a * flick * 0.45), (col, 0.0)], [0, 0.55, 1])
        p = paint(col, 1.0, blur=blur, blend=blend)
        p.setShader(sh)
        c.drawPath(poly(pts), p)


def light_sweep(c, x0, y0, w, h, prog, col=(1, 1, 1), a=0.6, angle=0.35, band=120):
    """Diagonal glint band moving across a rect (for titles)."""
    if prog <= 0 or prog >= 1:
        return
    cx = x0 - band * 2 + (w + band * 4) * prog
    c.save()
    c.clipRect(skia.Rect.MakeXYWH(x0, y0, w, h))
    dx = math.tan(angle) * h
    pts = [(cx - band / 2, y0), (cx + band / 2, y0), (cx + band / 2 - dx, y0 + h), (cx - band / 2 - dx, y0 + h)]
    c.drawPath(poly(pts), paint(col, a * math.sin(prog * math.pi), blur=band * 0.35, blend='plus'))
    c.restore()


# ------------------------------------------------------------------ fog

@functools.lru_cache(maxsize=32)
def fog_layer(w=3200, h=700, cell=260, seed=5, col=(1, 1, 1), density=1.0, falloff='band', gamma=1.4):
    """A wide soft-noise fog Layer (alpha from noise). falloff: band|top|bottom|none."""
    s = rs()
    pw, ph = int(w * s), int(h * s)
    n = noise_tex(pw, ph, cell * s, 5, seed, gain=0.55)
    n = np.clip((n - 0.25) / 0.6, 0, 1) ** gamma
    yy = np.linspace(0, 1, ph, dtype=np.float32)[:, None]
    if falloff == 'band':
        v = np.clip(np.sin(np.clip(yy, 0, 1) * math.pi), 0, 1) ** 1.5
    elif falloff == 'top':
        v = (1 - yy) ** 1.3
    elif falloff == 'bottom':
        v = yy ** 1.3
    else:
        v = np.ones_like(yy)
    xx = np.linspace(0, 1, pw, dtype=np.float32)[None, :]
    edge = np.clip(np.minimum(xx, 1 - xx) / 0.06, 0, 1)
    m = n * v * edge * density
    arr = np.zeros((ph, pw, 4), np.uint8)
    a = np.clip(m, 0, 1)
    for i in range(3):
        arr[:, :, i] = (a * col[i] * 255).astype(np.uint8)
    arr[:, :, 3] = (a * 255).astype(np.uint8)
    return Layer(w, h, arr=arr)


def draw_fog(c, t, y, speed=12.0, alpha=0.5, seed=5, w=3200, h=700, cell=260, col=(1, 1, 1),
             offset=0.0, falloff='band', density=1.0, blend=None):
    L = fog_layer(w, h, cell, seed, col, density, falloff)
    x = -((t * speed + offset) % (w - W - 10)) if w > W + 20 else 0
    L.draw(c, x, y - h / 2, alpha, blend=blend)


# ------------------------------------------------------------------ particles

def motes(c, t, n=60, seed=11, box=(0, 0, W, H), size=(1.0, 3.5), col=(1, 1, 1), a=0.6,
          drift=(6, -10), wobble=18.0, blur_far=True, blend='plus', twinkle=True, depth=None):
    r = Rng(seed)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    for i in range(n):
        px_, py_ = r.u(0, 1), r.u(0, 1)
        z = r.u(0.2, 1.0) if depth is None else depth
        sz = lerp(size[0], size[1], z)
        vx, vy = drift[0] * (0.4 + z), drift[1] * (0.4 + z)
        ph = r.u(0, 6.28)
        x = x0 + ((px_ * bw + vx * t + math.sin(t * 0.5 + ph) * wobble) % bw)
        y = y0 + ((py_ * bh + vy * t + math.cos(t * 0.37 + ph) * wobble * 0.6) % bh)
        tw = 0.55 + 0.45 * math.sin(t * r.u(0.8, 2.2) + ph) if twinkle else 1.0
        al = a * tw * (0.35 + 0.65 * z)
        blur = sz * (0.6 if not blur_far else (0.5 + 1.8 * (1 - z)))
        c.drawCircle(x, y, sz, paint(col, al, blur=blur, blend=blend))


def snow(c, t, n=400, seed=21, wind=60.0, fall=70.0, size=(1.0, 3.2), col=(1, 1, 1), a=0.8,
         box=(-200, -100, W + 200, H + 100), gust=1.0, blur=0.6, streak=0.0):
    r = Rng(seed)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pts_by_size = {}
    for i in range(n):
        z = r.u(0.15, 1.0)
        px_, py_ = r.u(0, 1), r.u(0, 1)
        ph = r.u(0, 6.28)
        g = gust * (1 + 0.5 * math.sin(t * 0.7 + ph * 0.2))
        x = x0 + ((px_ * bw + wind * g * t * (0.4 + z) + math.sin(t * 1.3 + ph) * 12 * z) % bw)
        y = y0 + ((py_ * bh + fall * t * (0.4 + z)) % bh)
        sz = round(lerp(size[0], size[1], z * z), 1)
        key = min(4, int(z * 5))
        pts_by_size.setdefault(key, []).append((x, y, sz))
    for key, pts in pts_by_size.items():
        z = (key + 0.5) / 5
        if streak > 0:
            p = paint(col, a * (0.35 + 0.65 * z), blur=blur * (1.3 - z) + 0.3, stroke=lerp(size[0], size[1], z * z) * 1.2)
            for (x, y, sz) in pts:
                c.drawLine(x, y, x - wind * streak * (0.4 + z), y - fall * streak * (0.4 + z), p)
        else:
            p = paint(col, a * (0.35 + 0.65 * z), blur=blur * (1.3 - z) + (2.5 if key == 4 else 0))
            for (x, y, sz) in pts:
                c.drawCircle(x, y, sz * (1.6 if key == 4 else 1.0), p)


def rain(c, t, n=300, seed=31, angle=0.18, speed=1800, length=40, col=(0.8, 0.85, 0.9), a=0.25):
    r = Rng(seed)
    p = paint(col, a, stroke=1.2)
    dx, dy = math.sin(angle), math.cos(angle)
    for i in range(n):
        z = r.u(0.3, 1)
        x = r.u(-200, W + 200)
        y0 = r.u(0, H + 300)
        y = (y0 + speed * z * t) % (H + 300) - 150
        xx = x + dx * (y + 150)
        p.setAlphaf(a * z)
        p.setStrokeWidth(0.6 + z * 1.2)
        c.drawLine(xx, y, xx - dx * length * z, y - dy * length * z, p)


# ------------------------------------------------------------------ reveal masks (numpy)

class RevealField:
    """Distance field with watercolour-like noisy edge for colour-spread reveals."""

    def __init__(self, cx, cy, seed=3, amp=160.0, cell=150.0, aspect=1.0):
        s = 0.25
        self.lw, self.lh = int(W * s), int(H * s)
        yy, xx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32) / s
        d = np.sqrt(((xx - cx) * aspect) ** 2 + (yy - cy) ** 2)
        n = noise_tex(self.lw, self.lh, cell * s, 4, seed) - 0.5
        n2 = noise_tex(self.lw, self.lh, cell * s * 0.3, 3, seed + 1) - 0.5
        self.d = (d + n * amp * 2 + n2 * amp * 0.6).astype(np.float32)

    def mask(self, radius, edge=40.0):
        m = np.clip((radius - self.d) / edge + 0.5, 0, 1)
        m = m * m * (3 - 2 * m)
        w, h = int(round(W * rs())), int(round(H * rs()))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    def rim(self, radius, width=26.0):
        x = (self.d - radius) / width
        m = np.exp(-x * x)
        w, h = int(round(W * rs())), int(round(H * rs()))
        return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


@functools.lru_cache(maxsize=8)
def reveal_field(cx, cy, seed=3, amp=160.0, cell=150.0):
    return RevealField(cx, cy, seed, amp, cell)


def ink_wipe_mask(prog, seed=9, direction='lr', soft=0.08):
    """0..1 mask revealing along a direction with noisy edge (render res)."""
    w, h = int(round(W * rs())), int(round(H * rs()))
    key = (w, h, seed, direction)
    base = _wipe_base(*key)
    m = np.clip((prog * (1 + 2 * soft) - soft - base) / soft + 0.5, 0, 1)
    return m


@functools.lru_cache(maxsize=8)
def _wipe_base(w, h, seed, direction):
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    g = {'lr': xx, 'rl': 1 - xx, 'tb': yy, 'bt': 1 - yy}[direction]
    n = noise_tex(w, h, 120 * rs(), 4, seed) - 0.5
    return (g * 0.85 + n * 0.3).astype(np.float32)


# ------------------------------------------------------------------ frost

@functools.lru_cache(maxsize=2)
def _frost_fields(w, h, seed=4):
    s = w / W
    n1 = noise_tex(w, h, 26 * s, 4, seed)
    veins = (1 - np.abs(np.sin(n1 * 46.0))) ** 10
    n3 = noise_tex(w, h, 60 * s, 3, seed + 2)
    veins2 = (1 - np.abs(np.sin(n3 * 31.0))) ** 14
    fine = noise_tex(w, h, 5 * s, 2, seed + 1)
    tex = np.clip(veins * 0.7 + veins2 * 0.5 + (fine - 0.5) * 0.5 + 0.35, 0, 1).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx) / w * 1.25, np.minimum(yy, h - 1 - yy) / h)
    n2 = noise_tex(w, h, 90 * s, 4, seed + 5) - 0.5
    field = (d + n2 * 0.16).astype(np.float32)
    return tex, field


def frost(img, amount, col=(0.86, 0.92, 0.98), seed=4):
    """Ice creeping in from the frame edges (amount ~0..0.5 = how far it reaches)."""
    if amount <= 0.001:
        return img
    h, w = img.shape[:2]
    tex, field = _frost_fields(w, h, seed)
    m = np.clip((amount - field) / 0.05, 0, 1)
    m = (m * (0.35 + 0.55 * tex))[:, :, None]
    fc = np.array(col, np.float32) * (0.92 + 0.2 * tex[:, :, None])
    img[:] = img * (1 - m) + fc * m
    return img


# ------------------------------------------------------------------ watercolour

def wash(w, h, stops, seed=0, blotch=0.1, cell=200.0, pos=None, horizontal=False, blooms=6):
    """Opaque watercolour wash Layer: gradient + pigment blotches + a few cauliflower blooms."""
    s = rs()
    pw, ph = int(w * s), int(h * s)
    t = np.linspace(0, 1, pw if horizontal else ph, dtype=np.float32)
    cols = np.array([hexc_(c) for c in stops], np.float32)
    pos = np.linspace(0, 1, len(stops)) if pos is None else np.array(pos, np.float32)
    grad = np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], -1)
    if horizontal:
        img = np.repeat(grad[None, :, :], ph, 0)
    else:
        img = np.repeat(grad[:, None, :], pw, 1)
    n = noise_tex(pw, ph, cell * s, 5, seed, gain=0.55) - 0.5
    n2 = noise_tex(pw, ph, cell * s * 0.35, 3, seed + 1) - 0.5
    img *= (1 + (n * 2 * blotch + n2 * blotch * 0.6))[:, :, None]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    for _ in range(blooms):
        cx, cy = rng.uniform(0, pw), rng.uniform(0, ph)
        r = rng.uniform(80, 260) * s
        edge_n = noise_tex(pw, ph, 30 * s, 2, int(rng.integers(0, 9999))) - 0.5
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r + edge_n * 0.25
        ring = np.exp(-((d - 1) / 0.035) ** 2) * 0.06
        inside = np.clip(1 - d, 0, 1) * 0.04
        img *= (1 - ring + inside)[:, :, None]
    arr = np.zeros((ph, pw, 4), np.uint8)
    arr[:, :, :3] = np.clip(img * 255, 0, 255).astype(np.uint8)
    arr[:, :, 3] = 255
    return Layer(w, h, arr=arr)


def hexc_(c):
    if isinstance(c, str):
        c = c.lstrip('#')
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return c


def watercolorize(arr, seed=0, edge=0.45, gran=0.18, wobble=3.0, soft=0.6):
    """Give a flat premultiplied RGBA layer watercolour edges (dark rims, granulation, wobble)."""
    s = rs()
    h, w = arr.shape[:2]
    f = arr.astype(np.float32) / 255.0
    if wobble > 0:
        nx = (noise_tex(w, h, 40 * s, 3, seed) - 0.5) * 2 * wobble * s
        ny = (noise_tex(w, h, 40 * s, 3, seed + 7) - 0.5) * 2 * wobble * s
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        f = cv2.remap(f, gx + nx, gy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = f[:, :, 3]
    if soft > 0:
        f = cv2.GaussianBlur(f, (0, 0), soft * s + 0.3)
        a = f[:, :, 3]
    inner = np.clip(a - cv2.GaussianBlur(a, (0, 0), 5 * s + 0.5), 0, 1)
    g = noise_tex(w, h, 4 * s + 1, 2, seed + 3) - 0.5
    mul = (1 - edge * inner * 2.2) * (1 + gran * g * 2)
    f[:, :, :3] *= np.clip(mul, 0, 2)[:, :, None]
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def reflect(img, horizon_px, strength=0.45, t=0.0, ripple=4.0, blur_px=2.0, tint=(0.8, 0.9, 1.0), depth=None):
    """Mirror the image above a horizon row into the region below it, with ripples (numpy, in place)."""
    h, w = img.shape[:2]
    hz = int(horizon_px)
    n = min(hz, h - hz) if depth is None else min(int(depth), hz, h - hz)
    if n <= 2:
        return img
    src = img[hz - n:hz][::-1].copy()
    rows = np.arange(n, dtype=np.float32)
    shift = (np.sin(rows * 0.35 / max(rs(), 0.3) + t * 2.2) * ripple * rs() * (0.3 + rows / n)).astype(np.float32)
    gx = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    gy = np.repeat(rows[:, None], w, 1)
    src = cv2.remap(src, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if blur_px > 0:
        src = cv2.GaussianBlur(src, (0, 0), blur_px * rs() + 0.3)
    fade = (strength * (1 - rows / n) ** 1.2)[:, None, None]
    img[hz:hz + n] = img[hz:hz + n] * (1 - fade * 0.5) + src * np.array(tint, np.float32) * fade
    return img
