"""Numpy / OpenCV post-processing: grading, bloom, grain, vignette, paper."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .core import W, H, rs, noise_tex, smooth

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Post:
    _inst: "Post | None" = None
    _inst_scale = None

    @classmethod
    def get(cls) -> "Post":
        if cls._inst is None or cls._inst_scale != rs():
            cls._inst = Post()
            cls._inst_scale = rs()
        return cls._inst

    def __init__(self):
        s = rs()
        self.w, self.h = w, h = int(round(W * s)), int(round(H * s))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - w / 2) / (w / 2)
        ny = (yy - h / 2) / (h / 2)
        self.r = np.sqrt(nx * nx * 0.78 + ny * ny).astype(np.float32)
        self.ny = ny.astype(np.float32)
        self.nx = nx.astype(np.float32)
        rng = np.random.default_rng(1234)
        self.grain_bank = []
        for _ in range(6):
            g = rng.standard_normal((h, w)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.6 * max(s, 0.5))
            g /= g.std() + 1e-6
            self.grain_bank.append(g)
        self.paper = self._build_paper(rng)
        self._vig = {}
        self._dust = self._build_dust(rng)

    # ------------------------------------------------------------------ builders
    def _build_paper(self, rng):
        w, h, s = self.w, self.h, rs()
        fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.9 * s + 0.3)
        fine /= fine.std() + 1e-6
        blotch = noise_tex(w, h, cell=220 * s, octaves=5, seed=99) - 0.5
        blotch2 = noise_tex(w, h, cell=40 * s, octaves=3, seed=98) - 0.5
        # fibres
        fib = np.zeros((h, w), np.float32)
        for _ in range(int(900 * s * s) + 50):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            a = rng.uniform(0, math.pi)
            ln = rng.uniform(6, 30) * s
            cv2.line(fib, (int(x), int(y)), (int(x + math.cos(a) * ln), int(y + math.sin(a) * ln)),
                     float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
        fib = cv2.GaussianBlur(fib, (0, 0), 0.6)
        paper = 1.0 + fine * 0.018 + blotch * 0.16 + blotch2 * 0.06 - fib * 0.05
        return paper.astype(np.float32)

    def _build_dust(self, rng):
        """Sparse film dust / hair specks mask bank (several frames)."""
        w, h, s = self.w, self.h, rs()
        bank = []
        for k in range(8):
            m = np.zeros((h, w), np.float32)
            for _ in range(rng.integers(2, 7)):
                x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
                r = max(1, int(rng.uniform(0.8, 2.6) * s))
                cv2.circle(m, (x, y), r, float(rng.uniform(0.4, 1)), -1, cv2.LINE_AA)
            if rng.random() < 0.6:  # a hair
                x, y = rng.uniform(0, w), rng.uniform(0, h)
                pts = []
                a = rng.uniform(0, 6.28)
                for i in range(12):
                    a += rng.normal(0, 0.35)
                    x += math.cos(a) * 5 * s
                    y += math.sin(a) * 5 * s
                    pts.append((int(x), int(y)))
                cv2.polylines(m, [np.array(pts, np.int32)], False, 0.8, max(1, int(s)), cv2.LINE_AA)
            if rng.random() < 0.5:  # scratch
                x = int(rng.uniform(0, w))
                cv2.line(m, (x, 0), (x + int(rng.uniform(-20, 20) * s), h), 0.35, 1, cv2.LINE_AA)
            bank.append(m)
        return bank

    # ------------------------------------------------------------------ masks
    def vignette_mask(self, strength=0.5, inner=0.35, outer=1.25):
        key = (round(strength, 3), round(inner, 3), round(outer, 3))
        v = self._vig.get(key)
        if v is None:
            t = np.clip((self.r - inner) / (outer - inner), 0, 1)
            t = t * t * (3 - 2 * t)
            v = (1.0 - strength * t).astype(np.float32)
            if len(self._vig) > 32:
                self._vig = {k: v for k, v in self._vig.items() if k[0] == 'rad'}
            self._vig[key] = v
        return v


# ---------------------------------------------------------------------- ops

def radial_mask(cx, cy, r, power=2.0):
    """Soft radial falloff (1 at centre -> 0 at r) in design coordinates, render resolution."""
    P = Post.get()
    s = rs()
    key = ('rad', P.w, P.h)
    grid = P._vig.get(key)
    if grid is None:
        yy, xx = np.mgrid[0:P.h, 0:P.w].astype(np.float32)
        grid = (xx / s, yy / s)
        P._vig[key] = grid
    d2 = ((grid[0] - cx) ** 2 + (grid[1] - cy) ** 2) / (r * r)
    return np.exp(-d2 * power).astype(np.float32)


def luma(img):
    if img.ndim != 3:
        return img
    out = cv2.transform(img, LUMA.reshape(1, 3))
    return out[:, :, 0] if out.ndim == 3 else out


def color_matrix(img, sat=1.0, mul=(1, 1, 1), add=(0, 0, 0)):
    """Saturation + per-channel gain in one cv2.transform call."""
    l = LUMA
    m = np.empty((3, 3), np.float32)
    for r in range(3):
        for c in range(3):
            m[r, c] = (1 - sat) * l[c] + (sat if r == c else 0.0)
        m[r] *= mul[r]
    out = cv2.transform(img, m)
    if any(add):
        out += np.array(add, np.float32)
    return out


def contrast(img, k=1.0, pivot=0.5):
    if k != 1.0:
        img -= pivot
        img *= k
        img += pivot
    return img


def curve_s(img, amount=0.3):
    """Gentle filmic S-curve (applied in-place, expects ~0..1)."""
    if amount <= 0:
        return img
    x = np.clip(img, 0, 1)
    s = x * x * (3 - 2 * x)
    img[:] = x + (s - x) * amount
    return img


def lift_gamma_gain(img, lift=0.0, gamma=1.0, gain=1.0):
    np.clip(img, 0, None, out=img)
    if gamma != 1.0:
        cv2.pow(img, 1.0 / gamma, img)
    if gain != 1.0:
        img *= gain
    if lift != 0.0:
        img *= (1 - lift)
        img += lift
    return img


def vignette(img, strength=0.5, inner=0.35, outer=1.25):
    if strength <= 0:
        return img
    v = Post.get().vignette_mask(strength, inner, outer)
    img *= v[:, :, None]
    return img


def grain(img, amount=0.04, frame=0, mid_weight=True):
    if amount <= 0:
        return img
    P = Post.get()
    g = P.grain_bank[frame % len(P.grain_bank)]
    # random offset so the bank does not visibly repeat
    rng = np.random.default_rng(frame * 7919 + 13)
    dy, dx = int(rng.integers(0, P.h)), int(rng.integers(0, P.w))
    g = np.roll(g, (dy, dx), axis=(0, 1))
    if mid_weight:
        l = luma(img)
        wgt = 0.45 + 2.2 * np.clip(l, 0, 1) * (1 - np.clip(l, 0, 1))
        img += (g * wgt * amount)[:, :, None]
    else:
        img += (g * amount)[:, :, None]
    return img


def dust(img, amount=0.5, frame=0, dark=True):
    """Film dust & hair.  Shown on a random subset of frames."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(frame * 31 + 7)
    if rng.random() > 0.55:
        return img
    P = Post.get()
    m = P._dust[int(rng.integers(0, len(P._dust)))]
    if dark:
        img *= (1 - m * amount)[:, :, None]
    else:
        img += (m * amount)[:, :, None]
    return img


def bloom(img, thresh=0.72, strength=0.6, sigma=22.0, tint=(1, 1, 1)):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    sw, sh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    hi = np.maximum(small - thresh, 0) * (1.0 / max(1e-3, 1 - thresh))
    s = rs()
    b1 = cv2.GaussianBlur(hi, (0, 0), max(0.6, sigma * s / 4 * 0.35))
    b2 = cv2.GaussianBlur(hi, (0, 0), max(1.0, sigma * s / 4))
    b3 = cv2.GaussianBlur(hi, (0, 0), max(2.0, sigma * s / 4 * 2.6))
    b = b1 * 0.35 + b2 * 0.4 + b3 * 0.25
    up = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    if tint != (1, 1, 1):
        up *= np.array(tint, np.float32)
    img += up * strength
    return img


def diffuse(img, amount=0.25, sigma=8.0, mode='screen'):
    """Dreamy halation / softness."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // 3), max(8, h // 3)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), max(0.8, sigma * rs() / 3))
    up = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == 'screen':
        img[:] = 1 - (1 - img) * (1 - up * amount)
    else:
        img[:] = img * (1 - amount) + up * amount
    return img


def paper(img, amount=0.5):
    if amount <= 0:
        return img
    p = Post.get().paper
    img *= (1 + (p - 1) * amount)[:, :, None]
    return img


def chroma(img, amount_px=2.0):
    if amount_px <= 0.05:
        return img
    h, w = img.shape[:2]
    k = amount_px * rs() / (w / 2)
    for ch, sc in ((0, 1 + k), (2, 1 - k)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, sc)
        img[:, :, ch] = cv2.warpAffine(np.ascontiguousarray(img[:, :, ch]), M, (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img


def blur(img, sigma):
    if sigma <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma * rs())


def shake_warp(img, dx, dy, ang=0.0, zoom=1.0):
    if abs(dx) < 0.01 and abs(dy) < 0.01 and abs(ang) < 1e-4 and zoom == 1.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(ang), zoom)
    M[0, 2] += dx * rs()
    M[1, 2] += dy * rs()
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def vgrad(img, top=(0, 0, 0), bottom=(0, 0, 0), amount=0.0, mode='mul'):
    if amount <= 0:
        return img
    P = Post.get()
    t = ((P.ny + 1) * 0.5)[:, :, None]
    col = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    if mode == 'mul':
        img *= 1 + (col - 1) * amount
    else:
        img += col * amount
    return img


# ---------------------------------------------------------------------- looks

def look_limbo(img, frame, grain_amt=0.055, vig=0.62, haze=0.22, tone=(1.0, 0.985, 0.95),
               sat=0.0, contrast_k=1.12, flicker=0.025, dust_amt=0.55, lift=0.015):
    rng = np.random.default_rng(frame * 101 + 3)
    img = color_matrix(img, sat=sat, mul=tone)
    img = diffuse(img, haze, 10.0)
    img = contrast(img, contrast_k, 0.45)
    img *= 1 + (rng.random() - 0.5) * 2 * flicker
    img = vignette(img, vig, 0.2, 1.3)
    img = dust(img, dust_amt, frame)
    img = lift_gamma_gain(img, lift=lift)
    img = grain(img, grain_amt, frame)
    return img


def look_gris(img, frame, bloom_amt=0.45, paper_amt=0.7, vig=0.28, grain_amt=0.018, sat=1.05,
              haze=0.12, chroma_px=0.0, thresh=0.7):
    img = bloom(img, thresh, bloom_amt, 26.0)
    if sat != 1.0:
        img = color_matrix(img, sat=sat)
    img = diffuse(img, haze, 12.0)
    img = paper(img, paper_amt)
    img = vignette(img, vig, 0.3, 1.35)
    img = chroma(img, chroma_px)
    img = grain(img, grain_amt, frame, mid_weight=False)
    return img
