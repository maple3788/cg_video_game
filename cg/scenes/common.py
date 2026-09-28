"""Shared scene helpers: parallax layers, camera, fades."""
from __future__ import annotations

import math

import numpy as np
import skia

from ..core import W, H, Layer, clamp, paint, lerp, smooth, inv, snoise


class PL:
    """A parallax layer placed in world space."""

    def __init__(self, layer: Layer, p=1.0, x0=0.0, y0=0.0, py=None, zoomk=None):
        self.layer = layer
        self.p = p
        self.py = p if py is None else py
        self.x0, self.y0 = x0, y0
        self.zoomk = p if zoomk is None else zoomk


def draw_layers(c, layers, camx=0.0, camy=0.0, zoom=1.0, zc=(W / 2, H / 2), alpha=1.0, between=None):
    """Draw parallax layers.  ``between`` maps layer index -> callable(c) drawn after it."""
    for i, L in enumerate(layers):
        z = 1 + (zoom - 1) * L.zoomk
        c.save()
        c.translate(zc[0], zc[1])
        c.scale(z, z)
        c.translate(-zc[0], -zc[1])
        L.layer.draw(c, L.x0 - camx * L.p, L.y0 - camy * L.py, alpha)
        c.restore()
        if between and i in between:
            between[i](c)


class Cam:
    """Camera transform for p=1 dynamic elements."""

    def __init__(self, x=0.0, y=0.0, zoom=1.0, zc=(W / 2, H / 2), rot=0.0):
        self.x, self.y, self.zoom, self.zc, self.rot = x, y, zoom, zc, rot

    def begin(self, c, p=1.0):
        z = 1 + (self.zoom - 1) * p
        c.save()
        c.translate(self.zc[0], self.zc[1])
        if self.rot:
            c.rotate(math.degrees(self.rot))
        c.scale(z, z)
        c.translate(-self.zc[0], -self.zc[1])
        c.translate(-self.x * p, -self.y * p)

    def end(self, c):
        c.restore()


def shake(t, hits, amp=18.0, decay=5.0, freq=38.0):
    """Sum of decaying oscillations after each hit time.  Returns (dx, dy, rot)."""
    dx = dy = rot = 0.0
    for (th, k) in hits:
        if t >= th:
            e = math.exp(-(t - th) * decay) * k
            dx += math.sin((t - th) * freq) * amp * e
            dy += math.cos((t - th) * freq * 1.3 + 1) * amp * 0.7 * e
            rot += math.sin((t - th) * freq * 0.7) * 0.004 * e
    return dx, dy, rot


def fade_black(img, k):
    if k < 0.999:
        img *= max(0.0, k)
    return img


def fade_white(img, k, col=(1, 1, 1)):
    if k > 0.001:
        img[:] = img * (1 - k) + np.array(col, np.float32) * k
    return img


def lookat(tx, ty, zoom=1.0, sx=W / 2, sy=H / 2):
    """Camera so that world point (tx, ty) lands on screen (sx, sy) at the given zoom."""
    return tx - sx, ty - sy, zoom, (sx, sy)


def lerp_cam(a, b, k):
    return (lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k),
            (lerp(a[3][0], b[3][0], k), lerp(a[3][1], b[3][1], k)))


class Walker:
    """Piecewise walk: list of (t_start, t_end, x_start, x_end); position + gait phase."""

    def __init__(self, legs, h=118.0, stride=0.68):
        self.legs = sorted(legs)
        self.h = h
        self.cycle = stride * h

    def x(self, t):
        x = self.legs[0][2]
        for (t0, t1, x0, x1) in self.legs:
            if t < t0:
                return x
            if t <= t1:
                u = (t - t0) / max(1e-6, t1 - t0)
                # accelerate/decelerate over the first/last 12%
                return x0 + (x1 - x0) * _ease_walk(u)
            x = x1
        return x

    def moving(self, t):
        for (t0, t1, x0, x1) in self.legs:
            if t0 <= t <= t1 and x0 != x1:
                u = (t - t0) / max(1e-6, t1 - t0)
                return min(1.0, u / 0.1, (1 - u) / 0.1)
        return 0.0

    def phase(self, t):
        return self.x(t) / self.cycle


def _ease_walk(u, e=0.1):
    # constant speed with smooth ramps; normalised so f(1) = 1
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    v = 1.0 / (1 - e)  # peak speed
    if u < e:
        return v * u * u / (2 * e)
    if u > 1 - e:
        w = 1 - u
        return 1 - v * w * w / (2 * e)
    return v * (u - e / 2)
