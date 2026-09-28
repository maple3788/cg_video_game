"""Procedural silhouettes and set pieces."""
from __future__ import annotations

import math

import numpy as np
import skia

from .core import (W, H, Rng, c4, clamp, fbm, lerp, paint, poly, smooth, smooth_path, snoise,
                   vnoise, ribbon, radial, linear)
from .text import draw_text, font, layout


# ------------------------------------------------------------------ ground

def ground_fn(base_y, amp=40.0, scale=900.0, seed=0, amp2=8.0):
    def g(x):
        return base_y + (fbm(np.asarray(x) / scale, 3, seed) - 0.5) * 2 * amp + snoise(np.asarray(x) / 90.0, seed + 5) * amp2
    return g


def draw_ground(c, x0, x1, gfn, col, a=1.0, bottom=H + 400, step=8.0, blur=0.0):
    xs = np.arange(x0, x1 + step, step)
    ys = gfn(xs)
    pts = list(zip(xs, ys)) + [(x1, bottom), (x0, bottom)]
    c.drawPath(poly(pts), paint(col, a, blur=blur))


def grass(c, x0, x1, gfn, col, a=1.0, density=0.9, hmin=8, hmax=28, seed=1, lean=0.0, blur=0.0, wide=1.0):
    r = Rng(seed)
    path = skia.Path()
    x = x0
    while x < x1:
        x += r.u(0.3, 2.2) / density
        gy = float(gfn(x))
        # clumps
        hh = r.u(hmin, hmax) * (0.55 + 0.9 * vnoise(x / 60.0, seed))
        ln = lean + r.n(0, 0.28)
        bw = r.u(1.6, 4.0) * wide
        tipx = x + math.sin(ln) * hh
        tipy = gy - math.cos(ln) * hh
        midx = x + math.sin(ln * 0.5) * hh * 0.55
        midy = gy - math.cos(ln * 0.5) * hh * 0.55
        path.moveTo(x - bw, gy + 2)
        path.quadTo(midx - bw * 0.4, midy, tipx, tipy)
        path.quadTo(midx + bw * 0.4, midy, x + bw, gy + 2)
        path.close()
    c.drawPath(path, paint(col, a, blur=blur))


# ------------------------------------------------------------------ trees

def trunk(c, x, gy, top_y=-200, w=40.0, lean=0.0, seed=0, col=(0, 0, 0), a=1.0, branches=3,
          blur=0.0, roots=True, twig_scale=1.0, bend=18.0):
    r = Rng(seed)
    left, right = [], []
    n = 28
    for i in range(n + 1):
        u = i / n
        y = gy + 6 - u * (gy + 6 - top_y)
        hfrac = (gy - y)
        xx = x + lean * hfrac + snoise(u * 3.0, seed) * bend + snoise(u * 11.0, seed + 3) * 3
        ww = w * (1 - 0.45 * u) * (1 + 0.1 * snoise(u * 8, seed + 7))
        if roots:
            ww *= 1 + 1.3 * math.exp(-hfrac / (w * 0.9 + 1e-3))
        left.append((xx - ww / 2, y))
        right.append((xx + ww / 2, y))
    pts = left + right[::-1]
    pnt = paint(col, a, blur=blur)
    c.drawPath(smooth_path(pts, close=True, tension=0.6), pnt)
    # branches
    for b in range(branches):
        u = r.u(0.25, 0.85)
        y = gy - u * (gy - top_y)
        hfrac = gy - y
        bx = x + lean * hfrac + snoise(u * 3.0, seed) * bend
        side = r.choice([-1, 1])
        ang = side * r.u(0.45, 1.1)  # from vertical
        ln = r.u(0.8, 2.6) * w * 2.2 * twig_scale
        branch(c, bx, y, ang, ln, w * 0.32 * (1 - u * 0.4), seed * 13 + b, col, a, blur, depth=2)


def branch(c, x, y, ang, ln, w, seed, col, a=1.0, blur=0.0, depth=2):
    r = Rng(seed)
    pts = []
    widths = []
    n = 8
    cx, cy = x, y
    for i in range(n + 1):
        u = i / n
        aa = ang + snoise(u * 2 + seed, seed) * 0.25 + u * 0.15 * math.copysign(1, ang)
        pts.append((cx, cy))
        widths.append(max(0.8, w * (1 - 0.85 * u)))
        cx += math.sin(aa) * ln / n
        cy -= math.cos(aa) * ln / n * (1 - 0.35 * u)
    c.drawPath(ribbon(pts, widths), paint(col, a, blur=blur))
    if depth > 0:
        for k in range(r.i(1, 3)):
            j = r.i(3, n)
            sx, sy = pts[j]
            na = ang + r.choice([-1, 1]) * r.u(0.3, 0.9)
            branch(c, sx, sy, na, ln * r.u(0.3, 0.55), widths[j] * 0.7, seed * 7 + k + 1, col, a, blur, depth - 1)


def bare_tree(c, x, gy, h, seed, col, a=1.0, blur=0.0, spread=1.0):
    """Winter tree: trunk with fractal crown."""
    r = Rng(seed)
    w = h * 0.06
    trunk_top = gy - h * 0.45
    pts_l = [(x - w * 1.3, gy + 4), (x - w * 0.5, gy - h * 0.1), (x - w * 0.42, trunk_top)]
    pts_r = [(x + w * 0.42, trunk_top), (x + w * 0.5, gy - h * 0.1), (x + w * 1.3, gy + 4)]
    c.drawPath(smooth_path(pts_l + pts_r, close=True, tension=0.5), paint(col, a, blur=blur))
    for k in range(5):
        ang = (k - 2) * 0.42 * spread + r.n(0, 0.12)
        branch(c, x, trunk_top + h * 0.03, ang, h * r.u(0.35, 0.55), w * 0.75, seed * 11 + k, col, a, blur, depth=3)


# ------------------------------------------------------------------ mechanical

def gear_path(cx, cy, r, teeth=12, tooth=0.16, ang=0.0, hole=0.35, spokes=5):
    p = skia.Path()
    n = teeth * 4
    pts = []
    for i in range(n):
        k = i % 4
        base = (i // 4) * 2 * math.pi / teeth
        off = [0.0, 0.18, 0.5, 0.68][k] * 2 * math.pi / teeth
        rr = r * (1 + tooth) if k in (1, 2) else r
        a_ = ang + base + off
        pts.append((cx + math.cos(a_) * rr, cy + math.sin(a_) * rr))
    p.addPoly([skia.Point(*q) for q in pts], True)
    # spoke holes
    p.setFillType(skia.PathFillType.kEvenOdd)
    if spokes:
        for s in range(spokes):
            a0 = ang + s * 2 * math.pi / spokes + 0.25
            a1 = a0 + 2 * math.pi / spokes - 0.5
            inner, outer = r * hole, r * 0.78
            q = skia.Path()
            arc = []
            for j in range(9):
                t = a0 + (a1 - a0) * j / 8
                arc.append((cx + math.cos(t) * outer, cy + math.sin(t) * outer))
            for j in range(9):
                t = a1 - (a1 - a0) * j / 8
                arc.append((cx + math.cos(t) * inner, cy + math.sin(t) * inner))
            p.addPoly([skia.Point(*q_) for q_ in arc], True)
    p.addCircle(cx, cy, r * 0.12)
    return p


def gear(c, cx, cy, r, ang=0.0, col=(0, 0, 0), a=1.0, blur=0.0, teeth=12, spokes=5):
    c.drawPath(gear_path(cx, cy, r, teeth, 0.14, ang, 0.3, spokes), paint(col, a, blur=blur))


def hanging_sign(c, x, top_y, rope, w, h, text, col=(0.02, 0.02, 0.02), a=1.0, sway=0.0,
                 hole_alpha=1.0, font_key='cinzel_b', size=None, text_col=None):
    """Rope-hung wooden plank with the text punched out (shows what is behind)."""
    ang = sway
    ex = x + math.sin(ang) * rope
    ey = top_y + math.cos(ang) * rope
    c.drawLine(x, top_y, ex - w * 0.3, ey, paint(col, a, stroke=2.2))
    c.drawLine(x, top_y, ex + w * 0.3, ey, paint(col, a, stroke=2.2))
    c.save()
    c.translate(ex, ey)
    c.rotate(math.degrees(-ang * 0.6))
    rect = skia.Rect.MakeXYWH(-w / 2, 0, w, h)
    c.saveLayer(skia.Rect.MakeXYWH(-w, -h, w * 2, h * 3), None)
    rr = skia.RRect.MakeRectXY(rect, 3, 3)
    c.drawRRect(rr, paint(col, a))
    sz = size or h * 0.62
    if text_col is None:
        draw_text(c, text, 0, h * 0.5 + sz * 0.36, font_key, sz, (1, 1, 1), hole_alpha, tracking=0.08,
                  blend='dstout')
    c.restore()
    if text_col is not None:
        draw_text(c, text, 0, h * 0.5 + sz * 0.36, font_key, sz, text_col, a, tracking=0.08)
    c.restore()


def rope(c, x, y0, y1, col, a=1.0, sway=0.0, w=2.0):
    p = skia.Path()
    p.moveTo(x, y0)
    mx = x + sway * (y1 - y0) * 0.5
    p.quadTo(mx, (y0 + y1) / 2, x + sway * (y1 - y0), y1)
    c.drawPath(p, paint(col, a, stroke=w))


def rock(c, x, y, w, h, seed, col, a=1.0, blur=0.0):
    r = Rng(seed)
    pts = []
    n = 11
    for i in range(n):
        t = math.pi + i / (n - 1) * math.pi
        rr = 1 + r.n(0, 0.08)
        pts.append((x + math.cos(t) * w / 2 * rr, y + math.sin(t) * h * rr))
    pts += [(x + w / 2, y + 10), (x - w / 2, y + 10)]
    c.drawPath(smooth_path(pts, close=True, tension=0.7), paint(col, a, blur=blur))


# ------------------------------------------------------------------ architecture (GRIS)

def arch(c, x, y, w, h, thick, col, a=1.0, blur=0.0, shader=None):
    """Classical arch standing on ground y (bottom), centred on x."""
    p = skia.Path()
    ow, iw = w / 2, w / 2 - thick
    spring = y - h + ow  # where the semicircle starts
    p.moveTo(x - ow, y)
    p.lineTo(x - ow, spring)
    p.arcTo(skia.Rect.MakeLTRB(x - ow, spring - ow, x + ow, spring + ow), 180, 180, False)
    p.lineTo(x + ow, y)
    p.lineTo(x + iw, y)
    p.lineTo(x + iw, spring)
    p.arcTo(skia.Rect.MakeLTRB(x - iw, spring - iw, x + iw, spring + iw), 0, -180, False)
    p.lineTo(x - iw, y)
    p.close()
    pn = paint(col, a, blur=blur)
    if shader is not None:
        pn.setShader(shader)
        pn.setAlphaf(a)
    c.drawPath(p, pn)


def column(c, x, y, w, h, col, a=1.0, blur=0.0, broken=0.0, seed=0):
    p = skia.Path()
    top = y - h
    if broken > 0:
        r = Rng(seed)
        pts = [(x - w / 2, y), (x - w / 2, top + r.u(0, broken)), (x - w * 0.1, top + r.u(0, broken) * 0.3),
               (x + w * 0.2, top + r.u(0, broken)), (x + w / 2, top + r.u(0, broken) * 0.6), (x + w / 2, y)]
        c.drawPath(poly(pts), paint(col, a, blur=blur))
    else:
        c.drawRect(skia.Rect.MakeLTRB(x - w / 2, top, x + w / 2, y), paint(col, a, blur=blur))
        c.drawRect(skia.Rect.MakeLTRB(x - w * 0.65, top - w * 0.25, x + w * 0.65, top), paint(col, a, blur=blur))
    c.drawRect(skia.Rect.MakeLTRB(x - w * 0.65, y - w * 0.2, x + w * 0.65, y), paint(col, a, blur=blur))


def glow_dot(c, x, y, r, col, a=1.0, halo=4.0, core=1.0, halo_a=0.45):
    if a <= 0.003:
        return
    c.drawCircle(x, y, r * halo, paint(col, a * halo_a * 0.5, blur=r * halo * 0.6, blend='plus'))
    c.drawCircle(x, y, r * 1.6, paint(col, a * 0.6, blur=r * 0.9, blend='plus'))
    c.drawCircle(x, y, r * core, paint((1, 1, 1), a * 0.95))


def diamond(c, x, y, s, col, a=1.0, stroke=0.0, blur=0.0, blend=None):
    c.drawPath(poly([(x, y - s), (x + s, y), (x, y + s), (x - s, y)]),
               paint(col, a, stroke=stroke, blur=blur, blend=blend))
