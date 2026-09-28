"""Chapter II — The Long Winter (1969–1986): AI winter, then backpropagation melts the ice.

The first colour of the film appears here: the red error signal (and the hero's scarf).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, VLayer, Rng, clamp, ease_in_out, ease_out, env, inv, lerp, paint,
                    poly, radial, linear, smooth, seg, snoise, fbm, smooth_path)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width
from .common import PL, Cam, Walker, draw_layers, fade_black, fade_white, lookat, lerp_cam

HH = 118.0
GY = 860.0
TOPW = -700
BOTW = 1300
RED = (1.0, 0.22, 0.16)
EMBER = (1.0, 0.55, 0.25)
ICE = (0.86, 0.9, 0.94)
DARK = (0.16, 0.17, 0.19)

NODES = {'I1': (2520, 600), 'I2': (2520, 760), 'H1': (2770, 540), 'H2': (2770, 680), 'H3': (2770, 820),
         'O': (3025, 752)}
EDGES_OH = [('H1', 'O'), ('H2', 'O'), ('H3', 'O')]
EDGES_HI = [(h, i) for h in ('H1', 'H2', 'H3') for i in ('I1', 'I2')]
HERO_B = 3088.0
GATE_X = 3400.0
T_TOUCH = 15.9
T_OH = (16.0, 17.0)
T_HI = (17.0, 18.1)
T_FWD = (20.2, 21.8)


def snow_ground(y0, amp, scale, seed):
    def g(x):
        x = np.asarray(x, dtype=np.float64)
        r = y0 + (fbm(x / scale, 3, seed) - 0.5) * 2 * amp
        return float(r) if np.ndim(r) == 0 else r
    return g


GROUND = snow_ground(GY, 18, 900, 3)


def draw_snowfield(c, x0, x1, g, top_col, bot_col, bottom=BOTW):
    xs = np.arange(x0, x1 + 8, 8.0)
    ys = g(xs)
    pts = list(zip(xs, ys)) + [(x1, bottom), (x0, bottom)]
    ymin = float(np.min(ys))
    c.drawPath(poly(pts), paint((1, 1, 1), shader=linear(0, ymin, 0, ymin + 260, [top_col, bot_col])))


def buried_machine(c, x, gy, kind, s, col, seed):
    r = Rng(seed)
    if kind == 'gear':
        props.gear(c, x, gy + s * 0.25, s, r.u(0, 6), col, teeth=16, spokes=5)
    elif kind == 'terminal':
        c.save()
        c.translate(x, gy)
        c.rotate(r.u(-18, 18))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-s * 0.6, -s * 1.1, s * 1.2, s * 1.3), 12, 12),
                    paint(col))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-s * 0.45, -s * 0.95, s * 0.9, s * 0.6), 10, 10),
                    paint((0.08, 0.08, 0.09)))
        c.restore()
    elif kind == 'arm':
        pts = [(x, gy), (x + s * 0.3, gy - s * 0.9), (x + s * 1.0, gy - s * 1.2), (x + s * 1.3, gy - s * 0.8)]
        for a, b in zip(pts, pts[1:]):
            c.drawLine(a[0], a[1], b[0], b[1], paint(col, stroke=s * 0.12))
        for p_ in pts[1:]:
            c.drawCircle(p_[0], p_[1], s * 0.1, paint(col))
        c.drawLine(pts[-1][0], pts[-1][1], pts[-1][0] + s * 0.2, pts[-1][1] + s * 0.25, paint(col, stroke=s * 0.06))
        c.drawLine(pts[-1][0], pts[-1][1], pts[-1][0] - s * 0.05, pts[-1][1] + s * 0.3, paint(col, stroke=s * 0.06))
    elif kind == 'sign':
        c.save()
        c.translate(x, gy)
        c.rotate(r.u(-14, -6))
        c.drawLine(-s * 0.4, 0, -s * 0.4, -s * 0.9, paint(col, stroke=8))
        c.drawLine(s * 0.4, 0, s * 0.4, -s * 0.9, paint(col, stroke=8))
        rect = skia.Rect.MakeXYWH(-s * 0.75, -s * 1.25, s * 1.5, s * 0.42)
        c.saveLayer(skia.Rect.MakeXYWH(-s, -s * 1.6, s * 2, s * 1.2), None)
        c.drawRect(rect, paint(col))
        draw_text(c, 'EXPERT SYSTEM', 0, -s * 1.25 + s * 0.29, 'cinzel_b', s * 0.17, (1, 1, 1), 1.0,
                  tracking=0.08, blend='dstout')
        c.restore()
        c.restore()


@functools.lru_cache(maxsize=1)
def winter_layers():
    width = 5200
    Ht = BOTW - TOPW

    def mk(p, draw, blur, vector=False):
        x0 = -300 * p - 250

        def full(cv):
            cv.translate(-x0, -TOPW)
            draw(cv)
        lay = VLayer(width, Ht, full) if vector else Layer(width, Ht, full, blur=blur)
        return PL(lay, p=p, x0=x0, y0=TOPW)

    xa, xb = -600, 5000

    def far(c):
        for k, (y0, amp, sc, colv) in enumerate([(560, 120, 700, 0.8), (640, 90, 500, 0.72)]):
            g = snow_ground(y0, amp, sc, 30 + k)
            draw_snowfield(c, xa, xb, g, (colv, colv, colv + 0.01), (colv - 0.04,) * 3)

    def hill(c):
        g = snow_ground(700, 40, 600, 41)
        r = Rng(42)
        for i in range(26):
            x = r.u(xa, xb)
            props.bare_tree(c, x, float(g(x)) + 6, r.u(120, 220), r.i(0, 999), (0.55, 0.56, 0.58), spread=0.8)
        draw_snowfield(c, xa, xb, g, (0.8, 0.81, 0.83), (0.74, 0.75, 0.77))

    def mid(c):
        g = snow_ground(780, 30, 600, 51)
        r = Rng(52)
        x = xa
        while x < xb:
            x += r.u(220, 420)
            props.bare_tree(c, x, float(g(x)) + 8, r.u(260, 420), r.i(0, 999), (0.3, 0.31, 0.33))
        for (x, kind, s) in [(300, 'gear', 170), (760, 'sign', 170), (1150, 'terminal', 120), (1560, 'arm', 150),
                             (3150, 'gear', 220), (4300, 'terminal', 140)]:
            buried_machine(c, x, float(g(x)) + 10, kind, s, (0.28, 0.29, 0.31), int(x))
        draw_snowfield(c, xa, xb, g, (0.86, 0.87, 0.89), (0.8, 0.81, 0.83))

    def near(c):
        r = Rng(61)
        for i in range(60):
            x = r.u(xa, xb)
            if 2300 < x < 3600:
                continue
            gy = GROUND(x)
            props.branch(c, x, gy + 4, r.n(0, 0.5), r.u(20, 60), r.u(2, 4), r.i(0, 999), DARK, depth=1)
        for i in range(10):
            x = r.u(xa, xb)
            if 2300 < x < 3600:
                continue
            props.rock(c, x, GROUND(x) + 6, r.u(40, 110), r.u(14, 30), r.i(0, 99), DARK)
        draw_snowfield(c, xa, xb, GROUND, (0.93, 0.94, 0.95), (0.78, 0.79, 0.82))

    return [mk(0.08, far, 6), mk(0.3, hill, 2.5), mk(0.6, mid, 1.0), mk(1.0, near, 0, vector=True)]


@functools.lru_cache(maxsize=1)
def fg_layer():
    def draw(c):
        g = snow_ground(1010, 50, 300, 71)
        draw_snowfield(c, -800, 5200, g, (0.97, 0.97, 0.98), (0.9, 0.9, 0.92), bottom=BOTW)
        r = Rng(72)
        for i in range(14):
            x = r.u(-800, 5200)
            props.branch(c, x, float(g(x)) + 10, r.n(0, 0.4), r.u(120, 260), r.u(6, 12), r.i(0, 99),
                         (0.1, 0.1, 0.11), depth=2)
    lay = Layer(6000, BOTW - TOPW, lambda c: (c.translate(800, -TOPW), draw(c)), blur=9)
    return PL(lay, p=1.45, x0=-800 - 300, y0=TOPW)


def lantern_keepers(c, t):
    """Three tiny figures holding warm lamps on the distant hill (hill plane)."""
    g = snow_ground(700, 40, 600, 41)
    for k, x in enumerate((1180, 1230, 1290)):
        gy = float(g(x))
        s = 34
        c.drawCircle(x, gy - s * 0.86, s * 0.12, paint((0.4, 0.41, 0.43)))
        c.drawPath(poly([(x - s * 0.1, gy - s * 0.74), (x + s * 0.1, gy - s * 0.74), (x + s * 0.16, gy),
                         (x - s * 0.16, gy)]), paint((0.4, 0.41, 0.43)))
        lx, ly = x + s * 0.2, gy - s * 0.5
        c.drawLine(x + s * 0.05, gy - s * 0.65, lx, ly - 4, paint((0.4, 0.41, 0.43), stroke=1.5))
        fl = 0.8 + 0.2 * snoise(t * 5 + k * 3, k)
        props.glow_dot(c, lx, ly, 3.2, (1.0, 0.72, 0.38), fl, halo=9, halo_a=0.8)


# ------------------------------------------------------------------ network

def spark(c, x0, y0, x1, y1, u, col=RED, r=7.0, trail=0.35, label=None):
    if u <= 0 or u >= 1:
        return
    x, y = lerp(x0, x1, u), lerp(y0, y1, u)
    tu = max(0.0, u - trail)
    tx, ty = lerp(x0, x1, tu), lerp(y0, y1, tu)
    sh = linear(tx, ty, x, y, [(col, 0.0), (col, 0.9)])
    p = paint(col, 1.0, stroke=r * 0.9, blur=1.5)
    p.setShader(sh)
    c.drawLine(tx, ty, x, y, p)
    props.glow_dot(c, x, y, r, col, 1.0, halo=5.5, halo_a=0.9)
    if label:
        draw_text(c, label, x + 12, y - 16, 'garamond_m', 30, (1, 0.8, 0.75), 0.95)


def ice_edge(c, a, b, frozen, warm, t, seed, age=0.0):
    (x0, y0), (x1, y1) = a, b
    # ice rod
    c.drawLine(x0, y0, x1, y1, paint(ICE, 0.35 + 0.35 * frozen, stroke=10))
    c.drawLine(x0, y0, x1, y1, paint((1, 1, 1), 0.6 * frozen, stroke=2.2))
    if frozen > 0.02:
        r = Rng(seed)
        L = math.hypot(x1 - x0, y1 - y0)
        nx, ny = -(y1 - y0) / L, (x1 - x0) / L
        for k in range(9):
            u = r.u(0.08, 0.92)
            cx, cy = lerp(x0, x1, u), lerp(y0, y1, u)
            s = r.u(6, 16) * frozen
            side = r.choice([-1, 1])
            tip = (cx + nx * s * side + r.n(0, 3), cy + ny * s * side + r.n(0, 3))
            c.drawPath(poly([(cx - (x1 - x0) / L * 5, cy - (y1 - y0) / L * 5), tip,
                             (cx + (x1 - x0) / L * 5, cy + (y1 - y0) / L * 5)]), paint(ICE, 0.7 * frozen))
    if warm > 0:
        col = (lerp(RED[0], 1.0, age), lerp(RED[1], 0.78, age), lerp(RED[2], 0.52, age))
        c.drawLine(x0, y0, x1, y1, paint(col, 0.9 * warm, stroke=4))
        c.drawLine(x0, y0, x1, y1, paint(col, 0.45 * warm, stroke=16, blur=9, blend='plus'))


def ice_node(c, x, y, frozen, warm, t, r=24):
    hexp = poly([(x + math.cos(a) * r * 1.15, y + math.sin(a) * r * 1.15)
                 for a in [k * math.pi / 3 + math.pi / 6 for k in range(6)]])
    c.drawPath(hexp, paint(ICE, 0.55 + 0.4 * frozen))
    c.drawPath(hexp, paint((1, 1, 1), 0.9, stroke=2))
    c.drawCircle(x, y, r * 0.55, paint((0.98, 0.99, 1.0), 0.7))
    if warm > 0:
        pulse = 0.8 + 0.2 * math.sin(t * 5 + x * 0.01)
        c.drawCircle(x, y, r * 3.2, paint(EMBER, 0.35 * warm * pulse, blur=r * 1.5, blend='plus'))
        c.drawCircle(x, y, r * 0.8, paint((1, 0.85, 0.7), warm))


def falling_shards(c, t, t0, x0, y0, x1, y1, seed, n=10):
    if t < t0:
        return
    r = Rng(seed)
    dt = t - t0
    for k in range(n):
        u = r.u(0, 1)
        x, y = lerp(x0, x1, u), lerp(y0, y1, u)
        vx, vy = r.n(0, 40), r.u(-120, -20)
        px_ = x + vx * dt
        py_ = y + vy * dt + 0.5 * 900 * dt * dt
        if py_ > GROUND(px_) + 10:
            continue
        s = r.u(3, 8)
        ang = r.u(0, 6) + dt * r.n(0, 6)
        pts = [(px_ + math.cos(ang + j * 2.1) * s, py_ + math.sin(ang + j * 2.1) * s) for j in range(3)]
        c.drawPath(poly(pts), paint(ICE, 0.9))


def gate(c, x, gy, t, frost_k, solved, open_k):
    col = DARK
    gw, gh = 330, 500
    left, top = x - gw / 2, gy - gh
    cx, cy = x, gy - 250
    # light behind the opening door
    if open_k > 0:
        c.drawCircle(cx, cy, 900, paint((1, 1, 1), 1.0, shader=radial(cx, cy, 900, [
            ((1, 0.97, 0.9), 0.9 * open_k), ((1, 0.95, 0.85), 0.25 * open_k), ((1, 1, 1), 0.0)], [0, 0.35, 1]),
            blend='plus'))
        fx.god_rays(c, cx, cy, t, n=10, spread=2.8, length=1400, col=(1, 0.96, 0.88), a=0.35 * open_k,
                    base_angle=math.pi, width=(30, 90), blur=20, seed=9)
    split = open_k * 150

    def half(sign):
        c.save()
        c.translate(sign * split, 0)
        body = skia.Path()
        if sign < 0:
            body.moveTo(left, gy + 10)
            body.lineTo(left, top + gw / 2)
            body.arcTo(skia.Rect.MakeXYWH(left, top, gw, gw), 180, 90, False)
            body.lineTo(cx, gy + 10)
        else:
            body.moveTo(cx, gy + 10)
            body.lineTo(cx, top)
            body.arcTo(skia.Rect.MakeXYWH(left, top, gw, gw), 270, 90, False)
            body.lineTo(left + gw, gy + 10)
        body.close()
        c.drawPath(body, paint(col))
        c.restore()
    half(-1)
    half(1)
    if open_k < 0.05:
        panel = skia.Rect.MakeXYWH(cx - 125, cy - 125, 250, 250)
        c.drawRect(panel, paint((0.05, 0.05, 0.06)))
        c.drawRect(panel, paint((0.4, 0.4, 0.42), stroke=1.5))
        draw_text(c, 'X O R', x, top + 92, 'cinzel_b', 34, (0.55, 0.56, 0.58), 0.9, tracking=0.2)
        if solved > 0:
            c.save()
            c.clipRect(panel)
            # curved band containing the two class-1 points (top-left & bottom-right)
            upper = [(cx - 170 + k * 34, cy - 170 + k * 34 - 58 + 14 * math.sin(k * 0.8)) for k in range(11)]
            lower = [(cx - 170 + k * 34, cy - 170 + k * 34 + 58 + 14 * math.sin(k * 0.8 + 1)) for k in range(11)]
            region = poly(upper + lower[::-1])
            c.drawPath(region, paint(EMBER, 0.28 * solved))
            c.drawPath(smooth_path(upper), paint((1, 0.9, 0.75), 0.95 * solved, stroke=3))
            c.drawPath(smooth_path(lower), paint((1, 0.9, 0.75), 0.95 * solved, stroke=3))
            c.drawPath(smooth_path(upper), paint(EMBER, 0.5 * solved, stroke=12, blur=8, blend='plus'))
            c.drawPath(smooth_path(lower), paint(EMBER, 0.5 * solved, stroke=12, blur=8, blend='plus'))
            c.restore()
        for (gx, gyy), v in (((-60, -60), 1), ((60, -60), 0), ((-60, 60), 0), ((60, 60), 1)):
            px_, py_ = cx + gx, cy + gyy
            if v:
                props.glow_dot(c, px_, py_, 12, (1, 1, 1), 0.95, halo=3)
            else:
                c.drawCircle(px_, py_, 13, paint((0.75, 0.75, 0.78), 0.9, stroke=3))
        if solved > 0.9:
            draw_text(c, '✓', cx + 100, cy + 118, 'dejavu', 30, (1, 0.9, 0.7), solved)
    # ice coat
    if frost_k > 0:
        c.save()
        c.translate(0, 0)
        r = Rng(88)
        for k in range(26):
            u = r.u(0, 1)
            ix = left + u * gw
            L = r.u(20, 70) * frost_k
            c.drawPath(poly([(ix - 7, top + gw / 2 * (1 - math.sin(u * math.pi)) + 6),
                             (ix + 7, top + gw / 2 * (1 - math.sin(u * math.pi)) + 6),
                             (ix, top + gw / 2 * (1 - math.sin(u * math.pi)) + 6 + L)]), paint(ICE, 0.85 * frost_k))
        c.drawRect(skia.Rect.MakeXYWH(left, top + gw / 2, gw, gh - gw / 2), paint(ICE, 0.18 * frost_k))
        c.restore()


class Winter:
    DUR = 30.0
    NAME = 'winter'

    def __init__(self):
        self.walk = Walker([(0.0, 13.0, 380.0, 1100.0)], h=HH)
        self.walk_b = Walker([(26.2, 30.0, HERO_B, 3290.0)], h=HH)

    def camera(self, t):
        if t < 13.0:
            hx = self.walk.x(t)
            return lookat(hx + 330, 600, 1.0)
        a = lookat(2840, 640, 1.12)
        b = lookat(3190, 640, 1.32)
        k = ease_in_out(inv(24.5, 30.0, t))
        return lerp_cam(a, b, k)

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        camx, camy, zoom, zc = self.camera(t)
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=linear(0, 0, 0, H, [
            (0.56, 0.58, 0.61), (0.82, 0.83, 0.85), (0.9, 0.9, 0.91)], [0, 0.6, 1])))
        L = winter_layers()

        def plane(fn, p):
            def f(cv):
                cam = Cam(camx, camy, zoom, zc)
                cam.begin(cv, p)
                fn(cv)
                cam.end(cv)
            return f

        def fog(y, a, seed, sp):
            return lambda cv: fx.draw_fog(cv, t, y - camy * 0.4, speed=sp, alpha=a, seed=seed, w=3600, h=520,
                                          cell=220, col=(0.95, 0.96, 0.97))
        between = {0: fog(600, 0.5, 21, 30), 1: lambda cv: (plane(lambda c2: lantern_keepers(c2, t), 0.3)(cv),
                                                           fog(700, 0.45, 22, 45)(cv)),
                   2: fog(790, 0.35, 23, 60)}
        draw_layers(c, L[:3], camx, camy, zoom, zc, between=between)
        draw_layers(c, L[3:], camx, camy, zoom, zc)

        cam = Cam(camx, camy, zoom, zc)
        cam.begin(c)
        scarf_red = smooth(inv(T_TOUCH, T_TOUCH + 0.6, t)) if t >= 13 else 0.0
        scol = (lerp(0.1, RED[0], scarf_red), lerp(0.1, RED[1], scarf_red), lerp(0.11, RED[2], scarf_red))
        if t < 13.0:
            hx = self.walk.x(t)
            mv = self.walk.moving(t)
            pose = ch.walk(self.walk.phase(t), stride=0.28)
            pose.torso += 0.16
            pose.head -= 0.05
            pose = ch.blend(ch.idle(t), pose, mv)
            ch.Hero(HH, col=(0.07, 0.07, 0.08)).draw(
                c, hx, GROUND(hx) + 4, pose, t, eyes=0.9,
                scarf=dict(length=1.0, width=0.06, cols=[scol], wind=1.0, speed=0.4, seed=3.0, n=16))
        else:
            self.draw_network(c, t, scol, scarf_red)
        cam.end(c)
        draw_layers(c, [fg_layer()], camx, camy, zoom, zc)
        fx.snow(c, t, n=520, seed=11, wind=-260, fall=120, a=0.9, size=(1.0, 3.4), gust=1.3)
        fx.snow(c, t, n=40, seed=12, wind=-420, fall=160, a=0.6, size=(5, 11), blur=4)
        img = fr.to_float()
        img = post.look_limbo(img, idx, sat=1.0, tone=(0.95, 0.98, 1.03), grain_amt=0.045, vig=0.5, haze=0.25,
                              contrast_k=1.05, dust_amt=0.4)
        img = fx.frost(img, 0.26 * (1 - seg(t, 0.0, 2.4)))
        img = fade_white(img, (1 - seg(t, 0.0, 1.2)) * 0.9, (0.9, 0.93, 0.97))
        img = fade_black(img, 1 - 0.9 * env(t, 12.7, 13.3, 0.3, 0.3))
        dk = seg(t, 23.8, 26.0) * (1 - seg(t, 27.5, 29.0))
        if dk > 0:
            gsx = (GATE_X - camx - zc[0]) * zoom + zc[0]
            gsy = (GROUND(GATE_X) - 250 - camy - zc[1]) * zoom + zc[1]
            m = post.radial_mask(gsx, gsy, 520)
            img *= (1 - 0.55 * dk * (1 - m))[:, :, None]
            img += (m * 0.35 * dk)[:, :, None] * np.array([1.0, 0.95, 0.85], np.float32)
        img = fade_white(img, seg(t, 27.8, 29.3) ** 1.5, (1.0, 0.98, 0.94))
        img = fade_black(img, 1 - seg(t, 29.3, 30.0))
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        ui.chapter_card(c, t, 0.6, 5.4, '第 二 章', '漫长寒冬', 'CHAPTER II · THE LONG WINTER', '1969 — 1986',
                        col=ui.INK)
        ui.subtitle(c, t, 5.6, 9.2, '资金冻结，信念熄灭。', 'Funding froze. Faith went dark.')
        ui.subtitle(c, t, 9.4, 12.8, '只有少数人，还在雪中守着一盏灯。', 'Only a few still kept a lamp alight in the snow.')
        ui.year_stamp(c, t, 13.6, 17.8, '1986', '反向传播', x=170, y=330,
                      cap_en='BACKPROPAGATION · RUMELHART · HINTON · WILLIAMS', col=(0.1, 0.1, 0.12), glow_a=0.0,
                      cap_col=(0.55, 0.2, 0.15))
        ui.subtitle(c, t, 15.2, 19.2, '让误差逆流而上——每一个连接，都学会修正自己。',
                    'Let the error flow backward — so every connection learns to correct itself.')
        ui.skill_popup(c, t, 18.8, 23.0, '反向传播', 'BACKPROPAGATION', '沿链式法则，把误差传回每一层：∂L/∂w = ∂L/∂y · ∂y/∂w',
                       y=260, accent=(1.0, 0.62, 0.5))
        ui.subtitle(c, t, 23.2, 26.8, '有了隐藏层，XOR 终于被解开。', 'With a hidden layer, XOR was finally solved.')
        ui.subtitle(c, t, 27.0, 29.8, '冰，开始融化。', 'And the ice began to melt.', fo=0.6)

    def draw_network(self, c, t, scol, scarf_red):
        N = NODES
        oh = inv(*T_OH, t)
        hi = inv(*T_HI, t)
        fwd = inv(*T_FWD, t)
        solved = smooth(inv(21.8, 22.6, t))
        open_k = ease_in_out(inv(24.2, 26.0, t))
        gate(c, GATE_X, GROUND(GATE_X) + 2, t, 1 - smooth(inv(23.4, 24.2, t)), solved, open_k)
        if 23.6 < t < 26:
            falling_shards(c, t, 23.8, GATE_X - 165, GROUND(GATE_X) - 480, GATE_X + 165, GROUND(GATE_X) - 250, 3, 30)
        # supports
        for k, (x, y) in N.items():
            c.drawLine(x, y + 20, x + 6, GROUND(x) + 6, paint(ICE, 0.45, stroke=7))
        for i, (a, b) in enumerate(EDGES_OH):
            warm = smooth(inv(T_OH[1] - 0.05, T_OH[1] + 0.4, t))
            ice_edge(c, N[a], N[b], 1 - warm, warm, t, i * 7 + 1, age=inv(T_OH[1], T_OH[1] + 3.5, t))
            falling_shards(c, t, T_OH[1] - 0.1, *N[a], *N[b], seed=i + 10)
        for i, (a, b) in enumerate(EDGES_HI):
            warm = smooth(inv(T_HI[1] - 0.05, T_HI[1] + 0.4, t))
            ice_edge(c, N[a], N[b], 1 - warm, warm, t, i * 7 + 30, age=inv(T_HI[1], T_HI[1] + 3.5, t))
            falling_shards(c, t, T_HI[1] - 0.1, *N[a], *N[b], seed=i + 30)
        for k, (x, y) in N.items():
            if k == 'O':
                warm = smooth(inv(T_TOUCH, T_TOUCH + 0.3, t))
            elif k.startswith('H'):
                warm = smooth(inv(T_OH[1] - 0.1, T_OH[1] + 0.3, t))
            else:
                warm = smooth(inv(T_HI[1] - 0.1, T_HI[1] + 0.3, t))
            ice_node(c, x, y, 1 - warm, warm, t)
        # backward sparks (red, "δ")
        if 0 < oh < 1:
            for (a, b) in EDGES_OH:
                spark(c, *N[b], *N[a], ease_in_out(oh), label='δ' if a == 'H1' else None)
        if 0 < hi < 1:
            for (a, b) in EDGES_HI:
                spark(c, *N[a], *N[b], ease_in_out(hi))
        # forward pass (gold)
        if 0 < fwd < 1:
            f1 = inv(0.0, 0.5, fwd)
            f2 = inv(0.5, 1.0, fwd)
            for (h, i) in EDGES_HI:
                if 0 < f1 < 1:
                    spark(c, *N[i], *N[h], f1, col=(1.0, 0.85, 0.5), r=5)
            for (h, o) in EDGES_OH:
                if 0 < f2 < 1:
                    spark(c, *N[h], *N[o], f2, col=(1.0, 0.85, 0.5), r=5)
        # touch flash + ring
        if t >= T_TOUCH:
            dt = t - T_TOUCH
            ox, oy = N['O']
            if dt < 1.6:
                c.drawCircle(ox, oy, 30 + dt * 380, paint(RED, 0.8 * (1 - dt / 1.6), stroke=3 + 6 * (1 - dt / 1.6)))
            if dt < 0.6:
                c.drawCircle(ox, oy, 140, paint(RED, 0.7 * (1 - dt / 0.6), blur=40, blend='plus'))
            a = env(t, T_TOUCH + 0.1, 20.0, 0.3, 0.5)
            draw_text(c, 'ŷ − y', ox + 4, oy - 44, 'garamond_m', 30, (1, 0.8, 0.75), a, glow=8, glow_col=RED)
        # hero
        if t < 26.2:
            hx = HERO_B
            reach = smooth(inv(T_TOUCH - 0.9, T_TOUCH - 0.1, t)) * (1 - smooth(inv(19.0, 20.0, t)))
            reach_up = ch.Pose(py=0.455, torso=0.1, head=-0.15, th=(0.12, -0.12), kn=(0.12, 0.05),
                               sh=(2.0, 0.1), el=(0.05, 0.3))
            pose = ch.blend(ch.idle(t), reach_up, reach)
            pose = ch.blend(pose, ch.LOOK_UP, 0.4 * smooth(inv(21.5, 22.5, t)))
            facing = -1 if t < 25.6 else 1
        else:
            hx = self.walk_b.x(t)
            pose = ch.blend(ch.idle(t), ch.walk(self.walk_b.phase(t)), self.walk_b.moving(t))
            facing = 1
        ch.Hero(HH, col=(0.07, 0.07, 0.08)).draw(
            c, hx, GROUND(hx) + 4, pose, t, facing=facing, eyes=1.0,
            scarf=dict(length=1.05, width=0.065, cols=[scol], wind=0.8, speed=0.2, seed=3.0, n=16,
                       glow=10 * scarf_red))
