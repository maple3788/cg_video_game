"""LIMBO-style forest: reusable parallax set, prologue (wake up) and title card."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, VLayer, Rng, clamp, ease_in_out, ease_out, env, inv, lerp,
                    paint, radial, linear, smooth, seg, snoise, fbm)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width
from .common import PL, Cam, draw_layers, fade_black

TOP = -1000
BOT = 1250
NEAR = (0.012, 0.012, 0.015)


def canopy(c, x0, x1, y0, y1, col, seed, n=14, a=1.0, blur=0.0, wmax=16):
    r = Rng(seed)
    for i in range(n):
        x = r.u(x0, x1)
        y = r.u(y0, y1)
        ang = r.choice([-1, 1]) * r.u(0.9, 1.9)
        props.branch(c, x, y, ang, r.u(160, 520), r.u(4, wmax), r.i(0, 99999), col, a, blur, depth=3)


class ForestSet:
    """Parallax LIMBO forest.  World x is screen x when camx == 0 (p=1 plane)."""

    def __init__(self, seed=1, width=3600, ground_y=842, near_trunks=None, clearings=(), x_origin=-550.0,
                 mid_extra=None, near_extra=None, far_extra=None, grass_k=1.0):
        self.seed, self.width, self.gy0 = seed, width, ground_y
        self.clearings = list(clearings)
        self.x_origin = x_origin
        self.near_trunks = near_trunks if near_trunks is not None else [
            (170, 74), (1420, 60), (1960, 88), (2620, 66), (3150, 80)]
        self.mid_extra, self.near_extra, self.far_extra = mid_extra, near_extra, far_extra
        self.grass_k = grass_k
        self._layers = None
        self._fg = None

    # -- ground -------------------------------------------------------------
    def clear_k(self, x):
        x = np.asarray(x, dtype=np.float64)
        k = np.zeros_like(x)
        for (cx, r) in self.clearings:
            k = np.maximum(k, np.clip(1 - np.abs(x - cx) / r, 0, 1) ** 1.5)
        return k

    def ground(self, x):
        x = np.asarray(x, dtype=np.float64)
        n = (fbm(x / 700.0, 3, self.seed + 3) - 0.5) * 52 + snoise(x / 90.0, self.seed + 8) * 5
        r = self.gy0 + n * (1 - 0.88 * self.clear_k(x))
        return float(r) if np.ndim(r) == 0 else r

    # -- layers -------------------------------------------------------------
    def layers(self):
        if self._layers is None:
            self._layers = self._build()
        return self._layers

    def _build(self):
        seed, width = self.seed, self.width
        Ht = BOT - TOP

        def mk(p, draw, blur, vector=False):
            x0 = -300 * p - 250 if p < 1 else self.x_origin

            def full(c):
                c.translate(-x0, -TOP)  # draw in world coords of this plane
                draw(c)
            if vector:
                lay = VLayer(width, Ht, full)
            else:
                lay = Layer(width, Ht, full, blur=blur)
            return PL(lay, p=p, x0=x0, y0=TOP)

        xa, xb = self.x_origin - 200, self.x_origin + width + 200

        def far(c):
            rr = Rng(seed + 1)
            g = props.ground_fn(700, 60, 900, seed + 11)
            for i in range(int(36 * width / 3600)):
                x = rr.u(xa, xb)
                props.trunk(c, x, 760, TOP, w=rr.u(10, 22), lean=rr.n(0, 0.02), seed=rr.i(0, 9999),
                            col=(0.72, 0.72, 0.72), branches=rr.i(0, 3), roots=False, bend=10)
            canopy(c, xa, xb, TOP + 100, -150, (0.72, 0.72, 0.72), seed + 21, n=int(20 * width / 3600), wmax=8)
            if self.far_extra:
                self.far_extra(c)
            props.draw_ground(c, xa, xb, g, (0.68, 0.68, 0.68))

        def midfar(c):
            rr = Rng(seed + 2)
            g = props.ground_fn(760, 50, 800, seed + 12)
            for i in range(int(20 * width / 3600)):
                x = rr.u(xa, xb)
                props.trunk(c, x, float(g(x)) + 10, TOP, w=rr.u(18, 34), lean=rr.n(0, 0.03), seed=rr.i(0, 9999),
                            col=(0.5, 0.5, 0.5), branches=rr.i(1, 4), bend=14)
            canopy(c, xa, xb, TOP + 50, -50, (0.5, 0.5, 0.5), seed + 22, n=int(16 * width / 3600), wmax=10)
            props.draw_ground(c, xa, xb, g, (0.48, 0.48, 0.48))
            props.grass(c, xa, xb, g, (0.48, 0.48, 0.48), density=0.5, hmin=6, hmax=18, seed=seed + 5)

        def mid(c):
            rr = Rng(seed + 3)
            g = props.ground_fn(805, 36, 650, seed + 13)
            x = xa
            while x < xb:
                x += rr.u(280, 480)
                props.trunk(c, x, float(g(x)) + 10, TOP, w=rr.u(28, 52), lean=rr.n(0, 0.035), seed=rr.i(0, 9999),
                            col=(0.26, 0.26, 0.26), branches=rr.i(2, 5), bend=18)
            canopy(c, xa, xb, TOP, 0, (0.26, 0.26, 0.26), seed + 23, n=int(12 * width / 3600), wmax=13)
            for i in range(int(9 * width / 3600)):
                x = rr.u(xa, xb)
                props.rope(c, x, TOP + 400, rr.u(80, 480), (0.26, 0.26, 0.26), sway=rr.n(0, 0.03), w=2)
            if self.mid_extra:
                self.mid_extra(c)
            props.draw_ground(c, xa, xb, g, (0.24, 0.24, 0.24))
            props.grass(c, xa, xb, g, (0.24, 0.24, 0.24), density=0.7, hmin=8, hmax=26, seed=seed + 6)

        def near(c):
            rr = Rng(seed + 4)
            for x, wd in self.near_trunks:
                props.trunk(c, x, self.ground(x) + 14, TOP, w=wd, lean=rr.n(0, 0.015),
                            seed=rr.i(0, 9999), col=NEAR, branches=rr.i(2, 5), bend=22)
            canopy(c, xa, xb, TOP, -380, NEAR, seed + 24, n=int(10 * width / 3600), wmax=18)
            if self.near_extra:
                self.near_extra(c)
            props.draw_ground(c, xa, xb, self.ground, NEAR)
            r2 = Rng(seed + 7)
            path = skia.Path()
            x = xa
            while x < xb:
                x += r2.u(0.3, 2.0) / 1.2
                gy = self.ground(x)
                k = (1 - 0.8 * float(self.clear_k(x))) * self.grass_k
                hh = r2.u(6, 32) * (0.5 + 0.9 * float(fbm(x / 60.0, 2, 3))) * k
                ln = r2.n(0, 0.3)
                bw = r2.u(1.4, 3.6)
                tipx, tipy = x + math.sin(ln) * hh, gy - math.cos(ln) * hh
                mx, my = x + math.sin(ln * 0.5) * hh * 0.55, gy - math.cos(ln * 0.5) * hh * 0.55
                path.moveTo(x - bw, gy + 2)
                path.quadTo(mx - bw * 0.4, my, tipx, tipy)
                path.quadTo(mx + bw * 0.4, my, x + bw, gy + 2)
                path.close()
            c.drawPath(path, paint(NEAR))
            for i in range(int(7 * width / 3600)):
                x = rr.u(xa, xb)
                if float(self.clear_k(x)) > 0.01:
                    continue
                props.rock(c, x, self.ground(x) + 4, rr.u(30, 90), rr.u(12, 30), rr.i(0, 999), NEAR)

        return [mk(0.12, far, 7), mk(0.3, midfar, 3.5), mk(0.58, mid, 1.6), mk(1.0, near, 0, vector=True)]

    def foreground(self):
        if self._fg is None:
            seed = self.seed + 9
            w = self.width + 800

            def draw(c):
                col = (0.0, 0.0, 0.0)
                g = props.ground_fn(1030, 40, 400, seed)
                props.draw_ground(c, 0, w, g, col)
                props.grass(c, 0, w, g, col, density=0.45, hmin=40, hmax=170, seed=seed + 1, wide=3.0)
                rr = Rng(seed)
                x = rr.u(0, 400)
                while x < w:
                    props.trunk(c, x, 1150, TOP, w=rr.u(120, 170), lean=rr.n(0, 0.02), seed=rr.i(0, 99), col=col,
                                branches=2)
                    x += rr.u(2400, 3400)
            lay = Layer(w, BOT - TOP, lambda c: (c.translate(0, -TOP), draw(c)), blur=10)
            self._fg = PL(lay, p=1.5, x0=-700, y0=TOP)
        return self._fg

    # -- drawing ------------------------------------------------------------
    def draw_sky(self, c, camy=0.0, glow=(1180, 440), bright=1.0):
        oy = -camy * 0.1
        sh = linear(0, TOP * 0.55 + oy, 0, 900 + oy,
                    [(0.06, 0.06, 0.065), (0.3, 0.3, 0.3), (0.8, 0.8, 0.78), (0.6, 0.6, 0.58)],
                    [0.0, 0.45, 0.8, 1.0])
        p = paint((1, 1, 1))
        p.setShader(sh)
        c.drawRect(skia.Rect.MakeWH(W, H), p)
        c.drawCircle(glow[0], glow[1] + oy, 760, paint((1, 1, 1), 1.0, shader=radial(
            glow[0], glow[1] + oy, 760, [((1, 1, 0.97), 0.55 * bright), ((1, 1, 1), 0.0)])))

    def draw(self, c, t, camx, camy, zoom, zc, rays=1.0, fog=1.0, dyn_far=None, dyn_mid=None,
             ray_origin=(1500, -700), glow=(1180, 440)):
        layers = self.layers()
        self.draw_sky(c, camy, glow)

        def fogband(y, alpha, seed, speed):
            def f(cv):
                fx.draw_fog(cv, t, y - camy * 0.45, speed=speed, alpha=alpha * fog, seed=seed, w=3600, h=560,
                            cell=240, col=(0.93, 0.93, 0.92))
            return f

        def plane(fn, p):
            def f(cv):
                cam = Cam(camx, camy, zoom, zc)
                cam.begin(cv, p)
                fn(cv)
                cam.end(cv)
            return f

        between = {0: fogband(690, 0.55, 5, 9), 1: fogband(745, 0.45, 6, 14), 2: fogband(805, 0.3, 7, 20)}
        if dyn_far is not None:
            f0 = between[0]
            between[0] = lambda cv: (plane(dyn_far, 0.2)(cv), f0(cv))
        if dyn_mid is not None:
            f2 = between[2]
            between[2] = lambda cv: (plane(dyn_mid, 0.58)(cv), f2(cv))
        draw_layers(c, layers[:3], camx, camy, zoom, zc, between=between)
        if rays > 0:
            cam = Cam(camx * 0.6, camy * 0.6, 1 + (zoom - 1) * 0.6, zc)
            cam.begin(c, 1.0)
            fx.god_rays(c, ray_origin[0] + camx * 0.4, ray_origin[1], t, n=9, spread=0.5, length=2000,
                        col=(1, 1, 0.95), a=0.3 * rays, base_angle=math.pi / 2 + 0.3, width=(26, 110), blur=26,
                        seed=5)
            cam.end(c)
        draw_layers(c, layers[3:], camx, camy, zoom, zc)


# ============================================================ prologue & title

HERO_X = 860.0
_FOREST = None


def forest():
    global _FOREST
    if _FOREST is None:
        _FOREST = ForestSet(seed=1, width=3600, clearings=[(HERO_X, 260)])
    return _FOREST


def shot(t):
    """Camera for the prologue: close-up on the lying hero, pull back, tilt up."""
    head = (HERO_X - 0.36 * 118, 842 - 0.12 * 118)
    S = (905.0, 690.0)
    k = ease_in_out(inv(11.2, 17.2, t))
    z = lerp(2.35, 1.0, k)
    camx = lerp(head[0] - S[0], 0.0, k)
    camy = lerp(head[1] - S[1], 0.0, k)
    camy += -170 * ease_in_out(inv(15.4, 18.0, t)) ** 1.6
    return camx, camy, z, S


def hero_scarf(t, length=0.7):
    return dict(length=length, width=0.055, cols=[(0.02, 0.02, 0.025)], wind=0.12, speed=0.0, seed=1.0, n=14)


class Prologue:
    DUR = 18.0
    NAME = 'prologue'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        F = forest()
        fade = seg(t, 7.0, 10.2)
        if fade > 0:
            camx, camy, zoom, zc = shot(t)
            F.draw(c, t, camx, camy, zoom, zc)
            cam = Cam(camx, camy, zoom, zc)
            cam.begin(c)
            if t < 11.9:
                pose = ch.LIE
            elif t < 13.3:
                pose = ch.blend(ch.LIE, ch.CROUCH, ease_in_out(inv(11.9, 13.3, t)))
            elif t < 14.8:
                pose = ch.blend(ch.CROUCH, ch.idle(t), ease_in_out(inv(13.3, 14.8, t)))
            else:
                pose = ch.blend(ch.idle(t), ch.LOOK_UP, smooth(inv(15.0, 16.6, t)) * 0.7)
            eyes = smooth(inv(10.35, 10.6, t))
            if 10.75 < t < 10.86:
                eyes *= 0.15  # blink
            ln = lerp(0.3, 0.7, smooth(inv(12.5, 14.8, t)))
            ch.Hero(118).draw(c, HERO_X, F.ground(HERO_X) + 3, pose, t, facing=1, eyes=eyes,
                              scarf=hero_scarf(t, ln))
            cam.end(c)
            draw_layers(c, [F.foreground()], camx, camy, zoom, zc)
            fx.motes(c, t, n=70, seed=4, col=(1, 1, 1), a=0.45, size=(1.0, 3.2), drift=(5, -7))
        img = fr.to_float()
        img = post.look_limbo(img, idx)
        img = fade_black(img, fade)
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        a1 = env(t, 1.0, 4.3, 1.0, 0.9)
        a2 = env(t, 4.6, 8.2, 1.0, 1.0)
        for a, cn, en in ((a1, '起初，机器没有语言。', 'In the beginning, machines had no language.'),
                          (a2, '只有规则，和漫长的沉默。', 'Only rules — and a long silence.')):
            if a > 0:
                draw_text(c, cn, W / 2, 535, 'serif', 46, ui.INK, a, tracking=0.2, blur=(1 - a) * 8)
                draw_text(c, en, W / 2, 590, 'garamond', 30, ui.INK, a * 0.7, tracking=0.12, blur=(1 - a) * 8)


class Title:
    DUR = 10.0
    NAME = 'title'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        F = forest()
        camx, camy, zoom, zc = shot(18.0)
        camy += -640 * ease_out(inv(0.0, 6.0, t))
        zoom += 0.04 * inv(0, 10, t)
        tt = t + 18.0
        F.draw(c, tt, camx, camy, zoom, zc, rays=1.4)
        cam = Cam(camx, camy, zoom, zc)
        cam.begin(c)
        ch.Hero(118).draw(c, HERO_X, F.ground(HERO_X) + 3, ch.blend(ch.idle(tt), ch.LOOK_UP, 0.7), tt, eyes=1.0,
                          scarf=hero_scarf(tt))
        cam.end(c)
        draw_layers(c, [F.foreground()], camx, camy, zoom, zc)
        fx.motes(c, tt, n=70, seed=4, col=(1, 1, 1), a=0.45, size=(1.0, 3.2), drift=(5, -7))
        img = fr.to_float()
        img = post.look_limbo(img, idx)
        img *= 1 - 0.45 * env(t, 1.0, 9.6, 1.2, 1.2)
        img = fade_black(img, 1 - seg(t, 8.8, 10.0))
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        self.title(c, t)

    def title(self, c, t):
        a = env(t, 1.6, 9.4, 1.4, 1.2)
        if a <= 0:
            return
        yt = 500
        spread = 0.55 + 0.1 * ease_out(inv(1.6, 7.0, t))
        c.drawRect(skia.Rect.MakeWH(W, H), paint((0, 0, 0), 1.0, shader=radial(W / 2, yt, 700,
                   [((0, 0, 0), 0.45 * a), ((0, 0, 0), 0.0)])))
        draw_text(c, 'TOKEN', W / 2, yt, 'cinzel', 170, ui.INK, a, tracking=spread, blur=(1 - a) * 14,
                  glow=26, glow_a=0.35)
        tw = text_width('TOKEN', 'cinzel', 170, spread)
        fx.light_sweep(c, W / 2 - tw / 2 - 40, yt - 150, tw + 80, 180, inv(3.2, 5.0, t), a=0.9, band=90)
        a2 = env(t, 2.6, 9.4, 1.2, 1.2)
        draw_text(c, '词   元', W / 2, yt + 110, 'serif_m', 60, ui.INK, a2, tracking=0.35, blur=(1 - a2) * 10)
        a3 = env(t, 3.8, 9.4, 1.2, 1.2)
        ly = yt + 150
        lw = 240 * ease_out(inv(3.8, 5.5, t))
        p = paint(ui.INK, a3 * 0.6, stroke=1.2)
        c.drawLine(W / 2 - 20 - lw, ly, W / 2 - 20, ly, p)
        c.drawLine(W / 2 + 20, ly, W / 2 + 20 + lw, ly, p)
        props.diamond(c, W / 2, ly, 5, ui.INK, a3, stroke=1.2)
        draw_text(c, '一个词元，穿越沉默纪元的旅程', W / 2, ly + 56, 'serif', 30, ui.INK, a3 * 0.9, tracking=0.3,
                  blur=(1 - a3) * 6)
        draw_text(c, "A TOKEN'S JOURNEY THROUGH THE AGE OF SILENCE", W / 2, ly + 96, 'cinzel', 18, ui.GOLD,
                  a3 * 0.8, tracking=0.3)
