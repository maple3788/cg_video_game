"""Chapter III — The Chain of Memory (1990–2014): RNN lantern bridge, the vanishing-gradient
spider (boss), and LSTM gates with the cell-state 'memory highway'."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, Rng, clamp, ease_in, ease_in_out, ease_out, env, inv, lerp, paint,
                    poly, radial, linear, smooth, seg, snoise, fbm, smooth_path, ribbon)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width, draw_rich
from .common import PL, Cam, Walker, draw_layers, fade_black, lookat, lerp_cam, shake

HH = 118.0
DECK = 700.0
X0, SP = 300.0, 150.0
TOKENS = ['我', '在', '法', '国', '长', '大', '，', '后', '来', '…', '所', '以', '我', '能', '说', '流', '利', '的', '？']
NPOST = 27
AMBER = (1.0, 0.7, 0.34)
GOLD = (1.0, 0.88, 0.55)
RED = (1.0, 0.2, 0.15)
WOOD = (0.035, 0.035, 0.04)
STABS = [12.9, 14.4, 15.9, 17.3, 18.6, 19.9, 21.1]
T_LSTM_BLOCK = 25.6
GATES = [(3480.0, '遗忘门', 'f', 23.0), (3640.0, '输入门', 'i', 23.4), (3800.0, '输出门', 'o', 23.8)]
CELL_Y = DECK - 250.0
T_CELL = 24.0


def post_x(k):
    return X0 + SP * k


def token_post(i):
    return i + 1


def deck_y(x):
    k = (x - X0) / SP
    frac = k - math.floor(k)
    return DECK + 11 * math.sin(math.pi * frac)


class Memory:
    DUR = 36.0
    NAME = 'memory'

    def __init__(self):
        self.walk = Walker([(0.0, 12.6, 330.0, 1520.0)], h=HH)
        self.run = Walker([(13.4, 22.6, 1520.0, 3280.0)], h=HH, stride=1.24)
        self.walk2 = Walker([(31.0, 36.5, 3280.0, 3980.0)], h=HH)

    # ---------------------------------------------------------------- hero
    def hero(self, t):
        if t < 13.4:
            x = self.walk.x(t)
            mv = self.walk.moving(t)
            pose = ch.blend(ch.idle(t), ch.walk(self.walk.phase(t)), mv)
            if 12.9 < t < 13.4:
                pose = ch.blend(pose, ch.CROUCH, 0.35 * env(t, 12.9, 13.5, 0.1, 0.3))
            return x, pose, 0.3 * mv, 1
        if t < 22.6:
            x = self.run.x(t)
            mv = self.run.moving(t)
            pose = ch.blend(ch.idle(t), ch.run(x / (1.24 * HH)), mv)
            return x, pose, 1.0 * mv + 0.2, 1
        if t < 31.0:
            x = 3280.0
            pose = ch.idle(t)
            if 25.2 < t < 27.0:
                pose = ch.blend(pose, ch.CROUCH, 0.5 * env(t, 25.4, 27.0, 0.2, 0.8))
            if t > 28.4:
                pose = ch.blend(pose, ch.LOOK_UP, 0.3 * env(t, 28.4, 31.0, 0.6, 0.6))
            return x, pose, 0.1, -1 if 28.3 < t < 30.9 else 1
        x = self.walk2.x(t)
        mv = self.walk2.moving(t)
        return x, ch.blend(ch.idle(t), ch.walk(self.walk2.phase(t)), mv), 0.3 * mv, 1

    def read_index(self, t):
        hx = self.hero(t)[0]
        j = -1
        for i in range(len(TOKENS)):
            if post_x(token_post(i)) < hx + 30:
                j = i
        return j

    def destroyed(self, t, x):
        for ts in STABS:
            if t >= ts and abs(x - self.stab_x(ts)) < 95:
                return ts
        return None

    def stab_x(self, ts):
        return self.hero(ts)[0] - 230 if ts > 13 else self.hero(ts)[0] - 300

    # ---------------------------------------------------------------- camera
    def camera(self, t):
        hx = self.hero(t)[0]
        a = lookat(hx + 260, DECK - 170, 1.0)
        b = lookat(hx + 180, DECK - 240, 0.9)
        c_ = lookat(3420, DECK - 240, 0.94)
        d = lookat(hx + 120, DECK - 270, 0.8)
        if t < 12.6:
            cam = a
        elif t < 14.0:
            cam = lerp_cam(a, b, ease_in_out(inv(12.6, 14.0, t)))
        elif t < 22.2:
            cam = b
        elif t < 24.0:
            cam = lerp_cam(b, c_, ease_in_out(inv(22.2, 24.0, t)))
        elif t < 31.0:
            cam = c_
        else:
            cam = lerp_cam(c_, d, ease_in_out(inv(31.0, 36.0, t)))
        hits = [(ts, 1.0) for ts in STABS] + [(T_LSTM_BLOCK, 1.3)]
        dx, dy, rot = shake(t, hits, amp=16, decay=6, freq=40)
        return (cam[0] + dx, cam[1] + dy, cam[2], cam[3]), rot

    # ---------------------------------------------------------------- drawing
    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        (camx, camy, zoom, zc), rot = self.camera(t)
        self.draw_bg(c, t, camx, camy)
        cam = Cam(camx, camy, zoom, zc, rot)
        cam.begin(c)
        self.draw_bridge(c, t, camx, zoom)
        self.draw_cell_state(c, t)
        hx, pose, spd, facing = self.hero(t)
        ch.Hero(HH).draw(c, hx, deck_y(hx) + 2, pose, t, facing=facing, eyes=1.0,
                         scarf=dict(length=1.05, width=0.065, cols=[RED], wind=0.35, speed=spd, seed=4.0, n=16,
                                    glow=6))
        self.draw_spider(c, t)
        self.draw_memory_token(c, t)
        cam.end(c)
        fx.motes(c, t, n=40, seed=9, col=AMBER, a=0.35, size=(1, 2.6), drift=(4, -10))
        img = fr.to_float()
        img = post.look_limbo(img, idx, sat=1.0, tone=(0.93, 0.96, 1.05), vig=0.66, haze=0.2, grain_amt=0.05,
                              contrast_k=1.08)
        img = fade_black(img, seg(t, 0.0, 1.4) * (1 - seg(t, 34.6, 36.0) * 0.85))
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        self.overlays(c, t)

    def draw_bg(self, c, t, camx, camy):
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=linear(0, 0, 0, H, [
            (0.02, 0.022, 0.03), (0.1, 0.105, 0.125), (0.22, 0.23, 0.26)], [0, 0.55, 1])))
        mx, my = 360 - camx * 0.02, 300 - camy * 0.03
        c.drawCircle(mx, my, 640, paint((1, 1, 1), shader=radial(mx, my, 640, [
            ((0.8, 0.82, 0.86), 0.42), ((0.8, 0.82, 0.86), 0.0)])))
        c.drawCircle(mx, my, 150, paint((0.9, 0.9, 0.91)))
        c.drawCircle(mx - 34, my - 22, 110, paint((0.82, 0.83, 0.85), 0.45, blur=16))
        c.drawCircle(mx + 50, my + 40, 30, paint((0.8, 0.81, 0.83), 0.4, blur=6))
        for L in mountain_layers():
            L.layer.draw(c, L.x0 - camx * L.p, L.y0 - camy * L.py)
        hy = DECK - 90 - camy * 0.9
        c.drawRect(skia.Rect.MakeXYWH(0, hy - 260, W, 520), paint((1, 1, 1), shader=linear(0, hy - 260, 0, hy + 260, [
            ((0.5, 0.52, 0.58), 0.0), ((0.5, 0.52, 0.58), 0.42), ((0.5, 0.52, 0.58), 0.0)])))
        fx.draw_fog(c, t, 860 - camy * 0.5, speed=14, alpha=0.5, seed=31, w=3600, h=520, cell=230,
                    col=(0.55, 0.57, 0.62))
        fx.draw_fog(c, t, 1010 - camy * 0.8, speed=24, alpha=0.55, seed=32, w=3600, h=520, cell=230,
                    col=(0.62, 0.64, 0.68))

    def draw_bridge(self, c, t, camx, zoom):
        j = self.read_index(t)
        vis0 = camx - 600
        vis1 = camx + W / zoom + 600
        # support ropes & rails
        rail = paint(WOOD, stroke=3)
        for k in range(NPOST - 1):
            x0, x1 = post_x(k), post_x(k + 1)
            if x1 < vis0 or x0 > vis1:
                continue
            gone = self.destroyed(t, (x0 + x1) / 2)
            for hgt in (72, 38):
                p = skia.Path()
                p.moveTo(x0, DECK - hgt)
                p.quadTo((x0 + x1) / 2, DECK - hgt + 30, x1, DECK - hgt)
                if gone is None:
                    c.drawPath(p, rail)
            # deck planks
            r = Rng(k * 13 + 1)
            x = x0 + 4
            while x < x1 - 4:
                w_ = r.u(12, 17)
                if self.destroyed(t, x) is None:
                    y = deck_y(x)
                    c.save()
                    c.translate(x, y)
                    c.rotate(r.n(0, 2))
                    c.drawRect(skia.Rect.MakeXYWH(0, -2, w_ - 3, 13), paint(WOOD))
                    c.restore()
                else:
                    ts = self.destroyed(t, x)
                    dt = t - ts
                    if dt < 2.0:
                        y = deck_y(x) + 0.5 * 1400 * dt * dt
                        c.save()
                        c.translate(x + r.n(0, 30) * dt, y)
                        c.rotate(r.n(0, 200) * dt)
                        c.drawRect(skia.Rect.MakeXYWH(0, -2, w_ - 3, 13), paint(WOOD))
                        c.restore()
                x += w_
            # under-ropes
            c.drawLine(x0, DECK + 14, (x0 + x1) / 2, DECK + 60, paint(WOOD, stroke=2))
            c.drawLine((x0 + x1) / 2, DECK + 60, x1, DECK + 14, paint(WOOD, stroke=2))
        # posts, lanterns, tokens
        for k in range(NPOST):
            x = post_x(k)
            if x < vis0 or x > vis1:
                continue
            gone = self.destroyed(t, x)
            if gone is not None and t - gone > 2.0:
                continue
            fall = 0.0 if gone is None else 0.5 * 1400 * (t - gone) ** 2
            c.save()
            c.translate(0, fall)
            if gone is not None:
                c.rotate(min(30, (t - gone) * 60))
            c.drawRect(skia.Rect.MakeXYWH(x - 5, DECK - 150, 10, 175), paint(WOOD))
            c.drawLine(x, DECK - 148, x + 34, DECK - 150, paint(WOOD, stroke=5))
            ti = k - 1
            b = 0.0
            if 0 <= ti < len(TOKENS):
                b = 0.05
                if ti <= j:
                    b = 0.1 + 0.9 * math.exp(-(j - ti) * 0.42)
            if gone is not None:
                b *= max(0.0, 1 - (t - gone) * 3)
            lx, ly = x + 34, DECK - 118
            c.drawLine(lx, DECK - 150, lx, ly - 14, paint(WOOD, stroke=1.5))
            if b > 0.06:
                c.drawCircle(lx, ly, 150, paint(AMBER, 1.0, shader=radial(lx, ly, 150, [
                    (AMBER, 0.42 * b), (AMBER, 0.0)]), blend='plus'))
                c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(lx - 9, ly - 13, 18, 26), 4, 4),
                            paint((1, 0.9, 0.6), min(1.0, b * 1.2)))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(lx - 10, ly - 14, 20, 28), 4, 4),
                        paint(WOOD, stroke=2.5))
            c.drawLine(lx - 10, ly, lx + 10, ly, paint(WOOD, stroke=1.5))
            if 0 <= ti < len(TOKENS):
                glyph = TOKENS[ti]
                ga = 0.22 + 0.78 * b
                if glyph == '？' and t > 30.8:
                    k2 = smooth(inv(30.8, 31.4, t))
                    draw_text(c, '？', lx, DECK - 188, 'serif_m', 40, (1, 0.95, 0.85), ga * (1 - k2), glow=10,
                              glow_col=AMBER)
                    draw_text(c, '法语', lx, DECK - 188, 'serif_m', 40, (1, 0.95, 0.85), k2, glow=16,
                              glow_col=GOLD, glow_a=0.8)
                else:
                    draw_text(c, glyph, lx, DECK - 188, 'serif_m', 40, (1, 0.95, 0.85), ga, glow=10 * b,
                              glow_col=AMBER, glow_a=0.6)
            c.restore()
            if gone is not None and t - gone < 1.2:
                self.sparks(c, t - gone, x + 34, DECK - 118, k)
        # memory thread between consecutive lit lanterns
        for i in range(max(0, j - 12), j):
            x0, x1 = post_x(token_post(i)) + 34, post_x(token_post(i + 1)) + 34
            if self.destroyed(t, x0 - 34) is not None or self.destroyed(t, x1 - 34) is not None:
                continue
            b0 = 0.1 + 0.9 * math.exp(-(j - i) * 0.42)
            p = skia.Path()
            p.moveTo(x0, DECK - 118)
            p.quadTo((x0 + x1) / 2, DECK - 170, x1, DECK - 118)
            c.drawPath(p, paint(AMBER, 0.7 * b0, stroke=2))
            c.drawPath(p, paint(AMBER, 0.35 * b0, stroke=8, blur=6, blend='plus'))

    @staticmethod
    def sparks(c, dt, x, y, seed):
        r = Rng(seed + 101)
        a = max(0.0, 1 - dt / 1.2)
        p = paint(AMBER, a)
        for k in range(18):
            vx, vy = r.n(0, 260), r.u(-420, -60)
            px_, py_ = x + vx * dt, y + vy * dt + 0.5 * 1300 * dt * dt
            c.drawCircle(px_, py_, 2.2, p)
        if dt < 0.3:
            c.drawCircle(x, y, 90, paint(AMBER, 0.8 * (1 - dt / 0.3), blur=30, blend='plus'))

    def draw_cell_state(self, c, t):
        # gates rising
        for (gx, cn, sym, tg) in GATES:
            k = ease_out(inv(tg, tg + 0.9, t))
            if k <= 0:
                continue
            top = DECK - 330
            yoff = (1 - k) * 380
            c.save()
            c.translate(0, yoff)
            glow = 0.6 + 0.4 * math.sin(t * 3 + gx)
            for dx in (-34, 34):
                c.drawRect(skia.Rect.MakeXYWH(gx + dx - 5, top, 10, 330), paint(WOOD))
            c.drawRect(skia.Rect.MakeXYWH(gx - 42, top - 10, 84, 14), paint(WOOD))
            opening = 0.5 + 0.35 * math.sin(t * 1.6 + gx * 0.01)
            ph = 150 * (1 - opening)
            c.drawRect(skia.Rect.MakeXYWH(gx - 28, CELL_Y + 18, 56, ph), paint((0.12, 0.12, 0.14)))
            c.drawRect(skia.Rect.MakeXYWH(gx - 28, CELL_Y + 18 + ph - 3, 56, 3), paint(GOLD, 0.9 * k * glow))
            draw_text(c, 'σ', gx, top + 50, 'dejavu_serif', 28, GOLD, 0.9 * k)
            draw_text(c, cn, gx, top - 30, 'serif_m', 26, (1, 0.95, 0.85), k, glow=8, glow_col=GOLD)
            draw_text(c, sym, gx, top - 62, 'garamond_m', 30, GOLD, 0.9 * k)
            c.restore()
        k = inv(T_CELL, T_CELL + 0.9, t)
        if k <= 0:
            return
        x_right = 4100
        x_left = lerp(x_right, post_x(1) - 60, ease_out(k))
        pulse = 1.0
        if t > T_LSTM_BLOCK:
            pulse += 1.5 * math.exp(-(t - T_LSTM_BLOCK) * 3)
        c.drawLine(x_left, CELL_Y, x_right, CELL_Y, paint(GOLD, 0.95, stroke=3.2))
        c.drawLine(x_left, CELL_Y, x_right, CELL_Y, paint(GOLD, 0.5 * pulse, stroke=18, blur=12, blend='plus'))
        # flowing highlights
        for q in range(14):
            u = ((t * 0.18 + q / 14) % 1.0)
            x = lerp(x_left, x_right, u)
            props.glow_dot(c, x, CELL_Y, 3, GOLD, 0.8, halo=4)
        draw_text(c, 'cₜ', x_right - 40, CELL_Y - 22, 'garamond_m', 34, GOLD, 0.9 * k)
        if t > T_LSTM_BLOCK:
            dt = t - T_LSTM_BLOCK
            if dt < 1.2:
                bx = self.stab_x(T_LSTM_BLOCK) + 0
                c.drawCircle(bx, CELL_Y, 30 + dt * 500, paint(GOLD, 0.9 * (1 - dt / 1.2), stroke=4))
                c.drawCircle(bx, CELL_Y, 90, paint(GOLD, 0.8 * (1 - dt / 1.2), blur=30, blend='plus'))

    def draw_memory_token(self, c, t):
        if t < 28.4 or t > 31.2:
            return
        k = inv(28.4, 30.4, t)
        xa = 2400
        x = lerp(xa, post_x(token_post(18)) + 34, ease_in_out(k))
        y = CELL_Y - 4
        if t > 30.4:
            d = ease_in(inv(30.4, 30.8, t))
            y = lerp(CELL_Y, DECK - 200, d)
        a = env(t, 28.4, 31.2, 0.3, 0.4)
        c.drawCircle(x, y, 60, paint(GOLD, 0.35 * a, blur=26, blend='plus'))
        draw_text(c, '法国', x, y + 12, 'serif_m', 34, (1, 0.97, 0.9), a, glow=12, glow_col=GOLD, glow_a=0.9)
        if 30.8 < t < 31.6:
            dt = t - 30.8
            c.drawCircle(x, DECK - 200, 20 + dt * 300, paint(GOLD, 0.8 * (1 - dt / 0.8), stroke=3))

    # ---------------------------------------------------------------- spider
    def spider_body(self, t):
        hx = self.hero(min(t, 22.6))[0]
        appear = ease_out(inv(12.2, 13.4, t))
        leave = ease_in(inv(26.0, 28.8, t))
        bx = hx - 600 + 50 * math.sin(t * 0.9)
        by = DECK - 600 - (1 - appear) * 520 - leave * 800 + 16 * math.sin(t * 2.1)
        return bx, by, appear

    @staticmethod
    def leg(c, hip, tip, L1=430, L2=520, bend=None, w0=26):
        hx, hy = hip
        tx, ty = tip
        if bend is None:
            bend = 1 if tx >= hx else -1
        d = math.hypot(tx - hx, ty - hy)
        d = min(max(d, 1.0), (L1 + L2) * 0.995)
        base = math.atan2(ty - hy, tx - hx)
        a = math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1))
        ang = base - a * bend
        kx, ky = hx + math.cos(ang) * L1, hy + math.sin(ang) * L1
        col = (0.005, 0.005, 0.007)
        # femur (slightly arched), tibia tapering to a needle
        mx, my = (hx + kx) / 2, (hy + ky) / 2
        nx, ny = -(ky - hy) / L1, (kx - hx) / L1
        mx += nx * 18 * bend
        my += ny * 18 * bend
        c.drawPath(ribbon([(hx, hy), (mx, my), (kx, ky)], [w0, w0 * 0.9, w0 * 0.7]), paint(col))
        dx, dy = tx - kx, ty - ky
        c.drawPath(ribbon([(kx, ky), (kx + dx * 0.55, ky + dy * 0.55), (kx + dx * 0.85, ky + dy * 0.85), (tx, ty)],
                          [w0 * 0.7, w0 * 0.5, w0 * 0.3, 1.2]), paint(col))
        c.drawCircle(kx, ky, w0 * 0.5, paint(col))
        # bristles
        r = Rng(int(hx * 7 + hy))
        for k in range(10):
            u = r.u(0.1, 0.9)
            px_, py_ = lerp(kx, tx, u * 0.8), lerp(ky, ty, u * 0.8)
            c.drawLine(px_, py_, px_ + r.n(0, 10), py_ + r.n(0, 10), paint(col, stroke=1.4))

    def draw_spider(self, c, t):
        if t < 12.1 or t > 29.5:
            return
        bx, by, vis = self.spider_body(t)
        col = (0.005, 0.005, 0.007)
        # it hangs from a thread in the darkness above
        c.drawLine(bx - 130, by - 120, bx - 150, by - 2000, paint((0.35, 0.36, 0.4), 0.8, stroke=2.2))
        # stabbing legs
        all_stabs = STABS + [T_LSTM_BLOCK]
        for n, ts in enumerate(all_stabs):
            if not (ts - 0.7 < t < ts + 1.6):
                continue
            tx_ = self.stab_x(ts)
            ty_ = DECK + 8 if ts != T_LSTM_BLOCK else CELL_Y + 4
            rest = (bx + 260, by + 240)
            if t < ts - 0.14:
                u = ease_out(inv(ts - 0.7, ts - 0.14, t))
                tip = (lerp(rest[0], tx_ - 60, u), lerp(rest[1], ty_ - 360, u))
            elif t < ts:
                u = ease_in(inv(ts - 0.14, ts, t))
                tip = (lerp(tx_ - 60, tx_, u), lerp(ty_ - 360, ty_, u))
            elif t < ts + 0.5:
                if ts != T_LSTM_BLOCK:
                    tip = (tx_, ty_)
                else:
                    tip = (tx_ - 40 * (t - ts), ty_ - 300 * ease_out((t - ts) / 0.5))
            else:
                u = ease_in(inv(ts + 0.5, ts + 1.6, t))
                base_tip = (tx_, ty_) if ts != T_LSTM_BLOCK else (tx_ - 20, ty_ - 300)
                tip = (lerp(base_tip[0], rest[0], u), lerp(base_tip[1], rest[1], u))
            hip = (bx + 90 + (n % 2) * 40, by + 30)
            self.leg(c, hip, tip, w0=30)
        # other legs curled, twitching
        for k, (ox, oy, dx, dy) in enumerate([(-60, 30, -420, 250), (-20, 50, -250, 380), (30, 60, 120, 400),
                                             (60, 40, 330, 300), (-90, 0, -480, 60), (80, 10, 420, 120)]):
            ph = t * 1.6 + k * 1.9
            tip = (bx + dx + 30 * math.sin(ph), by + dy + 24 * math.cos(ph * 1.3))
            self.leg(c, (bx + ox, by + oy), tip, L1=300, L2=360, w0=22)
        # abdomen + cephalothorax
        c.drawOval(skia.Rect.MakeXYWH(bx - 380, by - 200, 460, 340), paint(col))
        c.drawOval(skia.Rect.MakeXYWH(bx + 20, by - 90, 240, 196), paint(col))
        r = Rng(3)
        for k in range(60):  # hair along the outline
            a = r.u(0, 6.28)
            ex, ey = bx - 130 + math.cos(a) * 200, by - 20 + math.sin(a) * 150
            c.drawLine(ex, ey, ex + math.cos(a) * r.u(6, 16), ey + math.sin(a) * r.u(6, 16), paint(col, stroke=2))
        # eyes on the front of the head
        for e, (ex, ey, rr) in enumerate([(205, 0, 9), (222, 18, 7), (190, 22, 7), (230, 40, 5), (200, 44, 6),
                                          (178, 42, 4), (238, 4, 4), (168, 8, 5)]):
            flick = 0.75 + 0.25 * math.sin(t * 6 + e * 1.3)
            x, y = bx + ex, by + ey
            c.drawCircle(x, y, rr * 4.2, paint(RED, 0.45 * vis * flick, blur=rr * 2.4, blend='plus'))
            c.drawCircle(x, y, rr, paint((1.0, 0.18, 0.12), vis))
            c.drawCircle(x - rr * 0.3, y - rr * 0.3, rr * 0.35, paint((1.0, 0.6, 0.5), 0.9 * vis))

    # ---------------------------------------------------------------- overlays
    def overlays(self, c, t):
        ui.chapter_card(c, t, 0.8, 5.6, '第 三 章', '记忆之链', 'CHAPTER III · THE CHAIN OF MEMORY', '1990 — 2014')
        ui.subtitle(c, t, 5.8, 9.4, '循环神经网络：一次只读一个字，把记忆交给下一步。',
                    'A recurrent network reads one word at a time, handing its memory to the next step.')
        ui.subtitle(c, t, 9.6, 12.8, '可链条越长，最初的那盏灯，就越暗。',
                    'But the longer the chain, the dimmer the very first lamp.')
        a = env(t, 6.0, 12.4, 0.8, 0.8)
        if a > 0:
            draw_rich(c, [('h', 'garamond_m', 46, 0, 0), ('t', 'garamond_m', 28, 12, 0),
                          (' = tanh( W · h', 'garamond_m', 46, 0, 0.02), ('t−1', 'garamond_m', 28, 12, 0),
                          (' + U · x', 'garamond_m', 46, 0, 0.02), ('t', 'garamond_m', 28, 12, 0),
                          (' )', 'garamond_m', 46, 0, 0)], W / 2 + 330, 260, col=(1, 0.93, 0.8), a=a * 0.9,
                      glow=10, glow_col=AMBER)
        ui.boss_card(c, t, 13.3, 16.9, '遗 忘 之 蛛', 'THE VANISHING GRADIENT')
        a = env(t, 18.4, 22.6, 0.4, 0.6)
        if a > 0:
            jit = (snoise(t * 30, 5) * 4, snoise(t * 30, 6) * 3) if int(t * 8) % 5 == 0 else (0, 0)
            draw_rich(c, [('∂h', 'garamond_m', 58, 0, 0), ('t', 'garamond_m', 34, 14, 0),
                          (' / ∂h', 'garamond_m', 58, 0, 0), ('1', 'garamond_m', 34, 14, 0),
                          ('  =  ∏ ∂h', 'garamond_m', 58, 0, 0), ('k', 'garamond_m', 34, 14, 0),
                          (' / ∂h', 'garamond_m', 58, 0, 0), ('k−1', 'garamond_m', 34, 14, 0),
                          ('   →  0', 'garamond_m', 58, 0, 0)], W / 2 + 180 + jit[0], 300 + jit[1],
                      col=(1, 0.85, 0.82), a=a, glow=16, glow_col=RED, glow_a=0.7)
        ui.subtitle(c, t, 18.8, 22.8, '梯度在长链中层层相乘，渐渐消失——远处的记忆被吞噬。',
                    'Multiplied step after step, the gradient vanishes — and distant memories are devoured.')
        ui.skill_popup(c, t, 27.0, 31.2, '长短期记忆', 'LSTM · 1997', '遗忘门 · 输入门 · 输出门：让记忆沿直线流淌',
                       y=260, accent=GOLD)
        ui.subtitle(c, t, 31.5, 33.8, '门，让记忆穿过了漫长的路……', 'Gates let memory survive the long road…')
        ui.subtitle(c, t, 34.0, 36.0, '……可它仍然只能，一步一步地走。', '…yet it could still only walk, one step at a time.',
                    fo=0.6)


@functools.lru_cache(maxsize=1)
def mountain_layers():
    out = []
    for k, (p, y0, amp, col, blur) in enumerate([(0.05, 560, 140, 0.11, 5), (0.15, 640, 110, 0.075, 3),
                                                  (0.3, 760, 90, 0.05, 2)]):
        width = 3200 + 4200 * p

        def draw(c, y0=y0, amp=amp, col=col, k=k, width=width):
            g = props.ground_fn(y0, amp, 520 - k * 80, 60 + k, amp2=14)
            props.draw_ground(c, 0, width, g, (col, col, col + 0.015), bottom=1400)
        out.append(PL(Layer(width, 1400, draw, blur=blur), p=p, x0=-200, y0=0))
    return out
