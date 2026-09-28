"""Chapter I — The Forest of Rules (1950–1969): IF/THEN signs, ELIZA, perceptron, XOR."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, Rng, clamp, ease_in_out, ease_out, env, inv, lerp, paint, poly,
                    radial, linear, smooth, seg, snoise)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width
from .common import Cam, Walker, draw_layers, fade_black, lookat, lerp_cam
from .forest import ForestSet, NEAR

HH = 118.0
ELIZA_X = 2330.0
HERO_STOP = 2140.0
PER_X = 3470.0      # perceptron pedestal
HERO_P = 3290.0
GATE_X = 3830.0

SIGN_WORDS = ['IF', 'THEN', 'ELSE', 'GOTO', 'AND', 'OR', 'NOT', 'END IF', 'WHILE', 'RETURN']


def _mid_signs(c):
    r = Rng(77)
    x = -300
    while x < 4600:
        x += r.u(260, 520)
        w = r.u(90, 150)
        props.hanging_sign(c, x, -200, r.u(420, 760), w, w * 0.42, r.choice(SIGN_WORDS),
                           col=(0.26, 0.26, 0.26), sway=r.n(0, 0.05))


_SET = None


def forest_set():
    global _SET
    if _SET is None:
        trunks = [(120, 70), (1500, 62), (3050, 84), (4250, 70), (4700, 80)]
        _SET = ForestSet(seed=2, width=5600, near_trunks=trunks,
                         clearings=[(ELIZA_X + 150, 520), (PER_X + 100, 700)], mid_extra=_mid_signs)
    return _SET


@functools.lru_cache(maxsize=8)
def gear_sprite(r, teeth, spokes, shade, blur):
    pad = r * 0.3 + blur * 3
    size = 2 * (r * 1.2 + pad)
    return Layer(size, size, lambda c: props.gear(c, size / 2, size / 2, r, 0.0, (shade,) * 3, 1.0,
                                                   teeth=teeth, spokes=spokes), blur=blur)


GEARS = [  # far plane gears: (x, y, r, teeth, spokes, speed)
    (260, 300, 330, 18, 6, 0.05), (760, 90, 230, 14, 5, -0.07), (1320, 360, 420, 22, 6, 0.035),
    (1900, 150, 270, 16, 5, -0.06), (2420, 330, 360, 20, 6, 0.045), (2980, 120, 250, 15, 5, -0.065),
    (3450, 330, 300, 18, 6, 0.05),
]


def draw_gears(c, t):
    for (x, y, r, teeth, spokes, spd) in GEARS:
        spr = gear_sprite(r, teeth, spokes, 0.5, 3.0)
        c.save()
        c.translate(x, y)
        c.rotate(math.degrees(t * spd))
        spr.draw(c, -spr.W / 2, -spr.H / 2)
        c.restore()


NEAR_SIGNS = [(820, 250, 'IF', 190), (1180, 360, 'THEN', 230), (1620, 190, 'ELSE', 210),
              (1960, 300, 'GOTO', 220), (2960, 260, 'END', 200)]


def draw_near_signs(c, t):
    for i, (x, rope_len, word, w) in enumerate(NEAR_SIGNS):
        sway = 0.035 * math.sin(t * 0.8 + i * 1.7) + 0.015 * math.sin(t * 1.9 + i)
        props.hanging_sign(c, x, -420, rope_len + 420, w, w * 0.4, word, col=NEAR, sway=sway)


# ------------------------------------------------------------------ ELIZA terminal

CRT_LINES = [  # (t_start, text, style)
    (15.0, 'ELIZA · DOCTOR  1966', 'dim'),
    (16.0, '> 我觉得很孤独。', 'in'),
    (18.3, '匹配   “我觉得 *”', 'rule'),
    (18.9, '套用   “你为什么觉得 * ？”', 'rule'),
    (19.5, '< 你为什么觉得很孤独？', 'out'),
]


def eliza(c, x, gy, t, on=1.0):
    col = NEAR
    bx, by, bw, bh = x, gy - 440, 400, 470
    # light spill from the screen onto the fog (behind the machine)
    sx, sy = x + 200, gy - 330
    flick = 0.92 + 0.08 * snoise(t * 9, 3)
    c.drawCircle(sx, sy, 520, paint((1, 1, 1), 1.0, shader=radial(sx, sy, 520, [
        ((1, 1, 1), 0.33 * on * flick), ((1, 1, 1), 0.0)]), blend='plus'))
    # cables into the ground
    for k in range(5):
        p = skia.Path()
        x0 = bx + 20 + k * 22
        p.moveTo(x0, gy - 60)
        p.cubicTo(x0 - 60 - k * 30, gy - 40, x0 - 120 - k * 40, gy + 10, x0 - 160 - k * 50, gy + 20)
        c.drawPath(p, paint(col, stroke=6 - k * 0.6))
    # antennas
    for (ax, h_, bend) in ((bx + 70, 170, -30), (bx + 330, 120, 25)):
        p = skia.Path()
        p.moveTo(ax, by)
        p.quadTo(ax + bend, by - h_ * 0.6, ax + bend * 0.4, by - h_)
        c.drawPath(p, paint(col, stroke=4))
        c.drawCircle(ax + bend * 0.4, by - h_, 7, paint(col))
        blink = 0.5 + 0.5 * math.sin(t * 3 + ax)
        props.glow_dot(c, ax + bend * 0.4, by - h_, 3.5, (1, 1, 1), 0.8 * on * blink, halo=5)
    # hood + body
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(bx + 40, by - 46, bw - 80, 60), 10, 10), paint(col))
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(bx, by, bw, bh), 20, 20), paint(col))
    # engraved label on hood
    draw_text(c, 'E L I Z A', bx + bw / 2, by - 10, 'cinzel_b', 22, (0.55, 0.55, 0.55), 0.8 * on, tracking=0.2)
    # vents
    for k in range(6):
        c.drawRect(skia.Rect.MakeXYWH(bx + 40 + k * 14, by + bh - 140, 6, 60), paint((0.1, 0.1, 0.1)))
    # tubes
    for k in range(4):
        tx = bx + 250 + k * 30
        props.glow_dot(c, tx, by + bh - 100, 4, (1, 1, 1), 0.6 * on * (0.6 + 0.4 * math.sin(t * 2 + k)), halo=4)
    # keyboard ledge
    kb = poly([(bx + 10, gy - 190), (bx + bw - 10, gy - 190), (bx + bw + 40, gy - 150), (bx - 40, gy - 150)])
    c.drawPath(kb, paint(col))
    for row in range(2):
        for k in range(12):
            kx = bx + 20 + k * 30 - row * 8
            c.drawRect(skia.Rect.MakeXYWH(kx, gy - 182 + row * 14, 22, 9), paint((0.09, 0.09, 0.1)))
    # screen
    sr = skia.Rect.MakeXYWH(bx + 32, by + 22, bw - 64, 236)
    c.drawRRect(skia.RRect.MakeRectXY(sr, 22, 22), paint((0.0, 0.0, 0.0)))
    inner = skia.Rect.MakeXYWH(sr.left() + 12, sr.top() + 12, sr.width() - 24, sr.height() - 24)
    c.drawRRect(skia.RRect.MakeRectXY(inner, 18, 18), paint((1, 1, 1), 1.0, shader=radial(
        inner.centerX(), inner.centerY(), 230, [((0.34, 0.36, 0.34), on), ((0.1, 0.11, 0.1), on)])))
    c.save()
    c.clipRRect(skia.RRect.MakeRectXY(inner, 18, 18))
    for yy in range(int(inner.top()), int(inner.bottom()), 4):
        c.drawLine(inner.left(), yy, inner.right(), yy, paint((0, 0, 0), 0.18 * on, stroke=1.2))
    ty = inner.top() + 34
    for (ts, text, style) in CRT_LINES:
        if t < ts:
            break
        k = (t - ts) * 16 if style != 'dim' else 99
        colr = {'dim': (0.7, 0.72, 0.7), 'in': (0.95, 0.97, 0.95), 'rule': (0.8, 0.84, 0.8),
                'out': (1.0, 1.0, 1.0)}[style]
        size = 17 if style in ('dim',) else 19
        if style == 'rule':
            hl = min(1.0, (t - ts) * 2) * on
            c.drawRect(skia.Rect.MakeXYWH(inner.left() + 10, ty - 19, inner.width() - 20, 26),
                       paint((1, 1, 1), 0.12 * hl))
        draw_text(c, text, inner.left() + 16, ty, 'cjk_fallback', size, colr, on, align='left',
                  reveal=min(k, len(text)), glow=6, glow_a=0.5)
        ty += 36
    # cursor
    if int(t * 2.2) % 2 == 0:
        c.drawRect(skia.Rect.MakeXYWH(inner.left() + 16, ty - 16, 11, 18), paint((1, 1, 1), 0.85 * on))
    c.restore()
    # glass highlight
    c.drawRRect(skia.RRect.MakeRectXY(inner, 18, 18), paint((1, 1, 1), 0.08 * on, blur=6, blend='plus'))


# ------------------------------------------------------------------ perceptron + XOR gate

def perceptron(c, x, gy, t, lit):
    col = NEAR
    # input stones
    ins = [(x - 120, gy - 34, 'x₁'), (x - 72, gy - 52, 'x₂')]
    for (sx, sy, lab) in ins:
        props.rock(c, sx, gy + 2, 34, gy - sy, 5 + int(sx), col)
    # wires
    for i, (sx, sy, lab) in enumerate(ins):
        p = skia.Path()
        p.moveTo(sx, sy + 4)
        p.cubicTo(sx + 10, sy - 60, x - 60, gy - 200, x - 22, gy - 178)
        c.drawPath(p, paint(col, stroke=3.2))
        if lit > 0:
            c.drawPath(p, paint((1, 1, 1), 0.55 * lit, stroke=2.0, blur=3, blend='plus'))
            # pulses travelling up the wire
            m = skia.PathMeasure(p, False)
            L = m.getLength()
            for k in range(2):
                u = ((t * 0.7 + k * 0.5 + i * 0.25) % 1.0)
                pos, tan = m.getPosTan(u * L)
                props.glow_dot(c, pos.x(), pos.y(), 3.0, (1, 1, 1), lit, halo=4)
        draw_text(c, lab, sx, sy - 18, 'garamond_m', 30, (0.85, 0.85, 0.85), 0.9 * lit, glow=8, glow_a=0.4)
        draw_text(c, f'w{"₁₂"[i]}', sx + 36 + i * 10, sy - 70 - i * 30, 'garamond_m', 24, (0.75, 0.75, 0.75),
                  0.8 * lit)
    # pedestal
    props.column(c, x, gy + 4, 70, 150, col)
    c.drawPath(poly([(x - 60, gy - 150), (x + 60, gy - 150), (x + 42, gy - 168), (x - 42, gy - 168)]), paint(col))
    # orb
    ox, oy = x, gy - 196
    c.drawCircle(ox, oy, 28, paint(col))
    if lit > 0:
        pulse = 0.85 + 0.15 * math.sin(t * 4)
        c.drawCircle(ox, oy, 240, paint((1, 1, 1), 1.0, shader=radial(ox, oy, 240, [
            ((1, 1, 1), 0.3 * lit * pulse), ((1, 1, 1), 0.0)]), blend='plus'))
        props.glow_dot(c, ox, oy, 20, (1, 1, 1), lit, halo=3.5, core=1.0)
        draw_text(c, 'Σ', ox, oy + 8, 'dejavu_serif', 24, (0.1, 0.1, 0.1), lit)
    return ox, oy


GRID = [((-60, -60), 1), ((60, -60), 0), ((-60, 60), 0), ((60, 60), 1)]
# (dx, dy) relative to panel centre; value = XOR(x1, x2) with x1 = right, x2 = up:
# top-left (x1=0,x2=1)=1, top-right (1,1)=0, bottom-left (0,0)=0, bottom-right (1,0)=1

ATTEMPTS = [  # (time, angle(rad), offset)
    (31.0, math.pi / 2, 0.0), (32.25, math.pi / 4, -38.0), (33.5, 0.0, 0.0)]


def xor_gate(c, x, gy, t, beam):
    col = NEAR
    gw, gh = 330, 500
    left, top = x - gw / 2, gy - gh
    body = skia.Path()
    body.moveTo(left, gy + 10)
    body.lineTo(left, top + gw / 2)
    body.arcTo(skia.Rect.MakeXYWH(left, top, gw, gw), 180, 180, False)
    body.lineTo(left + gw, gy + 10)
    body.close()
    c.drawPath(body, paint(col))
    draw_text(c, 'X O R', x, top + 92, 'cinzel_b', 34, (0.5, 0.5, 0.5), 0.9, tracking=0.2)
    cx, cy = x, gy - 250
    panel = skia.Rect.MakeXYWH(cx - 125, cy - 125, 250, 250)
    c.drawRect(panel, paint((0.06, 0.06, 0.065)))
    c.drawRect(panel, paint((0.35, 0.35, 0.35), stroke=1.5))
    # axis ticks
    draw_text(c, '0', cx - 60, cy + 150, 'garamond_m', 24, (0.6, 0.6, 0.6), 0.9)
    draw_text(c, '1', cx + 60, cy + 150, 'garamond_m', 24, (0.6, 0.6, 0.6), 0.9)
    draw_text(c, '0', cx - 146, cy + 68, 'garamond_m', 24, (0.6, 0.6, 0.6), 0.9)
    draw_text(c, '1', cx - 146, cy - 52, 'garamond_m', 24, (0.6, 0.6, 0.6), 0.9)
    # decision line attempts
    att = None
    for (ta, ang, off) in ATTEMPTS:
        if t >= ta - 0.9:
            att = (ta, ang, off)
    fail_flash = 0.0
    if att is not None and beam > 0.5:
        ta, ang, off = att
        idx = ATTEMPTS.index(att)
        if idx > 0 and t < ta:
            pa = ATTEMPTS[idx - 1]
            k = ease_in_out(inv(ta - 0.9, ta - 0.1, t))
            ang = lerp(pa[1], ang, k)
            off = lerp(pa[2], off, k)
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        px_, py_ = cx + nx * off, cy + ny * off
        c.save()
        c.clipRect(panel)
        # shade the positive half-plane
        big = 600
        half = poly([(px_ - dx * big, py_ - dy * big), (px_ + dx * big, py_ + dy * big),
                     (px_ + dx * big + nx * big, py_ + dy * big + ny * big),
                     (px_ - dx * big + nx * big, py_ - dy * big + ny * big)])
        c.drawPath(half, paint((1, 1, 1), 0.08))
        c.drawLine(px_ - dx * big, py_ - dy * big, px_ + dx * big, py_ + dy * big,
                   paint((1, 1, 1), 0.9, stroke=3, blur=0.6))
        c.drawLine(px_ - dx * big, py_ - dy * big, px_ + dx * big, py_ + dy * big,
                   paint((1, 1, 1), 0.5, stroke=10, blur=8, blend='plus'))
        c.restore()
        if t >= ta:
            fail_flash = math.exp(-(t - ta) * 3.0)
        # which points are misclassified: side sign vs label (either labelling)
        sides = []
        for (gx, gy_), v in GRID:
            s = (gx - nx * off) * nx + (gy_ - ny * off) * ny
            sides.append(1 if s > 0 else 0)
        wrong_a = [i for i, ((g, v), sd) in enumerate(zip(GRID, sides)) if v != sd]
        wrong_b = [i for i, ((g, v), sd) in enumerate(zip(GRID, sides)) if v != 1 - sd]
        wrong = wrong_a if len(wrong_a) <= len(wrong_b) else wrong_b
    else:
        wrong = []
    for i, ((gx, gy_), v) in enumerate(GRID):
        px_, py_ = cx + gx, cy + gy_
        if v == 1:
            props.glow_dot(c, px_, py_, 13, (1, 1, 1), 0.95, halo=3.2)
        else:
            c.drawCircle(px_, py_, 14, paint((0.75, 0.75, 0.75), 0.9, stroke=3))
        if i in wrong and t >= (att[0] if att else 99):
            a = 0.35 + 0.65 * fail_flash
            s = 20
            p = paint(ui.RED, a, stroke=4)
            c.drawLine(px_ - s, py_ - s, px_ + s, py_ + s, p)
            c.drawLine(px_ + s, py_ - s, px_ - s, py_ + s, p)
    if fail_flash > 0:
        c.drawRect(panel, paint(ui.RED, 0.25 * fail_flash, blend='plus'))
    # cracks after the last failure
    k = seg(t, 33.9, 34.6)
    if k > 0:
        r = Rng(5)
        for j in range(7):
            p = skia.Path()
            x0, y0 = cx + r.u(-30, 30), cy + r.u(-30, 30)
            p.moveTo(x0, y0)
            a = r.u(0, 6.28)
            L = r.u(120, 260) * k
            steps = 6
            for s_ in range(steps):
                a += r.n(0, 0.45)
                x0 += math.cos(a) * L / steps
                y0 += math.sin(a) * L / steps
                p.lineTo(x0, y0)
            c.drawPath(p, paint((0.9, 0.9, 0.9), 0.8, stroke=1.8))
    return cx, cy


# ------------------------------------------------------------------ scene

class ForestOfRules:
    DUR = 36.0
    NAME = 'rules'

    def __init__(self):
        self.walker = Walker([(0.0, 14.2, 560.0, HERO_STOP)], h=HH)

    def camera(self, t):
        F = forest_set()
        gy = F.gy0
        if t < 26.0:
            hx = self.walker.x(t)
            walk_cam = lookat(hx + 190, gy - 250, 1.08)
            eliza_cam = lookat(ELIZA_X + 60, gy - 290, 1.32)
            k = ease_in_out(inv(12.0, 16.5, t))
            return lerp_cam(walk_cam, eliza_cam, k)
        a = lookat(PER_X + 20, gy - 230, 1.3)
        b = lookat(3650, gy - 270, 1.1)
        k = ease_in_out(inv(29.4, 31.2, t))
        cam = lerp_cam(a, b, k)
        # gentle drift
        return (cam[0] + (t - 26) * 4, cam[1], cam[2] + (t - 26) * 0.004, cam[3])

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        F = forest_set()
        camx, camy, zoom, zc = self.camera(t)
        F.draw(c, t + 40, camx, camy, zoom, zc, rays=0.8, dyn_far=lambda cv: draw_gears(cv, t),
               ray_origin=(1500 + camx * 0.3, -700))
        cam = Cam(camx, camy, zoom, zc)
        cam.begin(c)
        draw_near_signs(c, t)
        gy = F.ground
        if t < 26.0:
            eliza(c, ELIZA_X, F.gy0 + 2, t, on=0.35 + 0.65 * smooth(inv(14.2, 15.0, t)))
            hx = self.walker.x(t)
            mv = self.walker.moving(t)
            pose = ch.blend(ch.idle(t), ch.walk(self.walker.phase(t)), mv)
            if t > 14.4:
                pose = ch.blend(pose, ch.LOOK_UP, 0.55 * smooth(inv(14.4, 15.6, t)))
            ch.Hero(HH).draw(c, hx, gy(hx) + 3, pose, t, eyes=1.0,
                             scarf=dict(length=0.72, width=0.055, cols=[(0.02, 0.02, 0.025)], wind=0.1,
                                        speed=0.5 * mv, seed=2.0, n=14))
        else:
            lit = smooth(inv(27.4, 27.8, t))
            ox, oy = perceptron(c, PER_X, F.gy0 + 2, t, lit)
            beam = seg(t, 29.8, 30.6)
            gxc, gyc = GATE_X, F.gy0 - 248
            if beam > 0:
                ex, ey = lerp(ox, gxc - 150, beam), lerp(oy, gyc, beam)
                c.drawLine(ox + 26, oy, ex, ey, paint((1, 1, 1), 0.85, stroke=2.5))
                c.drawLine(ox + 26, oy, ex, ey, paint((1, 1, 1), 0.4, stroke=12, blur=10, blend='plus'))
            xor_gate(c, GATE_X, F.gy0 + 2, t, beam)
            reach = smooth(inv(26.8, 27.5, t)) * (1 - smooth(inv(28.6, 29.4, t)))
            pose = ch.blend(ch.idle(t), ch.REACH, reach)
            pose = ch.blend(pose, ch.LOOK_UP, 0.35 * smooth(inv(29.5, 30.5, t)))
            ch.Hero(HH).draw(c, HERO_P, gy(HERO_P) + 3, pose, t, eyes=1.0,
                             scarf=dict(length=0.72, width=0.055, cols=[(0.02, 0.02, 0.025)], wind=0.12 + 0.5 *
                                        seg(t, 33.5, 36), seed=2.0, n=14))
        cam.end(c)
        draw_layers(c, [F.foreground()], camx, camy, zoom, zc)
        fx.motes(c, t, n=60, seed=5, col=(1, 1, 1), a=0.4, size=(1.0, 3.0), drift=(5, -7))
        winter = seg(t, 33.8, 36.0)
        if winter > 0:
            fx.snow(c, t, n=int(420 * winter), seed=7, wind=160, fall=90, a=0.85 * winter)
        img = fr.to_float()
        img = post.look_limbo(img, idx, tone=(lerp(1.0, 0.93, winter), lerp(0.985, 0.97, winter),
                                              lerp(0.95, 1.03, winter)))
        img = fx.frost(img, 0.34 * seg(t, 33.9, 36.0) ** 1.3)
        # cut from ELIZA to perceptron: quick dip
        img = fade_black(img, seg(t, 0.0, 0.6) * (1 - 0.85 * env(t, 25.7, 26.3, 0.3, 0.3)))
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        # ------------------------------------------------ overlays
        ui.chapter_card(c, t, 0.5, 5.6, '第 一 章', '规则之森', 'CHAPTER I · THE FOREST OF RULES', '1950 — 1969')
        ui.subtitle(c, t, 5.8, 9.9, '在规则之森，每一句话，都必须被人预先写好。',
                    'In the Forest of Rules, every sentence had to be written by hand, in advance.')
        ui.subtitle(c, t, 10.2, 14.3, '机器只会查表：如果……就……否则……',
                    'Machines could only look things up: IF … THEN … ELSE …')
        ui.dialogue_box(c, t, 15.6, 18.6, '词元', '……我觉得很孤独。', speaker_en='TOKEN', x=1300, w=980, y=835)
        ui.dialogue_box(c, t, 19.3, 22.8, 'ELIZA · 1966', '你为什么觉得很孤独？', speaker_en='MIT', x=1300, w=980,
                        y=835)
        ui.subtitle(c, t, 23.0, 26.0, '它像是在倾听，其实只是在匹配关键词。',
                    'It seemed to listen — but it was only matching keywords.')
        ui.year_stamp(c, t, 26.5, 30.6, '1958', '感知机 · 罗森布拉特', x=170, y=330,
                      cap_en='PERCEPTRON · F. ROSENBLATT')
        ui.subtitle(c, t, 27.8, 30.8, '一个神经元：输入加权求和，越过阈值，就点亮。',
                    'One neuron: weigh the inputs, sum them, and fire past a threshold.')
        ui.subtitle(c, t, 31.0, 34.0, '1969 · 可是，一条直线永远分不开 XOR。',
                    '1969 · But no single straight line can ever separate XOR.')
        ui.subtitle(c, t, 34.2, 36.0, '寒冬，降临了。', 'And then, winter came.', fo=0.3)
