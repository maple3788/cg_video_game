"""Chapter V — ATTENTION (2017).  The climax: self-attention web, the colour bloom that
opens the world (GRIS), the Transformer tower and its apex."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, VLayer, Rng, clamp, ease_in, ease_in_out, ease_out, ease_out_expo, ease_out_back, env,
                    inv, lerp, paint, poly, radial, linear, smooth, seg, snoise, smooth_path, rs)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width, draw_rich
from .common import fade_black, fade_white

TOKS = list('那只动物没有过马路，因为它太累了')
N = len(TOKS)
IT = 12  # 它
GROUND = 905.0
HEAD_COLS = [(0.93, 0.27, 0.35), (1.0, 0.76, 0.3), (0.26, 0.76, 0.74), (0.62, 0.46, 0.95),
             (1.0, 0.52, 0.28), (0.38, 0.6, 1.0)]
SKY_STOPS = [(0.5, 0.47, 0.76), (0.9, 0.6, 0.72), (1.0, 0.78, 0.64), (1.0, 0.91, 0.76)]
GOLD = (1.0, 0.84, 0.5)
CREAM = (0.98, 0.95, 0.9)


def tok_pos(i):
    return 210 + i * 100, 500 - 30 * math.sin(math.pi * i / (N - 1))


def head_edges():
    E = []  # (head, i, j, w, above)
    for (i, j, w) in [(12, 2, 0.71), (12, 3, 0.62), (12, 8, 0.12), (14, 12, 0.45), (14, 3, 0.3)]:
        E.append((0, i, j, w, True))
    for i in range(1, N):
        E.append((1, i, i - 1, 0.55, False))
    for i in range(0, N - 1, 2):
        E.append((2, i, i + 1, 0.42, False))
    for i in range(N):
        if i != 9 and i % 2 == 1:
            E.append((3, i, 9, 0.3, True))
    for (i, j, w) in [(6, 7, 0.6), (6, 8, 0.55), (4, 6, 0.5), (5, 6, 0.4), (14, 13, 0.5), (10, 14, 0.4),
                      (11, 12, 0.3), (0, 3, 0.35)]:
        E.append((4, i, j, w, True))
    for (i, j, w) in [(0, 2, 0.5), (1, 3, 0.45), (15, 14, 0.6), (13, 14, 0.5), (12, 14, 0.35), (7, 8, 0.6),
                      (3, 7, 0.25), (2, 12, 0.4)]:
        E.append((5, i, j, w, True))
    return E


EDGES = head_edges()


def arc_path(i, j, above=True):
    x0, y0 = tok_pos(i)
    x1, y1 = tok_pos(j)
    if above:
        y0 -= 34
        y1 -= 34
        cy = min(y0, y1) - 40 - abs(x1 - x0) * 0.42
    else:
        y0 += 34
        y1 += 34
        cy = max(y0, y1) + 30 + abs(x1 - x0) * 0.5
    p = skia.Path()
    p.moveTo(x0, y0)
    p.quadTo((x0 + x1) / 2, cy, x1, y1)
    return p


@functools.lru_cache(maxsize=1)
def arc_paths():
    return [arc_path(i, j, ab) for (h, i, j, w, ab) in EDGES]


def partial(p, k):
    if k >= 0.999:
        return p
    m = skia.PathMeasure(p, False)
    dst = skia.Path()
    m.getSegment(0, m.getLength() * max(0.0, k), dst, True)
    return dst


# ------------------------------------------------------------------ colour world backdrop

@functools.lru_cache(maxsize=1)
def color_world():
    sky = fx.wash(W, H, SKY_STOPS, seed=61, blotch=0.1, cell=280, pos=[0, 0.35, 0.62, 0.75], blooms=6)

    def hills(c):
        r = Rng(62)
        for (y0, amp, sc, col, seed) in [(760, 60, 520, (0.93, 0.74, 0.8), 1), (815, 50, 420, (0.78, 0.5, 0.64), 2),
                                         (870, 34, 380, (0.46, 0.28, 0.46), 3)]:
            g = props.ground_fn(y0, amp, sc, 70 + seed, amp2=4)
            props.draw_ground(c, -50, W + 50, g, col, bottom=H + 10)
        # the hero's ground: a soft plateau
        pts = [(-50, H + 10), (-50, 930), (500, 915), (800, GROUND + 2), (1120, GROUND + 2), (1420, 918),
               (W + 50, 940), (W + 50, H + 10)]
        c.drawPath(smooth_path(pts, close=True, tension=0.5), paint((0.2, 0.12, 0.24)))
    hl = Layer(W, H, arr=fx.watercolorize(Layer(W, H, hills).arr, seed=63, edge=0.4, gran=0.2, wobble=2.5))

    def shapes(c):
        r = Rng(64)
        for k in range(7):
            x, y = r.u(80, W - 80), r.u(140, 480)
            s = r.u(20, 70)
            if k % 2 == 0:
                c.drawCircle(x, y, s, paint((1, 1, 1), 0.35, stroke=2))
            else:
                c.drawPath(poly([(x, y - s), (x + s * 0.87, y + s / 2), (x - s * 0.87, y + s / 2)]),
                           paint((1, 1, 1), 0.3, stroke=2))
    sh = Layer(W, H, shapes, blur=0.6)
    return sky, hl, sh


def draw_petals(c, t, t0, cx, cy, radius, n=140, seed=5, a=1.0):
    r = Rng(seed)
    cols = [(1.0, 0.72, 0.78), (1.0, 0.86, 0.55), (1, 1, 1), (0.78, 0.62, 1.0), (0.5, 0.85, 0.85)]
    for k in range(n):
        ang = r.u(0, 6.283)
        born = t0 + r.u(0, 3.0)
        if t < born:
            continue
        age = t - born
        rr = radius * r.u(0.55, 1.0) + age * r.u(20, 90)
        x = cx + math.cos(ang) * rr + math.sin(age * 2 + k) * 10
        y = cy + math.sin(ang) * rr * 0.75 + age * 30
        al = a * max(0.0, 1 - age / 4.0)
        if al <= 0:
            continue
        col = cols[k % len(cols)]
        c.save()
        c.translate(x, y)
        c.rotate(age * r.u(-120, 120))
        s = r.u(3, 8)
        c.drawOval(skia.Rect.MakeXYWH(-s, -s * 0.45, s * 2, s * 0.9), paint(col, al))
        c.restore()


# ------------------------------------------------------------------ tower

T_BASE = GROUND
FLOORS = []  # (y_top, y_bot, kind, label_cn, label_en, side)


def _floors():
    if FLOORS:
        return FLOORS
    y = GROUND - 60
    FLOORS.append((y, GROUND, 'plinth', '', '', 0))
    FLOORS.append((y - 180, y, 'embed', '输入嵌入 + 位置编码', 'EMBEDDING + POSITIONAL ENCODING', -1))
    y -= 180
    for k in range(4):
        FLOORS.append((y - 110, y, 'attn', '多头注意力', 'MULTI-HEAD ATTENTION', 1 if k == 0 else 0))
        FLOORS.append((y - 140, y - 110, 'norm', '残差 & 归一化', 'ADD & NORM', -1 if k == 0 else 0))
        FLOORS.append((y - 220, y - 140, 'ffn', '前馈网络', 'FEED-FORWARD', 1 if k == 0 else 0))
        FLOORS.append((y - 250, y - 220, 'norm', '', '', 0))
        y -= 250
    FLOORS.append((y - 170, y, 'out', '线性层 + Softmax → 下一个词', 'LINEAR + SOFTMAX → NEXT TOKEN', -1))
    return FLOORS


CROWN_Y = GROUND - 60 - 180 - 1000 - 170 - 330   # top of the spire


@functools.lru_cache(maxsize=1)
def tower_layer():
    fl = _floors()

    def draw(c):
        c.translate(0, 1400)
        cx = W / 2
        stone = (0.97, 0.93, 0.88)
        shade = (0.78, 0.7, 0.82)
        for (yt, yb, kind, lc, le, side) in fl:
            if kind == 'plinth':
                for s_ in range(3):
                    w_ = 780 - s_ * 60
                    c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yb - (s_ + 1) * 20, cx + w_ / 2, yb - s_ * 20),
                               paint(stone if s_ % 2 == 0 else shade))
                continue
            w_ = 600 if kind != 'out' else 470
            if kind == 'norm':
                w_ = 640
            rect = skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yb)
            c.drawRect(rect, paint(stone))
            # right-side shading for volume
            c.drawRect(skia.Rect.MakeLTRB(cx + w_ / 2 - w_ * 0.22, yt, cx + w_ / 2, yb), paint(shade, 0.55))
            if kind == 'attn':
                band = (0.93, 0.45, 0.52)
                c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yt + 10), paint(band))
                for a_ in range(6):
                    ax = cx - w_ / 2 + 50 + a_ * 100
                    props.arch(c, ax, yb - 8, 64, 84, 14, band)
                    c.drawRect(skia.Rect.MakeLTRB(ax - 25, yb - 70, ax + 25, yb - 8), paint((0.3, 0.2, 0.34), 0.9))
            elif kind == 'ffn':
                band = (0.3, 0.72, 0.72)
                c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yt + 8), paint(band))
                for a_ in range(9):
                    ax = cx - w_ / 2 + 42 + a_ * 64
                    c.drawRect(skia.Rect.MakeLTRB(ax - 12, yt + 24, ax + 12, yb - 12), paint(band, 0.85))
            elif kind == 'norm':
                c.drawRect(rect, paint((1.0, 0.82, 0.45)))
                c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yt + 4), paint((1, 0.95, 0.8)))
            elif kind == 'embed':
                c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yt + 10), paint((0.62, 0.48, 0.92)))
                for a_ in range(16):
                    ax = cx - w_ / 2 + 25 + a_ * 36.5
                    c.drawRect(skia.Rect.MakeLTRB(ax - 11, yt + 40, ax + 11, yb - 20), paint((0.62, 0.48, 0.92), 0.8))
            elif kind == 'out':
                c.drawRect(skia.Rect.MakeLTRB(cx - w_ / 2, yt, cx + w_ / 2, yt + 10), paint(GOLD))
                p = skia.Path()
                p.addArc(skia.Rect.MakeLTRB(cx - w_ / 2, yt - w_ / 2 * 0.6, cx + w_ / 2, yt + w_ / 2 * 0.6), 180, 180)
                c.drawPath(p, paint(stone))
            if lc and side != 0:
                lx = cx - w_ / 2 - 40 if side < 0 else cx + w_ / 2 + 40
                ly = (yt + yb) / 2
                ex = lx - 150 * (1 if side < 0 else -1) * 0.35
                c.drawLine(cx + side * (w_ / 2 + 6), ly, lx, ly, paint((0.3, 0.12, 0.34), 0.8, stroke=1.6))
                c.drawCircle(cx + side * (w_ / 2 + 6), ly, 5, paint((0.3, 0.12, 0.34), 0.9))
                al = 'right' if side < 0 else 'left'
                draw_text(c, lc, lx + (-10 if side < 0 else 10), ly + 4, 'serif_b', 34, (0.26, 0.1, 0.3), 0.95,
                          align=al, glow=10, glow_col=(1, 1, 1), glow_a=0.55)
                draw_text(c, le, lx + (-10 if side < 0 else 10), ly + 34, 'cinzel_b', 17, (0.36, 0.16, 0.4), 0.9,
                          align=al, tracking=0.2)
        # xN bracket
        y_top_blocks = GROUND - 60 - 180 - 1000
        y_bot_blocks = GROUND - 60 - 180 - 250
        bx = cx + 360
        c.drawLine(bx, y_top_blocks, bx, y_bot_blocks, paint((0.3, 0.12, 0.34), 0.85, stroke=2.5))
        c.drawLine(bx - 14, y_top_blocks, bx, y_top_blocks, paint((0.3, 0.12, 0.34), 0.85, stroke=2.5))
        c.drawLine(bx - 14, y_bot_blocks, bx, y_bot_blocks, paint((0.3, 0.12, 0.34), 0.85, stroke=2.5))
        draw_text(c, '× N', bx + 20, (y_top_blocks + y_bot_blocks) / 2 + 20, 'garamond_m', 72, (0.3, 0.12, 0.34), 0.95,
                  align='left', glow=10, glow_col=(1, 1, 1), glow_a=0.5)
        # spire
        yt = GROUND - 60 - 180 - 1000 - 170
        c.drawPath(poly([(cx - 60, yt - 60), (cx + 60, yt - 60), (cx + 16, CROWN_Y + 40), (cx - 16, CROWN_Y + 40)]),
                   paint((0.86, 0.8, 0.88)))
        c.drawPath(poly([(cx, CROWN_Y - 30), (cx + 30, CROWN_Y + 20), (cx, CROWN_Y + 70), (cx - 30, CROWN_Y + 20)]),
                   paint(GOLD))
    return VLayer(W, 2600, draw)


# ------------------------------------------------------------------ scene

class Attention:
    DUR = 44.0
    NAME = 'attention'

    def render(self, fr: Frame, t: float, idx: int):
        if t < 6.6:
            self.intro(fr, t, idx)
        elif t < 19.0:
            self.bloom(fr, t, idx)
        else:
            self.tower(fr, t, idx)

    # ---------------------------------------------------------------- A: intro
    def intro(self, fr, t, idx):
        c = fr.c
        fr.clear((0, 0, 0))
        beat = sum(math.exp(-((t - b) / 0.08) ** 2) for b in (3.9, 4.25, 5.1, 5.45))
        img = fr.to_float()
        img = post.grain(img, 0.03, idx)
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        a0 = env(t, 0.3, 6.4, 0.6, 0.5)
        draw_text(c, '第 五 章  ·  注 意 力', W / 2, 330, 'serif', 26, ui.INK, 0.7 * a0, tracking=0.3)
        draw_text(c, 'CHAPTER V · ATTENTION', W / 2, 368, 'cinzel', 16, ui.GOLD, 0.7 * a0, tracking=0.4)
        txt = 'Attention Is All You Need'
        k = clamp((t - 1.0) * 11, 0, len(txt))
        diss = seg(t, 5.9, 6.6)
        a = (1 - diss)
        tw = draw_text(c, txt, W / 2, 560, 'garamond_m', 84, (1, 1, 1), a, reveal=k, tracking=0.04,
                       glow=14 + 20 * beat, glow_a=0.35 + 0.4 * beat)
        full_w = text_width(txt, 'garamond_m', 84, 0.04)
        if k < len(txt) or int(t * 2.4) % 2 == 0:
            from ..text import layout
            items, _ = layout(txt, 'garamond_m', 84.0, 0.04)
            n = int(k)
            cx = W / 2 - full_w / 2 + (items[n - 1][2] + items[n - 1][3] if n > 0 else 0) + 6
            if t < 5.9:
                c.drawRect(skia.Rect.MakeXYWH(cx, 490, 5, 84), paint((1, 1, 1), 0.9))
        a2 = env(t, 3.4, 6.2, 0.6, 0.4)
        draw_text(c, 'Vaswani et al.  ·  2017', W / 2, 640, 'garamond', 36, ui.GOLD, a2, tracking=0.1)
        if diss > 0:
            # letters scatter into sparks falling to where the tokens will be
            r = Rng(3)
            for q in range(90):
                sx = W / 2 - full_w / 2 + r.u(0, full_w)
                sy = 540 + r.u(-30, 20)
                i = q % N
                tx, ty = tok_pos(i)
                u = ease_in_out(diss)
                x, y = lerp(sx, tx, u), lerp(sy, ty, u)
                props.glow_dot(c, x, y, 2.2, (1, 1, 1), 0.9, halo=4)

    # ---------------------------------------------------------------- B: bloom
    def draw_tokens_arcs(self, c, t, color):
        paths = arc_paths()
        for e, (h, i, j, w, ab) in enumerate(EDGES):
            if h == 0:
                ts = 9.1 + (0 if j in (2, 3) else 0.5) + (0.8 if i == 14 else 0)
            else:
                ts = 11.1 + h * 0.2 + (i / N) * 0.45
            k = ease_out(inv(ts, ts + 0.8, t))
            if k <= 0:
                continue
            pulse = 0.8 + 0.2 * math.sin(t * 3 + e)
            col = HEAD_COLS[h] if color else (0.92, 0.92, 0.92)
            p = partial(paths[e], k)
            a = (0.4 + 0.6 * w) * pulse
            if h == 0 and t < 11.1:
                a = 1.0
            c.drawPath(p, paint(col, a * 0.5, stroke=6 + 16 * w, blur=7, blend='plus'))
            c.drawPath(p, paint(col, a, stroke=1.2 + 4.5 * w))
        # weight label on the key arc
        la = env(t, 10.0, 18.6, 0.5, 0.6)
        if la > 0:
            x0, y0 = tok_pos(IT)
            x1, y1 = tok_pos(2)
            mx, my = (x0 + x1) / 2 + 50, min(y0, y1) - 34 - 40 - abs(x1 - x0) * 0.42 / 2 - 44
            col = (1, 0.85, 0.85) if color else (1, 1, 1)
            draw_text(c, '「它」→「动物」  0.71', mx, my, 'serif_m', 30, col, la, glow=10,
                      glow_col=HEAD_COLS[0] if color else (1, 1, 1), glow_a=0.5)
        for i, tk in enumerate(TOKS):
            ts = 7.0 + i * 0.1
            k = ease_out_back(inv(ts, ts + 0.45, t))
            if k <= 0:
                continue
            x, y = tok_pos(i)
            r = 30 * k
            hi = 0.0
            if i == IT:
                hi = env(t, 8.8, 19.0, 0.3, 0.5)
            if i in (2, 3):
                hi = max(hi, env(t, 9.8, 19.0, 0.4, 0.5) * 0.8)
            if color:
                c.drawCircle(x, y, r + 16, paint(GOLD, 0.35 + 0.4 * hi, blur=14, blend='plus'))
                c.drawCircle(x, y, r, paint(CREAM))
                c.drawCircle(x, y, r, paint(HEAD_COLS[i % 6], 0.9, stroke=2.5))
                tc = (0.25, 0.15, 0.3)
            else:
                c.drawCircle(x, y, r + 12, paint((1, 1, 1), 0.12 + 0.3 * hi, blur=12, blend='plus'))
                c.drawCircle(x, y, r, paint((0.06, 0.06, 0.07)))
                c.drawCircle(x, y, r, paint((0.9, 0.9, 0.9), 0.9, stroke=2))
                tc = (0.95, 0.95, 0.95)
            if hi > 0:
                ring = 40 + 10 * math.sin(t * 5)
                c.drawCircle(x, y, ring, paint(HEAD_COLS[0] if color else (1, 1, 1), 0.8 * hi, stroke=2.5))
            if k > 0.5:
                draw_text(c, tk, x, y + 12, 'serif_m', 34, tc, smooth(inv(0.5, 1.0, k)))

    def bloom(self, fr, t, idx):
        c = fr.c
        cx, cy = tok_pos(IT)
        R = ease_in_out(inv(12.0, 15.4, t)) * 2400
        need_dark = t < 15.5
        need_color = t > 11.95
        img_dark = img_color = None
        wind = 0.2 + 0.8 * seg(t, 12.0, 13.5)
        hero_pose = ch.blend(ch.idle(t), ch.LOOK_UP, 0.6 * smooth(inv(7.5, 9.0, t)))
        if t > 12.2:
            hero_pose = ch.blend(hero_pose, ch.ARMS_UP, 0.7 * smooth(inv(12.2, 13.4, t)))
        if need_dark:
            fr.clear((0.02, 0.02, 0.025))
            c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=radial(W / 2, 620, 1100, [
                (0.1, 0.1, 0.115), (0.015, 0.015, 0.02)])))
            c.drawLine(0, GROUND, W, GROUND, paint((0.3, 0.3, 0.32), 0.6, stroke=1.2))
            c.drawRect(skia.Rect.MakeLTRB(0, GROUND, W, H), paint((0.01, 0.01, 0.012)))
            fx.motes(c, t, n=50, seed=8, col=(1, 1, 1), a=0.35, size=(1, 2.5))
            self.draw_tokens_arcs(c, t, False)
            ch.Hero(120, col=(0.0, 0.0, 0.0)).draw(c, W / 2, GROUND + 2, hero_pose, t, eyes=1.0,
                                                   glow_col=(1, 1, 1), glow_r=8, glow_a=0.2,
                                                   scarf=dict(length=1.3, width=0.065, cols=[(0.9, 0.2, 0.16),
                                                              (0.3, 0.5, 1.0)], wind=wind, speed=0.0, seed=6.0, n=18))
            img_dark = fr.to_float()
            img_dark = post.look_limbo(img_dark, idx, sat=0.25, tone=(1, 1, 1), vig=0.6, haze=0.15, dust_amt=0.3)
        if need_color:
            sky, hills, shapes = color_world()
            sky.draw(c, 0, 0)
            shapes.draw(c, 0, 0, 0.8)
            sx, sy = W / 2, 820
            c.drawCircle(sx, sy, 700, paint((1, 1, 1), shader=radial(sx, sy, 700, [
                ((1, 0.9, 0.74), 0.4), ((1, 0.9, 0.75), 0.0)])))
            c.drawCircle(sx, sy, 215, paint((1.0, 0.93, 0.8), 0.9))
            hills.draw(c, 0, 0)
            c.drawRect(skia.Rect.MakeXYWH(0, 150, W, 430), paint((1, 1, 1), shader=linear(0, 150, 0, 580, [
                ((0.3, 0.14, 0.34), 0.0), ((0.3, 0.14, 0.34), 0.28), ((0.3, 0.14, 0.34), 0.0)])))
            self.draw_tokens_arcs(c, t, True)
            draw_petals(c, t, 12.0, cx, cy, R * 0.8, n=160)
            ch.Hero(120, col=(0.16, 0.08, 0.2)).draw(
                c, W / 2, GROUND + 2, hero_pose, t, eyes=1.0, glow_col=(1, 0.85, 0.6), glow_r=10, glow_a=0.45,
                scarf=dict(length=2.0, width=0.07, cols=[(0.93, 0.25, 0.3), (1.0, 0.78, 0.3), (0.3, 0.75, 0.75),
                                                         (0.55, 0.45, 0.95)], wind=wind, speed=0.3, seed=6.0, n=24,
                           glow=10))
            fx.motes(c, t, n=70, seed=9, col=(1, 0.95, 0.85), a=0.6, size=(1.2, 3.2), drift=(8, -16))
            img_color = fr.to_float()
            img_color = post.look_gris(img_color, idx, bloom_amt=0.38, paper_amt=0.8, vig=0.3, sat=1.08,
                                       thresh=0.8)
        if img_dark is not None and img_color is not None:
            field = fx.reveal_field(float(cx), float(cy), seed=7, amp=170.0, cell=160.0)
            m = field.mask(R, edge=60.0)[:, :, None]
            rim = field.rim(R, width=30.0)[:, :, None]
            img = img_dark * (1 - m) + img_color * m
            img += rim * np.array([1.0, 0.9, 0.7], np.float32) * 0.9 * (1 - seg(t, 14.8, 15.5))
            flash = math.exp(-((t - 12.05) / 0.12) ** 2) * 0.6
            img += flash
        else:
            img = img_color if img_color is not None else img_dark
        fr.from_float(img)
        ui.letterbox(c, 1 - ease_in_out(inv(12.4, 14.0, t)))
        ui.subtitle(c, t, 9.4, 11.9, '「它」指的是谁？——它，注意到了「动物」。', 'What does "it" refer to? — It attends to "animal".',
                    style='bar')
        ui.subtitle(c, t, 14.4, 18.8, '自注意力：每一个词，都同时看见所有的词。',
                    'Self-attention: every word sees every other word — all at once.', style='float', y=985)

    # ---------------------------------------------------------------- C/D: tower & apex
    def tower(self, fr, t, idx):
        c = fr.c
        tl = tower_layer()
        k = ease_in_out(inv(19.0, 31.0, t))
        cam_y = lerp(0.0, CROWN_Y - 560, k)       # world y (minus H/2) at the screen centre
        zoom = lerp(0.82, 1.0, k)
        if t > 31.0:
            zoom = lerp(1.0, 1.12, ease_in_out(inv(31.0, 44.0, t)))
        sky, hills, shapes = color_world()
        # sky that deepens toward the heavens
        up = clamp(-cam_y / 1600, 0, 1)
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=linear(0, 0, 0, H, [
            (lerp(0.5, 0.2, up), lerp(0.47, 0.18, up), lerp(0.76, 0.42, up)),
            (lerp(0.9, 0.62, up), lerp(0.6, 0.45, up), lerp(0.72, 0.72, up)),
            (lerp(1.0, 0.95, up), lerp(0.85, 0.72, up), lerp(0.7, 0.78, up))])))
        if up > 0.2:
            r = Rng(4)
            for i in range(160):
                x, y = r.u(0, W), r.u(0, H * 0.7)
                c.drawCircle(x, y, r.u(0.6, 1.8), paint((1, 1, 1), (up - 0.2) * r.u(0.3, 0.9)))
        sky.draw(c, 0, -cam_y * 0.25 - 40, 1 - up * 0.7)
        c.save()
        c.translate(W / 2, H / 2)
        c.scale(zoom, zoom)
        c.translate(-W / 2, -H / 2 - cam_y)
        # far hills only near the ground
        if cam_y > -900:
            hills.draw(c, -320, -40, sx=(W + 640) / W, sy=1.08)
        # positional-encoding ribbons around the base
        fl = _floors()
        emb = [f for f in fl if f[2] == 'embed'][0]
        tl.draw(c, 0, -1400)
        ey = (emb[0] + emb[1]) / 2
        for q, (col, fq, amp) in enumerate([((0.93, 0.35, 0.45), 1.0, 60), ((1.0, 0.75, 0.3), 2.0, 45),
                                            ((0.3, 0.75, 0.75), 4.0, 30), ((0.55, 0.45, 0.95), 8.0, 18)]):
            pts = []
            for s_ in range(90):
                x = -40 + s_ * (W + 80) / 89
                y = ey + math.sin(x / W * fq * 2 * math.pi - t * (1.4 + q * 0.4)) * amp + (q - 1.5) * 26
                pts.append((x, y))
            c.drawPath(smooth_path(pts), paint(col, 0.9, stroke=5))
            c.drawPath(smooth_path(pts), paint(col, 0.35, stroke=16, blur=8, blend='plus'))
        if t < 22.5:
            draw_text(c, 'sin(pos / 10000^(2i/d))', W / 2 + 470, ey - 120, 'garamond_m', 34, (1, 1, 1),
                      0.9 * env(t, 19.3, 22.5, 0.5, 0.5), align='left', shadow=8, shadow_a=0.35)
        # parallel light streams rising through all floors at once
        ybot, ytop = GROUND - 60, CROWN_Y + 60
        for s_ in range(16):
            x = W / 2 - 270 + s_ * 36
            for q in range(3):
                u = ((t * 0.35 + q / 3 + (s_ % 4) * 0.02) % 1.0)
                y = lerp(ybot, ytop, u)
                L = 160
                sp = paint(GOLD, 0.9, stroke=3)
                sp.setShader(linear(x, y + L, x, y, [(GOLD, 0.0), ((1, 1, 1), 0.95)]))
                c.drawLine(x, y + L, x, y, sp)
                c.drawCircle(x, y, 14, paint(GOLD, 0.45, blur=10, blend='plus'))
        # the hero rising on a column of light
        if t < 31.0:
            hy = lerp(GROUND, CROWN_Y + 60, ease_in_out(inv(19.4, 31.0, t)))
            hx = W / 2 + 430 - 400 * seg(t, 29.0, 31.0)
            c.drawRect(skia.Rect.MakeLTRB(hx - 30, hy, hx + 30, GROUND), paint(GOLD, 0.25, blur=20, blend='plus'))
            pose = ch.blend(ch.FLOAT, ch.idle(t), seg(t, 30.0, 31.0))
            sc = dict(length=2.2, width=0.07, cols=[(0.93, 0.25, 0.3), (1.0, 0.78, 0.3), (0.3, 0.75, 0.75),
                                                   (0.55, 0.45, 0.95)], wind=1.0, speed=1.0, seed=7.0, n=24,
                      lift=-2.05 * (1 - seg(t, 30.0, 31.0)), glow=10)
        else:
            hx, hy = W / 2, CROWN_Y + 60
            pose = ch.blend(ch.idle(t), ch.ARMS_UP, smooth(inv(31.0, 31.8, t)))
            sc = dict(length=2.4, width=0.07, cols=[(0.93, 0.25, 0.3), (1.0, 0.78, 0.3), (0.3, 0.75, 0.75),
                                                   (0.55, 0.45, 0.95)], wind=1.0, speed=0.6, seed=7.0, n=24,
                      glow=12)
        # apex burst
        if t > 31.6:
            bx, by = W / 2, CROWN_Y - 10
            dt = t - 31.6
            fx.god_rays(c, bx, by, t, n=18, spread=6.28, length=2200, col=(1, 0.93, 0.8), a=0.22 * min(1, dt * 2),
                        base_angle=0, width=(30, 110), blur=24, seed=11)
            for q, col in enumerate(HEAD_COLS):
                rr = (dt * 380 + q * 90) % 1600
                c.drawCircle(bx, by, rr, paint(col, 0.55 * max(0, 1 - rr / 1600), stroke=3 + 3 * (1 - rr / 1600)))
            c.drawCircle(bx, by, 110 + 12 * math.sin(t * 3), paint((1, 0.9, 0.75), 0.35, blur=60, blend='plus'))
            props.glow_dot(c, bx, by, 18, (1, 0.97, 0.9), 1.0, halo=6, halo_a=0.9)
            draw_petals(c, t, 31.6, bx, by + 200, 300, n=120, seed=8)
        # crown platform
        c.drawRect(skia.Rect.MakeLTRB(W / 2 - 90, CROWN_Y + 60, W / 2 + 90, CROWN_Y + 76), paint((0.55, 0.45, 0.62)))
        ch.Hero(110, col=(0.16, 0.08, 0.2)).draw(c, hx, hy, pose, t, eyes=1.0, glow_col=(1, 0.85, 0.6), glow_r=10,
                                                 glow_a=0.5, scarf=sc)
        c.restore()
        img = fr.to_float()
        img = post.look_gris(img, idx, bloom_amt=0.32 + 0.1 * seg(t, 31.4, 33.0), paper_amt=0.75, vig=0.3,
                             sat=1.08, thresh=0.85)
        img = fade_white(img, math.exp(-((t - 31.65) / 0.25) ** 2) * 0.85, (1, 0.97, 0.9))
        img = fade_white(img, seg(t, 43.0, 44.0) ** 1.5, (1, 0.98, 0.94))
        fr.from_float(img)
        # overlays
        ui.subtitle(c, t, 19.6, 23.6, '位置编码：用正弦波，为每个词标记它的位置。',
                    'Positional encoding: sine waves mark where every word stands.', style='float', y=985)
        ui.subtitle(c, t, 23.9, 27.9, '不再一步一步，而是一瞬之间——并行。',
                    'No longer step by step — all at once, in parallel.', style='float', y=985)
        fa = env(t, 25.0, 30.6, 0.8, 0.8)
        if fa > 0:
            c.drawRect(skia.Rect.MakeXYWH(0, 250, W, 150), paint((0.15, 0.08, 0.2), 0.0, shader=linear(0, 0, W, 0, [
                ((0.15, 0.08, 0.2), 0.0), ((0.15, 0.08, 0.2), 0.45 * fa), ((0.15, 0.08, 0.2), 0.45 * fa),
                ((0.15, 0.08, 0.2), 0.0)], [0, 0.25, 0.75, 1])))
            ui.attention_formula(c, W / 2, 348, size=68, col=(1, 0.97, 0.9), a=fa, glow=14, glow_col=GOLD)
        ta = env(t, 32.4, 43.4, 1.2, 0.8)
        if ta > 0:
            c.drawRect(skia.Rect.MakeXYWH(0, 90, W, 230), paint((0.1, 0.05, 0.16), 1.0, shader=linear(0, 0, W, 0, [
                ((0.1, 0.05, 0.16), 0.0), ((0.1, 0.05, 0.16), 0.35 * ta), ((0.1, 0.05, 0.16), 0.35 * ta),
                ((0.1, 0.05, 0.16), 0.0)], [0, 0.3, 0.7, 1])))
            draw_text(c, 'TRANSFORMER', W / 2, 225, 'cinzel', 124, (1, 0.97, 0.9), ta, tracking=0.32,
                      blur=(1 - ta) * 12, glow=30, glow_col=GOLD, glow_a=0.6)
            draw_text(c, '2017  ·  VASWANI ET AL.', W / 2, 288, 'cinzel', 24, (1, 0.9, 0.7), ta * 0.9, tracking=0.4)
        ui.skill_popup(c, t, 34.2, 38.6, '自注意力', 'SELF-ATTENTION', '任意两个词之间，一步直达：并行、全局、可扩展',
                       y=820, accent=GOLD)
        ui.subtitle(c, t, 38.8, 42.9, '它，成为此后所有大语言模型的基石。',
                    'It became the cornerstone of every large language model to come.', style='float', y=985)
