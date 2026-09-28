"""Chapter IV — The Sea of Meaning (2013–2014): word2vec constellations, seq2seq islands,
and the first threads of attention.  The film turns GRIS-blue."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, Rng, clamp, ease_in, ease_in_out, ease_out, ease_out_back, env, inv,
                    lerp, paint, poly, radial, linear, smooth, seg, snoise, smooth_path)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width
from .common import fade_black

HZ = 720.0
STAR = (0.86, 0.93, 1.0)
PALE = (0.72, 0.84, 0.98)
DEEP = (0.03, 0.05, 0.13)
RED = (1.0, 0.22, 0.16)
BLUE = (0.35, 0.6, 1.0)

WORDS = {  # word: (x, y, cluster)
    '国王': (560, 250, 0), '女王': (760, 290, 0), '王子': (640, 360, 0), '公主': (830, 385, 0),
    '男人': (450, 460, 1), '女人': (650, 500, 1), '男孩': (505, 560, 1), '女孩': (700, 590, 1),
    '法国': (1060, 340, 2), '巴黎': (1160, 240, 2), '日本': (1300, 390, 2), '东京': (1400, 290, 2),
    '中国': (1540, 340, 2), '北京': (1640, 240, 2),
    '猫': (1620, 500, 3), '狗': (1730, 545, 3), '老虎': (1560, 600, 3), '狮子': (1760, 625, 3),
}
CLUSTER_LINES = [('国王', '女王'), ('国王', '王子'), ('女王', '公主'), ('王子', '公主'),
                 ('男人', '女人'), ('男人', '男孩'), ('女人', '女孩'), ('男孩', '女孩'),
                 ('法国', '巴黎'), ('日本', '东京'), ('中国', '北京'), ('法国', '日本'), ('日本', '中国'),
                 ('猫', '狗'), ('猫', '老虎'), ('老虎', '狮子'), ('狗', '狮子')]
ARROWS = [('男人', '女人', 9.0, '性别'), ('国王', '女王', 9.5, '性别'), ('男人', '国王', 10.1, ''),
          ('女人', '女王', 10.4, ''), ('法国', '巴黎', 10.8, '首都'), ('日本', '东京', 11.0, '首都'),
          ('中国', '北京', 11.2, '首都')]

IN_TOK = ['猫', '坐', '在', '垫', '子', '上']
OUT_TOK = ['The', 'cat', 'sat', 'on', 'the', 'mat']
LONG_TOK = list('那只在窗边晒了一下午太阳的老猫终于')
ALIGN = np.array([  # rows: outputs, cols: inputs
    [0.55, 0.1, 0.1, 0.1, 0.05, 0.1],
    [0.9, 0.04, 0.02, 0.02, 0.01, 0.01],
    [0.05, 0.85, 0.05, 0.02, 0.01, 0.02],
    [0.02, 0.05, 0.5, 0.02, 0.01, 0.4],
    [0.05, 0.02, 0.1, 0.45, 0.3, 0.08],
    [0.02, 0.02, 0.03, 0.55, 0.36, 0.02],
])


@functools.lru_cache(maxsize=1)
def backdrops():
    sky = fx.wash(W, HZ + 40, [(0.02, 0.035, 0.1), (0.05, 0.1, 0.24), (0.14, 0.26, 0.48), (0.3, 0.46, 0.7)],
                  seed=51, blotch=0.14, cell=260, pos=[0, 0.4, 0.8, 1.0], blooms=5)
    sea = fx.wash(W, H - HZ, [(0.1, 0.19, 0.36), (0.03, 0.06, 0.14)], seed=52, blotch=0.12, cell=200, blooms=3)

    def stars(c):
        r = Rng(5)
        for i in range(520):
            x, y = r.u(0, W), r.u(0, HZ)
            z = r.u(0, 1) ** 3
            c.drawCircle(x, y, 0.6 + 1.6 * z, paint(STAR, 0.25 + 0.6 * z))
        for i in range(40):
            x, y = r.u(0, W), r.u(0, HZ * 0.8)
            c.drawCircle(x, y, 6, paint(STAR, 0.18, blur=5))
    star_layer = Layer(W, HZ, stars)

    def sea_strokes(c):
        r = Rng(6)
        for i in range(260):
            y = HZ + (r.u(0, 1) ** 1.6) * (H - HZ)
            x = r.u(-100, W)
            ln = r.u(60, 420) * (1 + (y - HZ) / 200)
            c.drawLine(x, y, x + ln, y, paint(PALE, r.u(0.03, 0.1), stroke=r.u(1, 3)))
    strokes = Layer(W, H, sea_strokes, blur=0.8)

    def islands(c):
        col = (0.02, 0.035, 0.08)
        for (cx, w_, top) in ((700, 330, 612), (1400, 330, 612)):
            pts = []
            r = Rng(int(cx))
            for k in range(13):
                u = k / 12
                x = cx - w_ / 2 + u * w_
                y = HZ + 8 - math.sin(u * math.pi) ** 0.7 * (HZ - top) + r.n(0, 3)
                pts.append((x, y))
            pts += [(cx + w_ / 2, HZ + 12), (cx - w_ / 2, HZ + 12)]
            c.drawPath(smooth_path(pts, close=True, tension=0.6), paint(col))
            c.drawPath(poly([(cx - 34, top + 6), (cx + 34, top + 6), (cx + 6, 372), (cx - 6, 372)]), paint(col))
    isl_arr = Layer(W, H, islands).arr
    isl = Layer(W, H, arr=fx.watercolorize(isl_arr, seed=8, edge=0.3, gran=0.15, wobble=2.5))

    def rock(c):
        col = (0.015, 0.025, 0.06)
        pts = [(0, H), (20, 800), (90, 752), (170, 736), (260, 738), (350, 760), (430, 820), (480, H)]
        c.drawPath(smooth_path(pts, close=True, tension=0.7), paint(col))
    rk = Layer(W, H, arr=fx.watercolorize(Layer(W, H, rock).arr, seed=9, edge=0.25, wobble=3))
    return sky, sea, star_layer, strokes, isl, rk


def word_star(c, x, y, word, a, t, glow=1.0, size=30, col=STAR, lab_a=None):
    if a <= 0.01:
        return
    tw = 0.8 + 0.2 * math.sin(t * 2.3 + x * 0.1)
    props.glow_dot(c, x, y, 3.6, col, a * tw * glow, halo=6, halo_a=0.7)
    la = a if lab_a is None else lab_a
    draw_text(c, word, x + 12, y - 12, 'serif', size, col, la * 0.95, align='left', tracking=0.06, glow=8,
              glow_col=BLUE, glow_a=0.5)


def arrow(c, x0, y0, x1, y1, k, col=(1, 1, 1), a=1.0, w=2.2, head=13, label='', inset=18):
    if k <= 0:
        return
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    sx, sy = x0 + ux * inset, y0 + uy * inset
    ex, ey = x1 - ux * inset, y1 - uy * inset
    cx, cy = lerp(sx, ex, k), lerp(sy, ey, k)
    c.drawLine(sx, sy, cx, cy, paint(col, a * 0.35, stroke=w * 4, blur=5, blend='plus'))
    c.drawLine(sx, sy, cx, cy, paint(col, a, stroke=w))
    nx, ny = -uy, ux
    hp = poly([(cx, cy), (cx - ux * head + nx * head * 0.5, cy - uy * head + ny * head * 0.5),
               (cx - ux * head - nx * head * 0.5, cy - uy * head - ny * head * 0.5)])
    c.drawPath(hp, paint(col, a))
    if label and k > 0.6:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        draw_text(c, label, mx + nx * 22, my + ny * 22 + 8, 'serif', 22, col, a * smooth(inv(0.6, 1.0, k)) * 0.85)


class Sea:
    DUR = 30.0
    NAME = 'sea'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        sky, sea, stars, strokes, isl, rk = backdrops()
        tilt = -50 * ease_in_out(inv(0.0, 13.5, t))       # sky drifts down as we look up
        partB = seg(t, 13.4, 14.6)
        sky.draw(c, 0, 0)
        stars.draw(c, 0, -tilt * 0.3, 0.9)
        # the great pale moon (GRIS circle)
        mx, my, mr = 1180, HZ + 10 - tilt * 0.15, 262
        c.drawCircle(mx, my, mr * 2.2, paint(PALE, 1.0, shader=radial(mx, my, mr * 2.2, [
            (PALE, 0.3), (PALE, 0.0)])))
        c.drawCircle(mx, my, mr, paint((0.66, 0.78, 0.92), 0.9))
        c.drawCircle(mx + 40, my - 30, mr * 0.8, paint((0.8, 0.88, 0.98), 0.3, blur=30))
        if t > 13.4:
            isl.draw(c, 0, 0, partB)
        if t < 14.6:
            self.part_a(c, t, tilt, 1 - partB)
        if t > 13.4:
            self.part_bc(c, t, partB)
            self.part_bc_front(c, t, partB)
        # sea + reflection of everything above the horizon
        sea.draw(c, 0, HZ)
        img = fr.to_float()
        from ..core import rs
        img = fx.reflect(img, HZ * rs(), strength=0.55, t=t, ripple=6, blur_px=2.2, tint=(0.75, 0.88, 1.0))
        fr.from_float(img)
        strokes.draw(c, 0, 0)
        rk.draw(c, 0, 0)
        # hero on the rock
        pose = ch.blend(ch.idle(t), ch.LOOK_UP, 0.75 * (1 - partB) + 0.15)
        ch.Hero(104, col=(0.01, 0.015, 0.035)).draw(
            c, 200, 736, pose, t, eyes=1.0, glow_col=BLUE, glow_r=10, glow_a=0.35,
            scarf=dict(length=1.2, width=0.065, cols=[RED, (0.25, 0.45, 1.0)], wind=0.45, speed=0.0, seed=5.0,
                       n=18, glow=6))
        fx.motes(c, t, n=60, seed=21, col=PALE, a=0.4, size=(1, 2.6), drift=(3, -8))
        img = fr.to_float()
        img = post.look_gris(img, idx, bloom_amt=0.55, paper_amt=0.55, vig=0.45, grain_amt=0.02, sat=1.05,
                             haze=0.1, thresh=0.6)
        img = fade_black(img, seg(t, 0.0, 1.8) * (1 - 0.8 * env(t, 13.4, 14.6, 0.6, 0.6)) *
                         (1 - seg(t, 28.8, 30.0)))
        fr.from_float(img)
        ui.letterbox(c, 1.0)
        self.overlays(c, t)

    # ------------------------------------------------------------ constellations
    def part_a(self, c, t, tilt, vis):
        oy = -tilt * 0.5
        r = Rng(12)
        pos = {}
        for i, (wd, (x, y, cl)) in enumerate(WORDS.items()):
            t0 = 0.8 + i * 0.3 + r.u(0, 0.3)
            sx, sy = x + r.u(-120, 120), HZ + r.u(40, 140)
            k = ease_out(inv(t0, t0 + 2.4, t))
            px_, py_ = lerp(sx, x, k), lerp(sy, y + oy, k)
            pos[wd] = (x, y + oy)
            if 0 < k < 1:
                tr = max(0.0, k - 0.25)
                tx, ty = lerp(sx, x, tr), lerp(sy, y + oy, tr)
                p = paint(STAR, 0.5 * vis, stroke=2, blur=1.5)
                p.setShader(linear(tx, ty, px_, py_, [(STAR, 0.0), (STAR, 0.6 * vis)]))
                c.drawLine(tx, ty, px_, py_, p)
                props.glow_dot(c, px_, py_, 3, STAR, vis, halo=5)
            elif k >= 1:
                word_star(c, px_, py_, wd, vis * smooth(inv(t0 + 2.2, t0 + 3.0, t)), t)
        la = smooth(inv(5.0, 8.0, t)) * vis
        for a_, b_ in CLUSTER_LINES:
            (x0, y0), (x1, y1) = pos[a_], pos[b_]
            c.drawLine(x0, y0, x1, y1, paint(PALE, 0.22 * la, stroke=1.2))
        for (a_, b_, ta, lab) in ARROWS:
            k = ease_in_out(inv(ta, ta + 0.9, t))
            (x0, y0), (x1, y1) = pos[a_], pos[b_]
            col = (1.0, 0.9, 0.7) if lab == '性别' else ((0.75, 0.9, 1.0) if lab == '首都' else (1, 1, 1))
            arrow(c, x0, y0, x1, y1, k, col=col, a=vis * 0.95, label=lab)
        ea = env(t, 11.6, 14.2, 0.8, 0.6) * vis
        if ea > 0:
            from ..text import draw_rich
            draw_rich(c, [('国王', 'serif_m', 54, 0, 0.1), ('  −  ', 'garamond_m', 54, 0, 0),
                          ('男人', 'serif_m', 54, 0, 0.1), ('  +  ', 'garamond_m', 54, 0, 0),
                          ('女人', 'serif_m', 54, 0, 0.1), ('  ≈  ', 'dejavu_serif', 46, 0, 0),
                          ('女王', 'serif_m', 54, 0, 0.1)], W / 2 + 40, 880, col=(1, 0.95, 0.85), a=ea,
                      glow=18, glow_col=(1.0, 0.8, 0.5), glow_a=0.55)

    # ------------------------------------------------------------ seq2seq & attention
    def part_bc(self, c, t, vis):
        """Things behind the islands: bridge arc, orb, threads."""
        br = env(t, 14.2, 22.8, 0.8, 0.8) * vis
        if br > 0:
            p = skia.Path()
            p.moveTo(700, 372)
            p.quadTo(1050, 150, 1400, 372)
            c.drawPath(p, paint(PALE, 0.55 * br, stroke=2))
            c.drawPath(p, paint(BLUE, 0.35 * br, stroke=10, blur=8, blend='plus'))
            m = skia.PathMeasure(p, False)
            L = m.getLength()
            # context vector orb (short sentence)
            k = inv(16.6, 17.8, t)
            if 16.2 < t < 18.2:
                pos, _ = m.getPosTan(ease_in_out(k) * L)
                ga = env(t, 16.2, 18.2, 0.3, 0.3)
                props.glow_dot(c, pos.x(), pos.y() - 6, 13, (0.9, 0.96, 1.0), ga, halo=4.5, halo_a=0.9)
            # overloaded orb (long sentence)
            if 20.2 < t < 22.8:
                grow = ease_out(inv(20.2, 21.2, t))
                rr = 12 + 26 * grow
                x, y = 700, 340 - 30 * grow
                jit = snoise(t * 40, 3) * 4 * seg(t, 21.2, 22.0)
                props.glow_dot(c, x + jit, y, rr, (0.9, 0.96, 1.0), env(t, 20.2, 22.8, 0.3, 0.4), halo=3.2,
                               halo_a=0.8)
                ck = seg(t, 21.3, 21.9)
                if ck > 0:
                    rng = Rng(4)
                    for j in range(8):
                        a = rng.u(0, 6.28)
                        L2 = rr * 1.6 * ck
                        c.drawLine(x, y, x + math.cos(a) * L2, y + math.sin(a) * L2,
                                   paint((0.02, 0.04, 0.1), 0.9, stroke=2.2))
                    draw_text(c, '瓶颈', x, y - rr - 30, 'serif_m', 30, (1, 0.8, 0.7), ck * env(t, 21.3, 22.8, 0.3, 0.4))
        # attention threads
        th = env(t, 22.2, 30.0, 0.8, 0.8) * vis
        if th > 0:
            conv = ease_in(inv(27.0, 29.0, t))
            xi = [280 + i * 62 for i in range(len(IN_TOK))]
            xo = [1520 + j * 70 for j in range(len(OUT_TOK))]
            yb = 668
            for j in range(len(OUT_TOK)):
                hi = env(t, 22.6 + j * 0.72, 22.6 + j * 0.72 + 1.3, 0.25, 0.4)
                for i in range(len(IN_TOK)):
                    wgt = ALIGN[j, i]
                    a = th * (0.1 + 0.9 * wgt) * (0.35 + 0.65 * hi) * (1 - conv)
                    if a < 0.02:
                        continue
                    x0, x1 = xo[j], xi[i]
                    cy = yb - 260 - abs(x1 - x0) * 0.18
                    p = skia.Path()
                    p.moveTo(x0, yb - 26)
                    p.quadTo((x0 + x1) / 2, cy, x1, yb - 26)
                    col = (1.0, 0.86, 0.6) if hi > 0.3 and wgt > 0.3 else PALE
                    c.drawPath(p, paint(col, a, stroke=0.8 + 3.5 * wgt))
                    if wgt > 0.3:
                        c.drawPath(p, paint(col, a * 0.5, stroke=10 * wgt + 2, blur=7, blend='plus'))
            if conv > 0:
                # light gathers into one star high above
                sx, sy = W / 2, 210
                for j in range(len(OUT_TOK)):
                    for i in range(0, len(IN_TOK), 2):
                        u = conv
                        x0 = lerp((xo[j] + xi[i]) / 2, sx, u)
                        y0 = lerp(yb - 300, sy, u)
                        props.glow_dot(c, x0, y0, 2.5, (1, 0.93, 0.8), th * (1 - u * 0.5), halo=4)
                props.glow_dot(c, sx, sy, 6 + 16 * conv, (1, 0.95, 0.85), th * conv, halo=6, halo_a=0.9)

    def part_bc_front(self, c, t, vis):
        """Tokens on the water in front of the islands."""
        # encoder / decoder glow + labels
        eg = env(t, 14.3, 22.9, 0.5, 0.6) * vis
        if eg > 0:
            for (x, cn, en, al) in ((650, '编码器', 'ENCODER', 'right'), (1450, '解码器', 'DECODER', 'left')):
                draw_text(c, cn, x, 470, 'serif_m', 32, STAR, eg, align=al, tracking=0.2, glow=10, glow_col=BLUE)
                draw_text(c, en, x, 505, 'cinzel', 18, PALE, eg * 0.8, align=al, tracking=0.3)
            charge = seg(t, 14.4, 16.4) * (1 - seg(t, 16.6, 17.4)) + seg(t, 19.6, 21.0) * (1 - seg(t, 21.8, 22.6))
            c.drawCircle(700, 470, 150, paint(BLUE, 0.4 * charge * eg, blur=60, blend='plus'))
            dec = seg(t, 17.6, 18.2) * (1 - seg(t, 19.2, 19.8))
            c.drawCircle(1400, 470, 150, paint(BLUE, 0.4 * dec * eg, blur=60, blend='plus'))
        yb = 668
        # short sentence into the encoder
        for i, tok in enumerate(IN_TOK):
            t0 = 14.3 + i * 0.28
            k = ease_in(inv(t0, t0 + 1.4, t))
            if t < 22.0:
                if 0 <= k < 1 and t >= t0 - 0.6:
                    x = lerp(280 + i * 62, 700, k)
                    a = env(t, t0 - 0.6, t0 + 1.4, 0.4, 0.15) * vis * (1 - smooth(inv(0.75, 1.0, k)))
                    draw_text(c, tok, x, yb, 'serif_m', 34, STAR, a, glow=10, glow_col=BLUE)
            if t >= 22.0:
                a = smooth(inv(22.0, 22.8, t)) * vis * (1 - seg(t, 28.0, 29.2))
                draw_text(c, tok, 280 + i * 62, yb, 'serif_m', 34, STAR, a, glow=10, glow_col=BLUE)
        # decoder outputs
        for j, tok in enumerate(OUT_TOK):
            t0 = 17.9 + j * 0.24
            k = ease_out(inv(t0, t0 + 0.9, t))
            if t < 22.0:
                a = smooth(inv(t0, t0 + 0.4, t)) * (1 - seg(t, 19.2, 19.8)) * vis
                if a > 0:
                    x = lerp(1400, 1520 + j * 70, k)
                    draw_text(c, tok, x, yb, 'garamond_m', 36, STAR, a, glow=10, glow_col=BLUE)
            else:
                a = smooth(inv(22.0, 22.8, t)) * vis * (1 - seg(t, 28.0, 29.2))
                hi = env(t, 22.6 + j * 0.72, 22.6 + j * 0.72 + 1.3, 0.25, 0.4)
                draw_text(c, tok, 1520 + j * 70, yb, 'garamond_m', 36, (1, lerp(0.95, 0.85, hi), lerp(0.98, 0.6, hi)),
                          a, glow=10 + 8 * hi, glow_col=(1, 0.8, 0.5) if hi > 0.2 else BLUE)
        # long sentence overloading the encoder
        for i, tok in enumerate(LONG_TOK):
            t0 = 19.5 + i * 0.09
            k = ease_in(inv(t0, t0 + 1.1, t))
            if 0 <= k < 1 and t >= t0 - 0.5:
                x = lerp(200 + i * 30, 700, k)
                a = env(t, t0 - 0.5, t0 + 1.1, 0.3, 0.15) * vis * (1 - smooth(inv(0.7, 1.0, k)))
                draw_text(c, tok, x, yb + (i % 2) * 22, 'serif_m', 28, STAR, a, glow=8, glow_col=BLUE)

    def overlays(self, c, t):
        ui.chapter_card(c, t, 0.8, 5.6, '第 四 章', '意义之海', 'CHAPTER IV · THE SEA OF MEANING', '2013 — 2014')
        ui.subtitle(c, t, 5.8, 9.4, '2013 · word2vec：每一个词，都有了自己的坐标。',
                    '2013 · word2vec: every word found its own coordinates.')
        ui.subtitle(c, t, 9.7, 13.6, '意义，变成了空间里的方向与距离。', 'Meaning became direction and distance in space.')
        ui.subtitle(c, t, 14.6, 18.2, '2014 · Seq2Seq：把一整句话，压进一个向量。',
                    '2014 · Seq2Seq: squeeze a whole sentence into a single vector.')
        ui.subtitle(c, t, 18.5, 21.9, '句子越长，就越装不下。', 'The longer the sentence, the less it can hold.')
        ui.subtitle(c, t, 22.3, 26.4, '2014 · 注意力：每译一个词，都回头看最相关的那几个。',
                    '2014 · Attention: for every word, look back at the ones that matter most.')
        ui.subtitle(c, t, 26.7, 29.6, '……如果，只需要注意力呢？', '… What if attention was all you needed?', fo=0.8)
