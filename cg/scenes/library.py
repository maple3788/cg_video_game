"""Chapter VIII — The Endless Library (2020–): retrieval-augmented generation, then a montage of
the frontier (chain-of-thought, mixture of experts, agents, multimodality, reasoning)."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, Rng, clamp, ease_in, ease_in_out, ease_out, ease_out_back, env, inv,
                    lerp, lerp3, paint, poly, radial, linear, smooth, seg, snoise, smooth_path)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width
from .common import fade_black, fade_white

INK = (0.2, 0.26, 0.3)
TEAL = (0.3, 0.62, 0.64)
GOLD = (0.98, 0.78, 0.4)
IVORY = (0.98, 0.95, 0.88)
BOOK_COLS = [(0.85, 0.4, 0.42), (0.3, 0.6, 0.62), (0.95, 0.72, 0.36), (0.5, 0.45, 0.78), (0.4, 0.66, 0.46),
             (0.9, 0.55, 0.35), (0.25, 0.4, 0.62)]
SCARF = [(0.93, 0.25, 0.3), (1.0, 0.78, 0.3), (0.3, 0.75, 0.75), (0.55, 0.45, 0.95)]

SHELVES = [(380, 300, 520), (1020, 250, 560), (700, 440, 480), (1420, 400, 440), (260, 560, 380),
           (1150, 590, 400)]   # (x, y, width)
TARGETS = [(1, 7, 0.91, '[1] 报销制度 v3'), (3, 4, 0.87, '[2] 财务 FAQ'), (5, 9, 0.82, '[3] 审批流程')]
STEPS = [('① 提问', 'QUERY', 4.8), ('② 向量化', 'EMBED', 6.0), ('③ 检索', 'RETRIEVE', 7.2),
         ('④ 增强', 'AUGMENT', 8.6), ('⑤ 生成', 'GENERATE', 10.2)]


def book_rect(si, bi):
    x, y, w = SHELVES[si]
    n = int(w / 26)
    bx = x - w / 2 + 8 + bi * 26
    r = Rng(si * 100 + bi)
    h = r.u(56, 80)
    return bx, y - h, 20, h


@functools.lru_cache(maxsize=1)
def library_layers():
    bg = fx.wash(W, H, [(0.34, 0.6, 0.66), (0.66, 0.8, 0.8), (0.93, 0.9, 0.82)], seed=81, blotch=0.12, cell=240,
                 blooms=5)

    def arches(c):
        for k, (x, w, h, a) in enumerate([(200, 300, 900, 0.5), (620, 360, 1000, 0.4), (1080, 330, 950, 0.45),
                                          (1520, 380, 1020, 0.4), (1880, 300, 880, 0.5)]):
            props.arch(c, x, H + 40, w, h, 46, (0.98, 0.96, 0.9), a)
            props.arch(c, x + 12, H + 40, w, h, 10, (0.5, 0.7, 0.72), a * 0.5)

    ar = Layer(W, H, arches, blur=2.0)

    def shelves(c):
        for si, (x, y, w) in enumerate(SHELVES):
            c.drawRect(skia.Rect.MakeXYWH(x - w / 2, y, w, 10), paint((0.45, 0.36, 0.3)))
            c.drawRect(skia.Rect.MakeXYWH(x - w / 2, y + 10, w, 6), paint((0.3, 0.24, 0.2), 0.6))
            n = int(w / 26)
            for bi in range(n):
                bx, by, bw, bh = book_rect(si, bi)
                col = BOOK_COLS[(si * 3 + bi) % len(BOOK_COLS)]
                r = Rng(si * 7 + bi)
                tilt = r.n(0, 2) if bi % 5 == 4 else 0
                c.save()
                c.translate(bx + bw / 2, by + bh)
                c.rotate(tilt)
                c.drawRect(skia.Rect.MakeXYWH(-bw / 2, -bh, bw, bh), paint(col))
                c.drawRect(skia.Rect.MakeXYWH(-bw / 2, -bh * 0.8, bw, 3), paint((1, 1, 1), 0.5))
                c.restore()
    sh = Layer(W, H, arr=fx.watercolorize(Layer(W, H, shelves).arr, seed=82, edge=0.3, gran=0.15, wobble=1.2))
    return bg, ar, sh


def flying_book(c, x, y, s, t, col, ph=0.0):
    flap = math.sin(t * 7 + ph)
    c.save()
    c.translate(x, y)
    for side in (-1, 1):
        p = poly([(0, 0), (side * s, -s * 0.3 * flap - s * 0.1), (side * s * 0.95, s * 0.5 - s * 0.3 * flap),
                  (0, s * 0.55)])
        c.drawPath(p, paint(col if side < 0 else IVORY, 0.95))
    c.drawLine(0, 0, 0, s * 0.55, paint((0.3, 0.24, 0.2), stroke=2))
    c.restore()


class Library:
    DUR = 26.0
    NAME = 'library'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        if t < 14.0:
            self.rag(c, t)
        else:
            self.montage(c, t)
        img = fr.to_float()
        img = post.look_gris(img, idx, bloom_amt=0.32, paper_amt=0.85, vig=0.3, sat=1.04, thresh=0.8)
        img = fade_white(img, (1 - seg(t, 0.0, 1.4)) * 0.9, (1, 0.97, 0.9))
        for tc in (14.0, 16.4, 18.8, 21.2, 23.6):
            img = fade_white(img, math.exp(-((t - tc) / 0.12) ** 2) * 0.7, (1, 0.98, 0.94))
        img = fade_black(img, 1 - seg(t, 25.2, 26.0))
        fr.from_float(img)
        if t < 14.0:
            self.rag_overlays(c, t)
        else:
            self.montage_overlays(c, t)

    # ------------------------------------------------------------------ RAG
    def rag(self, c, t):
        bg, ar, sh = library_layers()
        bg.draw(c, 0, 0)
        ar.draw(c, 0, 0)
        sh.draw(c, 0, 0)
        r = Rng(3)
        for k in range(7):
            ph = r.u(0, 6.28)
            x = (r.u(0, W) + t * r.u(40, 90)) % (W + 200) - 100
            y = r.u(140, 520) + math.sin(t * 0.8 + ph) * 20
            flying_book(c, x, y, r.u(14, 24), t, BOOK_COLS[k % len(BOOK_COLS)], ph)
        # platform + hero with the question lantern
        c.drawPath(smooth_path([(90, 900), (180, 880), (340, 876), (430, 890), (380, 930), (140, 932)], close=True),
                   paint((0.9, 0.86, 0.78)))
        c.drawRect(skia.Rect.MakeXYWH(120, 920, 290, 200), paint((0.72, 0.8, 0.8), 0.35, blur=10))
        pose = ch.blend(ch.idle(t), ch.REACH, 0.6)
        pose = ch.Pose(py=0.455, torso=0.05, head=-0.2, th=(0.1, -0.1), kn=(0.1, 0.05), sh=(1.9, 0.1),
                       el=(0.2, 0.3)) if t > 4.4 else pose
        hero = ch.Hero(118, col=(0.12, 0.14, 0.2))
        hero.draw(c, 260, 880, pose, t, eyes=1.0, glow_col=(1, 0.95, 0.8), glow_r=8, glow_a=0.35,
                  scarf=dict(length=1.8, width=0.07, cols=SCARF, wind=0.4, speed=0.0, seed=10.0, n=22, glow=4))
        lx, ly = ch.Hero.hand_pos(260, 880, pose, 118, 1, 0)
        c.drawCircle(lx, ly - 14, 34, paint(GOLD, 0.5, blur=18, blend='plus'))
        props.glow_dot(c, lx, ly - 14, 9, GOLD, 1.0, halo=4)
        # ① query bubble
        qa = env(t, 4.8, 6.4, 0.3, 0.3)
        qx, qy = 470, 700
        if qa > 0:
            tw = text_width('公司的报销流程是怎样的？', 'serif_m', 32, 0.04) + 50
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(qx - 20, qy - 44, tw, 64), 20, 20),
                        paint(IVORY, 0.95 * qa))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(qx - 20, qy - 44, tw, 64), 20, 20),
                        paint(TEAL, 0.8 * qa, stroke=2))
            draw_text(c, '公司的报销流程是怎样的？', qx + 5, qy, 'serif_m', 32, INK, qa, align='left', tracking=0.04)
        # ② embedding vector
        va = env(t, 6.0, 8.0, 0.3, 0.4)
        vx, vy = 600, 470
        if va > 0:
            r2 = Rng(8)
            k = ease_out(inv(6.0, 6.6, t))
            for q in range(12):
                v = r2.u(-1, 1)
                col = lerp3((0.35, 0.55, 0.95), (0.95, 0.45, 0.45), (v + 1) / 2)
                c.drawRect(skia.Rect.MakeXYWH(vx - 18, vy + q * 22 * k, 36, 18), paint(col, va))
            draw_text(c, '[0.12, −0.83, 0.47, …]', vx + 40, vy + 140, 'garamond_m', 26, INK, va, align='left')
        # ③ retrieve: beams to three books
        ra = env(t, 7.2, 13.8, 0.3, 0.5)
        if ra > 0:
            sx, sy = vx, vy + 60
            dim = smooth(inv(7.2, 7.8, t))
            c.drawRect(skia.Rect.MakeWH(W, H), paint((0.25, 0.35, 0.4), 0.18 * dim * ra))
            for n_, (si, bi, sc, lab) in enumerate(TARGETS):
                bx, by, bw, bh = book_rect(si, bi)
                k = ease_out(inv(7.2 + n_ * 0.2, 7.8 + n_ * 0.2, t))
                ex, ey = lerp(sx, bx + bw / 2, k), lerp(sy, by + bh / 2, k)
                c.drawLine(sx, sy, ex, ey, paint(GOLD, 0.9 * ra, stroke=2.5))
                c.drawLine(sx, sy, ex, ey, paint(GOLD, 0.4 * ra, stroke=12, blur=8, blend='plus'))
                if k > 0.95:
                    c.drawRect(skia.Rect.MakeXYWH(bx - 4, by - 4, bw + 8, bh + 8), paint(GOLD, 0.6 * ra, blur=8,
                                                                                          blend='plus'))
                    draw_text(c, f'{sc:.2f}', bx + bw / 2, by - 14, 'garamond_m', 28, (0.6, 0.35, 0.05), ra,
                              glow=8, glow_col=(1, 1, 1), glow_a=0.7)
        # ④ augment: pages fly into the context window
        ca = env(t, 8.6, 13.8, 0.4, 0.5)
        cx, cy, cw, chh = 1400, 690, 600, 300
        if ca > 0:
            rect = skia.Rect.MakeXYWH(cx - cw / 2, cy - chh / 2, cw, chh)
            c.drawRRect(skia.RRect.MakeRectXY(rect, 12, 12), paint(IVORY, 0.92 * ca))
            c.drawRRect(skia.RRect.MakeRectXY(rect, 12, 12), paint(TEAL, 0.9 * ca, stroke=2))
            draw_text(c, '上下文窗口  CONTEXT', cx - cw / 2 + 22, cy - chh / 2 + 36, 'serif_m', 24, TEAL, ca,
                      align='left', tracking=0.1)
            for n_, (si, bi, sc, lab) in enumerate(TARGETS):
                bx, by, bw, bh = book_rect(si, bi)
                k = ease_in_out(inv(8.7 + n_ * 0.3, 9.6 + n_ * 0.3, t))
                tx, ty = cx - cw / 2 + 30, cy - chh / 2 + 70 + n_ * 44
                if 0 < k < 1:
                    px_, py_ = lerp(bx, tx, k), lerp(by, ty, k) - math.sin(k * math.pi) * 120
                    c.save()
                    c.translate(px_, py_)
                    c.rotate((1 - k) * 40)
                    c.drawRect(skia.Rect.MakeXYWH(0, 0, 40, 52), paint((1, 1, 1), 0.95))
                    c.drawRect(skia.Rect.MakeXYWH(0, 0, 40, 52), paint(TEAL, 0.6, stroke=1))
                    c.restore()
                if k >= 1:
                    draw_text(c, lab, tx, ty + 24, 'serif_m', 26, INK, ca, align='left')
            if t > 9.8:
                draw_text(c, '+ 问题：公司的报销流程是怎样的？', cx - cw / 2 + 30, cy - chh / 2 + 70 + 3 * 44 + 24,
                          'serif', 24, INK, ca * smooth(inv(9.8, 10.2, t)), align='left')
        # ⑤ generate
        ga = env(t, 10.2, 13.8, 0.3, 0.5)
        if ga > 0:
            ans = '提交发票 → 主管审批 → 财务打款（约 5 个工作日） [1][3]'
            k = clamp((t - 10.3) * 14, 0, len(ans))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(cx - cw / 2 - 20, cy + chh / 2 + 24, cw + 40, 70), 14, 14),
                        paint((0.14, 0.2, 0.24), 0.85 * ga))
            draw_text(c, ans, cx, cy + chh / 2 + 70, 'serif_m', 27, (1, 0.96, 0.86), ga, reveal=k)

    def rag_overlays(self, c, t):
        ui.chapter_card(c, t, 0.6, 4.6, '第 八 章', '无尽书库', 'CHAPTER VIII · THE ENDLESS LIBRARY', '2020 — ',
                        dim=0.45)
        sa = env(t, 4.6, 13.9, 0.4, 0.5)
        if sa > 0:
            total = sum(text_width(s_, 'serif_m', 26, 0.08) for (s_, e_, t_) in STEPS) + 4 * 90
            x = W / 2 - total / 2
            for i, (cn, en, ts) in enumerate(STEPS):
                on = smooth(inv(ts, ts + 0.4, t))
                col = lerp3((0.45, 0.55, 0.58), (0.72, 0.4, 0.05), on)
                wd = draw_text(c, cn, x, 150, 'serif_m', 26, col, sa, align='left', tracking=0.08,
                               glow=10 * on, glow_col=(1, 1, 1), glow_a=0.6)
                draw_text(c, en, x + wd / 2, 178, 'cinzel', 14, col, sa * 0.85, tracking=0.2)
                x += wd
                if i < len(STEPS) - 1:
                    draw_text(c, '→', x + 45, 150, 'dejavu', 24, (0.45, 0.55, 0.58), sa)
                    x += 90
        ui.subtitle(c, t, 4.8, 8.8, '它学会了在回答之前，先去翻书。', 'It learned to look things up before answering.',
                    style='float', y=985)
        ui.skill_popup(c, t, 10.6, 13.9, '检索增强生成', 'RAG', '先检索，再生成：让每个回答都有据可查', y=330,
                       accent=(0.85, 0.55, 0.15))

    # ------------------------------------------------------------------ montage
    def montage(self, c, t):
        k = int((t - 14.0) // 2.4)
        lt = (t - 14.0) - k * 2.4
        palettes = [[(0.98, 0.84, 0.72), (0.95, 0.62, 0.62)], [(0.72, 0.84, 0.95), (0.5, 0.52, 0.85)],
                    [(0.82, 0.93, 0.84), (0.36, 0.64, 0.6)], [(0.98, 0.9, 0.7), (0.9, 0.5, 0.4)],
                    [(0.3, 0.26, 0.5), (0.12, 0.1, 0.24)]]
        pa = palettes[min(k, 4)]
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=linear(0, 0, 0, H, [pa[0], pa[1]])))
        c.drawCircle(W / 2, H / 2, 900, paint((1, 1, 1), shader=radial(W / 2, H / 2, 900, [
            ((1, 1, 1), 0.25), ((1, 1, 1), 0.0)])))
        getattr(self, f'mont_{min(k, 4)}')(c, t, lt)

    def mont_0(self, c, t, lt):   # chain of thought: a staircase of reasoning steps
        steps = ['① 设未知数', '② 列出方程', '③ 逐步求解', '∴ 得出答案']
        for i, s_ in enumerate(steps):
            x, y = 420 + i * 330, 840 - i * 150
            a = smooth(inv(0.1 + i * 0.35, 0.4 + i * 0.35, lt))
            c.drawRect(skia.Rect.MakeXYWH(x - 150, y, 300, 40), paint(IVORY, a))
            c.drawRect(skia.Rect.MakeXYWH(x - 150, y + 40, 300, 400), paint((0.9, 0.6, 0.6), 0.45 * a))
            c.drawRect(skia.Rect.MakeXYWH(x - 150, y - 6, 300, 6), paint(GOLD, a))
            draw_text(c, s_, x, y - 30, 'serif_m', 36, (0.35, 0.12, 0.2), a)
        u = clamp(lt / 2.2, 0, 1)
        i = min(3, int(u * 4))
        f = u * 4 - i
        x = 420 + i * 330 + f * 330 * 0.6
        y = 840 - i * 150 - (f > 0.6) * 150 * ((f - 0.6) / 0.4)
        ch.Hero(110, col=(0.25, 0.1, 0.18)).draw(c, x - 60, y, ch.walk(lt * 1.4), t, eyes=1.0,
                                                 scarf=dict(length=1.6, width=0.07, cols=SCARF, wind=0.5, speed=0.6,
                                                            seed=11.0, n=20))

    def mont_1(self, c, t, lt):   # mixture of experts
        rx, ry = 420, 560
        props.glow_dot(c, rx, ry, 26, (1, 1, 1), 1.0, halo=4)
        draw_text(c, '路由器', rx, ry + 80, 'serif_m', 34, (0.15, 0.15, 0.4))
        active = (2, 5)
        for i in range(8):
            ang = -0.9 + i * 1.8 / 7
            ex, ey = rx + 900 * math.cos(ang) * 0.95, ry + 440 * math.sin(ang)
            on = i in active
            props.arch(c, ex, ey + 70, 110, 150, 16, (1, 0.9, 0.55) if on else (0.9, 0.92, 1.0), 0.95)
            draw_text(c, f'专家 {i + 1}', ex, ey + 104, 'serif_m', 22, (0.15, 0.15, 0.4), 0.9)
            if on:
                k = ease_out(inv(0.2, 0.8, lt))
                c.drawLine(rx, ry, lerp(rx, ex, k), lerp(ry, ey, k), paint(GOLD, 0.95, stroke=5))
                c.drawLine(rx, ry, lerp(rx, ex, k), lerp(ry, ey, k), paint(GOLD, 0.5, stroke=22, blur=12, blend='plus'))
                c.drawCircle(ex, ey + 20, 70, paint(GOLD, 0.45 * k, blur=26, blend='plus'))
            else:
                c.drawLine(rx, ry, ex, ey, paint((1, 1, 1), 0.25, stroke=1.2))

    def mont_2(self, c, t, lt):   # agents with tools: think -> act -> observe
        cx, cy = W / 2, 560
        c.drawCircle(cx, cy, 250, paint((0.2, 0.4, 0.38), 0.8, stroke=3))
        labels = ['思考', '行动', '观察']
        for i, lb in enumerate(labels):
            a = -math.pi / 2 + i * 2 * math.pi / 3 + lt * 0.6
            x, y = cx + math.cos(a) * 250, cy + math.sin(a) * 250
            c.drawCircle(x, y, 58, paint(IVORY))
            c.drawCircle(x, y, 58, paint((0.2, 0.4, 0.38), stroke=3))
            draw_text(c, lb, x, y + 12, 'serif_m', 34, (0.15, 0.3, 0.28))
        tools = [('搜索', '⌕'), ('代码', '</>'), ('邮件', '✉'), ('日历', '▦')]
        for i, (nm, ic) in enumerate(tools):
            a = i * math.pi / 2 - lt * 0.8 + 0.4
            x, y = cx + math.cos(a) * 470, cy + math.sin(a) * 300
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x - 70, y - 50, 140, 100), 18, 18), paint((1, 1, 1), 0.9))
            draw_text(c, ic, x, y + 4, 'dejavu', 38, (0.2, 0.4, 0.38))
            draw_text(c, nm, x, y + 38, 'serif_m', 22, (0.2, 0.4, 0.38))
        props.glow_dot(c, cx, cy, 20, GOLD, 1.0, halo=5)

    def mont_3(self, c, t, lt):   # multimodal: the eye opens
        cx, cy = W / 2, 540
        op = ease_out(inv(0.1, 0.9, lt))
        p = skia.Path()
        p.moveTo(cx - 360, cy)
        p.quadTo(cx, cy - 300 * op, cx + 360, cy)
        p.quadTo(cx, cy + 300 * op, cx - 360, cy)
        c.drawPath(p, paint(IVORY))
        c.save()
        c.clipPath(p, True)
        c.drawCircle(cx, cy, 150, paint((0.25, 0.5, 0.6)))
        c.drawCircle(cx, cy, 70, paint((0.08, 0.08, 0.12)))
        c.drawCircle(cx + 40, cy - 40, 24, paint((1, 1, 1), 0.9))
        c.restore()
        c.drawPath(p, paint((0.5, 0.2, 0.15), stroke=6))
        for i in range(6):
            u = ((lt * 0.6 + i / 6) % 1.0)
            x = lerp(80, cx - 330, u)
            y = 300 + i * 90
            if i % 2 == 0:
                c.drawRect(skia.Rect.MakeXYWH(x - 50, y - 36, 100, 72), paint((1, 1, 1), 0.95))
                c.drawPath(poly([(x - 40, y + 28), (x - 10, y - 12), (x + 10, y + 10), (x + 22, y - 2), (x + 40, y + 28)]),
                           paint((0.4, 0.6, 0.5)))
                c.drawCircle(x + 26, y - 20, 8, paint((0.95, 0.7, 0.3)))
            else:
                pts = [(x - 60 + q * 6, y + math.sin(q * 0.9 + t * 8) * 22 * math.sin(q / 20 * math.pi)) for q in range(21)]
                c.drawPath(smooth_path(pts), paint((0.5, 0.2, 0.15), 0.9, stroke=3))

    def mont_4(self, c, t, lt):   # reasoning: an hourglass of thought
        cx, cy = W / 2, 540
        c.drawPath(poly([(cx - 180, cy - 300), (cx + 180, cy - 300), (cx + 20, cy), (cx + 180, cy + 300),
                         (cx - 180, cy + 300), (cx - 20, cy)]), paint((0.85, 0.8, 1.0), 0.9, stroke=4))
        r = Rng(7)
        glyphs = list('若设则因故且即得所以验证')
        for q in range(60):
            u = ((lt * 0.35 + r.u(0, 1)) % 1.0)
            if u < 0.5:
                y = lerp(cy - 280, cy, u * 2)
                w_ = lerp(160, 16, u * 2)
            else:
                y = lerp(cy, cy + 280, (u - 0.5) * 2)
                w_ = lerp(16, 160, (u - 0.5) * 2)
            x = cx + r.u(-1, 1) * w_
            draw_text(c, glyphs[q % len(glyphs)], x, y, 'serif_m', 22, GOLD, 0.9, glow=8, glow_col=GOLD)
        dots = '.' * (1 + int(lt * 3) % 3)
        draw_text(c, '思考中' + dots, cx + 260, cy + 10, 'serif_m', 40, (0.95, 0.9, 1.0), 0.95, align='left')

    def montage_overlays(self, c, t):
        items = [('思维链', 'CHAIN-OF-THOUGHT · 2022', '一步一步地想'),
                 ('混合专家', 'MIXTURE OF EXPERTS', '每个词，只唤醒最合适的专家'),
                 ('智能体', 'AGENTS · TOOL USE', '会思考，也会动手'),
                 ('多模态', 'MULTIMODAL', '看见图像，听见声音'),
                 ('推理模型', 'REASONING · 2024', '回答之前，先想得更久')]
        k = min(4, int((t - 14.0) // 2.4))
        lt = (t - 14.0) - k * 2.4
        cn, en, sub = items[k]
        a = env(lt, 0.1, 2.35, 0.25, 0.25)
        dark = k == 4
        col = (1, 0.96, 0.9) if dark else (0.22, 0.12, 0.25)
        draw_text(c, cn, 120, 920, 'serif_b', 64, col, a, align='left', tracking=0.1,
                  glow=10, glow_col=(1, 1, 1) if not dark else GOLD, glow_a=0.4)
        draw_text(c, en, 124, 964, 'cinzel_b', 20, col, a * 0.85, align='left', tracking=0.25)
        draw_text(c, sub, 124, 1010, 'serif', 28, col, a * 0.9, align='left', tracking=0.06)
