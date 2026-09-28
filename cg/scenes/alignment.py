"""Chapter VII — Alignment (2022): human feedback lanterns (RLHF) calm the storm; the giant
speaks to us for the first time."""
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
from .giants import statue, STONE, SHADE

LAMP = (1.0, 0.78, 0.42)
PAIRS = [  # (t, good, bad, good_is_left)
    (4.6, '水开后煮 8 分钟，再过一遍冷水。', '鸡蛋是一种水果。', True),
    (7.0, '我不确定，但我可以帮你一起查证。', '当然！长城在月球上清晰可见。', False),
    (9.4, '别灰心，我们一步一步来。', '这个问题也太蠢了吧。', True),
]
QUESTIONS = ['怎样煮一个溏心蛋？', '长城能从月球上看到吗？', '我考砸了，怎么办？']


@functools.lru_cache(maxsize=1)
def hills_layer():
    def draw(c):
        for (y0, amp, col, seed) in [(820, 40, (0.34, 0.24, 0.42), 1), (900, 30, (0.2, 0.13, 0.26), 2),
                                     (980, 24, (0.1, 0.06, 0.14), 3)]:
            g = props.ground_fn(y0, amp, 400, 90 + seed, amp2=4)
            props.draw_ground(c, -100, W + 100, g, col, bottom=H + 20)
    return Layer(W, H, arr=fx.watercolorize(Layer(W, H, draw).arr, seed=91, edge=0.35, wobble=2.0))


def people(c, t, lit_t0=1.0, lit_t1=9.0):
    r = Rng(33)
    lamps = []
    for row, (y0, amp, seed, sc, n) in enumerate([(820, 40, 91, 0.7, 16), (900, 30, 92, 0.9, 13),
                                                   (980, 24, 93, 1.15, 10)]):
        g = props.ground_fn(y0, amp, 400, seed, amp2=4)
        for k in range(n):
            x = 60 + (k + r.u(-0.3, 0.3)) * (W - 120) / (n - 1)
            if row == 2 and 700 < x < 1250:
                continue
            gy = float(g(x)) + 4
            h = 46 * sc * r.u(0.85, 1.15)
            col = [(0.3, 0.2, 0.38), (0.17, 0.11, 0.22), (0.08, 0.05, 0.1)][row]
            lt = lerp(lit_t0, lit_t1, x / W) + r.u(-0.4, 0.4)
            up = smooth(inv(lt, lt + 0.8, t))
            # body
            c.drawCircle(x, gy - h * 0.86, h * 0.13, paint(col))
            c.drawPath(poly([(x - h * 0.12, gy - h * 0.72), (x + h * 0.12, gy - h * 0.72), (x + h * 0.2, gy),
                             (x - h * 0.2, gy)]), paint(col))
            # raised arm + lantern
            ax, ay = x + h * 0.08, gy - h * 0.7
            hx, hy = ax + h * lerp(0.2, 0.08, up), ay - h * lerp(-0.05, 0.42, up)
            c.drawLine(ax, ay, hx, hy, paint(col, stroke=h * 0.07))
            ly = hy + h * 0.1
            c.drawLine(hx, hy, hx, ly, paint(col, stroke=1.2))
            fl = 0.85 + 0.15 * math.sin(t * 6 + k * 1.3 + row)
            if up > 0.02:
                c.drawCircle(hx, ly + 4, h * 0.9, paint(LAMP, 0.35 * up * fl, blur=h * 0.4, blend='plus'))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(hx - h * 0.06, ly, h * 0.12, h * 0.16), 2, 2),
                        paint(lerp3((0.4, 0.35, 0.3), (1, 0.9, 0.65), up)))
            lamps.append((hx, ly + h * 0.08, up))
    return lamps


def answer_card(c, x, y, text, a, good, verdict, w=520):
    if a <= 0.01:
        return
    h = 92
    rect = skia.Rect.MakeXYWH(x - w / 2, y - h / 2, w, h)
    glow = (1, 0.85, 0.5) if good else (0.9, 0.4, 0.45)
    if verdict > 0:
        c.drawRRect(skia.RRect.MakeRectXY(rect.makeOutset(10, 10), 18, 18),
                    paint(glow, 0.5 * verdict * a, blur=16, blend='plus'))
    c.drawRRect(skia.RRect.MakeRectXY(rect, 14, 14), paint((0.99, 0.97, 0.93), 0.9 * a))
    c.drawRRect(skia.RRect.MakeRectXY(rect, 14, 14), paint((0.55, 0.45, 0.6), 0.6 * a, stroke=1.5))
    draw_text(c, text, x - w / 2 + 28, y + 11, 'serif_m', 27, (0.25, 0.16, 0.3), a, align='left')
    if verdict > 0:
        mark = '✓' if good else '✗'
        draw_text(c, mark, x + w / 2 - 36, y + 14, 'dejavu', 40, (0.9, 0.6, 0.15) if good else (0.85, 0.2, 0.25),
                  a * verdict)


class Alignment:
    DUR = 26.0
    NAME = 'alignment'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        dawn = smooth(inv(10.5, 15.0, t))
        top = lerp3((0.2, 0.15, 0.3), (0.6, 0.62, 0.85), dawn)
        mid = lerp3((0.36, 0.26, 0.44), (0.98, 0.72, 0.68), dawn)
        low = lerp3((0.5, 0.36, 0.5), (1.0, 0.88, 0.66), dawn)
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1, 1, 1), shader=linear(0, 0, 0, H, [top, mid, low],
                                                                          [0, 0.55, 0.85])))
        # sun rising behind the giant
        sy = lerp(900, 520, dawn)
        c.drawCircle(1480, sy, 700, paint((1, 1, 1), shader=radial(1480, sy, 700, [
            ((1, 0.9, 0.7), 0.6 * dawn), ((1, 0.9, 0.7), 0.0)])))
        c.drawCircle(1480, sy, 170, paint((1, 0.95, 0.84), 0.9 * dawn))
        # storm clouds parting
        storm = 1 - smooth(inv(8.0, 13.0, t))
        if storm > 0:
            r = Rng(4)
            for k in range(14):
                x = r.u(-100, W + 100) + (t * 20 * (1 if k % 2 else -1)) + (1 - storm) * (600 if x_side(k) else -600)
                y = r.u(40, 380)
                c.drawOval(skia.Rect.MakeXYWH(x - 260, y - 70, 520, 140), paint((0.18, 0.12, 0.24), 0.55 * storm,
                                                                                blur=30))
        close = smooth(inv(12.7, 13.5, t))
        if close < 1:
            self.part_feedback(c, t, 1 - close)
        if close > 0:
            self.part_speak(c, t, close)
        img = fr.to_float()
        img = post.look_gris(img, idx, bloom_amt=0.3, paper_amt=0.75, vig=0.32, sat=1.06, thresh=0.84)
        img = fade_black(img, 1 - 0.6 * (1 - seg(t, 0.0, 1.5)))
        img = fade_white(img, seg(t, 24.6, 26.0) ** 1.4 * 0.9, (1, 0.97, 0.9))
        fr.from_float(img)
        self.overlays(c, t)

    def part_feedback(self, c, t, a):
        gx, gy, gs = 1450, 360, 420
        c.save()
        c.translate(gx, 0)
        c.scale(-1, 1)
        c.translate(-gx, 0)
        statue(c, gx, gy, gs, 'gpt', eyes=0.6 + 0.4 * smooth(inv(9.0, 12.0, t)), t=t, bust=1.0, alpha=a)
        c.restore()
        hills_layer().draw(c, 0, 0, a)
        lamps = people(c, t)
        # threads of feedback rising to the giant
        head = (gx - 0.25 * gs, gy - 0.1 * gs)
        for i, (lx, ly, up) in enumerate(lamps):
            if up < 0.05:
                continue
            p = skia.Path()
            p.moveTo(lx, ly)
            p.quadTo((lx + head[0]) / 2, min(ly, head[1]) - 160, head[0], head[1])
            m = skia.PathMeasure(p, False)
            L = m.getLength()
            dst = skia.Path()
            m.getSegment(0, L * min(1.0, up * 1.2), dst, True)
            c.drawPath(dst, paint(LAMP, 0.28 * up * a, stroke=1.4))
            u = (t * 0.4 + i * 0.137) % 1.0
            if up > 0.9:
                pos, _ = m.getPosTan(u * L)
                c.drawCircle(pos.x(), pos.y(), 3, paint(LAMP, 0.8 * a))
        # preference pairs
        for k, (tp, good, bad, good_left) in enumerate(PAIRS):
            ca = env(t, tp, tp + 2.6, 0.35, 0.5) * a
            if ca <= 0:
                continue
            verdict = smooth(inv(tp + 1.0, tp + 1.4, t))
            y = 250 + k * 0
            draw_text(c, 'Q：' + QUESTIONS[k], 620, 180, 'serif_m', 30, (1, 1, 1), ca, shadow=8, shadow_a=0.5)
            lx, rx = 350, 890
            gxp, bxp = (lx, rx) if good_left else (rx, lx)
            fly = ease_in(inv(tp + 1.6, tp + 2.6, t))
            answer_card(c, lerp(gxp, head[0], fly), lerp(y, head[1], fly), good, ca * (1 - fly * 0.8), True, verdict)
            answer_card(c, bxp, y + 60 * seg(t, tp + 1.5, tp + 2.6), bad, ca * (1 - 0.7 * verdict), False, verdict)
        # reward meter
        ra = env(t, 4.4, 12.4, 0.5, 0.6) * a
        if ra > 0:
            fill = clamp((t - 4.4) / 7.5, 0, 1)
            draw_text(c, '奖励  REWARD', 120, 380, 'serif_m', 24, (1, 1, 1), ra, align='left', tracking=0.1)
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(120, 398, 300, 16), 8, 8), paint((1, 1, 1), 0.3 * ra))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(120, 398, 300 * fill, 16), 8, 8),
                        paint(LAMP, 0.95 * ra))

    def part_speak(self, c, t, a):
        # close-up: the giant lowers its head toward the hero
        k = ease_in_out(inv(12.8, 17.0, t))
        gx, gy, gs = lerp(1500, 1380, k), lerp(330, 470, k), lerp(760, 820, k)
        c.save()
        c.translate(gx, 0)
        c.scale(-1, 1)
        c.translate(-gx, 0)
        tilt = lerp(0, 12, k)
        c.translate(gx, gy)
        c.rotate(tilt)
        c.translate(-gx, -gy)
        statue(c, gx, gy, gs, 'gpt', eyes=1.0, t=t, bust=1.0, alpha=a, stone=(0.8, 0.72, 0.74),
               shade=(0.45, 0.36, 0.55), eye_col=(1.0, 0.9, 0.62))
        c.restore()
        # the hero on a hill, looking up
        pts = [(-50, H + 10), (-50, 860), (200, 830), (420, 820), (640, 850), (780, 930), (820, H + 10)]
        c.drawPath(smooth_path(pts, close=True, tension=0.6), paint((0.22, 0.13, 0.26), a))
        pose = ch.blend(ch.idle(t), ch.LOOK_UP, 0.9)
        ch.Hero(120, col=(0.14, 0.08, 0.18)).draw(
            c, 430, 822, pose, t, facing=1, eyes=1.0, glow_col=(1, 0.85, 0.6), glow_r=10, glow_a=0.5 * a,
            alpha=a, scarf=dict(length=2.0, width=0.07, cols=[(0.93, 0.25, 0.3), (1.0, 0.78, 0.3),
                                                              (0.3, 0.75, 0.75), (0.55, 0.45, 0.95)],
                                wind=0.5, speed=0.0, seed=9.0, n=22, glow=8))
        fx.motes(c, t, n=60, seed=41, col=(1, 0.92, 0.75), a=0.55 * a, size=(1.2, 3.0), drift=(6, -12))

    def overlays(self, c, t):
        ui.chapter_card(c, t, 0.8, 5.0, '第 七 章', '对 齐', 'CHAPTER VII · ALIGNMENT', '2022', dim=0.45)
        ui.subtitle(c, t, 5.2, 9.0, '人们一次又一次地告诉它：哪一个回答更好。',
                    'Again and again, people showed it which answer was better.', style='float', y=975)
        ui.subtitle(c, t, 9.2, 12.6, 'RLHF · 基于人类反馈的强化学习', 'Reinforcement Learning from Human Feedback',
                    style='float', y=975)
        ui.dialogue_box(c, t, 14.0, 24.4, 'ChatGPT', '你好！有什么我可以帮你的吗？', speaker_en='2022.11.30',
                        x=W / 2 + 120, w=1100, y=900, rate=11)
        ui.subtitle(c, t, 18.2, 23.2, '那一天，机器第一次，用我们的语言回答了我们。',
                    'That day, a machine answered us — in our own language.', style='float', y=160)


def x_side(k):
    return k % 2 == 0
