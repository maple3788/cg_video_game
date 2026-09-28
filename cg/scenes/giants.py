"""Chapter VI — The Giants Awaken (2018–2021): BERT & GPT statues, scaling laws, emergence,
and the storm of hallucination."""
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

HZ = 700.0
STONE = (0.9, 0.85, 0.8)
SHADE = (0.58, 0.5, 0.7)
EYE = (1.0, 0.85, 0.5)

GPT_PROFILE = [(-0.45, -0.55), (-0.1, -0.72), (0.32, -0.56), (0.42, -0.22), (0.45, -0.12), (0.61, 0.08),
               (0.47, 0.16), (0.51, 0.26), (0.44, 0.31), (0.49, 0.36), (0.42, 0.52), (0.12, 0.63),
               (0.09, 0.95), (-0.33, 0.95), (-0.41, 0.5), (-0.55, 0.0)]


def janus_profile():
    right = [(0.0, -0.74), (0.36, -0.56), (0.44, -0.22), (0.46, -0.12), (0.6, 0.06), (0.47, 0.15), (0.5, 0.26),
             (0.44, 0.31), (0.48, 0.36), (0.4, 0.52), (0.2, 0.63), (0.19, 0.95)]
    left = [(-x, y) for (x, y) in reversed(right[1:])]
    return right + left


def statue(c, cx, cy, s, kind='gpt', eyes=1.0, t=0.0, bust=0.0, alpha=1.0, stone=STONE, shade=SHADE,
           eye_col=EYE):
    pts = GPT_PROFILE if kind == 'gpt' else janus_profile()
    P = [(cx + x * s, cy + y * s) for (x, y) in pts]
    path = smooth_path(P, close=True, tension=0.7)
    if bust > 0:
        # shoulders & chest emerging below the neck
        sw = s * 1.5
        body = smooth_path([(cx - sw, cy + 2.4 * s), (cx - sw * 0.95, cy + 1.35 * s), (cx - sw * 0.55, cy + 1.02 * s),
                            (cx - 0.3 * s, cy + 0.92 * s), (cx + 0.2 * s, cy + 0.92 * s), (cx + sw * 0.55, cy + 1.02 * s),
                            (cx + sw * 0.95, cy + 1.35 * s), (cx + sw, cy + 2.4 * s)], close=True, tension=0.6)
        bp = paint(stone, alpha)
        bp.setShader(linear(cx - sw, 0, cx + sw, 0, [(stone, alpha), (stone, alpha), (shade, alpha)], [0, 0.55, 1]))
        c.drawPath(body, bp)
    p = paint(stone, alpha)
    if kind == 'gpt':
        p.setShader(linear(cx - 0.55 * s, 0, cx + 0.6 * s, 0, [(shade, alpha), (stone, alpha), (stone, alpha)],
                           [0, 0.45, 1]))
    else:
        p.setShader(linear(cx - 0.6 * s, 0, cx + 0.6 * s, 0, [(shade, alpha), (stone, alpha), (stone, alpha),
                                                              (shade, alpha)], [0, 0.3, 0.7, 1]))
    c.drawPath(path, p)
    c.drawPath(path, paint((0.42, 0.34, 0.52), 0.55 * alpha, stroke=max(1.2, s * 0.006)))
    # carved lines
    lp = paint(shade, 0.6 * alpha, stroke=max(1.0, s * 0.008))
    if kind == 'gpt':
        c.drawArc(skia.Rect.MakeXYWH(cx - 0.2 * s, cy - 0.05 * s, 0.16 * s, 0.26 * s), 100, 200, False, lp)
        eyes_at = [(0.28, -0.08)]
    else:
        eyes_at = [(0.27, -0.08), (-0.27, -0.08)]
        c.drawLine(cx, cy - 0.7 * s, cx, cy + 0.6 * s, paint(shade, 0.35 * alpha, stroke=max(1.0, s * 0.006)))
    for (ex, ey) in eyes_at:
        x, y = cx + ex * s, cy + ey * s
        w_, h_ = 0.15 * s, 0.035 * s * (0.25 + 0.75 * eyes)
        eye = poly([(x - w_ / 2, y), (x - w_ * 0.1, y - h_), (x + w_ / 2, y), (x - w_ * 0.1, y + h_)])
        c.drawPath(eye, paint((0.3, 0.26, 0.36), alpha))
        if eyes > 0.05:
            c.drawCircle(x, y, s * 0.16, paint(eye_col, 0.45 * eyes * alpha, blur=s * 0.08, blend='plus'))
            c.drawPath(eye, paint(eye_col, eyes * alpha))


@functools.lru_cache(maxsize=1)
def backdrops():
    sky = fx.wash(W, HZ + 20, [(0.56, 0.72, 0.8), (0.88, 0.78, 0.8), (1.0, 0.86, 0.72)], seed=71, blotch=0.1,
                  cell=260, blooms=5)
    sea = fx.wash(W, H - HZ + 400, [(0.62, 0.78, 0.8), (0.3, 0.52, 0.6), (0.16, 0.3, 0.4)], seed=72, blotch=0.12,
                  cell=200, blooms=4)
    return sky, sea


GPT_STAGES = [(9.5, 300.0, 520.0), (12.5, 520.0, 900.0), (15.5, 900.0, 1500.0)]
COUNTERS = [(9.8, 12.4, 'GPT-1', 1.17, '亿参数', '2018'), (12.8, 15.4, 'GPT-2', 15, '亿参数', '2019'),
            (15.8, 18.6, 'GPT-3', 1750, '亿参数', '2020')]
HALLU = ['长城在月球上清晰可见', '2 + 2 = 5', '巴黎是意大利的首都', '鲸鱼是鱼', '爱因斯坦发明了电话', '水在 50°C 沸腾',
         '莎士比亚写了《红楼梦》', '1 kg 铁比 1 kg 棉花重']
ABILITIES = ['翻译', '写诗', '编程', '算术', '总结', '问答', '推理', '类比']


class Giants:
    DUR = 26.0
    NAME = 'giants'

    def gpt_scale(self, t):
        s = 300.0
        for (ts, a, b) in GPT_STAGES:
            if t >= ts:
                s = lerp(a, b, ease_out_back(inv(ts, ts + 1.1, t), 1.2))
        return s

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        sky, sea = backdrops()
        sg = self.gpt_scale(t)
        zoom = (300.0 / sg) ** 0.8
        pivot = (960.0, 760.0)
        storm = seg(t, 21.2, 23.0)
        # sky (screen space) darkens in the storm
        c.drawRect(skia.Rect.MakeWH(W, H), paint((1.0, 0.86, 0.72)))
        sky.draw(c, 0, 0, sy=(H * 0.78) / (HZ + 20))
        if storm > 0:
            c.drawRect(skia.Rect.MakeWH(W, H), paint((0.2, 0.14, 0.28), 0.7 * storm))
        sun_a = 1 - storm
        c.drawCircle(1500, 330, 520, paint((1, 1, 1), shader=radial(1500, 330, 520, [
            ((1, 0.95, 0.85), 0.5 * sun_a), ((1, 0.95, 0.85), 0.0)])))
        c.drawCircle(1500, 330, 120, paint((1, 0.97, 0.9), 0.9 * sun_a))
        c.save()
        c.translate(pivot[0], pivot[1])
        c.scale(zoom, zoom)
        c.translate(-pivot[0], -pivot[1])
        # sea (world)
        sea.draw(c, -3000, HZ, sx=(W + 6000) / W, sy=3.0)
        c.drawRect(skia.Rect.MakeLTRB(-4000, HZ - 2, 6000, HZ + 3), paint((1, 0.95, 0.9), 0.5))
        # BERT (left) and GPT (right) rising
        rise = ease_out(inv(0.3, 6.0, t))
        bert_a = 1 - seg(t, 9.0, 10.5)
        if bert_a > 0:
            by = lerp(HZ + 420, 430, rise) + 60 * seg(t, 9.0, 10.5)
            self.stream(c, t, 520, by, 300, rise)
            statue(c, 520, by, 300, 'bert', eyes=smooth(inv(4.0, 5.0, t)), t=t, alpha=bert_a)
        gx = 1420 + (sg - 300) * 0.25
        gy = lerp(HZ + 420, 430, rise) - (sg - 300) * 0.62
        self.stream(c, t, gx, gy, sg, rise)
        statue(c, gx, gy, sg, 'gpt', eyes=smooth(inv(4.6, 5.6, t)), t=t, bust=smooth(inv(12.0, 13.5, t)))
        # water line in front of statues
        c.drawRect(skia.Rect.MakeLTRB(-4000, HZ, 6000, HZ + 3000), paint((0.3, 0.52, 0.6), 0.55))
        # sandbar + hero
        c.drawPath(smooth_path([(800, 772), (850, 758), (960, 752), (1080, 758), (1130, 772)], close=True),
                   paint((0.96, 0.9, 0.8)))
        pose = ch.blend(ch.idle(t), ch.LOOK_UP, 0.8)
        ch.Hero(72, col=(0.14, 0.1, 0.2)).draw(c, 960, 754, pose, t, eyes=1.0, glow_col=(1, 0.9, 0.7), glow_r=6,
                                               glow_a=0.4,
                                               scarf=dict(length=2.0, width=0.07, cols=[(0.93, 0.25, 0.3),
                                                          (1.0, 0.78, 0.3), (0.3, 0.75, 0.75), (0.55, 0.45, 0.95)],
                                                          wind=0.6 + 0.4 * storm, speed=0.0, seed=8.0, n=22, glow=6))
        self.emergence(c, t)
        c.restore()
        if storm > 0:
            self.storm(c, t, storm, gx, gy, sg, zoom, pivot)
        img = fr.to_float()
        img = post.look_gris(img, idx, bloom_amt=0.25, paper_amt=0.75, vig=0.3 + 0.2 * storm, sat=1.08 - 0.3 * storm,
                             thresh=0.85, chroma_px=4 * storm)
        img = fade_white(img, (1 - seg(t, 0.0, 1.6)) * 0.95, (1, 0.98, 0.94))
        fr.from_float(img)
        self.overlays(c, t)

    def stream(self, c, t, x, y, s, rise):
        """Water pouring off a rising statue."""
        if rise >= 0.98 or rise <= 0.02:
            return
        r = Rng(int(x))
        for k in range(26):
            sx = x + r.u(-0.45, 0.55) * s
            top = y + r.u(-0.3, 0.6) * s
            ln = r.u(40, 160) * (1 - rise)
            ph = (t * 3 + r.u(0, 1)) % 1.0
            c.drawLine(sx, top + ph * 80, sx, top + ph * 80 + ln, paint((0.85, 0.95, 1.0), 0.5 * (1 - rise), stroke=2))

    def emergence(self, c, t):
        if t < 18.2:
            return
        r = Rng(21)
        fade = 1 - seg(t, 23.0, 25.0)
        cols = [(1.0, 0.6, 0.7), (1.0, 0.8, 0.4), (0.6, 0.85, 0.8), (0.75, 0.6, 1.0), (1, 1, 1)]
        for k in range(90):
            x = r.u(-1600, 3500)
            y = HZ + 20 + r.u(0, 1) ** 1.5 * 1400
            tb = 18.3 + r.u(0, 2.6)
            g = ease_out_back(inv(tb, tb + 0.6, t)) * fade
            if g <= 0:
                continue
            s = r.u(55, 120) * (1 + (y - HZ) / 900)
            col = cols[k % len(cols)]
            for pp in range(5):
                a = pp * 2 * math.pi / 5 + t * 0.3
                c.drawCircle(x + math.cos(a) * s * 0.6 * g, y + math.sin(a) * s * 0.35 * g, s * 0.42 * g, paint(col, 0.85))
            c.drawCircle(x, y, s * 0.25 * g, paint((1, 0.95, 0.7)))
            if k < len(ABILITIES) * 3 and k % 3 == 0:
                word = ABILITIES[k // 3]
                u = inv(tb + 0.3, tb + 3.0, t)
                draw_text(c, word, x, y - 120 - u * 900, 'serif_b', 150 * (1 + (y - HZ) / 900), (0.32, 0.14, 0.36),
                          env(t, tb + 0.3, tb + 3.2, 0.4, 0.8) * fade, glow=40, glow_col=(1, 1, 1), glow_a=0.7)

    def storm(self, c, t, k, gx, gy, sg, zoom, pivot):
        # swirling hallucinated "facts" around the giant's head (screen space)
        hx = (gx - pivot[0]) * zoom + pivot[0]
        hy = (gy - pivot[1]) * zoom + pivot[1]
        r = Rng(5)
        for q, txt in enumerate(HALLU * 2):
            ang = t * (0.5 + 0.12 * (q % 3)) + q * 0.785
            rad = 260 + 90 * (q % 4) + 30 * math.sin(t + q)
            x, y = hx + math.cos(ang) * rad * 1.5, hy + math.sin(ang) * rad * 0.7
            a = k * (0.55 + 0.45 * math.sin(t * 4 + q))
            gl = int(t * 12 + q) % 7 == 0
            dx = r.n(0, 10) if gl else 0
            col = (1.0, 0.45, 0.55) if q % 2 else (0.8, 0.6, 1.0)
            draw_text(c, txt, x + dx, y, 'serif_m', 30, col, a, glow=10, glow_col=col, glow_a=0.6)
            if gl:
                draw_text(c, txt, x + dx + 6, y + 2, 'serif_m', 30, (0.3, 0.9, 1.0), a * 0.5)
        # lightning
        for (tl, xl) in ((22.3, 400), (23.6, 1500), (24.9, 700)):
            if tl < t < tl + 0.18:
                rr = Rng(int(tl * 10))
                x, y = xl, 120
                p = skia.Path()
                p.moveTo(x, y)
                while y < 600:
                    x += rr.n(0, 40)
                    y += rr.u(30, 70)
                    p.lineTo(x, y)
                c.drawPath(p, paint((1, 0.95, 1), 0.9 * k, stroke=3))
                c.drawPath(p, paint((0.8, 0.7, 1), 0.6 * k, stroke=16, blur=12, blend='plus'))
                c.drawRect(skia.Rect.MakeWH(W, H), paint((0.9, 0.85, 1), 0.2 * k, blend='plus'))

    def overlays(self, c, t):
        ui.chapter_card(c, t, 1.0, 5.4, '第 六 章', '巨人苏醒', 'CHAPTER VI · THE GIANTS AWAKEN', '2018 — 2021',
                        dim=0.5)
        # BERT / GPT demos
        a = env(t, 5.2, 9.2, 0.5, 0.6)
        if a > 0:
            draw_text(c, 'BERT · 2018', 520, 180, 'cinzel_b', 30, (0.3, 0.2, 0.4), a, tracking=0.2)
            draw_text(c, '双向阅读 · 完形填空', 520, 222, 'serif_m', 28, (0.3, 0.2, 0.4), a)
            fill = smooth(inv(6.6, 7.2, t))
            x0 = 520 - 170
            for i, tk in enumerate(['我', '爱', None, '书']):
                x = x0 + i * 110
                if tk is None:
                    c.drawRect(skia.Rect.MakeXYWH(x - 60, 810, 120, 56), paint((0.3, 0.2, 0.4), 0.85 * a, stroke=2))
                    draw_text(c, '[MASK]', x, 848, 'garamond_m', 30, (0.95, 0.95, 1), a * (1 - fill))
                    draw_text(c, '读', x, 852, 'serif_b', 40, (1, 0.85, 0.5), a * fill, glow=14, glow_col=EYE)
                else:
                    draw_text(c, tk, x, 852, 'serif_m', 40, (1, 1, 1), a, shadow=6, shadow_a=0.4)
            draw_text(c, 'GPT · 2018', 1420, 180, 'cinzel_b', 30, (0.3, 0.2, 0.4), a, tracking=0.2)
            draw_text(c, '从左到右 · 预测下一个词', 1420, 222, 'serif_m', 28, (0.3, 0.2, 0.4), a)
            seq = ['从', '前', '有', '座', '山', '，', '山', '里', '有', '座', '庙']
            n = 5 + int(clamp((t - 6.0) / 0.45, 0, 6))
            x0 = 1420 - 5 * 52
            for i in range(n):
                new = i >= 5
                ka = 1.0 if not new else smooth(inv(6.0 + (i - 5) * 0.45, 6.0 + (i - 5) * 0.45 + 0.3, t))
                draw_text(c, seq[i], x0 + i * 52, 852, 'serif_m', 40, (1, 0.85, 0.5) if new else (1, 1, 1), a * ka,
                          shadow=6, shadow_a=0.4, glow=12 if new else 0, glow_col=EYE)
            if int(t * 3) % 2 == 0:
                c.drawRect(skia.Rect.MakeXYWH(x0 + n * 52 - 16, 818, 4, 42), paint((1, 1, 1), a))
        ui.subtitle(c, t, 5.0, 9.0, '2018 · BERT 学会了阅读，GPT 学会了续写。',
                    '2018 · BERT learned to read; GPT learned to write on.', style='float', y=975)
        for (t0, t1, name, val, unit, year) in COUNTERS:
            a = env(t, t0, t1, 0.4, 0.5)
            if a <= 0:
                continue
            prev = {1.17: 0.0, 15: 1.17, 1750: 15}[val]
            v = lerp(prev, val, ease_out(inv(t0, t0 + 1.2, t)))
            vs = f'{v:.2f}' if val < 10 else f'{v:.0f}'
            draw_text(c, name, 170, 330, 'cinzel_b', 44, (0.25, 0.15, 0.32), a, align='left', tracking=0.15)
            draw_text(c, vs, 160, 470, 'garamond_m', 150, (0.25, 0.15, 0.32), a, align='left', glow=16,
                      glow_col=(1, 1, 1), glow_a=0.4)
            draw_text(c, f'{unit} · {year}', 172, 530, 'serif_m', 34, (0.25, 0.15, 0.32), a, align='left',
                      tracking=0.1)
        ui.subtitle(c, t, 12.9, 17.2, '规模定律：更多的数据，更多的参数，更多的算力。',
                    'Scaling laws: more data, more parameters, more compute.', style='float', y=975)
        ui.subtitle(c, t, 18.6, 21.4, '于是，能力开始涌现。', 'And then — abilities began to emerge.', style='float',
                    y=975)
        ui.subtitle(c, t, 21.8, 25.8, '可它还不懂：人真正想要的是什么。',
                    'But it did not yet understand what people truly wanted.', style='float', y=975)
