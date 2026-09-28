"""Epilogue: sitting in the giant's palm at sunrise; the final title — the next token is yours."""
from __future__ import annotations

import functools
import math

import numpy as np
import skia

from ..core import (W, H, Frame, Layer, Rng, clamp, ease_in, ease_in_out, ease_out, env, inv, lerp, lerp3, paint,
                    poly, radial, linear, smooth, seg, snoise, smooth_path)
from .. import character as ch
from .. import fx, post, props, ui
from ..text import draw_text, text_width, layout
from .common import fade_black
from .giants import STONE, SHADE

HZ = 690.0
SCARF = [(0.93, 0.25, 0.3), (1.0, 0.78, 0.3), (0.3, 0.75, 0.75), (0.55, 0.45, 0.95)]
MEMORIES = ['IF', 'THEN', 'ELIZA', 'δ', 'hₜ', 'cₜ', '国王', '女王', '法语', '它', 'Q', 'K', 'V', '你好', 'RAG',
            '∂L/∂w', 'softmax', '词元', '注意力', 'token']


@functools.lru_cache(maxsize=1)
def backdrops():
    sky = fx.wash(W, HZ + 20, [(0.52, 0.5, 0.78), (0.95, 0.66, 0.72), (1.0, 0.84, 0.66), (1.0, 0.93, 0.78)],
                  seed=101, blotch=0.1, cell=280, pos=[0, 0.45, 0.8, 1.0], blooms=5)
    sea = fx.wash(W, H - HZ, [(0.95, 0.74, 0.7), (0.55, 0.42, 0.62), (0.28, 0.2, 0.4)], seed=102, blotch=0.12,
                  cell=220, blooms=3)

    def hand(c):
        # a colossal open stone hand rising from the sea, palm up, fingers curling (GRIS homage)
        def shaded(path, x0, x1):
            pn = paint(STONE)
            pn.setShader(linear(x0, 0, x1, 0, [SHADE, STONE, STONE, SHADE], [0, 0.3, 0.75, 1]))
            c.drawPath(path, pn)
            c.drawPath(path, paint((0.45, 0.36, 0.52), 0.45, stroke=2))
        # forearm + palm
        arm = smooth_path([(470, H + 40), (560, 930), (680, 858), (800, 822), (960, 806), (1120, 800), (1215, 792),
                           (1250, 842), (1230, 900), (1120, 930), (960, 950), (820, 990), (720, H + 40)],
                          close=True, tension=0.55)
        shaded(arm, 470, 1260)
        # four fingers, curling upward at the tips (side view, slightly fanned)
        for k, (by, tipx, tipy, w) in enumerate([(810, 1470, 640, 44), (830, 1450, 670, 46), (852, 1420, 700, 46),
                                                 (872, 1380, 730, 42)]):
            f = smooth_path([(1190, by - w * 0.5), (1320, by - w * 0.7 - 10), (tipx - 20, (by + tipy) / 2 - 30),
                             (tipx, tipy), (tipx + 26, tipy + 20), (tipx + 10, (by + tipy) / 2 + 10),
                             (1330, by + w * 0.55), (1200, by + w * 0.6)], close=True, tension=0.6)
            shaded(f, 1180, tipx + 30)
            c.drawLine(1330, by - w * 0.1, 1345, by + w * 0.35, paint(SHADE, 0.6, stroke=2))
        # thumb raised on the near side
        th = smooth_path([(820, 850), (800, 780), (820, 690), (860, 640), (900, 632), (910, 662), (888, 720),
                          (880, 800), (900, 850)], close=True, tension=0.6)
        shaded(th, 790, 915)
        c.drawLine(812, 752, 860, 748, paint(SHADE, 0.55, stroke=2))
        c.drawPath(smooth_path([(900, 812), (1010, 830), (1150, 822)]), paint(SHADE, 0.5, stroke=3))
    hl = Layer(W, H, arr=fx.watercolorize(Layer(W, H, hand).arr, seed=103, edge=0.3, gran=0.05, wobble=1.5))
    return sky, sea, hl


class Epilogue:
    DUR = 24.0
    NAME = 'epilogue'

    def render(self, fr: Frame, t: float, idx: int):
        c = fr.c
        if t < 12.6:
            sky, sea, hand = backdrops()
            z = lerp(1.22, 1.0, ease_in_out(inv(0.0, 12.6, t)))
            c.save()
            c.translate(1100, 700)
            c.scale(z, z)
            c.translate(-1100, -700)
            sky.draw(c, -100, -60, sx=(W + 200) / W, sy=1.1)
            sx, sy = 1480, lerp(700, 610, inv(0, 12.6, t))
            c.drawCircle(sx, sy, 800, paint((1, 1, 1), shader=radial(sx, sy, 800, [
                ((1, 0.9, 0.72), 0.55), ((1, 0.9, 0.72), 0.0)])))
            c.drawCircle(sx, sy, 190, paint((1.0, 0.95, 0.84), 0.95))
            sea.draw(c, -100, HZ, sx=(W + 200) / W, sy=1.2)
            c.restore()
            img = fr.to_float()
            from ..core import rs
            img = fx.reflect(img, HZ * rs(), strength=0.5, t=t, ripple=5, blur_px=2.5, tint=(1.0, 0.9, 0.85))
            fr.from_float(img)
            c.save()
            c.translate(1100, 700)
            c.scale(z, z)
            c.translate(-1100, -700)
            # memories drifting toward the horizon on the water
            r = Rng(5)
            for q, wd in enumerate(MEMORIES * 2):
                u = ((t * 0.05 + r.u(0, 1)) % 1.0)
                x0 = r.u(-200, W + 200)
                x = lerp(x0, sx + (x0 - sx) * 0.1, u)
                y = lerp(H + 40, HZ + 6, u ** 0.7)
                s = lerp(40, 10, u)
                a = min(1.0, u * 4) * (1 - u) * 1.3
                draw_text(c, wd, x, y, 'serif_m', s, (1, 0.97, 0.9), clamp(a), glow=s * 0.4, glow_col=(1, 0.8, 0.6),
                          glow_a=0.6)
            hand.draw(c, 0, 0)
            pose = ch.SIT
            ch.Hero(104, col=(0.2, 0.1, 0.2)).draw(
                c, 1060, 806, ch.blend(pose, ch.Pose(py=0.02, torso=-0.02, head=-0.2, th=(1.42, 1.3),
                                                     kn=(0.6, 0.8), sh=(0.45, 0.25), el=(0.6, 0.4)), 0.5),
                t, facing=1, eyes=1.0, glow_col=(1, 0.85, 0.6), glow_r=8, glow_a=0.5,
                scarf=dict(length=2.6, width=0.075, cols=SCARF, wind=0.9, speed=0.3, seed=12.0, n=26, glow=8))
            fx.motes(c, t, n=80, seed=51, col=(1, 0.93, 0.8), a=0.55, size=(1.2, 3.2), drift=(-10, -8))
            c.restore()
            img = fr.to_float()
            img = post.look_gris(img, idx, bloom_amt=0.4, paper_amt=0.8, vig=0.32, sat=1.05, thresh=0.78)
            img = fade_black(img, seg(t, 0.0, 1.6) * (1 - seg(t, 11.2, 12.6)))
            fr.from_float(img)
            ui.subtitle(c, t, 1.6, 5.6, '从沉默，到言语。', 'From silence, to speech.', style='float', y=975)
            ui.subtitle(c, t, 6.0, 10.8, '而这个故事，还远远没有结束。', 'And this story is far from over.', style='float',
                        y=975)
        else:
            fr.clear((0, 0, 0))
            img = fr.to_float()
            img = post.grain(img, 0.03, idx)
            fr.from_float(img)
            self.final_card(c, t - 12.6)

    def final_card(self, c, lt):
        a = env(lt, 0.6, 11.0, 1.4, 1.2)
        if a > 0:
            draw_text(c, 'TOKEN', W / 2, 430, 'cinzel', 150, (0.96, 0.95, 0.91), a, tracking=0.6, blur=(1 - a) * 12,
                      glow=24, glow_a=0.3)
            draw_text(c, '词   元', W / 2, 525, 'serif_m', 52, (0.96, 0.95, 0.91), a, tracking=0.35)
        b = env(lt, 3.0, 11.0, 1.0, 1.2)
        if b > 0:
            txt = '下一个词，由你写下。'
            k = clamp((lt - 3.0) * 7, 0, len(txt))
            draw_text(c, txt, W / 2, 660, 'serif', 40, (0.96, 0.95, 0.91), b, tracking=0.2, reveal=k)
            draw_text(c, 'The next word is yours to write.', W / 2, 712, 'garamond', 30, ui.GOLD, b * 0.85 *
                      smooth(inv(4.6, 5.6, lt)), tracking=0.1)
        # blinking cursor = the next token
        items, tw = layout('下一个词，由你写下。', 'serif', 40.0, 0.2)
        cx = W / 2 + tw / 2 + 18
        if lt > 3.0 and int(lt * 1.9) % 2 == 0:
            ca = 1.0 if lt < 11.2 else clamp(1 - (lt - 11.2) / 0.2)
            c.drawRect(skia.Rect.MakeXYWH(cx, 624, 5, 46), paint((0.96, 0.95, 0.91), ca))
        d = env(lt, 7.0, 11.0, 0.8, 1.0)
        draw_text(c, '敬 请 期 待   ·   COMING SOON', W / 2, 880, 'cinzel', 20, (0.8, 0.78, 0.72), d * 0.8,
                  tracking=0.3)
