"""Minimal RPG-style UI overlays: subtitles, chapter cards, skill pop-ups, boss cards,
dialogue boxes, year stamps and letterbox bars."""
from __future__ import annotations

import math

import skia

from .core import W, H, clamp, ease_out, ease_in_out, env, inv, lerp, paint, poly, smooth, radial, linear
from .props import diamond
from .text import draw_text, text_width

INK = (0.96, 0.95, 0.91)
GOLD = (0.95, 0.8, 0.5)
RED = (0.88, 0.22, 0.2)
BAR = 118.0


def letterbox(c, amount=1.0):
    if amount <= 0.001:
        return
    hb = BAR * amount
    p = paint((0, 0, 0), 1.0)
    c.drawRect(skia.Rect.MakeLTRB(0, 0, W, hb), p)
    c.drawRect(skia.Rect.MakeLTRB(0, H - hb, W, H), p)


def subtitle(c, t, t0, t1, cn, en=None, style='bar', col=INK, fi=0.45, fo=0.45, y=None, size=38):
    a = env(t, t0, t1, fi, fo)
    if a <= 0.003:
        return
    bl = (1 - a) * 5
    if style == 'bar':
        y1 = y or 1003.0
        y2 = y1 + 38
        shadow = 0.0
    else:
        y1 = y or 955.0
        y2 = y1 + 40
        shadow = 10.0
    drift = (1 - a) * 6
    draw_text(c, cn, W / 2, y1 - drift, 'serif', size, col, a, tracking=0.12, blur=bl, shadow=shadow,
              shadow_a=0.75)
    if en:
        draw_text(c, en, W / 2, y2 - drift, 'garamond_m', size * 0.7, col, a * 0.78, tracking=0.07,
                  blur=bl, shadow=shadow, shadow_a=0.7)


def chapter_card(c, t, t0, t1, num_cn, title_cn, en, years, y=470, dim=0.68, col=INK):
    a = env(t, t0, t1, 1.0, 0.9)
    if a <= 0.003:
        return
    lt = t - t0
    # legibility vignette
    c.drawRect(skia.Rect.MakeWH(W, H), paint((0, 0, 0), 1.0, shader=radial(W / 2, y + 40, 620,
                                                                          [((0, 0, 0), dim * a), ((0, 0, 0), 0.0)])))
    bl = (1 - a) * 10
    spread = 1 + (1 - a) * 0.25
    draw_text(c, num_cn, W / 2, y - 62, 'serif', 30, col, a * 0.85, tracking=0.6 * spread, blur=bl)
    draw_text(c, title_cn, W / 2, y + 42, 'serif_m', 92, col, a, tracking=0.22 * spread, blur=bl,
              glow=14, glow_a=0.25)
    grow = ease_out(inv(0.3, 1.6, lt)) * (1 if t < t1 - 0.9 else a)
    lw = 330 * grow
    yl = y + 92
    p = paint(col, a * 0.7, stroke=1.3)
    c.drawLine(W / 2 - 22, yl, W / 2 - 22 - lw, yl, p)
    c.drawLine(W / 2 + 22, yl, W / 2 + 22 + lw, yl, p)
    diamond(c, W / 2, yl, 7, col, a * 0.9, stroke=1.3)
    diamond(c, W / 2, yl, 2.5, col, a)
    draw_text(c, en, W / 2, y + 142, 'cinzel', 25, col, a * 0.85, tracking=0.28 * spread, blur=bl)
    draw_text(c, years, W / 2, y + 188, 'garamond', 30, GOLD, a * 0.85, tracking=0.25, blur=bl)


def skill_popup(c, t, t0, t1, title_cn, title_en, desc, x=W / 2, y=250, accent=GOLD,
                caption_cn='获得技能', caption_en='SKILL ACQUIRED'):
    if t < t0 or t > t1:
        return
    lt, rt = t - t0, t1 - t
    tw = max(text_width(title_cn, 'serif_m', 50, 0.18), text_width(desc, 'serif', 23, 0.06), 420)
    bw = tw + 250
    bh = 196
    open_w = ease_out(inv(0.0, 0.35, lt)) * ease_out(inv(0.0, 0.35, rt))
    open_h = ease_out(inv(0.3, 0.75, lt)) * ease_out(inv(0.25, 0.6, rt))
    txt_a = smooth(inv(0.55, 1.0, lt)) * smooth(inv(0.1, 0.45, rt))
    cw = bw * open_w
    ch = max(2, bh * open_h)
    left, top = x - cw / 2, y - ch / 2
    rect = skia.Rect.MakeXYWH(left, top, cw, ch)
    # glow backdrop
    c.drawRect(skia.Rect.MakeXYWH(left - 30, top - 30, cw + 60, ch + 60),
               paint((0, 0, 0), 0.35 * open_h, blur=30))
    fill = paint((0.03, 0.03, 0.04), 0.62 * max(open_h, 0.2))
    fill.setShader(linear(0, top, 0, top + ch, [((0.07, 0.06, 0.05), 0.78), ((0.01, 0.01, 0.02), 0.62)]))
    c.drawRect(rect, fill)
    border = paint(accent, 0.85 * open_w, stroke=1.4)
    c.drawRect(rect, border)
    inset = skia.Rect.MakeXYWH(left + 7, top + 7, max(0, cw - 14), max(0, ch - 14))
    if ch > 20:
        c.drawRect(inset, paint(accent, 0.3 * open_h, stroke=0.9))
    for (cx, cy) in ((left, top), (left + cw, top), (left, top + ch), (left + cw, top + ch)):
        diamond(c, cx, cy, 6, accent, open_w)
    if txt_a <= 0.01:
        return
    # icon
    ix, iy = left + 88, y
    spin = lt * 1.3
    s = 34
    pts = [(ix + math.cos(spin + k * math.pi / 2) * s * (1 if k % 2 == 0 else 0.62),
            iy + math.sin(spin + k * math.pi / 2) * s * 1.0) for k in range(4)]
    c.drawPath(poly(pts), paint(accent, txt_a * 0.45, blur=16, blend='plus'))
    c.drawPath(poly(pts), paint(accent, txt_a, stroke=1.6))
    diamond(c, ix, iy, 9, (1, 1, 1), txt_a)
    c.drawCircle(ix, iy, 46, paint(accent, txt_a * 0.5, stroke=0.8))
    tx = left + 160
    draw_text(c, f'{caption_cn}  ·  {caption_en}', tx, top + 46, 'serif', 21, accent, txt_a, align='left',
              tracking=0.14)
    draw_text(c, title_cn, tx, top + 112, 'serif_m', 50, INK, txt_a, align='left', tracking=0.16,
              glow=12, glow_col=accent, glow_a=0.35)
    draw_text(c, title_en, tx + text_width(title_cn, 'serif_m', 50, 0.16) + 26, top + 110, 'cinzel', 24,
              accent, txt_a * 0.95, align='left', tracking=0.14)
    draw_text(c, desc, tx, top + 160, 'serif', 23, INK, txt_a * 0.82, align='left', tracking=0.05)


def boss_card(c, t, t0, t1, name_cn, name_en, sub='', y=520):
    if t < t0 or t > t1:
        return
    lt, rt = t - t0, t1 - t
    band = ease_out(inv(0.0, 0.3, lt)) * smooth(inv(0.0, 0.4, rt))
    bh = 190 * band
    top = y - bh / 2
    sh = paint((0, 0, 0), 1.0)
    sh.setShader(linear(0, 0, W, 0, [((0, 0, 0), 0.0), ((0, 0, 0), 0.78), ((0, 0, 0), 0.78), ((0, 0, 0), 0.0)],
                        [0, 0.25, 0.75, 1]))
    c.drawRect(skia.Rect.MakeXYWH(0, top, W, bh), sh)
    lp = paint(RED, 0.9 * band, stroke=2)
    lw = W * 0.55 * ease_out(inv(0.05, 0.45, lt))
    c.drawLine(W / 2 - lw / 2, top + 8, W / 2 + lw / 2, top + 8, lp)
    c.drawLine(W / 2 - lw / 2 + 60, top + bh - 8, W / 2 + lw / 2 - 60, top + bh - 8, lp)
    slide = (1 - ease_out(inv(0.1, 0.55, lt))) * 380
    ta = smooth(inv(0.1, 0.4, lt)) * smooth(inv(0.0, 0.35, rt))
    shake = math.sin(lt * 60) * 4 * math.exp(-max(0, lt - 0.55) * 8) if lt > 0.55 else 0
    draw_text(c, 'B O S S', W / 2 + slide * 0.5, y - 44, 'cinzel_b', 22, RED, ta, tracking=0.5)
    draw_text(c, name_cn, W / 2 - slide + shake, y + 26, 'serif_b', 76, INK, ta, tracking=0.3,
              glow=18, glow_col=RED, glow_a=0.35)
    draw_text(c, name_en, W / 2 + slide * 0.7, y + 70, 'cinzel', 24, RED, ta * 0.9, tracking=0.3)


def dialogue_box(c, t, t0, t1, speaker, text, x=W / 2, y=800, w=1180, h=150, rate=13.0,
                 accent=INK, text_col=INK, font_key='serif', size=36, speaker_en=''):
    if t < t0 or t > t1:
        return
    lt, rt = t - t0, t1 - t
    a = smooth(inv(0.0, 0.35, lt)) * smooth(inv(0.0, 0.35, rt))
    left, top = x - w / 2, y - h / 2 + (1 - a) * 12
    rect = skia.Rect.MakeXYWH(left, top, w, h)
    c.drawRect(skia.Rect.MakeXYWH(left - 20, top - 20, w + 40, h + 40), paint((0, 0, 0), 0.4 * a, blur=24))
    c.drawRect(rect, paint((0.02, 0.02, 0.025), 0.72 * a))
    c.drawRect(rect, paint(accent, 0.55 * a, stroke=1.2))
    for (cx, cy) in ((left, top), (left + w, top), (left, top + h), (left + w, top + h)):
        diamond(c, cx, cy, 5, accent, a)
    # name plaque
    nw = text_width(speaker, 'serif_m', 26, 0.12) + 60
    pr = skia.Rect.MakeXYWH(left + 34, top - 22, nw, 44)
    c.drawRect(pr, paint((0.02, 0.02, 0.025), 0.95 * a))
    c.drawRect(pr, paint(accent, 0.7 * a, stroke=1.2))
    draw_text(c, speaker, left + 34 + nw / 2, top + 9, 'serif_m', 26, accent, a, tracking=0.12)
    if speaker_en:
        draw_text(c, speaker_en, left + 34 + nw + 18, top + 7, 'cinzel', 18, accent, a * 0.7, align='left',
                  tracking=0.2)
    n = len(text)
    k = clamp((lt - 0.35) * rate, 0, n)
    lines = text.split('\n')
    yy = top + 72
    shown = k
    for ln in lines:
        if shown <= 0:
            break
        draw_text(c, ln, left + 60, yy, font_key, size, text_col, a, align='left', tracking=0.06,
                  reveal=min(shown, len(ln)))
        shown -= len(ln) + 1
        yy += size * 1.45
    if k >= n:
        blink = 0.5 + 0.5 * math.cos(lt * 6)
        tx, ty = left + w - 40, top + h - 28 + math.sin(lt * 5) * 3
        c.drawPath(poly([(tx - 9, ty - 6), (tx + 9, ty - 6), (tx, ty + 6)]), paint(accent, a * blink))


def year_stamp(c, t, t0, t1, year, caption='', x=150, y=300, align='left', col=INK, size=150, cap_en='',
               glow_a=0.18, cap_col=GOLD):
    a = env(t, t0, t1, 0.8, 0.8)
    if a <= 0.003:
        return
    drift = (1 - a) * 14
    draw_text(c, year, x, y - drift, 'garamond', size, col, a * 0.9, align=align, tracking=0.08,
              blur=(1 - a) * 8, glow=20 if glow_a > 0 else 0, glow_a=glow_a)
    if caption:
        yy = y + 58 - drift
        draw_text(c, caption, x + (6 if align == 'left' else 0), yy, 'serif', 30, col, a * 0.85, align=align,
                  tracking=0.18)
        if cap_en:
            draw_text(c, cap_en, x + (6 if align == 'left' else 0), yy + 38, 'cinzel', 19, cap_col, a * 0.8,
                      align=align, tracking=0.25)


def attention_formula(c, x, y, size=64, col=INK, a=1.0, glow=0.0, glow_col=GOLD):
    """Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V typeset by hand."""
    k = size
    parts = [
        ('Attention(', 'garamond_m', k, 0, 0.02),
        ('Q, K, V', 'garamond_m', k, 0, 0.02),
        (')  =  softmax(', 'garamond_m', k, 0, 0.02),
        ('QK', 'garamond_m', k, 0, 0.02),
        ('T', 'garamond_m', k * 0.58, -k * 0.38, 0.0),
        (' / √', 'dejavu_serif', k * 0.82, 0, 0.0),
        ('d', 'garamond_m', k, 0, 0.0),
        ('k', 'garamond_m', k * 0.58, k * 0.2, 0.0),
        (')', 'garamond_m', k, 0, 0.0),
        (' V', 'garamond_m', k, 0, 0.0),
    ]
    from .text import draw_rich
    return draw_rich(c, parts, x, y, col=col, a=a, glow=glow, glow_col=glow_col)
