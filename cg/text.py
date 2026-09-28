"""Typography: font loading with per-glyph fallback, tracking, glow, typewriter."""
from __future__ import annotations

import functools
from pathlib import Path

import skia

from .core import c4, clamp, paint

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / 'assets' / 'fonts'

FONT_FILES = {
    'serif': FONT_DIR / 'NotoSerifSC-Light.ttf',
    'serif_m': FONT_DIR / 'NotoSerifSC-Medium.ttf',
    'serif_b': FONT_DIR / 'NotoSerifSC-ExtraBold.ttf',
    'cinzel': FONT_DIR / 'Cinzel-Regular.ttf',
    'cinzel_b': FONT_DIR / 'Cinzel-SemiBold.ttf',
    'garamond': FONT_DIR / 'CormorantGaramond-Light.ttf',
    'garamond_m': FONT_DIR / 'CormorantGaramond-Medium.ttf',
    'xiaowei': FONT_DIR / 'ZCOOLXiaoWei-Regular.ttf',
    'brush': FONT_DIR / 'MaShanZheng-Regular.ttf',
    'mono': Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),
    'dejavu_serif': Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'),
    'dejavu': Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    'cjk_fallback': Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
}

FALLBACK = ['serif', 'dejavu_serif', 'dejavu', 'cjk_fallback']


@functools.lru_cache(maxsize=None)
def typeface(key: str) -> skia.Typeface:
    path = FONT_FILES[key]
    tf = skia.Typeface.MakeFromFile(str(path))
    if tf is None:
        raise FileNotFoundError(f'font {key}: {path} (run tools/fetch_fonts.py)')
    return tf


@functools.lru_cache(maxsize=4096)
def font(key: str, size: float) -> skia.Font:
    f = skia.Font(typeface(key), float(size))
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    f.setHinting(skia.FontHinting.kNone)
    return f


@functools.lru_cache(maxsize=65536)
def _has_glyph(key: str, ch: str) -> bool:
    return typeface(key).unicharToGlyph(ord(ch)) != 0


def _pick(key: str, ch: str) -> str:
    if ch == ' ' or _has_glyph(key, ch):
        return key
    for k in FALLBACK:
        if _has_glyph(k, ch):
            return k
    return key


@functools.lru_cache(maxsize=8192)
def layout(text: str, key: str, size: float, tracking: float = 0.0):
    """Return ((char, fontkey, x), ...), total_width.  tracking is in em."""
    items = []
    x = 0.0
    trk = tracking * size
    for i, ch in enumerate(text):
        k = _pick(key, ch)
        adv = font(k, size).measureText(ch) if ch != ' ' else size * (0.28 if key in ('serif', 'serif_m', 'serif_b', 'xiaowei', 'brush') else 0.26)
        items.append((ch, k, x, adv))
        x += adv + (trk if i < len(text) - 1 else 0.0)
    return tuple(items), x


def text_width(text, key='serif', size=40, tracking=0.0):
    return layout(text, key, float(size), float(tracking))[1]


def draw_text(c: skia.Canvas, text: str, x: float, y: float, key='serif', size=40.0,
              col=(1, 1, 1), a=1.0, align='center', tracking=0.0, blur=0.0,
              glow=0.0, glow_col=None, glow_a=0.55, reveal: float | None = None,
              shadow=0.0, shadow_a=0.6, blend=None, per_char_alpha=None, yoffs=None):
    """Draw a single line of text. y is the baseline.

    reveal: number of characters to show (fractional -> last char fades in).
    per_char_alpha: optional callable(i, n) -> alpha multiplier.
    """
    if a <= 0.003 or not text:
        return 0.0
    items, tw = layout(text, key, float(size), float(tracking))
    if align == 'center':
        x0 = x - tw / 2
    elif align == 'right':
        x0 = x - tw
    else:
        x0 = x
    n = len(items)
    passes = []
    if shadow > 0:
        passes.append(('shadow', paint((0, 0, 0), a * shadow_a, blur=shadow)))
    if glow > 0:
        passes.append(('glow', paint(glow_col or col, a * glow_a, blur=glow, blend='plus' if blend is None else blend)))
    passes.append(('main', paint(col, a, blur=blur, blend=blend)))
    for kind, p in passes:
        base_alpha = p.getAlphaf()
        for i, (ch, k, dx, adv) in enumerate(items):
            if ch == ' ':
                continue
            m = 1.0
            if reveal is not None:
                if i >= reveal:
                    break
                m = clamp(reveal - i)
            if per_char_alpha is not None:
                m *= per_char_alpha(i, n)
            if m <= 0.003:
                continue
            p.setAlphaf(base_alpha * m)
            yy = y + (yoffs(i, n) if yoffs else 0.0)
            c.drawString(ch, x0 + dx, yy, font(k, size), p)
    return tw


def draw_rich(c, parts, x, y, align='center', col=(1, 1, 1), a=1.0, glow=0.0, glow_col=None,
              glow_a=0.5, blur=0.0):
    """parts: list of (text, fontkey, size, dy, tracking) drawn in sequence on one baseline."""
    widths = [text_width(t, k, s, tr) for (t, k, s, dy, tr) in parts]
    gap = 0.0
    tw = sum(widths)
    x0 = x - tw / 2 if align == 'center' else (x - tw if align == 'right' else x)
    cx = x0
    for (t, k, s, dy, tr), wd in zip(parts, widths):
        draw_text(c, t, cx, y + dy, k, s, col, a, align='left', tracking=tr, glow=glow,
                  glow_col=glow_col, glow_a=glow_a, blur=blur)
        cx += wd + gap
    return tw
