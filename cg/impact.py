"""Global hit effects applied after a scene renders: zoom punch + chromatic fringe."""
from __future__ import annotations

import math

from . import post


def hits():
    from . import timeline
    S = {s.NAME: st for s, st in zip(timeline.scenes(), timeline.starts())}
    out = [(S['title'] + 1.6, 1.0), (S['memory'] + 13.3, 0.8), (S['memory'] + 25.6, 0.7),
           (S['attention'] + 12.0, 1.3), (S['attention'] + 31.6, 1.1)]
    out += [(S['giants'] + ts, 0.6) for ts in (9.5, 12.5, 15.5)]
    return out


_HITS = None


def apply(fr, t):
    global _HITS
    if _HITS is None:
        _HITS = hits()
    k = 0.0
    for (th, a) in _HITS:
        dt = t - th
        if 0 <= dt < 1.2:
            k += a * math.exp(-dt * 5.5)
    if k < 0.02:
        return
    img = fr.to_float()
    from .core import rs
    from .ui import BAR
    bp = int(round(BAR * rs())) - 1
    bars = img[:bp].max() < 0.03 and img[-bp:].max() < 0.03
    top, bot = img[:bp].copy(), img[-bp:].copy()
    img = post.shake_warp(img, 0, 0, 0, 1 + 0.028 * k)
    img = post.chroma(img, 5.0 * k)
    if bars:
        img[:bp] = top
        img[-bp:] = bot
    fr.from_float(img)
