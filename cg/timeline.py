"""Global timeline: ordered list of scenes and time lookup."""
from __future__ import annotations

import bisect
import importlib

# (module, class) in playback order
SCENE_SPECS = [
    ('forest', 'Prologue'),
    ('forest', 'Title'),
    ('rules', 'ForestOfRules'),
    ('winter', 'Winter'),
    ('memory', 'Memory'),
    ('sea', 'Sea'),
    ('attention', 'Attention'),
    ('giants', 'Giants'),
    ('alignment', 'Alignment'),
    ('library', 'Library'),
    ('epilogue', 'Epilogue'),
]

_scenes = None
_starts = None


def scenes():
    global _scenes, _starts
    if _scenes is None:
        _scenes = []
        for mod, cls in SCENE_SPECS:
            m = importlib.import_module(f'cg.scenes.{mod}')
            _scenes.append(getattr(m, cls)())
        _starts = []
        acc = 0.0
        for s in _scenes:
            _starts.append(acc)
            acc += s.DUR
    return _scenes


def starts():
    scenes()
    return list(_starts)


def total():
    sc = scenes()
    return _starts[-1] + sc[-1].DUR


def locate(t: float):
    sc = scenes()
    i = max(0, bisect.bisect_right(_starts, t) - 1)
    i = min(i, len(sc) - 1)
    return sc[i], t - _starts[i]


def scene_start(name: str) -> float:
    for s, st in zip(scenes(), _starts):
        if s.NAME == name:
            return st
    raise KeyError(name)
