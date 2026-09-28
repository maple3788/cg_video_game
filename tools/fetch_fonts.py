#!/usr/bin/env python3
"""Download the (SIL OFL) fonts used by the renderer and create static weight instances.

    python tools/fetch_fonts.py

Fonts come from the google/fonts repository.  Variable fonts are instanced with fontTools
because skia-python ignores variation coordinates on this build.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / 'assets' / 'fonts'
BASE = 'https://raw.githubusercontent.com/google/fonts/main/'

SOURCES = {
    'NotoSerifSC[wght].ttf': 'ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf',
    'Cinzel[wght].ttf': 'ofl/cinzel/Cinzel%5Bwght%5D.ttf',
    'CormorantGaramond[wght].ttf': 'ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf',
    'ZCOOLXiaoWei-Regular.ttf': 'ofl/zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf',
    'MaShanZheng-Regular.ttf': 'ofl/mashanzheng/MaShanZheng-Regular.ttf',
    'OFL.txt': 'ofl/notoserifsc/OFL.txt',
}

INSTANCES = [
    ('Cinzel[wght].ttf', 400, 'Cinzel-Regular.ttf'),
    ('Cinzel[wght].ttf', 600, 'Cinzel-SemiBold.ttf'),
    ('CormorantGaramond[wght].ttf', 300, 'CormorantGaramond-Light.ttf'),
    ('CormorantGaramond[wght].ttf', 500, 'CormorantGaramond-Medium.ttf'),
    ('NotoSerifSC[wght].ttf', 300, 'NotoSerifSC-Light.ttf'),
    ('NotoSerifSC[wght].ttf', 500, 'NotoSerifSC-Medium.ttf'),
    ('NotoSerifSC[wght].ttf', 800, 'NotoSerifSC-ExtraBold.ttf'),
]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        out = DST / name
        if out.exists() and out.stat().st_size > 0:
            continue
        print('download', name, flush=True)
        urllib.request.urlretrieve(BASE + path, out)
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    for src, wght, name in INSTANCES:
        out = DST / name
        if out.exists() and out.stat().st_size > 0:
            continue
        print('instance', name, flush=True)
        font = instancer.instantiateVariableFont(TTFont(DST / src), {'wght': wght})
        font.save(out)
    print('fonts ready in', DST)


if __name__ == '__main__':
    sys.exit(main())
