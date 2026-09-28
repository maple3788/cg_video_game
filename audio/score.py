#!/usr/bin/env python3
"""Compose and render the soundtrack (LIMBO-like ambience -> GRIS-like score), synced to the
scene timeline.  Usage:  python audio/score.py out/score.wav
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.synth import (SR, Bus, bell, boom, braam, celesta, choir, click, convolve, crash, drone, footstep,
                         glitch, heartbeat, hz, ice_crack, impulse, noise_burst, pad, piano, reverse_cymbal, riser,
                         strings_stac, waves, whoosh, wind, n_of)
from cg import timeline
from cg.core import Rng

# ---------------------------------------------------------------- harmony helpers

CH = {
    'Dm': ['D3', 'F3', 'A3', 'D4'], 'D': ['D3', 'F#3', 'A3', 'D4'], 'Bb': ['Bb2', 'D3', 'F3', 'Bb3'],
    'F': ['F2', 'C3', 'F3', 'A3'], 'C': ['C3', 'E3', 'G3', 'C4'], 'Gm': ['G2', 'D3', 'G3', 'Bb3'],
    'A': ['A2', 'E3', 'A3', 'C#4'], 'Bm': ['B2', 'F#3', 'B3', 'D4'], 'G': ['G2', 'D3', 'G3', 'B3'],
    'Em': ['E3', 'G3', 'B3', 'E4'], 'Dadd9': ['D3', 'A3', 'E4', 'F#4'],
}


def fs(names):
    return [hz(n) for n in names]


def up(names, octaves=1):
    out = []
    for n in names:
        p, o = n[:-1], int(n[-1])
        out.append(f'{p}{o + octaves}')
    return out


THEME = [  # (beat, note, beats) — 8 bars of 4/4
    (0, 'D5', 2), (2, 'A4', 1), (3, 'B4', 1), (4, 'F#4', 4),
    (8, 'G4', 1), (9, 'A4', 1), (10, 'B4', 1), (11, 'D5', 1), (12, 'A4', 4),
    (16, 'B4', 2), (18, 'C#5', 1), (19, 'D5', 1), (20, 'E5', 2), (22, 'F#5', 1), (23, 'E5', 1),
    (24, 'D5', 2), (26, 'C#5', 2), (28, 'D5', 4)]
THEME_CHORDS = [(0, 'D'), (4, 'Bm'), (8, 'G'), (12, 'A'), (16, 'Bm'), (20, 'G'), (24, 'Em'), (26, 'A'), (28, 'D')]


def play_theme(music, t0, bpm, bar0=0, bar1=8, vel=0.55, strings=True, choir_on=False, octave=0):
    beat = 60.0 / bpm
    b0, b1 = bar0 * 4, bar1 * 4
    for (b, note, ln) in THEME:
        if b0 <= b < b1:
            nm = up([note], octave)[0] if octave else note
            music.add(t0 + (b - b0) * beat, piano(hz(nm), ln * beat * 0.95, vel))
    for i, (b, chn) in enumerate(THEME_CHORDS):
        if b0 <= b < b1:
            nxt = THEME_CHORDS[i + 1][0] if i + 1 < len(THEME_CHORDS) else 32
            ln = (min(nxt, b1) - b) * beat
            notes = CH[chn]
            music.add(t0 + (b - b0) * beat, piano(hz(notes[0].replace('3', '2').replace('2', '2')), ln, vel * 0.45))
            for k, n in enumerate(notes[1:]):
                music.add(t0 + (b - b0) * beat + (k + 1) * beat * 0.5, piano(hz(n), ln * 0.8, vel * 0.3))
            if strings:
                music.add(t0 + (b - b0) * beat, pad(fs(notes), ln, vel * 0.5, attack=0.8, release=1.6, cutoff=1500))
            if choir_on:
                music.add(t0 + (b - b0) * beat, choir(fs(up(notes[1:], 1)), ln, vel * 0.35, attack=0.8, release=1.8))


def steps_from_walker(walker, t_from, t_to, dt=1 / 240):
    """Times where the gait phase crosses a half cycle (a foot lands)."""
    out = []
    t = t_from
    prev = walker.phase(t) * 2
    while t < t_to:
        t += dt
        ph = walker.phase(t) * 2
        if math.floor(ph) != math.floor(prev) and walker.moving(t) > 0.2:
            out.append(t)
        prev = ph
    return out


# ---------------------------------------------------------------- the score

def compose():
    total = timeline.total()
    S = {s.NAME: st for s, st in zip(timeline.scenes(), timeline.starts())}
    SC = {s.NAME: s for s in timeline.scenes()}
    music, sfx, amb = Bus(total), Bus(total), Bus(total)

    # ============ PROLOGUE & TITLE (LIMBO) ============
    P = S['prologue']
    amb.add(P + 0.3, drone(fs(['D2', 'A2', 'D3']), 29.0, 0.42, cutoff=380, attack=5, release=5))
    amb.add(P + 0.0, wind(30.0, 0.16, seed=1, lo=180, hi=900, gust=0.5))
    for k, tb in enumerate(np.arange(P + 0.9, P + 9.6, 1.05)):
        sfx.add(tb, heartbeat(0.55 * min(1, (k + 1) / 3) * (1 if tb < P + 8 else 0.5)))
    music.add(P + 1.0, piano(hz('D2'), 3.5, 0.5, bright=0.5))
    music.add(P + 1.02, piano(hz('A2'), 3.5, 0.3, bright=0.5))
    music.add(P + 4.6, piano(hz('F2'), 3.5, 0.45, bright=0.5))
    music.add(P + 4.62, piano(hz('C3'), 3.5, 0.25, bright=0.5))
    amb.add(P + 7.0, wind(22.0, 0.22, seed=2, lo=500, hi=2600, gust=0.7, rate=0.2))    # forest canopy
    sfx.add(P + 9.4, reverse_cymbal(1.0, 0.25))
    sfx.add(P + 10.4, bell(hz('D6'), 4.0, 0.35, ratio=2.0, index=1.0, decay=1.0))
    sfx.add(P + 10.42, bell(hz('A6'), 4.0, 0.18, ratio=2.0, index=0.8, decay=1.2, pan=0.3))
    music.add(P + 10.5, pad(fs(['D3', 'A3', 'E4']), 6.0, 0.25, attack=2.5, release=3.5, cutoff=1200))
    for k in range(6):
        sfx.add(P + 11.9 + k * 0.45, noise_burst(0.2, 400, 4000, 0.08, 0.02, 12, -0.2 + k * 0.08))
    music.add(P + 15.2, pad(fs(CH['Dm']), 5.0, 0.35, attack=3.5, release=1.0, cutoff=1400))
    T = S['title']
    sfx.add(T + 0.6, reverse_cymbal(1.0, 0.4))
    sfx.add(T + 1.6, boom(4.0, 90, 30, 0.9))
    sfx.add(T + 1.6, braam(hz('D1'), 4.0, 0.55))
    for k, n in enumerate(['D5', 'A5', 'D6']):
        sfx.add(T + 1.6 + k * 0.04, bell(hz(n), 6.0, 0.3, ratio=1.0, index=0.6, decay=0.6, pan=(k - 1) * 0.4))
    music.add(T + 1.7, pad(fs(CH['Dm']), 7.0, 0.35, attack=0.4, release=3.0, cutoff=1600))
    for k, n in enumerate(['D4', 'F4', 'A4', 'D5']):
        music.add(T + 2.6 + k * 0.22, piano(hz(n), 3.0, 0.4))
    music.add(T + 3.8, pad(fs(CH['Bb']), 5.0, 0.28, attack=1.5, release=2.5, cutoff=1500))
    sfx.add(T + 3.2, whoosh(1.8, 2000, 9000, 0.12))

    # ============ CHAPTER I · FOREST OF RULES ============
    R = S['rules']
    rules_sc = SC['rules']
    amb.add(R, wind(37.0, 0.13, seed=3, lo=200, hi=1500, gust=0.4))
    amb.add(R, drone(fs(['D2', 'A2']), 36.0, 0.3, cutoff=320, attack=2, release=4))
    for k in range(int(26 / 0.5)):
        tt = R + 0.5 + k * 0.5
        sfx.add(tt, click(0.07 if k % 2 else 0.1, 900 if k % 2 else 1300, pan=-0.3 + 0.6 * (k % 2)))
    sfx.add(R + 0.5, whoosh(1.4, 200, 2500, 0.2))
    music.add(R + 0.6, piano(hz('D2'), 4.0, 0.45, bright=0.5))
    music.add(R + 0.62, piano(hz('A2'), 4.0, 0.3, bright=0.5))
    for st in steps_from_walker(rules_sc.walker, 0.0, 14.3):
        sfx.add(R + st, footstep('grass', 0.28, -0.1))
    # ELIZA powers on
    sfx.add(R + 14.3, noise_burst(0.3, 60, 400, 0.4, 0.001, 12))
    sfx.add(R + 14.3, bell(hz('E7'), 1.2, 0.05, ratio=1.0, index=0.1, decay=2.5))
    hum_n = n_of(11.5)
    tt = np.arange(hum_n) / SR
    hum = (np.sin(2 * np.pi * 50 * tt) * 0.6 + np.sin(2 * np.pi * 100 * tt) * 0.3 + np.sin(2 * np.pi * 150 * tt) * 0.15)
    hum *= np.minimum(1, tt / 0.8) * np.minimum(1, (tt[-1] - tt) / 1.0) * 0.035
    amb.add(R + 14.4, np.stack([hum, hum], 1).astype(np.float32))
    from cg.scenes.rules import CRT_LINES
    for (ts, text, style) in CRT_LINES:
        if style == 'dim':
            continue
        for i in range(len(text)):
            sfx.add(R + ts + i / 16, click(0.05, 2400 + (i % 3) * 300, pan=0.35))
    for (t0, text) in ((15.6, '……我觉得很孤独。'), (19.3, '你为什么觉得很孤独？')):
        for i in range(len(text)):
            sfx.add(R + t0 + 0.35 + i / 13, click(0.09, 1800, pan=0.0))
    music.add(R + 19.3, pad(fs(['D3', 'F3', 'C4']), 6.5, 0.18, attack=1.5, release=2.5, cutoff=1000))
    music.add(R + 23.0, piano(hz('Bb2'), 3.0, 0.35, bright=0.5))
    # the perceptron lights up
    sfx.add(R + 25.7, noise_burst(0.5, 40, 200, 0.4, 0.001, 6))
    sfx.add(R + 27.5, bell(hz('A5'), 5.0, 0.4, ratio=1.0, index=0.7, decay=0.8))
    sfx.add(R + 27.52, bell(hz('E6'), 5.0, 0.2, ratio=1.0, index=0.5, decay=1.0, pan=0.3))
    music.add(R + 27.6, pad(fs(['D3', 'A3', 'E4', 'F4']), 4.0, 0.3, attack=1.0, release=2.0, cutoff=1800))
    sfx.add(R + 29.8, whoosh(0.9, 800, 6000, 0.12, pan=0.4))
    from cg.scenes.rules import ATTEMPTS
    for k, (ta, ang, off) in enumerate(ATTEMPTS):
        sfx.add(R + ta - 0.8, whoosh(0.7, 500, 1500, 0.06))
        for n in (['D2', 'Eb2', 'Ab2'] if k % 2 == 0 else ['C#2', 'D2', 'G#2']):
            music.add(R + ta, piano(hz(n), 1.4, 0.5, bright=0.4))
        sfx.add(R + ta, noise_burst(0.25, 80, 600, 0.35, 0.001, 14))
    sfx.add(R + 33.9, ice_crack(0.7, 1))
    sfx.add(R + 34.3, ice_crack(0.5, 2))
    amb.add(R + 33.0, wind(4.0, 0.3, seed=4, lo=300, hi=3000, gust=0.8, rate=0.4))
    sfx.add(R + 33.8, riser(2.2, 0.12, 2000, 9000))

    # ============ CHAPTER II · THE LONG WINTER ============
    Wn = S['winter']
    wsc = SC['winter']
    amb.add(Wn, wind(14.0, 0.42, seed=5, lo=250, hi=3500, gust=0.9, rate=0.18))
    amb.add(Wn + 12.5, wind(18.0, 0.14, seed=6, lo=300, hi=2500, gust=0.5))
    sfx.add(Wn + 0.6, whoosh(1.5, 300, 3000, 0.25))
    music.add(Wn + 0.7, piano(hz('A1'), 5.0, 0.45, bright=0.4))
    music.add(Wn + 0.72, piano(hz('E2'), 5.0, 0.3, bright=0.4))
    music.add(Wn + 5.6, piano(hz('D3'), 3.0, 0.25, bright=0.5))
    music.add(Wn + 9.4, piano(hz('F3'), 3.0, 0.25, bright=0.5))
    music.add(Wn + 6.0, pad(fs(['A3', 'C#4', 'E4']), 7.0, 0.09, attack=3, release=3, cutoff=900))   # warm lamps
    for st in steps_from_walker(wsc.walk, 0.0, 13.0):
        sfx.add(Wn + st, footstep('snow', 0.25, -0.2))
    amb.add(Wn + 13.0, drone(fs(['D5', 'A5', 'E6']), 4.0, 0.06, cutoff=6000, attack=1, release=2))  # glassy ice
    # the touch -> red spark
    sfx.add(Wn + 15.1, reverse_cymbal(0.8, 0.2))
    sfx.add(Wn + 15.9, boom(3.0, 70, 35, 0.5))
    sfx.add(Wn + 15.9, bell(hz('D6'), 4.0, 0.45, ratio=1.41, index=2.5, decay=0.9))
    sfx.add(Wn + 15.95, bell(hz('A5'), 4.0, 0.3, ratio=1.41, index=2.0, decay=1.0, pan=-0.3))
    for k, n in enumerate(['A5', 'F#5', 'D5']):
        sfx.add(Wn + 16.2 + k * 0.28, celesta(hz(n), 0.35, pan=0.3 - k * 0.2))
    for k, n in enumerate(['A4', 'F#4', 'E4', 'D4']):
        sfx.add(Wn + 17.1 + k * 0.25, celesta(hz(n), 0.3, pan=-0.1 - k * 0.1))
    sfx.add(Wn + 17.0, ice_crack(0.45, 3))
    sfx.add(Wn + 18.1, ice_crack(0.45, 4))
    music.add(Wn + 16.0, pad(fs(['D3', 'A3', 'D4', 'E4']), 7.0, 0.3, attack=1.5, release=3, cutoff=1600))
    SKILL = ['D5', 'F#5', 'A5', 'D6']

    def skill_chime(t):
        for k, n in enumerate(SKILL):
            sfx.add(t + k * 0.07, bell(hz(n), 3.0, 0.22, ratio=1.0, index=0.9, decay=1.2, pan=-0.3 + k * 0.2))
        sfx.add(t, whoosh(0.8, 1500, 8000, 0.08))
    skill_chime(Wn + 18.8)
    for k, n in enumerate(['D4', 'F#4', 'A4', 'D5']):
        sfx.add(Wn + 20.2 + k * 0.4, celesta(hz(n), 0.33, pan=-0.3 + k * 0.2))
    music.add(Wn + 21.8, pad(fs(CH['D']), 6.0, 0.35, attack=0.8, release=2.5, cutoff=2000))
    music.add(Wn + 21.8, piano(hz('D3'), 3, 0.35))
    music.add(Wn + 21.9, piano(hz('F#4'), 3, 0.3))
    music.add(Wn + 22.0, piano(hz('A4'), 3, 0.3))
    sfx.add(Wn + 23.6, ice_crack(0.8, 5))
    sfx.add(Wn + 23.8, crash(3.0, 0.25))
    sfx.add(Wn + 24.2, noise_burst(2.5, 30, 250, 0.35, 0.3, 1.2))
    music.add(Wn + 24.2, choir(fs(['D3', 'A3', 'D4', 'F#4', 'A4']), 5.2, 0.55, attack=3.0, release=2.0))
    music.add(Wn + 24.2, pad(fs(['D3', 'A3', 'D4', 'F#4']), 5.2, 0.4, attack=3.0, release=2.0, cutoff=2400))
    sfx.add(Wn + 27.8, reverse_cymbal(2.0, 0.3))

    # ============ CHAPTER III · THE CHAIN OF MEMORY ============
    M = S['memory']
    msc = SC['memory']
    from cg.scenes import memory as mem
    amb.add(M, wind(36.0, 0.12, seed=7, lo=150, hi=900, gust=0.4))
    amb.add(M, drone(fs(['A2', 'E3']), 13.0, 0.3, cutoff=360, attack=2, release=3))
    amb.add(M, waves(36.0, 0.08, seed=8))
    sfx.add(M + 0.9, whoosh(1.4, 200, 2500, 0.2))
    for st in steps_from_walker(msc.walk, 0.0, 12.6):
        sfx.add(M + st, footstep('wood', 0.3, -0.1))
    for st in steps_from_walker(msc.run, 13.4, 22.6):
        sfx.add(M + st, footstep('wood', 0.38, 0.0))
    penta = ['D5', 'E5', 'F#5', 'A5', 'B5', 'D6']
    prev_j = -1
    tt = 0.0
    while tt < 23.0:
        j = msc.read_index(tt)
        if j > prev_j and j >= 0:
            if msc.destroyed(tt, mem.post_x(mem.token_post(j))) is None:
                sfx.add(M + tt, celesta(hz(penta[j % len(penta)]), 0.22 if tt < 13 else 0.15, pan=0.2))
            prev_j = j
        tt += 1 / 120
    sfx.add(M + 6.0, bell(hz('A5'), 3.0, 0.12, ratio=1.0, index=0.5, decay=1.5))
    # spider
    sfx.add(M + 11.6, riser(1.4, 0.12, 100, 1500))
    sfx.add(M + 12.4, noise_burst(3.0, 40, 300, 0.4, 0.8, 1.0))
    music.add(M + 12.4, pad(fs(['D2', 'Eb2', 'A2', 'Bb2']), 11.0, 0.35, attack=0.8, release=2.0, cutoff=700))
    sfx.add(M + 13.3, braam(hz('D1'), 4.0, 0.9))
    sfx.add(M + 13.3, boom(3.5, 100, 28, 0.9))
    for ts in mem.STABS:
        sfx.add(M + ts - 0.3, whoosh(0.35, 300, 1800, 0.18, pan=-0.4))
        sfx.add(M + ts, boom(2.2, 110, 38, 0.55))
        sfx.add(M + ts, noise_burst(0.5, 200, 3000, 0.45, 0.001, 9, -0.2))
        sfx.add(M + ts + 0.02, ice_crack(0.35, int(ts * 10)))
    beat = 60 / 150
    for k in range(int((22.6 - 13.4) / (beat / 2))):
        t_ = M + 13.4 + k * beat / 2
        n = ['D2', 'D2', 'D3', 'D2'][k % 4]
        music.add(t_, strings_stac(hz(n), 0.18, 0.55 if k % 4 == 0 else 0.35))
        if k % 8 == 0:
            sfx.add(t_, boom(0.8, 70, 40, 0.35))
    for k, n in enumerate(['A3', 'Bb3', 'C4', 'D4']):
        music.add(M + 14.0 + k * 2.2, pad(fs([n, up([n])[0]]), 2.4, 0.18, attack=1.5, release=0.8, cutoff=3000))
    for k in range(10):
        sfx.add(M + 18.5 + k * 0.4, glitch(0.12, 0.08, seed=k))
    # LSTM gates & the cell state
    for (gx, cn, sym, tg) in mem.GATES:
        sfx.add(M + tg, noise_burst(0.4, 60, 500, 0.35, 0.001, 10))
        sfx.add(M + tg + 0.05, click(0.15, 700))
    sfx.add(M + 24.0, whoosh(1.0, 3000, 400, 0.15))
    music.add(M + 24.0, choir(fs(['A3', 'D4', 'F#4', 'A4']), 6.5, 0.4, attack=1.2, release=3.0))
    music.add(M + 24.0, pad(fs(['D3', 'A3', 'F#4']), 7.0, 0.3, attack=1.0, release=3.0, cutoff=2200))
    sfx.add(M + mem.T_LSTM_BLOCK, boom(3.0, 90, 30, 0.9))
    for k, n in enumerate(['D6', 'F#6', 'A6', 'D7']):
        sfx.add(M + mem.T_LSTM_BLOCK + k * 0.02, bell(hz(n), 3.0, 0.2, ratio=2.76, index=2.0, decay=1.5,
                                                   pan=-0.4 + k * 0.25))
    sfx.add(M + mem.T_LSTM_BLOCK + 0.3, whoosh(2.5, 1200, 150, 0.18))
    skill_chime(M + 27.0)
    for k in range(10):
        sfx.add(M + 28.4 + k * 0.2, celesta(hz(['D5', 'E5', 'F#5', 'A5', 'B5'][k % 5]) * (1 + k // 5), 0.12))
    for k, n in enumerate(['D5', 'F#5', 'A5', 'E6']):
        sfx.add(M + 30.8 + k * 0.05, bell(hz(n), 4.0, 0.3, ratio=1.0, index=0.6, decay=0.9, pan=-0.3 + k * 0.2))
    music.add(M + 30.8, pad(fs(CH['D']), 5.2, 0.3, attack=0.5, release=2.0, cutoff=2000))
    music.add(M + 31.5, piano(hz('A3'), 2.0, 0.3))
    music.add(M + 33.9, piano(hz('F#3'), 2.0, 0.25))

    # ============ CHAPTER IV · THE SEA OF MEANING ============
    Se = S['sea']
    from cg.scenes import sea as sea_mod
    amb.add(Se, waves(30.5, 0.18, seed=9))
    sfx.add(Se + 1.0, whoosh(1.5, 200, 2500, 0.15))
    prog = ['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'Gm', 'A']
    beat = 60 / 66 / 2
    for ci, chn in enumerate(prog):
        t0 = Se + 0.8 + ci * 8 * beat
        notes = CH[chn]
        arp = [notes[0], notes[1], notes[2], up([notes[0]])[0], up([notes[1]])[0], notes[3], up([notes[2]])[0],
               notes[3]]
        for k, n in enumerate(arp):
            music.add(t0 + k * beat, piano(hz(n), beat * 3, 0.26 + (0.06 if k == 0 else 0)))
        music.add(t0, pad(fs(notes), 8 * beat, 0.16, attack=1.2, release=2.0, cutoff=1300))
    r = Rng(12)
    names = list(sea_mod.WORDS.keys())
    pent = ['D5', 'F5', 'G5', 'A5', 'C6', 'D6']
    for i, wd in enumerate(names):
        t0 = 0.8 + i * 0.3 + r.u(0, 0.3)
        r.u()
        r.u()
        sfx.add(Se + t0 + 2.4, celesta(hz(pent[i % len(pent)]), 0.14, pan=(i % 5 - 2) * 0.25))
    for (a_, b_, ta, lab) in sea_mod.ARROWS:
        sfx.add(Se + ta, whoosh(0.8, 1500, 7000, 0.05, pan=0.2))
        sfx.add(Se + ta + 0.8, celesta(hz('A5'), 0.1))
    music.add(Se + 11.6, choir(fs(['D4', 'F4', 'A4']), 3.0, 0.2, attack=1.0, release=2.0, vowel='o'))
    sfx.add(Se + 13.4, whoosh(1.4, 400, 3000, 0.15))
    for i in range(len(sea_mod.IN_TOK)):
        sfx.add(Se + 14.3 + i * 0.28 + 1.3, piano(hz(['D4', 'F4', 'A4', 'C5', 'D5', 'F5'][i]), 0.3, 0.2))
    sfx.add(Se + 16.6, whoosh(1.2, 600, 5000, 0.08))
    for j in range(len(sea_mod.OUT_TOK)):
        sfx.add(Se + 17.9 + j * 0.24, piano(hz(['A4', 'C5', 'D5', 'F5', 'A5', 'D6'][j]), 0.3, 0.18))
    for i in range(len(sea_mod.LONG_TOK)):
        sfx.add(Se + 19.5 + i * 0.09 + 1.1, click(0.05, 2000 + (i % 5) * 200))
    sfx.add(Se + 20.4, noise_burst(1.4, 40, 400, 0.3, 0.8, 1.5))
    sfx.add(Se + 21.3, ice_crack(0.5, 7))
    for j in range(len(sea_mod.OUT_TOK)):
        for k in range(4):
            sfx.add(Se + 22.6 + j * 0.72 + k * 0.05, celesta(hz(['D5', 'F5', 'A5', 'D6'][k]) * 2 ** (j / 12), 0.08,
                                                            pan=0.4 - k * 0.2))
    sfx.add(Se + 26.8, riser(2.4, 0.08, 800, 8000))
    sfx.add(Se + 28.8, bell(hz('A6'), 3.0, 0.2, ratio=1.0, index=0.4, decay=1.2))

    # ============ CHAPTER V · ATTENTION ============
    A = S['attention']
    amb.add(A + 0.2, drone(fs(['D2', 'A2']), 11.8, 0.4, cutoff=300, attack=3, release=1))
    txt = 'Attention Is All You Need'
    for i in range(len(txt)):
        if txt[i] != ' ':
            sfx.add(A + 1.0 + i / 11, click(0.14, 1600 + (i % 4) * 150, dur=0.05))
    for b in (3.9, 5.1):
        sfx.add(A + b, heartbeat(0.9))
    sfx.add(A + 5.9, whoosh(0.9, 3000, 300, 0.15))
    for i in range(16):
        sfx.add(A + 7.0 + i * 0.1, celesta(hz(['D5', 'E5', 'F#5', 'A5', 'B5'][i % 5]) * (1 + (i // 5) * 0.0), 0.1,
                                          pan=-0.7 + i * 0.09))
    sfx.add(A + 8.8, heartbeat(0.7))
    for k in range(3):
        sfx.add(A + 9.1 + k * 0.5, whoosh(0.8, 800, 6000, 0.07, pan=0.3 - k * 0.3))
    music.add(A + 9.0, pad(fs(['D3', 'A3', 'E4']), 3.0, 0.22, attack=1.5, release=0.3, cutoff=1500))
    sfx.add(A + 9.8, riser(2.25, 0.3, 200, 9000))
    sfx.add(A + 10.2, reverse_cymbal(1.8, 0.55))
    # THE BLOOM
    tb = A + 12.0
    sfx.add(tb, boom(5.0, 110, 30, 1.0))
    sfx.add(tb, crash(5.0, 0.55))
    music.add(tb, choir(fs(['D3', 'A3', 'D4', 'F#4', 'A4', 'D5']), 6.5, 0.85, attack=0.15, release=3.0))
    music.add(tb, pad(fs(['D2', 'A2', 'D3', 'F#3', 'A3', 'D4']), 6.5, 0.6, attack=0.1, release=3.0, cutoff=3200))
    for k, n in enumerate(['D5', 'F#5', 'A5', 'D6', 'F#6', 'A6']):
        sfx.add(tb + k * 0.06, bell(hz(n), 5.0, 0.2, ratio=1.0, index=0.6, decay=0.8, pan=-0.6 + k * 0.24))
    play_theme(music, tb + 0.4, 96, 0, 2, vel=0.5, strings=False)
    music.add(tb + 3.6, pad(fs(CH['Bm']), 2.6, 0.4, attack=0.4, release=1.5, cutoff=2600))
    music.add(tb + 5.4, pad(fs(CH['G']), 2.0, 0.4, attack=0.4, release=1.5, cutoff=2600))
    music.add(tb + 6.6, pad(fs(CH['A']), 1.4, 0.45, attack=0.3, release=1.2, cutoff=2600))
    # tower ostinato: 120 bpm, bars of 2 s
    t_tower = A + 19.6
    bar = 2.0
    prog = ['D', 'Bm', 'G', 'A', 'Bm', 'A']
    for bi, chn in enumerate(prog):
        t0 = t_tower + bi * bar
        notes = CH[chn]
        root, fifth = notes[0], notes[2] if chn != 'A' else 'E3'
        patt = [root, up([root])[0], fifth, up([root])[0]] * 2
        for k, n in enumerate(patt):
            music.add(t0 + k * 0.25, strings_stac(hz(n), 0.2, 0.45 + 0.08 * bi / 5, pan=(-0.3 if k % 2 else 0.3)))
        sfx.add(t0, boom(1.2, 80, 38, 0.35 + 0.08 * bi))
        music.add(t0, pad(fs(notes), bar, 0.3 + 0.05 * bi, attack=0.15, release=0.8, cutoff=2200 + 300 * bi))
        if bi >= 2:
            music.add(t0, choir(fs(up(notes[1:], 1)), bar, 0.25 + 0.08 * (bi - 2), attack=0.3, release=0.9))
        mel = [('A4', 0), ('B4', 0.5), ('D5', 1.0), ('F#5', 1.5)] if bi % 2 == 0 else [('E5', 0), ('D5', 1.0)]
        for (n, off) in mel:
            music.add(t0 + off, piano(hz(n), 0.6, 0.35))
    sfx.add(t_tower + 8.0, riser(4.0, 0.3, 300, 9000))
    sfx.add(A + 29.8, reverse_cymbal(1.8, 0.6))
    # apex
    ta = A + 31.6
    sfx.add(ta, boom(6.0, 120, 28, 1.0))
    sfx.add(ta, crash(6.0, 0.6))
    music.add(ta, choir(fs(['D3', 'A3', 'D4', 'F#4', 'A4', 'D5', 'E5']), 11.0, 0.9, attack=0.2, release=4.0))
    music.add(ta, pad(fs(['D2', 'A2', 'D3', 'F#3', 'A3', 'E4', 'D4']), 11.0, 0.55, attack=0.1, release=4.0, cutoff=3000))
    for k, n in enumerate(['D5', 'A5', 'D6', 'F#6', 'A6', 'D7']):
        sfx.add(ta + k * 0.08, bell(hz(n), 6.0, 0.18, ratio=1.0, index=0.7, decay=0.6, pan=-0.6 + k * 0.24))
    sfx.add(A + 32.4, boom(3.0, 90, 40, 0.45))
    sfx.add(A + 32.4, braam(hz('D2'), 3.0, 0.3))
    play_theme(music, A + 33.0, 84, 4, 8, vel=0.45, strings=False)
    skill_chime(A + 34.2)
    sfx.add(A + 42.6, reverse_cymbal(1.4, 0.35))
    sfx.add(A + 43.2, whoosh(1.5, 800, 9000, 0.15))

    # ============ CHAPTER VI · GIANTS ============
    G = S['giants']
    amb.add(G, waves(26.0, 0.16, seed=10))
    sfx.add(G + 0.3, noise_burst(6.0, 30, 220, 0.3, 1.5, 0.35))       # stone rising
    for k in range(8):
        sfx.add(G + 0.5 + k * 0.7, noise_burst(0.5, 800, 6000, 0.08, 0.05, 6, -0.6 + k * 0.17))  # water pouring
    music.add(G + 0.4, pad(fs(CH['Dm']), 5.0, 0.35, attack=2.0, release=2.0, cutoff=1200))
    music.add(G + 0.4, choir(fs(['D3', 'A3', 'D4']), 5.0, 0.3, attack=2.0, release=2.0, vowel='o'))
    music.add(G + 5.0, pad(fs(CH['Bb']), 4.8, 0.3, attack=1.0, release=2.0, cutoff=1400))
    sfx.add(G + 6.9, celesta(hz('D6'), 0.3))
    for i in range(6):
        sfx.add(G + 6.0 + i * 0.45, piano(hz(['D5', 'E5', 'F5', 'G5', 'A5', 'C6'][i]), 0.3, 0.2))
    for (ts, key) in ((9.5, 'F'), (12.5, 'G'), (15.5, 'A')):
        sfx.add(G + ts, boom(3.0, 90, 30, 0.85))
        sfx.add(G + ts, braam(hz({'F': 'F1', 'G': 'G1', 'A': 'A1'}[key]), 2.8, 0.35))
        music.add(G + ts, pad(fs(CH[key]), 3.0, 0.45, attack=0.2, release=1.5, cutoff=2200))
        music.add(G + ts, choir(fs(up(CH[key][1:], 1)), 3.0, 0.35, attack=0.3, release=1.5))
        for k in range(12):
            sfx.add(G + ts + 0.3 + k * 0.08, click(0.05, 3000, pan=-0.5))
    tb = G + 18.3
    music.add(tb, pad(fs(CH['D']), 4.0, 0.45, attack=0.3, release=2.0, cutoff=2600))
    music.add(tb, choir(fs(['A3', 'D4', 'F#4', 'A4']), 3.5, 0.4, attack=0.3, release=2.0))
    rr = Rng(21)
    for k in range(26):
        sfx.add(tb + rr.u(0, 2.8), celesta(hz(['D6', 'E6', 'F#6', 'A6', 'B6'][k % 5]), 0.1, pan=rr.u(-0.8, 0.8)))
    # storm of hallucination
    music.add(G + 21.2, pad(fs(['D2', 'Eb3', 'A3', 'Bb3']), 9.5, 0.4, attack=1.2, release=2.0, cutoff=900))
    amb.add(G + 21.0, wind(9.5, 0.3, seed=11, lo=150, hi=1500, gust=1.0, rate=0.3))
    for k in range(18):
        sfx.add(G + 21.5 + k * 0.25, glitch(0.15, 0.1, seed=100 + k))
    for tl in (22.3, 23.6, 24.9):
        sfx.add(G + tl, noise_burst(3.0, 30, 400, 0.7, 0.01, 1.2))
        sfx.add(G + tl, noise_burst(0.3, 500, 6000, 0.35, 0.001, 12))

    # ============ CHAPTER VII · ALIGNMENT ============
    Al = S['alignment']
    from cg.scenes import alignment as al_mod
    amb.add(Al, wind(10.0, 0.18, seed=12, lo=150, hi=1200, gust=0.6))
    music.add(Al + 0.5, pad(fs(CH['Dm']), 4.0, 0.3, attack=1.0, release=1.5, cutoff=1200))
    music.add(Al + 4.3, pad(fs(CH['Bb']), 3.0, 0.3, attack=0.8, release=1.5, cutoff=1300))
    music.add(Al + 7.2, pad(fs(CH['Gm']), 3.0, 0.3, attack=0.8, release=1.5, cutoff=1400))
    music.add(Al + 10.0, pad(fs(CH['A']), 3.0, 0.32, attack=0.8, release=1.5, cutoff=1600))
    rr = Rng(33)
    for k in range(40):
        sfx.add(Al + 1.0 + rr.u(0, 8.5), celesta(hz(['D6', 'E6', 'F#6', 'A6', 'B6', 'D7'][k % 6]), 0.06,
                                                pan=rr.u(-0.9, 0.9)))
    for (tp, good, bad, gl) in al_mod.PAIRS:
        sfx.add(Al + tp, whoosh(0.6, 1500, 5000, 0.05))
        sfx.add(Al + tp + 1.0, bell(hz('F#5'), 2.0, 0.2, ratio=1.0, index=0.5, decay=1.5, pan=-0.2))
        sfx.add(Al + tp + 1.08, bell(hz('A5'), 2.0, 0.2, ratio=1.0, index=0.5, decay=1.5, pan=0.2))
        sfx.add(Al + tp + 1.0, piano(hz('C3'), 0.4, 0.15, bright=0.3))
    tb = Al + 12.8
    music.add(tb, pad(fs(CH['D']), 4.0, 0.35, attack=2.0, release=1.5, cutoff=2000))
    music.add(tb, choir(fs(['D4', 'F#4', 'A4']), 4.0, 0.25, attack=2.0, release=2.0))
    txt = '你好！有什么我可以帮你的吗？'
    for i in range(len(txt)):
        sfx.add(Al + 14.35 + i / 11, click(0.08, 1900))
    play_theme(music, Al + 14.2, 80, 0, 4, vel=0.5, strings=True)
    sfx.add(Al + 24.4, reverse_cymbal(1.6, 0.3))

    # ============ CHAPTER VIII · LIBRARY ============
    L = S['library']
    from cg.scenes import library as lib_mod
    sfx.add(L + 0.6, whoosh(1.4, 200, 2500, 0.15))
    beat = 60 / 104 / 4
    prog = ['D', 'G', 'Bm', 'A', 'D', 'G', 'A']
    for ci, chn in enumerate(prog):
        t0 = L + 0.4 + ci * 16 * beat * 1.0
        notes = CH[chn]
        patt = [up([notes[0]])[0], notes[2], notes[1], notes[2]] * 4
        for k, n in enumerate(patt):
            music.add(t0 + k * beat, strings_stac(hz(up([n])[0]), 0.1, 0.18, pan=(-0.4 if k % 2 else 0.4)))
        music.add(t0, pad(fs(notes), 16 * beat, 0.22, attack=0.3, release=1.0, cutoff=1800))
        music.add(t0, piano(hz(notes[0].replace('3', '2')), 16 * beat, 0.25))
    rr = Rng(3)
    for k in range(14):
        sfx.add(L + rr.u(0.5, 13.5), noise_burst(0.12, 1500, 6000, 0.05, 0.005, 25, rr.u(-0.8, 0.8)))
    for i, (cn, en, ts) in enumerate(lib_mod.STEPS):
        sfx.add(L + ts, bell(hz(['D5', 'E5', 'F#5', 'A5', 'B5'][i]), 2.0, 0.22, ratio=1.0, index=0.7, decay=1.8))
    sfx.add(L + 7.2, whoosh(0.8, 1000, 8000, 0.1))
    for n_ in range(3):
        sfx.add(L + 8.7 + n_ * 0.3, whoosh(0.9, 600, 4000, 0.07, pan=0.4))
    ans = '提交发票 → 主管审批 → 财务打款（约 5 个工作日） [1][3]'
    for i in range(len(ans)):
        if ans[i] != ' ':
            sfx.add(L + 10.3 + i / 14, click(0.05, 2100))
    skill_chime(L + 10.6)
    for k, (tc, chn) in enumerate(((14.0, 'D'), (16.4, 'Bm'), (18.8, 'G'), (21.2, 'A'), (23.6, 'Dadd9'))):
        sfx.add(L + tc - 0.3, whoosh(0.6, 400, 6000, 0.14))
        music.add(L + tc, pad(fs(CH[chn]), 2.4, 0.32, attack=0.1, release=1.2, cutoff=2400))
        music.add(L + tc, piano(hz(CH[chn][0]), 2.0, 0.35))
    for k, n in enumerate(['D4', 'F#4', 'A4', 'D5']):
        sfx.add(L + 14.2 + k * 0.5, celesta(hz(n), 0.2))
    for k in range(4):
        sfx.add(L + 16.6 + k * 0.1, click(0.12, 4000 + k * 300))
    sfx.add(L + 17.0, whoosh(0.4, 3000, 8000, 0.1))
    for k in range(6):
        sfx.add(L + 19.0 + k * 0.33, click(0.1, 1200 + (k % 3) * 400, pan=-0.5 + k * 0.2))
    sfx.add(L + 21.4, click(0.25, 5000, dur=0.05))
    for k in range(10):
        sfx.add(L + 23.7 + k * 0.23 * (1 + k * 0.08), click(0.1, 2600, pan=0.2))

    # ============ EPILOGUE ============
    E = S['epilogue']
    amb.add(E, waves(13.0, 0.15, seed=13))
    play_theme(music, E + 0.3, 76, 4, 8, vel=0.5, strings=True, choir_on=True)
    music.add(E + 12.8, piano(hz('D2'), 5.0, 0.4, bright=0.5))
    music.add(E + 13.2, bell(hz('D6'), 6.0, 0.25, ratio=1.0, index=0.4, decay=0.5))
    music.add(E + 13.0, pad(fs(['D3', 'A3', 'E4', 'F#4']), 9.0, 0.22, attack=3.0, release=3.0, cutoff=1400))
    txt = '下一个词，由你写下。'
    for i in range(len(txt)):
        sfx.add(E + 12.6 + 3.0 + i / 7, click(0.09, 1700))
    for k in range(9):
        sfx.add(E + 12.6 + 3.0 + 1.6 + k * 1.052, click(0.03, 1200))
    music.add(E + 12.6 + 7.2, pad(fs(['D3', 'A3', 'D4', 'E4', 'F#4', 'A4']), 4.0, 0.25, attack=1.5, release=4.0,
                                  cutoff=1800))
    music.add(E + 12.6 + 7.2, bell(hz('A5'), 6.0, 0.15, ratio=1.0, index=0.3, decay=0.6))
    return total, music, sfx, amb


GAINS = {  # per-scene bus gains in dB: (music, sfx, amb)
    'prologue': (3, 0, -5), 'title': (3, 0, -5), 'rules': (2, 2, -4), 'winter': (2, 0, -1), 'memory': (1, -3, 0),
    'sea': (4, 2, 0), 'attention': (0, -1, -3), 'giants': (2, -2, 0), 'alignment': (6, 2, 0), 'library': (7, 3, 0),
    'epilogue': (6, 3, 0),
}


def automate(buf, which, xfade=1.0):
    n = len(buf)
    g = np.ones(n, np.float32)
    for sc, st in zip(timeline.scenes(), timeline.starts()):
        a, b = int(st * SR), min(n, int((st + sc.DUR) * SR))
        g[a:b] = 10 ** (GAINS[sc.NAME][which] / 20)
    g[int(timeline.total() * SR):] = g[int(timeline.total() * SR) - 1]
    k = int(xfade * SR)
    kern = np.hanning(k)
    kern /= kern.sum()
    from scipy import signal as _sig
    g = _sig.oaconvolve(g, kern, mode='same').astype(np.float32)
    return buf * g[:, None]


def render(out: Path):
    t0 = time.time()
    total, music, sfx, amb = compose()
    print(f'composed in {time.time() - t0:.1f}s', flush=True)
    from audio.synth import highpass
    music.buf = automate(music.buf, 0)
    sfx.buf = automate(sfx.buf, 1)
    amb.buf = automate(highpass(amb.buf, 45).astype(np.float32), 2)
    sfx.buf = highpass(sfx.buf, 28).astype(np.float32)
    hall = impulse(4.2, damp=5200, seed=5)
    room = impulse(1.2, damp=7000, seed=7)
    m = music.buf
    m = m * 0.8 + convolve(m, hall) * 0.55
    s = sfx.buf
    s = s * 0.9 + convolve(s, room) * 0.25 + convolve(s, hall) * 0.22
    a = amb.buf
    a = a * 0.9 + convolve(a, hall) * 0.25
    mix = m * 1.0 + s * 0.85 + a * 0.75
    n = int(total * SR)
    mix = mix[:n]
    # master: gentle glue + soft limiter
    peak = np.max(np.abs(mix)) + 1e-9
    mix = mix / peak * 1.25
    mix = np.tanh(mix) / np.tanh(1.25)
    mix *= 10 ** (-1.0 / 20)
    # fades
    fi, fo = int(0.05 * SR), int(1.5 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out), SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    print(f'wrote {out} ({total:.1f}s) in {time.time() - t0:.1f}s')
    # loudness report per scene
    for sc, st in zip(timeline.scenes(), timeline.starts()):
        seg_ = mix[int(st * SR):int((st + sc.DUR) * SR)]
        rms = 20 * np.log10(np.sqrt(np.mean(seg_ ** 2)) + 1e-9)
        pk = 20 * np.log10(np.max(np.abs(seg_)) + 1e-9)
        print(f'  {sc.NAME:10s} rms {rms:6.1f} dBFS   peak {pk:6.1f} dBFS')


if __name__ == '__main__':
    render(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'out' / 'score.wav'))
