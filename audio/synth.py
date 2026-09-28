"""Tiny numpy synthesiser: instruments, effects and a mixing bus.

Everything returns float32 stereo arrays shaped (n, 2) at SR.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

SR = 48000
RNG = np.random.default_rng(2017)


def n_of(sec):
    return max(1, int(round(sec * SR)))


def tarr(n):
    return np.arange(n, dtype=np.float64) / SR


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def hz(name):
    """'D4', 'F#3', 'Bb2' -> frequency."""
    p = name[:-1]
    o = int(name[-1])
    return midi(12 * (o + 1) + NOTE[p])


def stereo(x, pan=0.0):
    """Equal-power pan, pan in [-1, 1]."""
    a = (pan + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], 1).astype(np.float32)


def env_ar(n, attack, release, hold=None, curve=2.0):
    """Attack / sustain / release envelope over n samples (release at the end)."""
    e = np.ones(n)
    na = min(n, n_of(attack))
    nr = min(n - na, n_of(release))
    if na > 0:
        e[:na] = (np.arange(na) / na) ** curve
    if nr > 0:
        e[n - nr:] *= (1 - np.arange(nr) / nr) ** curve
    return e


def lowpass(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def highpass(x, fc, order=2):
    sos = signal.butter(order, max(fc, 5), 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def bandpass(x, lo, hi, order=2):
    sos = signal.butter(order, [max(lo, 5), min(hi, SR * 0.45)], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def peak(x, fc, q=8.0, gain=1.0):
    b, a = signal.iirpeak(fc, q, fs=SR)
    return signal.lfilter(b, a, x, axis=0) * gain


def saw_phase(freq_arr):
    ph = np.cumsum(freq_arr / SR)
    return 2.0 * (ph - np.floor(ph)) - 1.0


def sweep_filter(x, f0, f1, kind='low', blocks=48):
    """Time-varying filter by overlap-added blocks (cheap & smooth)."""
    n = len(x)
    out = np.zeros_like(x, dtype=np.float64)
    bl = max(256, n // blocks)
    hop = bl // 2
    win = np.hanning(bl)
    for s in range(0, n, hop):
        seg = x[s:s + bl]
        if len(seg) < 8:
            break
        u = min(1.0, s / max(1, n - bl))
        fc = f0 * (f1 / f0) ** u
        if kind == 'low':
            y = lowpass(seg, fc)
        elif kind == 'high':
            y = highpass(seg, fc)
        else:
            y = bandpass(seg, fc * 0.7, fc * 1.4)
        w = win[:len(seg)]
        out[s:s + len(seg)] += y * (w[:, None] if y.ndim == 2 else w)
    return out


# ---------------------------------------------------------------- instruments

def piano(freq, dur=1.5, vel=0.7, bright=1.0, pan=None):
    ring = 3.2
    n = n_of(dur + ring)
    t = tarr(n)
    out = np.zeros(n)
    B = 0.00035
    fscale = (freq / 261.6) ** 0.55
    for k in range(1, 16):
        fk = k * freq * math.sqrt(1 + B * k * k)
        if fk > 14000:
            break
        amp = (1.0 / k ** (1.35 - 0.3 * bright)) * (0.9 if k == 1 else 1.0)
        dec = (0.55 + 0.42 * k) * fscale
        ph = RNG.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * fk * t + ph) * np.exp(-dec * t)
    # damper after key release
    off = n_of(dur)
    out[off:] *= np.exp(-np.arange(n - off) / SR * 6.0)
    # hammer
    hn = n_of(0.012)
    ham = lowpass(RNG.standard_normal(hn), 2500 * bright) * np.linspace(1, 0, hn) * 0.25
    out[:hn] += ham
    out[:n_of(0.003)] *= np.linspace(0, 1, n_of(0.003))
    out *= vel * 0.28
    if pan is None:
        pan = max(-0.7, min(0.7, (math.log2(freq / 261.6)) * 0.35))
    return stereo(out, pan)


def pad(freqs, dur, vel=0.5, attack=1.6, release=2.5, cutoff=1800, voices=4, detune=10.0, vib=0.25,
        bright_env=True):
    n = n_of(dur + release)
    t = tarr(n)
    L = np.zeros(n)
    R = np.zeros(n)
    for fi, f in enumerate(freqs):
        for v in range(voices):
            cents = (v - (voices - 1) / 2) * detune / max(1, voices - 1) * 2 + RNG.normal(0, 1.5)
            fr = f * 2 ** (cents / 1200)
            vibr = 1 + (vib / 100) * np.sin(2 * np.pi * (4.6 + 0.4 * v) * t + RNG.uniform(0, 6))
            w = saw_phase(fr * vibr) * 0.6 + np.sin(2 * np.pi * np.cumsum(fr * vibr / SR)) * 0.4
            if v % 2 == 0:
                L += w
            else:
                R += w
    x = np.stack([L, R], 1) / (len(freqs) * voices ** 0.5)
    x = lowpass(x, cutoff, order=2)
    e = env_ar(n, attack, release, curve=1.6)
    if bright_env:
        x = x * 0.85 + lowpass(x, cutoff * 0.35) * 0.15
    return (x * e[:, None] * vel * 0.5).astype(np.float32)


def strings_stac(freq, dur=0.22, vel=0.5, pan=0.0):
    """Short bowed/pizz-like note for ostinati."""
    n = n_of(dur + 0.4)
    t = tarr(n)
    w = saw_phase(np.full(n, freq)) * 0.5 + saw_phase(np.full(n, freq * 1.003)) * 0.5
    w = lowpass(w, 2600)
    e = np.minimum(1, t / 0.015) * np.exp(-t * 6.0)
    return stereo(w * e * vel * 0.3, pan)


def choir(freqs, dur, vel=0.5, attack=1.2, release=2.5, vowel='a'):
    forms = {'a': [(800, 1.0, 80), (1150, 0.5, 90), (2900, 0.25, 120), (3900, 0.12, 130)],
             'o': [(450, 1.0, 70), (800, 0.4, 80), (2830, 0.12, 100)],
             'u': [(325, 1.0, 60), (700, 0.3, 60), (2530, 0.08, 100)]}[vowel]
    n = n_of(dur + release)
    t = tarr(n)
    src = np.zeros((n, 2))
    for f in freqs:
        for v in range(6):
            cents = RNG.normal(0, 9)
            fr = f * 2 ** (cents / 1200) * (1 + 0.004 * np.sin(2 * np.pi * (5.2 + RNG.uniform(-0.5, 0.5)) * t
                                                                 + RNG.uniform(0, 6)))
            w = saw_phase(fr)
            src[:, v % 2] += w
    out = np.zeros_like(src)
    for (fc, g, bw) in forms:
        out += bandpass(src, fc - bw, fc + bw) * g
    out = lowpass(out, 5000)
    e = env_ar(n, attack, release, curve=1.5)
    out *= e[:, None]
    out /= (len(freqs) * 6) ** 0.5
    return (out * vel * 1.4).astype(np.float32)


def bell(freq, dur=3.0, vel=0.5, ratio=3.5, index=2.5, decay=1.6, pan=0.0):
    n = n_of(dur)
    t = tarr(n)
    mod = index * np.exp(-t * 4.0) * np.sin(2 * np.pi * freq * ratio * t)
    x = np.sin(2 * np.pi * freq * t + mod) * np.exp(-t * decay)
    x += 0.3 * np.sin(2 * np.pi * freq * 2.001 * t) * np.exp(-t * decay * 2.2)
    x[:n_of(0.002)] *= np.linspace(0, 1, n_of(0.002))
    return stereo(x * vel * 0.35, pan)


def celesta(freq, vel=0.4, pan=0.0):
    a = bell(freq, 2.2, vel, ratio=1.0, index=0.8, decay=2.4, pan=pan)
    b = bell(freq * 2, 1.2, vel * 0.3, ratio=1.0, index=0.3, decay=4.0, pan=pan)
    a[:len(b)] += b
    return a


def boom(dur=3.0, f0=80.0, f1=28.0, vel=1.0, noise=0.3):
    n = n_of(dur)
    t = tarr(n)
    f = f1 + (f0 - f1) * np.exp(-t * 7.0)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.3)
    nz = lowpass(RNG.standard_normal(n), 900) * np.exp(-t * 10) * noise
    x = np.tanh((x + nz) * 1.4)
    x[:n_of(0.004)] *= np.linspace(0, 1, n_of(0.004))
    return stereo(x * vel * 0.8, 0.0)


def braam(root=36.7, dur=3.5, vel=1.0):
    n = n_of(dur)
    t = tarr(n)
    x = np.zeros(n)
    for mult, a in ((1, 1.0), (1.5, 0.6), (2, 0.7), (3, 0.35), (4, 0.2)):
        for d in (-6, 0, 7):
            x += a * saw_phase(np.full(n, root * mult * 2 ** (d / 1200)))
    x /= 6
    y = sweep_filter(x, 180, 1400, 'low', blocks=40)
    e = np.minimum(1, t / 0.08) * np.exp(-t * 0.9)
    y = np.tanh(y * 2.2) * e
    return stereo(y * vel * 0.7, 0.0)


def noise_burst(dur, lo, hi, vel=0.5, attack=0.002, decay=20.0, pan=0.0):
    n = n_of(dur)
    t = tarr(n)
    x = bandpass(RNG.standard_normal(n), lo, hi) * np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t * decay)
    return stereo(x * vel, pan)


def whoosh(dur=1.2, f0=300, f1=4000, vel=0.5, pan=0.0):
    n = n_of(dur)
    x = RNG.standard_normal(n)
    y = sweep_filter(x, f0, f1, 'band', blocks=30)
    t = tarr(n)
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return stereo(y * e * vel * 1.2, pan)


def riser(dur=4.0, vel=0.6, f0=200, f1=6000):
    n = n_of(dur)
    t = tarr(n)
    nz = sweep_filter(RNG.standard_normal(n), f0, f1, 'band', blocks=60)
    fr = 110 * 2 ** (t / dur * 3)
    tone = saw_phase(fr) * 0.3
    tone = lowpass(tone, 3000)
    e = (t / dur) ** 2.2
    return np.stack([(nz + tone) * e, (nz * 0.9 + tone) * e], 1).astype(np.float32) * vel


def reverse_cymbal(dur=2.0, vel=0.5):
    n = n_of(dur)
    t = tarr(n)
    x = highpass(RNG.standard_normal((n, 2)), 3000) * ((t / dur) ** 3)[:, None]
    return (x * vel * 0.5).astype(np.float32)


def crash(dur=4.0, vel=0.6):
    n = n_of(dur)
    t = tarr(n)
    x = highpass(RNG.standard_normal((n, 2)), 2500) * np.exp(-t * 1.2)[:, None]
    x += bandpass(RNG.standard_normal((n, 2)), 400, 3000) * np.exp(-t * 3.0)[:, None] * 0.4
    return (x * vel * 0.35).astype(np.float32)


def heartbeat(vel=0.8):
    out = np.zeros((n_of(0.9), 2), np.float32)
    for off, a in ((0.0, 1.0), (0.24, 0.7)):
        n = n_of(0.35)
        t = tarr(n)
        f = 42 + 30 * np.exp(-t * 30)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14) * a
        s = n_of(off)
        out[s:s + n] += stereo(x * vel * 0.9)
    return out


def click(vel=0.3, freq=3000, pan=0.0, dur=0.03):
    n = n_of(dur)
    t = tarr(n)
    x = peak(RNG.standard_normal(n) * np.exp(-t * 400), freq, 6) + np.sin(2 * np.pi * freq * t) * np.exp(-t * 180) * 0.3
    return stereo(x * vel, pan)


def footstep(kind='grass', vel=0.35, pan=0.0):
    if kind == 'grass':
        a = noise_burst(0.12, 300, 3000, vel * 0.5, 0.004, 35, pan)
        b = noise_burst(0.1, 60, 250, vel * 0.9, 0.002, 40, pan)
    elif kind == 'snow':
        a = noise_burst(0.16, 1500, 7000, vel * 0.55, 0.01, 24, pan)
        b = noise_burst(0.1, 100, 500, vel * 0.5, 0.002, 40, pan)
    else:  # wood
        a = noise_burst(0.1, 150, 1200, vel * 0.8, 0.001, 45, pan)
        n = n_of(0.15)
        t = tarr(n)
        b = stereo(np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40) * vel * 0.6, pan)
    m = max(len(a), len(b))
    out = np.zeros((m, 2), np.float32)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def wind(dur, vel=0.4, seed=0, lo=200, hi=1200, gust=0.5, rate=0.12):
    n = n_of(dur)
    rng = np.random.default_rng(seed)
    t = tarr(n)
    x = rng.standard_normal((n, 2))
    b1 = bandpass(x, lo, lo * 2.2)
    b2 = bandpass(x, lo * 2.2, hi)
    m1 = 0.6 + gust * np.sin(2 * np.pi * rate * t + 1) * np.sin(2 * np.pi * rate * 0.37 * t)
    m2 = 0.5 + gust * np.sin(2 * np.pi * rate * 1.6 * t + 2)
    y = b1 * np.clip(m1, 0.05, None)[:, None] + b2 * np.clip(m2, 0.05, None)[:, None] * 0.6
    return (y * vel).astype(np.float32)


def waves(dur, vel=0.4, seed=1):
    n = n_of(dur)
    rng = np.random.default_rng(seed)
    t = tarr(n)
    x = lowpass(rng.standard_normal((n, 2)), 900)
    sw = 0.5 + 0.5 * np.sin(2 * np.pi * t / 6.5) ** 2
    y = x * sw[:, None] + highpass(rng.standard_normal((n, 2)), 3000) * 0.05 * sw[:, None]
    return (y * vel).astype(np.float32)


def drone(freqs, dur, vel=0.4, cutoff=300, attack=3.0, release=4.0):
    n = n_of(dur)
    t = tarr(n)
    x = np.zeros((n, 2))
    for i, f in enumerate(freqs):
        for d in (-4, 4):
            fr = f * 2 ** (d / 1200)
            x[:, 0 if d < 0 else 1] += np.sin(2 * np.pi * fr * t) + 0.35 * saw_phase(np.full(n, fr))
    x = lowpass(x, cutoff)
    e = env_ar(n, attack, release, curve=1.3)
    return (x * e[:, None] * vel * 0.3).astype(np.float32)


def glitch(dur=0.3, vel=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_of(dur)
    x = np.zeros(n)
    s = 0
    while s < n:
        ln = int(rng.uniform(0.01, 0.06) * SR)
        f = rng.uniform(200, 3000)
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(ln) / SR)) * rng.uniform(0.2, 1)
        if rng.random() < 0.4:
            seg = np.round(rng.standard_normal(ln) * 3) / 3
        x[s:s + ln] = seg[:max(0, min(ln, n - s))]
        s += ln
    return stereo(lowpass(x, 6000) * vel * 0.4, rng.uniform(-0.5, 0.5))


def ice_crack(vel=0.6, seed=0):
    rng = np.random.default_rng(seed)
    out = noise_burst(0.35, 1500, 9000, vel * 0.8, 0.0005, 18, rng.uniform(-0.4, 0.4))
    for k in range(6):
        f = rng.uniform(2500, 6000)
        b = bell(f, 0.8, vel * 0.2, ratio=2.7, index=1.2, decay=6, pan=rng.uniform(-0.6, 0.6))
        s = n_of(rng.uniform(0, 0.25))
        if s + len(b) > len(out):
            out = np.pad(out, ((0, s + len(b) - len(out)), (0, 0)))
        out[s:s + len(b)] += b
    return out


# ---------------------------------------------------------------- reverb & bus

def impulse(seconds=3.5, pre=0.02, damp=4000, seed=5, early=True):
    rng = np.random.default_rng(seed)
    n = n_of(seconds)
    t = tarr(n)
    ir = rng.standard_normal((n, 2)) * np.exp(-t * (6.9 / seconds))[:, None]
    ir = lowpass(ir, damp)
    if early:
        for k in range(10):
            d = n_of(rng.uniform(0.008, 0.08))
            ir[d, rng.integers(0, 2)] += rng.uniform(0.3, 0.8)
    p = n_of(pre)
    ir = np.concatenate([np.zeros((p, 2)), ir])
    ir /= np.sqrt((ir ** 2).sum(0))[None, :]
    return ir.astype(np.float32)


def convolve(x, ir):
    out = np.stack([signal.oaconvolve(x[:, c], ir[:, c], mode='full')[:len(x)] for c in range(2)], 1)
    return out.astype(np.float32)


class Bus:
    def __init__(self, seconds):
        self.buf = np.zeros((n_of(seconds) + SR * 8, 2), np.float32)

    def add(self, t, x, gain=1.0):
        if t < 0:
            cut = n_of(-t)
            x = x[cut:]
            t = 0
        s = n_of(t)
        e = min(len(self.buf), s + len(x))
        if e > s:
            self.buf[s:e] += x[:e - s] * gain
