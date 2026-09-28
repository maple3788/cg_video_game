#!/usr/bin/env python3
"""Render the TOKEN game-CG trailer.

Examples
--------
  python render.py --scale 0.5 --out out/preview.mp4             # fast half-res preview
  python render.py --stills 5,12,40 --scale 0.5                   # PNG stills at given seconds
  python render.py --start 60 --end 90 --scale 0.5 --out out/part.mp4
  python render.py --out out/token_trailer_1080p.mp4 --audio out/score.wav   # final
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


_SCALE = 1.0


def _init(scale):
    import cv2
    cv2.setNumThreads(1)
    from cg.core import set_render_scale
    set_render_scale(scale)


def render_frame(i: int) -> bytes:
    from cg.core import FPS, Frame
    from cg import timeline
    t = i / FPS
    scene, lt = timeline.locate(t)
    fr = Frame()
    scene.render(fr, lt, i)
    from cg import impact
    impact.apply(fr, t)
    return fr.rgb_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--out', default='out/token_trailer.mp4')
    ap.add_argument('--audio', default=None)
    ap.add_argument('--stills', default=None, help='comma separated seconds')
    ap.add_argument('--stills-dir', default='out/stills')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--preset', default='medium')
    args = ap.parse_args()

    _init(args.scale)
    from cg.core import FPS, W, H
    from cg import timeline

    if args.stills:
        from PIL import Image
        out = Path(args.stills_dir)
        out.mkdir(parents=True, exist_ok=True)
        times = [float(x) for x in args.stills.split(',') if x.strip()]
        idxs = [int(round(tt * FPS)) for tt in times]
        with mp.get_context('fork').Pool(min(args.workers, len(idxs)), initializer=_init,
                                         initargs=(args.scale,)) as pool:
            frames = pool.map(render_frame, idxs)
        w, h = int(round(W * args.scale)), int(round(H * args.scale))
        for tt, fb in zip(times, frames):
            im = Image.frombytes('RGB', (w, h), fb)
            p = out / f'still_{tt:07.2f}.png'
            im.save(p)
            print(p)
        return

    total = timeline.total()
    end = min(args.end if args.end is not None else total, total)
    i0, i1 = int(round(args.start * FPS)), int(round(end * FPS))
    w, h = int(round(W * args.scale)), int(round(H * args.scale))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    vid_path = outp if not args.audio else outp.with_suffix('.video.mp4')
    cmd = [_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', args.preset, '-crf', str(args.crf),
           '-pix_fmt', 'yuv420p', '-tune', 'grain', '-movflags', '+faststart', str(vid_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    n = i1 - i0
    with mp.get_context('fork').Pool(args.workers, initializer=_init, initargs=(args.scale,)) as pool:
        for k, fb in enumerate(pool.imap(render_frame, range(i0, i1), chunksize=6)):
            proc.stdin.write(fb)
            if k % 48 == 0:
                el = time.time() - t0
                eta = el / max(1, k) * (n - k)
                print(f'[{k}/{n}] t={(i0 + k) / FPS:7.2f}s  elapsed {el:6.1f}s  eta {eta:6.1f}s', flush=True)
    proc.stdin.close()
    proc.wait()
    print(f'video done in {time.time() - t0:.1f}s -> {vid_path}')
    if args.audio:
        a_start = args.start
        cmd = [_ffmpeg(), '-y', '-loglevel', 'error', '-i', str(vid_path), '-ss', f'{a_start:.3f}', '-i', args.audio,
               '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-shortest',
               '-movflags', '+faststart', str(outp)]
        subprocess.check_call(cmd)
        vid_path.unlink()
        print(f'muxed -> {outp}')


if __name__ == '__main__':
    main()
