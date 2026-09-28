"""The protagonist: a small silhouette child (LIMBO) with a flowing scarf (GRIS).

Skeleton is angle based (FK) so poses can be blended; walk/run cycles are
generated with two-bone IK so the feet stay planted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import skia

from .core import clamp, lerp, paint, ribbon, smooth, smooth_path, c4

L_THIGH = 0.235
L_SHIN = 0.235
L_TORSO = 0.25
L_NECK = 0.035
R_HEAD = 0.125
L_UARM = 0.15
L_FARM = 0.14


@dataclass
class Pose:
    px: float = 0.0
    py: float = 0.462
    torso: float = 0.04
    head: float = 0.0
    th: tuple = (0.05, -0.05)
    kn: tuple = (0.08, 0.06)
    sh: tuple = (0.06, -0.08)
    el: tuple = (0.18, 0.2)


def _l(a, b, t):
    return a + (b - a) * t


def blend(a: Pose, b: Pose, t: float) -> Pose:
    t = clamp(t)
    return Pose(
        _l(a.px, b.px, t), _l(a.py, b.py, t), _l(a.torso, b.torso, t), _l(a.head, b.head, t),
        (_l(a.th[0], b.th[0], t), _l(a.th[1], b.th[1], t)),
        (_l(a.kn[0], b.kn[0], t), _l(a.kn[1], b.kn[1], t)),
        (_l(a.sh[0], b.sh[0], t), _l(a.sh[1], b.sh[1], t)),
        (_l(a.el[0], b.el[0], t), _l(a.el[1], b.el[1], t)),
    )


def leg_ik(hx, hy, fx, fy):
    """Return (thigh, knee) angles for a leg from hip (hx,hy) to foot (fx,fy), y up."""
    dx, dy = fx - hx, hy - fy
    d = math.hypot(dx, dy)
    d = min(max(d, 1e-4), (L_THIGH + L_SHIN) * 0.9995)
    phi = math.atan2(dx, dy)
    a = math.acos(clamp((L_THIGH ** 2 + d * d - L_SHIN ** 2) / (2 * L_THIGH * d), -1, 1))
    b = math.acos(clamp((L_THIGH ** 2 + L_SHIN ** 2 - d * d) / (2 * L_THIGH * L_SHIN), -1, 1))
    return phi + a, math.pi - b


# ---------------------------------------------------------------- pose library

def idle(t=0.0) -> Pose:
    br = math.sin(t * 2 * math.pi / 3.4)
    return Pose(py=0.462 + 0.004 * br, torso=0.035 + 0.012 * br, head=0.02 * br,
                th=(0.07, -0.07), kn=(0.1, 0.07), sh=(0.07 + 0.02 * br, -0.1), el=(0.2, 0.25))


def walk(phase: float, stride=0.34, lift=0.075, speed=1.0) -> Pose:
    p = phase % 1.0
    hip_y = 0.448 + 0.012 * math.cos(4 * math.pi * p)
    angs = []
    for i in (0, 1):
        q = (p + 0.5 * i) % 1.0
        if q < 0.56:
            u = q / 0.56
            fx = (0.5 - u) * stride
            fy = 0.0
        else:
            u = (q - 0.56) / 0.44
            fx = (-0.5 + smooth(u)) * stride
            fy = lift * math.sin(math.pi * u)
        angs.append(leg_ik(0.0, hip_y, fx + 0.02, fy))
    sw = 0.42 * math.sin(2 * math.pi * p)
    return Pose(py=hip_y, torso=0.07, head=0.03 + 0.02 * math.sin(4 * math.pi * p),
                th=(angs[0][0], angs[1][0]), kn=(angs[0][1], angs[1][1]),
                sh=(-sw + 0.05, sw + 0.05), el=(0.25 + 0.15 * max(0, sw), 0.25 + 0.15 * max(0, -sw)))


def run(phase: float, stride=0.62, lift=0.17) -> Pose:
    p = phase % 1.0
    hip_y = 0.43 + 0.03 * math.cos(4 * math.pi * p + 0.6)
    angs = []
    for i in (0, 1):
        q = (p + 0.5 * i) % 1.0
        if q < 0.38:
            u = q / 0.38
            fx = (0.45 - u) * stride
            fy = 0.0
        else:
            u = (q - 0.38) / 0.62
            fx = (-0.55 + smooth(u) * 1.0) * stride
            fy = lift * math.sin(math.pi * u) ** 0.8
        angs.append(leg_ik(0.0, hip_y, fx + 0.06, fy))
    sw = 0.95 * math.sin(2 * math.pi * p)
    return Pose(py=hip_y, torso=0.3, head=-0.12,
                th=(angs[0][0], angs[1][0]), kn=(angs[0][1], angs[1][1]),
                sh=(-sw + 0.2, sw + 0.2), el=(1.35, 1.35))


LIE = Pose(px=0.0, py=0.075, torso=-1.5, head=0.15, th=(1.52, 1.42), kn=(0.25, 0.5),
           sh=(1.35, 1.0), el=(0.4, 0.9))
CROUCH = Pose(px=0.0, py=0.25, torso=0.55, head=-0.15, th=(1.3, 0.3), kn=(2.0, 1.9),
              sh=(0.7, 0.35), el=(0.5, 0.6))
KNEEL = CROUCH
REACH = Pose(py=0.455, torso=0.12, head=0.05, th=(0.12, -0.12), kn=(0.12, 0.05),
             sh=(1.35, 0.1), el=(0.12, 0.3))
LOOK_UP = Pose(py=0.462, torso=-0.06, head=-0.5, th=(0.07, -0.07), kn=(0.1, 0.07),
               sh=(0.02, -0.12), el=(0.15, 0.2))
SIT = Pose(px=0.0, py=0.02, torso=-0.02, head=0.05, th=(1.42, 1.3), kn=(1.25, 1.5),
           sh=(0.45, 0.25), el=(0.6, 0.4))
FLOAT = Pose(py=0.46, torso=0.0, head=-0.25, th=(0.25, -0.12), kn=(0.5, 0.3),
             sh=(-0.9, 0.9), el=(-0.4, 0.4))
ARMS_UP = Pose(py=0.462, torso=-0.04, head=-0.35, th=(0.07, -0.07), kn=(0.1, 0.07),
               sh=(2.6, 2.3), el=(0.2, 0.3))


# ---------------------------------------------------------------- forward kinematics

def fk(p: Pose):
    """Joint positions in body units, y up, facing +x."""
    P = (p.px, p.py)
    S = (P[0] + L_TORSO * math.sin(p.torso), P[1] + L_TORSO * math.cos(p.torso))
    ha = p.torso + p.head
    Hc = (S[0] + (L_NECK + R_HEAD) * math.sin(ha), S[1] + (L_NECK + R_HEAD) * math.cos(ha))
    legs = []
    for i in (0, 1):
        th, kn = p.th[i], p.kn[i]
        K = (P[0] + L_THIGH * math.sin(th), P[1] - L_THIGH * math.cos(th))
        F = (K[0] + L_SHIN * math.sin(th - kn), K[1] - L_SHIN * math.cos(th - kn))
        legs.append((P, K, F, th - kn))
    arms = []
    for i in (0, 1):
        sh, el = p.sh[i], p.el[i]
        E = (S[0] + L_UARM * math.sin(sh + p.torso * 0.5), S[1] - L_UARM * math.cos(sh + p.torso * 0.5))
        Hn = (E[0] + L_FARM * math.sin(sh + el + p.torso * 0.5), E[1] - L_FARM * math.cos(sh + el + p.torso * 0.5))
        arms.append((S, E, Hn))
    return P, S, Hc, ha, legs, arms


class Hero:
    """Draws the protagonist.  (x, y) is the ground point under the pelvis."""

    def __init__(self, height=120.0, col=(0.02, 0.02, 0.025), far_col=None):
        self.h = height
        self.col = col
        self.far_col = far_col or col

    def draw(self, c: skia.Canvas, x, y, pose: Pose, t=0.0, facing=1, eyes=1.0, eye_col=(1, 1, 1),
             scarf=None, glow_col=None, glow_r=0.0, glow_a=0.0, alpha=1.0, height=None):
        """scarf: dict(length, width, cols(list), wind, speed, wave, seed)."""
        h = height or self.h
        f = facing
        P, S, Hc, ha, legs, arms = fk(pose)

        def T(pt):
            return (x + f * pt[0] * h, y - pt[1] * h)

        # --- build geometry

        def limb(pts, w0, w1):
            q = [T(pp) for pp in pts]
            path = skia.Path()
            path.moveTo(*q[0])
            for pp in q[1:]:
                path.lineTo(*pp)
            return (path, w0 * h, w1 * h)

        # legs & feet
        leg_geo = []
        for i, (Pp, K, F, sa) in enumerate(legs):
            g = [limb([Pp, K], 0.088, 0.08), limb([K, F], 0.07, 0.06)]
            # foot: short stroke forward from ankle
            fa = sa + math.pi / 2 - 0.1  # perpendicular to shin -> forward
            toe = (F[0] + 0.075 * math.sin(fa), F[1] - 0.075 * math.cos(fa))
            g.append(limb([F, toe], 0.052, 0.04))
            leg_geo.append(g)
        arm_geo = []
        for i, (Sp, E, Hn) in enumerate(arms):
            arm_geo.append([limb([Sp, E], 0.058, 0.05), limb([E, Hn], 0.05, 0.042)])

        # torso polygon (tunic)
        ta = pose.torso
        ux, uy = math.sin(ta), math.cos(ta)      # torso up vector
        nx, ny = uy, -ux                         # forward normal
        sw, hw, hem = 0.074, 0.07, 0.092
        Pp = P
        top_b = (S[0] - nx * sw, S[1] - ny * sw)
        top_f = (S[0] + nx * sw * 0.9, S[1] + ny * sw * 0.9)
        hip_f = (Pp[0] + nx * hw, Pp[1] + ny * hw)
        hip_b = (Pp[0] - nx * hw, Pp[1] - ny * hw)
        hem_f = (Pp[0] + nx * hem - ux * 0.06, Pp[1] + ny * hem - uy * 0.06)
        hem_b = (Pp[0] - nx * (hem + 0.01) - ux * 0.07, Pp[1] - ny * (hem + 0.01) - uy * 0.07)
        neck = (S[0] + ux * 0.04, S[1] + uy * 0.04)
        torso_pts = [T(neck), T(top_f), T(hip_f), T(hem_f), T(hem_b), T(hip_b), T(top_b)]
        torso_path = smooth_path(torso_pts, close=True, tension=0.55)

        # head with hair tufts
        hx, hy = T(Hc)
        r = R_HEAD * h
        head_path = skia.Path()
        head_path.addCircle(hx, hy, r)
        hair = skia.Path()
        for k, (ang, ln, wd) in enumerate([(102, 0.04, 0.05), (126, 0.048, 0.05), (150, 0.044, 0.05),
                                           (172, 0.036, 0.045), (80, 0.028, 0.04), (194, 0.026, 0.04)]):
            a = math.radians(ang) - ha * 1.0  # rotate with head (screen: y down)
            # direction in body space (x right, y up) -> apply facing
            dxu, dyu = math.cos(a), math.sin(a)
            base_c = (Hc[0] + dxu * R_HEAD * 0.92, Hc[1] + dyu * R_HEAD * 0.92)
            tip = (Hc[0] + dxu * (R_HEAD + ln) - 0.012, Hc[1] + dyu * (R_HEAD + ln))
            px_, py_ = -dyu * wd * 0.5, dxu * wd * 0.5
            b1 = (base_c[0] + px_, base_c[1] + py_)
            b2 = (base_c[0] - px_, base_c[1] - py_)
            q1, q2, q3 = T(b1), T(tip), T(b2)
            hair.moveTo(*q1)
            hair.lineTo(*q2)
            hair.lineTo(*q3)
            hair.close()

        # scarf
        scarf_path = None
        scarf_shader = None
        if scarf:
            ln = scarf.get('length', 0.9)
            n = scarf.get('n', 18)
            wind = scarf.get('wind', 0.4)
            spd = scarf.get('speed', 0.0)
            wave = scarf.get('wave', 1.0)
            ph = scarf.get('seed', 0.0)
            anchor = (S[0] - nx * 0.02 + ux * 0.035, S[1] - ny * 0.02 + uy * 0.035)
            ax, ay = T(anchor)
            flow = clamp(0.6 * spd + 0.9 * wind, 0, 1)
            lift = clamp(0.05 + 0.22 * spd + 0.3 * wind, 0, 0.9) + scarf.get('lift', 0.0)
            base = lerp(math.pi * 0.56, math.pi + lift, flow)
            droop = 0.12 + 0.55 * flow * (1 - 0.55 * flow)
            amp = wave * (0.25 + 0.75 * flow)
            pts = [(ax, ay)]
            segl = ln * h / n
            px_, py_ = ax, ay
            for i in range(1, n + 1):
                u = i / n
                wv = amp * (0.18 + 0.5 * u) * (math.sin(t * (4.0 + 4 * flow) - i * 0.55 + ph)
                                                + 0.4 * math.sin(t * 9.0 - i * 1.1 + ph * 1.7))
                ang = base - droop * u * u + wv * 0.35 + scarf.get('curl', 0.22) * math.sin(math.pi * u) * (1 - flow)
                dx, dy = math.cos(ang), math.sin(ang)
                px_ += dx * segl * f
                py_ += dy * segl
                pts.append((px_, py_))
            w0 = scarf.get('width', 0.06) * h
            widths = [w0 * (1 - 0.72 * (i / n) ** 1.3) * (1 + 0.15 * math.sin(i * 0.9 + t * 4)) for i in range(n + 1)]
            scarf_path = ribbon(pts, widths)
            self._wrap = (ax, ay, w0 * 0.95)
            cols = scarf.get('cols')
            if cols and len(cols) > 1:
                scarf_shader = skia.GradientShader.MakeLinear(
                    [skia.Point(*pts[0]), skia.Point(*pts[-1])], [c4(cc).toColor() for cc in cols])
            self._scarf_col = cols[0] if cols else self.col

        # --- draw
        def draw_all(pnt_fn, far_col, near_col, glow=False):
            # far limbs
            for g in (leg_geo[1],):
                for (pth, w0, w1) in g:
                    c.drawPath(pth, pnt_fn(far_col, (w0 + w1) / 2))
            for (pth, w0, w1) in arm_geo[1]:
                c.drawPath(pth, pnt_fn(far_col, (w0 + w1) / 2))
            if scarf_path is not None and not glow:
                sp = paint(self._scarf_col, alpha * scarf.get('alpha', 1.0))
                if scarf_shader is not None:
                    sp.setShader(scarf_shader)
                    sp.setAlphaf(alpha * scarf.get('alpha', 1.0))
                if scarf.get('glow', 0) > 0:
                    gp = paint(self._scarf_col, alpha * 0.5, blur=scarf['glow'], blend='plus')
                    if scarf_shader is not None:
                        gp.setShader(scarf_shader)
                    c.drawPath(scarf_path, gp)
                c.drawPath(scarf_path, sp)
            for (pth, w0, w1) in leg_geo[0]:
                c.drawPath(pth, pnt_fn(near_col, (w0 + w1) / 2))
            fill = pnt_fn(near_col, 0)
            c.drawPath(torso_path, fill)
            c.drawPath(head_path, fill)
            c.drawPath(hair, fill)
            if scarf_path is not None and not glow:
                wx, wy, wr = self._wrap
                c.drawOval(skia.Rect.MakeXYWH(wx - wr * 1.1, wy - wr * 0.55, wr * 2.3, wr * 1.1),
                           paint(self._scarf_col, alpha * scarf.get('alpha', 1.0)))
            for (pth, w0, w1) in arm_geo[0]:
                c.drawPath(pth, pnt_fn(near_col, (w0 + w1) / 2))

        if glow_col is not None and glow_a > 0 and glow_r > 0:
            def gp(col, w):
                return paint(glow_col, glow_a * alpha, blur=glow_r, stroke=w + glow_r * 0.6 if w else 0,
                             blend='plus') if w else paint(glow_col, glow_a * alpha, blur=glow_r, blend='plus')
            draw_all(gp, glow_col, glow_col, glow=True)

        def sp(col, w):
            return paint(col, alpha, stroke=w) if w else paint(col, alpha)
        draw_all(sp, self.far_col, self.col)

        # eyes
        if eyes > 0.01:
            ea = ha
            for (ox, oy, er) in ((0.045, 0.012, 0.017), (0.088, 0.014, 0.0135)):
                ex = Hc[0] + ox * math.cos(ea) + oy * math.sin(ea)
                ey = Hc[1] - ox * math.sin(ea) + oy * math.cos(ea)
                sx, sy = T((ex, ey))
                c.drawCircle(sx, sy, er * h * 1.9, paint(eye_col, 0.3 * eyes * alpha, blur=er * h * 1.1, blend='plus'))
                c.drawCircle(sx, sy, er * h, paint(eye_col, eyes * alpha))
        return T(Hc), T(S)

    @staticmethod
    def head_pos(x, y, pose: Pose, h, facing=1):
        P, S, Hc, ha, legs, arms = fk(pose)
        return x + facing * Hc[0] * h, y - Hc[1] * h

    @staticmethod
    def hand_pos(x, y, pose: Pose, h, facing=1, which=0):
        P, S, Hc, ha, legs, arms = fk(pose)
        Hn = arms[which][2]
        return x + facing * Hn[0] * h, y - Hn[1] * h
