#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

r"""
2-D Cartesian MMS for the multigroup S_N solver
DiscreteOrdinates2D.mg_sn_solver_2d.mg_temp_solve_2d.

Manufactured field per group:
    psi_{g,n}(x,y) = B_g(T(x,y)) + Ox_n * Jx_g(x) + Oy_n * Jy_g(y)

with B_g(T) = p_g * (a c) * T^4 and Sum_g p_g = 1.

Using symmetric 2-D quadrature (Sum_n W_n Ox_n = Sum_n W_n Oy_n = 0),
    phi_g = Sum_n W_n psi_{g,n} = B_g(T),
so the material coupling term vanishes at the exact solution.

Steady manufactured angle-dependent source:
    Q_{g,n} = Ox*(dB/dx + sigma_t,g * Jx_g) + Oy*(dB/dy + sigma_t,g * Jy_g)
              + Ox^2 * dJx_g/dx + Oy^2 * dJy_g/dy

This script uses mg_temp_solve_2d(..., q_ext_ang=...) to inject Q_{g,n}.
"""

import argparse
import os
import sys
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_do2d_dir = os.path.dirname(_this_dir)
_repo_dir = os.path.dirname(_do2d_dir)
sys.path.insert(0, _do2d_dir)
sys.path.insert(0, _repo_dir)

from DiscreteOrdinates2D.sn_solver_2d import ac
from DiscreteOrdinates2D.quadratures import get_2d_quadrature
from DiscreteOrdinates2D.mg_sn_solver_2d import mg_temp_solve_2d

Lx = 1.0
Ly = 1.0
CV = 0.3


def group_fractions(G):
    raw = 1.0 + np.arange(G, dtype=np.float64)
    return raw / raw.sum()


def sigma_a_list(G):
    return [3.0 / (1.0 + g) for g in range(G)]


def sigma_s_list(G):
    return [0.5 * (1.0 + g) for g in range(G)]


class Manufactured2D:
    def __init__(self, G, mode):
        self.G = G
        self.mode = mode
        self.p = group_fractions(G)
        self.sa = sigma_a_list(G)
        self.ss = sigma_s_list(G)
        self.st = [self.sa[g] + self.ss[g] for g in range(G)]
        self.Jx_amp = [0.02 / (1.0 + g) for g in range(G)]
        self.Jy_amp = [0.015 / (1.0 + g) for g in range(G)]

    def Theta(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if self.mode == "linear":
            return 1.0 + 0.8 * (x / Lx) + 0.6 * (y / Ly)
        return 1.3 + 0.2 * np.sin(np.pi * x / Lx) * np.cos(np.pi * y / Ly)

    def dTheta_dx(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if self.mode == "linear":
            return np.full_like(x, 0.8 / Lx)
        return 0.2 * (np.pi / Lx) * np.cos(np.pi * x / Lx) * np.cos(np.pi * y / Ly)

    def dTheta_dy(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if self.mode == "linear":
            return np.full_like(y, 0.6 / Ly)
        return -0.2 * (np.pi / Ly) * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)

    def T(self, x, y):
        return self.Theta(x, y) ** 0.25

    def Jx(self, g, x):
        x = np.asarray(x)
        if self.mode == "linear":
            return self.Jx_amp[g] * (0.4 + 0.6 * x / Lx)
        return self.Jx_amp[g] * (1.0 + 0.25 * np.cos(np.pi * x / Lx))

    def dJx_dx(self, g, x):
        x = np.asarray(x)
        if self.mode == "linear":
            return np.full_like(x, self.Jx_amp[g] * 0.6 / Lx)
        return -self.Jx_amp[g] * 0.25 * (np.pi / Lx) * np.sin(np.pi * x / Lx)

    def Jy(self, g, y):
        y = np.asarray(y)
        if self.mode == "linear":
            return self.Jy_amp[g] * (0.3 + 0.7 * y / Ly)
        return self.Jy_amp[g] * (1.0 + 0.20 * np.sin(np.pi * y / Ly))

    def dJy_dy(self, g, y):
        y = np.asarray(y)
        if self.mode == "linear":
            return np.full_like(y, self.Jy_amp[g] * 0.7 / Ly)
        return self.Jy_amp[g] * 0.20 * (np.pi / Ly) * np.cos(np.pi * y / Ly)

    def B(self, g, x, y):
        return self.p[g] * ac * self.Theta(x, y)

    def dB_dx(self, g, x, y):
        return self.p[g] * ac * self.dTheta_dx(x, y)

    def dB_dy(self, g, x, y):
        return self.p[g] * ac * self.dTheta_dy(x, y)


def corner_coords(x_faces, y_faces):
    Ix = len(x_faces) - 1
    Iy = len(y_faces) - 1
    xL = x_faces[:-1][:, None]
    xR = x_faces[1:][:, None]
    yB = y_faces[:-1][None, :]
    yT = y_faces[1:][None, :]

    xc = np.zeros((Ix, Iy, 4))
    yc = np.zeros((Ix, Iy, 4))

    xc[:, :, 0] = xR
    yc[:, :, 0] = yT
    xc[:, :, 1] = xL
    yc[:, :, 1] = yT
    xc[:, :, 2] = xL
    yc[:, :, 2] = yB
    xc[:, :, 3] = xR
    yc[:, :, 3] = yB
    return xc, yc


def make_bc_group(mms, g, x_faces, y_faces, Ox, Oy):
    Iy = len(y_faces) - 1
    Ix = len(x_faces) - 1
    M = len(Ox)

    yB = y_faces[:-1]
    yT = y_faces[1:]
    xL = x_faces[:-1]
    xR = x_faces[1:]

    xlo = np.zeros((Iy, M, 2))
    xhi = np.zeros((Iy, M, 2))
    ylo = np.zeros((Ix, M, 2))
    yhi = np.zeros((Ix, M, 2))

    for n in range(M):
        ox = Ox[n]
        oy = Oy[n]

        B_xlo_bot = mms.B(g, 0.0, yB)
        B_xlo_top = mms.B(g, 0.0, yT)
        Jx_xlo = mms.Jx(g, 0.0)
        Jy_bot = mms.Jy(g, yB)
        Jy_top = mms.Jy(g, yT)
        xlo[:, n, 0] = B_xlo_bot + ox * Jx_xlo + oy * Jy_bot
        xlo[:, n, 1] = B_xlo_top + ox * Jx_xlo + oy * Jy_top

        B_xhi_bot = mms.B(g, Lx, yB)
        B_xhi_top = mms.B(g, Lx, yT)
        Jx_xhi = mms.Jx(g, Lx)
        xhi[:, n, 0] = B_xhi_bot + ox * Jx_xhi + oy * Jy_bot
        xhi[:, n, 1] = B_xhi_top + ox * Jx_xhi + oy * Jy_top

        B_ylo_l = mms.B(g, xL, 0.0)
        B_ylo_r = mms.B(g, xR, 0.0)
        Jy_ylo = mms.Jy(g, 0.0)
        Jx_l = mms.Jx(g, xL)
        Jx_r = mms.Jx(g, xR)
        ylo[:, n, 0] = B_ylo_l + ox * Jx_l + oy * Jy_ylo
        ylo[:, n, 1] = B_ylo_r + ox * Jx_r + oy * Jy_ylo

        B_yhi_l = mms.B(g, xL, Ly)
        B_yhi_r = mms.B(g, xR, Ly)
        Jy_yhi = mms.Jy(g, Ly)
        yhi[:, n, 0] = B_yhi_l + ox * Jx_l + oy * Jy_yhi
        yhi[:, n, 1] = B_yhi_r + ox * Jx_r + oy * Jy_yhi

    return {"xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi}


def run_mesh(mms, Ix, Iy, N, tfinal, dt_min, dt_max, K, maxits, loud=False):
    x_faces = np.linspace(0.0, Lx, Ix + 1)
    y_faces = np.linspace(0.0, Ly, Iy + 1)
    dx = np.diff(x_faces)
    dy = np.diff(y_faces)

    Ox, Oy, W = get_2d_quadrature("product_square", N)
    M = len(Ox)
    G = mms.G

    xc, yc = corner_coords(x_faces, y_faces)

    q_ext_iso = [np.zeros((Ix, Iy, 4)) for _ in range(G)]
    q_ext_ang = []

    for g in range(G):
        qg = np.zeros((Ix, Iy, M, 4))
        st = mms.st[g]

        dBdx = mms.dB_dx(g, xc, yc)
        dBdy = mms.dB_dy(g, xc, yc)
        Jx = mms.Jx(g, xc)
        Jy = mms.Jy(g, yc)
        dJxdx = mms.dJx_dx(g, xc)
        dJydy = mms.dJy_dy(g, yc)

        for n in range(M):
            ox = Ox[n]
            oy = Oy[n]
            qg[:, :, n, :] = (
                ox * (dBdx + st * Jx)
                + oy * (dBdy + st * Jy)
                + ox * ox * dJxdx
                + oy * oy * dJydy
            )
        q_ext_ang.append(qg)

    BCs = [make_bc_group(mms, g, x_faces, y_faces, Ox, Oy) for g in range(G)]

    sigma_a_funcs = [lambda T, g=g: np.full_like(T, mms.sa[g]) for g in range(G)]
    scat_funcs = [lambda T, g=g: np.full_like(T, mms.ss[g]) for g in range(G)]
    Bg_funcs = [lambda T, g=g: mms.p[g] * ac * T ** 4 for g in range(G)]
    dBdT_funcs = [lambda T, g=g: 4.0 * mms.p[g] * ac * T ** 3 for g in range(G)]

    Cv_func = lambda T: np.full_like(T, CV)
    EOS = lambda T: CV * T
    invEOS = lambda e: e / CV

    T_init = mms.T(xc, yc)

    phi_init = [mms.B(g, xc, yc) for g in range(G)]
    psi_init = []
    for g in range(G):
        psi_g = np.zeros((Ix, Iy, M, 4))
        B = mms.B(g, xc, yc)
        Jx = mms.Jx(g, xc)
        Jy = mms.Jy(g, yc)
        for n in range(M):
            psi_g[:, :, n, :] = B + Ox[n] * Jx + Oy[n] * Jy
        psi_init.append(psi_g)

    def bc_func(_t):
        return BCs[0]

    bc_funcs = [lambda _t, b=BCs[g]: b for g in range(G)]

    phi_hist, T_hist, iters, ts, its_per_step = mg_temp_solve_2d(
        Ix,
        Iy,
        dx,
        dy,
        G,
        sigma_a_funcs,
        scat_funcs,
        Bg_funcs,
        dBdT_funcs,
        q_ext_iso,
        "product_square",
        N,
        bc_funcs,
        EOS,
        invEOS,
        Cv_func,
        phi_init,
        psi_init,
        T_init,
        dt_min=dt_min,
        dt_max=dt_max,
        tfinal=tfinal,
        Linf_tol=1e-12,
        tolerance=1e-13,
        maxits=maxits,
        LOUD=loud,
        K=K,
        R=3,
        use_dmd=True,
        store_full_history=True,
        fleck_mode="legacy",
        q_ext_ang=q_ext_ang,
    )

    phi_num = phi_hist[-1]
    T_num = T_hist[-1]

    phi_man = [mms.B(g, xc, yc) for g in range(G)]
    T_man = mms.T(xc, yc)

    T_prev = T_hist[-2]
    dT_last = np.max(np.abs(T_num - T_prev)) / (np.max(np.abs(T_num)) + 1e-300)

    return phi_num, T_num, phi_man, T_man, dT_last


def error_norms(phi_num, T_num, phi_man, T_man, dx, dy):
    num = np.concatenate([p.ravel() for p in phi_num])
    man = np.concatenate([p.ravel() for p in phi_man])
    err = num - man

    w = 0.25 * np.mean(dx) * np.mean(dy)
    l2_rad = np.sqrt(np.sum(err ** 2) * w) / (np.sqrt(np.sum(man ** 2) * w) + 1e-300)
    linf_rad = np.max(np.abs(err)) / (np.max(np.abs(man)) + 1e-300)

    Terr = (T_num - T_man).ravel()
    l2_T = np.sqrt(np.sum(Terr ** 2) * w) / (np.sqrt(np.sum(T_man.ravel() ** 2) * w) + 1e-300)
    linf_T = np.max(np.abs(Terr)) / (np.max(np.abs(T_man)) + 1e-300)

    return dict(l2_rad=l2_rad, linf_rad=linf_rad, l2_T=l2_T, linf_T=linf_T)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["linear", "smooth"], default="linear")
    parser.add_argument("--G", type=int, default=3)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--I", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--tfinal", type=float, default=5.0)
    parser.add_argument("--dt-min", type=float, default=1e-3)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--maxits", type=int, default=200)
    parser.add_argument("--loud", action="store_true")
    args = parser.parse_args(argv)

    mms = Manufactured2D(args.G, args.mode)

    print(f"\\n2-D Multigroup MMS (mode={args.mode}, G={args.G}, N={args.N})")
    print(f"  p_g     = {np.array2string(mms.p, precision=4)}")
    print(f"  sigma_a = {mms.sa}")
    print(f"  sigma_s = {mms.ss}")
    print(f"{'I':>6} {'hx':>12} {'L2(rad)':>13} {'Linf(rad)':>13} {'L2(T)':>13} {'Linf(T)':>13} {'dT_last':>11}")

    h = []
    l2 = []
    linf = []

    for I in args.I:
        Ix = I
        Iy = I
        dx = np.full(Ix, Lx / Ix)
        dy = np.full(Iy, Ly / Iy)

        phi_num, T_num, phi_man, T_man, dT = run_mesh(
            mms, Ix, Iy, args.N, args.tfinal, args.dt_min, args.dt_max, args.K, args.maxits, args.loud
        )
        e = error_norms(phi_num, T_num, phi_man, T_man, dx, dy)
        hx = Lx / Ix

        h.append(hx)
        l2.append(e["l2_rad"])
        linf.append(e["linf_rad"])

        print(
            f"{I:6d} {hx:12.5e} {e['l2_rad']:13.4e} {e['linf_rad']:13.4e} "
            f"{e['l2_T']:13.4e} {e['linf_T']:13.4e} {dT:11.2e}"
        )

    h = np.array(h)
    l2 = np.array(l2)
    linf = np.array(linf)

    if args.mode == "linear":
        worst = max(l2.max(), linf.max())
        print(f"\\nLinear-exactness check: worst relative radiation error = {worst:.3e}")
    elif len(h) >= 2:
        rate_l2 = np.log(l2[:-1] / l2[1:]) / np.log(h[:-1] / h[1:])
        rate_linf = np.log(linf[:-1] / linf[1:]) / np.log(h[:-1] / h[1:])
        print("\\nObserved convergence order (radiation field):")
        for k in range(len(rate_l2)):
            print(
                f"  {args.I[k]:4d} -> {args.I[k + 1]:4d}: "
                f"p_L2 = {rate_l2[k]:.3f}, p_Linf = {rate_linf[k]:.3f}"
            )
        print(f"  mean p_L2 = {rate_l2.mean():.3f}")


if __name__ == "__main__":
    main()
