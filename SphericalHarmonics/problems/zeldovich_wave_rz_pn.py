#!/usr/bin/env python3
"""
2-D cylindrical (r-z) Zel'dovich wave using the P_N solver.

This mirrors the S_N driver in DiscreteOrdinates2D/problems/zeldovich_wave_rz.py
but calls the cylindrical P_N transport solver.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_pn_dir = os.path.join(_here, "..")
if _pn_dir not in sys.path:
    sys.path.insert(0, _pn_dir)

from SphericalHarmonics.src.pn_solver_2d_rz import temp_solve_pn_rz, ac


# ===========================================================================
# Physics parameters
# ===========================================================================

_n = 3.0
_sigma0 = 3000.0
_cv_vol = 100.0
_E0 = 10.0
_K0 = 4.0 * ac / (3.0 * _sigma0)
_m_diff = _n + 3.0

JACOBIAN_DIR = os.path.join(_pn_dir, "Jacobians")


# ===========================================================================
# Self-similar solution
# ===========================================================================

def zeldovich_self_similar(r, t, N=2):
    from scipy.special import beta as Beta_func
    from math import gamma, pi

    r = np.asarray(r, dtype=float)
    m = _m_diff
    D = _K0 / _cv_vol

    beta = 1.0 / (N * m + 2.0)
    alpha = N * beta
    B = m * beta / (2.0 * D)

    beta_int = Beta_func(N / 2.0, 1.0 / m + 1.0)
    S_N = 2.0 * pi ** (N / 2.0) / gamma(N / 2.0)

    Q = _E0 / _cv_vol
    power = 1.0 / m + N / 2.0
    A = (Q * 2.0 * B ** (N / 2.0) / (S_N * beta_int)) ** (1.0 / power)

    eta_f = np.sqrt(A / B)
    R_front = eta_f * t ** beta

    T = np.zeros_like(r, dtype=float)
    mask = r < R_front
    if np.any(mask):
        xi = r[mask] / R_front
        T[mask] = t ** (-alpha) * A ** (1.0 / m) * (1.0 - xi ** 2) ** (1.0 / m)

    return T, R_front


# ===========================================================================
# Materials and grid helpers
# ===========================================================================

def _make_materials():
    def EOS(T):
        return _cv_vol * T

    def invEOS(e):
        return e / _cv_vol

    def sigma_func(T):
        return _sigma0 * np.maximum(T, 1e-6) ** (-_n)

    def scat_func(T):
        return np.zeros_like(T)

    return EOS, invEOS, sigma_func, scat_func


def _make_radial_grid(Ir, Lr, stretch=1.0):
    if abs(stretch - 1.0) < 1e-10:
        return np.full(Ir, Lr / Ir)
    ratio = stretch ** (1.0 / max(Ir - 1, 1))
    dr0 = Lr * (ratio - 1.0) / (ratio ** Ir - 1.0) if abs(ratio - 1.0) > 1e-12 else Lr / Ir
    return np.array([dr0 * ratio ** k for k in range(Ir)])


def _make_axial_grid(Iz, Lz):
    return np.full(Iz, Lz / Iz)


# ===========================================================================
# Runner
# ===========================================================================

def run_zeldovich_rz_pn(
    Ir=40,
    Iz=8,
    N_pn=3,
    Lr=1.5,
    Lz=0.2,
    t_start=0.1,
    tfinal=1.0,
    dt_start=1e-5,
    dt_max=1e-3,
    tolerance=1e-6,
    maxits=120,
    W=3,
    n_gs=1,
    stretch=1.5,
    LOUD=False,
    print_stride=50,
    output_times=None,
):
    print(f"\n{'=' * 60}")
    print(f"  2-D r-z Zel'dovich Wave - P{N_pn} version")
    print(f"  Grid: Ir={Ir}, Iz={Iz}")
    print(f"  Domain: r in [0,{Lr}] cm, z in [0,{Lz}] cm")
    print(f"  Time: {t_start:.3f} -> {tfinal:.3f} ns")
    print(f"{'=' * 60}\n")

    dr_arr = _make_radial_grid(Ir, Lr, stretch)
    dz_arr = _make_axial_grid(Iz, Lz)
    r_faces = np.concatenate([[0.0], np.cumsum(dr_arr)])
    z_faces = np.concatenate([[0.0], np.cumsum(dz_arr)])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])

    T_floor = 1e-4
    T_init = np.full((Ir, Iz, 4), T_floor)

    dr_off = np.array([+0.25, -0.25, -0.25, +0.25])
    for i in range(Ir):
        for j in range(Iz):
            for cc in range(4):
                rc = max(r_centers[i] + dr_off[cc] * dr_arr[i], 0.0)
                T_val, _ = zeldovich_self_similar(np.array([rc]), t_start, N=2)
                T_init[i, j, cc] = max(float(T_val[0]), T_floor)

    phi_init = ac * T_init ** 4

    EOS, invEOS, sigma_func, scat_func = _make_materials()
    q_ext = np.zeros((Ir, Iz, 4))

    if output_times is None:
        output_times = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
        output_times = output_times[(output_times > t_start) & (output_times <= tfinal)]
    else:
        output_times = np.asarray(output_times, dtype=float)

    phis = [phi_init.copy()]
    Ts = [T_init.copy()]
    ts = [0.0]

    phi_curr = phi_init.copy()
    T_curr = T_init.copy()
    I_curr = None
    t_done = 0.0
    total_sweeps = 0

    for t_target in output_times:
        if t_target <= t_done + 1e-14:
            continue

        seg_time = float(t_target - t_done)
        phi_curr, T_curr, I_curr, _, hist = temp_solve_pn_rz(
            Ir,
            Iz,
            dr_arr,
            dz_arr,
            r_faces,
            z_faces,
            q_ext,
            sigma_func,
            scat_func,
            N_pn,
            JACOBIAN_DIR,
            EOS,
            invEOS,
            phi_curr,
            T_curr,
            I_init=I_curr,
            dt_start=dt_start,
            t_end=seg_time,
            reflect_rlo=False,
            reflect_rhi=True,
            reflect_zlo=True,
            reflect_zhi=True,
            tolerance=tolerance,
            maxits=maxits,
            W=W,
            n_gs=n_gs,
            loud=LOUD,
            print_stride=print_stride,
            dt_max=dt_max,
            T_floor=1e-6,
        )

        t_done = t_target
        phis.append(phi_curr.copy())
        Ts.append(T_curr.copy())
        ts.append(float(t_done))
        total_sweeps += int(sum(int(h["sweeps"]) for h in hist))

    print(f"\nFinished: P{N_pn} run with {total_sweeps} total source iterations")

    return {
        "phis": phis,
        "Ts": Ts,
        "ts": np.asarray(ts, dtype=float),
        "r_centers": r_centers,
        "z_centers": z_centers,
        "r_faces": r_faces,
        "z_faces": z_faces,
    }


# ===========================================================================
# Post-processing
# ===========================================================================

def extract_radial_profile(result, t_target):
    ts = result["ts"]
    phis = result["phis"]
    r_centers = result["r_centers"]

    idx = int(np.argmin(np.abs(ts - t_target)))
    t_actual = float(ts[idx])
    phi_snap = phis[idx]
    phi_cell = np.mean(phi_snap, axis=2)
    phi_r = np.mean(phi_cell, axis=1)
    T_rad = (np.maximum(phi_r, 0.0) / ac) ** 0.25

    return r_centers, T_rad, t_actual


def plot_radial_profiles(result, output_times, savefile="zeldovich_rz_pn_profiles.png"):
    _, R_max = zeldovich_self_similar(np.array([0.0]), max(output_times), N=2)
    r_ref = np.linspace(0.0, max(result["r_centers"][-1], R_max * 1.05), 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t_tgt in output_times:
        r_num, T_rad, t_act = extract_radial_profile(result, t_tgt)
        ax.plot(r_num, T_rad, lw=2, label=f"P_N  t={t_act:.2f} ns")

        T_ref, R_front = zeldovich_self_similar(r_ref, t_tgt, N=2)
        ax.plot(r_ref, T_ref, lw=1.5, ls="--", label=f"Self-similar  t={t_tgt:.2f} ns")
        ax.axvline(R_front, color="k", ls=":", lw=0.8, alpha=0.35)

    ax.set_xlabel("r (cm)")
    ax.set_ylabel(r"Radiation temperature $T_r$ (keV)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_xlim(0.0, result["r_centers"][-1])
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savefile}")


def plot_rz_heatmap(result, t_target, savefile=None):
    r_faces = result["r_faces"]
    z_faces = result["z_faces"]
    ts = result["ts"]
    phis = result["phis"]

    idx = int(np.argmin(np.abs(ts - t_target)))
    t_actual = float(ts[idx])
    phi_snap = phis[idx]
    phi_cell = np.mean(phi_snap, axis=2)
    T_rad = (np.maximum(phi_cell, 0.0) / ac) ** 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(r_faces, z_faces, T_rad.T, shading="flat", cmap="plasma")
    ax.set_xlabel("r (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title(f"P_N radiation temperature  t={t_actual:.2f} ns")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label=r"$T_r$ (keV)")
    plt.tight_layout()

    if savefile is None:
        savefile = f"zeldovich_rz_pn_heatmap_t{t_target:.2f}ns.png"
    plt.savefig(savefile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {savefile}")


def print_z_symmetry(result, t_target):
    ts = result["ts"]
    phis = result["phis"]
    idx = int(np.argmin(np.abs(ts - t_target)))
    phi_snap = phis[idx]
    phi_cell = np.mean(phi_snap, axis=2)
    phi_z_mean = np.mean(phi_cell, axis=1, keepdims=True)
    z_var = np.max(np.abs(phi_cell - phi_z_mean)) / (phi_z_mean.max() + 1e-30)
    print(f"  t={ts[idx]:.3f} ns: max relative z-variation = {z_var:.3e}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="2-D r-z Zel'dovich wave P_N")
    parser.add_argument("--Ir", type=int, default=40)
    parser.add_argument("--Iz", type=int, default=8)
    parser.add_argument("--Npn", type=int, default=3, help="P_N order")
    parser.add_argument("--Lr", type=float, default=1.5)
    parser.add_argument("--Lz", type=float, default=0.2)
    parser.add_argument("--t-start", type=float, default=0.1)
    parser.add_argument("--tfinal", type=float, default=1.0)
    parser.add_argument("--dt-start", type=float, default=1e-5)
    parser.add_argument("--dt-max", type=float, default=1e-3)
    parser.add_argument("--stretch", type=float, default=1.5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxits", type=int, default=120)
    parser.add_argument("--W", type=int, default=3)
    parser.add_argument("--n-gs", type=int, default=1)
    parser.add_argument("--output-times", type=float, nargs="+", default=None)
    parser.add_argument("--prefix", type=str, default="zeldovich_rz_pn")
    args = parser.parse_args()

    output_times = args.output_times
    if output_times is None:
        output_times = [t for t in [0.3, 0.5, 1.0, 2.0] if t > args.t_start and t <= args.tfinal]

    result = run_zeldovich_rz_pn(
        Ir=args.Ir,
        Iz=args.Iz,
        N_pn=args.Npn,
        Lr=args.Lr,
        Lz=args.Lz,
        t_start=args.t_start,
        tfinal=args.tfinal,
        dt_start=args.dt_start,
        dt_max=args.dt_max,
        tolerance=args.tol,
        maxits=args.maxits,
        W=args.W,
        n_gs=args.n_gs,
        stretch=args.stretch,
        output_times=output_times,
    )

    for t in output_times:
        print_z_symmetry(result, t)

    plot_radial_profiles(result, output_times, savefile=f"{args.prefix}_profiles.png")
    for t in output_times:
        plot_rz_heatmap(result, t, savefile=f"{args.prefix}_heatmap_t{t:.2f}ns.png")


if __name__ == "__main__":
    main()
