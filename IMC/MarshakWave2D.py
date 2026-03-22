#!/usr/bin/env python3
"""2D Marshak-wave validation driver for IMC2D.

This script runs three checks:
1) Cartesian directional symmetry: x-driven vs y-driven wave should match.
2) Cylindrical planar-limit check: rz with large radius should match 1D slab.
3) Snapshot/profile output and error summaries.

It uses the same material model as IMC/MarshakWave.py:
  sigma_a(T) = 300 * T^-3,   e(T) = c_v * T,  c_v = 0.3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import IMC1D as imc1d
import IMC2D as imc2d


CV_VAL = 0.3


def sigma_a_f(T):
    T_safe = np.maximum(T, 1e-4)
    return 300.0 * T_safe ** -3


def eos(T):
    return CV_VAL * T


def inv_eos(u):
    return u / CV_VAL


def cv(T):
    return 0.0 * T + CV_VAL


def _capture_snapshots_1d(
    output_times,
    dt,
    x_edges,
    Ntarget,
    Nboundary,
    Nmax,
    T_boundary,
    reflect,
):
    n = len(x_edges) - 1
    mesh = np.column_stack([x_edges[:-1], x_edges[1:]])

    Tinit = np.full(n, 1e-4)
    Trinit = np.full(n, 1e-4)
    source = np.zeros(n)

    state = imc1d.init_simulation(
        Ntarget,
        Tinit,
        Trinit,
        mesh,
        eos,
        inv_eos,
        geometry="slab",
    )

    snaps = {}
    step_count = 0
    for tout in sorted(output_times):
        while state.time < tout - 1e-12:
            step_dt = min(dt, tout - state.time)
            state, info = imc1d.step(
                state,
                Ntarget,
                Nboundary,
                0,
                Nmax,
                T_boundary,
                step_dt,
                mesh,
                sigma_a_f,
                inv_eos,
                cv,
                source,
                reflect=reflect,
                geometry="slab",
            )
            step_count += 1
            if step_count % 10 == 0:
                print("{:.6f}".format(info['time']), info['N_particles'],
                      "{:.6f}".format(info['total_energy']),
                      "{:.6f}".format(info['total_internal_energy']),
                      "{:.6f}".format(info['total_radiation_energy']),
                      "{:.6f}".format(info['boundary_emission']),
                      "{:.6e}".format(info['energy_loss']), sep='\t')
        snaps[float(tout)] = state.temperature.copy()
    return snaps


def _capture_snapshots_2d(
    geometry,
    output_times,
    dt,
    edges1,
    edges2,
    Ntarget,
    Nboundary,
    Nmax,
    T_boundary,
    reflect,
    max_events_per_particle,
):
    nx = len(edges1) - 1
    ny = len(edges2) - 1

    Tinit = np.full((nx, ny), 1e-4)
    Trinit = np.full((nx, ny), 1e-4)
    source = np.zeros((nx, ny))

    state = imc2d.init_simulation(
        Ntarget,
        Tinit,
        Trinit,
        edges1,
        edges2,
        eos,
        inv_eos,
        geometry=geometry,
    )

    snaps = {}
    step_count = 0
    for tout in sorted(output_times):
        while state.time < tout - 1e-12:
            step_dt = min(dt, tout - state.time)
            state, info = imc2d.step(
                state,
                Ntarget,
                Nboundary,
                0,
                Nmax,
                T_boundary,
                step_dt,
                edges1,
                edges2,
                sigma_a_f,
                inv_eos,
                cv,
                source,
                reflect=reflect,
                geometry=geometry,
                rz_linear_source=True,
                max_events_per_particle=max_events_per_particle,
            )
            step_count += 1
            if step_count % 10 == 0:
                print("{:.6f}".format(info["time"]), info["N_particles"],
                      "{:.6f}".format(info["total_energy"]),
                      "{:.6f}".format(info["total_internal_energy"]),
                      "{:.6f}".format(info["total_radiation_energy"]),
                      "{:.6f}".format(info["boundary_emission"]),
                      "{:.6e}".format(info["energy_loss"]), sep="\t")
        snaps[float(tout)] = state.temperature.copy()
    return snaps


def _rel_l2(a, b):
    denom = np.linalg.norm(b)
    if denom <= 1e-30:
        return np.linalg.norm(a - b)
    return np.linalg.norm(a - b) / denom


def _self_similar_T(x_arr, t, T_bc=1.0):
    """Marshak-wave self-similar solution for sigma_a = 300*T^{-3}, cv = 0.3.

    Parameters from Box 9.1 / MarshakWave.py:
      xi_max = 1.11305,  omega = 0.05989
      K = 8 a c / (7 * 3 * sigma_0 * rho * cv)
    """
    _a = imc2d.__a
    _c = imc2d.__c
    sigma_0 = 300.0 * T_bc ** -3      # sigma_a evaluated at boundary temperature
    rho = 1.0
    xi_max = 1.11305
    omega = 0.05989
    K = 8.0 * _a * _c / (7.0 * 3.0 * sigma_0 * rho * CV_VAL)
    xi = x_arr / np.sqrt(K * max(float(t), 1e-30))
    T_ss = np.where(
        xi < xi_max,
        T_bc * np.power(
            np.maximum((1.0 - xi / xi_max) * (1.0 + omega * xi / xi_max), 1e-30),
            1.0 / 6.0,
        ),
        0.0,
    )
    return T_ss


def run_xy_symmetry_check(
    output_times,
    dt,
    L=0.2,
    n=60,
    Ntarget=20000,
    Nboundary=12000,
    Nmax=80000,
    max_events_per_particle=100,
):
    x_edges = np.linspace(0.0, L, n + 1)
    y_edges = np.linspace(0.0, L, n + 1)

    # x-propagating case: source at left boundary.
    Tbc_x = (1.0, 0.0, 0.0, 0.0)
    ref_x = (False, True, True, True)
    snaps_x = _capture_snapshots_2d(
        "xy",
        output_times,
        dt,
        x_edges,
        y_edges,
        Ntarget,
        Nboundary,
        Nmax,
        Tbc_x,
        ref_x,
        max_events_per_particle,
    )

    # y-propagating case: source at bottom boundary.
    Tbc_y = (0.0, 0.0, 1.0, 0.0)
    ref_y = (True, True, False, True)
    snaps_y = _capture_snapshots_2d(
        "xy",
        output_times,
        dt,
        x_edges,
        y_edges,
        Ntarget,
        Nboundary,
        Nmax,
        Tbc_y,
        ref_y,
        max_events_per_particle,
    )

    # Compare 1D profiles: average over transverse dimension.
    errs = {}
    profiles = {}
    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    for t in output_times:
        Tx = snaps_x[float(t)].mean(axis=1)  # avg over y
        Ty = snaps_y[float(t)].mean(axis=0)  # avg over x (propagation in y)
        errs[float(t)] = _rel_l2(Tx, Ty)
        profiles[float(t)] = (Tx, Ty)

    return centers, profiles, errs


def run_rz_planar_limit_check(
    output_times,
    dt,
    Lz=0.2,
    nz=60,
    r0=10.0,
    dr=0.2,
    nr=10,
    Ntarget=24000,
    Nboundary=12000,
    Nmax=120000,
    max_events_per_particle=100,
):
    z_edges = np.linspace(0.0, Lz, nz + 1)
    r_edges = np.linspace(r0, r0 + dr, nr + 1)

    # 1D reference in z-direction (slab x mapped to z).
    Tbc_1d = (1.0, 0.0)
    ref_1d = (False, True)
    snaps_1d = _capture_snapshots_1d(
        output_times,
        dt,
        z_edges,
        Ntarget,
        Nboundary,
        Nmax,
        Tbc_1d,
        ref_1d,
    )

    # RZ case: source at zmin, reflecting r boundaries, reflecting zmax.
    Tbc_rz = (0.0, 0.0, 1.0, 0.0)
    ref_rz = (True, True, False, True)
    snaps_rz = _capture_snapshots_2d(
        "rz",
        output_times,
        dt,
        r_edges,
        z_edges,
        Ntarget,
        Nboundary,
        Nmax,
        Tbc_rz,
        ref_rz,
        max_events_per_particle,
    )

    vols = imc2d._cell_volumes_rz(r_edges, z_edges)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    errs = {}
    profiles = {}
    for t in output_times:
        T1 = snaps_1d[float(t)]
        Trz = snaps_rz[float(t)]
        # Volume-weighted radial average per z bin.
        w = vols / np.sum(vols, axis=0, keepdims=True)
        Trz_z = np.sum(w * Trz, axis=0)
        errs[float(t)] = _rel_l2(Trz_z, T1)
        profiles[float(t)] = (Trz_z, T1)

    return z_centers, profiles, errs


def save_plots_and_npz(out_prefix, output_times, xy_centers, xy_profiles, xy_errs, rz_centers, rz_profiles, rz_errs, T_bc=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ss_xy = []
    for i, t in enumerate(output_times):
        color = f"C{i}"
        Tx, Ty = xy_profiles[float(t)]
        T_ss = _self_similar_T(xy_centers, t, T_bc)
        ss_xy.append(T_ss)
        axes[0].plot(xy_centers, Tx, color=color, lw=2, label=f"x-prop t={t:g} ns")
        axes[0].plot(xy_centers, Ty, color=color, lw=1.5, ls="--", label=f"y-prop t={t:g} ns")
        axes[0].plot(xy_centers, T_ss, color=color, lw=1.5, ls=":", label=f"self-similar t={t:g} ns")
    axes[0].set_xlabel("distance (cm)")
    axes[0].set_ylabel("T (keV)")
    axes[0].set_title("Cartesian Directional Symmetry")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    ss_rz = []
    for i, t in enumerate(output_times):
        color = f"C{i}"
        Trz, T1 = rz_profiles[float(t)]
        T_ss = _self_similar_T(rz_centers, t, T_bc)
        ss_rz.append(T_ss)
        axes[1].plot(rz_centers, Trz, color=color, lw=2, label=f"rz avg t={t:g} ns")
        axes[1].plot(rz_centers, T1, color=color, lw=1.5, ls="--", label=f"1D ref t={t:g} ns")
        axes[1].plot(rz_centers, T_ss, color=color, lw=1.5, ls=":", label=f"self-similar t={t:g} ns")
    axes[1].set_xlabel("z (cm)")
    axes[1].set_ylabel("T (keV)")
    axes[1].set_title("RZ Planar-Limit vs 1D")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    fig_path = f"{out_prefix}_checks.png"
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    np.savez(
        f"{out_prefix}.npz",
        output_times=np.array(output_times),
        xy_centers=xy_centers,
        rz_centers=rz_centers,
        xy_err=np.array([xy_errs[float(t)] for t in output_times]),
        rz_err=np.array([rz_errs[float(t)] for t in output_times]),
        xy_x_profiles=np.array([xy_profiles[float(t)][0] for t in output_times]),
        xy_y_profiles=np.array([xy_profiles[float(t)][1] for t in output_times]),
        rz_profiles=np.array([rz_profiles[float(t)][0] for t in output_times]),
        ref1d_profiles=np.array([rz_profiles[float(t)][1] for t in output_times]),
        self_similar_xy=np.array(ss_xy),
        self_similar_rz=np.array(ss_rz),
    )

    return fig_path


def main():
    parser = argparse.ArgumentParser(description="2D Marshak-wave validation checks.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--times", type=float, nargs="+", default=[1.0])#, 5.0, 10.0])
    parser.add_argument("--save-prefix", type=str, default="marshak_wave_2d_validation")

    parser.add_argument("--nxy", type=int, default=60)
    parser.add_argument("--nr", type=int, default=10)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--r0", type=float, default=10.0)
    parser.add_argument("--dr", type=float, default=0.2)

    parser.add_argument("--Ntarget", type=int, default=20000)
    parser.add_argument("--Nboundary", type=int, default=12000)
    parser.add_argument("--Nmax", type=int, default=120000)
    parser.add_argument("--max-events-per-particle", type=int, default=10**6)
    args = parser.parse_args()

    output_times = sorted(args.times)

    print("=" * 72)
    print("Running RZ planar-limit check against 1D slab...")
    print("=" * 72)
    rz_centers, rz_profiles, rz_errs = run_rz_planar_limit_check(
        output_times,
        dt=args.dt,
        nz=args.nz,
        r0=args.r0,
        dr=args.dr,
        nr=args.nr,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        Nmax=args.Nmax,
        max_events_per_particle=args.max_events_per_particle,
    )

    print("=" * 72)
    print("Running Cartesian symmetry check (x-prop vs y-prop)...")
    print("=" * 72)
    xy_centers, xy_profiles, xy_errs = run_xy_symmetry_check(
        output_times,
        dt=args.dt,
        n=args.nxy,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        Nmax=args.Nmax,
        max_events_per_particle=args.max_events_per_particle,
    )

    print("\nError summary:")
    for t in output_times:
        Tx_profile = xy_profiles[float(t)][0]
        T_ss_xy = _self_similar_T(xy_centers, t)
        xy_vs_ss = _rel_l2(Tx_profile, T_ss_xy)

        Trz_profile = rz_profiles[float(t)][0]
        T_ss_rz = _self_similar_T(rz_centers, t)
        rz_vs_ss = _rel_l2(Trz_profile, T_ss_rz)

        print(
            f"  t={t:6.3f} ns  xy_sym_relL2={xy_errs[float(t)]:.4e}  "
            f"xy_vs_ss_relL2={xy_vs_ss:.4e}  "
            f"rz_vs_1d_relL2={rz_errs[float(t)]:.4e}  "
            f"rz_vs_ss_relL2={rz_vs_ss:.4e}"
        )

    fig_path = save_plots_and_npz(
        args.save_prefix,
        output_times,
        xy_centers,
        xy_profiles,
        xy_errs,
        rz_centers,
        rz_profiles,
        rz_errs,
    )
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {args.save_prefix}.npz")


if __name__ == "__main__":
    main()
