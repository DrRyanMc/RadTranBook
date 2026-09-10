#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Dilute Spectrum Shell — Multigroup **spherical S_N** run script.

Deterministic discrete-ordinates counterpart of the multigroup IMC driver
``MG_IMC/problems/run_dilute_spectrum_shell.py``.  It solves the same
benchmark (blackbody source, near-vacuum cavity, dense re-emitting shell,
free-streaming exterior) with the spherical LD-S_N solver
``mg_sn_solver_ld_sphere.mg_temp_solve_sph_ld`` and writes snapshots in the
**exact** ``.npz`` field layout consumed by
``MG_IMC/visualization/plot_dilute_spectrum_shell.py``.

Usage
-----
  python run_dilute_spectrum_shell_sn.py --mode quick
  python run_dilute_spectrum_shell_sn.py --mode publication --N 8 --G 32

Output
------
  results/dilute_spectrum_shell/<tag>/snapshot_t_<time>ns.npz
  with tag = sn_<G>g_s<N>.
"""

import argparse
import os
import sys

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_dosolve = os.path.dirname(_here)                       # DiscreteOrdinates
_root = os.path.dirname(_dosolve)                       # RadTranBook
for _p in (_dosolve, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DiscreteOrdinates.src.mg_sn_solver_ld_sphere import (
    mg_temp_solve_sph_ld, radiation_flux_by_group, c as C_LIGHT_SN, ac as AC,
)
from DiscreteOrdinates.src.sn_solver_ld_sphere import _get_quadrature

# Planck group integrals (rational-approximation library)
from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar

# Shared problem definition (constants + mesh)
from MG_IMC.problems.dilute_spectrum_shell import (
    A_RAD,
    R_S, R_1, R_2, R_OUT,
    T_S, T_INIT, T_FLOOR,
    RHO_CAVITY, RHO_SHELL, CV_SPEC,
    C_OPA, A_OPA, B_OPA, SIGMA_MAX,
    N_GROUPS_DEFAULT,
    T_FINAL, DUMP_TIMES,
    make_mesh, make_energy_edges,
    print_optical_depth_audit,
)


# ===========================================================================
# (I, 2)-aware physics closures
# ===========================================================================
# NOTE: the problem module's make_sigma_a_funcs / make_eos_functions operate on
# 1-D (n_cells,) temperature arrays and multiply by rho_per_cell (n_cells,);
# that broadcast fails on the LD solver's (I, 2) edge arrays.  We therefore
# rebuild the physics here with rho broadcast over the edge axis.

def make_sigma_a_funcs_2d(energy_edges, rho_per_cell):
    """Per-group absorption opacity, accepting/returning (I, 2) arrays."""
    rho_col = np.asarray(rho_per_cell, dtype=float)[:, None]   # (I, 1)
    funcs = []
    for g in range(len(energy_edges) - 1):
        nu_bar = np.sqrt(energy_edges[g] * energy_edges[g + 1])
        nu_fac = nu_bar ** B_OPA

        def _make(nf):
            def sigma_g(T):
                T_use = np.maximum(T, T_FLOOR)
                raw = rho_col * C_OPA * T_use ** A_OPA * nf
                return np.minimum(raw, SIGMA_MAX)
            return sigma_g
        funcs.append(_make(nu_fac))
    return funcs


def make_eos_funcs_2d(rho_per_cell):
    """Linear EOS / heat-capacity closures on (I, 2) arrays.

    Matches the problem module: cv_vol = ρ·CV_SPEC in the dense shell
    (ρ > 1) and 1.0 elsewhere (a heavy cavity heat capacity so the
    near-vacuum barely heats).
    """
    cv_vol = np.where(np.asarray(rho_per_cell) > 1.0,
                      np.asarray(rho_per_cell) * CV_SPEC, 1.0)
    cv_col = cv_vol[:, None]                                    # (I, 1)

    def eos(T):
        return cv_col * T

    def inv_eos(e):
        return e / cv_col

    def cv_func(T):
        return cv_col * np.ones_like(T)

    return eos, inv_eos, cv_func


def make_planck_funcs(energy_edges):
    """Return (Bg_funcs, dBdT_funcs) giving 4π B_g and 4π dB_g/dT on (I, 2)."""
    Bg_funcs, dBdT_funcs = [], []
    for g in range(len(energy_edges) - 1):
        El, Eh = energy_edges[g], energy_edges[g + 1]

        def _mkB(El_, Eh_):
            def Bg(T):
                Tf = np.maximum(np.asarray(T, float), 1e-6).ravel()
                out = np.array([4.0 * np.pi * _Bg_scalar(El_, Eh_, t) for t in Tf])
                return out.reshape(np.shape(T))
            return Bg

        def _mkdB(El_, Eh_):
            def dBdT(T):
                Tf = np.maximum(np.asarray(T, float), 1e-6).ravel()
                out = np.array([4.0 * np.pi * _dBgdT_scalar(El_, Eh_, t) for t in Tf])
                return out.reshape(np.shape(T))
            return dBdT

        Bg_funcs.append(_mkB(El, Eh))
        dBdT_funcs.append(_mkdB(El, Eh))
    return Bg_funcs, dBdT_funcs


# ===========================================================================
# Snapshot writing (matches the IMC .npz layout exactly)
# ===========================================================================

def save_snapshot(out_dir, snap, mesh, r_centers, rho_per_cell, energy_edges, N):
    """Write one snapshot dict to ``snapshot_t_<time>ns.npz``."""
    t = snap['time']
    fname = os.path.join(out_dir, f"snapshot_t_{t:.5f}ns.npz")

    G = len(snap['phi_g'])
    n_cells = mesh.shape[0]

    # Cell-centred scalar flux per group  (φ = c · E_rad)
    phi_c = np.array([0.5 * (snap['phi_g'][g][:, 0] + snap['phi_g'][g][:, 1])
                      for g in range(G)])                       # (G, n_cells)
    E_rad_by_group = phi_c / C_LIGHT_SN                          # energy density
    E_rad_total = np.sum(E_rad_by_group, axis=0)                 # (n_cells,)

    # Radiation temperature from total energy density: E = a T_r^4
    T_rad = np.where(E_rad_total > 0.0,
                     (np.maximum(E_rad_total, 0.0) / A_RAD) ** 0.25, 0.0)

    # Material temperature (cell-centred)
    T_mat = 0.5 * (snap['T'][:, 0] + snap['T'][:, 1])

    # Net radial flux per group  (same units as φ)
    F_list = radiation_flux_by_group(snap['psi_g'], N)
    F_rad_by_group = np.array(F_list)                           # (G, n_cells)
    F_rad_total = np.sum(F_rad_by_group, axis=0)

    np.savez_compressed(
        fname,
        r_centers=r_centers,
        r_edges=np.concatenate([mesh[:, 0], [mesh[-1, 1]]]),
        T_mat=T_mat,
        T_rad=T_rad,
        E_rad=E_rad_total,
        E_rad_by_group=E_rad_by_group,
        F_rad=F_rad_total,
        F_rad_by_group=F_rad_by_group,
        energy_edges=energy_edges,
        rho=rho_per_cell,
        time=np.float64(t),
    )
    print(f"  *** Snapshot saved → {fname}")


# ===========================================================================
# Main
# ===========================================================================

def run(args):
    n_groups = args.G
    N = args.N
    mesh_mode = "quick" if args.mode == "quick" else "standard"

    # Geometry / physics
    mesh, r_centers, rho_per_cell = make_mesh(mode=mesh_mode)
    energy_edges = make_energy_edges(n_groups)

    sigma_a_funcs = make_sigma_a_funcs_2d(energy_edges, rho_per_cell)
    scat_funcs = [lambda T: np.zeros_like(T) for _ in range(n_groups)]
    Bg_funcs, dBdT_funcs = make_planck_funcs(energy_edges)
    eos, inv_eos, cv_func = make_eos_funcs_2d(rho_per_cell)

    I = mesh.shape[0]
    r_left = np.ascontiguousarray(mesh[:, 0])
    dr = np.ascontiguousarray(mesh[:, 1] - mesh[:, 0])

    tfinal = args.tfinal if args.tfinal is not None else T_FINAL
    dump_times = np.array(sorted(t for t in DUMP_TIMES if t <= tfinal + 1e-12))

    # ── Boundary conditions ────────────────────────────────────────────
    MU, _W = _get_quadrature(N)
    pos = MU > 0.0

    # Inner wall: isotropic blackbody at T_S.  Incoming (μ > 0) intensity in
    # the φ-convention equilibrium is ½·(4π B_g) = ½·ac T_S⁴ χ_g.
    inner_vals = np.array([0.5 * 4.0 * np.pi * _Bg_scalar(energy_edges[g],
                                                          energy_edges[g + 1], T_S)
                           for g in range(n_groups)])

    def _make_inner(g):
        val = inner_vals[g]
        bc = np.zeros((N, 2))
        bc[pos, 1] = val                       # μ > 0 inflow at inner wall
        def bc_inner(t):
            return bc
        return bc_inner

    BCs_inner = [_make_inner(g) for g in range(n_groups)]

    # Outer wall: vacuum (no incoming, no starting-direction inflow).
    _zero_outer = np.zeros((N, 2))
    def _bc_outer_vac(t):
        return _zero_outer, 0.0
    BCs_outer = [_bc_outer_vac for _ in range(n_groups)]

    # ── Initial conditions (cold, near-Planck at T_INIT) ───────────────
    T0 = np.full((I, 2), T_INIT)
    phi_g0 = [Bg_funcs[g](T0) for g in range(n_groups)]            # ac T⁴ χ_g
    psi_g0 = [0.5 * phi_g0[g][:, None, :] * np.ones((1, N, 1))
              for g in range(n_groups)]
    g_g0 = [0.5 * phi_g0[g].copy() for g in range(n_groups)]

    # ── Output dir ─────────────────────────────────────────────────────
    tag = f"sn_{n_groups}g_s{N}"
    out_dir = os.path.join(_root, "results", "dilute_spectrum_shell", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Dilute Spectrum Shell — Multigroup spherical S_N")
    print(f"  Mode:        {args.mode}")
    print(f"  Groups:      {n_groups}   Energy: [{energy_edges[0]:.2e}, "
          f"{energy_edges[-1]:.1f}] keV")
    print(f"  Ordinates:   S_{N}")
    print(f"  Cells:       {I}   r ∈ [{R_S}, {R_OUT}] cm")
    print(f"  dt ∈ [{args.dt_min:.1e}, {args.dt_max:.1e}] ns   t_final = {tfinal} ns")
    print(f"  Inner BC:    blackbody T = {T_S} keV at r = {R_S} cm")
    print(f"  Output dir:  {out_dir}")
    print("=" * 72)
    print_optical_depth_audit(mesh, energy_edges, rho_per_cell)

    # Planck normalisation sanity check
    T_chk = np.array([[0.1, 0.5], [1.0, T_INIT]])
    Bsum = sum(Bg_funcs[g](T_chk) for g in range(n_groups))
    print(f"Planck normalisation: max frac err = "
          f"{np.max(np.abs(Bsum - AC * T_chk**4) / (AC * T_chk**4)):.2e}")

    # ── Solve ──────────────────────────────────────────────────────────
    snapshots, iterations, ts = mg_temp_solve_sph_ld(
        I, r_left, dr,
        n_groups, sigma_a_funcs, scat_funcs, Bg_funcs, dBdT_funcs,
        None, None, N,
        BCs_outer, BCs_inner,
        eos, inv_eos, cv_func,
        phi_g0, psi_g0, g_g0, T0,
        dt_min=args.dt_min, dt_max=args.dt_max, tfinal=tfinal,
        tolerance=args.tol, Linf_tol=args.tol, maxits=args.maxits,
        fix=args.fix, K=args.K, R=3,
        time_outputs=dump_times,
        print_stride=args.print_stride,
        dt_T_frac=args.dt_T_frac,
    )

    print(f"\nDone.  Steps: {len(ts) - 1}   Total transport sweeps: {iterations}")
    print(f"Snapshots captured: {len(snapshots)} / {len(dump_times)}")

    for snap in snapshots:
        save_snapshot(out_dir, snap, mesh, r_centers, rho_per_cell,
                      energy_edges, N)

    print(f"\nAll output written to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["quick", "standard", "publication"],
                   default="publication")
    p.add_argument("--G", type=int, default=N_GROUPS_DEFAULT)
    p.add_argument("--N", type=int, default=8, help="S_N order (ordinates)")
    p.add_argument("--tfinal", type=float, default=None)
    p.add_argument("--dt_min", type=float, default=1e-5)
    p.add_argument("--dt_max", type=float, default=2.5e-3)
    p.add_argument("--dt_T_frac", type=float, default=0.02,
                   help="Throttle dt to ~this fraction of peak-T change/step "
                        "(0 disables).")
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--maxits", type=int, default=400)
    p.add_argument("--K", type=int, default=60)
    p.add_argument("--fix", type=int, default=1)
    p.add_argument("--print_stride", type=int, default=25)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
