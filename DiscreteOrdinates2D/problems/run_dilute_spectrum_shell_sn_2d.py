#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Dilute Spectrum Shell — Multigroup 2-D S_N approximation.

This script builds a 2-D Cartesian analogue of the spherical benchmark:
- radial regions are defined by s = sqrt(x^2 + y^2),
- a small source core (s <= R_S) drives blackbody emission at T_S,
- a dense re-emitting shell (R_1 < s <= R_2),
- dilute cavity/exterior elsewhere.

It solves with DiscreteOrdinates2D.mg_sn_solver_2d.mg_temp_solve_2d and writes
snapshot .npz files under results/dilute_spectrum_shell_2d/<tag>/.

Notes
-----
- Geometry is 2-D Cartesian and therefore not exactly spherical. This is a
  best-effort radial mimic via concentric circular regions.
- The output includes full 2-D fields and radial-bin reductions for easier
  comparison with 1-D spherical runs.
"""

import argparse
import os
import sys
import numpy as np

# Path setup
_here = os.path.dirname(os.path.abspath(__file__))
_do2d = os.path.dirname(_here)                     # DiscreteOrdinates2D
_root = os.path.dirname(_do2d)                     # RadTranBook
for _p in (_do2d, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from DiscreteOrdinates2D.mg_sn_solver_2d import mg_temp_solve_2d
from DiscreteOrdinates2D.sn_solver_2d import c as C_LIGHT_SN, ac as AC
from DiscreteOrdinates2D.quadratures import get_2d_quadrature

from planck_integrals import Bg as _Bg_scalar, dBgdT as _dBgdT_scalar

from MG_IMC.problems.dilute_spectrum_shell import (
    A_RAD,
    R_S,
    R_1,
    R_2,
    R_OUT,
    T_S,
    T_INIT,
    T_FLOOR,
    RHO_CAVITY,
    RHO_SHELL,
    CV_SPEC,
    C_OPA,
    A_OPA,
    B_OPA,
    SIGMA_MAX,
    N_GROUPS_DEFAULT,
    T_FINAL,
    DUMP_TIMES,
    make_mesh,
    make_energy_edges,
    print_optical_depth_audit,
)


def make_cartesian_faces(mode, nxy=None):
    if nxy is not None:
        nx = int(nxy)
    elif mode == "quick":
        nx = 80
    elif mode == "standard":
        nx = 140
    else:
        nx = 200
    x_faces = np.linspace(-R_OUT, R_OUT, nx + 1)
    y_faces = np.linspace(-R_OUT, R_OUT, nx + 1)
    return x_faces, y_faces


def corner_coordinates(x_faces, y_faces):
    Ix = len(x_faces) - 1
    Iy = len(y_faces) - 1

    xL = x_faces[:-1][:, None]
    xR = x_faces[1:][:, None]
    yB = y_faces[:-1][None, :]
    yT = y_faces[1:][None, :]

    xc = np.zeros((Ix, Iy, 4))
    yc = np.zeros((Ix, Iy, 4))

    # Corner indexing matches solver: 0=NE, 1=NW, 2=SW, 3=SE
    xc[:, :, 0] = xR
    yc[:, :, 0] = yT
    xc[:, :, 1] = xL
    yc[:, :, 1] = yT
    xc[:, :, 2] = xL
    yc[:, :, 2] = yB
    xc[:, :, 3] = xR
    yc[:, :, 3] = yB
    return xc, yc


def radial_regions(r):
    """Return masks for source core, dense shell, dilute region."""
    in_source = r <= R_S
    in_shell = (r > R_1) & (r <= R_2)
    return in_source, in_shell


def make_density_map(xc, yc):
    r = np.sqrt(xc * xc + yc * yc)
    in_source, in_shell = radial_regions(r)
    rho = np.full_like(r, RHO_CAVITY)
    rho[in_shell] = RHO_SHELL
    # Keep source core dilute like cavity; heating is imposed via q_ext.
    rho[in_source] = RHO_CAVITY
    return rho


def make_sigma_a_funcs_2d(energy_edges, rho_corner):
    funcs = []
    for g in range(len(energy_edges) - 1):
        nu_bar = np.sqrt(energy_edges[g] * energy_edges[g + 1])
        nu_fac = nu_bar ** B_OPA

        def _mk(nf):
            def sigma_g(T):
                T_use = np.maximum(T, T_FLOOR)
                raw = rho_corner * C_OPA * T_use ** A_OPA * nf
                return np.minimum(raw, SIGMA_MAX)

            return sigma_g

        funcs.append(_mk(nu_fac))
    return funcs


def make_eos_funcs_2d(rho_corner):
    cv_vol = np.where(rho_corner > 1.0, rho_corner * CV_SPEC, 1.0)

    def eos(T):
        return cv_vol * T

    def inv_eos(e):
        return e / cv_vol

    def cv_func(T):
        return cv_vol * np.ones_like(T)

    return eos, inv_eos, cv_func


def make_planck_funcs(energy_edges):
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


def make_source_qext(energy_edges, xc, yc, source_sigma):
    """
    Build isotropic manufactured source in the source core (r <= R_S):
      q_g = source_sigma * 4pi B_g(T_S)
    """
    r = np.sqrt(xc * xc + yc * yc)
    in_source = r <= R_S

    q_ext = []
    for g in range(len(energy_edges) - 1):
        BgTs = 4.0 * np.pi * _Bg_scalar(energy_edges[g], energy_edges[g + 1], T_S)
        qg = np.zeros_like(r)
        qg[in_source] = source_sigma * BgTs
        q_ext.append(qg)
    return q_ext


def build_radial_profile(r_cell, val_cell, r_edges):
    prof = np.zeros(len(r_edges) - 1)
    counts = np.zeros(len(r_edges) - 1, dtype=np.int64)
    for k in range(len(r_edges) - 1):
        lo = r_edges[k]
        hi = r_edges[k + 1]
        m = (r_cell >= lo) & (r_cell < hi)
        if np.any(m):
            prof[k] = float(np.mean(val_cell[m]))
            counts[k] = int(np.count_nonzero(m))
        else:
            prof[k] = np.nan
    return prof, counts


def save_snapshot(out_dir, t, phi_g, T, x_faces, y_faces, r_edges_ref, rho_ref, energy_edges):
    Ix = len(x_faces) - 1
    Iy = len(y_faces) - 1

    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    r_cell = np.sqrt(Xc * Xc + Yc * Yc)

    G = len(phi_g)
    phi_cell_g = np.array([np.mean(phi_g[g], axis=2) for g in range(G)])
    E_rad_by_group_2d = phi_cell_g / C_LIGHT_SN
    E_rad_2d = np.sum(E_rad_by_group_2d, axis=0)

    T_rad_2d = np.where(E_rad_2d > 0.0, (np.maximum(E_rad_2d, 0.0) / A_RAD) ** 0.25, 0.0)
    T_mat_2d = np.mean(T, axis=2)

    # Radial reductions for 1-D-like comparison
    E_rad_by_group_r = []
    for g in range(G):
        prof, _ = build_radial_profile(r_cell, E_rad_by_group_2d[g], r_edges_ref)
        E_rad_by_group_r.append(prof)
    E_rad_by_group_r = np.array(E_rad_by_group_r)
    E_rad_r = np.nansum(E_rad_by_group_r, axis=0)

    T_mat_r, counts = build_radial_profile(r_cell, T_mat_2d, r_edges_ref)
    T_rad_r, _ = build_radial_profile(r_cell, T_rad_2d, r_edges_ref)

    # No direct per-group flux output from mg_temp_solve_2d.
    F_rad_by_group_r = np.zeros_like(E_rad_by_group_r)
    F_rad_r = np.zeros_like(E_rad_r)

    fname = os.path.join(out_dir, f"snapshot_t_{t:.5f}ns.npz")
    np.savez_compressed(
        fname,
        # 2-D fields
        x_centers=x_centers,
        y_centers=y_centers,
        x_faces=x_faces,
        y_faces=y_faces,
        r_cell=r_cell,
        T_mat_2d=T_mat_2d,
        T_rad_2d=T_rad_2d,
        E_rad_2d=E_rad_2d,
        E_rad_by_group_2d=E_rad_by_group_2d,
        # Radial reductions (1-D-like)
        r_centers=0.5 * (r_edges_ref[:-1] + r_edges_ref[1:]),
        r_edges=r_edges_ref,
        T_mat=T_mat_r,
        T_rad=T_rad_r,
        E_rad=E_rad_r,
        E_rad_by_group=E_rad_by_group_r,
        F_rad=F_rad_r,
        F_rad_by_group=F_rad_by_group_r,
        radial_counts=counts,
        # Shared metadata
        energy_edges=energy_edges,
        rho=rho_ref,
        time=np.float64(t),
    )
    print(f"  *** Snapshot saved -> {fname}")


def run(args):
    n_groups = args.G
    mesh_mode = "quick" if args.mode == "quick" else "standard"

    # Use the reference 1-D mesh only for radial bins and diagnostics.
    mesh_ref, r_centers_ref, rho_ref = make_mesh(mode=mesh_mode)
    r_edges_ref = np.concatenate([mesh_ref[:, 0], [mesh_ref[-1, 1]]])

    energy_edges = make_energy_edges(n_groups)

    x_faces, y_faces = make_cartesian_faces(args.mode, args.nxy)
    Ix = len(x_faces) - 1
    Iy = len(y_faces) - 1

    dx = np.diff(x_faces)
    dy = np.diff(y_faces)

    xc, yc = corner_coordinates(x_faces, y_faces)
    rho_corner = make_density_map(xc, yc)

    sigma_a_funcs = make_sigma_a_funcs_2d(energy_edges, rho_corner)
    scat_funcs = [lambda T: np.zeros_like(T) for _ in range(n_groups)]
    Bg_funcs, dBdT_funcs = make_planck_funcs(energy_edges)
    eos, inv_eos, cv_func = make_eos_funcs_2d(rho_corner)

    q_ext = make_source_qext(energy_edges, xc, yc, args.source_sigma)

    # Vacuum boundaries on all sides.
    bc_vac = {"xlo": None, "xhi": None, "ylo": None, "yhi": None}
    BCs = [lambda _t, bc=bc_vac: bc for _ in range(n_groups)]

    T0 = np.full((Ix, Iy, 4), T_INIT)
    phi_g0 = [Bg_funcs[g](T0) for g in range(n_groups)]

    # Build isotropic initial psi per group with correct M.
    quad_Ox, _, _ = get_2d_quadrature("product_square", args.N)
    M = len(quad_Ox)
    psi_g0 = [np.broadcast_to(phi_g0[g][:, :, None, :], (Ix, Iy, M, 4)).copy() for g in range(n_groups)]

    tfinal = args.tfinal if args.tfinal is not None else T_FINAL
    dump_times = np.array(sorted(t for t in DUMP_TIMES if t <= tfinal + 1e-12), dtype=float)

    tag = f"sn2d_{n_groups}g_s{args.N}_n{Ix}"
    out_dir = os.path.join(_root, "results", "dilute_spectrum_shell_2d", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("Dilute Spectrum Shell - Multigroup 2-D Cartesian S_N approximation")
    print(f"  Mode:        {args.mode}")
    print(f"  Groups:      {n_groups}   Energy: [{energy_edges[0]:.2e}, {energy_edges[-1]:.1f}] keV")
    print(f"  Ordinates:   S_{args.N} (product_square)")
    print(f"  Grid:        {Ix} x {Iy} over x,y in [-{R_OUT}, {R_OUT}] cm")
    print(f"  dt in [{args.dt_min:.1e}, {args.dt_max:.1e}] ns   t_final = {tfinal} ns")
    print(f"  Source core: s <= R_S with source_sigma = {args.source_sigma:.3e} 1/cm")
    print(f"  Output dir:  {out_dir}")
    print("=" * 72)
    print_optical_depth_audit(mesh_ref, energy_edges, rho_ref)

    T_chk = np.array([[0.1, 0.5], [1.0, T_INIT]])
    Bsum = sum(Bg_funcs[g](T_chk) for g in range(n_groups))
    print(
        "Planck normalisation: max frac err = "
        f"{np.max(np.abs(Bsum - AC * T_chk**4) / (AC * T_chk**4)):.2e}"
    )

    phi_hist, T_hist, iterations, ts, its_per_step = mg_temp_solve_2d(
        Ix,
        Iy,
        dx,
        dy,
        n_groups,
        sigma_a_funcs,
        scat_funcs,
        Bg_funcs,
        dBdT_funcs,
        q_ext,
        "product_square",
        args.N,
        BCs,
        eos,
        inv_eos,
        cv_func,
        phi_g0,
        psi_g0,
        T0,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        tfinal=tfinal,
        tolerance=args.tol,
        Linf_tol=args.tol,
        maxits=args.maxits,
        LOUD=args.loud,
        K=args.K,
        R=3,
        time_outputs=dump_times,
        use_dmd=True,
        print_stride=args.print_stride,
        store_full_history=False,
    )

    print(
        f"\nDone. Solver steps: {len(its_per_step)}   "
        f"Stored snapshots: {len(ts)}   Total transport sweeps: {iterations}"
    )

    for k, t in enumerate(ts):
        phi_g = phi_hist[k]
        T = T_hist[k]
        save_snapshot(out_dir, float(t), phi_g, T, x_faces, y_faces, r_edges_ref, rho_ref, energy_edges)

    np.savez_compressed(
        os.path.join(out_dir, "history_summary.npz"),
        times=np.asarray(ts, dtype=float),
        its_per_step=np.asarray(its_per_step, dtype=int),
        iterations_total=np.int64(iterations),
        x_faces=x_faces,
        y_faces=y_faces,
        r_edges_ref=r_edges_ref,
        r_centers_ref=r_centers_ref,
        rho_ref=rho_ref,
        energy_edges=energy_edges,
    )

    print(f"\nAll output written to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["quick", "standard", "publication"], default="quick")
    p.add_argument("--G", type=int, default=N_GROUPS_DEFAULT)
    p.add_argument("--N", type=int, default=6, help="S_N order in 2-D product square quadrature")
    p.add_argument("--nxy", type=int, default=None, help="Override square grid resolution")
    p.add_argument("--tfinal", type=float, default=None)
    p.add_argument("--dt_min", type=float, default=1e-5)
    p.add_argument("--dt_max", type=float, default=2.5e-3)
    p.add_argument("--source_sigma", type=float, default=50.0, help="Core source strength factor [1/cm]")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--maxits", type=int, default=300)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--print_stride", type=int, default=20)
    p.add_argument("--loud", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
