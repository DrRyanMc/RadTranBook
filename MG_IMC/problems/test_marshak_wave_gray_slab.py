#!/usr/bin/env python3
"""
test_marshak_wave_gray_slab.py

Gray Marshak wave equivalence test: 1-D MG IMC (G=1, large-R slab limit)
vs. the self-similar equilibrium diffusion solution.

Parameters match IMC/MarshakWave.py exactly:
  L       = 0.20 cm,  N_CELLS = 50 (uniform)
  sigma_a = 300 * T^{-3} cm^{-1}  (gray, frequency-independent)
  rho     = 1.0 g/cm^3,  c_v = 0.3 GJ/(g·keV)
  T_bc    = 1.0 keV (left),  right = reflecting
  T_init  = 1e-4 keV
    Output times: configurable from CLI (defaults: 1.0, 5.0, 10.0 ns)

The large-R slab limit:  R_inner = 1e4 cm  =>  L/R = 2e-5  (geometry correction negligible)

Usage
-----
    python test_marshak_wave_gray_slab.py              # standard preset
    python test_marshak_wave_gray_slab.py --mode quick --dt 0.01
    python test_marshak_wave_gray_slab.py --Ntarget 10000 --Nmax 40000 --dt 0.01
  python test_marshak_wave_gray_slab.py --save
"""

import argparse
import os
import sys
import time as _time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from MG_IMC.MG_IMC1D import init_simulation, step

try:
    from MG_IMC import A_RAD, C_LIGHT
except ImportError:
    A_RAD   = 0.01372   # GJ / (cm³ · keV⁴)
    C_LIGHT = 29.98     # cm / ns

# ===========================================================================
# Problem constants — must match IMC/MarshakWave.py
# ===========================================================================
R_INNER  = 1.0e4   # cm  — slab limit (L/R = L/R_INNER ~ 2e-5)
L        = 0.20    # cm
N_CELLS  = 50

RHO      = 1.0     # g/cm³
CV_SPEC  = 0.3     # GJ / (g·keV)  — specific heat

SIGMA_0  = 300.0   # cm^{-1}  at T = 1 keV
N_OPA    = 3       # temperature exponent  (sigma ~ T^{-N_OPA})

T_BC     = 1.0     # keV  — left boundary temperature
T_INIT   = 1.0e-3  # keV  — initial uniform temperature
T_FLOOR  = 1.0e-2  # keV  — opacity / emission floor

# Single wide group; covers >99.999 % of the Planck spectrum at T <= 1 keV.
ENERGY_LO = 1.0e-4   # keV
ENERGY_HI = 1.0e3    # keV
ENERGY_EDGES = np.array([ENERGY_LO, ENERGY_HI])

DEFAULT_OUTPUT_TIMES = [1.0, 5.0, 10.0]   # ns
DT_DEFAULT   = 0.01               # ns

# Self-similar eigenvalues  (n=3 opacity, from Larsen / Su-Olson tabulation)
XI_MAX = 1.11305
OMEGA  = 0.05989


# ===========================================================================
# Mesh
# ===========================================================================

def make_mesh():
    """50-cell uniform slab mesh embedded in the large-R spherical domain."""
    x_edges  = np.linspace(0.0, L, N_CELLS + 1)
    r_edges  = R_INNER + x_edges
    mesh     = np.column_stack([r_edges[:-1], r_edges[1:]])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    rho_arr  = np.full(N_CELLS, RHO)
    return mesh, x_centers, rho_arr


# ===========================================================================
# Gray opacity  sigma(T) = SIGMA_0 * T^{-N_OPA}   (G = 1, no nu dependence)
# ===========================================================================

def make_sigma_funcs():
    def sigma_g0(T_arr):
        T_use = np.maximum(T_arr, T_FLOOR)
        return SIGMA_0 * T_use**(-N_OPA)
    return [sigma_g0]


# ===========================================================================
# EOS  (volumetric)
# ===========================================================================

def make_eos():
    cv_vol = RHO * CV_SPEC   # GJ / (cm³·keV)

    def eos(T):
        return cv_vol * T

    def inv_eos(u):
        return u / cv_vol

    def cv_func(T):
        return cv_vol * np.ones_like(T)

    return eos, inv_eos, cv_func


# ===========================================================================
# Self-similar solution  (equilibrium radiation diffusion, sigma ~ T^{-n})
# Matches the formula in IMC/MarshakWave.py exactly.
# ===========================================================================

def _K_const():
    """Diffusion coefficient K such that x_front = xi_max * sqrt(K * t)."""
    sigma_bc = SIGMA_0 * T_BC**(-N_OPA)
    # K = 8 a c / ( (4+n) * n * sigma_bc * rho * cv )
    #   = 8 a c / ( 7 * 3 * 300 * 1.0 * 0.3 )
    return 8.0 * A_RAD * C_LIGHT / ((4 + N_OPA) * N_OPA * sigma_bc * RHO * CV_SPEC)


def self_similar_T(x, t):
    """Gray Marshak wave self-similar temperature  T_mat(x, t)."""
    K  = _K_const()
    xi = x / np.sqrt(K * max(float(t), 1e-300))
    raw = np.where(xi < XI_MAX,
                   (1.0 - xi / XI_MAX) * (1.0 + OMEGA * xi / XI_MAX),
                   0.0)
    return T_BC * np.power(np.maximum(raw, 0.0), 1.0 / 6.0)


# ===========================================================================
# Main
# ===========================================================================

def main(args):
    if args.dt > DT_DEFAULT + 1.0e-15:
        raise ValueError(f"This test requires dt <= {DT_DEFAULT:.2f} ns; got {args.dt:.5f} ns")

    output_times = sorted(float(t) for t in args.output_times)
    if not output_times:
        raise ValueError("At least one output time is required")
    if output_times[0] <= 0.0:
        raise ValueError(f"Output times must be positive; got {output_times}")

    mode_params = {
        "quick":       dict(Ntarget=5_000,   Nmax_init=10_000,  Nmax_growth=1_000,   Nmax_final=20_000),
        "standard":    dict(Ntarget=50_000,  Nmax_init=100_000, Nmax_growth=30_000,  Nmax_final=1_300_000),
        "publication": dict(Ntarget=100_000, Nmax_init=400_000, Nmax_growth=50_000,  Nmax_final=3_000_000),
    }
    p = mode_params[args.mode]
    Ntarget = p["Ntarget"] if args.Ntarget is None else args.Ntarget
    Nboundary = Ntarget
    Nmax_init = p["Nmax_init"]
    Nmax_growth = p["Nmax_growth"]
    Nmax_final = p["Nmax_final"]

    if args.Nmax is not None:
        Nmax_init = args.Nmax
        Nmax_growth = 0 if args.Nmax < 0 else Nmax_growth
        Nmax_final = args.Nmax if args.Nmax < 0 else Nmax_final

    mesh, x_centers, rho_arr = make_mesh()
    sigma_funcs = make_sigma_funcs()
    eos, inv_eos, cv_func = make_eos()

    T_init_arr = np.full(N_CELLS, T_INIT)
    source     = np.zeros(N_CELLS)
    reflect    = (False, True)   # left open, right reflecting — matches MarshakWave.py

    state = init_simulation(
        Ntarget, T_init_arr, T_init_arr, mesh,
        ENERGY_EDGES, eos, inv_eos,
        T_emit_floor=T_FLOOR,
    )

    K = _K_const()
    print("=" * 70)
    print("Gray Marshak Wave Test — MG IMC (G=1, large-R slab limit)")
    print(f"  N_cells = {N_CELLS},  dx = {L/N_CELLS:.4f} cm,  L = {L} cm")
    print(f"  sigma_a = {SIGMA_0} * T^{{-{N_OPA}}}  (gray)")
    print(f"  rho = {RHO},  c_v = {CV_SPEC},  T_bc = {T_BC} keV")
    print(f"  T_init = {T_INIT} keV,  right BC = reflecting")
    print(f"  K = {K:.4e} cm²/ns  =>  x_front(10 ns) ~ "
          f"{XI_MAX * (K * 10.0)**0.5:.4f} cm")
    print(f"  Mode = {args.mode},  Ntarget = {Ntarget},  dt = {args.dt} ns")
    print(f"  Nmax = {Nmax_init} -> {Nmax_final} (growth {Nmax_growth}/step)")
    print(f"  R_inner = {R_INNER:.1e} cm  (L/R = {L/R_INNER:.1e})")
    print("=" * 70)

    Nmax_current = Nmax_init
    snapshots = []  # (time, T_mat, T_rad)

    for target_t in output_times:
        while state.time < target_t - 1e-12:
            step_dt = min(args.dt, target_t - state.time)
            t0 = _time.perf_counter()
            state, info = step(
                state, Ntarget, Nboundary, 0,
                Nmax_current,
                (T_BC, 0.0), step_dt, mesh, ENERGY_EDGES,
                sigma_funcs, inv_eos, cv_func, source, reflect,
                theta=1.0,
                use_scalar_intensity_Tr=False,
                conserve_comb_energy=True,
                T_emit_floor=T_FLOOR,
                particle_budget_fmin=0.0,
                Nmax_growth=Nmax_growth,
                Nmax_final=Nmax_final,
            )
            wall = _time.perf_counter() - t0
            print(f"  t={state.time:.4f}  N={info['N_particles']:6d}  "
                  f"E_tot={info['total_energy']:.4e}  "
                f"resid={info['energy_residual']:.2e}  [{wall:.1f}s]")
            if Nmax_growth > 0 and Nmax_current >= 0:
                Nmax_current = min(Nmax_current + Nmax_growth, Nmax_final)

        snapshots.append((
            state.time,
            state.temperature.copy(),
            state.radiation_temperature.copy(),
        ))
        print(f"  >>> Snapshot captured at t = {state.time:.2f} ns")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_T, ax_err = axes
    colors = [f"C{i}" for i in range(len(snapshots))]
    x_fine = np.linspace(0.0, L, 500)

    for (t_snap, T_mat, T_rad), color in zip(snapshots, colors):
        T_ss = self_similar_T(x_fine, t_snap)
        ax_T.plot(x_centers, T_mat, color=color, ls="-",
                  label=f"$T_m$  t={t_snap:.0f} ns  (MG IMC)")
        ax_T.plot(x_centers, T_rad, color=color, ls="--",
                  label=f"$T_r$  t={t_snap:.0f} ns")
        ax_T.plot(x_fine, T_ss, color=color, ls=":", lw=1.8,
                  label=f"self-similar  t={t_snap:.0f} ns")

        # Relative error T_mat vs. self-similar (only in cells above floor)
        T_ss_c = self_similar_T(x_centers, t_snap)
        rel_err = np.full_like(T_ss_c, np.nan, dtype=np.float64)
        valid = T_ss_c > T_INIT * 5
        rel_err[valid] = (T_mat[valid] - T_ss_c[valid]) / T_ss_c[valid]
        ax_err.plot(x_centers, rel_err, color=color, ls="-",
                    label=f"t={t_snap:.0f} ns")

    ax_T.set_xlabel("x  (cm)")
    ax_T.set_ylabel("T  (keV)")
    ax_T.set_title("Gray Marshak Wave: MG IMC G=1 vs self-similar")
    ax_T.set_xlim([0, L])
    ax_T.legend(fontsize=7)

    ax_err.axhline(0, color="k", lw=0.7, ls="--")
    ax_err.set_xlabel("x  (cm)")
    ax_err.set_ylabel(r"$(T_m - T_{ss}) \,/\, T_{ss}$")
    ax_err.set_title(r"Relative error: $T_m$ vs. self-similar")
    ax_err.set_xlim([0, L])
    ax_err.legend(fontsize=7)

    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    tag      = f"Nt{Ntarget}_dt{args.dt:.3f}ns"
    out_base = os.path.join(args.out_dir, f"test_marshak_gray_{tag}")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    print(f"\nFigure → {out_base}.pdf/.png")
    plt.show()

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    K = _K_const()
    print("\nSummary: T_mat vs. self-similar at mid-domain and wave-front cells")
    print(f"  {'t (ns)':>7}  {'x_front (cm)':>12}  {'T_mat_front':>12}  "
          f"{'T_ss_front':>11}  {'rel_err':>8}")
    for t_snap, T_mat, T_rad in snapshots:
        x_front = XI_MAX * (K * t_snap) ** 0.5
        T_ss_front = self_similar_T(np.array([x_front * 0.9]), t_snap)[0]
        # nearest cell to 0.9 * x_front
        idx = np.argmin(np.abs(x_centers - x_front * 0.9))
        rel = (T_mat[idx] - T_ss_front) / T_ss_front if T_ss_front > T_INIT else np.nan
        print(f"  {t_snap:>7.1f}  {x_front:>12.4f}  {T_mat[idx]:>12.5f}  "
              f"{T_ss_front:>11.5f}  {rel:>8.3f}")

    # -----------------------------------------------------------------------
    # Save results if requested
    # -----------------------------------------------------------------------
    if args.save:
        out_npz = out_base + ".npz"
        np.savez(
            out_npz,
            x_centers    = x_centers,
            snap_times   = np.array([s[0] for s in snapshots]),
            snap_T_mat   = np.array([s[1] for s in snapshots]),
            snap_T_rad   = np.array([s[2] for s in snapshots]),
            K_const      = np.float64(K),
            xi_max       = np.float64(XI_MAX),
            omega        = np.float64(OMEGA),
        )
        print(f"Data → {out_npz}")


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gray Marshak wave equivalence test for MG IMC (G=1)"
    )
    parser.add_argument("--mode", choices=["quick", "standard", "publication"],
                        default="standard",
                        help="Particle-budget preset (default standard)")
    parser.add_argument("--Ntarget",  type=int,   default=None,
                        help="Override target particle count per step")
    parser.add_argument("--Nmax",     type=int,   default=None,
                        help="Override initial comb cap. Negative = threshold mode")
    parser.add_argument("--dt",       type=float, default=DT_DEFAULT,
                        help="Time step in ns; must be <= 0.01 (default 0.01)")
    parser.add_argument("--output_times", type=float, nargs="+",
                        default=DEFAULT_OUTPUT_TIMES,
                        help="Snapshot times in ns (default: 1 5 10)")
    parser.add_argument("--save",     action="store_true",
                        help="Save temperature snapshots to .npz")
    parser.add_argument("--out_dir",  default="results/test_marshak_gray",
                        help="Output directory for figures")
    args = parser.parse_args()
    main(args)
