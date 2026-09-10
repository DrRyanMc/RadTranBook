#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Converging Marshak Wave Test Problem 3 — Spherical LD-S_N Transport

Spherically convergent radiative heat wave in a material with a power-law
decreasing density profile, solved with the LD-S_N transport scheme.  This is
the transport counterpart of the non-equilibrium diffusion script
``nonEquilibriumDiffusion/problems/converging_marshak_wave_test3_marshak.py``.

Physical setup
--------------
  Domain:        r in [0, R],  R = 0.001 cm = 10 um
  Geometry:      Spherical (full sphere; origin regularity at r = 0)
  Density:       rho(r) = (r/cm)^{-0.45}  g/cm^3      (omega = 0.45)
  Opacity:       k_t = 10^3 (T/HeV)^{-3.5} (rho/(g/cm^3))^{1.4}  cm^{-1}
  Energy:        u(T,rho) = 10^{13} (T/HeV)^2 (rho/(g/cm^3))^{0.75}  erg/cm^3
                          = 10^{-3} (T/HeV)^2 (rho/(g/cm^3))^{0.75}  GJ/cm^3

Boundary conditions
-------------------
  r = 0:  reflecting (symmetry / origin regularity, automatic for a full sphere)
  r = R:  incoming Planckian intensity from a radiation bath at T_bath(t).

  The diffusion script imposes a Robin/Marshak condition
        phi/2 + (c/3k) dphi/dr = (a c T_bath^4)/2 .
  This is the P_1 / diffusion approximation of the *exact* transport boundary
  condition, namely a prescribed incoming intensity for inward ordinates.  For
  the S_N transport problem we therefore drive the outer wall directly with the
  isotropic incident intensity
        psi_inc = a c T_bath^4 / 2          (mu < 0 ordinates),
  using the SAME corrected bath temperature T_bath(t) (eq. 8.88) as the
  diffusion script, so the two calculations are driven identically.

Time convention
---------------
  Physical time runs from t_init = -(10^{1/delta}) ~ -7.875084 ns to
  t_final = -1 ns.  The solver elapsed time tau = t_phys - t_init runs from 0
  to ~6.875084 ns.

Self-similar solution (T in HeV)
--------------------------------
  T(r,t) = 1.1982 (-t/ns)^{0.0276392} W^{1/2}(xi(r,t))
  xi(r,t) = (r / 10^{-4} cm) / (-t/ns)^{1.1157536}
  W(xi) = (xi-1)^{0.357506} (1.9792  - 0.619497 xi + 0.110644  xi^2),  1 < xi <= 2
          (xi-1)^{0.210071} (1.27048 - 0.0470724 xi + 0.00179721 xi^2),  xi > 2
          0,                                                              xi <= 1

Units note: solver temperatures are in keV, 1 HeV = 0.1 keV; material energies
are in GJ/cm^3.  Plots use HeV and 10^{13} erg/cm^3 to match the diffusion
reference figures.

Run from the DiscreteOrdinates directory:
    python problems/converging_marshak_wave_test3_sn.py
    python problems/converging_marshak_wave_test3_sn.py --zones 200 --N 8
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── solver import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates.src.sn_solver_ld_sphere import temp_solve_sph_ld
from DiscreteOrdinates.src.sn_solver import a as A_RAD, c as C_LIGHT, ac as AC

# ── project-root utilities (optional pretty PDF saver) ────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
try:
    from utils.plotfuncs import show
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False


# =============================================================================
# PROBLEM PARAMETERS  (mirrors the diffusion Test 3 script)
# =============================================================================

R = 0.001                       # outer radius (cm) = 10 um
OMEGA = 0.45                    # density power-law exponent
T_HEV_PER_KEV = 10.0           # 1 keV = 10 HeV

# Opacity:  k_t = 10^3 T_HeV^{-3.5} rho^{1.4}  ->  keV units prefactor 10^{-0.5}
_OPACITY_PREFACTOR = 1e3 * (T_HEV_PER_KEV ** (-3.5))   # = 10^{-0.5}
_OPACITY_TEXP = -3.5
_OPACITY_REXP = 1.4

# Energy density:  u = 10^{-3} T_HeV^2 rho^{0.75}  ->  keV units prefactor 0.1
_ENERGY_PREFACTOR = 1e-3 * (T_HEV_PER_KEV ** 2)        # = 0.1
_ENERGY_TEXP = 2.0
_ENERGY_REXP = 0.75

# Self-similar solution parameters
DELTA          = 1.1157535873060416
T_ANALYTIC_AMP = 1.198199926365715    # HeV
T_ANALYTIC_EXP = 0.0276392

# Physical time span
T_INIT_NS  = -(10.0 ** (1.0 / DELTA))   # ~ -7.875084 ns (front at r = R)
OUTPUT_TIMES_NS = (-6.591897629554719, -3.926450981261105, -1.0)
#T_FINAL_NS is the min of OUTPUT_TIMES_NS and -1
T_FINAL_NS = min(max(OUTPUT_TIMES_NS), -1.0)                        # ns

# Initial (cold) material temperature
T_INIT_KEV = 1.0e-4
T_OPACITY_FLOOR = 1.0e-8        # keV floor inside the opacity (matches diffusion)


# =============================================================================
# SELF-SIMILAR (analytic) SOLUTION
# =============================================================================

def _Wxsi_scalar(xi):
    """W(xi) similarity profile (scalar)."""
    if xi >= 2.0:
        return (xi - 1.0) ** 0.210071 * (1.27048 - 0.0470724 * xi + 0.00179721 * xi ** 2)
    elif xi > 1.0:
        return (xi - 1.0) ** 0.357506 * (1.9792 - 0.619497 * xi + 0.110644 * xi ** 2)
    else:
        return 0.0


Wxsi = np.vectorize(_Wxsi_scalar)


def _Vxsi_scalar(xi):
    """V(xi) similarity profile for Test 3 (scalar)."""
    return 0.8879 * xi ** (-2.233) + 0.2278 * xi ** (-1.037)


def xsi_rt(r, t_ns):
    """Similarity coordinate xi(r, t) = (r / 1e-4 cm) / (-t/ns)^delta."""
    return (r / 1e-4) / (-t_ns) ** DELTA


def T_analytic_HeV(r, t_ns):
    """Analytic temperature T(r,t) = amp (-t)^{exp} W^{1/2}(xi)  [HeV]."""
    xi = xsi_rt(np.asarray(r, dtype=float), t_ns)
    return T_ANALYTIC_AMP * (-t_ns) ** T_ANALYTIC_EXP * Wxsi(xi) ** 0.5


def T_analytic_keV(r, t_ns):
    """Analytic temperature in keV."""
    return T_analytic_HeV(r, t_ns) / T_HEV_PER_KEV


def rho_r(r):
    """Density rho(r) = (r/cm)^{-omega} g/cm^3."""
    return np.maximum(r, 1e-30) ** (-OMEGA)


def u_analytic_per_1e13(r, t_ns):
    """Analytic energy density in units of 10^{13} erg/cm^3 = T_HeV^2 rho^{0.75}."""
    T_HeV = T_analytic_HeV(r, t_ns)
    rho = rho_r(np.asarray(r, dtype=float))
    return T_HeV ** 2 * rho ** _ENERGY_REXP


# =============================================================================
# CORRECTED BATH TEMPERATURE  (eq. 8.88, identical to the diffusion script)
# =============================================================================

def _surface_T_keV(t_ns):
    """Self-similar surface temperature T(R, t) in keV."""
    return max(float(T_analytic_HeV(R, t_ns)) / T_HEV_PER_KEV, T_INIT_KEV)


def _Lambda(xi):
    """Lambda(xi) = xi^{0.6625} V(xi) W^{-1}(xi)  (eq. 8.88)."""
    W_val = _Wxsi_scalar(xi)
    if W_val < 1e-30:
        return 0.0
    return (xi ** 0.6625) * _Vxsi_scalar(xi) * (W_val ** (-1.0))


def bath_T_keV(t_ns):
    """Corrected bath (drive) temperature in keV from eq. (8.88).

    T_bath = [1 + 0.075821 Lambda(xi_R) (-t)^{-0.316092}]^{1/4} T_s(t)
    """
    t_clamped = float(t_ns)
    xi_R = (R / 1e-4) / (-t_clamped) ** DELTA
    Lambda_R = _Lambda(xi_R)
    correction = 1.0 + 0.075821 * Lambda_R * (-t_clamped) ** (-0.316092)
    correction = max(correction, 0.0) ** 0.25
    return _surface_T_keV(t_clamped) * correction


# =============================================================================
# MATERIAL-PROPERTY CLOSURES  (bake in the radial density profile)
# =============================================================================

def make_material_funcs(r_edges):
    """Build (sigma, scat, eos, invEOS) closures for a fixed (I,2) radius array.

    The transport solver calls these with an (I, 2) temperature/energy array and
    expects an (I, 2) result, so the position dependence enters purely through
    the captured per-edge density ``rho_edges``.
    """
    rho_edges = np.maximum(r_edges, 1e-30) ** (-OMEGA)          # (I, 2)
    rho_op = rho_edges ** _OPACITY_REXP                          # (I, 2)
    rho_en = rho_edges ** _ENERGY_REXP                          # (I, 2)

    def sigma_func(T):
        T_safe = np.maximum(T, T_OPACITY_FLOOR)
        return _OPACITY_PREFACTOR * T_safe ** _OPACITY_TEXP * rho_op

    def scat_func(T):
        return np.zeros_like(T)

    def eos(T):
        return _ENERGY_PREFACTOR * np.maximum(T, 0.0) ** _ENERGY_TEXP * rho_en

    def invEOS(e):
        return np.sqrt(np.maximum(e, 0.0) / (_ENERGY_PREFACTOR * rho_en))

    return sigma_func, scat_func, eos, invEOS


# =============================================================================
# SETUP AND RUN
# =============================================================================

def setup_and_run(I=200, N=8, K=50, maxits=500, LOUD=0,
                  dt_min=1.0e-4, dt_max=2.0e-2,
                  output_times=OUTPUT_TIMES_NS):
    """Run the spherical converging Marshak wave with LD-S_N transport."""
    T_FINAL_NS = min(max(output_times), -1.0)                        # ns
    print(f"Converging Marshak Wave (Test 3)  LD-S_N  (N={N}, I={I})")
    print(f"  Domain: r in [0, {R*1e4:.0f} um],  rho(r) = (r/cm)^(-{OMEGA})")
    print(f"  t_init = {T_INIT_NS:.6f} ns  ->  t_final = {T_FINAL_NS} ns")

    # ── mesh ──────────────────────────────────────────────────────────────────
    dr_val   = R / I
    r_left   = np.arange(I, dtype=np.float64) * dr_val
    dr       = np.full(I, dr_val, dtype=np.float64)
    r_right  = r_left + dr
    r_centers = r_left + 0.5 * dr_val
    r_edges  = np.column_stack([r_left, r_right])         # (I, 2)

    # ── material closures (capture the density profile) ────────────────────────
    sigma_func, scat_func, eos, invEOS = make_material_funcs(r_edges)

    # ── initial condition: cold material in equilibrium ────────────────────────
    T_ic   = np.full((I, 2), T_INIT_KEV)
    phi    = AC * T_ic ** 4
    psi    = np.broadcast_to((phi / 2.0)[:, None, :], (I, N, 2)).copy()
    g_init = phi / 2.0

    # ── no external source ──────────────────────────────────────────────────────
    q_n = np.zeros((I, N, 2))
    q_g = np.zeros((I, 2))

    # ── outer Marshak boundary: incoming Planckian at T_bath(t) ─────────────────
    # bc_outer[n, 0] = inflow for mu_n < 0;  bc_g_outer = starting-direction inflow.
    def BCs(t_elapsed):
        t_phys = T_INIT_NS + t_elapsed
        T_bath = bath_T_keV(t_phys)
        I_inc = AC * T_bath ** 4 / 2.0
        bc = np.zeros((N, 2))
        bc[:, 0] = I_inc
        return bc, I_inc

    # ── output / time controls ──────────────────────────────────────────────────
    output_times = np.array(sorted(output_times), dtype=float)
    if np.any(output_times <= T_INIT_NS) or np.any(output_times > T_FINAL_NS + 1e-9):
        raise ValueError(f"output_times must lie in ({T_INIT_NS}, {T_FINAL_NS}] ns")
    time_outputs_rel = output_times - T_INIT_NS       # solver (elapsed) times
    tfinal = float(T_FINAL_NS - T_INIT_NS)

    print(f"  Solver: 0 -> {tfinal:.4f} ns elapsed   dt in [{dt_min:.1e}, {dt_max:.1e}]")

    phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
        I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
        BCs, eos, invEOS, phi, psi, T_ic, g_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        maxits=maxits, K=K, R=3, LOUD=bool(LOUD),
        reflect_outer=False,
        time_outputs=time_outputs_rel,
    )
    print(f"  Total transport sweeps: {its}")

    # ── extract snapshots at the requested physical times ──────────────────────
    ts_arr = np.asarray(ts)
    solutions = {}
    for t_phys in output_times:
        t_rel = t_phys - T_INIT_NS
        idx = int(np.argmin(np.abs(ts_arr[1:] - t_rel)))
        t_actual = float(ts_arr[idx + 1]) + T_INIT_NS
        T_snap = Ts[idx + 1]
        solutions[float(t_phys)] = {
            'T':    T_snap.copy(),
            'phi':  phis[idx + 1].copy(),
            't_ns': t_actual,
        }
        T_c = 0.5 * (T_snap[:, 0] + T_snap[:, 1])
        print(f"  Saved t = {t_phys:.4f} ns (actual {t_actual:.4f}),"
              f"  T_max = {T_c.max()*T_HEV_PER_KEV:.4f} HeV")

    return {
        'solutions':  solutions,
        'r_centers':  r_centers,
        'r_left':     r_left,
        'dr':         dr,
        'I':          I,
        'N':          N,
        'iterations': its,
    }


# =============================================================================
# PLOTTING  (T in HeV and u in 10^{13} erg/cm^3 vs r in um)
# =============================================================================

def plot_results(results, save_prefix='converging_marshak_wave_test3_sn'):
    solutions = results['solutions']
    r_centers = results['r_centers']
    t_vals = sorted(solutions.keys())
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    r_anal = np.linspace(1e-8, R, 2000)

    # ── temperature ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.25))
    for i, t_phys in enumerate(t_vals):
        sol = solutions[t_phys]
        col = colors[i % len(colors)]
        T_c = 0.5 * (sol['T'][:, 0] + sol['T'][:, 1]) * T_HEV_PER_KEV
        ax.plot(r_centers / 1e-4, T_c, '-', color=col, lw=2.0,
                label=rf'$t = {t_phys:.2f}$ ns')
        ax.plot(r_anal / 1e-4, T_analytic_HeV(r_anal, t_phys), '--',
                color=col, lw=1.5, alpha=0.7)

    legend_elements = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=2.0,
               label=rf'$t = {t:.2f}$ ns')
        for i, t in enumerate(t_vals)
    ]
    legend_elements += [
        Line2D([0], [0], color='k', lw=2.0, ls='-', label=r'LD-S$_N$ transport'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label='Self-similar'),
    ]
    ax.set_xlabel(r'$r$ ($\mu$m)', fontsize=13)
    ax.set_ylabel(r'$T$ (HeV)', fontsize=13)
    # ax.set_title('Converging Marshak Wave (Test 3) — Spherical LD-S$_N$ vs Self-Similar',
    #              fontsize=12)
    ax.set_xlim(0, R / 1e-4)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    #ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    plt.tight_layout()
    _save(fig, f'{save_prefix}_T')

    # ── energy density ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.25))
    for i, t_phys in enumerate(t_vals):
        sol = solutions[t_phys]
        col = colors[i % len(colors)]
        T_c = 0.5 * (sol['T'][:, 0] + sol['T'][:, 1])
        # u in 10^{13} erg/cm^3 = T_HeV^2 rho^{0.75}
        u_sim = (T_c * T_HEV_PER_KEV) ** 2 * rho_r(r_centers) ** _ENERGY_REXP
        ax.plot(r_centers / 1e-4, u_sim, '-', color=col, lw=2.0,
                label=rf'$t = {t_phys:.2f}$ ns')
        ax.plot(r_anal / 1e-4, u_analytic_per_1e13(r_anal, t_phys), '--',
                color=col, lw=1.5, alpha=0.7)
    ax.set_xlabel(r'$r$ ($\mu$m)', fontsize=13)
    ax.set_ylabel(r'$u$ ($10^{13}$ erg/cm$^3$)', fontsize=13)
    # ax.set_title('Converging Marshak Wave (Test 3) — energy density',
    #              fontsize=12)
    ax.set_xlim(0, R / 1e-4)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    #ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    plt.tight_layout()
    _save(fig, f'{save_prefix}_u')


def _save(fig, stem):
    if HAS_PLOTFUNCS:
        show(stem + '.pdf', close_after=True)
        print(f"Plot saved as '{stem}.pdf'")
    else:
        fig.savefig(stem + '.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{stem}.png'")
    plt.close(fig)


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_npz(results, filename):
    data = {
        'r_centers': results['r_centers'],
        'r_left':    results['r_left'],
        'dr':        results['dr'],
        'I':         np.array(results['I']),
        'N':         np.array(results['N']),
    }
    for t_phys, sol in results['solutions'].items():
        key = f't_{t_phys:.6f}'
        data[f'{key}_T']    = sol['T']
        data[f'{key}_phi']  = sol['phi']
        data[f'{key}_t_ns'] = np.array(sol['t_ns'])
    np.savez_compressed(filename, **data)
    print(f"Results saved to {filename}")


def load_npz(filename):
    data = np.load(filename)
    solutions = {}
    for key in data.files:
        if key.startswith('t_') and key.endswith('_T'):
            t_str = key[len('t_'):-len('_T')]
            t_phys = float(t_str)
            solutions[t_phys] = {
                'T':    data[f't_{t_str}_T'],
                'phi':  data[f't_{t_str}_phi'],
                't_ns': float(data[f't_{t_str}_t_ns']),
            }
    return {
        'solutions':  solutions,
        'r_centers':  data['r_centers'],
        'r_left':     data['r_left'],
        'dr':         data['dr'],
        'I':          int(data['I']),
        'N':          int(data['N']),
        'iterations': 0,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Spherical converging Marshak wave (Test 3) — LD-S_N transport')
    parser.add_argument('--zones', type=int, default=200,
                        help='Number of radial cells (default: 200)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--K', type=int, default=50,
                        help='DMD history length (default: 50)')
    parser.add_argument('--maxits', type=int, default=500,
                        help='Max iterations per time step (default: 500)')
    parser.add_argument('--dt-min', type=float, default=1.0e-4,
                        help='Minimum time step in ns (default: 1e-4)')
    parser.add_argument('--dt-max', type=float, default=2.0e-2,
                        help='Maximum time step in ns (default: 2e-2)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=list(OUTPUT_TIMES_NS),
                        help='Physical output times in ns (negative)')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Prefix for figure output (overrides default)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip auto-caching to/from .npz')
    args = parser.parse_args()

    _problems_dir = os.path.dirname(os.path.abspath(__file__))
    _sn_dir = os.path.dirname(_problems_dir)
    _default_npz = os.path.join(
        _sn_dir, f'converging_marshak_test3_sn_N{args.N}_I{args.zones}.npz')

    npz_path = '' if args.no_cache else _default_npz

    if npz_path and os.path.exists(npz_path):
        print(f"Loading cached results from {npz_path}")
        results = load_npz(npz_path)
    else:
        results = setup_and_run(
            I=args.zones, N=args.N, K=args.K, maxits=args.maxits,
            LOUD=args.loud, dt_min=args.dt_min, dt_max=args.dt_max,
            output_times=tuple(args.output_times),
        )
        if npz_path:
            save_npz(results, npz_path)

    prefix = args.save_fig or 'converging_marshak_wave_test3_sn'
    plot_results(results, save_prefix=prefix)


if __name__ == '__main__':
    main()
