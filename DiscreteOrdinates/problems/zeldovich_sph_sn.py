#!/usr/bin/env python3
"""
Zeldovich Wave Problem - Spherical LD-S_N Transport

Radiative heat wave initialized from the Zeldovich self-similar solution
in spherical geometry (N=3), solved with the LD-S_N transport scheme.

Problem setup:
  - 1-D spherical geometry, r in [0, R_MAX] cm
  - No external source
  - sigma = 300 * T^{-3}  cm^{-1}  (temperature-dependent)
  - Material energy: e = c_v * T,  c_v = 3e-6 GJ/(cm^3 keV),  rho = 1 g/cm^3
  - Reflecting BC at r = 0  (origin regularity, built-in for full sphere)
  - Reflecting BC at r = R_MAX
  - Initial condition: Zeldovich self-similar solution at t_init = 0.01 ns (N=3)
  - Output at physical times: 0.1, 0.3, 1.0, 3.0 ns

Compares with the self-similar (equilibrium-diffusion limit) solution and
with the companion diffusion solver in nonEquilibriumDiffusion/problems/.

Run from the DiscreteOrdinates directory:
    python problems/zeldovich_sph_sn.py
    python problems/zeldovich_sph_sn.py --zones 400 --N 16 --save-fig zeldovich_sph.pdf
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import njit

# ── solver import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import a as A_RAD, c as C_LIGHT, ac as AC

# ── project-root utilities ────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
try:
    from utils.plotfuncs import show
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ── Zeldovich analytical solution ─────────────────────────────────────────────
# zeldovich.py uses `from plotfuncs import *`; add the directory that holds it.
for _zel_path in [
    os.path.join(_project_root, 'Problems'),
    os.path.join(_project_root, 'nonEquilibriumDiffusion'),
    os.path.join(_project_root, 'nonEquilibriumDiffusion', 'problems'),
]:
    if _zel_path not in sys.path:
        sys.path.insert(0, _zel_path)

try:
    from zeldovich import T_of_r_t
    HAS_ANALYTICAL = True
except ImportError:
    HAS_ANALYTICAL = False
    print("Warning: zeldovich analytical module not found — will plot S_N only.")

# ── Physical / material constants ─────────────────────────────────────────────
CV_VOL = 3.0e-6   # volumetric heat capacity  GJ / (cm^3 keV)
RHO    = 1.0      # density                   g / cm^3

# ── Problem geometry ──────────────────────────────────────────────────────────
R_MAX  = 3.0    # outer radius (cm)
T_INIT = 0.01   # initial condition time (ns) — self-similar IC
T_COLD = 0.01   # cold background temperature (keV)

# ── Material functions (njit; receive and return (I, 2) arrays) ───────────────

@njit
def sigma_func(T):
    """sigma(T) = 300 * T^{-3}  [cm^{-1}], floored at 1e-3 keV."""
    T_safe = np.maximum(T, 1.0e-3)
    return 300.0 * T_safe ** (-3)


@njit
def scat_func(T):
    """No scattering."""
    return np.zeros_like(T)


@njit
def eos(T):
    """e = c_v * T  [GJ/cm^3]"""
    return CV_VOL * np.maximum(T, 0.0)


@njit
def invEOS(e):
    """T = e / c_v  [keV]"""
    return np.maximum(e, 0.0) / CV_VOL


# ── Main solver routine ───────────────────────────────────────────────────────

def setup_and_run(I=200, N=8, K=50, maxits=500, LOUD=0,
                  dt_min=1.0e-4, dt_max=0.01,
                  output_times=(0.1, 0.3, 1.0, 3.0)):
    """Run the spherical Zeldovich wave with LD-S_N transport.

    Parameters
    ----------
    I : int
        Number of radial cells.
    N : int
        Number of discrete ordinates (must be even).
    K : int
        DMD history length.
    maxits : int
        Maximum source iterations per time step.
    LOUD : int
        Verbosity level (0 = quiet).
    dt_min, dt_max : float
        Adaptive time-step bounds (ns).
    output_times : tuple of float
        Physical output times in ns (must all be > T_INIT = 0.01 ns).

    Returns
    -------
    results : dict
        Keys: solutions, r_centers, r_left, dr, I, N, iterations.
    """
    print(f"Spherical Zeldovich Wave  LD-S_N  (N={N}, I={I})")

    # ── mesh ──────────────────────────────────────────────────────────────────
    dr_val   = R_MAX / I
    r_left   = np.arange(I, dtype=np.float64) * dr_val   # left edges  (I,)
    dr       = np.full(I, dr_val, dtype=np.float64)       # cell widths (I,)
    r_right  = r_left + dr                                 # right edges (I,)
    r_centers = r_left + 0.5 * dr_val                     # cell centres (I,)

    # ── initial condition: self-similar solution at t = T_INIT ────────────────
    # Evaluate at both LD degrees of freedom (left and right edge of each cell).
    r_edges_flat = np.concatenate([r_left, r_right])       # (2I,)
    T_ic = np.full((I, 2), T_COLD)

    if HAS_ANALYTICAL:
        try:
            T_flat, R_front = T_of_r_t(r_edges_flat, T_INIT, N=3)
            # Interleave back into (I, 2): column 0 = left edge, column 1 = right edge
            T_ic[:, 0] = np.maximum(T_flat[:I],   T_COLD)
            T_ic[:, 1] = np.maximum(T_flat[I:],   T_COLD)
            print(f"  IC: self-similar at t={T_INIT} ns,  front R = {R_front:.4f} cm")
        except Exception as exc:
            print(f"  Warning: analytical IC failed ({exc}); using T_COLD background")
    else:
        print(f"  IC: cold background T = {T_COLD} keV (analytical unavailable)")

    # Equilibrium radiation flux and angular intensity
    phi    = AC * T_ic ** 4                                             # (I, 2)
    psi    = np.broadcast_to((phi / 2)[:, None, :], (I, N, 2)).copy()  # I_n = phi/2 at equil.
    g_init = phi / 2.0                                                  # g = phi/2 at equil.

    # ── no external source ────────────────────────────────────────────────────
    q_n = np.zeros((I, N, 2))
    q_g = np.zeros((I, 2))

    # ── boundary conditions ───────────────────────────────────────────────────
    # Inner (r=0): full sphere — origin regularity applied automatically.
    # Outer (r=R_MAX): reflecting — no incoming radiation from outside.
    _bc_zero = np.zeros((N, 2))

    def BCs(t):
        return _bc_zero, 0.0

    # ── time parameters ───────────────────────────────────────────────────────
    output_times = np.array(sorted(output_times), dtype=float)
    if np.any(output_times <= T_INIT):
        bad = output_times[output_times <= T_INIT]
        raise ValueError(f"All output_times must be > T_INIT={T_INIT} ns; got {bad}")
    time_outputs_rel = output_times - T_INIT      # solver times (start at 0)
    tfinal           = float(time_outputs_rel[-1])

    print(f"  Solver: 0 → {tfinal:.3f} ns  "
          f"(physical: {T_INIT} → {T_INIT + tfinal:.3f} ns)")
    print(f"  dt_min={dt_min:.1e}  dt_max={dt_max:.1e}  maxits={maxits}  K={K}")

    phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
        I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
        BCs, eos, invEOS, phi, psi, T_ic, g_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        maxits=maxits, K=K, R=3, LOUD=bool(LOUD),
        reflect_outer=True,
        time_outputs=time_outputs_rel,
    )

    print(f"  Total transport sweeps: {its}")

    # ── extract snapshots ─────────────────────────────────────────────────────
    # phis[i] corresponds to ts[i+1] (ts[0] = 0 is the initial time).
    ts_arr = np.asarray(ts)   # shape (n_steps + 1,): ts[0]=0, ts[k]=time after step k
    solutions = {}
    for t_phys in output_times:
        t_rel = t_phys - T_INIT
        idx   = int(np.argmin(np.abs(ts_arr[1:] - t_rel)))  # index into ts_arr[1:]
        # phis/Ts have shape (n_steps + 1,): [0]=initial, [k]=after step k
        # ts_arr[1:][idx] corresponds to phis/Ts[idx + 1]
        t_actual = float(ts_arr[idx + 1]) + T_INIT
        solutions[t_phys] = {
            'T':    Ts[idx + 1].copy(),   # (I, 2)  keV
            'phi':  phis[idx + 1].copy(),  # (I, 2)  GJ/cm^3
            't_ns': t_actual,
        }
        print(f"  Saved  t = {t_phys:.3f} ns  (actual = {t_actual:.5f} ns,"
              f"  T_max = {Ts[idx+1].max():.4f} keV,"
              f"  phi_max = {phis[idx+1].max():.3e} GJ/cm^3)")

    return {
        'solutions':  solutions,
        'r_centers':  r_centers,
        'r_left':     r_left,
        'dr':         dr,
        'I':          I,
        'N':          N,
        'iterations': its,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, savefile=''):
    """Plot material temperature profiles vs. the self-similar solution.

    Solid lines: LD-S_N transport.
    Dashed lines: equilibrium-diffusion self-similar solution.
    """
    solutions = results['solutions']
    r_centers = results['r_centers']
    r_plot    = np.linspace(0.0, R_MAX, 2000)
    t_vals    = sorted(solutions.keys())
    colors    = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_vals)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.25))

    for idx, t_phys in enumerate(t_vals):
        sol   = solutions[t_phys]
        color = colors[idx]

        # Cell-centre temperature (average of LD left/right values)
        T_snap    = sol['T']                           # (I, 2)
        T_centers = 0.5 * (T_snap[:, 0] + T_snap[:, 1])

        ax.plot(r_centers, T_centers, '-', color=color, lw=2.0, alpha=0.90,
                label=rf'$t = {t_phys:.2g}$ ns (LD-S$_N$)')

        # Analytical self-similar solution (spherical, N=3 dimensions)
        if HAS_ANALYTICAL:
            try:
                T_anal, R_front = T_of_r_t(r_plot, t_phys, N=3)
                ax.plot(r_plot, np.maximum(T_anal, T_COLD), '--',
                        color=color, lw=1.5, alpha=0.65)
            except Exception:
                pass

    # Combined legend: time colours + line-style key
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2.0,
               label=rf'$t = {t:.2g}$ ns')
        for i, t in enumerate(t_vals)
    ]
    if HAS_ANALYTICAL:
        legend_elements += [
            Line2D([0], [0], color='k', lw=2.0, ls='-',
                   label=r'LD-S$_N$ transport'),
            Line2D([0], [0], color='k', lw=1.5, ls='--',
                   label='Self-similar (eq. diffusion)'),
        ]

    ax.set_xlabel('Radius $r$ (cm)', fontsize=14)
    ax.set_ylabel('Temperature $T$ (keV)', fontsize=14)
    ax.set_title('Zeldovich Wave — Spherical LD-S$_N$ vs Self-Similar',
                 fontsize=13)
    ax.set_xlim(0.0, R_MAX)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', ncol=1)

    plt.tight_layout()

    if savefile:
        if savefile.endswith('.pdf') and HAS_PLOTFUNCS:
            show(savefile, close_after=True)
        else:
            plt.savefig(savefile, dpi=150, bbox_inches='tight')
            print(f"Plot saved as '{savefile}'")
        plt.close()
    else:
        plt.savefig('zeldovich_sph_sn.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'zeldovich_sph_sn.png'")
        plt.close()


# ── Save / load ───────────────────────────────────────────────────────────────

def save_npz(results, filename):
    """Save results dict to a compressed .npz file."""
    data = {
        'r_centers': results['r_centers'],
        'r_left':    results['r_left'],
        'dr':        results['dr'],
        'I':         np.array(results['I']),
        'N':         np.array(results['N']),
    }
    for t_phys, sol in results['solutions'].items():
        key = f't_{t_phys:.4f}'
        data[f'{key}_T']     = sol['T']
        data[f'{key}_phi']   = sol['phi']
        data[f'{key}_t_ns']  = np.array(sol['t_ns'])
    np.savez_compressed(filename, **data)
    print(f"Results saved to {filename}")


def load_npz(filename):
    """Reconstruct a results dict from a previously saved .npz file."""
    data      = np.load(filename)
    solutions = {}
    for key in data.files:
        if key.endswith('_T') and key.startswith('t_'):
            t_str  = key[len('t_'):-len('_T')]
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Spherical Zeldovich wave — LD-S_N transport')
    parser.add_argument('--zones', type=int, default=50,
                        help='Number of radial cells (default: 50)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--K', type=int, default=50,
                        help='DMD history length (default: 50)')
    parser.add_argument('--maxits', type=int, default=500,
                        help='Max iterations per time step (default: 500)')
    parser.add_argument('--dt-min', type=float, default=1.0e-4,
                        help='Minimum time step in ns (default: 1e-4)')
    parser.add_argument('--dt-max', type=float, default=1.0e-2,
                        help='Maximum time step in ns (default: 1e-2)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[0.1, 0.3, 1.0, 3.0],
                        help='Physical output times in ns (default: 0.1 0.3 1.0 3.0)')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save plot to this file (e.g. zeldovich_sph.pdf)')
    parser.add_argument('--save-npz', type=str, default='',
                        help='Save results to this .npz file')
    parser.add_argument('--load-npz', type=str, default='',
                        help='Load results from a .npz file (skip computation)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip auto-caching to/from .npz')
    args = parser.parse_args()

    # Default cache path mirrors the pattern of sn_order_comparison.py
    _problems_dir = os.path.dirname(os.path.abspath(__file__))
    _sn_dir       = os.path.dirname(_problems_dir)
    _default_npz  = os.path.join(
        _sn_dir, f'zeldovich_sph_sn_N{args.N}_I{args.zones}.npz')

    npz_load = args.load_npz or ('' if args.no_cache else _default_npz)
    npz_save = args.save_npz or ('' if args.no_cache else _default_npz)

    if npz_load and os.path.exists(npz_load):
        print(f"Loading cached results from {npz_load}")
        results = load_npz(npz_load)
    else:
        results = setup_and_run(
            I=args.zones,
            N=args.N,
            K=args.K,
            maxits=args.maxits,
            LOUD=args.loud,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            output_times=tuple(args.output_times),
        )
        if npz_save:
            save_npz(results, npz_save)

    savefile = (args.save_fig
                or ('zeldovich_sph_sn.pdf' if HAS_PLOTFUNCS else ''))
    plot_results(results, savefile=savefile)


if __name__ == '__main__':
    main()
