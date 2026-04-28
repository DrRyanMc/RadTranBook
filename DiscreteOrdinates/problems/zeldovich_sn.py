"""
Zeldovich Wave Problem — S_N Transport

Radiative heat wave initialized from the Zeldovich self-similar solution.
No external source; energy propagates from the initial hot pulse.

Problem setup:
  - 1-D planar slab geometry
  - No external source
  - σ_R = σ_P = 300 · T^{-3}  cm^{-1}  (temperature-dependent)
  - Material energy: e = ρ c_v T,  c_v = 3×10^{-6} GJ/(cm³·keV),  ρ = 1 g/cm³
  - Reflecting BC at x = 0  (emulated via symmetric full domain)
  - Vacuum BC at x = R_MAX
  - Initial condition: Zeldovich self-similar solution at t_init = 0.01 ns
  - Output at physical times: 0.1, 0.3, 1.0, 3.0 ns

The reflecting BC at x=0 is enforced by solving on the symmetric
full domain [0, 2*R_MAX] with the initial condition mirrored about
the centre and vacuum BCs on both sides.  Only the right half is reported.

Run from the DiscreteOrdinates directory:
    python problems/zeldovich_sn.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from matplotlib.lines import Line2D
from numba import jit, njit, float64

# ---- solver import ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver

# ---- project-root utilities ----
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ---- Zeldovich analytical solution ----
# zeldovich.py depends on 'plotfuncs'; add the Problems/ directory which
# contains that module, then the nonEquilibriumDiffusion directory.
for _zel_path in [
    os.path.join(project_root, 'Problems'),
    os.path.join(project_root, 'nonEquilibriumDiffusion'),
    os.path.join(project_root, 'nonEquilibriumDiffusion', 'problems'),
]:
    if _zel_path not in sys.path:
        sys.path.insert(0, _zel_path)

try:
    from zeldovich import T_of_r_t
    HAS_ANALYTICAL = True
except ImportError:
    HAS_ANALYTICAL = False
    print("Warning: zeldovich analytical module not found — will plot S_N only.")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
A_RAD   = sn_solver.a      # radiation constant      (GJ / cm³ keV⁴)
C_LIGHT = sn_solver.c      # speed of light          (cm / ns)
AC      = sn_solver.ac     # a · c
RHO     = 1.0              # density                 (g / cm³)
CV_VOL  = 3.0e-6           # volumetric heat capacity (GJ / (cm³ · keV))

# Problem geometry / time parameters
R_MAX   = 3.0              # half-domain length       (cm)
T_INIT  = 0.01             # IC start time            (ns)
T_COLD  = 0.01             # background temperature   (keV)
T_MIN   = 0.001            # opacity floor temperature (keV)

# ---------------------------------------------------------------------------
# Numba-compiled material functions
# ---------------------------------------------------------------------------

@njit
def sigma_func(T):
    """σ(T) = 300 · T^{-3}  [cm^{-1}], floored at T_MIN to prevent overflow."""
    T_safe = np.maximum(T, 0.001)
    return 300.0 * T_safe ** (-3)


@njit
def scat_func(T):
    return np.zeros_like(T)


@njit
def eos(T):
    """e = ρ c_v T  [GJ / cm³]"""
    return CV_VOL * np.maximum(T, 0.0)


@njit
def invEOS(e):
    """T = e / (ρ c_v)  [keV]"""
    return np.maximum(e, 0.0) / CV_VOL


# ---------------------------------------------------------------------------
# Setup and run
# ---------------------------------------------------------------------------

def setup_and_run(I=400, order=2, N=8, K=800, maxits=2000, LOUD=0,
                  output_times=(0.1, 0.3, 1.0, 3.0)):
    """Run the Zeldovich wave with S_N transport.

    The reflecting BC at x = 0 is emulated by solving on the symmetric
    full domain [0, 2·R_MAX] with the initial condition mirrored about the
    centre and vacuum BCs on both sides.  Only the right half is returned.

    Parameters
    ----------
    I : int
        Number of spatial zones (half-domain; full domain uses 2*I).
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates.
    K : int
        DMD inner iterations.
    maxits : int
        Max iterations per time step.
    LOUD : int
        Verbosity level.
    output_times : tuple of float
        Physical output times in ns (must all be > T_INIT = 0.01 ns).

    Returns
    -------
    results : dict
        Keys: solutions, x, hx, Lx, order, iterations.
    """
    # ---- full symmetric domain ----
    I_full  = 2 * I
    Lx_full = 2.0 * R_MAX
    hx      = Lx_full / I_full
    center  = R_MAX
    x_comp  = np.linspace(hx / 2.0, Lx_full - hx / 2.0, I_full)
    x_phys  = x_comp[I:] - center     # right half, x_phys[0] ≈ hx/2

    print(f"Zeldovich Wave S_N  (N={N}, I={I}, order={order})")
    print(f"  Full domain: [0, {Lx_full:.1f}] cm  ({I_full} cells, hx={hx:.4f} cm)")

    # ---- initial condition from self-similar solution ----
    x_abs = np.abs(x_comp - center)          # distance from symmetry plane
    T_init_1d = np.full(I_full, T_COLD)

    if HAS_ANALYTICAL:
        try:
            T_anal, R_front = T_of_r_t(x_abs, T_INIT, N=1)
            T_init_1d = np.maximum(T_anal, T_COLD)
            print(f"  IC: self-similar solution at t={T_INIT} ns, "
                  f"wave front R = {R_front:.4f} cm")
        except Exception as exc:
            print(f"  Warning: analytical IC failed ({exc}); using cold background")
    else:
        print(f"  IC: cold background T = {T_COLD} keV (analytical unavailable)")

    # Expand to Bernstein coefficient arrays: constant within each cell
    T   = np.ones((I_full, order + 1)) * T_init_1d[:, None]
    phi = AC * T ** 4                  # equilibrium: φ = a c T⁴
    psi = np.zeros((I_full, N, order + 1)) + phi[:, None, :]

    # ---- no external source ----
    q_zero = np.zeros((I_full, N, order + 1))

    # ---- vacuum BCs on both sides ----
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        return np.zeros((N, order + 1))

    # ---- time parameters ----
    # At T ~ 1 keV: σ ~ 300 cm^{-1} → mfp ~ 0.003 cm → mft ~ 1e-4 ns
    dt_min = 1.0e-5   # ns
    dt_max = 0.005     # ns

    output_times = np.array(sorted(output_times), dtype=float)
    if np.any(output_times <= T_INIT):
        raise ValueError(f"All output_times must be > T_INIT={T_INIT} ns; "
                         f"got {output_times[output_times <= T_INIT]}")
    time_outputs_rel = output_times - T_INIT   # solver time (starts at 0)
    tfinal           = float(time_outputs_rel[-1])

    print(f"  Running solver: 0 → {tfinal:.3f} ns  "
          f"(physical: {T_INIT} → {T_INIT + tfinal:.3f} ns)")

    phis, Ts, its, ts = sn_solver.temp_solve_dmd_inc(
        I_full, hx, q_zero, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3,
        time_outputs=time_outputs_rel)

    print(f"  Total transport sweeps: {its}")

    # ---- extract right-half snapshots ----
    solutions = {}
    for t_phys in output_times:
        t_rel  = t_phys - T_INIT
        tstep  = np.argmin(np.abs(t_rel - ts))
        T_snap   = Ts[tstep][I:, :]
        phi_snap = phis[tstep][I:, :]
        t_actual = float(ts[tstep]) + T_INIT
        solutions[t_phys] = {
            'T':     T_snap,     # Bernstein coeffs, shape (I, order+1), keV
            'phi':   phi_snap,   # Bernstein coeffs, shape (I, order+1), GJ/cm²/ns
            't_ns':  t_actual,
        }
        print(f"  Saved t = {t_phys:.3f} ns  (actual = {t_actual:.4f} ns)")

    return {
        'solutions': solutions,
        'x':   x_phys,
        'hx':  hx,
        'Lx':  R_MAX,
        'order': order,
        'iterations': its,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, savefile=''):
    """Plot temperature profiles at each output time vs. the self-similar solution."""
    solutions = results['solutions']
    x         = results['x']
    Lx        = results['Lx']
    order     = results['order']
    nI        = len(x)
    hx        = results['hx']
    edges     = np.linspace(0.0, Lx, nI + 1)
    xplot     = np.linspace(hx / 2.0, Lx, 2000)

    t_vals = sorted(solutions.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_vals)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.25))

    for idx, t_phys in enumerate(t_vals):
        sol   = solutions[t_phys]
        color = colors[idx]

        # S_N result
        T_interp = BPoly(sol['T'].T, edges)(xplot)
        ax.plot(xplot, T_interp, '-', color=color, lw=2.0, alpha=0.85)

        # Analytical self-similar solution
        if HAS_ANALYTICAL:
            try:
                T_anal, R_front = T_of_r_t(xplot, t_phys, N=1)
                ax.plot(xplot, np.maximum(T_anal, T_COLD), '--',
                        color=color, lw=1.5, alpha=0.6)
            except Exception:
                pass

    # ---- legend ----
    legend_elements = [
        Line2D([0], [0], color=colors[idx], lw=2.0,
               label=rf'$t = {t:.2g}$ ns')
        for idx, t in enumerate(t_vals)
    ]
    if HAS_ANALYTICAL:
        legend_elements += [
            Line2D([0], [0], color='k', lw=2.0,  ls='-',  label=r'S$_N$ transport'),
            Line2D([0], [0], color='k', lw=1.5,  ls='--', label='Self-similar'),
        ]

    ax.set_xlabel('Position (cm)', fontsize=14)
    ax.set_ylabel('Temperature $T$ (keV)', fontsize=14)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim(hx, R_MAX)
    ax.set_ylim(T_COLD * 0.5, None)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)

    if savefile and HAS_PLOTFUNCS:
        plt.tight_layout()
        show(savefile, close_after=True)
    else:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_npz(results, filename):
    """Save results dict to a compressed .npz file."""
    save_data = {
        'x':    results['x'],
        'hx':   np.array(results['hx']),
        'Lx':   np.array(results['Lx']),
        'order': np.array(results['order']),
    }
    for t_phys, sol in results['solutions'].items():
        key = f't_{t_phys:.4f}'
        save_data[f'{key}_T']     = sol['T']
        save_data[f'{key}_phi']   = sol['phi']
        save_data[f'{key}_t_ns']  = np.array(sol['t_ns'])
    np.savez_compressed(filename, **save_data)
    print(f"Results saved to {filename}")


def load_npz(filename):
    """Reconstruct a results dict from a previously saved .npz file."""
    data  = np.load(filename)
    x     = data['x']
    hx    = float(data['hx'])
    Lx    = float(data['Lx'])
    order = int(data['order'])
    solutions = {}
    for key in data.files:
        if key.endswith('_T') and key.startswith('t_'):
            t_str   = key[len('t_'):-len('_T')]
            t_phys  = float(t_str)
            solutions[t_phys] = {
                'T':    data[f't_{t_str}_T'],
                'phi':  data[f't_{t_str}_phi'],
                't_ns': float(data[f't_{t_str}_t_ns']),
            }
    return {'solutions': solutions, 'x': x, 'hx': hx, 'Lx': Lx, 'order': order}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Zeldovich wave problem with S_N transport')
    parser.add_argument('--zones', type=int, default=400,
                        help='Spatial zones per half-domain (default: 400)')
    parser.add_argument('--order', type=int, default=2,
                        help='Bernstein polynomial order (default: 2)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per time step (default: 2000)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[0.1, 0.3, 1.0, 3.0],
                        help='Physical output times in ns (default: 0.1 0.3 1.0 3.0)')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save plot to this file (PDF/PNG)')
    parser.add_argument('--save-npz', type=str, default='',
                        help='Save results to this .npz file')
    parser.add_argument('--load-npz', type=str, default='',
                        help='Load results from this .npz file (skip computation)')
    args = parser.parse_args()

    # Auto-generate a default npz cache path based on N and zone count,
    # mirroring the pattern used by sn_order_comparison.py.
    problems_dir = os.path.dirname(os.path.abspath(__file__))
    sn_dir       = os.path.dirname(problems_dir)
    default_npz  = os.path.join(sn_dir, f'zeldovich_sn_N{args.N}_I{args.zones}.npz')

    npz_load = args.load_npz if args.load_npz else default_npz
    npz_save = args.save_npz if args.save_npz else default_npz

    if os.path.exists(npz_load):
        print(f"Loading cached results from {npz_load}")
        results = load_npz(npz_load)
    else:
        results = setup_and_run(
            I=args.zones,
            order=args.order,
            N=args.N,
            K=args.K,
            maxits=args.maxits,
            LOUD=args.loud,
            output_times=tuple(args.output_times),
        )
        save_npz(results, npz_save)

    savefile = args.save_fig or ('zeldovich_sn.pdf' if HAS_PLOTFUNCS else '')
    plot_results(results, savefile=savefile)


if __name__ == '__main__':
    main()
