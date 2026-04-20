"""
Su-Olson test problem for S_N transport.

Compares discrete ordinates transport results against tabulated reference
data from Su & Olson (1997).

Problem setup:
  - 1-D slab geometry
  - Radiation source Q = a c T_0^4 in [0, 0.5] cm for t < 10 tau
  - sigma_a = 1.0 cm^{-1} (constant), no scattering
  - Material energy: e = a T^4  (radiation-dominated)
  - tau = 1 / (c sigma_a) ~ 0.03336 ns
  - Reflecting BC at x = 0  (emulated via full domain [-Lx, Lx])
  - Vacuum BC at x = Lx
  - Output at: 0.1, 1.0, 3.16228, 10.0, 31.6228, 100.0  mean free times

The reflecting BC at x=0 is enforced by solving on the symmetric
full domain [0, 2*Lx] with the source centered and vacuum BCs on
both sides; only the right half is reported.

Run from the DiscreteOrdinates directory:
    python problems/test_su_olson.py
"""

import sys
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from matplotlib.lines import Line2D
from numba import jit, njit, float64

# Add parent directory so we can import the solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver

# Add project root for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
A_RAD = sn_solver.a       # radiation constant  (GJ / cm^3 keV^4)
C_LIGHT = sn_solver.c     # speed of light      (cm / ns)
AC = sn_solver.ac          # a * c
RHO = 1.0                  # density (g/cm^3)
T_0 = 1.0                  # reference temperature (keV)

# ---------------------------------------------------------------------------
# Su-Olson reference data (from Su & Olson 1997, Tables II-III)
# ---------------------------------------------------------------------------
su_olson_x = np.array([
    0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5,
    0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228,
    5.62341, 10.0, 17.78279])

su_olson_tau = np.array([
    0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100.0])

# Transport reference: radiation energy density (normalised)
transport_rad_energy = np.array([
    [0.09531, 0.27526, 0.64308, 1.20052, 2.23575, 0.69020, 0.35720],
    [0.09531, 0.27526, 0.63585, 1.18869, 2.21944, 0.68974, 0.35714],
    [0.09532, 0.27527, 0.61958, 1.16190, 2.18344, 0.68878, 0.35702],
    [0.09529, 0.26262, 0.56187, 1.07175, 2.06448, 0.68569, 0.35664],
    [0.08823, 0.20312, 0.44711, 0.90951, 1.86072, 0.68111, 0.35599],
    [0.04765, 0.13762, 0.35801, 0.79902, 1.73178, 0.67908, 0.35574],
    [0.00375, 0.06277, 0.25374, 0.66678, 1.57496, 0.67619, 0.35538],
    [np.nan, 0.00280, 0.11430, 0.44675, 1.27398, 0.66548, 0.35393],
    [np.nan, np.nan, 0.03648, 0.27540, 0.98782, 0.64691, 0.35141],
    [np.nan, np.nan, 0.00291, 0.14531, 0.70822, 0.61538, 0.34697],
    [np.nan, np.nan, np.nan, 0.05968, 0.45016, 0.56353, 0.33924],
    [np.nan, np.nan, np.nan, 0.00123, 0.09673, 0.36965, 0.30346],
    [np.nan, np.nan, np.nan, np.nan, 0.00375, 0.10830, 0.21382],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00390, 0.07200],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00272],
])

# Transport reference: material energy density (normalised)
transport_mat_energy = np.array([
    [0.00468, 0.04093, 0.27126, 0.94670, 2.11186, 0.70499, 0.35914],
    [0.00468, 0.04093, 0.26839, 0.93712, 2.09585, 0.70452, 0.35908],
    [0.00468, 0.04093, 0.26261, 0.91525, 2.06052, 0.70348, 0.35895],
    [0.00468, 0.04032, 0.23978, 0.84082, 1.94365, 0.70020, 0.35854],
    [0.00455, 0.03314, 0.18826, 0.70286, 1.74291, 0.69532, 0.35793],
    [0.00234, 0.02046, 0.14187, 0.60492, 1.61536, 0.69308, 0.35766],
    [0.00005, 0.00635, 0.08838, 0.48843, 1.46027, 0.68994, 0.35728],
    [np.nan, 0.00005, 0.03014, 0.30656, 1.16591, 0.67850, 0.35581],
    [np.nan, np.nan, 0.00625, 0.17519, 0.88992, 0.65868, 0.35326],
    [np.nan, np.nan, 0.00017, 0.08352, 0.62521, 0.62507, 0.34875],
    [np.nan, np.nan, np.nan, 0.02935, 0.38688, 0.57003, 0.34086],
    [np.nan, np.nan, np.nan, 0.00025, 0.07642, 0.36727, 0.30517],
    [np.nan, np.nan, np.nan, np.nan, 0.00253, 0.10312, 0.21377],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00342, 0.07122],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00261],
])


# ---------------------------------------------------------------------------
# Problem setup & run
# ---------------------------------------------------------------------------
def setup_and_run(I=400, order=2, N=8, K=800, maxits=2000, LOUD=0,
                  output_tau=(0.1, 1.0, 3.16228, 10.0)):
    """Run the Su-Olson problem with S_N transport.

    The reflecting BC at x=0 is emulated by solving on a full symmetric
    domain [0, 2*Lx_half] with the source centered at Lx_half and vacuum
    BCs on both sides.  Only the right half is returned.

    Parameters
    ----------
    I : int
        Number of spatial zones (per half-domain; internal grid uses 2*I).
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates.
    K : int
        DMD inner iterations.
    maxits : int
        Max DMD solver iterations per time step.
    LOUD : int
        Verbosity level.
    output_tau : tuple of float
        Output times in mean-free-times.

    Returns
    -------
    results : dict
        Keys: solutions (dict tau->data), x, hx, Lx, order, iterations.
    """
    # --- geometry (full domain for reflecting BC) ---
    Lx_half = 20.0                     # physical half-domain
    I_full = 2 * I                     # total cells (symmetric domain)
    Lx_full = 2 * Lx_half
    hx = Lx_full / I_full              # = Lx_half / I
    center = Lx_half                   # symmetry centre
    x_comp = np.linspace(hx / 2, Lx_full - hx / 2, I_full)
    # physical x (right half only, reported to user)
    x_phys = x_comp[I:] - center       # starts at hx/2

    # --- problem parameters ---
    sigma_a = 1.0
    source_half = 0.5                   # source occupies [center-0.5, center+0.5]
    tau_mft = 1.0 / (C_LIGHT * sigma_a)
    source_duration = 10.0 * tau_mft
    Tinit = 0.001
    Q_ext = AC * T_0**4

    print(f"Mean-free time  tau = {tau_mft:.6e} ns")
    print(f"Source duration      = {source_duration:.6e} ns  (10 tau)")
    print(f"Full domain cells    = {I_full}  (2 x {I})")

    ac = AC

    # --- EOS: e = a T^4 ---
    @njit
    def eos(T):
        return A_RAD * np.maximum(T, 0.0)**4

    @njit
    def invEOS(E):
        return (np.maximum(E, 0.0) / A_RAD)**0.25

    # --- opacity (constant) ---
    @njit
    def sigma_func(T):
        return np.ones_like(T) * sigma_a

    # --- no scattering ---
    @njit
    def scat_func(T):
        return np.zeros_like(T)

    # --- boundary conditions (vacuum both sides) ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        return np.zeros((N, order + 1))

    # --- initial conditions (full domain) ---
    T = np.ones((I_full, order + 1)) * Tinit
    phi = ac * T**4
    psi = np.zeros((I_full, N, order + 1)) + phi[:, None, :]

    # --- external source (centred) ---
    q_on = np.zeros((I_full, N, order + 1))
    for i in range(I_full):
        if abs(x_comp[i] - center) < source_half:
            q_on[i, :, :] = Q_ext
    q_off = np.zeros((I_full, N, order + 1))

    # --- time parameters ---
    dt_min = 0.001 * tau_mft
    dt_max = 0.5 * tau_mft

    # split output times into early (source on) and late (source off)
    output_tau = np.array(sorted(output_tau))
    output_times_ns = output_tau * tau_mft
    early_mask = output_times_ns <= source_duration * (1 + 1e-10)
    late_mask = ~early_mask
    early_outputs = output_times_ns[early_mask]
    late_outputs = output_times_ns[late_mask]

    solutions = {}
    total_iterations = 0

    # ---- Phase 1: source ON, t in [0, source_duration] ----
    phase1_final = source_duration
    if early_outputs.size > 0:
        phase1_final = max(phase1_final, early_outputs[-1])

    print(f"\n--- Phase 1: source ON, 0 -> {phase1_final / tau_mft:.2f} tau ---")
    phis1, Ts1, its1, ts1 = sn_solver.temp_solve_dmd_inc(
        I_full, hx, q_on, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=phase1_final,
        order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3,
        time_outputs=early_outputs if early_outputs.size > 0 else None)
    total_iterations += its1
    print(f"Phase 1 sweeps: {its1}")

    # extract early-time snapshots (right half only)
    for tau_val in output_tau[early_mask]:
        t_ns = tau_val * tau_mft
        tstep = np.argmin(np.abs(t_ns - ts1))
        phi_snap = phis1[tstep][I:, :]     # right half
        T_snap = Ts1[tstep][I:, :]
        E_rad_norm = phi_snap / (ac * T_0**4)
        E_mat_norm = T_snap**4 / T_0**4
        solutions[tau_val] = {
            'phi': phi_snap, 'T': T_snap,
            'E_rad': E_rad_norm, 'E_mat': E_mat_norm,
            'tstep': tstep, 't_ns': ts1[tstep]
        }
        print(f"  Saved tau = {tau_val:.4f}  (t = {ts1[tstep]:.6e} ns)")

    # ---- Phase 2: source OFF ----
    if late_outputs.size > 0:
        phi2 = phis1[-1].copy()
        T2 = Ts1[-1].copy()
        psi2 = np.zeros((I_full, N, order + 1)) + phi2[:, None, :]

        late_final = late_outputs[-1] - source_duration
        late_outs_rel = late_outputs - source_duration

        print(f"\n--- Phase 2: source OFF, 0 -> {late_final / tau_mft:.2f} tau ---")
        phis2, Ts2, its2, ts2 = sn_solver.temp_solve_dmd_inc(
            I_full, hx, q_off, sigma_func, scat_func, N, BCFunc, eos, invEOS,
            phi2, psi2, T2,
            dt_min=dt_min, dt_max=dt_max, tfinal=late_final,
            order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3,
            time_outputs=late_outs_rel)
        total_iterations += its2
        print(f"Phase 2 sweeps: {its2}")

        for tau_val in output_tau[late_mask]:
            t_rel = tau_val * tau_mft - source_duration
            tstep = np.argmin(np.abs(t_rel - ts2))
            phi_snap = phis2[tstep][I:, :]
            T_snap = Ts2[tstep][I:, :]
            E_rad_norm = phi_snap / (ac * T_0**4)
            E_mat_norm = T_snap**4 / T_0**4
            solutions[tau_val] = {
                'phi': phi_snap, 'T': T_snap,
                'E_rad': E_rad_norm, 'E_mat': E_mat_norm,
                'tstep': tstep, 't_ns': ts2[tstep] + source_duration
            }
            print(f"  Saved tau = {tau_val:.4f}  (t = {solutions[tau_val]['t_ns']:.6e} ns)")

    print(f"\nTotal transport sweeps: {total_iterations}")
    return {
        'solutions': solutions, 'x': x_phys, 'hx': hx, 'Lx': Lx_half,
        'order': order, 'iterations': total_iterations,
        'tau_mft': tau_mft
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, savefile=''):
    """Plot S_N results vs. transport reference data."""
    solutions = results['solutions']
    x = results['x']
    Lx = results['Lx']
    order = results['order']
    nI = len(x)
    hx = results['hx']
    edges = np.linspace(0, Lx, nI + 1)
    xplot = np.linspace(hx / 2, Lx, 2000)

    taus_available = sorted(solutions.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(taus_available)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, tau_val in enumerate(taus_available):
        sol = solutions[tau_val]
        col = colors[idx]

        # Bernstein interpolation for smooth curves
        E_rad_interp = BPoly(sol['E_rad'].T, edges)(xplot)
        E_mat_interp = BPoly(sol['E_mat'].T, edges)(xplot)

        label = rf'$\tau={tau_val:.2f}$'
        ax1.plot(xplot, E_rad_interp, '-', color=col, lw=1.5, label=label)
        ax2.plot(xplot, E_mat_interp, '-', color=col, lw=1.5, label=label)

        # overlay transport reference
        ti = np.argmin(np.abs(su_olson_tau - tau_val))
        if abs(su_olson_tau[ti] - tau_val) < 0.01 * tau_val:
            ref_rad = transport_rad_energy[:, ti]
            ref_mat = transport_mat_energy[:, ti]
            valid = ~np.isnan(ref_rad)
            ax1.plot(su_olson_x[valid], ref_rad[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)
            valid = ~np.isnan(ref_mat)
            ax2.plot(su_olson_x[valid], ref_mat[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)

    for ax, title, ylabel in [
        (ax1, 'Radiation Energy', r'$E_r / (a\,T_0^4)$'),
        (ax2, 'Material Energy', r'$e / (a\,T_0^4) = (T/T_0)^4$')
    ]:
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.2, 12)
        ax.set_ylim(1e-3, 3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # custom legend entry for reference markers
    ref_handle = Line2D([], [], marker='s', color='gray', ms=5,
                        mec='k', mew=0.5, ls='', label='Transport ref')
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[0]) + [ref_handle],
               fontsize=9, loc='best')

    fig.suptitle(r'Su-Olson Problem — S$_N$ Transport', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    if HAS_PLOTFUNCS:
        show('su_olson_sn.pdf', close_after=True)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Su-Olson test problem with S_N transport')
    parser.add_argument('--zones', type=int, default=400,
                        help='Number of spatial zones (default: 400)')
    parser.add_argument('--order', type=int, default=2,
                        help='Bernstein polynomial order (default: 2)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per step (default: 2000)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--output-tau', type=float, nargs='+',
                        default=[0.1, 1.0, 3.16228, 10.0],
                        help='Output times in mean-free-times '
                             '(default: 0.1 1.0 3.16228 10.0)')
    parser.add_argument('--save-npz', type=str, default='su_olson_sn.npz',
                        help='Output .npz file name')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Suppress interactive plot window')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save figure to file')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.zones, order=args.order, N=args.N,
        K=args.K, maxits=args.maxits, LOUD=args.loud,
        output_tau=tuple(args.output_tau))

    # save results
    tau_mft = results['tau_mft']
    sol_dict = results['solutions']
    save_data = {'x': results['x'], 'tau_mft': tau_mft}
    for tau_val, sol in sol_dict.items():
        key = f'tau_{tau_val:.4f}'
        save_data[f'{key}_E_rad'] = sol['E_rad']
        save_data[f'{key}_E_mat'] = sol['E_mat']
        save_data[f'{key}_t_ns'] = sol['t_ns']
    np.savez_compressed(args.save_npz, **save_data)
    print(f"Results saved to {args.save_npz}")

    if (not args.no_plot) or args.save_fig:
        plot_results(results, savefile=args.save_fig)


if __name__ == '__main__':
    main()
