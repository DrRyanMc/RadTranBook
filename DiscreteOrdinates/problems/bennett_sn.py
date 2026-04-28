"""
Bennett test problem for S_N transport.

Compares discrete ordinates transport results against tabulated reference
data from Su & Olson (1996) / Bennett.

Problem setup:
  - 1-D slab geometry
  - Radiation source Q = a*c in [0, 0.5] cm for t < 10 tau
  - sigma_a = 1.0 cm^{-1} (constant), no scattering
  - Material energy: e = Cv * rho * T  (linear EOS, Cv = 0.03 GJ/(g*keV))
  - tau = 1 / (c * sigma_a)
  - Reflecting BC at x = 0  (emulated via full domain [-Lx, Lx])
  - Vacuum BC at x = Lx = 20 cm
  - Output at: 0.1, 1.0, 3.16228, 10.0, 31.6228, 100.0  mean free times

The reflecting BC at x=0 is enforced by solving on the symmetric
full domain [0, 2*Lx] with the source centered and vacuum BCs on
both sides; only the right half is reported.

Normalization convention (matching Bennett transport tables):
  E_rad = phi / (a*c*T_0^4)  =  phi / (a*c)      (since T_0 = 1 keV)
  E_mat = e / (a*T_0^4)      =  Cv * rho * T / a  (since T_0 = 1 keV)

Run from the DiscreteOrdinates directory:
    python problems/bennett_sn.py
"""

import sys
import os
import argparse
import numpy as np
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
A_RAD   = sn_solver.a      # radiation constant  (GJ / cm^3 keV^4)
C_LIGHT = sn_solver.c      # speed of light      (cm / ns)
AC      = sn_solver.ac     # a * c
RHO     = 1.0              # density (g/cm^3)
T_0     = 1.0              # reference temperature (keV)
CV_CONST = 0.03            # specific heat (GJ / (g * keV)) -- constant for Bennett

# ---------------------------------------------------------------------------
# Bennett reference data (transport solution)
# x values (cm) and tau (mean-free-time) are the same grid as Su-Olson
# ---------------------------------------------------------------------------
su_olson_x = np.array([
    0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5,
    0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228,
    5.62341, 10.0, 17.78279])

su_olson_tau = np.array([
    0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100.0])

# Transport reference: radiation energy density (normalised by a*c)
transport_rad_energy = np.array([
    [0.095162, 0.271108, 0.563683, 0.765084, 1.96832,  0.267247, 0.085108],  # x=0.01
    [0.095162, 0.271108, 0.557609, 0.756116, 1.950367, 0.266877, 0.085054],  # x=0.10
    [0.095162, 0.271108, 0.543861, 0.736106, 1.910675, 0.266071, 0.084937],  # x=0.17783
    [0.095162, 0.258592, 0.495115, 0.668231, 1.779896, 0.263527, 0.084565],  # x=0.31623
    [0.08809,  0.199962, 0.396442, 0.543721, 1.558248, 0.259729, 0.084008],  # x=0.45
    [0.047581, 0.135554, 0.316071, 0.453151, 1.420865, 0.257976, 0.083750],  # x=0.50
    [0.00376,  0.061935, 0.222261, 0.349209, 1.252213, 0.255538, 0.083392],  # x=0.56234
    [np.nan,   0.002788, 0.102348, 0.210780, 0.908755, 0.246543, 0.082061],  # x=0.75
    [np.nan,   np.nan,   0.034228, 0.124305, 0.562958, 0.230831, 0.079715],  # x=1.0
    [np.nan,   np.nan,   0.002864, 0.067319, 0.277520, 0.203718, 0.075591],  # x=1.33352
    [np.nan,   np.nan,   np.nan,   0.031357, 0.120054, 0.158039, 0.068419],  # x=1.77828
    [np.nan,   np.nan,   np.nan,   0.001057, 0.013737, 0.022075, 0.036021],  # x=3.16228
    [np.nan,   np.nan,   np.nan,   np.nan,   0.000413, 0.000814, 0.001068],  # x=5.62341
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   5e-06,    5e-06   ],  # x=10.0
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan  ],  # x=17.78279
])

# Transport reference: material energy density (normalised by a*c)
transport_mat_energy = np.array([
    [0.004837, 0.045121, 0.354022, 1.613529, 2.574610,  1.592549, 1.190296],  # x=0.01
    [0.004837, 0.045121, 0.350958, 1.601467, 2.568476,  1.591998, 1.190108],  # x=0.10
    [0.004837, 0.045121, 0.343803, 1.573757, 2.554747,  1.590795, 1.189698],  # x=0.17783
    [0.004837, 0.044507, 0.316063, 1.470780, 2.507772,  1.586979, 1.188398],  # x=0.31623
    [0.004705, 0.036765, 0.249325, 1.238666, 2.421019,  1.581228, 1.186445],  # x=0.45
    [0.002419, 0.022562, 0.183937, 1.025219, 2.361647,  1.578549, 1.185538],  # x=0.50
    [5.1e-05,  0.006779, 0.108887, 0.759317, 2.280932,  1.574800, 1.184271],  # x=0.56234
    [np.nan,   6.4e-05,  0.034842, 0.416175, 2.069946,  1.560710, 1.179537],  # x=0.75
    [np.nan,   np.nan,   0.006872, 0.214491, 1.685160,  1.535052, 1.171036],  # x=1.0
    [np.nan,   np.nan,   0.000168, 0.094966, 1.028758,  1.487096, 1.155611],  # x=1.33352
    [np.nan,   np.nan,   np.nan,   0.032116, 0.471906,  1.391456, 1.127131],  # x=1.77828
    [np.nan,   np.nan,   np.nan,   0.000196, 0.049604,  0.471468, 0.954827],  # x=3.16228
    [np.nan,   np.nan,   np.nan,   np.nan,   0.001163,  0.019493, 0.082189],  # x=5.62341
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    0.000113, 0.000487],  # x=10.0
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,    np.nan,   np.nan  ],  # x=17.78279
])


# ---------------------------------------------------------------------------
# Problem setup & run
# ---------------------------------------------------------------------------
def setup_and_run(I=400, order=2, N=8, K=800, maxits=2000, LOUD=0,
                  output_tau=(0.1, 1.0, 3.16228, 10.0)):
    """Run the Bennett problem with S_N transport.

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
    Lx_half = 10.0                      # physical half-domain
    I_full  = 2 * I                     # total cells (symmetric domain)
    Lx_full = 2 * Lx_half
    hx      = Lx_full / I_full          # = Lx_half / I
    center  = Lx_half                   # symmetry centre
    x_comp  = np.linspace(hx / 2, Lx_full - hx / 2, I_full)
    # physical x (right half only, reported to user)
    x_phys  = x_comp[I:] - center       # starts at hx/2

    # --- problem parameters ---
    sigma_a         = 1.0
    source_half     = 0.5               # source occupies [center-0.5, center+0.5]
    tau_mft         = 1.0 / (C_LIGHT * sigma_a)
    source_duration = 10.0 * tau_mft
    Tinit           = 0.001             # keV
    Q_ext           = AC * T_0**4      # = AC since T_0 = 1 keV  [GJ/(cm^2 ns)]

    print(f"Bennett S_N problem  (N={N}, I={I}, order={order})")
    print(f"Mean-free time  tau = {tau_mft:.6e} ns")
    print(f"Source duration      = {source_duration:.6e} ns  (10 tau)")
    print(f"Full domain cells    = {I_full}  (2 x {I})")

    # --- EOS: e = Cv * rho * T  (linear) ---
    @njit
    def eos(T):
        return CV_CONST * RHO * np.maximum(T, 0.0)

    @njit
    def invEOS(E):
        return np.maximum(E, 0.0) / (CV_CONST * RHO)

    # --- opacity (constant) ---
    @njit
    def sigma_func(T):
        return np.ones_like(T) * sigma_a

    # --- no scattering ---
    @njit
    def scat_func(T):
        return np.zeros_like(T)

    # --- boundary conditions (vacuum both sides of full domain) ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        return np.zeros((N, order + 1))

    # --- initial conditions (full domain) ---
    T   = np.ones((I_full, order + 1)) * Tinit
    phi = AC * T**4
    psi = np.zeros((I_full, N, order + 1)) + phi[:, None, :]

    # --- external source (centred) ---
    q_on = np.zeros((I_full, N, order + 1))
    for i in range(I_full):
        if abs(x_comp[i] - center) < source_half:
            q_on[i, :, :] = Q_ext
    q_off = np.zeros((I_full, N, order + 1))

    # --- time parameters ---
    dt_min = 0.001 * tau_mft
    dt_max = 0.5   * tau_mft

    # split output times into early (source on) and late (source off)
    output_tau      = np.array(sorted(output_tau))
    output_times_ns = output_tau * tau_mft
    early_mask      = output_times_ns <= source_duration * (1 + 1e-10)
    late_mask       = ~early_mask
    early_outputs   = output_times_ns[early_mask]
    late_outputs    = output_times_ns[late_mask]

    solutions        = {}
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
        t_ns   = tau_val * tau_mft
        tstep  = np.argmin(np.abs(t_ns - ts1))
        phi_snap = phis1[tstep][I:, :]
        T_snap   = Ts1[tstep][I:, :]
        E_rad_norm = phi_snap / (AC * T_0**4)
        E_mat_norm = CV_CONST * RHO * T_snap / (A_RAD * T_0**4)
        solutions[tau_val] = {
            'phi': phi_snap, 'T': T_snap,
            'E_rad': E_rad_norm, 'E_mat': E_mat_norm,
            'tstep': tstep, 't_ns': ts1[tstep]
        }
        print(f"  Saved tau = {tau_val:.4f}  (t = {ts1[tstep]:.6e} ns)")

    # ---- Phase 2: source OFF ----
    if late_outputs.size > 0:
        phi2 = phis1[-1].copy()
        T2   = Ts1[-1].copy()
        psi2 = np.zeros((I_full, N, order + 1)) + phi2[:, None, :]

        late_final    = late_outputs[-1] - source_duration
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
            t_rel  = tau_val * tau_mft - source_duration
            tstep  = np.argmin(np.abs(t_rel - ts2))
            phi_snap = phis2[tstep][I:, :]
            T_snap   = Ts2[tstep][I:, :]
            E_rad_norm = phi_snap / (AC * T_0**4)
            E_mat_norm = CV_CONST * RHO * T_snap / (A_RAD * T_0**4)
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
    x   = results['x']
    Lx  = results['Lx']
    order = results['order']
    nI  = len(x)
    hx  = results['hx']
    edges = np.linspace(0, Lx, nI + 1)
    xplot = np.linspace(hx / 2, Lx, 2000)

    taus_available = sorted(solutions.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(taus_available)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for idx, tau_val in enumerate(taus_available):
        sol = solutions[tau_val]
        col = colors[idx]

        E_rad_interp = BPoly(sol['E_rad'].T, edges)(xplot)
        E_mat_interp = BPoly(sol['E_mat'].T, edges)(xplot)

        label = rf'$\tau={tau_val:.2f}$'
        ax1.plot(xplot, E_rad_interp, '-', color=col, lw=1.5, label=label)
        ax2.plot(xplot, E_mat_interp, '-', color=col, lw=1.5, label=label)

        # overlay transport reference
        ti   = np.argmin(np.abs(su_olson_tau - tau_val))
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
        (ax1, 'Radiation Energy', r'$\phi\,/\,(a\,c)$'),
        (ax2, 'Material Energy',  r'$C_v\,\rho\,T\,/\,(a\,c)$')
    ]:
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.2, 12)
        ax.set_ylim(1e-3, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ref_handle = Line2D([], [], marker='s', color='gray', ms=5,
                        mec='k', mew=0.5, ls='', label='Transport ref')
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[0]) + [ref_handle],
               fontsize=9, loc='best')

    fig.suptitle(r'Bennett Problem — S$_N$ Transport', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    if HAS_PLOTFUNCS:
        show('bennett_sn.pdf', close_after=True)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers: save / load
# ---------------------------------------------------------------------------
def save_npz(results, filename):
    tau_mft  = results['tau_mft']
    sol_dict = results['solutions']
    save_data = {'x': results['x'], 'tau_mft': tau_mft}
    for tau_val, sol in sol_dict.items():
        key = f'tau_{tau_val:.4f}'
        save_data[f'{key}_E_rad'] = sol['E_rad']
        save_data[f'{key}_E_mat'] = sol['E_mat']
        save_data[f'{key}_t_ns']  = sol['t_ns']
    np.savez_compressed(filename, **save_data)
    print(f"Results saved to {filename}")


def load_npz(filename):
    """Reconstruct a results dict from a previously saved .npz file."""
    data      = np.load(filename)
    x         = data['x']
    tau_mft   = float(data['tau_mft'])
    hx        = x[0] * 2          # x[0] = hx/2
    Lx        = x[-1] + hx / 2
    solutions = {}
    for key in data.files:
        if key.endswith('_E_rad'):
            tau_str = key[len('tau_'):-len('_E_rad')]
            tau_val = float(tau_str)
            solutions[tau_val] = {
                'E_rad': data[f'tau_{tau_str}_E_rad'],
                'E_mat': data[f'tau_{tau_str}_E_mat'],
                't_ns':  float(data[f'tau_{tau_str}_t_ns']),
            }
    return {'solutions': solutions, 'x': x, 'hx': hx, 'Lx': Lx,
            'tau_mft': tau_mft}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Bennett test problem with S_N transport')
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
    parser.add_argument('--save-npz', type=str, default='',
                        help='Output .npz file name '
                             '(default: bennett_sn_N{N}_I{zones}.npz)')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Suppress interactive plot window')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save figure to file')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.zones, order=args.order, N=args.N,
        K=args.K, maxits=args.maxits, LOUD=args.loud,
        output_tau=tuple(args.output_tau))

    npz_name = args.save_npz or f'bennett_sn_N{args.N}_I{args.zones}.npz'
    save_npz(results, npz_name)

    if (not args.no_plot) or args.save_fig:
        plot_results(results, savefile=args.save_fig)


if __name__ == '__main__':
    main()
