"""
Bennett problem with a thin Gaussian source for S_N transport.

Source:  S^{gs}(x, t) = exp(-x^2 / x_0^2) * Theta(t_0 - t)
         x_0 = 0.5 cm,  t_0 = 10 * tau  (same as the flat-top Bennett problem)

All other physics are identical to the flat-top Bennett problem:
  - sigma_a = 1.0 cm^{-1}, no scattering
  - Material energy: e = Cv * rho * T  (Cv = 0.03 GJ/(g*keV), rho = 1 g/cm^3)
  - Reflecting BC at x = 0  (emulated via full symmetric domain)
  - Vacuum BC at x = Lx = 20 cm

Normalization (matches tabulated transport reference):
  E_rad = phi / (a * c)      [since T_0 = 1 keV]  -- φ is energy flux
  E_mat = Cv * rho * T / a   [since T_0 = 1 keV]  -- e is energy density

Reference data: Su & Olson transport tables for the thin Gaussian source,
constant Cv problem (x_0 = 0.5, t_0 = 10, Cv0 = 0.03 GJ/cm^3/keV).

Run from the DiscreteOrdinates directory:
    python problems/bennett_gaussian_sn.py
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
A_RAD    = sn_solver.a     # radiation constant  (GJ / cm^3 keV^4)
C_LIGHT  = sn_solver.c     # speed of light      (cm / ns)
AC       = sn_solver.ac    # a * c
RHO      = 1.0             # density  (g/cm^3)
T_0      = 1.0             # reference temperature (keV)
CV_CONST = 0.03            # specific heat GJ/(g*keV) -- constant

# Gaussian source parameters
SOURCE_X0 = 0.5            # Gaussian half-width  (cm)

# ---------------------------------------------------------------------------
# Reference data: thin Gaussian source, constant Cv
# (Su & Olson transport solution)
# ---------------------------------------------------------------------------
ref_x = np.array([
    0.01, 0.1, 0.17783, 0.31623, 0.45, 0.5,
    0.56234, 0.75, 1.0, 1.33352, 1.77828, 3.16228,
    5.62341, 10.0, 17.78279])

ref_tau = np.array([0.1, 0.31623, 1.0, 3.16228, 10.0, 31.6228, 100.0])

# Transport reference: radiation scalar flux phi normalised by (a*c)
transport_rad_energy = np.array([
    [0.094715, 0.260632, 0.501868, 0.653649, 1.639873, 0.218959, 0.069516],  # x=0.01
    [0.091069, 0.251252, 0.488575, 0.636913, 1.614763, 0.218620, 0.069466],  # x=0.1
    [0.083585, 0.231922, 0.460879, 0.602409, 1.561824, 0.217881, 0.069356],  # x=0.17783
    [0.063732, 0.180063, 0.384158, 0.509408, 1.410010, 0.215547, 0.069011],  # x=0.31623
    [0.042445, 0.123229, 0.294366, 0.405067, 1.219261, 0.212063, 0.068495],  # x=0.45
    [0.035158, 0.103372, 0.260874, 0.367198, 1.142975, 0.210455, 0.068256],  # x=0.5
    [0.027037, 0.080916, 0.221011, 0.322683, 1.047037, 0.208219, 0.067923],  # x=0.56234
    [0.010181, 0.032561, 0.122095, 0.213179, 0.768814, 0.199970, 0.066689],  # x=0.75
    [0.001796, 0.006473, 0.045352, 0.123297, 0.461235, 0.185567, 0.064514],  # x=1.0
    [8.2e-05,  0.000367, 0.008352, 0.064418, 0.223392, 0.160718, 0.060695],  # x=1.33352
    [np.nan,   2e-06,    0.000371, 0.029558, 0.097211, 0.118847, 0.054065],  # x=1.77828
    [np.nan,   np.nan,   np.nan,   0.001165, 0.011091, 0.014263, 0.024308],  # x=3.16228
    [np.nan,   np.nan,   np.nan,   np.nan,   0.000340, 0.000570, 0.000634],  # x=5.62341
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   4e-06,    3e-06   ],  # x=10.0
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan  ],  # x=17.78279
])

# Transport reference: material energy density e normalised by (a*c)
transport_mat_energy = np.array([
    [0.004825, 0.044224, 0.323377, 1.468400, 2.455993, 1.518526, 1.133064],  # x=0.01
    [0.004638, 0.042571, 0.313295, 1.438115, 2.446023, 1.517940, 1.132861],  # x=0.1
    [0.004255, 0.039172, 0.292399, 1.373348, 2.424567, 1.516657, 1.132417],  # x=0.17783
    [0.003241, 0.030112, 0.235379, 1.183116, 2.359381, 1.512586, 1.131013],  # x=0.31623
    [0.002154, 0.020302, 0.170708, 0.944125, 2.267966, 1.506441, 1.128901],  # x=0.45
    [0.001783, 0.016913, 0.147332, 0.851718, 2.227558, 1.503576, 1.127920],  # x=0.5
    [0.001369, 0.013111, 0.120189, 0.740141, 2.172608, 1.499564, 1.126549],  # x=0.56234
    [0.000513, 0.005089, 0.057328, 0.458276, 1.969460, 1.484442, 1.121422],  # x=0.75
    [8.9e-05,  0.000949, 0.016339, 0.231272, 1.546952, 1.456735, 1.112196],  # x=1.0
    [4e-06,    4.8e-05,  0.002002, 0.097311, 0.889883, 1.404200, 1.095386],  # x=1.33352
    [np.nan,   np.nan,   5.5e-05,  0.032461, 0.400128, 1.294119, 1.064084],  # x=1.77828
    [np.nan,   np.nan,   np.nan,   0.000322, 0.042204, 0.349552, 0.857807],  # x=3.16228
    [np.nan,   np.nan,   np.nan,   np.nan,   0.001039, 0.014856, 0.054666],  # x=5.62341
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   8.8e-05,  0.000336],  # x=10.0
    [np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan  ],  # x=17.78279
])


# ---------------------------------------------------------------------------
# Problem setup & run
# ---------------------------------------------------------------------------
def setup_and_run(I=400, order=2, N=8, K=800, maxits=2000, LOUD=0,
                  output_tau=(0.1, 1.0, 3.16228, 10.0)):
    """Run the Bennett Gaussian-source problem with S_N transport.

    The reflecting BC at x=0 is emulated by solving on the full symmetric
    domain [0, 2*Lx_half] with the source centred at Lx_half and vacuum BCs
    on both sides.  Only the right half is returned.

    Parameters
    ----------
    I : int
        Spatial zones per half-domain (internal grid uses 2*I).
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates.
    K, maxits, LOUD : int
        DMD parameters / verbosity.
    output_tau : tuple of float
        Output times in mean-free-times.

    Returns
    -------
    results : dict  (solutions, x, hx, Lx, order, iterations, tau_mft)
    """
    # --- geometry ---
    Lx_half = 10.0
    I_full  = 2 * I
    Lx_full = 2 * Lx_half
    hx      = Lx_full / I_full
    center  = Lx_half
    x_comp  = np.linspace(hx / 2, Lx_full - hx / 2, I_full)
    x_phys  = x_comp[I:] - center   # right half, starting at hx/2

    # --- problem parameters ---
    sigma_a         = 1.0
    tau_mft         = 1.0 / (C_LIGHT * sigma_a)
    source_duration = 10.0 * tau_mft  # t_0 = 10 tau
    Tinit           = 0.0001           # keV
    Q_peak          = AC * T_0**4     # peak source magnitude (at x=0)

    print(f"Bennett Gaussian S_N problem  (N={N}, I={I}, order={order})")
    print(f"  x_0 = {SOURCE_X0} cm,  t_0 = 10 tau")
    print(f"  tau = {tau_mft:.6e} ns,  source off at {source_duration:.6e} ns")
    print(f"  Full domain cells = {I_full}  (2 x {I})")

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

    # --- vacuum boundary conditions (both sides of full domain) ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        return np.zeros((N, order + 1))

    # --- initial conditions ---
    T   = np.ones((I_full, order + 1)) * Tinit
    phi = AC * T**4
    psi = np.zeros((I_full, N, order + 1)) + phi[:, None, :]

    # --- Gaussian source  exp(-(x - center)^2 / x_0^2) ---
    q_on = np.zeros((I_full, N, order + 1))
    for i in range(I_full):
        q_on[i, :, :] = Q_peak * np.exp(-((x_comp[i] - center) / SOURCE_X0)**2)
    q_off = np.zeros((I_full, N, order + 1))

    # --- time stepping ---
    dt_min = 0.001 * tau_mft
    dt_max = 0.5   * tau_mft

    output_tau      = np.array(sorted(output_tau))
    output_times_ns = output_tau * tau_mft
    early_mask      = output_times_ns <= source_duration * (1 + 1e-10)
    late_mask       = ~early_mask
    early_outputs   = output_times_ns[early_mask]
    late_outputs    = output_times_ns[late_mask]

    solutions        = {}
    total_iterations = 0

    # ---- Phase 1: source ON ----
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

    for tau_val in output_tau[early_mask]:
        t_ns     = tau_val * tau_mft
        tstep    = np.argmin(np.abs(t_ns - ts1))
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
            t_rel    = tau_val * tau_mft - source_duration
            tstep    = np.argmin(np.abs(t_rel - ts2))
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
    x    = results['x']
    Lx   = results['Lx']
    nI   = len(x)
    hx   = results['hx']
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
        ti = np.argmin(np.abs(ref_tau - tau_val))
        if abs(ref_tau[ti] - tau_val) < 0.01 * tau_val:
            rr = transport_rad_energy[:, ti]
            rm = transport_mat_energy[:, ti]
            valid = ~np.isnan(rr)
            ax1.plot(ref_x[valid], rr[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)
            valid = ~np.isnan(rm)
            ax2.plot(ref_x[valid], rm[valid], 's',
                     color=col, ms=5, mec='k', mew=0.5, alpha=0.8)

    for ax, title, ylabel in [
        (ax1, 'Radiation energy density', r'$\phi\,/\,(a\,c)$'),
        (ax2, 'Material energy density',  r'$C_v\,\rho\,T\,/\,(a\,c)$')
    ]:
        ax.set_xlabel('Position (cm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.01, 15)
        ax.set_ylim(1e-4, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ref_handle = Line2D([], [], marker='s', color='gray', ms=5,
                        mec='k', mew=0.5, ls='', label='Transport ref')
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[0]) + [ref_handle],
               fontsize=9, loc='best')

    fig.suptitle(r'Bennett Gaussian Source — S$_N$ Transport', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    if HAS_PLOTFUNCS:
        show('bennett_gaussian_sn.pdf', close_after=True)
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
    data    = np.load(filename)
    x       = data['x']
    tau_mft = float(data['tau_mft'])
    hx      = x[0] * 2
    Lx      = x[-1] + hx / 2
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
        description='Bennett Gaussian-source problem with S_N transport')
    parser.add_argument('--zones',  type=int,   default=400)
    parser.add_argument('--order',  type=int,   default=2)
    parser.add_argument('--N',      type=int,   default=8)
    parser.add_argument('--K',      type=int,   default=800)
    parser.add_argument('--maxits', type=int,   default=2000)
    parser.add_argument('--loud',   type=int,   default=0)
    parser.add_argument('--output-tau', type=float, nargs='+',
                        default=[0.1, 1.0, 3.16228, 10.0])
    parser.add_argument('--save-npz',  type=str, default='',
                        help='NPZ filename (default: bennett_gaussian_sn_N{N}_I{I}.npz)')
    parser.add_argument('--no-plot',   action='store_true', default=False)
    parser.add_argument('--save-fig',  type=str, default='')
    args = parser.parse_args()

    results  = setup_and_run(
        I=args.zones, order=args.order, N=args.N,
        K=args.K, maxits=args.maxits, LOUD=args.loud,
        output_tau=tuple(args.output_tau))

    npz_name = args.save_npz or f'bennett_gaussian_sn_N{args.N}_I{args.zones}.npz'
    save_npz(results, npz_name)

    if (not args.no_plot) or args.save_fig:
        plot_results(results, savefile=args.save_fig)


if __name__ == '__main__':
    main()
