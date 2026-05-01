"""
Simple Marshak wave problem.

Solves the classic thermal radiation Marshak wave with temperature-
dependent opacity using the 1-D discrete ordinates solver with
DMD-accelerated source iteration.

Problem setup (matches the non-equilibrium diffusion version):
  - Left boundary:  incoming blackbody radiation at T = 1 keV
  - Right boundary: vacuum (zero incoming)
  - Opacity:        sigma = 300 * T^{-3}  (cm^{-1}, T in keV)
  - Heat capacity:  c_v = 0.3  GJ/(cm^3 keV)
  - No scattering
  - Domain:         [0, 0.5] cm
  - Initial T:      0.01 keV

Run from the DiscreteOrdinates directory:
    python problems/test_simple_marshak.py
"""

import sys
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from numba import jit, njit, float64

# Add parent directory so we can import the solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# sn_solver uses nanosecond time units:
#   c  = 29.98  cm/ns
#   a  = 0.01372 GJ/(cm^3 keV^4)
#   ac = a * c
C_LIGHT = 29.98         # cm/ns
A_RAD = 0.01372         # GJ/(cm^3 keV^4)
RHO = 1.0               # g/cm^3
CV_VOL = 0.3            # GJ/(cm^3 keV)


# ---------------------------------------------------------------------------
# Setup & run
# ---------------------------------------------------------------------------
def setup_and_run(I=200, order=3, N=8, tfinal=10.0,
                  dt_min=1e-5, dt_max=0.05, K=800, maxits=2000,
                  LOUD=0):
    """Set up and run the simple Marshak wave.

    Parameters
    ----------
    I : int
        Number of spatial zones.
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates (S_N order).
    tfinal : float
        Final time in nanoseconds.
    dt_min, dt_max : float
        Adaptive time step bounds (ns).
    K : int
        Number of DMD inner iterations.
    maxits : int
        Maximum DMD solver iterations per time step.
    LOUD : int
        Verbosity level.

    Returns
    -------
    results : dict
        Dictionary with keys: phis, Ts, iterations, ts, x, hx, Lx, order
    """
    # --- geometry ---
    Lx = 0.5
    hx = Lx / I
    x = np.linspace(hx / 2, Lx - hx / 2, I)

    # --- material parameters ---
    sigma_coeff = 300.0   # opacity coefficient
    n_opacity = 3         # opacity temperature exponent
    Tinit = 0.001          # initial temperature (keV)
    T_bc = 1.0            # left-boundary blackbody temperature (keV)
    T_min = 0.0001         # floor to prevent overflow in T^{-n}

    ac = sn_solver.ac

    # --- quadrature ---
    MU, W = np.polynomial.legendre.leggauss(N)

    # --- external source (zero) ---
    q = np.zeros((I, N, order + 1))

    # --- EOS:  e = c_v * T  (linear) ---
    @njit
    def eos(T):
        return CV_VOL * T

    @njit
    def invEOS(E):
        return E / CV_VOL

    # --- absorption opacity: sigma_a = 300 * |T|^{-3} ---
    @njit
    def sigma_func(T):
        return sigma_coeff * np.maximum(np.abs(T), T_min)**(-n_opacity)

    # --- scattering (none) ---
    @njit
    def scat_func(T):
        return np.zeros_like(T)

    # --- initial conditions ---
    T = np.ones((I, order + 1)) * Tinit
    phi = ac * T**4
    psi = np.zeros((I, N, order + 1)) + phi[:, None, :]

    # --- boundary conditions ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        out = np.zeros((N, order + 1))
        out[MU > 0, :] = ac * T_bc**4       # left: blackbody at 1 keV
        out[MU < 0, :] = ac * Tinit**4       # right: cold background
        return out

    # --- run ---
    print(f"Running simple Marshak wave: I={I}, order={order}, N={N}, "
          f"tfinal={tfinal} ns")
    phis, Ts, iterations, ts = sn_solver.temp_solve_dmd_inc(
        I, hx, q, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=10)
    print(f"\nDone. Total transport sweeps: {iterations}")

    results = {
        'phis': phis, 'Ts': Ts, 'iterations': iterations, 'ts': ts,
        'x': x, 'hx': hx, 'Lx': Lx, 'order': order
    }
    return results


# ---------------------------------------------------------------------------
# Self-similar diffusion reference
# ---------------------------------------------------------------------------
def self_similar_T(x, t_ns):
    """Self-similar diffusion solution for the material temperature.

    Valid for the nonlinear Marshak wave with sigma = 300*T^{-3} and
    linear heat capacity c_v = 0.3.  Uses the Pomraning similarity
    variable.  *t_ns* is time in nanoseconds.
    """
    xi_max = 1.11305
    omega = 0.05989
    # diffusion similarity constant  K = 8*a*c / ((n+4)*3*sigma0*rho*cv)
    K_const = (8 * A_RAD * C_LIGHT
               / ((4 + 3) * 3 * 300.0 * RHO * CV_VOL))
    xi = x / np.sqrt(K_const * np.maximum(t_ns, 1e-30))
    T_ref = np.where(
        xi < xi_max,
        np.power(np.maximum((1 - xi / xi_max) * (1 + omega * xi / xi_max),
                             0.0), 1.0 / 6.0),
        0.0)
    return T_ref


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results, plot_times_ns=(1.0, 5.0, 10.0), savefile=''):
    """Plot T and T_r profiles and compare with the diffusion self-similar solution.

    Parameters
    ----------
    results : dict
        Output of ``setup_and_run``.
    plot_times_ns : tuple of float
        Times to plot, in nanoseconds.
    savefile : str
        If non-empty, save the figure to this path.
    """
    phis = results['phis']
    Ts = results['Ts']
    ts = results['ts']
    Lx = results['Lx']
    order = results['order']
    nI = len(Ts[0])
    ac = sn_solver.ac

    xplot = np.linspace(0, Lx, 1000)
    edges = np.linspace(0, Lx, nI + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for t_ns in plot_times_ns:
        tstep = np.argmin(np.abs(t_ns - ts))
        t_actual_ns = ts[tstep]

        # material temperature (Bernstein interpolation)
        T_sol = Ts[tstep]
        T_plot = BPoly(T_sol.transpose(), edges)(xplot)

        # radiation temperature
        Tr_sol = (phis[tstep] / ac)**0.25
        Tr_plot = BPoly(Tr_sol.transpose(), edges)(xplot)

        label_t = f't = {t_actual_ns:.1f} ns'
        p = ax.plot(xplot, T_plot, '-', label=f'$T$ {label_t}')
        color = p[0].get_color()
        ax.plot(xplot, Tr_plot, '--', color=color, label=f'$T_r$ {label_t}')

        # diffusion self-similar reference
        T_ss = self_similar_T(xplot, t_actual_ns)
        ax.plot(xplot, T_ss, ':', color=color, alpha=0.6,
                label=f'SS {label_t}')

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title(r'Simple Marshak Wave ($\sigma=300\,T^{-3}$, S$_N$)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, Lx)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run simple Marshak wave with S_N transport')
    parser.add_argument('--zones', type=int, default=200,
                        help='Number of spatial zones (default: 200)')
    parser.add_argument('--order', type=int, default=3,
                        help='Bernstein polynomial order (default: 3)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--tfinal', type=float, default=10.0,
                        help='Final time in ns (default: 10.0)')
    parser.add_argument('--dt-min', type=float, default=1e-5,
                        help='Minimum time step in ns (default: 1e-5)')
    parser.add_argument('--dt-max', type=float, default=0.05,
                        help='Maximum time step in ns (default: 0.05)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per step (default: 2000)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--save-npz', type=str,
                        default='simple_marshak_sn.npz',
                        help='Output .npz file name')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Show plots after running')
    parser.add_argument('--save-fig', type=str, default='',
                        help='Save figure to file (e.g. simple_marshak.pdf)')
    parser.add_argument('--plot-times', type=float, nargs='+',
                        default=[1.0, 5.0, 10.0],
                        help='Times to plot in ns (default: 1 5 10)')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.zones, order=args.order, N=args.N, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max, K=args.K,
        maxits=args.maxits, LOUD=args.loud)

    # save results
    ac = sn_solver.ac
    np.savez_compressed(
        args.save_npz,
        times_ns=results['ts'],
        Tr_keV=np.array([(p / ac)**0.25 for p in results['phis']]),
        T_keV=np.array(results['Ts']))
    print(f"Results saved to {args.save_npz}")

    if args.plot or args.save_fig:
        plot_results(results, plot_times_ns=tuple(args.plot_times),
                     savefile=args.save_fig)


if __name__ == '__main__':
    main()
