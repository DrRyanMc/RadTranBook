"""
Inhomogeneous Marshak wave problem (Test 1).

Solves a thermal radiation Marshak wave with spatially varying material
properties (power-law density profile) using the 1-D discrete ordinates
solver with DMD-accelerated source iteration.

Run from the DiscreteOrdinates directory:
    python problems/marshak_wave.py
"""

import sys
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from numba import jit, float64

# Add parent directory so we can import the solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver


def setup_and_run(I=200, order=3, N=8, tfinal=1.0,
                  dt_min=1e-5, dt_max=0.01, K=800, maxits=2000,
                  LOUD=0):
    """Set up and run the inhomogeneous Marshak wave (Test 1).

    Parameters
    ----------
    I : int
        Number of spatial zones.
    order : int
        Bernstein polynomial order.
    N : int
        Number of discrete ordinates (S_N order).
    tfinal : float
        Final time (ns).
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
    Lx = 1.0
    hx = Lx / I
    x = np.linspace(hx / 2, Lx - hx / 2, I)

    # --- material parameters ---
    alpha = 1.5      # absorption opacity temperature exponent
    beta = 3.4       # EOS temperature exponent
    u0 = 0.01        # EOS coefficient
    k0 = 0.1         # absorption opacity coefficient
    ks = 40 - k0     # scattering opacity coefficient
    mu_mat = 0.14    # EOS density exponent
    omega = -20 / 19 # density profile exponent: rho(x) = x^(-omega)
    lam = 0.2        # scattering density exponent
    lamp = 0.2       # absorption density exponent
    Tinit = 1e-3     # initial temperature (keV)

    # --- quadrature ---
    MU, W = np.polynomial.legendre.leggauss(N)
    ac = sn_solver.ac

    # --- external source (zero) ---
    q = np.zeros((I, N, order + 1))

    # --- EOS: e = u0 * T^beta * rho^(1 - mu_mat) ---
    @jit(float64[:, :](float64[:, :], float64[:], float64, float64,
                        float64, float64), nopython=True)
    def eos_full(T, x, u0, beta, mu_mat, omega):
        tmp_rho = np.zeros(T.shape)
        for col in range(T.shape[1]):
            tmp_rho[:, col] = x**(-omega)
        return u0 * T**beta * tmp_rho**(1 - mu_mat)

    eos = lambda T: eos_full(T, x, u0, beta, mu_mat, omega)

    @jit(float64[:, :](float64[:, :], float64[:], float64, float64,
                        float64, float64), nopython=True)
    def invEOS_full(E, x, u0, beta, mu_mat, omega):
        tmp_rho = np.zeros(E.shape)
        for col in range(E.shape[1]):
            tmp_rho[:, col] = x**(-omega)
        return (E / u0 / (tmp_rho**(1 - mu_mat)))**(1 / beta)

    invEOS = lambda E: invEOS_full(E, x, u0, beta, mu_mat, omega)

    # --- absorption opacity: sigma_a = k0 * T^(-alpha) * rho^(lamp+1) ---
    @jit(float64[:, :](float64[:, :], float64[:], float64, float64,
                        float64, float64), nopython=True)
    def sigma_func_nb(T, x, k0, alpha, lamp, omega):
        tmp_rho = np.zeros(T.shape)
        for col in range(T.shape[1]):
            tmp_rho[:, col] = x**(-omega)
        return k0 * np.abs(T)**(-alpha) * tmp_rho**(lamp + 1)

    sigma_func = lambda T: sigma_func_nb(T, x, k0, alpha, lamp, omega)

    # --- scattering cross-section ---
    @jit(float64[:, :](float64[:, :], float64[:], float64, float64,
                        float64, float64), nopython=True)
    def scat_func_nb(T, x, ks, alpha, lam, omega):
        tmp_rho = np.zeros(T.shape)
        for col in range(T.shape[1]):
            tmp_rho[:, col] = x**(-omega)
        return ks * np.abs(T)**(-alpha) * tmp_rho**(lam + 1)

    scat_func = lambda T: scat_func_nb(T, x, ks, alpha, lam, omega)

    # --- initial conditions ---
    T = np.ones((I, order + 1)) * Tinit
    phi = ac * T**4
    psi = np.zeros((I, N, order + 1)) + phi[:, None, :]

    # --- boundary conditions ---
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        out = np.zeros((N, order + 1))
        tau = 86 / 57
        out[MU > 0, :] = ac * (1.0470478 * t**tau)**4
        out[MU < 0, :] = ac * Tinit**4
        return out

    # --- run ---
    print(f"Running Marshak wave: I={I}, order={order}, N={N}, tfinal={tfinal}")
    phis, Ts, iterations, ts = sn_solver.temp_solve_dmd_inc(
        I, hx, q, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        order=order, LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3)
    print(f"\nDone. Total transport sweeps: {iterations}")

    results = {
        'phis': phis, 'Ts': Ts, 'iterations': iterations, 'ts': ts,
        'x': x, 'hx': hx, 'Lx': Lx, 'order': order
    }
    return results


def semi_analytic_Tr(x, t):
    """Semi-analytic radiation temperature profile for Test 1.

    Parameters
    ----------
    x : ndarray
        Spatial positions (cm).
    t : float
        Time in nanoseconds.
    """
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    f = 1 - 0.75567 * (xi / xi0)**2.0416
    f[xi > xi0 * 0.75] = 1.2527 * (1 - xi[xi > xi0 * 0.75] / xi0)**0.55623
    return t**(86 / 57) * f


def semi_analytic_T(x, t):
    """Semi-analytic material temperature profile for Test 1.

    Parameters
    ----------
    x : ndarray
        Spatial positions (cm).
    t : float
        Time in nanoseconds.
    """
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    g = -0.15937 * (xi / xi0)**0.2194 + (xi / xi0)**0.074842
    g[xi > xi0 * 0.05] = (
        0.63674 + 0.55611 * (xi[xi > xi0 * 0.05] / xi0)**0.56101
    ) * (1 - xi[xi > xi0 * 0.05] / xi0)**0.63964
    return t**(86 / 57) * g


def plot_results(results, plot_times=(0.2, 0.6, 1.0), savefile=''):
    """Plot temperature profiles and compare with semi-analytic solutions.

    Parameters
    ----------
    plot_times : tuple of float
        Times to plot, in nanoseconds.
    """
    phis = results['phis']
    Ts = results['Ts']
    ts = results['ts']
    Lx = results['Lx']
    order = results['order']
    I = len(Ts[0])
    ac = sn_solver.ac

    xplot = np.linspace(0, Lx, 1000)
    edges = np.linspace(0, Lx, I + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for tval in plot_times:
        tstep = np.argmin(np.abs(tval - ts))
        tval_actual = ts[tstep]

        # material temperature from Bernstein interpolation
        T_sol = Ts[tstep]
        T_plot = BPoly(T_sol.transpose(), edges)(xplot)

        # radiation temperature
        Tr_sol = (phis[tstep] / ac)**0.25
        Tr_plot = BPoly(Tr_sol.transpose(), edges)(xplot)

        label_t = f't = {tval_actual:.2f} ns'
        p = ax.plot(xplot, T_plot, '-', label=f'$T$ {label_t}')
        color = p[0].get_color()
        ax.plot(xplot, Tr_plot, '--', color=color, label=f'$T_r$ {label_t}')

        # semi-analytic
        ax.plot(xplot, semi_analytic_T(xplot, tval_actual), ':',
                color=color, alpha=0.6)
        ax.plot(xplot, semi_analytic_Tr(xplot, tval_actual), ':',
                color=color, alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Inhomogeneous Marshak Wave (Test 1)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, Lx)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Run inhomogeneous Marshak wave (Test 1)')
    parser.add_argument('--zones', type=int, default=200,
                        help='Number of spatial zones (default: 200)')
    parser.add_argument('--order', type=int, default=3,
                        help='Bernstein polynomial order (default: 3)')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--tfinal', type=float, default=1.0,
                        help='Final time in ns (default: 1.0)')
    parser.add_argument('--dt-min', type=float, default=1e-5,
                        help='Minimum time step in ns (default: 1e-5)')
    parser.add_argument('--dt-max', type=float, default=0.01,
                        help='Maximum time step in ns (default: 0.01)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per step (default: 2000)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--save-npz', type=str, default='marshak_wave_test1.npz',
                        help='Output .npz file name')
    parser.add_argument('--plot', action='store_true',
                        help='Show plots after running')
    parser.add_argument('--save-fig', type=str, default='sn_marshak_test.pdf',
                        help='Save figure to file (e.g. marshak_wave.pdf)')
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
        Tr_keV=(np.array(results['phis']) / ac)**0.25,
        T_keV=np.array(results['Ts']))
    print(f"Results saved to {args.save_npz}")

    if args.plot or args.save_fig:
        plot_results(results, savefile=args.save_fig)


if __name__ == '__main__':
    main()
