"""
Inhomogeneous Marshak wave problem (Test 1) — Linear Discontinuous S_N solver.

Solves the same thermal radiation Marshak wave as marshak_wave.py but uses
the sn_solver_ld Linear Discontinuous (LD) Galerkin spatial discretisation
with the conservative negative-intensity fixup and DMD-accelerated source
iteration described in Chapter 11 of the textbook.

Array shapes in this file
-------------------------
- phi, T, e, sigma : (I, 2)  — index 0 = left cell edge, 1 = right cell edge
- psi, source      : (I, N, 2)
- BCs              : (N, 2)  — BCs[n,1] left-boundary inflow (mu>0),
                               BCs[n,0] right-boundary inflow (mu<0)

Run from the DiscreteOrdinates directory:
    python problems/marshak_wave_ld.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, float64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver_ld


def setup_and_run(I=200, N=8, tfinal=1.0,
                  dt_min=1e-5, dt_max=0.01, K=800, maxits=2000,
                  LOUD=0):
    """Set up and run the inhomogeneous Marshak wave with the LD S_N solver.

    Parameters
    ----------
    I : int
        Number of spatial cells.
    N : int
        Number of discrete ordinates (S_N order).
    tfinal : float
        Final time (ns).
    dt_min, dt_max : float
        Adaptive time-step bounds (ns).
    K : int
        Number of DMD inner iterations.
    maxits : int
        Maximum DMD solver iterations per time step.
    LOUD : int
        Verbosity level.

    Returns
    -------
    results : dict
        Keys: phis, Ts, iterations, ts, x, x_2d, hx, Lx
    """
    # ── geometry ──────────────────────────────────────────────────────────────
    Lx = 1.0
    hx = Lx / I
    # Cell-centre coordinates (for reference and plotting)
    x = np.linspace(hx / 2, Lx - hx / 2, I)

    # Left- and right-edge positions for each cell (LD degrees of freedom).
    # Floor the left edge of cell 0 at a small positive value to avoid
    # x=0 in the power-law density profile.
    x_l = np.maximum(x - hx / 2, 1e-12)   # left  edges: (I,)
    x_r = x + hx / 2                        # right edges: (I,)
    x_2d = np.stack([x_l, x_r], axis=1)    # (I, 2) — used by material functions

    # ── material parameters ───────────────────────────────────────────────────
    alpha  = 1.5         # absorption opacity temperature exponent
    beta   = 3.4         # EOS temperature exponent
    u0     = 0.01        # EOS coefficient
    k0     = 0.1         # absorption opacity coefficient
    ks     = 40 - k0     # scattering opacity coefficient
    mu_mat = 0.14        # EOS density exponent
    omega  = -20 / 19    # density-profile exponent: rho(x) = x^(-omega)
    lam    = 0.2         # scattering density exponent
    lamp   = 0.2         # absorption density exponent
    Tinit  = 1e-3        # initial temperature (keV)

    # ── quadrature ────────────────────────────────────────────────────────────
    MU, W = np.polynomial.legendre.leggauss(N)
    ac = sn_solver_ld.ac

    # ── fixed external source (zero) ──────────────────────────────────────────
    q = np.zeros((I, N, 2))

    # ── EOS: e = u0 * T^beta * rho^(1-mu_mat), rho(x)=x^(-omega) ────────────
    @jit(float64[:, :](float64[:, :]), nopython=True)
    def eos(T):
        rho = x_2d ** (-omega)          # (I, 2)
        return u0 * T**beta * rho**(1.0 - mu_mat)

    @jit(float64[:, :](float64[:, :]), nopython=True)
    def invEOS(E):
        rho = x_2d ** (-omega)
        return (E / u0 / rho**(1.0 - mu_mat))**(1.0 / beta)

    # ── absorption opacity: sigma_a = k0 * T^(-alpha) * rho^(lamp+1) ─────────
    @jit(float64[:, :](float64[:, :]), nopython=True)
    def sigma_func(T):
        rho = x_2d ** (-omega)
        return k0 * T**(-alpha) * rho**(lamp + 1.0)

    # ── scattering: sigma_s = ks * T^(-alpha) * rho^(lam+1) ──────────────────
    @jit(float64[:, :](float64[:, :]), nopython=True)
    def scat_func(T):
        rho = x_2d ** (-omega)
        return ks * T**(-alpha) * rho**(lam + 1.0)

    # ── initial conditions ────────────────────────────────────────────────────
    T   = np.full((I, 2), Tinit)
    phi = ac * T**4
    psi = np.broadcast_to(phi[:, None, :], (I, N, 2)).copy()

    # ── boundary conditions ───────────────────────────────────────────────────
    # Left wall: incoming blackbody at T_b(t) = 1.0470478 * t^(86/57) keV.
    # Right wall: cold incoming radiation at Tinit.
    # BCs[n, 1] = left-wall inflow  (used when MU[n] > 0)
    # BCs[n, 0] = right-wall inflow (used when MU[n] < 0)
    @jit(float64[:, :](float64), nopython=True)
    def BCFunc(t):
        out = np.zeros((N, 2))
        tau = 86.0 / 57.0
        I_hot  = ac * (1.0470478 * t**tau)**4
        I_cold = ac * Tinit**4
        for n in range(N):
            if MU[n] > 0.0:
                out[n, 1] = I_hot    # left-boundary inflow
            else:
                out[n, 0] = I_cold   # right-boundary inflow
        return out

    # ── run ───────────────────────────────────────────────────────────────────
    print(f"Running LD Marshak wave: I={I}, N={N}, tfinal={tfinal}")
    phis, Ts, iterations, ts = sn_solver_ld.temp_solve_ld(
        I, hx, q, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        LOUD=LOUD, maxits=maxits, fix=1, K=K, R=3)

    print(f"\nDone. Total transport sweeps: {iterations}")

    return {
        'phis': phis, 'Ts': Ts, 'iterations': iterations, 'ts': ts,
        'x': x, 'x_2d': x_2d, 'hx': hx, 'Lx': Lx,
    }


# ── semi-analytic reference (unchanged from marshak_wave.py) ─────────────────

def semi_analytic_Tr(x, t):
    """Semi-analytic radiation temperature profile for Test 1."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    f = 1 - 0.75567 * (xi / xi0)**2.0416
    f[xi > xi0 * 0.75] = (
        1.2527 * (1 - xi[xi > xi0 * 0.75] / xi0)**0.55623)
    return t**(86 / 57) * f


def semi_analytic_T(x, t):
    """Semi-analytic material temperature profile for Test 1."""
    xheat = 0.8332614 * t
    xi0 = 1.2746051
    xi = xi0 * x / xheat
    g = -0.15937 * (xi / xi0)**0.2194 + (xi / xi0)**0.074842
    g[xi > xi0 * 0.05] = (
        0.63674 + 0.55611 * (xi[xi > xi0 * 0.05] / xi0)**0.56101
    ) * (1 - xi[xi > xi0 * 0.05] / xi0)**0.63964
    return t**(86 / 57) * g


# ── plotting ──────────────────────────────────────────────────────────────────

def _ld_edges_for_plot(vals, x_2d):
    """Interleave left and right edge values for a piecewise-linear plot.

    Returns (z_plot, v_plot) where adjacent cells share the interface
    position, revealing any inter-cell discontinuity.
    """
    I = vals.shape[0]
    z_plot = np.empty(2 * I)
    v_plot = np.empty(2 * I)
    z_plot[0::2] = x_2d[:, 0]
    z_plot[1::2] = x_2d[:, 1]
    v_plot[0::2] = vals[:, 0]
    v_plot[1::2] = vals[:, 1]
    return z_plot, v_plot


def plot_results(results, plot_times=(0.2, 0.6, 1.0), savefile=''):
    """Plot LD temperature profiles and compare with semi-analytic solutions.

    Parameters
    ----------
    results : dict
        Output of ``setup_and_run``.
    plot_times : tuple of float
        Times (ns) to include in the plot.
    savefile : str
        If non-empty, save the figure to this path instead of showing it.
    """
    phis  = results['phis']
    Ts    = results['Ts']
    ts    = results['ts']
    Lx    = results['Lx']
    x_2d  = results['x_2d']
    ac    = sn_solver_ld.ac

    xref = np.linspace(0, Lx, 2000)

    fig, ax = plt.subplots(figsize=(8, 5))

    for tval in plot_times:
        tstep = int(np.argmin(np.abs(tval - ts)))
        tact  = float(ts[tstep])

        # LD piecewise-linear profiles
        z_T,  v_T  = _ld_edges_for_plot(Ts[tstep],                     x_2d)
        z_Tr, v_Tr = _ld_edges_for_plot((phis[tstep] / ac)**0.25, x_2d)

        label = f't = {tact:.2f} ns'
        p  = ax.plot(z_T,  v_T,  '-',  label=f'$T$ {label}')
        col = p[0].get_color()
        ax.plot(z_Tr, v_Tr, '--', color=col, label=f'$T_r$ {label}')

        # semi-analytic
        ax.plot(xref, semi_analytic_T(xref.copy(),  tact), ':',
                color=col, alpha=0.6)
        ax.plot(xref, semi_analytic_Tr(xref.copy(), tact), ':',
                color=col, alpha=0.6)

    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Inhomogeneous Marshak Wave — LD $S_N$')
    ax.legend(fontsize=8)
    ax.set_xlim(0, Lx)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    else:
        plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Inhomogeneous Marshak wave — LD S_N solver')
    parser.add_argument('--zones',   type=int,   default=200,
                        help='Number of spatial cells (default: 200)')
    parser.add_argument('--N',       type=int,   default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--tfinal',  type=float, default=1.0,
                        help='Final time in ns (default: 1.0)')
    parser.add_argument('--dt-min',  type=float, default=1e-5,
                        help='Minimum time step in ns (default: 1e-5)')
    parser.add_argument('--dt-max',  type=float, default=0.01,
                        help='Maximum time step in ns (default: 0.01)')
    parser.add_argument('--K',       type=int,   default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits',  type=int,   default=2000,
                        help='Max iterations per step (default: 2000)')
    parser.add_argument('--loud',    type=int,   default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--save-npz', type=str,  default='marshak_wave_ld.npz',
                        help='Output .npz filename')
    parser.add_argument('--plot',    action='store_true',
                        help='Show plot after running')
    parser.add_argument('--save-fig', type=str,  default='',
                        help='Save figure to this file (e.g. marshak_ld.pdf)')
    args = parser.parse_args()

    results = setup_and_run(
        I=args.zones, N=args.N, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, maxits=args.maxits, LOUD=args.loud)

    ac = sn_solver_ld.ac
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
