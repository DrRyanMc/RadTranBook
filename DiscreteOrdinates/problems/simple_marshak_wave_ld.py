"""
Simple Marshak wave — Linear Discontinuous S_N solver.

Mirrors the IMC/MarshakWave.py problem setup:

  - sigma_a = 300 T^{-3}  (pure absorption, no scattering)
  - Linear EOS: e = cv * T,  cv = 0.3
  - Left wall: blackbody at T_bc = 1.0 keV
  - Right wall: reflecting
  - Cold start: T_init = 1e-4 keV

The self-similar (Su-Olson-like) solution is used for comparison.

Run from the DiscreteOrdinates directory:
    python problems/simple_marshak_wave_ld.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver_ld
from sn_solver_ld import ac, a, c


def setup_and_run(I=50, N=8, L=0.20, tfinal=10.0,
                  dt_min=1e-4, dt_max=0.1,
                  K=200, R=3, maxits=500, tolerance=1e-8, LOUD=0,
                  time_outputs=None, use_dmd=True, W=0,
                  tau_phi_max=0.1, tau_T=1e-6, C_T=1.0, omega_T=1.0):
    """Set up and run the simple Marshak wave with the LD S_N solver.

    Parameters
    ----------
    I : int
        Number of spatial cells.
    N : int
        Number of discrete ordinates.
    L : float
        Domain length (cm).
    tfinal : float
        Final simulation time (ns).
    dt_min, dt_max : float
        Adaptive time-step bounds (ns).
    K : int
        DMD inner iterations.
    maxits : int
        Max DMD solver iterations per time step.
    LOUD : int
        Verbosity level.
    time_outputs : array-like or None
        Extra times (ns) to hit exactly for output snapshots.

    Returns
    -------
    results : dict
        Keys: phis, Ts, iterations, ts, x, x_2d, hx, L
    """
    hx  = L / I
    # cell-centre positions
    x   = np.linspace(hx / 2, L - hx / 2, I)
    # left- and right-edge positions for each cell (LD DoFs)
    x_l = x - hx / 2
    x_r = x + hx / 2
    x_2d = np.stack([x_l, x_r], axis=1)   # (I, 2)

    # ── material parameters ───────────────────────────────────────────────────
    T_bc   = 1.0      # left-wall temperature (keV)
    Tinit  = 1e-4     # initial temperature (keV)
    cv_val = 0.3      # heat capacity

    # ── EOS: e = cv * T  (linear, node-wise) ─────────────────────────────────
    def eos(T):
        return cv_val * T

    def invEOS(e):
        return e / cv_val

    # ── opacities ─────────────────────────────────────────────────────────────
    def sigma_func(T):
        return 300.0 * T**(-3)

    def scat_func(T):
        return np.zeros_like(T)

    # ── initial conditions ────────────────────────────────────────────────────
    T   = np.full((I, 2), Tinit)
    phi = ac * T**4
    psi = np.broadcast_to(phi[:, None, :], (I, N, 2)).copy()

    # ── quadrature (needed only to build BCs) ─────────────────────────────────
    MU, _ = np.polynomial.legendre.leggauss(N)

    # ── boundary conditions ───────────────────────────────────────────────────
    # Left wall: Mark BC — incoming angular intensity = full isotropic blackbody
    #   value ac*T_bc^4 (weights in this solver are normalised to sum=1, so
    #   the isotropic ψ equals the scalar flux φ = ac*T^4).
    # Right wall: reflecting (handled by reflect_right=True in temp_solve_ld;
    #             the BCs[n,0] entries for mu < 0 will be overwritten internally)
    I_bc = ac * T_bc**4

    def BCFunc(t):
        out = np.zeros((N, 2))
        for n in range(N):
            if MU[n] > 0.0:
                out[n, 1] = I_bc   # left-boundary inflow (mu > 0)
            # mu < 0 entries left 0; overridden by reflecting BC machinery
        return out

    # ── output times ──────────────────────────────────────────────────────────
    if time_outputs is None:
        time_outputs = np.array([1.0, 5.0, 10.0])
    else:
        time_outputs = np.asarray(time_outputs)

    # ── fixed external source (zero) ─────────────────────────────────────────
    q = np.zeros((I, N, 2))

    # ── run ───────────────────────────────────────────────────────────────────
    print(f"Running simple LD Marshak wave: I={I}, N={N}, tfinal={tfinal} ns"
          f"  {'(DMD)' if use_dmd else '(Richardson only)'}"
          f"{f'  W={W}' if W > 0 else ''}")
    phis, Ts, iterations, ts, its_per_step = sn_solver_ld.temp_solve_ld(
        I, hx, q, sigma_func, scat_func, N, BCFunc, eos, invEOS,
        phi, psi, T,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        LOUD=LOUD, maxits=maxits, fix=1, K=K, R=R,
        tolerance=tolerance,
        time_outputs=time_outputs,
        reflect_left=False, reflect_right=True,
        print_stride=10,
        use_dmd=use_dmd,
        W=W,
        tau_phi_max=tau_phi_max,
        tau_T=tau_T,
        C_T=C_T,
        omega_T=omega_T,
    )
    print(f"\nDone. Total transport sweeps: {iterations}")

    return {
        'phis': phis, 'Ts': Ts, 'iterations': iterations,
        'its_per_step': its_per_step,
        'ts': ts, 'x': x, 'x_2d': x_2d, 'hx': hx, 'L': L,
        'T_bc': T_bc, 'cv_val': cv_val,
    }


# ── self-similar reference solution ──────────────────────────────────────────

def self_similar_solution(t, results):
    """Return (r, T_ss) for the self-similar Marshak wave at time *t* (ns).

    Uses the same parameters as IMC/MarshakWave.py.
    """
    T_bc   = results['T_bc']
    cv_val = results['cv_val']
    sigma_0 = 300.0 * T_bc**(-3)    # sigma_a at T_bc
    rho   = 1.0
    xi_max = 1.11305
    omega  = 0.05989
    K_const = 8 * a * c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)

    xi_vals = np.linspace(0, xi_max, 500)
    r_ss = xi_vals * np.sqrt(K_const * t)
    f_ss = (xi_vals < xi_max) * np.power(
        np.where(xi_vals < xi_max,
                 (1 - xi_vals / xi_max) * (1 + omega * xi_vals / xi_max),
                 1e-30),
        1.0 / 6.0)
    return r_ss, T_bc * f_ss


# ── plotting helpers ──────────────────────────────────────────────────────────

def _ld_cell_centres(vals, x):
    """Cell-centre averages from LD left/right edge values."""
    return 0.5 * (vals[:, 0] + vals[:, 1])


def plot_results(results, plot_times=(1.0, 5.0, 10.0), savefile=''):
    """Plot material and radiation temperatures with self-similar comparison.

    Parameters
    ----------
    results : dict
        Output of ``setup_and_run``.
    plot_times : tuple of float
        Times (ns) to include in the plot.
    savefile : str
        If non-empty, save figure here instead of displaying it.
    """
    phis = results['phis']
    Ts   = results['Ts']
    ts   = results['ts']
    x    = results['x']
    L    = results['L']

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, tval in enumerate(plot_times):
        tstep = int(np.argmin(np.abs(tval - ts)))
        tact  = float(ts[tstep])
        color = f'C{i}'

        T_mat = _ld_cell_centres(Ts[tstep], x)
        T_rad = _ld_cell_centres((phis[tstep] / ac)**0.25, x)

        ax.plot(x, T_mat, '-',  color=color,
                label=f'Material $T$ ($t={tact:.1f}$ ns)')
        ax.plot(x, T_rad, '--', color=color,
                label=f'Radiation $T_r$ ($t={tact:.1f}$ ns)')

        r_ss, T_ss = self_similar_solution(tact, results)
        ax.plot(r_ss, T_ss, ':', color=color, linewidth=1.5,
                label=f'Self-similar ($t={tact:.1f}$ ns)')

    ax.set_xlim(0, L)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Simple Marshak Wave — LD $S_N$')
    ax.legend(fontsize=7)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefile}")
    else:
        plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Simple Marshak wave — LD S_N solver')
    parser.add_argument('--zones',    type=int,   default=50,
                        help='Number of spatial cells (default: 50)')
    parser.add_argument('--N',        type=int,   default=8,
                        help='Number of discrete ordinates (default: 8)')
    parser.add_argument('--L',        type=float, default=0.20,
                        help='Domain length in cm (default: 0.20)')
    parser.add_argument('--tfinal',   type=float, default=10.0,
                        help='Final time in ns (default: 10.0)')
    parser.add_argument('--dt-min',   type=float, default=0.01,
                        help='Minimum time step in ns (default: 1e-4)')
    parser.add_argument('--dt-max',   type=float, default=0.01,
                        help='Maximum time step in ns (default: 0.1)')
    parser.add_argument('--K',        type=int,   default=200,
                        help='DMD snapshot count (default: 200)')
    parser.add_argument('--R',        type=int,   default=3,
                        help='Richardson steps between DMD updates (default: 3)')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='Inner iteration convergence tolerance (default: 1e-8)')
    parser.add_argument('--maxits',   type=int,   default=500,
                        help='Max iterations per time step (default: 500)')
    parser.add_argument('--loud',     type=int,   default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--save-npz', type=str,   default='simple_marshak_ld.npz',
                        help='Output .npz filename')
    parser.add_argument('--no-dmd',   action='store_true',
                        help='Disable DMD acceleration (use pure Richardson iteration)')
    parser.add_argument('--W',        type=int,   default=0,
                        help='Max T_star outer iterations: 0=single linearisation '
                             '(default), >0=update T_star up to W times per time step')
    parser.add_argument('--tau-phi-max', type=float, default=0.1,
                        help='Loose inner transport tolerance for k=0 outer iter '
                             '(default: 0.1)')
    parser.add_argument('--tau-T',    type=float, default=1e-6,
                        help='Outer T_star convergence tolerance (default: 1e-6)')
    parser.add_argument('--C-T',      type=float, default=1.0,
                        help='Coupling constant: tau_phi = C_T * eta_T (default: 1.0)')
    parser.add_argument('--omega-T',  type=float, default=1.0,
                        help='T_star damping factor: 1=undamped (default: 1.0)')
    parser.add_argument('--no-plot',  action='store_true',
                        help='Skip showing the plot')
    parser.add_argument('--save-fig', type=str,   default='simple_marshak_ld.pdf',
                        help='Save figure to file (default: simple_marshak_ld.pdf)')
    args = parser.parse_args()

    output_times = np.array([1.0, 5.0, args.tfinal])
    results = setup_and_run(
        I=args.zones, N=args.N, L=args.L, tfinal=args.tfinal,
        dt_min=args.dt_min, dt_max=args.dt_max,
        K=args.K, R=args.R, maxits=args.maxits,
        tolerance=args.tolerance, LOUD=args.loud,
        time_outputs=output_times, use_dmd=not args.no_dmd,
        W=args.W,
        tau_phi_max=args.tau_phi_max, tau_T=args.tau_T,
        C_T=args.C_T, omega_T=args.omega_T)

    ts_arr    = results['ts']
    T_arr     = np.array([_ld_cell_centres(T, results['x']) for T in results['Ts']])
    Tr_arr    = np.array([_ld_cell_centres((p / ac)**0.25, results['x'])
                          for p in results['phis']])
    np.savez_compressed(
        args.save_npz,
        times_ns=ts_arr,
        T_keV=T_arr,
        Tr_keV=Tr_arr,
        x=results['x'])
    print(f"Results saved to {args.save_npz}")

    if not args.no_plot:
        plot_results(results, savefile=args.save_fig)


if __name__ == '__main__':
    main()
