import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
0-D Equilibration Test — 2-D S_N solver.

Radiation and material come into equilibrium in a single cell with
reflecting boundaries on all sides (no leakage). This is the simplest
possible test of the radiation-material coupling.

Problem setup:
  - Single cell (Ix=1, Iy=1) with all reflecting BCs
  - σ_a = 1.0 cm⁻¹ (constant)
  - c_v = 0.01 GJ/(cm³·keV), linear EOS: e = c_v * T
  - T_init = 0.4 keV, T_r_init = 1.0 keV
  - No external source, no scattering
  - Radiation and material exchange energy until equilibrium
  - Compare against ODE solution (exact)

The ODE system is:
  d(aT_r⁴)/dt = -σ c a (T_r⁴ - T⁴)
  d(c_v T)/dt  =  σ c a (T_r⁴ - T⁴)

Run from the DiscreteOrdinates2D directory:
    python problems/test_equilibration.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ===========================================================================
# Problem parameters (matching IMC 0-D test)
# ===========================================================================
SIGMA_A = 1.0       # absorption opacity (cm⁻¹)
CV_VOL = 0.01       # volumetric heat capacity (GJ/(cm³·keV))
T_INIT = 0.4        # initial material temperature (keV)
TR_INIT = 1.0       # initial radiation temperature (keV)
L = 0.1             # cell size (cm) — arbitrary for 0-D
FINAL_TIME = 0.375  # ns


def run_equilibration(dt_values=(0.025,), N_quad=4, quad_type='level_symmetric'):
    """Run the 0-D equilibration test.

    Parameters
    ----------
    dt_values : tuple of float
        Time step sizes to test (ns).
    """
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    # --- Exact ODE solution ---
    def RHS(t, y):
        # y[0] = radiation energy density = a * Tr^4
        # y[1] = material energy density = cv * T
        Er = y[0]
        e_mat = y[1]
        Tr = (Er / a)**0.25
        T = e_mat / CV_VOL
        # Exchange: dEr/dt = -sigma*c*(Er - a*T^4) = -sigma*c*a*(Tr^4 - T^4)
        exchange = SIGMA_A * c * a * (Tr**4 - T**4)
        return [-exchange, exchange]

    Er_init = a * TR_INIT**4
    e_init = CV_VOL * T_INIT
    t_eval = np.linspace(0, FINAL_TIME, 1000)
    sol = solve_ivp(RHS, [0, FINAL_TIME], [Er_init, e_init],
                    t_eval=t_eval, rtol=1e-10, atol=1e-14)
    T_exact = sol.y[1] / CV_VOL
    Tr_exact = (sol.y[0] / a)**0.25

    print("="*60)
    print("  0-D Equilibration Test")
    print("="*60)
    print(f"  σ_a = {SIGMA_A}, c_v = {CV_VOL}, L = {L}")
    print(f"  T_init = {T_INIT} keV, Tr_init = {TR_INIT} keV")
    print(f"  Equilibrium T_eq = {((a*TR_INIT**4 + CV_VOL*T_INIT)/(a*1 + CV_VOL))**0.25:.4f} keV (approx)")
    print(f"  Final time = {FINAL_TIME} ns")

    # --- S_N solutions for each dt ---
    sn_results = {}
    for dt in dt_values:
        print(f"\n  Running with dt = {dt} ns...")
        Ix, Iy = 1, 1
        dx_arr = np.full(Ix, L)
        dy_arr = np.full(Iy, L)

        def EOS(T):
            return CV_VOL * T

        def invEOS(e):
            return e / CV_VOL

        def sigma_func(T):
            return np.full_like(T, SIGMA_A)

        def scat_func(T):
            return np.zeros_like(T)

        # Initial conditions
        T_init_arr = np.full((Ix, Iy, 4), T_INIT)
        phi_init_arr = np.full((Ix, Iy, 4), ac * TR_INIT**4)
        q_ext = np.zeros((Ix, Iy, 4))

        # All reflecting (no leakage)
        def BCs_func(t):
            return {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None}

        phis, Ts, iterations, ts, its_per = temp_solve_2d(
            Ix, Iy, dx_arr, dy_arr,
            q_ext, sigma_func, scat_func,
            quad_type, N_quad,
            BCs_func, EOS, invEOS,
            phi_init_arr, T_init_arr,
            dt_min=dt, dt_max=dt, tfinal=FINAL_TIME,
            tolerance=1e-12, Linf_tol=1e-10, maxits=500,
            K=20, R=3,
            reflect_xlo=True, reflect_xhi=True,
            reflect_ylo=True, reflect_yhi=True,
            use_dmd=False,
            print_stride=0,
        )

        # Extract time history
        ts_arr = np.array(ts)
        T_hist = np.array([np.mean(T_snap) for T_snap in Ts])
        Tr_hist = np.array([(np.mean(phi_snap) / ac)**0.25 for phi_snap in phis])

        sn_results[dt] = {
            'ts': ts_arr, 'T': T_hist, 'Tr': Tr_hist,
            'iterations': iterations,
        }
        print(f"    Steps: {len(ts)-1}, sweeps: {iterations}")
        print(f"    Final T = {T_hist[-1]:.6f}, Tr = {Tr_hist[-1]:.6f}")

    return {
        'exact': {'t': t_eval, 'T': T_exact, 'Tr': Tr_exact},
        'sn': sn_results,
    }


def plot_results(results, savefile='equilibration_0d.png'):
    """Plot equilibration results."""
    exact = results['exact']
    sn = results['sn']

    fig, ax = plt.subplots(figsize=(7, 5))

    # Exact solution
    ax.plot(exact['t'], exact['T'], '-', color='black', alpha=0.5, lw=2,
            label='Exact T')
    ax.plot(exact['t'], exact['Tr'], '--', color='black', alpha=0.5, lw=2,
            label='Exact $T_r$')

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(sn)))
    markers = ['^', 'o', 's', 'D']

    for idx, (dt, data) in enumerate(sorted(sn.items())):
        col = colors[idx]
        mk = markers[idx % len(markers)]
        ax.plot(data['ts'], data['T'], color=col, marker=mk, ms=6,
                markerfacecolor='white', markeredgewidth=0.8,
                lw=1, alpha=0.8,
                label=f'S$_N$ T, dt={dt}')
        ax.plot(data['ts'], data['Tr'], '--', color=col, marker=mk, ms=6,
                markerfacecolor='white', markeredgewidth=0.8,
                lw=1, alpha=0.8,
                label=f'S$_N$ $T_r$, dt={dt}')

    ax.set_xlabel('t (ns)')
    ax.set_ylabel('T (keV)')
    ax.set_title('0-D Equilibration: Radiation ↔ Material')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, FINAL_TIME)
    ax.set_ylim(T_INIT - 0.05, TR_INIT + 0.05)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {savefile}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='0-D equilibration test')
    parser.add_argument('--dt', type=float, nargs='+', default=[0.025, 0.01],
                        help='Time step sizes (ns)')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order')
    args = parser.parse_args()

    results = run_equilibration(dt_values=tuple(args.dt), N_quad=args.N)
    plot_results(results)
