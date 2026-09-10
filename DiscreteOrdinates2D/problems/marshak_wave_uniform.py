import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Marshak Wave with uniform material — 2-D S_N solver.

Classic radiative heat wave driven by a blackbody boundary at T_bc = 1 keV
into a cold uniform medium.

Problem setup:
  - Left boundary (x=0): incoming blackbody at T_bc = 1 keV
  - Material opacity: σ = 300 * T^{-3} cm^{-1} (T in keV)
  - Heat capacity: c_v = 0.3 GJ/(cm³·keV)  → e = c_v * T (linear EOS)
  - No scattering
  - Right boundary: vacuum
  - y boundaries: reflecting (effectively 1-D)
  - Output at: 1, 10, 20 ns
  - Compare with self-similar solution

Run from the DiscreteOrdinates2D directory:
    python problems/marshak_wave_uniform.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature


# ===========================================================================
# Physical constants and material parameters
# ===========================================================================
A_RAD = a           # radiation constant (GJ/(cm³ keV⁴))
C_LIGHT = c         # speed of light (cm/ns)
RHO = 1.0           # density (g/cm³)
CV_VOL = 0.3        # volumetric heat capacity (GJ/(cm³ keV))
SIGMA0 = 300.0      # opacity coefficient
N_OPACITY = 3       # opacity temperature exponent
T_BC = 1.0          # boundary temperature (keV)
T_INIT = 0.01       # initial temperature (keV)


# ===========================================================================
# Self-similar solution
# ===========================================================================

def self_similar_solution(x, t):
    """Self-similar Marshak wave solution for σ = σ₀ T^{-3}, e = c_v T.

    The self-similar variable is ξ = x / √(K*t) where
    K = 8 a c / ((4+n)*3*σ₀*ρ*c_v) with n=3.

    Returns T/T_bc.
    """
    xi_max = 1.11305
    omega = 0.05989
    K = 8 * A_RAD * C_LIGHT / ((4 + N_OPACITY) * 3 * SIGMA0 * RHO * CV_VOL)
    xi = x / np.sqrt(K * t)
    T_norm = np.zeros_like(xi)
    mask = xi < xi_max
    T_norm[mask] = ((1 - xi[mask] / xi_max) *
                    (1 + omega * xi[mask] / xi_max))**(1.0 / 6.0)
    return T_norm * T_BC


# ===========================================================================
# Problem setup and run
# ===========================================================================

def setup_and_run(Ix=200, Iy=1, N_quad=4, quad_type='level_symmetric',
                  Lx=0.5, tfinal=20.0, output_times=(1.0, 10.0, 20.0),
                  LOUD=False, use_dmd=True):
    """Run the uniform Marshak wave problem.

    Parameters
    ----------
    Ix : int
        Cells in x.
    Iy : int
        Cells in y.
    N_quad : int
        Quadrature order.
    Lx : float
        Domain length (cm).
    tfinal : float
        Final time (ns).
    output_times : tuple
        Times to output solution (ns).
    """
    Ly = 0.1  # small y-domain (problem is 1-D)

    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)

    x_faces = np.cumsum(np.concatenate([[0], dx_arr]))
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])

    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    print(f"Marshak Wave (uniform): Ix={Ix}, Iy={Iy}, M={M}")
    print(f"  σ = {SIGMA0} T^{{-{N_OPACITY}}}, c_v = {CV_VOL}")
    print(f"  T_bc = {T_BC} keV, T_init = {T_INIT} keV")
    print(f"  Lx = {Lx} cm, tfinal = {tfinal} ns")
    print(f"  output times: {output_times} ns")

    # --- Material functions ---
    def EOS(T):
        """e = c_v * T"""
        return CV_VOL * T

    def invEOS(e):
        """T = e / c_v"""
        return e / CV_VOL

    def sigma_func(T):
        """σ = σ₀ * T^{-n}"""
        T_safe = np.maximum(T, 0.005)  # floor to avoid overflow
        return SIGMA0 * T_safe**(-N_OPACITY)

    def scat_func(T):
        return np.zeros_like(T)

    # --- Initial conditions ---
    T_init = np.full((Ix, Iy, 4), T_INIT)
    phi_init = ac * T_init**4

    # --- External source (none) ---
    q_ext = np.zeros((Ix, Iy, 4))

    # --- Boundary conditions ---
    # Left: incoming blackbody at T_bc
    # Right: vacuum (zero)
    # y: reflecting
    I_bc = ac * T_BC**4

    def BCs_func(t):
        BCs_xlo = np.zeros((Iy, M, 2))
        for n in range(M):
            if Omega_x[n] > 0:
                BCs_xlo[:, n, :] = I_bc
        return {'xlo': BCs_xlo, 'xhi': None, 'ylo': None, 'yhi': None}

    # --- Time stepping ---
    dt_min = 1e-4   # ns
    dt_max = 0.05   # ns (matching 1-D solver)

    output_times = np.array(sorted(output_times))
    output_times = output_times[output_times <= tfinal]

    phis, Ts, iterations, ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        tolerance=1e-5, Linf_tol=1e-3, maxits=2000,
        K=800, R=10,
        reflect_xlo=False, reflect_xhi=False,
        reflect_ylo=(Iy > 1), reflect_yhi=(Iy > 1),
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=50,
        time_outputs=output_times,
    )

    print(f"\nTotal sweeps: {iterations}, steps: {len(ts)-1}")

    # Extract 1-D profiles at output times
    solutions = {}
    for t_target in output_times:
        idx = np.argmin(np.abs(np.array(ts) - t_target))
        T_snap = Ts[idx]
        phi_snap = phis[idx]
        T_1d = np.mean(T_snap, axis=(1, 2))
        phi_1d = np.mean(phi_snap, axis=(1, 2))
        Tr_1d = (phi_1d / ac)**0.25
        solutions[t_target] = {
            'T': T_1d, 'Tr': Tr_1d, 'phi': phi_1d,
            't_actual': ts[idx],
        }
        print(f"  t={ts[idx]:.3f} ns: T_max={T_1d.max():.4f} keV, "
              f"front~{x_centers[T_1d > 0.1*T_BC][-1] if np.any(T_1d > 0.1*T_BC) else 0:.3f} cm")

    return {
        'solutions': solutions, 'x': x_centers, 'Lx': Lx,
        'Ix': Ix, 'Iy': Iy, 'iterations': iterations,
        'output_times': output_times,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(results, savefile='marshak_wave_uniform_2d.png'):
    """Plot temperature profiles vs self-similar solution."""
    solutions = results['solutions']
    x = results['x']
    Lx = results['Lx']
    output_times = results['output_times']

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(output_times)))
    x_fine = np.linspace(1e-4, Lx, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for k, t_val in enumerate(output_times):
        if t_val not in solutions:
            continue
        sol = solutions[t_val]
        col = colors[k]

        # Material temperature
        ax1.plot(x, sol['T'], '-', color=col, lw=2,
                 label=f't={sol["t_actual"]:.1f} ns')
        # Self-similar
        T_ss = self_similar_solution(x_fine, t_val)
        ax1.plot(x_fine, T_ss, '--', color=col, lw=1.5, alpha=0.7)

        # Radiation temperature
        ax2.plot(x, sol['Tr'], '-', color=col, lw=2,
                 label=f't={sol["t_actual"]:.1f} ns')
        ax2.plot(x_fine, T_ss, '--', color=col, lw=1.5, alpha=0.7)

    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('Material Temperature T (keV)')
    ax1.set_title('Material Temperature')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, 1.1 * T_BC)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('Radiation Temperature $T_r$ (keV)')
    ax2.set_title('Radiation Temperature')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, 1.1 * T_BC)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(r'Marshak Wave (uniform) — 2-D S$_N$ Corner Balance'
                 '\n(dashed = self-similar)', fontsize=12)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {savefile}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uniform Marshak wave in 2-D S_N')
    parser.add_argument('--Ix', type=int, default=200, help='Cells in x')
    parser.add_argument('--Iy', type=int, default=1, help='Cells in y (1 for pure 1-D)')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order')
    parser.add_argument('--Lx', type=float, default=0.5, help='Domain length (cm)')
    parser.add_argument('--tfinal', type=float, default=20.0, help='Final time (ns)')
    parser.add_argument('--quad', type=str, default='level_symmetric')
    parser.add_argument('--no-dmd', action='store_true')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 5.0, 10.0, 20.0])
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix, Iy=args.Iy, N_quad=args.N,
        quad_type=args.quad, Lx=args.Lx, tfinal=args.tfinal,
        output_times=tuple(args.output_times),
        use_dmd=not args.no_dmd, LOUD=args.loud)

    plot_results(results)
