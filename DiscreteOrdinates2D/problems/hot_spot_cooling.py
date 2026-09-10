import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Hot-Spot Cooling Problem — S_N corner-balance solver.

Geometry: 5 cm × 5 cm Cartesian, uniform 100×100 mesh.
Material: σ = 1 cm⁻¹, c_v = 0.1 GJ/(cm³·keV), temperature-independent.
IC: Gaussian temperature pulse T = T_peak * exp(-r²/(2σ²)) + T_cold,
    σ = 1/8 cm, T_peak = 1 keV, centered at domain center. Radiation in equilibrium.
BCs: Vacuum on all faces.
Output times: 1, 5, 10 ns.

This problem demonstrates ray effects as the hot spot radiates isotropically
into the cold surrounding medium. The rotationally symmetric IC makes
angular artifacts from S_N clearly visible.

Run from the DiscreteOrdinates2D/problems directory:
    python hot_spot_cooling.py
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
if __package__:
    from .hot_spot_fiducials import save_fiducial_history
else:
    from hot_spot_fiducials import save_fiducial_history

# ===========================================================================
# Problem parameters
# ===========================================================================
LX = 5.0        # domain size (cm)
LY = 5.0
SIGMA = 1.0     # opacity (cm⁻¹) — constant
CV_VOL = 0.1    # heat capacity GJ/(cm³·keV)
T_PEAK = 1.0    # keV — Gaussian peak temperature
T_COLD = 0.01   # keV — background
GAUSS_SIGMA = 0.125  # cm — Gaussian standard deviation (1/8 cm)


# ===========================================================================
# Problem setup
# ===========================================================================

def setup_and_run(Ix=100, Iy=100, Lx=5.0, Ly=5.0,
                  N_quad=4, quad_type='level_symmetric',
                  output_times=(1.0, 5.0, 10.0),
                  LOUD=False, use_dmd=True,
                  dt_min=1e-3, dt_max=0.5):
    """Run the hot-spot cooling problem."""

    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)
    x_faces = np.linspace(0, Lx, Ix + 1)
    y_faces = np.linspace(0, Ly, Iy + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    # Uniform material — closures return constant arrays
    sigma_arr = np.full((Ix, Iy, 4), SIGMA)
    cv_arr = np.full((Ix, Iy, 4), CV_VOL)

    def sigma_func(T):
        return sigma_arr

    def scat_func(T):
        return np.zeros_like(T)

    def EOS(T):
        return cv_arr * T

    def invEOS(e):
        return e / cv_arr

    # Initial condition: Gaussian temperature pulse centered in domain
    cx, cy = Lx / 2, Ly / 2
    T_init = np.full((Ix, Iy, 4), T_COLD)

    # Corner offsets (NE, NW, SW, SE)
    cx_off = np.array([+0.25, -0.25, -0.25, +0.25])
    cy_off = np.array([+0.25, +0.25, -0.25, -0.25])

    for i in range(Ix):
        for j in range(Iy):
            for cc in range(4):
                xc = x_centers[i] + cx_off[cc] * dx_arr[i]
                yc = y_centers[j] + cy_off[cc] * dy_arr[j]
                r2 = (xc - cx)**2 + (yc - cy)**2
                T_init[i, j, cc] = T_COLD + T_PEAK * np.exp(-r2 / (2.0 * GAUSS_SIGMA**2))

    phi_init = ac * T_init**4
    q_ext = np.zeros((Ix, Iy, 4))

    # Boundary conditions: vacuum everywhere
    def BCs_func(t):
        return {'xlo': None, 'xhi': None, 'ylo': None, 'yhi': None}

    # Time stepping
    output_times = np.array(sorted(output_times))
    tfinal = output_times[-1]

    # Track only the requested point histories at every accepted step. Full
    # fields are retained only at requested output times by temp_solve_2d.
    fid_points = {
        'center': (cx, cy),
        'x+1 cm': (cx + 1.0, cy),
        'diagonal +1 cm': (cx + 0.707, cy + 0.707),
        'y+1 cm': (cx, cy + 1.0),
    }
    fid_indices = {}
    for label, (xv, yv) in fid_points.items():
        i = np.argmin(np.abs(x_centers - xv))
        j = np.argmin(np.abs(y_centers - yv))
        fid_indices[label] = (i, j)

    fid_times = []
    T_history = {label: [] for label in fid_indices}
    Tr_history = {label: [] for label in fid_indices}

    def record_fiducials(t, phi, T):
        fid_times.append(t)
        for label, (i, j) in fid_indices.items():
            T_history[label].append(np.mean(T[i, j, :]))
            Tr_history[label].append(
                (np.mean(phi[i, j, :]) / ac)**0.25)

    print(f"Hot-Spot Cooling: {Ix}×{Iy}, S_{N_quad} ({quad_type}), M={M}")
    print(f"  σ = {SIGMA} cm⁻¹, c_v = {CV_VOL} GJ/(cm³·keV)")
    print(f"  Gaussian IC: T_peak={T_PEAK} keV, σ_gauss={GAUSS_SIGMA} cm")
    print(f"  Background: T={T_COLD} keV")
    print(f"  Output times: {output_times}")

    phis, Ts, iterations, snapshot_ts, its_per_step = temp_solve_2d(
        Ix, Iy, dx_arr, dy_arr,
        q_ext, sigma_func, scat_func,
        quad_type, N_quad,
        BCs_func, EOS, invEOS,
        phi_init, T_init,
        dt_min=dt_min, dt_max=dt_max, tfinal=tfinal,
        tolerance=1e-6, Linf_tol=1e-3, maxits=100,
        K=50, R=3, W=3,
        reflect_xlo=False, reflect_xhi=False,
        reflect_ylo=False, reflect_yhi=False,
        use_dmd=use_dmd,
        LOUD=LOUD,
        print_stride=20,
        time_outputs=output_times,
        store_full_history=False,
        step_callback=record_fiducials,
    )

    print(f"\nTotal sweeps: {iterations}, steps: {len(its_per_step)}")

    # Extract snapshots at output times
    solutions = {}
    for t_target in output_times:
        idx = np.argmin(np.abs(snapshot_ts - t_target))
        solutions[t_target] = {
            'T': Ts[idx],
            'phi': phis[idx],
            't_actual': snapshot_ts[idx],
        }
        print(f"  t={snapshot_ts[idx]:.2f} ns: T_max={np.max(Ts[idx]):.4f} keV")

    fid_data = {
        label: {
            'T_mat': np.asarray(T_history[label]),
            'T_rad': np.asarray(Tr_history[label]),
        }
        for label in fid_indices
    }

    return {
        'solutions': solutions, 'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': x_faces, 'y_faces': y_faces,
        'Ix': Ix, 'Iy': Iy, 'ts': np.asarray(fid_times),
        'fid_data': fid_data, 'fid_indices': fid_indices,
        'output_times': output_times,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_snapshot(results, t_target, savefile_prefix='hot_spot'):
    """Plot material T and radiation T at a given time."""
    sol = results['solutions'][t_target]
    x_f = results['x_faces']
    y_f = results['y_faces']

    T_cell = np.mean(sol['T'], axis=2)
    Tr_cell = (np.mean(sol['phi'], axis=2) / ac)**0.25

    for field, label, data in [('material', 'T (keV)', T_cell),
                                ('radiation', r'$T_r$ (keV)', Tr_cell)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        vmax =  max(data.max(), 0.05)
        im = ax.pcolormesh(x_f, y_f, data.T, shading='flat', cmap='plasma',
                           vmin=0.0, vmax=vmax)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        #ax.set_title(f'{field.capitalize()} Temperature, t={sol["t_actual"]:.1f} ns')
        plt.colorbar(im, ax=ax, label=label)
        ax.set_aspect('equal')

        # Mark Gaussian 1-sigma contour
        cx, cy = 0.5 * (x_f[0] + x_f[-1]), 0.5 * (y_f[0] + y_f[-1])
        circle = plt.Circle((cx, cy), GAUSS_SIGMA,
                             fill=False, edgecolor='cyan', lw=1, ls='--')
        ax.add_patch(circle)

        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t{t_target:.0f}ns.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='hot_spot'):
    """Plot temperature history at fiducial points."""
    ts = results['ts']
    fid_data = results['fid_data']
    markers = ['o', 's', '^', 'd']
    colors = ['blue', 'red', 'green', 'purple']

    for field in ['T_mat', 'T_rad']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (label, data) in enumerate(fid_data.items()):
            ax.semilogy(ts, data[field],
                        marker=markers[idx % 4], color=colors[idx % 4],
                        lw=2, ms=5, markevery=max(1, len(ts)//20),
                        label=label, alpha=0.8)
        ylabel = ('Material Temperature (keV)' if field == 'T_mat'
                  else 'Radiation Temperature (keV)')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3, ls='--')
        ax.set_xlim(0, ts[-1])
        plt.tight_layout()
        fname = f'{savefile_prefix}_history_{field}.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot-Spot Cooling S_N')
    parser.add_argument('--Ix', type=int, default=100)
    parser.add_argument('--Iy', type=int, default=100)
    parser.add_argument('--Lx', type=float, default=5.0,
                        help='Domain size in x (cm)')
    parser.add_argument('--Ly', type=float, default=5.0,
                        help='Domain size in y (cm)')
    parser.add_argument('--N', type=int, default=4, help='Quadrature order')
    parser.add_argument('--quad', type=str, default='level_symmetric')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 5.0, 10.0])
    parser.add_argument('--dt-max', type=float, default=0.5)
    parser.add_argument('--dt-min', type=float, default=1e-3)
    parser.add_argument('--no-dmd', action='store_true')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument('--fiducial-output',
                        default='hot_spot_sn_fiducials.npz',
                        help='NPZ file for every-step fiducial histories')
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix, Iy=args.Iy, Lx=args.Lx, Ly=args.Ly, N_quad=args.N,
        quad_type=args.quad,
        output_times=tuple(args.output_times),
        use_dmd=not args.no_dmd,
        LOUD=args.loud,
        dt_max=args.dt_max,
        dt_min=args.dt_min
    )

    save_fiducial_history(results, args.fiducial_output, method='S_N')

    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t)
    plot_fiducial_history(results)
