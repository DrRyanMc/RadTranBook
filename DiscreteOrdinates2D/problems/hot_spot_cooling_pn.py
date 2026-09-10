import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Hot-Spot Cooling Problem — P_N corner-balance solver.

Geometry: 5 cm × 5 cm Cartesian, uniform mesh.
Material: sigma = 1 cm^-1, c_v = 0.1 GJ/(cm^3*keV), temperature-independent.
IC: Gaussian temperature pulse T = T_peak * exp(-r^2/(2*sigma^2)) + T_cold,
    sigma = 1/8 cm, T_peak = 1 keV, centered at domain center. Radiation in equilibrium.
BCs: Vacuum on all faces.
Output times: 1, 5, 10 ns by default.

Run from DiscreteOrdinates2D/problems:
    python hot_spot_cooling_pn.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .hot_spot_fiducials import save_fiducial_history
else:
    from hot_spot_fiducials import save_fiducial_history

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
_pn_dir = os.path.join(_root, 'SphericalHarmonics')
if _pn_dir not in sys.path:
    sys.path.insert(0, _pn_dir)

from SphericalHarmonics.src.pn_solver_2d import temp_solve_pn_2d, c as C_LIGHT, ac

JACOBIAN_DIR = os.path.join(_pn_dir, 'Jacobians')

# ===========================================================================
# Problem parameters
# ===========================================================================
LX = 5.0
LY = 5.0
SIGMA = 1.0
CV_VOL = 0.1
T_PEAK = 1.0
T_COLD = 0.01
GAUSS_SIGMA = 0.125


def _build_initial_temperature(Ix, Iy, x_centers, y_centers, dx_arr, dy_arr, Lx, Ly):
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
                r2 = (xc - cx) ** 2 + (yc - cy) ** 2
                T_init[i, j, cc] = T_COLD + T_PEAK * np.exp(-r2 / (2.0 * GAUSS_SIGMA ** 2))
    return T_init


def setup_and_run(
    Ix=100,
    Iy=100,
    Lx=5.0,
    Ly=5.0,
    N_pn=3,
    output_times=(1.0, 5.0, 10.0),
    dt_start=1e-3,
    dt_max=0.5,
    tolerance=1e-6,
    maxits=150,
    n_gs=1,
    use_gmres=False,
    LOUD=False,
    filter_type='lanczos',
    filter_strength=None,
    filter_exp_order=4,
):
    dx_arr = np.full(Ix, Lx / Ix)
    dy_arr = np.full(Iy, Ly / Iy)
    x_faces = np.linspace(0, Lx, Ix + 1)
    y_faces = np.linspace(0, Ly, Iy + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])

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

    T_init = _build_initial_temperature(Ix, Iy, x_centers, y_centers, dx_arr, dy_arr, Lx, Ly)
    phi_init = ac * T_init ** 4
    q_ext = np.zeros((Ix, Iy, 4))

    output_times = np.array(sorted(output_times), dtype=float)

    cx, cy = Lx / 2, Ly / 2
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
    Er_history = {label: [] for label in fid_indices}

    def record_snapshot(t_now, phi_now, T_now):
        fid_times.append(float(t_now))
        for label, (i, j) in fid_indices.items():
            T_history[label].append(float(np.mean(T_now[i, j, :])))
            phi_avg = float(np.mean(phi_now[i, j, :]))
            Er_history[label].append(phi_avg / C_LIGHT)
            Tr_history[label].append(float(np.maximum(phi_avg, 0.0) / ac) ** 0.25)

    print(f"Hot-Spot Cooling P_{N_pn}: {Ix}x{Iy}")
    print(f"  sigma = {SIGMA} cm^-1, c_v = {CV_VOL} GJ/(cm^3*keV)")
    print(f"  Gaussian IC: T_peak={T_PEAK} keV, sigma_gauss={GAUSS_SIGMA} cm")
    print(f"  Background: T={T_COLD} keV")
    print(f"  Output times: {output_times}")
    print(f"  Jacobians: {JACOBIAN_DIR}")

    solutions = {}
    phi_curr = phi_init.copy()
    T_curr = T_init.copy()
    I_curr = None
    t_done = 0.0
    total_iters = 0

    # Seed histories at t=0 so exported NPZ contains the initial state.
    record_snapshot(t_done, phi_curr, T_curr)

    for t_target in output_times:
        if t_target <= t_done + 1e-14:
            continue
        duration = float(t_target - t_done)

        # temp_solve_pn_2d reports callback times starting at 0 for each call.
        # Convert those segment-local times to absolute simulation time.
        t_base = t_done
        def _record_snapshot_abs(t_local, phi_now, T_now):
            record_snapshot(t_base + float(t_local), phi_now, T_now)

        phi_curr, T_curr, I_curr, _t_reached, history = temp_solve_pn_2d(
            Ix,
            Iy,
            dx_arr,
            dy_arr,
            q_ext,
            sigma_func,
            scat_func,
            N_pn,
            JACOBIAN_DIR,
            EOS,
            invEOS,
            phi_curr,
            T_curr,
            I_init=I_curr,
            dt_start=dt_start,
            t_end=duration,
            reflect_xlo=False,
            reflect_xhi=False,
            reflect_ylo=False,
            reflect_yhi=False,
            tolerance=tolerance,
            maxits=maxits,
            W=0,
            n_gs=n_gs,
            use_gmres=use_gmres,
            loud=LOUD,
            print_stride=20,
            dt_max=dt_max,
            T_floor=1e-6,
            step_callback=_record_snapshot_abs,
            filter_type=filter_type,
            filter_strength=filter_strength,
            filter_exp_order=filter_exp_order,
        )

        step_iters = int(sum(int(h['sweeps']) for h in history))
        total_iters += step_iters
        t_done = float(t_target)

        solutions[t_target] = {
            'T': T_curr.copy(),
            'phi': phi_curr.copy(),
            't_actual': t_done,
        }
        print(f"  t={t_done:.2f} ns: T_max={np.max(T_curr):.4f} keV  step-its={step_iters}")

    print(f"\nTotal source iterations: {total_iters}")

    fid_data = {
        label: {
            'T_mat': np.asarray(T_history[label]),
            'T_rad': np.asarray(Tr_history[label]),
            'E_rad': np.asarray(Er_history[label]),
        }
        for label in fid_indices
    }

    return {
        'solutions': solutions,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'x_faces': x_faces,
        'y_faces': y_faces,
        'Lx': float(Lx),
        'Ly': float(Ly),
        'Ix': Ix,
        'Iy': Iy,
        'ts': np.asarray(fid_times),
        'fid_data': fid_data,
        'fid_indices': fid_indices,
        'output_times': output_times,
    }


def plot_snapshot(results, t_target, savefile_prefix='hot_spot_pn'):
    sol = results['solutions'][t_target]
    x_f = results['x_faces']
    y_f = results['y_faces']
    t_tag = f"{sol['t_actual']:.3f}".rstrip('0').rstrip('.')
    t_tag = t_tag.replace('.', 'p')

    T_cell = np.mean(sol['T'], axis=2)
    phi_cell = np.mean(sol['phi'], axis=2)
    Tr_cell = (np.maximum(phi_cell, 0.0) / ac) ** 0.25
    Er_cell = phi_cell / C_LIGHT

    plot_specs = [
        ('material', 'T (keV)', T_cell, 'plasma', None),
        ('radiation', r'$T_r$ (keV)', Tr_cell, 'plasma', None),
        ('erad', r'$E_r$ (GJ/cm$^3$)', Er_cell, 'coolwarm', 'signed'),
    ]
    for field, label, data, cmap, mode in plot_specs:
        fig, ax = plt.subplots(figsize=(6, 6))
        if mode == 'signed':
            vmax = max(float(np.max(np.abs(data))), 1e-12)
            im = ax.pcolormesh(
                x_f, y_f, data.T, shading='flat', cmap=cmap,
                vmin=-vmax, vmax=vmax,
            )
        else:
            vmax = max(float(data.max()), 0.05)
            im = ax.pcolormesh(
                x_f, y_f, data.T, shading='flat', cmap=cmap,
                vmin=0.0, vmax=vmax,
            )
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        plt.colorbar(im, ax=ax, label=label)
        ax.set_aspect('equal')

        cx = 0.5 * (x_f[0] + x_f[-1])
        cy = 0.5 * (y_f[0] + y_f[-1])
        circle = plt.Circle((cx, cy), GAUSS_SIGMA, fill=False, edgecolor='cyan', lw=1, ls='--')
        ax.add_patch(circle)

        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t{t_tag}ns.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='hot_spot_pn'):
    ts = results['ts']
    fid_data = results['fid_data']
    markers = ['o', 's', '^', 'd']
    colors = ['blue', 'red', 'green', 'purple']

    for field in ['T_mat', 'T_rad']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (label, data) in enumerate(fid_data.items()):
            ax.semilogy(
                ts,
                data[field],
                marker=markers[idx % 4],
                color=colors[idx % 4],
                lw=2,
                ms=5,
                markevery=max(1, len(ts) // 20),
                label=label,
                alpha=0.8,
            )
        ylabel = 'Material Temperature (keV)' if field == 'T_mat' else 'Radiation Temperature (keV)'
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3, ls='--')
        ax.set_xlim(0, ts[-1] if len(ts) > 0 else 1.0)
        plt.tight_layout()
        fname = f'{savefile_prefix}_history_{field}.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot-Spot Cooling P_N')
    parser.add_argument('--Ix', type=int, default=100)
    parser.add_argument('--Iy', type=int, default=100)
    parser.add_argument('--Lx', type=float, default=5.0, help='Domain size in x (cm)')
    parser.add_argument('--Ly', type=float, default=5.0, help='Domain size in y (cm)')
    parser.add_argument('--Npn', type=int, default=3, help='P_N order (odd preferred)')
    parser.add_argument('--output-times', type=float, nargs='+', default=[1.0, 5.0, 10.0])
    parser.add_argument('--dt-start', type=float, default=1e-3)
    parser.add_argument('--dt-max', type=float, default=0.5)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--maxits', type=int, default=150)
    parser.add_argument('--n-gs', type=int, default=1)
    parser.add_argument('--gmres', action='store_true', help='Use matrix-free GMRES instead of GS')
    parser.add_argument('--loud', action='store_true')
    parser.add_argument(
        '--fiducial-output',
        default='hot_spot_pn_fiducials.npz',
        help='NPZ file for fiducial histories',
    )
    args = parser.parse_args()

    results = setup_and_run(
        Ix=args.Ix,
        Iy=args.Iy,
        Lx=args.Lx,
        Ly=args.Ly,
        N_pn=args.Npn,
        output_times=tuple(args.output_times),
        dt_start=args.dt_start,
        dt_max=args.dt_max,
        tolerance=args.tol,
        maxits=args.maxits,
        n_gs=args.n_gs,
        use_gmres=args.gmres,
        LOUD=args.loud,
    )

    save_fiducial_history(results, args.fiducial_output, method=f'P_{args.Npn}')

    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t)
    plot_fiducial_history(results)
