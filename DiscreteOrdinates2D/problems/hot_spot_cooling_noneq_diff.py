import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Hot-Spot Cooling — Flux-Limited Non-Equilibrium Diffusion.

Same physics as the S_N and IMC versions:
  Geometry: 5 cm × 5 cm Cartesian, 100×100 uniform mesh.
  Material: σ_R = σ_P = 1 cm⁻¹, c_v = 0.1 GJ/(cm³·keV).
  IC: T = 1 keV (equilibrium) in center 0.25×0.25 cm, T = 0.01 keV elsewhere.
  BCs: Vacuum (Marshak) on all faces.
  Output times: 1, 5, 10 ns.

Run:
    python hot_spot_cooling_noneq_diff.py
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as timer_mod

if __package__:
    from .hot_spot_fiducials import save_fiducial_history
else:
    from hot_spot_fiducials import save_fiducial_history

from nonEquilibriumDiffusion.finite_volume_2d.src.twoDFV import (
    NonEquilibriumRadiationDiffusionSolver2D,
    A_RAD, C_LIGHT, RHO,
    flux_limiter_levermore_pomraning,
)

# ===========================================================================
# Problem parameters
# ===========================================================================
LX = 5.0;  LY = 5.0
SIGMA = 1.0        # cm⁻¹
CV_VOL = 0.1       # GJ/(cm³·keV)
T_HOT = 1.0;  T_COLD = 0.01
HOT_HALF = 0.125   # half-width of hot square


# ===========================================================================
# Material property functions  (signature: f(T, x, y) → float)
# ===========================================================================

def rosseland_opacity(T, x, y):
    return SIGMA

def planck_opacity(T, x, y):
    return SIGMA

def specific_heat(T, x, y):
    return CV_VOL / RHO          # per unit mass

def material_energy(T, x, y):
    return RHO * (CV_VOL / RHO) * T   # = CV_VOL * T

def inverse_material_energy(e, x, y):
    return e / CV_VOL


# ===========================================================================
# Boundary conditions — Marshak vacuum: (1/2)φ + (1/(3σ)) dφ/dn = 0
# For Robin BC  A·φ + B·dφ/dn = C  with C=0, A=1/2, B=1/(3σ)
# ===========================================================================

def bc_vacuum(phi, pos, t, boundary='left', geometry='cartesian'):
    return 0.5, 1.0 / (3.0 * SIGMA), 0.0


# ===========================================================================
# Setup and run
# ===========================================================================

def setup_and_run(Nx=100, Ny=100, dt=0.01,
                  output_times=(1.0, 5.0, 10.0),
                  use_flux_limiter=True):

    cx, cy = LX / 2, LY / 2

    boundary_funcs = {
        'left': bc_vacuum, 'right': bc_vacuum,
        'bottom': bc_vacuum, 'top': bc_vacuum,
    }

    fl = flux_limiter_levermore_pomraning if use_flux_limiter else None

    solver = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=0.0, x_max=LX, nx_cells=Nx,
        y_min=0.0, y_max=LY, ny_cells=Ny,
        geometry='cartesian',
        dt=dt,
        max_newton_iter=50,
        newton_tol=1e-8,
        rosseland_opacity_func=rosseland_opacity,
        planck_opacity_func=planck_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        inverse_material_energy_func=inverse_material_energy,
        boundary_funcs=boundary_funcs,
        theta=1.0,
        flux_limiter_func=fl,
    )

    # Initial condition
    T_2d = np.full((Nx, Ny), T_COLD)
    x_c = solver.x_centers
    y_c = solver.y_centers
    for i in range(Nx):
        for j in range(Ny):
            if abs(x_c[i] - cx) <= HOT_HALF and abs(y_c[j] - cy) <= HOT_HALF:
                T_2d[i, j] = T_HOT

    phi_2d = C_LIGHT * A_RAD * T_2d**4
    solver.set_initial_condition(phi_init=phi_2d, T_init=T_2d)

    output_times = np.array(sorted(output_times))
    tfinal = output_times[-1]

    print(f"Hot-Spot Cooling NonEq Diffusion: {Nx}×{Ny}")
    print(f"  σ={SIGMA}, c_v={CV_VOL}, flux_limiter={use_flux_limiter}")
    print(f"  dt={dt}, tfinal={tfinal}")

    # Time loop
    solutions = {}
    current_time = 0.0
    step_count = 0
    wall_start = timer_mod.perf_counter()
    next_output = 0

    # Fiducial tracking
    fid_points = {
        'center': (cx, cy),
        'x+1 cm': (cx + 1.0, cy),
        'diagonal +1 cm': (cx + 0.707, cy + 0.707),
        'y+1 cm': (cx, cy + 1.0),
    }
    fid_indices = {}
    for label, (xv, yv) in fid_points.items():
        i = np.argmin(np.abs(x_c - xv))
        j = np.argmin(np.abs(y_c - yv))
        fid_indices[label] = (i, j)

    times_list = [0.0]
    T_history = {lab: [T_2d[i, j]] for lab, (i, j) in fid_indices.items()}
    Tr_history = {lab: [(phi_2d[i, j] / C_LIGHT / A_RAD)**0.25]
                  for lab, (i, j) in fid_indices.items()}

    while current_time < tfinal - 1e-12:
        step_dt = dt
        # Snap to output times
        if next_output < len(output_times):
            if current_time + step_dt > output_times[next_output]:
                step_dt = output_times[next_output] - current_time
        if current_time + step_dt > tfinal:
            step_dt = tfinal - current_time

        solver.dt = step_dt
        solver.time_step(n_steps=1, verbose=False)
        current_time += step_dt
        step_count += 1

        T_now = solver.get_T_2d()
        phi_now = solver.get_phi_2d()

        # Track fiducial points
        times_list.append(current_time)
        for lab, (i, j) in fid_indices.items():
            T_history[lab].append(T_now[i, j])
            Tr_history[lab].append((phi_now[i, j] / C_LIGHT / A_RAD)**0.25)

        # Save at output times
        if (next_output < len(output_times) and
                abs(current_time - output_times[next_output]) < 1e-10):
            t_out = output_times[next_output]
            solutions[t_out] = {
                'T': T_now.copy(),
                'Tr': (phi_now / C_LIGHT / A_RAD)**0.25,
                't_actual': current_time,
            }
            print(f"  t={current_time:.2f} ns: T_max={T_now.max():.4f}, "
                  f"Tr_max={solutions[t_out]['Tr'].max():.4f} keV")
            next_output += 1

        if step_count % 100 == 0:
            print(f"  step {step_count}, t={current_time:.3f} ns, "
                  f"T_max={T_now.max():.4f} keV")

    wall_time = timer_mod.perf_counter() - wall_start
    print(f"\n  Wall time: {wall_time:.1f} s, steps: {step_count}")

    fid_data = {}
    for lab in fid_indices:
        fid_data[lab] = {
            'T_mat': np.array(T_history[lab]),
            'T_rad': np.array(Tr_history[lab]),
        }

    return {
        'solutions': solutions,
        'x_centers': x_c, 'y_centers': y_c,
        'x_faces': solver.x_faces, 'y_faces': solver.y_faces,
        'ts': np.array(times_list),
        'fid_data': fid_data, 'fid_indices': fid_indices,
        'output_times': output_times,
    }


# ===========================================================================
# Plotting (reused pattern)
# ===========================================================================

def plot_snapshot(results, t_target, savefile_prefix='hot_spot_noneq'):
    sol = results['solutions'][t_target]
    x_f, y_f = results['x_faces'], results['y_faces']
    for field, label, data in [('material', 'T (keV)', sol['T']),
                                ('radiation', r'$T_r$ (keV)', sol['Tr'])]:
        fig, ax = plt.subplots(figsize=(6, 6))
        vmax = max(data.max(), 0.05)
        im = ax.pcolormesh(x_f, y_f, data.T, shading='flat', cmap='plasma',
                           vmin=0.0, vmax=vmax)
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
        #ax.set_title(f'NonEq Diff {field.capitalize()} T, t={sol["t_actual"]:.1f} ns')
        plt.colorbar(im, ax=ax, label=label); ax.set_aspect('equal')
        cx, cy = LX/2, LY/2
        ax.add_patch(plt.Rectangle((cx-HOT_HALF, cy-HOT_HALF), 2*HOT_HALF,
                     2*HOT_HALF, fill=False, ec='cyan', lw=1, ls='--'))
        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t{t_target:.0f}ns.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight'); print(f'  Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='hot_spot_noneq'):
    ts = results['ts']; fid_data = results['fid_data']
    markers = ['o', 's', '^', 'd']; colors = ['blue', 'red', 'green', 'purple']
    for field in ['T_mat', 'T_rad']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (label, data) in enumerate(fid_data.items()):
            ax.semilogy(ts, data[field], marker=markers[idx%4], color=colors[idx%4],
                        lw=2, ms=5, markevery=max(1,len(ts)//20), label=label, alpha=0.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Material T (keV)' if field=='T_mat' else 'Radiation T (keV)')
        ax.set_title('NonEq Diffusion Hot-Spot Cooling')
        ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3, ls='--')
        ax.set_xlim(0, ts[-1]); plt.tight_layout()
        fname = f'{savefile_prefix}_history_{field}.png'
        plt.savefig(fname, dpi=650, bbox_inches='tight'); print(f'  Saved: {fname}')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot-Spot NonEq Diffusion')
    parser.add_argument('--Nx', type=int, default=100)
    parser.add_argument('--Ny', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--output-times', type=float, nargs='+', default=[1.0, 5.0, 10.0])
    parser.add_argument('--no-limiter', action='store_true')
    parser.add_argument('--fiducial-output',
                        default='hot_spot_noneq_fiducials.npz',
                        help='NPZ file for every-step fiducial histories')
    args = parser.parse_args()

    results = setup_and_run(Nx=args.Nx, Ny=args.Ny, dt=args.dt,
                            output_times=tuple(args.output_times),
                            use_flux_limiter=not args.no_limiter)
    save_fiducial_history(results, args.fiducial_output,
                           method='nonequilibrium_diffusion')
    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t)
    plot_fiducial_history(results)
