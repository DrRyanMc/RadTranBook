import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Hot-Spot Cooling — Equilibrium Diffusion.

Same physics as the S_N, IMC, and nonequilibrium diffusion versions:
  Geometry: 5 cm × 5 cm Cartesian, 100×100 uniform mesh.
  Material: σ_R = 1 cm⁻¹, c_v = 0.1 GJ/(cm³·keV).
  IC: T = 1 keV (equilibrium) in center 0.25×0.25 cm, T = 0.01 keV elsewhere.
  BCs: Vacuum (Marshak) on all faces.
  Output times: 1, 5, 10 ns.

Run:
    python hot_spot_cooling_eq_diff.py
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

from EqDiffusion.utils.twoDFV import RadiationDiffusionSolver2D, temperature_from_Er, A_RAD, C_LIGHT, RHO

# ===========================================================================
# Problem parameters
# ===========================================================================
LX = 5.0;  LY = 5.0
SIGMA = 1.0        # cm⁻¹
CV_VOL = 0.1       # GJ/(cm³·keV)
T_HOT = 1.0;  T_COLD = 0.01
HOT_HALF = 0.125


# ===========================================================================
# Material property functions
# ===========================================================================

def rosseland_opacity(Er, coord1_val, coord2_val):
    """Equilibrium solver takes Er, not T."""
    return SIGMA

def specific_heat(T, coord1_val, coord2_val):
    return CV_VOL / RHO   # per unit mass

def material_energy(T, coord1_val, coord2_val):
    return CV_VOL * T


# ===========================================================================
# Boundary conditions — Marshak vacuum
# ===========================================================================

def bc_vacuum(Er_boundary, coord1_val, coord2_val, geometry='cartesian', time=0.0):
    return 0.5, 1.0 / (3.0 * SIGMA), 0.0


# ===========================================================================
# Setup and run
# ===========================================================================

def setup_and_run(Nx=100, Ny=100, dt=0.01,
                  output_times=(1.0, 5.0, 10.0)):

    cx, cy = LX / 2, LY / 2

    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=LX, n1_cells=Nx,
        coord2_min=0.0, coord2_max=LY, n2_cells=Ny,
        geometry='cartesian',
        dt=dt,
        max_newton_iter=50,
        newton_tol=1e-8,
        rosseland_opacity_func=rosseland_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        left_bc_func=bc_vacuum,
        right_bc_func=bc_vacuum,
        bottom_bc_func=bc_vacuum,
        top_bc_func=bc_vacuum,
        theta=1.0,
    )

    # Initial condition: Er = a T^4
    x_c = solver.coord1_centers
    y_c = solver.coord2_centers
    Er_init = np.full((Nx, Ny), A_RAD * T_COLD**4)
    for i in range(Nx):
        for j in range(Ny):
            if abs(x_c[i] - cx) <= HOT_HALF and abs(y_c[j] - cy) <= HOT_HALF:
                Er_init[i, j] = A_RAD * T_HOT**4

    solver.set_initial_condition(Er_init)

    output_times = np.array(sorted(output_times))
    tfinal = output_times[-1]

    print(f"Hot-Spot Cooling Eq Diffusion: {Nx}×{Ny}")
    print(f"  σ={SIGMA}, c_v={CV_VOL}, dt={dt}, tfinal={tfinal}")

    # Fiducial setup
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

    T_init_2d = temperature_from_Er(Er_init)
    times_list = [0.0]
    T_history = {lab: [T_init_2d[i, j]] for lab, (i, j) in fid_indices.items()}

    # Time loop
    solutions = {}
    current_time = 0.0
    step_count = 0
    wall_start = timer_mod.perf_counter()
    next_output = 0

    while current_time < tfinal - 1e-12:
        step_dt = dt
        if next_output < len(output_times):
            if current_time + step_dt > output_times[next_output]:
                step_dt = output_times[next_output] - current_time
        if current_time + step_dt > tfinal:
            step_dt = tfinal - current_time

        solver.dt = step_dt
        solver.time_step(n_steps=1, verbose=False)
        current_time += step_dt
        step_count += 1

        Er_now = solver.Er.reshape((Nx, Ny))
        T_now = temperature_from_Er(Er_now)

        times_list.append(current_time)
        for lab, (i, j) in fid_indices.items():
            T_history[lab].append(T_now[i, j])

        if (next_output < len(output_times) and
                abs(current_time - output_times[next_output]) < 1e-10):
            t_out = output_times[next_output]
            solutions[t_out] = {
                'T': T_now.copy(),   # equilibrium: T_mat = T_rad
                't_actual': current_time,
            }
            print(f"  t={current_time:.2f} ns: T_max={T_now.max():.4f} keV")
            next_output += 1

        if step_count % 100 == 0:
            print(f"  step {step_count}, t={current_time:.3f} ns, "
                  f"T_max={T_now.max():.4f} keV")

    wall_time = timer_mod.perf_counter() - wall_start
    print(f"\n  Wall time: {wall_time:.1f} s, steps: {step_count}")

    fid_data = {lab: {'T_mat': np.array(T_history[lab]),
                      'T_rad': np.array(T_history[lab])}  # equilibrium
                for lab in fid_indices}

    return {
        'solutions': solutions,
        'x_centers': x_c, 'y_centers': y_c,
        'x_faces': solver.coord1_faces, 'y_faces': solver.coord2_faces,
        'ts': np.array(times_list),
        'fid_data': fid_data, 'fid_indices': fid_indices,
        'output_times': output_times,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_snapshot(results, t_target, savefile_prefix='hot_spot_eqdiff'):
    sol = results['solutions'][t_target]
    x_f, y_f = results['x_faces'], results['y_faces']
    T = sol['T']
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(T.max(), 0.05)
    im = ax.pcolormesh(x_f, y_f, T.T, shading='flat', cmap='plasma',
                       vmin=0.0, vmax=vmax)
    ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
    ax.set_title(f'Eq Diffusion T, t={sol["t_actual"]:.1f} ns')
    plt.colorbar(im, ax=ax, label='T (keV)'); ax.set_aspect('equal')
    cx, cy = LX/2, LY/2
    ax.add_patch(plt.Rectangle((cx-HOT_HALF, cy-HOT_HALF), 2*HOT_HALF,
                 2*HOT_HALF, fill=False, ec='cyan', lw=1, ls='--'))
    plt.tight_layout()
    fname = f'{savefile_prefix}_t{t_target:.0f}ns.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight'); print(f'  Saved: {fname}')
    plt.close()


def plot_fiducial_history(results, savefile_prefix='hot_spot_eqdiff'):
    ts = results['ts']; fid_data = results['fid_data']
    markers = ['o', 's', '^', 'd']; colors = ['blue', 'red', 'green', 'purple']
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, data) in enumerate(fid_data.items()):
        ax.semilogy(ts, data['T_mat'], marker=markers[idx%4], color=colors[idx%4],
                    lw=2, ms=5, markevery=max(1,len(ts)//20), label=label, alpha=0.8)
    ax.set_xlabel('Time (ns)'); ax.set_ylabel('Temperature (keV)')
    ax.set_title('Eq Diffusion Hot-Spot Cooling')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3, ls='--')
    ax.set_xlim(0, ts[-1]); plt.tight_layout()
    fname = f'{savefile_prefix}_history.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight'); print(f'  Saved: {fname}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot-Spot Eq Diffusion')
    parser.add_argument('--Nx', type=int, default=100)
    parser.add_argument('--Ny', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--output-times', type=float, nargs='+', default=[1.0, 5.0, 10.0])
    parser.add_argument('--fiducial-output',
                        default='hot_spot_eqdiff_fiducials.npz',
                        help='NPZ file for every-step fiducial histories')
    args = parser.parse_args()

    results = setup_and_run(Nx=args.Nx, Ny=args.Ny, dt=args.dt,
                            output_times=tuple(args.output_times))
    save_fiducial_history(results, args.fiducial_output,
                           method='equilibrium_diffusion')
    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t)
    plot_fiducial_history(results)
