import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
2-D Hot-Spot Cooling Problem — Gray IMC (Fleck-Cummings).

Same physics as the S_N version:
  Geometry: 5 cm × 5 cm Cartesian, uniform mesh.
  Material: σ = 1 cm⁻¹, c_v = 0.1 GJ/(cm³·keV), temperature-independent.
  IC: T = 1 keV (equilibrium) in center 0.25×0.25 cm, T = 0.01 keV elsewhere.
  BCs: Vacuum on all faces.
  Output times: 1, 5, 10 ns.

Demonstrates ray effects (or lack thereof) in the IMC solution for comparison
with the S_N discrete-ordinates result.

Run from the IMC directory:
    python hot_spot_cooling_imc.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'IMC'))
import IMC2D as imc2d

# ===========================================================================
# Problem parameters (matching the S_N version)
# ===========================================================================
LX = 5.0        # domain size (cm)
LY = 5.0
SIGMA = 1.0     # opacity (cm⁻¹) — constant
CV_VOL = 0.1    # heat capacity GJ/(cm³·keV)
T_HOT = 1.0     # keV — hot spot
T_COLD = 0.01   # keV — background
HOT_HALF = 0.125  # half-width of hot region (0.25/2 cm)

__a = 0.01372
__c = 29.98
__ac = __a * __c


# ===========================================================================
# Problem setup and run
# ===========================================================================

def setup_and_run(Nx=100, Ny=100, Ntarget=100000, dt=0.1,
                  output_times=(1.0, 5.0, 10.0), Nmax=500000):
    """Run the hot-spot cooling problem with 2D IMC."""

    # Grid
    edges_x = np.linspace(0, LX, Nx + 1)
    edges_y = np.linspace(0, LY, Ny + 1)
    x_centers = 0.5 * (edges_x[:-1] + edges_x[1:])
    y_centers = 0.5 * (edges_y[:-1] + edges_y[1:])

    # Initial temperature fields
    cx, cy = LX / 2, LY / 2
    T_init = np.full((Nx, Ny), T_COLD)
    for i in range(Nx):
        for j in range(Ny):
            if (abs(x_centers[i] - cx) <= HOT_HALF and
                    abs(y_centers[j] - cy) <= HOT_HALF):
                T_init[i, j] = T_HOT

    # Radiation in equilibrium with material
    Tr_init = T_init.copy()

    # Material functions
    def sigma_a_func(T):
        return SIGMA + 0.0 * T  # constant

    def cv_func(T):
        return CV_VOL + 0.0 * T  # constant

    def eos(T):
        return CV_VOL * T

    def inv_eos(e):
        return e / CV_VOL

    # No external source, no boundary emission
    source = np.zeros((Nx, Ny))
    T_boundary = (0.0, 0.0, 0.0, 0.0)  # vacuum (no incoming)

    # Reflecting = False on all faces (vacuum)
    reflect = (False, False, False, False)

    output_times = np.array(sorted(output_times))
    tfinal = output_times[-1]

    print(f"Hot-Spot Cooling IMC: {Nx}×{Ny}, Ntarget={Ntarget}, dt={dt}")
    print(f"  σ = {SIGMA} cm⁻¹, c_v = {CV_VOL} GJ/(cm³·keV)")
    print(f"  Hot spot: T={T_HOT} keV in [{cx-HOT_HALF:.3f},{cx+HOT_HALF:.3f}]²")
    print(f"  Background: T={T_COLD} keV")
    print(f"  Output times: {output_times}")
    print(f"  Final time: {tfinal} ns, dt={dt} ns")

    wall_start = timer_mod.perf_counter()

    times, Tr_hist, T_hist = imc2d.run_simulation(
        Ntarget=Ntarget,
        Nboundary=0,
        Nsource=0,
        Nmax=Nmax,
        Tinit=T_init,
        Tr_init=Tr_init,
        T_boundary=T_boundary,
        dt=dt,
        edges1=edges_x,
        edges2=edges_y,
        sigma_a_func=sigma_a_func,
        eos=eos,
        inv_eos=inv_eos,
        cv=cv_func,
        source=source,
        final_time=tfinal,
        reflect=reflect,
        output_freq=1,
        theta=1.0,
        geometry="xy",
    )

    wall_time = timer_mod.perf_counter() - wall_start
    print(f"\n  Wall time: {wall_time:.1f} s, steps: {len(times)-1}")

    # Extract snapshots at output times
    solutions = {}
    for t_target in output_times:
        idx = np.argmin(np.abs(times - t_target))
        solutions[t_target] = {
            'T': T_hist[idx],
            'Tr': Tr_hist[idx],
            't_actual': times[idx],
        }
        print(f"  t={times[idx]:.2f} ns: T_max={T_hist[idx].max():.4f}, "
              f"Tr_max={Tr_hist[idx].max():.4f} keV")

    # Fiducial point histories
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

    fid_data = {}
    for label, (i, j) in fid_indices.items():
        fid_data[label] = {
            'T_mat': np.array([T_hist[k][i, j] for k in range(len(T_hist))]),
            'T_rad': np.array([Tr_hist[k][i, j] for k in range(len(Tr_hist))]),
        }

    return {
        'solutions': solutions, 'x_centers': x_centers, 'y_centers': y_centers,
        'x_faces': edges_x, 'y_faces': edges_y,
        'Nx': Nx, 'Ny': Ny, 'ts': times,
        'fid_data': fid_data, 'fid_indices': fid_indices,
        'output_times': output_times,
        'wall_time': wall_time,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_snapshot(results, t_target, savefile_prefix='hot_spot_imc'):
    """Plot material T and radiation T at a given time."""
    sol = results['solutions'][t_target]
    x_f = results['x_faces']
    y_f = results['y_faces']

    T_cell = sol['T']
    Tr_cell = sol['Tr']

    for field, label, data in [('material', 'T (keV)', T_cell),
                                ('radiation', r'$T_r$ (keV)', Tr_cell)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        vmax = max(data.max(), 0.05)
        im = ax.pcolormesh(x_f, y_f, data.T, shading='flat', cmap='plasma',
                           vmin=0.0, vmax=vmax)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'IMC {field.capitalize()} Temperature, t={sol["t_actual"]:.1f} ns')
        plt.colorbar(im, ax=ax, label=label)
        ax.set_aspect('equal')

        # Mark hot-spot outline
        cx, cy = LX / 2, LY / 2
        rect = plt.Rectangle((cx - HOT_HALF, cy - HOT_HALF),
                              2*HOT_HALF, 2*HOT_HALF,
                              fill=False, edgecolor='cyan', lw=1, ls='--')
        ax.add_patch(rect)

        plt.tight_layout()
        fname = f'{savefile_prefix}_{field}_t{t_target:.0f}ns.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


def plot_fiducial_history(results, savefile_prefix='hot_spot_imc'):
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
        ax.set_title('IMC Hot-Spot Cooling')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3, ls='--')
        ax.set_xlim(0, ts[-1])
        plt.tight_layout()
        fname = f'{savefile_prefix}_history_{field}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot-Spot Cooling IMC')
    parser.add_argument('--Nx', type=int, default=100)
    parser.add_argument('--Ny', type=int, default=100)
    parser.add_argument('--Ntarget', type=int, default=100000,
                        help='Target particles per step')
    parser.add_argument('--Nmax', type=int, default=500000,
                        help='Max particles')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (ns)')
    parser.add_argument('--output-times', type=float, nargs='+',
                        default=[1.0, 5.0, 10.0])
    parser.add_argument('--fiducial-output',
                        default='hot_spot_imc_fiducials.npz',
                        help='NPZ file for every-step fiducial histories')
    args = parser.parse_args()

    results = setup_and_run(
        Nx=args.Nx, Ny=args.Ny,
        Ntarget=args.Ntarget, Nmax=args.Nmax,
        dt=args.dt,
        output_times=tuple(args.output_times),
    )

    save_fiducial_history(results, args.fiducial_output, method='IMC')

    for t in args.output_times:
        if t in results['solutions']:
            plot_snapshot(results, t)
    plot_fiducial_history(results)
