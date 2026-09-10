#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Plot Zel'dovich wave S_N results — publication figures.

Produces two figures matching the EqDiffusion format:
  1. Radial lineout: T_rad vs ρ at t = 0.3, 0.5, 1.0 ns
  2. Quadrant comparison: numerical (right) vs self-similar (left)

Usage:
    python plot_zeldovich_sn.py path/to/zeldovich_2d_*.npz
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# plotfuncs path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EqDiffusion', 'utils'))
from plotfuncs import hide_spines, show

# S_N self-similar solution
sys.path.insert(0, os.path.dirname(__file__))
from zeldovich_wave_2d_converge import zeldovich_self_similar, ac

# Font (matching EqDiffusion style)
try:
    font = fm.FontProperties(family='Gill Sans',
                             fname='/Library/Fonts/GillSans.ttc', size=12)
except Exception:
    font = fm.FontProperties(size=12)


# ===========================================================================
# Data loading
# ===========================================================================

def load_result(npzfile):
    """Load npz and reconstruct useful arrays."""
    d = np.load(npzfile, allow_pickle=True)
    result = {}
    result['ts'] = d['ts']
    result['x_centers'] = d['x_centers']
    result['y_centers'] = d['y_centers']
    result['x_faces'] = d['x_faces']
    result['y_faces'] = d['y_faces']
    result['Lx'] = float(d['Lx'])

    # Handle both old format (snapshots) and new format (full arrays)
    if 'phis' in d:
        result['phis'] = d['phis']         # (n_steps+1, Ix, Iy, 4)
        result['Ts'] = d['Ts']
    elif 'phis_snapshots' in d:
        result['phis'] = d['phis_snapshots']
        result['Ts'] = d['Ts_snapshots']
        # Override ts with just the snapshot times
        result['ts'] = d['ts'][d['phis_snap_indices']]
    else:
        raise KeyError("npz has neither 'phis' nor 'phis_snapshots'")

    return result


def get_snapshot(result, t_target):
    """Get the snapshot closest to t_target. Returns (phi_cell, t_actual)."""
    ts = result['ts']
    idx = np.argmin(np.abs(ts - t_target))
    phi_snap = result['phis'][idx]       # (Ix, Iy, 4)
    phi_cell = np.mean(phi_snap, axis=2)  # cell-averaged
    t_actual = ts[idx]
    return phi_cell, t_actual


def T_rad_from_phi(phi_cell):
    """Radiation temperature from scalar flux."""
    return (np.maximum(phi_cell, 0.0) / ac) ** 0.25


# ===========================================================================
# Figure 1: Radial lineout
# ===========================================================================

def plot_lineout(result, times=[0.3, 0.5, 1.0], savefile='zeldovich_sn_lineout.pdf'):
    """Radial T_rad profiles vs self-similar at multiple times."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8 / 1.518))

    colors = ['blue', 'green', 'red']
    x_c = result['x_centers']
    y_c = result['y_centers']
    Lx = result['Lx']

    for i, t_target in enumerate(times):
        color = colors[i % len(colors)]

        # Numerical: extract along x-axis (y ≈ 0, j = 0)
        phi_cell, t_act = get_snapshot(result, t_target)
        T_rad = T_rad_from_phi(phi_cell)
        T_xaxis = T_rad[:, 0]  # along x-axis (y = y_centers[0] ≈ 0)

        ax.plot(x_c, T_xaxis, color=color, linewidth=2, linestyle='-',
                marker='o', markersize=3, markevery=5,
                label=f't = {t_act:.2f} ns')

        # Self-similar
        r_fine = np.linspace(0.001, Lx, 300)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            T_ss, R_front = zeldovich_self_similar(r_fine, t_act, N=2)
        ax.plot(r_fine, T_ss, color=color, linewidth=1.5, linestyle='--',
                alpha=0.7)
        ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)

    ax.set_xlabel('Radial Distance (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, Lx)

    plt.tight_layout()
    show(savefile)
    print(f"  Lineout saved: {savefile}")


# ===========================================================================
# Figure 2: Quadrant comparison (numerical right, self-similar left)
# ===========================================================================

def plot_quadrant(result, times=[0.3, 1.0], savefile='zeldovich_sn_quadrant.png'):
    """Split-view: left = self-similar, right = numerical.

    Upper half uses early time, lower half uses late time.
    """
    x_c = result['x_centers']
    y_c = result['y_centers']
    Ix = len(x_c)
    Iy = len(y_c)

    # Get numerical snapshots
    phi_early, t_early = get_snapshot(result, times[0])
    phi_late, t_late = get_snapshot(result, times[1])
    T_num_early = T_rad_from_phi(phi_early)
    T_num_late = T_rad_from_phi(phi_late)

    # Compute self-similar on same grid
    T_ss_early = np.zeros((Ix, Iy))
    T_ss_late = np.zeros((Ix, Iy))
    for i in range(Ix):
        for j in range(Iy):
            rho = np.sqrt(x_c[i]**2 + y_c[j]**2)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                T_val, _ = zeldovich_self_similar(np.array([rho]), t_early, N=2)
                T_ss_early[i, j] = T_val[0]
                T_val, _ = zeldovich_self_similar(np.array([rho]), t_late, N=2)
                T_ss_late[i, j] = T_val[0]

    # Build composite: mirror about x=0.
    # Domain is [0, Lx] × [0, Ly] (first quadrant). We'll create a full
    # [-Lx, Lx] × [-Ly, Ly] view where:
    #   x > 0: Numerical
    #   x < 0: Self-similar
    #   y > 0: early time
    #   y < 0: late time

    Lx = result['Lx']
    x_full = np.concatenate([-x_c[::-1], x_c])
    y_full = np.concatenate([-y_c[::-1], y_c])
    x_faces_full = np.concatenate([-result['x_faces'][::-1], result['x_faces'][1:]])
    y_faces_full = np.concatenate([-result['y_faces'][::-1], result['y_faces'][1:]])

    T_composite = np.zeros((2 * Ix, 2 * Iy))

    # Top-right (x>0, y>0): Numerical early
    T_composite[Ix:, Iy:] = T_num_early
    # Top-left (x<0, y>0): Self-similar early (mirrored)
    T_composite[:Ix, Iy:] = T_ss_early[::-1, :]
    # Bottom-right (x>0, y<0): Numerical late
    T_composite[Ix:, :Iy] = T_num_late[:, ::-1]
    # Bottom-left (x<0, y<0): Self-similar late (mirrored)
    T_composite[:Ix, :Iy] = T_ss_late[::-1, ::-1]

    # Global color limits
    T_max = max(T_num_early.max(), T_num_late.max(),
                T_ss_early.max(), T_ss_late.max())

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    im = ax.pcolormesh(x_faces_full, y_faces_full, T_composite.T,
                       shading='flat', cmap='plasma', vmin=0, vmax=T_max)

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('z (cm)')
    ax.set_aspect('equal')

    # Separator lines
    ax.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)

    # Front radius circles
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        _, R_early = zeldovich_self_similar(np.array([0.0]), t_early, N=2)
        _, R_late = zeldovich_self_similar(np.array([0.0]), t_late, N=2)
    circle1 = plt.Circle((0, 0), R_early, fill=False, color='white',
                          linestyle='--', alpha=0.3, linewidth=1)
    circle2 = plt.Circle((0, 0), R_late, fill=False, color='white',
                          linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Labels
    text_props = dict(fontsize=11, fontweight='bold', ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontproperties=font)
    ax.text(-Lx * 0.7, -Lx * 0.85, "Self-Similar", **text_props)
    ax.text(Lx * 0.7, Lx * 0.85, "Numerical", **text_props)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (keV)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    show(savefile, cbar_ax=cbar.ax)
    print(f"  Quadrant plot saved: {savefile}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default path
        npzfile = os.path.join(os.path.dirname(__file__), '..', '..',
                               'results', 'ZeldovichWave', 'S4_LS_100_1ns',
                               'zeldovich_2d_I40_N4_t1.0.npz')
    else:
        npzfile = sys.argv[1]

    if not os.path.exists(npzfile):
        print(f"Error: {npzfile} not found")
        sys.exit(1)

    print(f"Loading: {npzfile}")
    result = load_result(npzfile)
    print(f"  Time range: {result['ts'][0]:.4f} → {result['ts'][-1]:.4f} ns  ({len(result['ts'])} steps)")
    print(f"  Domain: [0, {result['Lx']:.4f}] cm, grid {len(result['x_centers'])}×{len(result['y_centers'])}")

    outdir = os.path.dirname(npzfile) or '.'

    plot_lineout(result, times=[0.3, 0.5, 1.0],
                 savefile=os.path.join(outdir, 'zeldovich_sn_lineout.pdf'))

    plot_quadrant(result, times=[0.2, 1.0],
                  savefile=os.path.join(outdir, 'zeldovich_sn_quadrant.png'))
