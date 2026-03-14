#!/usr/bin/env python3
"""
Plotting script for marshak_wave_multigroup_powerlaw NPZ output files.

Usage:
    python plot_marshak_multigroup_npz.py <npz_file>
    python plot_marshak_multigroup_npz.py  (looks for *_solutions.npz in cwd)

Expects structured NPZ with keys:
    times        (n_saved,)
    r            (n_cells,)
    energy_edges (n_groups+1,)
    T_mat        (n_saved, n_cells)
    T_rad        (n_saved, n_cells)
    E_r          (n_saved, n_cells)
    phi_groups   (n_saved, n_groups, n_cells)
    E_r_groups   (n_saved, n_groups, n_cells)
"""

import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# path setup — allow running from the problems/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, hide_spines, font

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def group_centers(energy_edges):
    """Geometric mean energy of each group (keV)."""
    return np.sqrt(energy_edges[:-1] * energy_edges[1:])


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    required = ['times', 'r', 'energy_edges', 'T_mat', 'T_rad', 'E_r',
                'phi_groups', 'E_r_groups']
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"NPZ file is missing keys: {missing}\n"
                         "Re-run the simulation to produce a structured output file.")
    return d


# ---------------------------------------------------------------------------
# individual plot functions
# ---------------------------------------------------------------------------

def plot_temperature_linear(r, times, T_mat, T_rad, base, colors):
    """Material and radiation temperature profiles — linear axes."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t in enumerate(times):
        ax.plot(r, T_mat[idx], color=colors[idx], linewidth=2,
                label='material' if idx == 0 else None)
        ax.plot(r, T_rad[idx], color=colors[idx], linewidth=2, linestyle='--',
                label='radiation' if idx == 0 else None)
    # time labels — second pass so colours match without cluttering legend
    for idx, t in enumerate(times):
        ax.plot([], [], color=colors[idx], linewidth=2,
                label=f't = {t:.2f} ns')
    ax.set_xlabel('position (cm)', fontsize=12)
    ax.set_ylabel('temperature (keV)', fontsize=12)
    ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    show(f"{base}_T_mat.pdf", close_after=True)
    print(f"Saved: {base}_T_mat.pdf")


def plot_temperature_log(r, times, T_mat, T_rad, base, colors):
    """Material and radiation temperature profiles — log–log axes."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t in enumerate(times):
        ax.plot(r, T_mat[idx], color=colors[idx], linewidth=2,
                label='material' if idx == 0 else None)
        ax.plot(r, T_rad[idx], color=colors[idx], linewidth=2, linestyle='--',
                label='radiation' if idx == 0 else None)
    for idx, t in enumerate(times):
        ax.plot([], [], color=colors[idx], linewidth=2,
                label=f't = {t:.2f} ns')
    ax.set_xlabel('position (cm)', fontsize=12)
    ax.set_ylabel('temperature (keV)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    show(f"{base}_T_log.pdf", close_after=True)
    print(f"Saved: {base}_T_log.pdf")


def plot_energy_density(r, times, E_r, base, colors):
    """Total radiation energy density profiles."""
    fig, ax = plt.subplots(figsize=(7.5, 5.25))
    for idx, t in enumerate(times):
        ax.semilogy(r, E_r[idx], color=colors[idx], linewidth=2,
                    label=f't = {t:.2f} ns')
    ax.set_xlabel('position (cm)', fontsize=12)
    ax.set_ylabel(r'radiation energy density (GJ/cm$^3$)', fontsize=12)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    show(f"{base}_E_rad.pdf", close_after=True)
    print(f"Saved: {base}_E_rad.pdf")


def plot_group_energy_spatial(r, times, E_r_groups, energy_edges, base, colors):
    """
    Group radiation energy density vs position.

    One panel per snapshot; each line is one energy group, coloured by
    group index with a shared colorbar.
    """
    n_saved, n_groups, _ = E_r_groups.shape
    E_centers = group_centers(energy_edges)
    cmap = cm.get_cmap('plasma', n_groups)

    fig, axes = plt.subplots(1, n_saved, figsize=(5 * n_saved, 4.5), sharey=True)
    if n_saved == 1:
        axes = [axes]

    for idx, (ax, t) in enumerate(zip(axes, times)):
        for g in range(n_groups):
            ax.semilogy(r, E_r_groups[idx, g, :], color=cmap(g), linewidth=1.5)
        ax.set_xlabel('position (cm)', fontsize=12)
        if idx == 0:
            ax.set_ylabel(r'$E_{r,g}$ (GJ/cm$^3$)', fontsize=12)
        ax.set_title(f't = {t:.2f} ns', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

    # Shared colorbar for energy groups
    sm = plt.cm.ScalarMappable(cmap='plasma',
                               norm=plt.Normalize(vmin=E_centers[0], vmax=E_centers[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label('group centre energy (keV)', fontsize=11)

    plt.tight_layout()
    show(f"{base}_Er_groups_spatial.pdf", close_after=True)
    print(f"Saved: {base}_Er_groups_spatial.pdf")


def plot_spectra_at_positions(r, times, E_r_groups, energy_edges, base, colors,
                               n_positions=5):
    """
    Energy spectrum (E_r_g vs E_g) at several spatial positions for each snapshot.

    Positions are chosen as evenly spaced fractions through the domain.
    """
    E_centers = group_centers(energy_edges)
    n_saved, n_groups, n_cells = E_r_groups.shape

    pos_indices = np.linspace(0, n_cells - 1, n_positions, dtype=int)
    pos_colors = [plt.cm.viridis(i / (n_positions - 1)) for i in range(n_positions)]

    fig, axes = plt.subplots(1, n_saved, figsize=(5.5 * n_saved, 4.5), sharey=True)
    if n_saved == 1:
        axes = [axes]

    for idx, (ax, t) in enumerate(zip(axes, times)):
        for pi, pos_idx in enumerate(pos_indices):
            spectrum = E_r_groups[idx, :, pos_idx]
            ax.loglog(E_centers, spectrum, color=pos_colors[pi], linewidth=2,
                      marker='o', markersize=4,
                      label=f'r = {r[pos_idx]:.2f} cm')
        ax.set_xlabel('photon energy (keV)', fontsize=12)
        if idx == 0:
            ax.set_ylabel(r'$E_{r,g}$ (GJ/cm$^3$)', fontsize=12)
        ax.set_title(f't = {t:.2f} ns', fontsize=11)
        ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    show(f"{base}_spectra.pdf", close_after=True)
    print(f"Saved: {base}_spectra.pdf")


def plot_group_fractions_spatial(r, times, E_r_groups, energy_edges, base, n_groups_show=None):
    """
    Fractional contribution of each group to the total E_r as a stacked area.

    Coloured by group with a shared plasma colorbar.
    """
    n_saved, n_groups, n_cells = E_r_groups.shape
    if n_groups_show is None:
        n_groups_show = n_groups
    E_centers = group_centers(energy_edges)
    cmap = cm.get_cmap('plasma', n_groups)

    fig, axes = plt.subplots(1, n_saved, figsize=(5.5 * n_saved, 4.5), sharey=True)
    if n_saved == 1:
        axes = [axes]

    for idx, (ax, t) in enumerate(zip(axes, times)):
        E_total = E_r_groups[idx].sum(axis=0)
        fractions = np.zeros((n_groups_show, n_cells))
        for g in range(n_groups_show):
            fractions[g] = E_r_groups[idx, g, :] / np.maximum(E_total, 1e-300)

        cumulative = np.zeros(n_cells)
        for g in range(n_groups_show):
            ax.fill_between(r, cumulative, cumulative + fractions[g],
                            color=cmap(g), alpha=0.85, linewidth=0)
            cumulative += fractions[g]

        ax.set_xlim(r[0], r[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel('position (cm)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('fractional group energy', fontsize=12)
        ax.set_title(f't = {t:.2f} ns', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    sm = plt.cm.ScalarMappable(cmap='plasma',
                               norm=plt.Normalize(vmin=E_centers[0], vmax=E_centers[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label('group centre energy (keV)', fontsize=11)

    plt.tight_layout()
    show(f"{base}_group_fractions.pdf", close_after=True)
    print(f"Saved: {base}_group_fractions.pdf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # --- resolve input file ---
    if len(sys.argv) >= 2:
        npz_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob("*_solutions.npz"))
        if not candidates:
            candidates = sorted(glob.glob("marshak_wave_multigroup_powerlaw_*.npz"))
        if not candidates:
            print("No NPZ file specified and none found in current directory.")
            sys.exit(1)
        npz_path = candidates[-1]
        print(f"No file specified — using: {npz_path}")

    print(f"Loading {npz_path} ...")
    d = load_npz(npz_path)

    times        = d['times']
    r            = d['r']
    energy_edges = d['energy_edges']
    T_mat        = d['T_mat']
    T_rad        = d['T_rad']
    E_r          = d['E_r']
    phi_groups   = d['phi_groups']
    E_r_groups   = d['E_r_groups']

    n_saved, n_groups, n_cells = E_r_groups.shape
    print(f"  snapshots: {n_saved},  groups: {n_groups},  cells: {n_cells}")
    print(f"  times: {times}")

    # base name for output files (strip directory and .npz)
    base = os.path.splitext(os.path.basename(npz_path))[0]

    # Colour cycle — same as original powerlaw script
    _colors_cycle = ['blue', 'red', 'green', 'orange', 'purple',
                     'cyan', 'magenta', 'brown', 'olive', 'teal']
    colors = [_colors_cycle[i % len(_colors_cycle)] for i in range(n_saved)]

    # ---- produce all plots ----
    plot_temperature_linear(r, times, T_mat, T_rad, base, colors)
    plot_temperature_log(r, times, T_mat, T_rad, base, colors)
    plot_energy_density(r, times, E_r, base, colors)
    plot_group_energy_spatial(r, times, E_r_groups, energy_edges, base, colors)
    plot_spectra_at_positions(r, times, E_r_groups, energy_edges, base, colors)
    plot_group_fractions_spatial(r, times, E_r_groups, energy_edges, base)

    print("\nAll plots saved.")


if __name__ == '__main__':
    main()
