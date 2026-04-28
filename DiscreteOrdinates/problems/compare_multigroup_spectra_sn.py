#!/usr/bin/env python3
"""
Compare multigroup S_N radiation spectra, with optional overlay of diffusion
and/or IMC results.

Each S_N NPZ file must contain the arrays saved by
test_marshak_wave_multigroup_powerlaw.py:
    times, r, energy_edges, T_mat, T_rad, phi_groups, E_r_groups, E_r

Diffusion NPZ files (marshak_wave_multigroup_powerlaw_*g*.npz) and IMC NPZ
files (marshak_wave_multigroup_powerlaw_imc_*g*.npz) share the same layout
and are overlaid when supplied via --diff-files / --imc-files.

Usage examples
--------------
# Auto-detect S_N files; plot spectrum at last time, first position
python compare_multigroup_spectra_sn.py

# Specific S_N files, add diffusion and IMC references
python compare_multigroup_spectra_sn.py \\
    marshak_wave_powerlaw_sn_10g_timeBC.npz \\
    marshak_wave_powerlaw_sn_50g_timeBC.npz \\
    --diff-files ../../nonEquilibriumDiffusion/problems/marshak_wave_multigroup_powerlaw_10g_precond_timeBC.npz \\
    --imc-files ../../MG_IMC/marshak_wave_multigroup_powerlaw_imc_10g_timeBC.npz \\
    --time 5.0 --r-position 1.5

# Temperature profiles (T_mat and T_rad vs x)
python compare_multigroup_spectra_sn.py --plot-T
"""

import re
import sys
import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, font


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_centers(energy_edges):
    return np.sqrt(energy_edges[:-1] * energy_edges[1:])


def group_widths(energy_edges):
    return energy_edges[1:] - energy_edges[:-1]


def nearest_index(arr, value):
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


def _ngroups_from_name(path):
    m = re.search(r'_(\d+)g[_.]', os.path.basename(path))
    return int(m.group(1)) if m else 0


def label_from_filename(path, prefix=''):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'_(\d+)g[_.]?', os.path.basename(path))
    ng = m.group(1) if m else '?'
    return f"{prefix}{ng}g"


def load_npz(path, required_keys=('times', 'r', 'energy_edges', 'E_r_groups')):
    d = np.load(path, allow_pickle=False)
    missing = [k for k in required_keys if k not in d]
    if missing:
        raise ValueError(
            f"'{os.path.basename(path)}' is missing keys: {missing}.")
    return d


# ---------------------------------------------------------------------------
# Spectrum plot: E_{r,g}/dE vs photon energy
# ---------------------------------------------------------------------------

def plot_spectra(sn_paths, diff_paths, imc_paths,
                 target_time, target_r, outbase=None):
    """Compare E_{r,g}/dE spectra at a chosen (t, r) for S_N + optional refs."""

    def _load_sorted(paths, label_prefix):
        out = []
        for p in paths:
            try:
                d = load_npz(p)
                out.append((p, d, label_prefix))
            except ValueError as e:
                print(f"  SKIPPING {os.path.basename(p)}: {e}")
        out.sort(key=lambda x: _ngroups_from_name(x[0]))
        return out

    sn_data   = _load_sorted(sn_paths,   'S_N ')
    diff_data = _load_sorted(diff_paths, 'Diff ')
    imc_data  = _load_sorted(imc_paths,  'IMC ')

    all_data = sn_data + diff_data + imc_data
    if not all_data:
        print("No usable NPZ files found.")
        return

    # Choose query values from first S_N dataset (or first overall)
    ref_d = all_data[0][1]
    t_query = target_time if target_time is not None else float(ref_d['times'][-1])
    r_query = target_r   if target_r   is not None else float(ref_d['r'][0])

    print(f"\nSpectrum plot at t ≈ {t_query:.3f} ns, r ≈ {r_query:.3f} cm")

    # Style: S_N=solid, Diff=dashed, IMC=dotted
    ls_map      = {'S_N ': '-', 'Diff ': '--', 'IMC ': ':'}
    colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Assign a unique color per group-count per method
    color_index = {}
    col_counter = 0
    for _, _, prefix in all_data:
        if prefix not in color_index:
            color_index[prefix] = col_counter
            col_counter += 1

    # Assign one color per entry grouped by number of groups
    ng_colors = {}
    ctr = 0
    for p, d, prefix in all_data:
        ng = _ngroups_from_name(p)
        if ng not in ng_colors:
            ng_colors[ng] = colors[ctr % len(colors)]
            ctr += 1

    fig, ax = plt.subplots(figsize=(7, 5))

    for path, d, prefix in all_data:
        t_idx  = nearest_index(d['times'], t_query)
        r_idx  = nearest_index(d['r'],     r_query)
        ee     = d['energy_edges']
        Ec     = group_centers(ee)
        dE     = group_widths(ee)
        spec   = d['E_r_groups'][t_idx, :, r_idx] / dE
        ng     = _ngroups_from_name(path)
        col    = ng_colors[ng]
        ls     = ls_map.get(prefix, '-')
        t_act  = float(d['times'][t_idx])
        r_act  = float(d['r'][r_idx])

        lbl = f"{prefix.strip()} {ng}g  (t={t_act:.2f} ns, r={r_act:.2f} cm)"
        ax.step(ee, np.append(spec, spec[-1]),
                where='post', color=col, ls=ls, lw=2, label=lbl)
        ax.plot(Ec, spec, 'o', color=col, ms=4, alpha=0.5)

    ax.set_xlabel('photon energy (keV)', fontsize=12)
    ax.set_ylabel(r'$E_{r,g}/\Delta E$  (GJ cm$^{-3}$ keV$^{-1}$)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=9)
    ax.grid(True, which='both', alpha=0.3, ls='--')
    fig.tight_layout()

    if outbase is None:
        outbase = 'multigroup_sn_spectra'
    t_tag = f"_t{t_query:.2f}ns".replace('.', 'p')
    r_tag = f"_r{r_query:.2f}cm".replace('.', 'p')
    outname = f"{outbase}{t_tag}{r_tag}.pdf"
    show(outname, close_after=True)
    print(f"Saved: {outname}")


# ---------------------------------------------------------------------------
# Temperature profile plots: T_mat and T_rad vs x at each saved time
# ---------------------------------------------------------------------------

def plot_temperatures(sn_paths, diff_paths, imc_paths, outbase=None):
    """Compare T_mat and T_rad spatial profiles at all saved times."""

    def _load_sorted(paths, label_prefix):
        out = []
        for p in paths:
            try:
                d = load_npz(p, required_keys=('times', 'r', 'T_mat', 'T_rad'))
                out.append((p, d, label_prefix))
            except ValueError as e:
                print(f"  SKIPPING {os.path.basename(p)}: {e}")
        out.sort(key=lambda x: _ngroups_from_name(x[0]))
        return out

    sn_data   = _load_sorted(sn_paths,   'S_N ')
    diff_data = _load_sorted(diff_paths, 'Diff ')
    imc_data  = _load_sorted(imc_paths,  'IMC ')

    all_data = sn_data + diff_data + imc_data
    if not all_data:
        print("No usable NPZ files found.")
        return

    # Collect all unique times across all datasets
    all_times = set()
    for _, d, _ in all_data:
        for t in d['times']:
            all_times.add(round(float(t), 4))
    all_times = sorted(all_times)

    t_colors = plt.cm.viridis(np.linspace(0, 1, len(all_times)))
    ls_map   = {'S_N ': '-', 'Diff ': '--', 'IMC ': ':'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plotted_labels = set()
    for path, d, prefix in all_data:
        ls   = ls_map.get(prefix, '-')
        ng   = _ngroups_from_name(path)
        for ti, t_val in enumerate(d['times']):
            t_col = t_colors[nearest_index(all_times, float(t_val))]
            lbl_mat = f"{prefix.strip()} {ng}g  mat  t={t_val:.1f} ns"
            lbl_rad = f"{prefix.strip()} {ng}g  rad  t={t_val:.1f} ns"

            k_mat = (prefix, ng, 'mat', round(float(t_val), 2))
            k_rad = (prefix, ng, 'rad', round(float(t_val), 2))

            ax1.plot(d['r'], d['T_mat'][ti], ls=ls, color=t_col, lw=1.8,
                     label=lbl_mat if k_mat not in plotted_labels else '_')
            ax2.plot(d['r'], d['T_mat'][ti], ls=ls, color=t_col, lw=1.8,
                     label='_')
            ax1.plot(d['r'], d['T_rad'][ti], ls=ls, color=t_col, lw=1.0,
                     alpha=0.6,
                     label=lbl_rad if k_rad not in plotted_labels else '_')
            ax2.plot(d['r'], d['T_rad'][ti], ls=ls, color=t_col, lw=1.0,
                     alpha=0.6, label='_')
            plotted_labels.add(k_mat)
            plotted_labels.add(k_rad)

    for ax in (ax1, ax2):
        ax.set_xlabel('x (cm)', fontsize=12)
        ax.set_ylabel('Temperature (keV)', fontsize=12)
        ax.grid(True, alpha=0.3)
    ax1.set_title('Linear scale', fontsize=12)
    ax2.set_title('Log scale', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Legend: solid=mat, dashed-ish=rad; entries per file
    ax1.legend(prop=font, facecolor='white', edgecolor='none', fontsize=8,
               ncol=2, loc='upper right')

    fig.suptitle('Multigroup Marshak Wave — Temperature profiles', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()

    if outbase is None:
        outbase = 'multigroup_sn_temperatures'
    outname = f"{outbase}.pdf"
    show(outname, close_after=True)
    print(f"Saved: {outname}")


# ---------------------------------------------------------------------------
# Spatial sweep: spectrum at several r positions for one file
# ---------------------------------------------------------------------------

def plot_spectra_spatial_sweep(sn_paths, diff_paths, imc_paths,
                                target_time, r_positions, outbase=None):
    """For each dataset, plot spectra at several r positions (one panel each)."""

    def _load_sorted(paths, lp):
        out = []
        for p in paths:
            try:
                out.append((p, load_npz(p), lp))
            except ValueError as e:
                print(f"  SKIPPING {os.path.basename(p)}: {e}")
        out.sort(key=lambda x: _ngroups_from_name(x[0]))
        return out

    all_data = (_load_sorted(sn_paths,   'S_N ')
                + _load_sorted(diff_paths, 'Diff ')
                + _load_sorted(imc_paths,  'IMC '))
    if not all_data:
        print("No usable NPZ files found.")
        return

    ref_d   = all_data[0][1]
    t_query = target_time if target_time is not None else float(ref_d['times'][-1])

    n_files = len(all_data)
    fig, axes = plt.subplots(1, n_files, figsize=(6.5 * n_files, 5),
                              squeeze=False, sharey=True)
    pos_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(r_positions)))

    for col, (path, d, prefix) in enumerate(all_data):
        ax = axes[0, col]
        t_idx = nearest_index(d['times'], t_query)
        ee = d['energy_edges']
        Ec = group_centers(ee)
        dE = group_widths(ee)

        for pi, r_val in enumerate(r_positions):
            r_idx  = nearest_index(d['r'], r_val)
            r_act  = float(d['r'][r_idx])
            spec   = d['E_r_groups'][t_idx, :, r_idx] / dE
            ax.step(ee, np.append(spec, spec[-1]),
                    where='post', color=pos_colors[pi], lw=2,
                    label=f'r = {r_act:.2f} cm')
            ax.plot(Ec, spec, 'o', color=pos_colors[pi], ms=4, alpha=0.5)

        ng = _ngroups_from_name(path)
        t_act = float(d['times'][t_idx])
        ax.set_title(f"{prefix.strip()} {ng}g  (t={t_act:.2f} ns)", fontsize=11)
        ax.set_xlabel('photon energy (keV)', fontsize=11)
        if col == 0:
            ax.set_ylabel(r'$E_{r,g}/\Delta E$  (GJ cm$^{-3}$ keV$^{-1}$)', fontsize=11)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=8)
        ax.grid(True, which='both', alpha=0.3, ls='--')

    fig.tight_layout()

    if outbase is None:
        outbase = 'multigroup_sn_spectra_spatial'
    t_tag = f"_t{t_query:.2f}ns".replace('.', 'p')
    outname = f"{outbase}{t_tag}_spatial.pdf"
    show(outname, close_after=True)
    print(f"Saved: {outname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare multigroup S_N spectra/temperatures; '
                    'overlay diffusion and/or IMC.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('files', nargs='*',
                   help='S_N NPZ files. Auto-detected if omitted.')
    p.add_argument('--diff-files', nargs='*', default=[],
                   help='Diffusion NPZ files to overlay.')
    p.add_argument('--imc-files', nargs='*', default=[],
                   help='IMC NPZ files to overlay.')
    p.add_argument('--time', '-t', type=float, default=None,
                   help='Target time (ns).')
    p.add_argument('--r-position', '-r', type=float, default=None,
                   help='Target spatial position (cm).')
    p.add_argument('--plot-T', action='store_true',
                   help='Plot T_mat / T_rad profiles instead of spectra.')
    p.add_argument('--spatial-sweep', action='store_true',
                   help='Plot spectra at multiple r positions (one panel per file).')
    p.add_argument('--r-sweep-positions', type=str, default=None,
                   help='Comma-separated r values (cm) for --spatial-sweep.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Prefix for output PDF filenames.')
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect S_N files
    if args.files:
        sn_paths = args.files
    else:
        sn_paths = sorted(glob.glob('marshak_wave_powerlaw_sn_*g*.npz'))
        if not sn_paths:
            print('No S_N NPZ files found. Specify files explicitly or run '
                  'from the directory containing marshak_wave_powerlaw_sn_*g*.npz files.')
            sys.exit(1)
        print(f"Auto-detected {len(sn_paths)} S_N file(s):")
        for p in sn_paths:
            print(f"  {p}")

    diff_paths = args.diff_files or []
    imc_paths  = args.imc_files  or []

    if args.plot_T:
        plot_temperatures(sn_paths, diff_paths, imc_paths,
                          outbase=args.output_base)
    elif args.spatial_sweep:
        if args.r_sweep_positions:
            r_pos = [float(x.strip()) for x in args.r_sweep_positions.split(',')]
        else:
            try:
                d0 = load_npz(sn_paths[0])
                r_pos = list(np.linspace(float(d0['r'][0]), float(d0['r'][-1]), 5))
            except Exception:
                r_pos = [0.5, 1.5, 3.0, 5.0, 6.5]
        plot_spectra_spatial_sweep(sn_paths, diff_paths, imc_paths,
                                   args.time, r_pos,
                                   outbase=args.output_base)
    else:
        plot_spectra(sn_paths, diff_paths, imc_paths,
                     target_time=args.time,
                     target_r=args.r_position,
                     outbase=args.output_base)


if __name__ == '__main__':
    main()
