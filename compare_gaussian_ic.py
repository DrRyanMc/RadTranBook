#!/usr/bin/env python3
"""
Gaussian IC Comparison: S_N Transport vs. Multigroup Diffusion

Loads the NPZ output files from:
  - DiscreteOrdinates/problems/test_gaussian_ic_sn.py
  - nonEquilibriumDiffusion/problems/gaussian_ic_diffusion.py

and produces comparison plots at each output time.

Usage:
    python compare_gaussian_ic.py
    python compare_gaussian_ic.py --sn  gaussian_ic_sn_10g.npz \\
                                   --diff gaussian_ic_diffusion_10g.npz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Physical / IC constants (must match problem files) ──────────────────
A_RAD  = 0.01372   # GJ/(cm³ keV⁴)
RHO    = 0.01      # g/cm³
CV_MASS = 0.05     # GJ/(g·keV)
CV_VOL = RHO * CV_MASS  # 5e-4 GJ/(cm³·keV)

T_MAX   = 0.2      # keV
X_0     = 3.5      # cm
SIGMA_G = 0.8      # cm
T_FLOOR = 0.001    # keV


def _gaussian_T0(r):
    """Analytic t=0 temperature profile on grid r."""
    return np.maximum(T_FLOOR, T_MAX * np.exp(-0.5 * ((r - X_0) / SIGMA_G)**2))


# ── helpers ──────────────────────────────────────────────────────────────

def load(path):
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    return dict(d)


def find_closest_time(times, t_target):
    return int(np.argmin(np.abs(np.asarray(times) - t_target)))


# ── plotting ─────────────────────────────────────────────────────────────

def compare_snapshot(ax_T, ax_Tr, ax_Er, sn, diff, tidx_sn, tidx_diff, t_target, label_extra=''):
    """Plot one time-snapshot on provided axes."""
    r_sn   = sn['r']
    r_diff = diff['r']

    T_sn   = sn['T_mat'][tidx_sn]
    Tr_sn  = sn['T_rad'][tidx_sn]
    Er_sn  = sn['E_r'][tidx_sn]

    T_diff  = diff['T_mat'][tidx_diff]
    Tr_diff = diff['T_rad'][tidx_diff]
    Er_diff = diff['E_r'][tidx_diff]

    t_sn   = float(sn['times'][tidx_sn])
    t_diff = float(diff['times'][tidx_diff])

    lbl_sn   = f'S$_N$   t={t_sn:.3f} ns'
    lbl_diff = f'Diff  t={t_diff:.3f} ns'

    ax_T.plot(r_sn,   T_sn,   '-',  color='C0', lw=1.8, label=lbl_sn)
    ax_T.plot(r_diff, T_diff, '--', color='C1', lw=1.8, label=lbl_diff)
    ax_T.set_ylabel('$T_{mat}$ (keV)')
    ax_T.legend(fontsize=8)
    ax_T.grid(True, alpha=0.3)

    ax_Tr.plot(r_sn,   Tr_sn,   '-',  color='C0', lw=1.8, label='$T_{rad}$ S$_N$')
    ax_Tr.plot(r_diff, Tr_diff, '--', color='C1', lw=1.8, label='$T_{rad}$ Diff')
    ax_Tr.set_ylabel('$T_{rad}$ (keV)')
    ax_Tr.legend(fontsize=8)
    ax_Tr.grid(True, alpha=0.3)

    ax_Er.semilogy(r_sn,   Er_sn,   '-',  color='C0', lw=1.8, label='$E_r$ S$_N$')
    ax_Er.semilogy(r_diff, Er_diff, '--', color='C1', lw=1.8, label='$E_r$ Diff')
    ax_Er.set_ylabel('$E_r$ (GJ/cm³)')
    ax_Er.set_xlabel('x (cm)')
    ax_Er.legend(fontsize=8)
    ax_Er.grid(True, which='both', alpha=0.3)

    for ax in (ax_T, ax_Tr, ax_Er):
        ax.set_xlim(float(r_sn[0]), float(r_sn[-1]))


def plot_spectrum(ax, sn, diff, tidx_sn, tidx_diff, x_target=3.5):
    """Per-group E_r spectrum at x_target at a single snapshot."""
    # Find closest cell
    i_sn   = int(np.argmin(np.abs(sn['r']   - x_target)))
    i_diff = int(np.argmin(np.abs(diff['r']  - x_target)))

    edges = sn['energy_edges']
    G     = len(edges) - 1
    E_mid = np.sqrt(edges[:-1] * edges[1:])  # geometric midpoint

    Er_sn_g   = sn['E_r_groups'][tidx_sn,   :, i_sn]
    Er_diff_g = diff['E_r_groups'][tidx_diff, :, i_diff]

    t_sn   = float(sn['times'][tidx_sn])
    t_diff = float(diff['times'][tidx_diff])

    ax.loglog(E_mid, Er_sn_g,   'o-', color='C0', lw=1.5, ms=5,
              label=f'S$_N$ t={t_sn:.3f} ns')
    ax.loglog(E_mid, Er_diff_g, 's--', color='C1', lw=1.5, ms=5,
              label=f'Diff t={t_diff:.3f} ns')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('$E_{r,g}$ (GJ/cm³)')
    ax.set_title(f'Group spectrum at x≈{x_target} cm')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)


def plot_all_times(sn, diff, plot_times, out_prefix='gaussian_ic_compare'):
    """Make a multi-panel figure for each output time."""
    for t_target in plot_times:
        ti_sn   = find_closest_time(sn['times'],   t_target)
        ti_diff = find_closest_time(diff['times'],  t_target)

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_T  = fig.add_subplot(gs[0, 0])
        ax_Tr = fig.add_subplot(gs[1, 0])
        ax_Er = fig.add_subplot(gs[2, 0])
        ax_sp = fig.add_subplot(gs[:2, 1])
        ax_rt = fig.add_subplot(gs[2, 1])

        compare_snapshot(ax_T, ax_Tr, ax_Er, sn, diff, ti_sn, ti_diff, t_target)
        fig.suptitle(f'Gaussian IC Equilibrium Test — target t ≈ {t_target} ns',
                     fontsize=13, fontweight='bold')

        plot_spectrum(ax_sp, sn, diff, ti_sn, ti_diff, x_target=3.5)

        # Ratio E_r(diff) / E_r(SN) on common grid
        r_sn   = sn['r']
        r_diff = diff['r']
        Er_sn  = sn['E_r'][ti_sn]
        Er_diff_interp = np.interp(r_sn, r_diff, diff['E_r'][ti_diff])
        ratio  = Er_diff_interp / (Er_sn + 1e-30)
        ax_rt.plot(r_sn, ratio, 'k-', lw=1.5)
        ax_rt.axhline(1.0, color='gray', ls='--', lw=1)
        ax_rt.set_xlabel('x (cm)')
        ax_rt.set_ylabel(r'$E_r^{diff}$ / $E_r^{SN}$')
        ax_rt.set_title('Diffusion / Transport ratio')
        ax_rt.set_xlim(r_sn[0], r_sn[-1])
        ax_rt.grid(True, alpha=0.3)

        fname = f'{out_prefix}_t{t_target:.2f}ns.png'.replace('.', 'p')
        fname = f'{out_prefix}_t{t_target:.3f}ns.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved {fname}")
        plt.close(fig)


def plot_summary(sn, diff, plot_times, out_file='gaussian_ic_summary.png'):
    """Overlay T_mat profiles from all times on a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_T, ax_Er = axes
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_times)))

    for ci, t_target in enumerate(plot_times):
        ti_sn   = find_closest_time(sn['times'],   t_target)
        ti_diff = find_closest_time(diff['times'],  t_target)
        col = colors[ci]
        t_sn   = float(sn['times'][ti_sn])
        t_diff = float(diff['times'][ti_diff])
        ax_T.plot(sn['r'],   sn['T_mat'][ti_sn],    '-',  color=col, lw=2.0,
                  label=f'S$_N$  t={t_sn:.2f} ns')
        ax_T.plot(diff['r'], diff['T_mat'][ti_diff], '--', color=col, lw=1.5,
                  label=f'Diff t={t_diff:.2f} ns')
        ax_Er.semilogy(sn['r'],   sn['E_r'][ti_sn],    '-',  color=col, lw=2.0)
        ax_Er.semilogy(diff['r'], diff['E_r'][ti_diff], '--', color=col, lw=1.5)

    ax_T.set_xlabel('x (cm)')
    ax_T.set_ylabel('$T_{mat}$ (keV)')
    ax_T.set_title('Material temperature')
    ax_T.legend(fontsize=7, ncol=2)
    ax_T.grid(True, alpha=0.3)

    ax_Er.set_xlabel('x (cm)')
    ax_Er.set_ylabel('$E_r$ (GJ/cm³)')
    ax_Er.set_title('Total radiation energy density')
    ax_Er.grid(True, which='both', alpha=0.3)

    # Add legend proxies for line styles
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='k', ls='-',  lw=2, label='S$_N$ transport'),
               Line2D([0], [0], color='k', ls='--', lw=2, label='Diffusion')]
    ax_Er.legend(handles=handles, fontsize=9)

    plt.suptitle('Gaussian IC Equilibrium Test — Reflecting BCs', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved {out_file}")
    plt.close(fig)


# ── diagnostics ──────────────────────────────────────────────────────────

def print_diagnostics(sn, diff):
    print("\n" + "=" * 60)
    print("DIAGNOSTICS — Gaussian IC Equilibrium Test")
    print("=" * 60)

    # t=0 is not stored, but check t=0.05 snapshot
    t0 = 0.05
    ti_sn   = find_closest_time(sn['times'],   t0)
    ti_diff = find_closest_time(diff['times'],  t0)

    r_sn   = sn['r']
    r_diff = diff['r']

    for label, ti, data, r in [('S_N', ti_sn, sn, r_sn),
                                ('Diff', ti_diff, diff, r_diff)]:
        t = float(data['times'][ti])
        T = data['T_mat'][ti]
        Er = data['E_r'][ti]
        print(f"\n{label} at t={t:.4f} ns:")
        print(f"  T_mat : max={T.max():.6f}  min={T.min():.6f}  "
              f"centre={np.interp(3.5, r, T):.6f} keV")
        print(f"  E_r   : max={Er.max():.3e}  centre={np.interp(3.5, r, Er):.3e} GJ/cm³")

    # Ratio at t=0.1 ns
    t1 = 0.1
    ti_sn   = find_closest_time(sn['times'],   t1)
    ti_diff = find_closest_time(diff['times'],  t1)
    Er_sn   = sn['E_r'][ti_sn]
    Er_diff_interp = np.interp(r_sn, r_diff, diff['E_r'][ti_diff])
    ratio = Er_diff_interp / (Er_sn + 1e-30)
    print(f"\nE_r ratio (Diff/S_N) at t≈{t1} ns:")
    print(f"  centre (x=3.5): {np.interp(3.5, r_sn, ratio):.4f}")
    print(f"  mean over domain: {ratio.mean():.4f}")
    print(f"  max: {ratio.max():.4f}  min: {ratio.min():.4f}")

    # ── Energy conservation check ────────────────────────────────────────
    # With reflecting BCs and no sources the total energy
    #   E_total = ∫ (ρ c_v T_mat + E_r) dx
    # must equal its analytic t=0 value at every stored snapshot.

    for label, data, r in [('S_N', sn, r_sn), ('Diff', diff, r_diff)]:
        dx = r[1] - r[0]   # uniform cell width

        # Analytic t=0 total energy (Gaussian IC, equilibrium)
        # T_mat stored at cell-center evaluation of Bernstein polynomial;
        # for a uniform Bernstein basis the cell average equals the
        # point evaluation at the cell centre (order-3 polynomial evaluated
        # at x_c is exact to second order, but the cell average is
        # (1/(p+1)) * sum of nodes).  We use the cell-centre values here
        # because that is what is stored in the NPZ, but note that the
        # S_N solver's internal energy is computed from node averages —
        # any residual (< ~1 %) between the two is a quadrature artefact.
        T0       = _gaussian_T0(r)
        E_mat_0  = np.dot(CV_VOL * T0,   np.ones_like(r)) * dx
        E_rad_0  = np.dot(A_RAD  * T0**4, np.ones_like(r)) * dx
        E_tot_0  = E_mat_0 + E_rad_0

        print(f"\n{label} energy conservation  (ρcᵥ = {CV_VOL:.2e} GJ/cm³/keV)")
        print(f"  {'Time (ns)':<12} {'E_mat':>12} {'E_rad':>12} "
              f"{'E_total':>12}  {'ΔE/E₀':>10}")
        print(f"  {'t=0 (IC)':<12} {E_mat_0:>12.5e} {E_rad_0:>12.5e} "
              f"{E_tot_0:>12.5e}  {'---':>10}")
        for ti in range(len(data['times'])):
            t     = float(data['times'][ti])
            E_mat = np.dot(CV_VOL * data['T_mat'][ti], np.ones_like(r)) * dx
            E_rad = np.dot(data['E_r'][ti],            np.ones_like(r)) * dx
            E_tot = E_mat + E_rad
            delta = (E_tot - E_tot_0) / E_tot_0
            print(f"  {t:<12.4f} {E_mat:>12.5e} {E_rad:>12.5e} "
                  f"{E_tot:>12.5e}  {delta:>+10.3e}")

    print("=" * 60)


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Gaussian IC S_N vs Diffusion solutions')
    parser.add_argument('--sn',   default='gaussian_ic_sn_10g.npz',
                        help='Path to S_N NPZ file')
    parser.add_argument('--diff', default='gaussian_ic_diffusion_10g.npz',
                        help='Path to diffusion NPZ file')
    parser.add_argument('--times', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Output times to compare')
    parser.add_argument('--prefix', default='gaussian_ic_compare',
                        help='Output filename prefix for per-time plots')
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────────────────
    sn_data   = load(args.sn)
    diff_data = load(args.diff)

    if sn_data is None:
        print(f"ERROR: S_N file not found: {args.sn}")
        print("Run from DiscreteOrdinates/problems/:")
        print("    python test_gaussian_ic_sn.py --save-npz ../../gaussian_ic_sn_10g.npz")
        sys.exit(1)

    if diff_data is None:
        print(f"ERROR: Diffusion file not found: {args.diff}")
        print("Run from nonEquilibriumDiffusion/problems/:")
        print("    python gaussian_ic_diffusion.py --save-npz ../../gaussian_ic_diffusion_10g.npz")
        sys.exit(1)

    print(f"Loaded S_N:   {args.sn}")
    print(f"  times = {sn_data['times']}")
    print(f"  r shape = {sn_data['r'].shape},  E_r shape = {sn_data['E_r'].shape}")
    print(f"Loaded Diff: {args.diff}")
    print(f"  times = {diff_data['times']}")
    print(f"  r shape = {diff_data['r'].shape},  E_r shape = {diff_data['E_r'].shape}")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_times = sorted(args.times)
    plot_all_times(sn_data, diff_data, plot_times, out_prefix=args.prefix)
    plot_summary(sn_data, diff_data, plot_times)
    print_diagnostics(sn_data, diff_data)
