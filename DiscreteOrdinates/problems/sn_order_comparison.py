"""
S_N order-convergence comparison for three benchmark problems:
  1. Su-Olson  (top-hat source, nonlinear EOS  e = a T^4)
  2. Bennett flat-top  (top-hat source, linear EOS  e = Cv rho T)
  3. Bennett Gaussian  (Gaussian source  exp(-x^2/x_0^2), same linear EOS)

For each S_N order in [2, 4, 6, 8] (default) the script:
  - Loads pre-computed '<problem>_N{N}_I{I}.npz' if it exists, otherwise
    runs the transport calculation and saves it.
  - Plots E_rad and E_mat at a chosen tau with one curve per order and
    benchmark symbols overlaid (one figure per problem).
  - Computes relative L2 errors against the transport reference and plots
    error vs. S_N order for all three problems on a single graph.

Run from the DiscreteOrdinates directory:
    python problems/sn_order_comparison.py [--zones 400] [--tau 10.0]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import BPoly

# ---- solver / problem imports ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sn_solver  # noqa: F401  (needed to access physical constants)

sys.path.insert(0, os.path.dirname(__file__))
from test_su_olson import (setup_and_run as su_olson_run,
                            save_npz      as su_olson_save,
                            load_npz      as su_olson_load,
                            su_olson_x    as SO_X,
                            su_olson_tau  as SO_TAU,
                            transport_rad_energy as SO_RAD_REF,
                            transport_mat_energy as SO_MAT_REF)

from bennett_sn import (setup_and_run as bennett_run,
                         save_npz      as bennett_save,
                         load_npz      as bennett_load,
                         su_olson_x    as BN_X,
                         su_olson_tau  as BN_TAU,
                         transport_rad_energy as BN_RAD_REF,
                         transport_mat_energy as BN_MAT_REF)

from bennett_gaussian_sn import (setup_and_run as bg_run,
                                   save_npz      as bg_save,
                                   load_npz      as bg_load,
                                   ref_x         as BG_X,
                                   ref_tau        as BG_TAU,
                                   transport_rad_energy as BG_RAD_REF,
                                   transport_mat_energy as BG_MAT_REF)

# Add project root for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from utils.plotfuncs import show, hide_spines, font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORDERS_DEFAULT = [2, 4, 6, 8, 10]
_LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # one per S_N order
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#b427d3']   # used in L2-error plot
_TAU_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']  # one per tau snapshot


def _interp_to_ref(results, tau_val, ref_x):
    """Interpolate E_rad and E_mat from a results dict onto ref_x positions.

    Uses piecewise-linear interpolation of the cell-centre values (the
    Bernstein polynomial coefficients at the midpoint are the cell-centre
    averages, which is accurate enough for error computation).
    """
    sol = results['solutions'].get(tau_val)
    if sol is None:
        return None, None
    x   = results['x']
    hx  = results['hx']
    # cell-centre average from Bernstein coefficients (mean of all nodes)
    E_rad_cc = sol['E_rad'].mean(axis=1)
    E_mat_cc = sol['E_mat'].mean(axis=1)
    E_rad_ref_x = np.interp(ref_x, x, E_rad_cc)
    E_mat_ref_x = np.interp(ref_x, x, E_mat_cc)
    return E_rad_ref_x, E_mat_ref_x


def _l2_error(computed, reference):
    """Relative L2 error, ignoring NaN reference entries."""
    mask = ~np.isnan(reference) & (reference > 0)
    if mask.sum() == 0:
        return np.nan
    diff = computed[mask] - reference[mask]
    return np.sqrt(np.sum(diff**2) / np.sum(reference[mask]**2))


_PROBLEM_MAP = {
    'su_olson':         ('su_olson_sn',          su_olson_run,   su_olson_save,  su_olson_load),
    'bennett':          ('bennett_sn',            bennett_run,    bennett_save,   bennett_load),
    'bennett_gaussian': ('bennett_gaussian_sn',   bg_run,         bg_save,        bg_load),
}


def _load_or_run(problem, N, I, order, K, maxits, output_tau, loud=0):
    """Load an existing .npz file or run the calculation and save it."""
    prefix, runner, saver, loader = _PROBLEM_MAP[problem]
    problems_dir = os.path.dirname(os.path.abspath(__file__))
    sn_dir       = os.path.dirname(problems_dir)
    npz_path     = os.path.join(sn_dir, f'{prefix}_N{N}_I{I}.npz')

    if os.path.exists(npz_path):
        print(f"[{problem} N={N}] Loading cached results from {npz_path}")
        return loader(npz_path)

    print(f"[{problem} N={N}] Running transport calculation...")
    results = runner(I=I, order=order, N=N, K=K, maxits=maxits, LOUD=loud,
                     output_tau=output_tau)
    saver(results, npz_path)
    return results


# ---------------------------------------------------------------------------
# Solution comparison plots (two separate figures per problem)
# ---------------------------------------------------------------------------

def _plot_solution_comparison(all_results, ref_x, ref_tau, ref_rad, ref_mat,
                               tau_plots, orders, title_prefix, savefile=''):
    """Plot E_rad and E_mat for multiple tau values and all S_N orders.

    Color encodes tau snapshot; line style encodes S_N order.
    Reference squares share the same color as their tau snapshot.

    Parameters
    ----------
    tau_plots : list of float
        Dimensionless times to overlay on each figure.

    Returns
    -------
    (fig_rad, fig_mat) — two separate single-panel figures.
    """
    fig_rad, ax_rad = plt.subplots(1, 1, figsize=(7.5, 5.25))
    fig_mat, ax_mat = plt.subplots(1, 1, figsize=(7.5, 5.25))

    for order_idx, N in enumerate(orders):
        results = all_results[N]
        if results is None:
            continue

        color = _COLORS[order_idx]
        hx    = results['hx']
        Lx    = results['Lx']
        nI    = len(results['x'])
        edges = np.linspace(0, Lx, nI + 1)
        xplot = np.linspace(hx / 2, Lx, 2000)

        for tau in tau_plots:
            sol = results['solutions'].get(tau)
            if sol is None:
                available = sorted(results['solutions'].keys())
                nearest   = min(available, key=lambda t: abs(t - tau))
                print(f"  Warning: tau={tau} not found for N={N}, using nearest tau={nearest}")
                sol       = results['solutions'][nearest]

            E_rad_curve = BPoly(sol['E_rad'].T, edges)(xplot)
            E_mat_curve = BPoly(sol['E_mat'].T, edges)(xplot)

            ls = _LINE_STYLES[order_idx]
            ax_rad.plot(xplot, E_rad_curve, linestyle=ls, color=color, lw=2.0, alpha=0.85)
            ax_mat.plot(xplot, E_mat_curve, linestyle=ls, color=color, lw=2.0, alpha=0.85)

    # Reference symbols (black squares) for each tau snapshot
    for tau in tau_plots:
        ref_tidx    = np.argmin(np.abs(ref_tau - tau))
        ref_rad_col = ref_rad[:, ref_tidx]
        ref_mat_col = ref_mat[:, ref_tidx]
        valid_r = ~np.isnan(ref_rad_col)
        valid_m = ~np.isnan(ref_mat_col)
        ax_rad.plot(ref_x[valid_r], ref_rad_col[valid_r], 's',
                    color='k', mec='k', mew=1.5, ms=6, zorder=10, alpha=0.8, linestyle='')
        ax_mat.plot(ref_x[valid_m], ref_mat_col[valid_m], 's',
                    color='k', mec='k', mew=1.5, ms=6, zorder=10, alpha=0.8, linestyle='')

    # Legend: one colored solid line per S_N order + transport ref square
    legend_elements = [
        Line2D([0], [0], color=_COLORS[order_idx], linestyle=_LINE_STYLES[order_idx],
               linewidth=2.0, label=rf'S$_{{{N}}}$')
        for order_idx, N in enumerate(orders)
    ]
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k',
               markeredgecolor='k', markersize=6, markeredgewidth=1.5,
               linestyle='', label='Transport ref')
    )

    for ax, ylabel in [
        (ax_rad, r'Radiation Energy Density $\frac{\phi}{a\,c\,T_0^4}$'),
        (ax_mat, r'Material Energy Density $\frac{e}{a\,T_0^4}$'),
    ]:
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Position (mean-free path)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.2, 15)
        ax.set_ylim(1e-3, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)

    if savefile and HAS_PLOTFUNCS:
        base = savefile.rsplit('.', 1)[0] if '.' in os.path.basename(savefile) else savefile
        plt.figure(fig_rad.number)
        plt.tight_layout()
        show(f'{base}_rad.pdf', close_after=True)
        plt.figure(fig_mat.number)
        plt.tight_layout()
        show(f'{base}_mat.pdf', close_after=True)
    else:
        fig_rad.tight_layout()
        fig_mat.tight_layout()

    return fig_rad, fig_mat


# ---------------------------------------------------------------------------
# L2 error vs. S_N order plot
# ---------------------------------------------------------------------------

def _plot_l2_error(orders, so_errors, bn_errors, bg_errors, savefile=''):
    """Plot radiation-energy L2 error vs. S_N order for all three problems."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(orders, so_errors, 'o-',  color=_COLORS[0], lw=2, ms=7,
            label='Su-Olson')
    ax.plot(orders, bn_errors, 's--', color=_COLORS[1], lw=2, ms=7,
            label='Bennett (flat)')
    ax.plot(orders, bg_errors, '^-.', color=_COLORS[2], lw=2, ms=7,
            label='Bennett (Gaussian)')

    ax.set_xlabel(r'S$_N$ order $N$', fontsize=13)
    ax.set_ylabel(r'Relative $L_2$ error  ($E_r$)', fontsize=13)
    #no title for publication quality, but keep for debugging
    #ax.set_title(r'S$_N$ order convergence', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if savefile and HAS_PLOTFUNCS:
        show(savefile, close_after=True)

    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_comparison(orders=None, I=400, order=2, K=800, maxits=2000,
                   tau_plot=10, loud=0,
                   save_figs=False, fig_prefix='sn_comparison'):
    """Run or load all transport calculations and produce the comparison plots.

    Parameters
    ----------
    orders : list of int
        S_N orders to compare (default: [2, 4, 6, 8]).
    I : int
        Spatial zones per half-domain.
    order : int
        Bernstein polynomial order.
    K : int
        DMD inner iterations.
    maxits : int
        Max iterations per timestep.
    tau_plot : float
        Dimensionless time at which to compare solutions.
    loud : int
        Verbosity passed to setup_and_run.
    save_figs : bool
        Save figures to PNG files.
    fig_prefix : str
        Prefix for saved figure filenames.
    """
    if orders is None:
        orders = _ORDERS_DEFAULT

    # Taus shown on solution plots: 1, 3.16228, and the user-specified tau_plot.
    # Near-duplicates (within 1%) are collapsed.
    _raw_sol_taus = sorted({1.0, 3.16228,10.0, tau_plot})
    sol_taus = [_raw_sol_taus[0]]
    for _t in _raw_sol_taus[1:]:
        if _t / sol_taus[-1] > 1.01:
            sol_taus.append(_t)

    # Output tau covers all solution taus plus standard survey times
    output_tau = tuple(sorted(
        set(sol_taus) | {0.1} |
        ({31.6228, 100.0} if tau_plot > 10 else set())
    ))

    so_results = {}
    bn_results = {}
    bg_results = {}

    for N in orders:
        so_results[N] = _load_or_run('su_olson',         N, I, order, K, maxits,
                                      output_tau, loud)
        bn_results[N] = _load_or_run('bennett',           N, I, order, K, maxits,
                                      output_tau, loud)
        bg_results[N] = _load_or_run('bennett_gaussian',  N, I, order, K, maxits,
                                      output_tau, loud)

    # ---- solution plots ----
    so_figs = _plot_solution_comparison(
        so_results, SO_X, SO_TAU, SO_RAD_REF, SO_MAT_REF,
        sol_taus, orders, 'Su-Olson',
        savefile=f'{fig_prefix}_su_olson_solution' if save_figs else '')

    bn_figs = _plot_solution_comparison(
        bn_results, BN_X, BN_TAU, BN_RAD_REF, BN_MAT_REF,
        sol_taus, orders, 'Bennett (flat source)',
        savefile=f'{fig_prefix}_bennett_solution' if save_figs else '')

    bg_figs = _plot_solution_comparison(
        bg_results, BG_X, BG_TAU, BG_RAD_REF, BG_MAT_REF,
        sol_taus, orders, r'Bennett (Gaussian source $e^{-x^2/x_0^2}$)',
        savefile=f'{fig_prefix}_bennett_gaussian_solution' if save_figs else '')

    # ---- L2 errors at tau_plot ----
    tau_idx_so = np.argmin(np.abs(SO_TAU - tau_plot))
    tau_idx_bn = np.argmin(np.abs(BN_TAU - tau_plot))
    tau_idx_bg = np.argmin(np.abs(BG_TAU - tau_plot))

    so_errors = []
    bn_errors = []
    bg_errors = []

    print(f"\nL2 errors at tau = {tau_plot}")
    print(f"{'N':>4}  {'Su-Olson':>14}  {'Bennett flat':>14}  {'Bennett Gaussian':>18}")
    print("-" * 58)

    for N in orders:
        E_rad_so, _ = _interp_to_ref(so_results[N], tau_plot, SO_X)
        err_so = _l2_error(E_rad_so, SO_RAD_REF[:, tau_idx_so]) \
                 if E_rad_so is not None else np.nan

        E_rad_bn, _ = _interp_to_ref(bn_results[N], tau_plot, BN_X)
        err_bn = _l2_error(E_rad_bn, BN_RAD_REF[:, tau_idx_bn]) \
                 if E_rad_bn is not None else np.nan

        E_rad_bg, _ = _interp_to_ref(bg_results[N], tau_plot, BG_X)
        err_bg = _l2_error(E_rad_bg, BG_RAD_REF[:, tau_idx_bg]) \
                 if E_rad_bg is not None else np.nan

        so_errors.append(err_so)
        bn_errors.append(err_bn)
        bg_errors.append(err_bg)
        print(f"{N:>4}  {err_so:>14.4e}  {err_bn:>14.4e}  {err_bg:>18.4e}")

    err_fig = _plot_l2_error(
        orders, so_errors, bn_errors, bg_errors,
        savefile=f'{fig_prefix}_l2_error.pdf' if save_figs else '')

    if not HAS_PLOTFUNCS or not save_figs:
        plt.show()

    return {
        'su_olson':         so_results,
        'bennett':          bn_results,
        'bennett_gaussian': bg_results,
        'so_errors': so_errors,
        'bn_errors': bn_errors,
        'bg_errors': bg_errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='S_N order-convergence comparison: Su-Olson, Bennett flat, Bennett Gaussian')
    parser.add_argument('--orders', type=int, nargs='+', default=[2, 4, 6, 8, 10],
                        help='S_N orders to compare (default: 2 4 6 8 10)')
    parser.add_argument('--zones', type=int, default=400,
                        help='Spatial zones per half-domain (default: 400)')
    parser.add_argument('--order', type=int, default=4,
                        help='Bernstein polynomial order (default: 4)')
    parser.add_argument('--K', type=int, default=800,
                        help='DMD inner iterations (default: 800)')
    parser.add_argument('--maxits', type=int, default=2000,
                        help='Max iterations per timestep (default: 2000)')
    parser.add_argument('--tau', type=float, default=10.0,
                        help='Dimensionless time for solution plot and L2 '
                             'error (default: 10.0)')
    parser.add_argument('--loud', type=int, default=0,
                        help='Verbosity (default: 0)')
    parser.add_argument('--save-figs', action='store_true', default=False,
                        help='Save figures to PNG files')
    parser.add_argument('--fig-prefix', type=str, default='sn_comparison',
                        help='Prefix for saved figure filenameS '
                             '(default: sn_comparison)')
    args = parser.parse_args()

    run_comparison(
        orders=args.orders,
        I=args.zones,
        order=args.order,
        K=args.K,
        maxits=args.maxits,
        tau_plot=args.tau,
        loud=args.loud,
        save_figs=args.save_figs,
        fig_prefix=args.fig_prefix,
    )


if __name__ == '__main__':
    main()
