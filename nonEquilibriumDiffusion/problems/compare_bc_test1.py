#!/usr/bin/env python3
"""
Compare Dirichlet vs Marshak boundary conditions for Converging Marshak Wave Test 1.

Usage:
  python compare_bc_test1.py [--force-run] [--n-cells N]

Options:
  --force-run    : Force re-run even if cached solutions exist
  --n-cells N    : Number of cells (default 100)
"""

import sys
import os
import pickle
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add utils to path for plotting
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'utils'))
from plotfuncs import show

# Import the test modules
import converging_marshak_wave as test_dirichlet
import converging_marshak_wave_marshak as test_marshak


def get_cache_path(bc_type, n_cells):
    """Return path for cached solution."""
    return f'cache_test1_{bc_type}_n{n_cells}.pkl'


def save_solution(bc_type, n_cells, solutions):
    """Save solution snapshots to cache."""
    cache_path = get_cache_path(bc_type, n_cells)
    with open(cache_path, 'wb') as f:
        pickle.dump(solutions, f)
    print(f"  Cached solution to {cache_path}")


def load_solution(bc_type, n_cells):
    """Load solution from cache if it exists."""
    cache_path = get_cache_path(bc_type, n_cells)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            solutions = pickle.load(f)
        print(f"  Loaded cached solution from {cache_path}")
        return solutions
    return None


def run_and_cache(test_module, bc_type, n_cells):
    """Run test and cache the solution snapshots."""
    print(f"\nRunning {bc_type} BC version...")
    print("=" * 70)
    
    # Run the test
    solver, solutions = test_module.run(
        n_cells=n_cells,
        dt_initial=0.0001,
        dt_max=0.01,
        dt_growth=1.05,
        t_duration=test_module.OUTPUT_TIMES_NS[-1] - test_module.T_INIT_NS,
    )
    
    # Cache the solutions
    save_solution(bc_type, n_cells, solutions)
    return solutions


def plot_comparison(solutions_dirichlet, solutions_marshak, save_prefix='compare_bc_test1'):
    """
    Plot comparison of Dirichlet BC, Marshak BC, and self-similar solution.
    
    Creates two figures:
      1) Temperature profiles at output times
      2) Energy density profiles at output times
    """
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    r_anal = np.linspace(0.0, test_dirichlet.R, 2000)
    
    T_HEV_PER_KEV = test_dirichlet.T_HEV_PER_KEV
    
    # --- Temperature comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for idx, (sol_d, sol_m, col) in enumerate(zip(solutions_dirichlet, solutions_marshak, colors)):
        t_ns = sol_d['t_ns']
        r_cm_d = sol_d['r_cm']
        r_cm_m = sol_m['r_cm']
        
        T_HeV_d = sol_d['T_keV'] * T_HEV_PER_KEV
        T_HeV_m = sol_m['T_keV'] * T_HEV_PER_KEV
        
        # Analytic solution
        T_ref = test_dirichlet.T_analytic_HeV_vec(r_anal, t_ns)
        
        label_base = f't = {t_ns:.1f} ns'
        ax.plot(r_cm_d / 1e-4, T_HeV_d/10, color=col, lw=2, ls='-', 
                label=f'{label_base} (Dirichlet)')
        ax.plot(r_cm_m / 1e-4, T_HeV_m/10, color=col, lw=2, ls='--',
                label=f'{label_base} (Marshak)')
        ax.plot(r_anal / 1e-4, T_ref/10, color=col, lw=1.5, ls=':',
                label=f'{label_base} (analytic)')
    
    ax.set_xlabel(r'$r$ ($\mu$m)', fontsize=12)
    ax.set_ylabel(r'$T$ (keV)', fontsize=12)
    ax.set_title('Test 1: Boundary Condition Comparison (Temperature)', fontsize=13)
    ticks = np.linspace(0, test_dirichlet.R / 1e-4, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outfile = f'{save_prefix}_T.pdf'
    show(outfile, close_after=True)
    print(f'Saved: {outfile}')
    
    # --- Energy density comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for idx, (sol_d, sol_m, col) in enumerate(zip(solutions_dirichlet, solutions_marshak, colors)):
        t_ns = sol_d['t_ns']
        r_cm_d = sol_d['r_cm']
        r_cm_m = sol_m['r_cm']
        
        # Convert GJ/cm³ → 10¹³ erg/cm³ (×1e3)
        u_d = test_dirichlet.test1_material_energy(sol_d['T_keV']) * 1e3
        u_m = test_marshak.test1_material_energy(sol_m['T_keV']) * 1e3
        
        # Analytic
        u_ref = test_dirichlet.u_analytic_erg_per_1e13(r_anal, t_ns)
        
        label_base = f't = {t_ns:.1f} ns'
        ax.plot(r_cm_d / 1e-4, u_d/1e3, color=col, lw=2, ls='-',
                label=f'{label_base} (Dirichlet)')
        ax.plot(r_cm_m / 1e-4, u_m/1e3, color=col, lw=2, ls='--',
                label=f'{label_base} (Marshak)')
        ax.plot(r_anal / 1e-4, u_ref/1e3, color=col, lw=1.5, ls=':',
                label=f'{label_base} (analytic)')
    
    ax.set_xlabel(r'$r$ ($\mu$m)', fontsize=12)
    ax.set_ylabel(r'$e(T)$ (GJ/cm$^3$)', fontsize=12)
    ax.set_title('Test 1: Boundary Condition Comparison (Energy Density)', fontsize=13)
    ticks = np.linspace(0, test_dirichlet.R / 1e-4, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    outfile = f'{save_prefix}_u.pdf'
    show(outfile, close_after=True)
    print(f'Saved: {outfile}')


def main():
    parser = argparse.ArgumentParser(
        description='Compare Dirichlet vs Marshak BCs for Test 1'
    )
    parser.add_argument('--force-run', action='store_true',
                        help='Force re-run even if cached solutions exist')
    parser.add_argument('--n-cells', type=int, default=100,
                        help='Number of spatial cells (default: 100)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Converging Marshak Wave Test 1: BC Comparison")
    print("=" * 70)
    print(f"n_cells = {args.n_cells}")
    print(f"force_run = {args.force_run}")
    print("=" * 70)
    
    # Load or run Dirichlet BC version
    if args.force_run:
        solutions_dirichlet = None
    else:
        solutions_dirichlet = load_solution('dirichlet', args.n_cells)
    
    if solutions_dirichlet is None:
        solutions_dirichlet = run_and_cache(test_dirichlet, 'dirichlet', args.n_cells)
    
    # Load or run Marshak BC version
    if args.force_run:
        solutions_marshak = None
    else:
        solutions_marshak = load_solution('marshak', args.n_cells)
    
    if solutions_marshak is None:
        solutions_marshak = run_and_cache(test_marshak, 'marshak', args.n_cells)
    
    # Generate comparison plots
    print("\n" + "=" * 70)
    print("Generating comparison plots...")
    print("=" * 70)
    plot_comparison(solutions_dirichlet, solutions_marshak)
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
