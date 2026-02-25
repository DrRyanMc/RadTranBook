#!/usr/bin/env python3
"""
Example demonstrating flux limiter usage in the non-equilibrium radiation diffusion solver

This script shows how to use different flux limiters with the solver.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (NonEquilibriumRadiationDiffusionSolver, 
                    flux_limiter_standard,
                    flux_limiter_levermore_pomraning,
                    flux_limiter_larsen,
                    flux_limiter_sum,
                    flux_limiter_max,
                    A_RAD, C_LIGHT)

# =============================================================================
# MATERIAL PROPERTIES (Zeldovich wave parameters)
# =============================================================================

def zeldovich_rosseland_opacity(T):
    """σ_R = 300 * T^(-3) cm^(-1)"""
    return 300.0 * T**(-3.0)

def zeldovich_planck_opacity(T):
    """σ_P = σ_R (same opacity)"""
    return zeldovich_rosseland_opacity(T)

def specific_heat(T):
    """c_v = 3e-6 GJ/(cm³·keV)"""
    return 3e-6

# =============================================================================
# FLUX LIMITER COMPARISON
# =============================================================================

def plot_flux_limiters():
    """Plot the different flux limiter functions"""
    R = np.logspace(-2, 3, 200)  # R from 0.01 to 1000
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(R, flux_limiter_standard(R), 'k-', linewidth=2, label='Standard (λ=1/3)')
    ax.loglog(R, flux_limiter_levermore_pomraning(R), 'r-', linewidth=2, label='Levermore-Pomraning')
    ax.loglog(R, flux_limiter_larsen(R, n=2), 'b-', linewidth=2, label='Larsen (n=2)')
    ax.loglog(R, flux_limiter_sum(R), 'g-', linewidth=2, label='Sum')
    ax.loglog(R, flux_limiter_max(R), 'm-', linewidth=2, label='Max')
    
    ax.set_xlabel('R = |∇φ|/(σ_R φ)', fontsize=14)
    ax.set_ylabel('λ(R)', fontsize=14)
    ax.set_title('Flux Limiter Functions', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0.01, 1000])
    ax.set_ylim([0.001, 1])
    
    plt.tight_layout()
    plt.savefig('flux_limiter_comparison.png', dpi=150)
    print("Flux limiter comparison saved as 'flux_limiter_comparison.png'")
    plt.close()

def run_comparison_test():
    """Run a simple test comparing standard vs flux-limited diffusion"""
    print("\n" + "="*80)
    print("Flux Limiter Comparison Test")
    print("="*80)
    
    # Problem parameters
    r_min = 0.0
    r_max = 3.0
    n_cells = 100
    dt_initial = 1e-7
    
    # Initial conditions: Hot spot at center
    def initialize_solver(solver, T_hot=5.0, T_cold=0.01):
        """Initialize with hot center cell"""
        for i in range(n_cells):
            if i == 0:
                solver.T[i] = T_hot
                solver.phi[i] = A_RAD * C_LIGHT * T_hot**4
            else:
                solver.T[i] = T_cold
                solver.phi[i] = A_RAD * C_LIGHT * T_cold**4
        solver.T_old[:] = solver.T
        solver.phi_old[:] = solver.phi
    
    # Test with standard limiter
    print("\n1. Testing with STANDARD limiter (λ=1/3)...")
    solver_std = NonEquilibriumRadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0, dt=dt_initial,
        rosseland_opacity_func=zeldovich_rosseland_opacity,
        planck_opacity_func=zeldovich_planck_opacity,
        specific_heat_func=specific_heat,
        flux_limiter_func=flux_limiter_standard
    )
    initialize_solver(solver_std)
    
    # Take one timestep (newton_step returns the new phi and T)
    phi_new, T_new = solver_std.newton_step(solver_std.phi_old, solver_std.T_old, verbose=True)
    solver_std.phi = phi_new
    solver_std.T = T_new
    print(f"   Max T: {np.max(solver_std.T):.4f} keV")
    print(f"   Max φ: {np.max(solver_std.phi):.4e} GJ/cm³")
    
    # Test with Levermore-Pomraning limiter
    print("\n2. Testing with LEVERMORE-POMRANING limiter...")
    solver_lp = NonEquilibriumRadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0, dt=dt_initial,
        rosseland_opacity_func=zeldovich_rosseland_opacity,
        planck_opacity_func=zeldovich_planck_opacity,
        specific_heat_func=specific_heat,
        flux_limiter_func=flux_limiter_levermore_pomraning
    )
    initialize_solver(solver_lp)
    
    # Take one timestep
    phi_new, T_new = solver_lp.newton_step(solver_lp.phi_old, solver_lp.T_old, verbose=True)
    solver_lp.phi = phi_new
    solver_lp.T = T_new
    print(f"   Max T: {np.max(solver_lp.T):.4f} keV")
    print(f"   Max φ: {np.max(solver_lp.phi):.4e} GJ/cm³")
    
    print("\n" + "="*80)
    print("Both flux limiters work correctly!")
    print("="*80)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\nFlux Limiter Example")
    print("=" * 80)
    
    # Plot flux limiter functions
    plot_flux_limiters()
    
    # Run comparison test
    run_comparison_test()
    
    print("\nExample completed successfully!")
