#!/usr/bin/env python3
"""
Test 1-group multigroup solver with constant opacity
Single timestep to check GMRES convergence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Physical constants
RHO = 1.0  # g/cm³

def constant_opacity(T, r):
    """Constant opacity: σ = 300 cm^-1"""
    return 300.0

def constant_diffusion_coeff(T, r):
    """Diffusion coefficient: D = c/(3σ)"""
    sigma = constant_opacity(T, r)
    return C_LIGHT / (3.0 * sigma)

def run_test():
    """Run 1-group test with constant opacity"""
    
    print("="*80)
    print("1-Group Test with Constant Opacity")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_a = 300 cm^-1 (constant)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("="*80)
    
    # Problem setup
    n_groups = 1
    r_min = 0.0
    r_max = 0.5
    n_cells = 50
    
    energy_edges = np.array([0.01, 10.0])
    
    dt = 0.001  # ns
    
    rho = RHO
    cv = 0.3 / rho
    
    T_bc = 1.0  # keV
    phi_bc_total = C_LIGHT * A_RAD * T_bc**4
    left_bc_values = [phi_bc_total]
    right_bc_values = [0.0]
    
    print(f"\nSetup: {n_cells} cells, dt = {dt} ns")
    print(f"Left boundary φ_bc = {phi_bc_total:.6e} GJ/cm²")
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=[constant_diffusion_coeff],
        absorption_coeff_funcs=[constant_opacity],
        left_bc='dirichlet',
        right_bc='neumann',
        left_bc_values=left_bc_values,
        right_bc_values=right_bc_values,
        rho=rho,
        cv=cv
    )
    
    # Initial condition
    T_init = 0.1  # keV
    solver.T = np.full(n_cells, T_init)
    solver.T_old = solver.T.copy()
    solver.E_r = np.full(n_cells, A_RAD * T_init**4)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    print(f"Initial: T = {T_init} keV, E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"Opacity: σ = {constant_opacity(T_init, 0.0):.2e} cm⁻¹ (constant)")
    
    # Take ONE timestep
    print("\n" + "="*80)
    print("Taking 1 timestep...")
    print("="*80)
    
    info = solver.step(max_newton_iter=10, newton_tol=1e-6,
                      gmres_tol=1e-6, gmres_maxiter=200,
                      verbose=True)
    
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    print(f"Newton iterations: {info.get('newton_iter', info.get('n_newton_iter', '?'))}")
    print(f"GMRES iterations: {info.get('gmres_iter', '?')}")
    print(f"Converged: {info.get('converged', False)}")
    print(f"T range: [{solver.T.min():.6f}, {solver.T.max():.6f}] keV")
    print(f"E_r range: [{solver.E_r.min():.6e}, {solver.E_r.max():.6e}] GJ/cm³")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    r = solver.r_centers
    
    # Temperature
    ax = axes[0, 0]
    ax.plot(r, solver.T, 'b-', linewidth=2)
    ax.axhline(T_init, color='gray', linestyle=':', label='Initial')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Radiation energy
    ax = axes[0, 1]
    ax.semilogy(r, solver.E_r, 'r-', linewidth=2)
    ax.axhline(A_RAD * T_init**4, color='gray', linestyle=':', label='Initial')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('E_r (GJ/cm³)')
    ax.set_title('Radiation Energy Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absorption rate κ
    ax = axes[1, 0]
    ax.plot(r, solver.kappa, 'g-', linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('κ = σ*φ (GJ/(cm³·ns))')
    ax.set_title('Absorption Rate Density')
    ax.grid(True, alpha=0.3)
    
    # Radiation flux (approximate)
    ax = axes[1, 1]
    phi = solver.E_r * C_LIGHT
    ax.semilogy(r, phi, 'm-', linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('φ = c·E_r (GJ/(cm²·ns))')
    ax.set_title('Radiation Intensity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_1group_constant_opacity.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_1group_constant_opacity.png'")

if __name__ == "__main__":
    run_test()
