"""
Test Newton step damping functionality.

Demonstrates how max_relative_change limits the Newton step size.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D

A_RAD = 0.01372  # GJ/(cm^3·keV^4)

def test_damping():
    """Test damping with different max_relative_change values."""
    
    # Problem setup
    n_groups = 10
    n_cells = 30
    r_min, r_max = 0.0, 5.0
    dt = 10.0  # Larger time step to force bigger Newton updates
    geometry = 'planar'
    
    # Energy edges (logarithmically spaced from 0.1 eV to 10 keV)
    energy_edges = np.logspace(-4, 1, n_groups + 1)
    
    # Uniform emission fractions
    chi = np.ones(n_groups) / n_groups
    
    # Simple constant coefficients
    def sigma_func(T, r, E_low, E_high):
        return 1.0
    
    def diff_func(T, r, g=0):
        return 1.0 / 3.0
    
    sigma_funcs = [lambda T, r, g=g: sigma_func(T, r, energy_edges[g], energy_edges[g+1]) 
                   for g in range(n_groups)]
    diff_funcs = [lambda T, r, g=g: diff_func(T, r, g) for g in range(n_groups)]
    
    # Neumann BCs
    def neumann_bc(phi, r):
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [neumann_bc] * n_groups
    right_bc_funcs = [neumann_bc] * n_groups
    
    print("="*70)
    print("Newton Damping Test")
    print("="*70)
    print(f"Problem: {n_groups} groups, {n_cells} cells")
    print(f"Time step: dt = {dt} ns")
    print()
    
    # Test different damping values
    damping_values = [1.0, 0.5, 0.2]
    
    for max_rel_change in damping_values:
        print(f"\n{'='*70}")
        print(f"Test with max_relative_change = {max_rel_change}")
        print(f"{'='*70}\n")
        
        # Create solver
        solver = MultigroupDiffusionSolver1D(
            n_groups=n_groups,
            r_min=r_min,
            r_max=r_max,
            n_cells=n_cells,
            energy_edges=energy_edges,
            geometry=geometry,
            dt=dt,
            diffusion_coeff_funcs=diff_funcs,
            absorption_coeff_funcs=sigma_funcs,
            left_bc_funcs=left_bc_funcs,
            right_bc_funcs=right_bc_funcs,
            emission_fractions=chi,
            rho=1.0,
            cv=0.1
        )
        
        # Initial temperature profile
        r_centers = solver.r_centers
        T_init = 0.1 + 0.9 * np.exp(-r_centers / 2.0)
        
        solver.T = T_init.copy()
        solver.T_old = T_init.copy()
        solver.E_r = A_RAD * T_init**4
        solver.E_r_old = solver.E_r.copy()
        solver.kappa = np.zeros(n_cells)
        solver.kappa_old = np.zeros(n_cells)
        
        # Take one Newton step
        info = solver.step(
            max_newton_iter=1,
            newton_tol=1e-8,
            gmres_tol=1e-8,
            gmres_maxiter=200,
            use_preconditioner=False,
            max_relative_change=max_rel_change,
            verbose=True
        )
        
        print(f"\nResult:")
        print(f"  GMRES iterations: {info['gmres_info']['iterations']}")
        print(f"  Temperature change: {info['T_change']:.3e}")
        print(f"  Final T_max: {solver.T.max():.6f} keV")
        print(f"  Final T_min: {solver.T.min():.6f} keV")
        
        if max_rel_change < 1.0:
            print(f"  ✓ Step was damped to limit relative change")

if __name__ == "__main__":
    test_damping()
