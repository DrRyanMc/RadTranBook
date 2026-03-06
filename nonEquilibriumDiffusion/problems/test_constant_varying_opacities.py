#!/usr/bin/env python3
"""
Test symmetric BC with CONSTANT opacities that vary greatly between groups.
This isolates whether the bug is:
1. Temperature-dependent opacities, OR
2. Large opacity contrast between groups
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def test_constant_varying_opacities(n_groups=5, T_bc=0.05, n_cells=1, max_steps=100):
    """Test with constant opacities that vary between groups"""
    
    r_min = 0.0
    r_max = 1.0e-3
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    dt = 0.01
    rho = 1.0
    cv = 0.05 / rho
    
    # Define opacities: use power-law at T_bc, but keep them CONSTANT (no T dependence)
    # This gives same opacity contrast as power-law case but without T dependence
    sigma_values = []
    for g in range(n_groups):
        E_mid = np.sqrt(energy_edges[g] * energy_edges[g+1])
        # Use power-law formula at T_bc
        sigma_g = 10.0 * rho * (T_bc**(-1/2)) * (E_mid**(-3.0))
        sigma_values.append(sigma_g)
    
    print(f"Opacity values (constant with T):")
    for g in range(n_groups):
        print(f"  Group {g}: σ_a = {sigma_values[g]:.6e} cm^-1")
    
    # Create constant opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        sigma_g_fixed = sigma_values[g]
        
        def make_opacity_func(sigma_fixed):
            def opacity_func(T, r):
                # Return constant value regardless of T
                return sigma_fixed * np.ones_like(T) if hasattr(T, '__len__') else sigma_fixed
            return opacity_func
        
        def make_diffusion_func(sigma_fixed):
            def diffusion_func(T, r):
                return (1.0 / (3.0 * sigma_fixed)) * np.ones_like(T) if hasattr(T, '__len__') else (1.0 / (3.0 * sigma_fixed))
            return diffusion_func
        
        sigma_funcs.append(make_opacity_func(sigma_g_fixed))
        diff_funcs.append(make_diffusion_func(sigma_g_fixed))
    
    # Emission fractions from Planck function at T_bc
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, sigma_a_groups=np.array(sigma_values))
    
    print(f"\nEmission fractions χ:")
    for g in range(n_groups):
        print(f"  Group {g}: χ_{g} = {chi[g]:.6e}")
    
    # BCs - SYMMETRIC
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    def make_left_bc_func(group_idx):
        def left_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[0])
            D_g = diff_funcs[group_idx](T_avg, 0.0)
            if hasattr(D_g, '__len__'):
                D_g = D_g[0]
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return left_bc
    
    def make_right_bc_func(group_idx):
        def right_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[-1])
            D_g = diff_funcs[group_idx](T_avg, r_max)
            if hasattr(D_g, '__len__'):
                D_g = D_g[0]
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return right_bc
    
    left_bc_funcs = [make_left_bc_func(g) for g in range(n_groups)]
    right_bc_funcs = [make_right_bc_func(g) for g in range(n_groups)]
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs, right_bc_funcs=right_bc_funcs,
        emission_fractions=chi, rho=rho, cv=cv
    )
    
    # Initial condition
    T_init = 0.5 * T_bc
    solver.T[:] = T_init
    solver.E_r[:] = A_RAD * T_init**4
    solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
    solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * T_init**4
    
    # Run to steady state
    T_old = solver.T.copy()
    converged = False
    for step in range(max_steps):
        solver.step()
        
        if step > 10:
            T_change = np.max(np.abs(solver.T - T_old))
            if T_change < 1e-6 * T_bc:
                converged = True
                break
            T_old = solver.T.copy()
    
    T_final = np.mean(solver.T)
    T_ratio = T_final / T_bc
    
    # Compute emission and absorption
    phi_total = solver.E_r * C_LIGHT
    phi_g = phi_total[np.newaxis, :] * solver.phi_g_fraction
    E_r_g = solver.E_r[np.newaxis, :] * solver.phi_g_fraction
    
    dx = (r_max - r_min) / n_cells
    total_absorption = 0.0
    total_emission = 0.0
    
    for g in range(n_groups):
        sigma_g = sigma_values[g]  # Constant value
        absorption_g = C_LIGHT * sigma_g * E_r_g[g, :]
        emission_g = C_LIGHT * sigma_g * chi[g] * A_RAD * solver.T**4
        total_absorption += np.sum(absorption_g) * dx
        total_emission += np.sum(emission_g) * dx
    
    return {
        'T_ratio': T_ratio,
        'converged': converged,
        'absorption': total_absorption,
        'emission': total_emission,
        'emission_to_absorption': total_emission / total_absorption if total_absorption > 0 else np.inf,
        'sigma_values': sigma_values
    }


if __name__ == "__main__":
    print("="*80)
    print("TEST: Constant opacities with large group-to-group variation")
    print("="*80)
    print(f"\nConfiguration: T_bc = 0.05 keV, n_cells = 1, symmetric BCs")
    print(f"Opacities: Computed from power-law at T_bc, but CONSTANT with T")
    print(f"Expected result: T/T_bc = 1.0 if temperature dependence is the bug")
    print(f"                T/T_bc >> 1.0 if opacity contrast is the bug")
    print()
    
    # Test with 5 groups (where we saw the problem before)
    print(f"{'='*80}")
    print(f"Testing with 5 groups (power-law-like opacity contrast)")
    print(f"{'='*80}")
    
    result = test_constant_varying_opacities(n_groups=5, T_bc=0.05, n_cells=1, max_steps=100)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"T/T_bc = {result['T_ratio']:.6f}")
    print(f"Error = {abs(result['T_ratio'] - 1.0):.6f} ({abs(result['T_ratio'] - 1.0)*100:.1f}%)")
    print(f"Converged: {result['converged']}")
    print(f"Emission/Absorption = {result['emission_to_absorption']:.6e}")
    
    print(f"\nOpacity contrast:")
    sigma_max = max(result['sigma_values'])
    sigma_min = min(result['sigma_values'])
    print(f"  σ_max / σ_min = {sigma_max / sigma_min:.6e}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    
    if abs(result['T_ratio'] - 1.0) < 0.1:
        print("✓ WORKS CORRECTLY with constant opacities!")
        print("→ Bug is in temperature-dependent opacity handling")
    elif result['T_ratio'] > 5.0:
        print("✗ STILL FAILS with constant opacities")
        print("→ Bug is in large opacity contrast handling")
    elif result['T_ratio'] < 0.1:
        print("✗ Material frozen (T → 0)")
        print("→ Different failure mode")
    else:
        print("? Partial failure (1 < T/T_bc < 5)")
        print("→ Some improvement but not fully resolved")
    
    # Also test with different group counts
    print(f"\n{'='*80}")
    print("VARYING NUMBER OF GROUPS (constant opacities)")
    print(f"{'='*80}")
    
    print(f"{'Groups':<10} {'T/T_bc':<15} {'Error':<15} {'σ_max/σ_min':<15}")
    print("-"*60)
    
    for n_groups in [2, 3, 4, 5, 6, 8]:
        result = test_constant_varying_opacities(n_groups=n_groups, T_bc=0.05, n_cells=1, max_steps=100)
        error = abs(result['T_ratio'] - 1.0)
        sigma_max = max(result['sigma_values'])
        sigma_min = min(result['sigma_values'])
        contrast = sigma_max / sigma_min
        print(f"{n_groups:<10} {result['T_ratio']:<15.6f} {error:<15.6f} {contrast:<15.6e}")
