#!/usr/bin/env python3
"""
Test symmetric BC case with varying number of energy groups
to see if the error depends on the number of groups.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    T_safe = 1e-2
    T_use = np.maximum(T, T_safe)
    return 10.0 * rho * ((T_use)**(-1/2)) * (E)**(-3.0)

def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    def opacity_func(T, r):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func

def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    def diffusion_func(T, r):
        sigma = opacity_func(T, r)
        return 1.0 / (3.0 * sigma)
    return diffusion_func

def test_n_groups(n_groups, T_bc=0.05, n_cells=1, max_steps=100):
    """Test with specified number of groups"""
    
    r_min = 0.0
    r_max = 1.0e-3
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    dt = 0.01
    rho = 1.0
    cv = 0.05 / rho
    
    # Opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        sigma_funcs.append(make_powerlaw_opacity_func(energy_edges[g], energy_edges[g+1], rho))
        diff_funcs.append(make_powerlaw_diffusion_func(energy_edges[g], energy_edges[g+1], rho))
    
    # Emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    sigma_a_groups = np.array([sigma_funcs[g](T_bc, 0.0) for g in range(n_groups)])
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, sigma_a_groups=sigma_a_groups)
    
    # BCs - SYMMETRIC
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    def make_left_bc_func(group_idx):
        def left_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[0])
            D_g = diff_funcs[group_idx](T_avg, 0.0)
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return left_bc
    
    def make_right_bc_func(group_idx):
        def right_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[-1])
            D_g = diff_funcs[group_idx](T_avg, r_max)
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
        sigma_g = np.array([sigma_funcs[g](solver.T[i], solver.r_centers[i]) for i in range(n_cells)])
        absorption_g = C_LIGHT * sigma_g * E_r_g[g, :]
        emission_g = C_LIGHT * sigma_g * chi[g] * A_RAD * solver.T**4
        total_absorption += np.sum(absorption_g) * dx
        total_emission += np.sum(emission_g) * dx
    
    return {
        'T_ratio': T_ratio,
        'converged': converged,
        'chi_sum': np.sum(chi),
        'absorption': total_absorption,
        'emission': total_emission,
        'emission_to_absorption': total_emission / total_absorption if total_absorption > 0 else np.inf
    }


if __name__ == "__main__":
    print("="*80)
    print("TEST: Varying number of energy groups (symmetric BC)")
    print("="*80)
    print(f"\nExpected result: T/T_bc = 1.0 for all cases")
    print(f"Configuration: T_bc = 0.05 keV, n_cells = 1")
    print()
    
    n_groups_list = [2, 3, 4, 5, 6, 8, 10]
    results = []
    
    for n_groups in n_groups_list:
        print(f"\n{'='*80}")
        print(f"Testing with {n_groups} groups...")
        print(f"{'='*80}")
        
        result = test_n_groups(n_groups, T_bc=0.05, n_cells=1, max_steps=100)
        results.append((n_groups, result))
        
        print(f"  T/T_bc = {result['T_ratio']:.6f}")
        print(f"  Error = {abs(result['T_ratio'] - 1.0):.6f}")
        print(f"  Converged: {result['converged']}")
        print(f"  Sum(χ_g) = {result['chi_sum']:.10f}")
        print(f"  Emission/Absorption = {result['emission_to_absorption']:.6e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Groups':<10} {'T/T_bc':<15} {'Error':<15} {'Em/Abs':<15}")
    print("-"*60)
    
    for n_groups, result in results:
        error = abs(result['T_ratio'] - 1.0)
        em_abs = result['emission_to_absorption']
        print(f"{n_groups:<10} {result['T_ratio']:<15.6f} {error:<15.6f} {em_abs:<15.6e}")
    
    # Check if there's a pattern
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    errors = [abs(result['T_ratio'] - 1.0) for _, result in results]
    if all(e < 0.01 for e in errors):
        print("✓ All cases converge correctly (error < 1%)")
    elif errors[0] < 0.01 and errors[-1] > 1.0:
        print("✗ Error increases with number of groups")
        print(f"  2 groups: error = {errors[0]:.6f}")
        print(f"  {n_groups_list[-1]} groups: error = {errors[-1]:.6f}")
    else:
        print(f"✗ Errors are inconsistent across group counts")
        print(f"  Min error: {min(errors):.6f} at {n_groups_list[errors.index(min(errors))]} groups")
        print(f"  Max error: {max(errors):.6f} at {n_groups_list[errors.index(max(errors))]} groups")
