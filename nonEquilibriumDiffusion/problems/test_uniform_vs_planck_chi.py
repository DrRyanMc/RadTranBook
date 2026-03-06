#!/usr/bin/env python3
"""
Test symmetric BC with UNIFORM emission fractions (like Su-Olson)
vs PLANCK emission fractions (like power-law case)
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

def test_emission_fractions(n_groups, use_planck, T_bc=0.05, n_cells=1, max_steps=100):
    """Test with uniform vs Planck emission fractions"""
    
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
    
    # Emission fractions: UNIFORM vs PLANCK
    if use_planck:
        from multigroup_diffusion_solver import compute_emission_fractions_from_edges
        sigma_a_groups = np.array([sigma_funcs[g](T_bc, 0.0) for g in range(n_groups)])
        chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, sigma_a_groups=sigma_a_groups)
        method = "Planck"
    else:
        chi = np.ones(n_groups) / n_groups  # UNIFORM (like Su-Olson)
        method = "Uniform"
    
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
    
    return {
        'T_ratio': T_ratio,
        'converged': converged,
        'method': method,
        'chi': chi
    }


if __name__ == "__main__":
    print("="*80)
    print("TEST: Uniform vs Planck Emission Fractions")
    print("="*80)
    print(f"\nConfiguration: T_bc = 0.05 keV, n_cells = 1, symmetric BCs")
    print(f"Expected result: T/T_bc = 1.0 for both cases\n")
    
    n_groups_list = [2, 3, 4, 5, 6, 8]
    
    print(f"{'Groups':<10} {'Uniform χ':<20} {'Planck χ':<20}")
    print(f"{'':10} {'T/T_bc':>10} {'Error':>8} {'T/T_bc':>10} {'Error':>8}")
    print("-"*70)
    
    for n_groups in n_groups_list:
        # Test with uniform emission fractions
        result_uniform = test_emission_fractions(n_groups, use_planck=False, T_bc=0.05)
        
        # Test with Planck emission fractions
        result_planck = test_emission_fractions(n_groups, use_planck=True, T_bc=0.05)
        
        error_uniform = abs(result_uniform['T_ratio'] - 1.0)
        error_planck = abs(result_planck['T_ratio'] - 1.0)
        
        print(f"{n_groups:<10} {result_uniform['T_ratio']:>10.6f} {error_uniform:>8.4f} "
              f"{result_planck['T_ratio']:>10.6f} {error_planck:>8.4f}")
    
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON FOR 5 GROUPS")
    print(f"{'='*80}")
    
    result_uniform_5 = test_emission_fractions(5, use_planck=False, T_bc=0.05)
    result_planck_5 = test_emission_fractions(5, use_planck=True, T_bc=0.05)
    
    print(f"\nUniform emission fractions (χ_g = 1/5 = 0.2 for all g):")
    print(f"  χ = {result_uniform_5['chi']}")
    print(f"  T/T_bc = {result_uniform_5['T_ratio']:.6f}")
    print(f"  Error = {abs(result_uniform_5['T_ratio'] - 1.0):.6f}")
    
    print(f"\nPlanck emission fractions (from Planck function):")
    print(f"  χ = {result_planck_5['chi']}")
    print(f"  T/T_bc = {result_planck_5['T_ratio']:.6f}")
    print(f"  Error = {abs(result_planck_5['T_ratio'] - 1.0):.6f}")
