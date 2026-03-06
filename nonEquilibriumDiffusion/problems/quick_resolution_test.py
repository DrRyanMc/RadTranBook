#!/usr/bin/env python3
"""
Quick test: Single zone vs 2 zones with HIGH drive temperature
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

def quick_test(n_cells, T_bc, n_groups=5):
    """Quick test to steady state"""
    
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
    
    # BCs
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    def make_left_bc_func(group_idx):
        def left_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[0])
            D_g = diff_funcs[group_idx](T_avg, 0.0)
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return left_bc
    
    def right_bc_func(phi, r):
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [make_left_bc_func(g) for g in range(n_groups)]
    right_bc_funcs = [right_bc_func] * n_groups
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs, right_bc_funcs=right_bc_funcs,
        emission_fractions=chi, rho=rho, cv=cv
    )
    
    # Initial condition - start warm
    T_init = 0.5 * T_bc
    solver.T = T_init * np.ones(n_cells)
    solver.T_old = solver.T.copy()
    solver.E_r = A_RAD * T_init**4 * np.ones(n_cells)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    solver.t = 0.0
    
    print(f"\nn_cells = {n_cells}, T_bc = {T_bc} keV, Δx = {(r_max-r_min)/n_cells:.6e} cm")
    print(f"{'Step':<8} {'Time':<10} {'T_avg':<12} {'T/T_bc':<10}")
    print("-"*50)
    
    # Run
    step = 0
    prev_T_max = solver.T.max()
    
    while step < 200:
        info = solver.step(max_newton_iter=50, newton_tol=1e-8, gmres_tol=1e-8,
                          gmres_maxiter=200, use_preconditioner=False, verbose=False)
        step += 1
        
        # Check convergence
        dT_max = abs(solver.T.max() - prev_T_max)
        if dT_max < 1e-8 and step > 20:
            T_avg = np.mean(solver.T)
            print(f"{step:<8} {solver.t:<10.4f} {T_avg:<12.8f} {T_avg/T_bc:<10.6f}  *** CONVERGED ***")
            break
        
        prev_T_max = solver.T.max()
        
        if step % 20 == 0:
            T_avg = np.mean(solver.T)
            print(f"{step:<8} {solver.t:<10.4f} {T_avg:<12.8f} {T_avg/T_bc:<10.6f}")
        
        solver.advance_time()
    
    T_final = np.mean(solver.T)
    return T_final / T_bc

if __name__ == "__main__":
    print("="*80)
    print("QUICK TEST: Single Zone vs Multiple Zones with Different Drive Temperatures")
    print("="*80)
    
    test_cases = [
        (0.05, "Low drive temp (baseline)"),
        (0.2, "High drive temp (4x)"),
        (0.5, "Very high drive temp (10x)")
    ]
    
    for T_bc, label in test_cases:
        print(f"\n{'='*80}")
        print(f"{label}: T_bc = {T_bc} keV")
        print("="*80)
        
        ratio_1 = quick_test(n_cells=1, T_bc=T_bc)
        ratio_2 = quick_test(n_cells=2, T_bc=T_bc)
        
        print(f"\nSummary for T_bc = {T_bc} keV:")
        print(f"  1 cell:  T/T_bc = {ratio_1:.6f}")
        print(f"  2 cells: T/T_bc = {ratio_2:.6f}")
        print(f"  Improvement: {abs(ratio_1 - 1.0):.4f} → {abs(ratio_2 - 1.0):.4f} (error reduced by {abs(ratio_1-1.0)/abs(ratio_2-1.0):.1f}x)")
