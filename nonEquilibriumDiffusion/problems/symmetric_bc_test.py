#!/usr/bin/env python3
"""
Test Marshak wave with SYMMETRIC boundary conditions on both sides.
Both boundaries held at T_bc, so steady state should give uniform T = T_bc.
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

def symmetric_bc_test(n_cells, T_bc, n_groups=5, max_steps=200):
    """Test with both boundaries at same temperature"""
    
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
    
    # BCs - SYMMETRIC: both sides at T_bc
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
    right_bc_funcs = [make_right_bc_func(g) for g in range(n_groups)]  # SYMMETRIC!
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs, right_bc_funcs=right_bc_funcs,
        emission_fractions=chi, rho=rho, cv=cv
    )
    
    # Initial condition: start at half the BC temperature
    T_init = 0.5 * T_bc
    solver.T[:] = T_init
    
    # Initialize radiation field to equilibrium at T_init
    solver.E_r[:] = A_RAD * T_init**4
    solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
    solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * T_init**4
    
    print(f"\nn_cells = {n_cells}, T_bc = {T_bc} keV, Δx = {(r_max-r_min)/n_cells:.6e} cm")
    print(f"Both boundaries at same temperature (symmetric)")
    print("Expected steady state: T = T_bc everywhere")
    print(f"\nStep     Time       T_avg        T/T_bc    ")
    print("-" * 50)
    
    converged = False
    T_old = solver.T.copy()
    
    for step in range(max_steps):
        solver.step()
        
        if step % 20 == 19:
            T_avg = np.mean(solver.T)
            T_ratio = T_avg / T_bc
            print(f"{step+1:<8} {solver.t:<10.4f} {T_avg:<12.8f} {T_ratio:<9.6f}")
        
        # Check convergence
        if step > 10:
            T_change = np.max(np.abs(solver.T - T_old))
            if T_change < 1e-6 * T_bc:
                T_avg = np.mean(solver.T)
                T_ratio = T_avg / T_bc
                print(f"{step+1:<8} {solver.t:<10.4f} {T_avg:<12.8f} {T_ratio:<9.6f}  *** CONVERGED ***")
                converged = True
                break
            T_old = solver.T.copy()
    
    T_final = np.mean(solver.T)
    T_ratio = T_final / T_bc
    
    return T_ratio, converged


if __name__ == "__main__":
    print("=" * 80)
    print("SYMMETRIC BC TEST: Both boundaries at same temperature")
    print("Expected result: T/T_bc = 1.0 at steady state")
    print("=" * 80)
    
    T_bc_values = [0.05, 0.2, 0.5]
    n_cells_values = [1, 2, 5]
    
    results = {}
    
    for T_bc in T_bc_values:
        print(f"\n{'='*80}")
        print(f"T_bc = {T_bc} keV")
        print(f"{'='*80}")
        
        results[T_bc] = {}
        
        for n_cells in n_cells_values:
            T_ratio, converged = symmetric_bc_test(n_cells, T_bc, n_groups=5, max_steps=200)
            results[T_bc][n_cells] = T_ratio
            
            if not converged:
                print(f"  WARNING: Did not converge in 200 steps")
        
        print(f"\nSummary for T_bc = {T_bc} keV:")
        for n_cells in n_cells_values:
            error = abs(results[T_bc][n_cells] - 1.0)
            print(f"  {n_cells} cell(s):  T/T_bc = {results[T_bc][n_cells]:.6f}  (error = {error:.6f})")
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"{'T_bc (keV)':<12} {'1 cell':<15} {'2 cells':<15} {'5 cells':<15}")
    print("-" * 60)
    for T_bc in T_bc_values:
        row = f"{T_bc:<12.2f}"
        for n_cells in n_cells_values:
            ratio = results[T_bc][n_cells]
            row += f" {ratio:<15.6f}"
        print(row)
