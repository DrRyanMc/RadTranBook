#!/usr/bin/env python3
"""
Diagnostic version: Show detailed energy balance for symmetric BC case
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB

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

def diagnostic_test(n_cells, T_bc, n_groups=5, max_steps=100):
    """Test with detailed energy diagnostics"""
    
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
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC TEST: n_cells={n_cells}, T_bc={T_bc} keV")
    print(f"{'='*80}")
    print(f"Cell width: Δx = {(r_max-r_min)/n_cells:.6e} cm")
    print(f"Expected steady state: T = {T_bc} keV everywhere")
    print(f"Total boundary flux: F_total = {F_total:.6e} GJ/(cm²·ns)")
    print(f"Emission fractions χ: {chi}")
    
    # Run to steady state
    T_old = solver.T.copy()
    for step in range(max_steps):
        solver.step()
        
        if step > 10:
            T_change = np.max(np.abs(solver.T - T_old))
            if T_change < 1e-6 * T_bc:
                print(f"\nConverged at step {step+1}, time = {solver.t:.4f} ns")
                break
            T_old = solver.T.copy()
    
    # Compute detailed diagnostics
    T_avg = np.mean(solver.T)
    print(f"\n{'='*80}")
    print(f"STEADY STATE DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Average temperature: T_avg = {T_avg:.6e} keV")
    print(f"Boundary temperature: T_bc  = {T_bc:.6e} keV")
    print(f"Ratio: T_avg/T_bc = {T_avg/T_bc:.6f}")
    print(f"ERROR: T_avg - T_bc = {T_avg - T_bc:.6e} keV ({(T_avg/T_bc - 1.0)*100:.2f}%)")
    
    # Get radiation field
    # phi_g_fraction is (n_groups, n_cells)
    # E_r is (n_cells,)
    phi_total = solver.E_r * C_LIGHT
    phi_g = phi_total[np.newaxis, :] * solver.phi_g_fraction  # (n_groups, n_cells)
    E_r_g = solver.E_r[np.newaxis, :] * solver.phi_g_fraction  # (n_groups, n_cells)
    
    print(f"\n{'-'*80}")
    print(f"GROUP-BY-GROUP ENERGY BALANCE")
    print(f"{'-'*80}")
    
    # For each group, compute absorption and emission
    total_absorption = np.zeros(n_groups)
    total_emission = np.zeros(n_groups)
    E_r_equilibrium = np.zeros(n_groups)
    
    for g in range(n_groups):
        # Get opacity at average material temperature
        sigma_g = np.array([sigma_funcs[g](solver.T[i], solver.r_centers[i]) for i in range(n_cells)])
        
        # Absorption: c * sigma_a,g * E_r,g  [GJ/(cm³·ns)]
        absorption_g = C_LIGHT * sigma_g * E_r_g[g, :]
        
        # Emission: c * sigma_a,g * chi_g * a * T^4  [GJ/(cm³·ns)]
        emission_g = C_LIGHT * sigma_g * chi[g] * A_RAD * solver.T**4
        
        # Total over all cells (multiply by cell volume)
        dx = (r_max - r_min) / n_cells
        total_absorption[g] = np.sum(absorption_g) * dx  # [GJ/(cm²·ns)]
        total_emission[g] = np.sum(emission_g) * dx      # [GJ/(cm²·ns)]
        
        # Equilibrium E_r,g for this group at material temperature
        E_r_eq_g = chi[g] * A_RAD * solver.T**4
        E_r_equilibrium[g] = np.mean(E_r_eq_g)
        
        print(f"\nGroup {g}: E ∈ [{energy_edges[g]:.4e}, {energy_edges[g+1]:.4e}] keV")
        print(f"  χ_{g} = {chi[g]:.6e}")
        print(f"  <E_r,g> = {np.mean(E_r_g[g, :]):.6e} GJ/cm³")
        print(f"  <E_r,eq,g> = {E_r_equilibrium[g]:.6e} GJ/cm³  (at T_avg)")
        print(f"  E_r,g / E_r,eq,g = {np.mean(E_r_g[g, :]) / E_r_equilibrium[g]:.6e}")
        print(f"  Absorption = {total_absorption[g]:.6e} GJ/(cm²·ns)")
        print(f"  Emission   = {total_emission[g]:.6e} GJ/(cm²·ns)")
        print(f"  Net (Abs - Em) = {total_absorption[g] - total_emission[g]:.6e} GJ/(cm²·ns)")
    
    print(f"\n{'-'*80}")
    print(f"BOUNDARY ENERGY FLUX")
    print(f"{'-'*80}")
    
    # Compute boundary fluxes
    # At steady state with symmetric BCs, net flux should be zero
    
    # Left boundary (incoming from boundary)
    left_flux_by_group = np.zeros(n_groups)
    right_flux_by_group = np.zeros(n_groups)
    
    for g in range(n_groups):
        # Get BC parameters at left boundary
        A_left, B_left, C_left = left_bc_funcs[g](phi_g[g, 0], r_min)
        # BC: A*phi + B*dphi/dr = C
        # Flux at boundary: J = (1/2)*phi - (1/2)*B*dphi/dr
        # From BC: dphi/dr = (C - A*phi) / B
        dphi_dr_left = (C_left - A_left * phi_g[g, 0]) / B_left
        J_left = 0.5 * phi_g[g, 0] - 0.5 * B_left * dphi_dr_left
        left_flux_by_group[g] = J_left
        
        # Right boundary
        A_right, B_right, C_right = right_bc_funcs[g](phi_g[g, -1], r_max)
        dphi_dr_right = (C_right - A_right * phi_g[g, -1]) / B_right
        J_right = 0.5 * phi_g[g, -1] - 0.5 * B_right * dphi_dr_right
        right_flux_by_group[g] = -J_right  # Negative because outgoing
        
        print(f"\nGroup {g}:")
        print(f"  Left BC:  C_g = {C_left:.6e} GJ/(cm²·ns)")
        print(f"  Left flux (in):  J_left  = {J_left:.6e} GJ/(cm²·ns)")
        print(f"  Right flux (out): J_right = {right_flux_by_group[g]:.6e} GJ/(cm²·ns)")
        print(f"  Net flux into domain = {J_left + right_flux_by_group[g]:.6e} GJ/(cm²·ns)")
    
    print(f"\n{'-'*80}")
    print(f"TOTAL ENERGY BALANCE")
    print(f"{'-'*80}")
    
    total_abs = np.sum(total_absorption)
    total_em = np.sum(total_emission)
    total_left_flux = np.sum(left_flux_by_group)
    total_right_flux = np.sum(right_flux_by_group)
    total_boundary_flux = total_left_flux + total_right_flux
    net_material_heating = total_abs - total_em
    
    print(f"\nTotal absorption (all groups):  {total_abs:.6e} GJ/(cm²·ns)")
    print(f"Total emission (all groups):    {total_em:.6e} GJ/(cm²·ns)")
    print(f"Net material heating (Abs-Em):  {net_material_heating:.6e} GJ/(cm²·ns)")
    print(f"\nTotal flux from left boundary:  {total_left_flux:.6e} GJ/(cm²·ns)")
    print(f"Total flux from right boundary: {total_right_flux:.6e} GJ/(cm²·ns)")
    print(f"Net boundary flux into domain:  {total_boundary_flux:.6e} GJ/(cm²·ns)")
    print(f"\nExpected at steady state:")
    print(f"  Net material heating should be ≈ 0 (converged)")
    print(f"  Net boundary flux should be ≈ 0 (symmetric)")
    
    # Check sum of chi
    print(f"\n{'-'*80}")
    print(f"EMISSION FRACTION CHECK")
    print(f"{'-'*80}")
    print(f"Sum of χ_g = {np.sum(chi):.10f} (should be exactly 1.0)")
    
    # Compare total emission to Planck function
    F_planck_T_avg = 0.5 * A_RAD * C_LIGHT * T_avg**4
    F_planck_T_bc = 0.5 * A_RAD * C_LIGHT * T_bc**4
    
    print(f"\n{'-'*80}")
    print(f"PLANCK FUNCTION COMPARISON")
    print(f"{'-'*80}")
    print(f"Expected boundary flux (at T_bc): {F_planck_T_bc:.6e} GJ/(cm²·ns)")
    print(f"Expected material flux (at T_avg): {F_planck_T_avg:.6e} GJ/(cm²·ns)")
    print(f"Sum of emission by group: {total_em:.6e} GJ/(cm²·ns)")
    print(f"Ratio (emission / Planck@T_avg): {total_em / (dx * rho * C_LIGHT * A_RAD * T_avg**4 * np.sum(sigma_a_groups)):.6f}")


if __name__ == "__main__":
    # Test single case with detailed diagnostics
    print("="*80)
    print("SYMMETRIC BC ENERGY BALANCE DIAGNOSTIC")
    print("="*80)
    
    # Start with low temperature and 1 cell to see the problem clearly
    diagnostic_test(n_cells=1, T_bc=0.05, n_groups=5, max_steps=100)
    
    # Also test with 2 cells
    diagnostic_test(n_cells=2, T_bc=0.05, n_groups=5, max_steps=100)
