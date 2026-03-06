#!/usr/bin/env python3
"""
Diagnose why material doesn't heat with constant opacities.
Check the material energy equation step-by-step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def diagnose_material_heating(n_groups=5, T_bc=0.05, n_cells=1):
    """Diagnose material energy equation with constant opacities"""
    
    r_min = 0.0
    r_max = 1.0e-3
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    dt = 0.01
    rho = 1.0
    cv = 0.05 / rho
    
    # Constant opacities
    sigma_values = []
    for g in range(n_groups):
        E_mid = np.sqrt(energy_edges[g] * energy_edges[g+1])
        sigma_g = 10.0 * rho * (T_bc**(-1/2)) * (E_mid**(-3.0))
        sigma_values.append(sigma_g)
    
    # Create constant opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        sigma_g_fixed = sigma_values[g]
        
        def make_opacity_func(sigma_fixed):
            def opacity_func(T, r):
                return sigma_fixed * np.ones_like(T) if hasattr(T, '__len__') else sigma_fixed
            return opacity_func
        
        def make_diffusion_func(sigma_fixed):
            def diffusion_func(T, r):
                return (1.0 / (3.0 * sigma_fixed)) * np.ones_like(T) if hasattr(T, '__len__') else (1.0 / (3.0 * sigma_fixed))
            return diffusion_func
        
        sigma_funcs.append(make_opacity_func(sigma_g_fixed))
        diff_funcs.append(make_diffusion_func(sigma_g_fixed))
    
    # Emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, sigma_a_groups=np.array(sigma_values))
    
    # BCs - SYMMETRIC
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    def make_left_bc_func(group_idx):
        def left_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[0])
            D_g_base = diff_funcs[group_idx](T_avg, 0.0)  # Returns 1/(3σ)
            if hasattr(D_g_base, '__len__'):
                D_g_base = D_g_base[0]
            D_g = C_LIGHT * D_g_base  # For φ equation, need D = c/(3σ)
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return left_bc
    
    def make_right_bc_func(group_idx):
        def right_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[-1])
            D_g_base = diff_funcs[group_idx](T_avg, r_max)
            if hasattr(D_g_base, '__len__'):
                D_g_base = D_g_base[0]
            D_g = C_LIGHT * D_g_base  # For φ equation, need D = c/(3σ)
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
    
    print(f"{'='*80}")
    print(f"MATERIAL ENERGY EQUATION DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"Configuration: {n_groups} groups, T_bc = {T_bc} keV")
    print(f"Initial T = {T_init} keV (should heat to {T_bc} keV)\n")
    
    # Take one time step and diagnose
    print(f"Before step:")
    print(f"  T = {solver.T[0]:.6e} keV")
    print(f"  E_r = {solver.E_r[0]:.6e} GJ/cm³")
    
    # Get radiation energy density by group
    E_r_g_before = solver.E_r[0] * solver.phi_g_fraction[:, 0]
    print(f"\n  Radiation energy by group (E_r,g):")
    for g in range(n_groups):
        E_r_eq = chi[g] * A_RAD * solver.T[0]**4
        print(f"    Group {g}: E_r,g = {E_r_g_before[g]:.6e}, E_r,eq = {E_r_eq:.6e}, ratio = {E_r_g_before[g]/E_r_eq:.6e}")
    
    # Compute expected heating rate
    print(f"\n  Expected material heating rate:")
    total_absorption = 0.0
    total_emission = 0.0
    for g in range(n_groups):
        abs_g = C_LIGHT * sigma_values[g] * E_r_g_before[g]
        em_g = C_LIGHT * sigma_values[g] * chi[g] * A_RAD * solver.T[0]**4
        total_absorption += abs_g
        total_emission += em_g
        print(f"    Group {g}: absorption = {abs_g:.6e}, emission = {em_g:.6e}")
    
    net_heating = total_absorption - total_emission
    dT_dt_expected = net_heating / (cv * rho)
    
    print(f"\n  Total absorption = {total_absorption:.6e} GJ/(cm³·ns)")
    print(f"  Total emission = {total_emission:.6e} GJ/(cm³·ns)")
    print(f"  Net heating = {net_heating:.6e} GJ/(cm³·ns)")
    print(f"  Expected dT/dt = {dT_dt_expected:.6e} keV/ns")
    print(f"  Expected ΔT after dt={dt} ns: {dT_dt_expected * dt:.6e} keV")
    
    # Take one step
    solver.step()
    
    print(f"\nAfter step:")
    print(f"  T = {solver.T[0]:.6e} keV")
    print(f"  E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"  Actual ΔT = {solver.T[0] - T_init:.6e} keV")
    
    # Get new radiation energy
    E_r_g_after = solver.E_r[0] * solver.phi_g_fraction[:, 0]
    print(f"\n  Radiation energy by group after step:")
    for g in range(n_groups):
        print(f"    Group {g}: E_r,g = {E_r_g_after[g]:.6e}, change = {E_r_g_after[g] - E_r_g_before[g]:.6e}")
    
    # Continue for a few more steps
    print(f"\n{'='*80}")
    print(f"Evolution over multiple steps:")
    print(f"{'='*80}")
    print(f"{'Step':<6} {'T (keV)':<12} {'ΔT':<12} {'E_r total':<12}")
    print("-"*50)
    print(f"{0:<6} {T_init:<12.6e} {0.0:<12.6e} {A_RAD * T_init**4:<12.6e}")
    
    for step in range(1, 21):
        T_old = solver.T[0]
        solver.step()
        delta_T = solver.T[0] - T_old
        print(f"{step:<6} {solver.T[0]:<12.6e} {delta_T:<12.6e} {solver.E_r[0]:<12.6e}")
        
        if abs(delta_T) < 1e-10:
            print(f"\n  → Material temperature FROZEN (no change)")
            break
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS")
    print(f"{'='*80}")
    
    if abs(solver.T[0] - T_init) < 1e-8:
        print(f"✗ Material did NOT heat (T stayed at {solver.T[0]:.6e} keV)")
        print(f"  Expected: T should increase from {T_init} to {T_bc} keV")
        print(f"\n  Possible causes:")
        print(f"  1. Radiation field not being set by boundary conditions")
        print(f"  2. Absorption term in material equation is zero or wrong")
        print(f"  3. Initial radiation field is wrong (should be set by BCs)")
        
        # Check if E_r is being updated
        E_r_expected = A_RAD * T_bc**4  # Should approach this
        E_r_init = A_RAD * T_init**4
        print(f"\n  Radiation energy density:")
        print(f"    Initial (at T_init): {E_r_init:.6e} GJ/cm³")
        print(f"    Current: {solver.E_r[0]:.6e} GJ/cm³")
        print(f"    Expected (at T_bc): {E_r_expected:.6e} GJ/cm³")
        
        if abs(solver.E_r[0] - E_r_init) < 1e-10 * E_r_init:
            print(f"  → Radiation field is NOT changing! BCs may not be working.")
    else:
        print(f"✓ Material is heating correctly")
        print(f"  T changed from {T_init} to {solver.T[0]:.6e} keV")


if __name__ == "__main__":
    diagnose_material_heating(n_groups=5, T_bc=0.05, n_cells=1)
