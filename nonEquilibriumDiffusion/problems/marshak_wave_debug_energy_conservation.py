#!/usr/bin/env python3
"""
ENERGY CONSERVATION DIAGNOSTIC
Focus on understanding why absorption doesn't balance emission in single-zone case.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB

# =============================================================================
# SIMPLE POWER-LAW OPACITY
# =============================================================================

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

# =============================================================================
# ENERGY CONSERVATION ANALYSIS
# =============================================================================

def compute_energy_balance(solver, sigma_funcs, energy_edges, n_groups):
    """Compute detailed energy balance for single zone"""
    
    print("\n" + "="*80)
    print("DETAILED ENERGY BALANCE ANALYSIS")
    print("="*80)
    
    T = solver.T[0]
    E_r_total = solver.E_r[0]
    T_rad = (E_r_total / A_RAD)**0.25
    
    print(f"\nState:")
    print(f"  T_material = {T:.8f} keV")
    print(f"  T_radiation = {T_rad:.8f} keV")
    print(f"  E_r_total = {E_r_total:.12e} GJ/cm³")
    
    # Material energy
    cv = solver.cv * solver.rho
    E_material = cv * T
    print(f"  E_material = c_v*ρ*T = {E_material:.12e} GJ/cm³")
    
    # Compute group-wise quantities
    solver.update_absorption_coefficients(solver.T)
    
    xi_g_list = [solver.compute_source_xi(g, solver.T, solver.t) for g in range(n_groups)]
    
    print(f"\n{'Group':<6} {'E_range(keV)':<20} {'σ_a(cm⁻¹)':<14} {'χ_g':<12} {'φ_g':<14} {'E_r,g':<14} {'E_r,eq':<14}")
    print("-"*110)
    
    phi_groups = []
    E_r_groups = []
    E_r_equilibrium = []
    absorption_groups = []
    emission_groups = []
    
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        
        # Absorption coefficient
        sigma_g = sigma_funcs[g](T, 0.0)
        
        # Emission fraction
        chi_g = solver.chi[g]
        
        # Compute phi_g
        phi_g = solver.compute_phi_g(g, solver.kappa, solver.T, xi_g_list)
        phi_groups.append(phi_g[0])
        
        # Radiation energy in this group
        E_r_g = phi_g[0] / C_LIGHT
        E_r_groups.append(E_r_g)
        
        # Equilibrium radiation energy for this group (when emission = absorption)
        # At equilibrium: E_r,g = χ_g * a * T^4
        E_r_eq_g = chi_g * A_RAD * T**4
        E_r_equilibrium.append(E_r_eq_g)
        
        print(f"{g:<6} [{E_low:6.3f}, {E_high:6.3f}] {sigma_g:<14.6e} {chi_g:<12.6e} {phi_g[0]:<14.6e} {E_r_g:<14.6e} {E_r_eq_g:<14.6e}")
        
        # Absorption rate for this group: c*σ_a,g*E_r,g
        absorption_g = C_LIGHT * sigma_g * E_r_g
        absorption_groups.append(absorption_g)
        
        # Emission rate for this group:
        # In the diffusion equation, the source term is c*σ_a,g*B_g where B_g is the Planck function
        # The Planck function integrated gives: B = (a*c*T^4)/(4π) in each direction
        # But we want the RATE at which material emits into group g, which is:
        # emission_g = c*σ_a,g * (χ_g * a*T^4)
        # where χ_g is the emission fraction and a*T^4 is the radiation energy density at temperature T
        
        emission_g = C_LIGHT * sigma_g * chi_g * A_RAD * T**4
        emission_groups.append(emission_g)
    
    print(f"\nSum of E_r,g: {sum(E_r_groups):.12e} GJ/cm³ (should equal {E_r_total:.12e})")
    print(f"Sum of E_r,eq: {sum(E_r_equilibrium):.12e} GJ/cm³ (equals a*T^4 = {A_RAD * T**4:.12e})")
    
    # Total rates
    absorption_total = sum(absorption_groups)
    emission_total = sum(emission_groups)
    
    print(f"\n" + "-"*80)
    print("GROUP-WISE ABSORPTION AND EMISSION RATES")
    print("-"*80)
    print(f"{'Group':<6} {'Absorption':<18} {'Emission':<18} {'Net (Abs-Em)':<18} {'E_r,g/E_r,eq':<12}")
    print("-"*80)
    
    for g in range(n_groups):
        net = absorption_groups[g] - emission_groups[g]
        ratio = E_r_groups[g] / E_r_equilibrium[g] if E_r_equilibrium[g] > 0 else 0.0
        print(f"{g:<6} {absorption_groups[g]:<18.12e} {emission_groups[g]:<18.12e} {net:<18.12e} {ratio:<12.6e}")
    
    print("-"*80)
    total_ratio = sum(E_r_groups) / sum(E_r_equilibrium) if sum(E_r_equilibrium) > 0 else 0.0
    print(f"{'TOTAL':<6} {absorption_total:<18.12e} {emission_total:<18.12e} {absorption_total - emission_total:<18.12e} {total_ratio:<12.6e}")
    
    # Material heating rate
    dE_material_dt = absorption_total - emission_total
    dT_dt = dE_material_dt / cv
    
    print(f"\nMaterial Energy Balance:")
    print(f"  Total absorption rate: {absorption_total:.12e} GJ/(cm³·ns)")
    print(f"  Total emission rate:   {emission_total:.12e} GJ/(cm³·ns)")
    print(f"  Net heating rate:      {dE_material_dt:.12e} GJ/(cm³·ns)")
    print(f"  c_v*ρ = {cv:.6e} GJ/(cm³·keV)")
    print(f"  dT/dt = {dT_dt:.12e} keV/ns")
    
    # Now check TOTAL energy conservation (radiation + material)
    # For a single zone, the change in total energy should equal the net flux through boundaries
    
    print(f"\n" + "-"*80)
    print("TOTAL ENERGY CONSERVATION CHECK")
    print("-"*80)
    
    # We need to compute the actual flux through boundaries
    # Left BC: F_in (from boundary)
    # Right BC: F_out (should be ~0)
    
    # Compute incoming flux from BC
    T_bc = 0.05  # keV (hardcoded for this test)
    F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    
    chi = solver.chi
    F_in_groups = [chi[g] * F_total_bc for g in range(n_groups)]
    F_in_total = sum(F_in_groups)
    
    print(f"Incoming flux (from BC): F_in = {F_in_total:.12e} GJ/(cm²·ns)")
    
    # For a slab of thickness Δx and area A, power in = F * A
    # In 1D planar geometry, "per unit area" so we just have F
    # But we need to convert flux to volumetric rate
    
    # Actually, for a finite volume of width Δx:
    # Power in = F_in * A
    # Volume = A * Δx
    # Power per unit volume = F_in / Δx
    
    Δx = solver.r_centers[0] * 2  # For single cell centered at r_centers[0]
    if len(solver.r_centers) == 1:
        # Single zone case
        Δx = (solver.r_max - solver.r_min)
    
    power_in_per_volume = F_in_total / Δx
    
    print(f"Cell width: Δx = {Δx:.6e} cm")
    print(f"Power in per unit volume: F_in/Δx = {power_in_per_volume:.12e} GJ/(cm³·ns)")
    
    # At steady state, this should equal the net absorption - emission
    print(f"\nEnergy Balance Check:")
    print(f"  Power in from BC:      {power_in_per_volume:.12e} GJ/(cm³·ns)")
    print(f"  Net absorption - emission: {dE_material_dt:.12e} GJ/(cm³·ns)")
    print(f"  Ratio: (Abs-Em) / (F_in/Δx) = {dE_material_dt / power_in_per_volume:.6e}")
    
    if abs(dE_material_dt / power_in_per_volume - 1.0) < 0.01:
        print(f"  ✓ GOOD: Energy is conserved!")
    else:
        print(f"  ✗ ERROR: Energy is NOT conserved!")
    
    # Check if we're at steady state
    print(f"\nSteady State Check:")
    if abs(dT_dt) < 1e-8:
        print(f"  ✓ AT STEADY STATE: dT/dt = {dT_dt:.6e} keV/ns")
        print(f"  Absorption and emission are balanced.")
    else:
        print(f"  ✗ NOT at steady state: dT/dt = {dT_dt:.6e} keV/ns")
        print(f"  Material is {'heating' if dT_dt > 0 else 'cooling'}.")
    
    print("="*80 + "\n")
    
    return {
        'T': T,
        'T_rad': T_rad,
        'absorption_total': absorption_total,
        'emission_total': emission_total,
        'dT_dt': dT_dt,
        'power_in': power_in_per_volume,
        'absorption_groups': absorption_groups,
        'emission_groups': emission_groups
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_single_zone_test(n_groups=5):
    """Run single zone to near steady-state and check energy balance"""
    
    print("="*80)
    print("SINGLE-ZONE ENERGY CONSERVATION TEST")
    print("="*80)
    
    # Setup
    r_min = 0.0
    r_max = 1.0e-3
    n_cells = 1
    
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    
    dt = 0.01
    max_time = 5.0  # Run until near steady state
    
    rho = 1.0
    cv = 0.05 / rho
    T_bc = 0.05
    
    # Create opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, rho))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, rho))
    
    # Emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    
    sigma_a_groups = np.zeros(n_groups)
    for g in range(n_groups):
        sigma_a_groups[g] = sigma_funcs[g](T_bc, 0.0)
    
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, 
                                                sigma_a_groups=sigma_a_groups)
    
    # Boundary conditions
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
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        emission_fractions=chi,
        rho=rho,
        cv=cv
    )
    
    # Initial condition
    T_init = 0.01
    solver.T = np.array([T_init])
    solver.T_old = solver.T.copy()
    solver.E_r = np.array([A_RAD * T_init**4])
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    solver.t = 0.0
    
    print(f"Configuration: {n_cells} cells, {n_groups} groups")
    print(f"T_bc = {T_bc} keV")
    print(f"T_init = {T_init} keV")
    print(f"Δx = {r_max - r_min:.6e} cm")
    
    # Initial energy balance
    print("\n" + "="*80)
    print("INITIAL STATE (t = 0)")
    print("="*80)
    compute_energy_balance(solver, sigma_funcs, energy_edges, n_groups)
    
    # Evolve
    print(f"\n{'Step':<6} {'Time(ns)':<10} {'T(keV)':<12} {'T/T_bc':<10} {'dT/dt':<14}")
    print("-"*60)
    
    step = 0
    check_steps = [1, 5, 10, 20, 50, 100, 200]
    
    while solver.t < max_time:
        info = solver.step(
            max_newton_iter=50,
            newton_tol=1e-8,
            gmres_tol=1e-8,
            gmres_maxiter=200,
            use_preconditioner=False,
            verbose=False
        )
        
        step += 1
        
        # Compute dT/dt
        dT = solver.T[0] - solver.T_old[0]
        dT_dt = dT / dt
        
        if step % 10 == 0:
            print(f"{step:<6} {solver.t:<10.4f} {solver.T[0]:<12.8f} {solver.T[0]/T_bc:<10.6f} {dT_dt:<14.6e}")
        
        # Detailed diagnostics at specific steps
        if step in check_steps:
            print(f"\n{'='*80}")
            print(f"STEP {step} (t = {solver.t:.4f} ns)")
            print(f"{'='*80}")
            result = compute_energy_balance(solver, sigma_funcs, energy_edges, n_groups)
            
            # Check if steady state
            if abs(result['dT_dt']) < 1e-10:
                print(f"\n*** STEADY STATE REACHED ***")
                print(f"Final T = {solver.T[0]:.8f} keV")
                print(f"T_bc = {T_bc:.8f} keV")
                print(f"T/T_bc = {solver.T[0]/T_bc:.6f}")
                break
        
        solver.advance_time()
    
    # Final state
    print(f"\n{'='*80}")
    print(f"FINAL STATE (t = {solver.t:.4f} ns, step = {step})")
    print(f"{'='*80}")
    final_result = compute_energy_balance(solver, sigma_funcs, energy_edges, n_groups)
    
    return solver, final_result


if __name__ == "__main__":
    solver, result = run_single_zone_test(n_groups=5)
