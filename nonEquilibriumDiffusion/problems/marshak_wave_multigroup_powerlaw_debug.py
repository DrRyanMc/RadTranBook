#!/usr/bin/env python3
"""
HEAVILY INSTRUMENTED DEBUG VERSION
Marshak Wave Problem - Multigroup Version with Power-Law Opacity

PURPOSE: Diagnose why steady-state temperature doesn't match boundary temperature
in single-zone configuration.

Key Debug Features:
- Detailed energy balance tracking
- Material-radiation coupling diagnostics
- BC parameter evolution
- Group-wise contributions
- Newton iteration details
- Convergence metrics per timestep
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB

# Physical constants
RHO = 1.0  # g/cm³

# =============================================================================
# POWER-LAW OPACITY FUNCTIONS
# =============================================================================

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """Power-law opacity at specific energy
    
    σ_a(T,E) = 10.0 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
    """
    T_safe = 1e-2  # Minimum temperature to avoid singularity
    T_use = np.maximum(T, T_safe)
    return 10.0 * rho * ((T_use)**(-1/2)) * (E)**(-3.0)


def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    """Create opacity function for a group using geometric mean at boundaries"""
    def opacity_func(T, r):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    """Create diffusion coefficient function for a group"""
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    
    def diffusion_func(T, r):
        sigma = opacity_func(T, r)
        return 1.0 / (3.0 * sigma)
    return diffusion_func


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def print_energy_balance(solver, T_bc_current, chi_current, sigma_funcs, diff_funcs, energy_edges, n_groups):
    """Print detailed energy balance diagnostics"""
    print("\n" + "="*80)
    print("ENERGY BALANCE DIAGNOSTICS")
    print("="*80)
    
    T = solver.T[0]
    T_array = solver.T  # Keep array version for solver functions
    E_r = solver.E_r[0]
    T_rad = (E_r / A_RAD)**0.25
    
    print(f"Material Temperature:   T = {T:.8f} keV")
    print(f"Radiation Temperature:  T_rad = {T_rad:.8f} keV")
    print(f"Radiation Energy:       E_r = {E_r:.12e} GJ/cm³")
    print(f"Expected E_r(T):        a*T^4 = {A_RAD * T**4:.12e} GJ/cm³")
    print(f"BC Temperature:         T_bc = {T_bc_current:.8f} keV")
    print(f"Expected E_r(T_bc):     a*T_bc^4 = {A_RAD * T_bc_current**4:.12e} GJ/cm³")
    
    # Temperature ratios
    print(f"\nTemperature Ratios:")
    print(f"  T/T_bc = {T/T_bc_current:.6f}")
    print(f"  T_rad/T_bc = {T_rad/T_bc_current:.6f}")
    print(f"  T_rad/T = {T_rad/T:.6f}")
    
    # Incoming flux from BC
    F_total_bc = (A_RAD * C_LIGHT * T_bc_current**4) / 2.0
    print(f"\nBoundary Condition:")
    print(f"  Incoming flux: F_in = (a*c*T_bc^4)/2 = {F_total_bc:.12e} GJ/(cm²·ns)")
    
    # Emission from material
    P_emit_total = 4 * np.pi * A_RAD * C_LIGHT * T**4
    print(f"\nMaterial Emission:")
    print(f"  Integrated emission: ∫ B dν = π*a*c*T^4 = {np.pi * A_RAD * C_LIGHT * T**4:.12e} GJ/(cm²·ns·sr)")
    print(f"  Total emission (4π): 4π*a*c*T^4 = {P_emit_total:.12e} GJ/(cm³·ns)")
    
    # Group-wise diagnostics
    print(f"\nGroup-wise Analysis:")
    print(f"{'Group':<6} {'E_range(keV)':<20} {'chi':<12} {'σ_a(cm⁻¹)':<14} {'D(cm)':<14} {'F_in':<14}")
    print("-"*90)
    
    F_in_sum = 0.0
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        chi_g = chi_current[g]
        sigma_g = sigma_funcs[g](T, 0.0)
        D_g = diff_funcs[g](T, 0.0)
        F_in_g = chi_g * F_total_bc
        F_in_sum += F_in_g
        
        print(f"{g:<6} [{E_low:6.3f}, {E_high:6.3f}] {chi_g:<12.6e} {sigma_g:<14.6e} {D_g:<14.6e} {F_in_g:<14.6e}")
    
    print(f"\nSum of F_in_g: {F_in_sum:.12e} (should equal {F_total_bc:.12e})")
    print(f"Chi sum: {np.sum(chi_current):.12e} (should equal 1.0)")
    
    # Absorption rate
    print(f"\nAbsorption/Emission Balance:")
    solver.update_absorption_coefficients(T_array)
    c_sigma_P_total = 0.0
    fleck = solver.compute_fleck_factor(T_array)
    print(f"  Fleck factor: f = {fleck[0]:.12e}")
    
    # Compute xi for all groups
    xi_g_list = [solver.compute_source_xi(g, T_array, solver.t) for g in range(n_groups)]
    
    for g in range(n_groups):
        sigma_g = sigma_funcs[g](T, 0.0)
        # Get phi for this group
        phi_g = solver.compute_phi_g(g, solver.kappa, T_array, xi_g_list)
        c_sigma_P = C_LIGHT * sigma_g * phi_g[0]
        c_sigma_P_total += c_sigma_P
        print(f"  Group {g}: c*σ_a*φ = {c_sigma_P:.12e} GJ/(cm³·ns), φ = {phi_g[0]:.12e}")
    
    print(f"  Total absorption: Σ c*σ_a*φ = {c_sigma_P_total:.12e} GJ/(cm³·ns)")
    print(f"  Total emission: 4π*a*c*T^4 = {P_emit_total:.12e} GJ/(cm³·ns)")
    print(f"  Net: emission - absorption = {P_emit_total - c_sigma_P_total:.12e} GJ/(cm³·ns)")
    
    # Material energy rate
    cv = solver.cv * solver.rho
    dT_dt = (c_sigma_P_total - P_emit_total) / cv
    print(f"\nMaterial Energy Rate:")
    print(f"  c_v*ρ = {cv:.6e} GJ/(cm³·keV)")
    print(f"  dT/dt = (absorption - emission)/(c_v*ρ) = {dT_dt:.12e} keV/ns")
    print(f"  Estimated T after 1 timestep: {T + dT_dt * solver.dt:.8f} keV")
    
    print("="*80 + "\n")


def print_bc_diagnostics(solver, bc_time, T_bc_current, chi_current, diff_funcs, energy_edges, n_groups):
    """Print boundary condition diagnostics"""
    print("\n" + "-"*80)
    print("BOUNDARY CONDITION DIAGNOSTICS")
    print("-"*80)
    print(f"Time: t = {bc_time:.6f} ns")
    print(f"BC Temperature: T_bc = {T_bc_current:.8f} keV")
    print(f"Material Temperature (cell 0): T = {solver.T[0]:.8f} keV")
    
    F_total = (A_RAD * C_LIGHT * T_bc_current**4) / 2.0
    print(f"Total incoming flux: F_total = {F_total:.12e} GJ/(cm²·ns)")
    
    # Average temperature for diffusion coefficient
    T_avg_bc = 0.5 * (T_bc_current + solver.T[0])
    print(f"Average temperature for D: T_avg = {T_avg_bc:.8f} keV")
    
    print(f"\n{'Group':<6} {'chi':<12} {'D(cm)':<14} {'F_in':<14} {'BC: A*phi + B*dphi/dr = C':<40}")
    print("-"*110)
    
    for g in range(n_groups):
        chi_g = chi_current[g]
        D_g = diff_funcs[g](T_avg_bc, 0.0)
        F_in_g = chi_g * F_total
        A = 0.5
        B = D_g
        C = F_in_g
        
        print(f"{g:<6} {chi_g:<12.6e} {D_g:<14.6e} {F_in_g:<14.6e} {A:.1f}*phi + {B:.6e}*dphi/dr = {C:.6e}")
    
    print("-"*80 + "\n")


def print_newton_diagnostics(solver, info, step):
    """Print Newton iteration diagnostics"""
    print(f"\nNewton Iteration Summary (Step {step}):")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['newton_iter']}")
    if 'final_residual' in info:
        print(f"  Final residual: {info['final_residual']:.6e}")
    print(f"  GMRES iterations: {info['gmres_info']['iterations']}")
    if 'residual' in info['gmres_info']:
        print(f"  GMRES residual: {info['gmres_info']['residual']:.6e}")
    
    if not info['converged']:
        print("  WARNING: Newton did not converge!")


# =============================================================================
# MARSHAK WAVE SIMULATION (DEBUG VERSION)
# =============================================================================

def run_marshak_wave_debug(use_preconditioner=False, n_groups=10):
    """Run heavily instrumented version for debugging"""
    
    print("="*80)
    print("MARSHAK WAVE PROBLEM - DEBUG VERSION")
    print("="*80)
    print("Single-zone steady-state test")
    print("Expect: Material temperature should approach boundary temperature")
    print("="*80)
    
    # Problem setup
    r_min = 0.0
    r_max = 1.0e-3  # 1 mm slab
    n_cells = 1     # SINGLE ZONE for steady-state test
    
    # Energy group structure
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    
    # Time stepping
    dt = 0.01  # ns
    max_time = 40.0  # ns - long enough for steady state
    
    # Material properties
    rho = RHO
    cv = 0.05 / rho  # GJ/(g·keV)
    
    # Boundary temperature - CONSTANT for easier diagnosis
    T_bc = 0.05  # keV
    
    print(f"\nConfiguration:")
    print(f"  Domain: [{r_min}, {r_max}] cm")
    print(f"  Cells: {n_cells}")
    print(f"  Groups: {n_groups}")
    print(f"  Timestep: dt = {dt} ns")
    print(f"  Max time: {max_time} ns")
    print(f"  Boundary temp: T_bc = {T_bc} keV (CONSTANT)")
    print(f"  Heat capacity: c_v = {cv} GJ/(g·keV)")
    print(f"  Density: ρ = {rho} g/cm³")
    
    # Create opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, rho))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, rho))
    
    # Compute emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    
    sigma_a_groups = np.zeros(n_groups)
    for g in range(n_groups):
        sigma_a_groups[g] = sigma_funcs[g](T_bc, 0.0)
    
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, 
                                                sigma_a_groups=sigma_a_groups)
    
    print(f"\nEnergy Groups:")
    for g in range(n_groups):
        print(f"  Group {g:2d}: [{energy_edges[g]:8.4f}, {energy_edges[g+1]:8.4f}] keV, "
              f"χ = {chi[g]:.6e}, σ_a = {sigma_a_groups[g]:.6e} cm⁻¹")
    
    # Boundary conditions
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    print(f"\nBoundary Condition Setup:")
    print(f"  Type: Marshak BC (Robin BC: A*phi + B*dphi/dr = C)")
    print(f"  A = {BC_A}")
    print(f"  Incoming flux: F_total = {F_total:.12e} GJ/(cm²·ns)")
    
    # Create BC functions
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
    print(f"\nInitializing solver...")
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
    
    # Initial condition - COLD START
    T_init = 0.01  # keV - cold
    solver.T = np.array([T_init])
    solver.T_old = solver.T.copy()
    solver.E_r = np.array([A_RAD * T_init**4])
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    solver.t = 0.0
    
    print(f"Initial condition: T = {T_init} keV, E_r = {solver.E_r[0]:.12e} GJ/cm³")
    
    # Print initial energy balance
    print_energy_balance(solver, T_bc, chi, sigma_funcs, diff_funcs, energy_edges, n_groups)
    print_bc_diagnostics(solver, 0.0, T_bc, chi, diff_funcs, energy_edges, n_groups)
    
    # Time evolution with DETAILED logging
    print("\n" + "="*80)
    print("TIME EVOLUTION")
    print("="*80)
    
    step = 0
    diagnostic_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    print(f"\n{'Step':<6} {'Time':<8} {'T(keV)':<12} {'T_rad':<12} {'T/T_bc':<10} {'dT/dt':<12} {'Newton':<7} {'GMRES':<7}")
    print("-"*90)
    
    T_prev = solver.T[0]
    
    while solver.t < max_time:
        # Take timestep
        info = solver.step(
            max_newton_iter=50,
            newton_tol=1e-8,
            gmres_tol=1e-8,
            gmres_maxiter=200,
            use_preconditioner=use_preconditioner,
            max_relative_change=1.0,
            verbose=False
        )
        
        step += 1
        
        # Compute dT/dt
        dT_dt = (solver.T[0] - T_prev) / dt
        T_prev = solver.T[0]
        
        # Current radiation temperature
        T_rad = (solver.E_r[0] / A_RAD)**0.25
        
        # Print summary
        print(f"{step:<6} {solver.t:<8.4f} {solver.T[0]:<12.8f} {T_rad:<12.8f} "
              f"{solver.T[0]/T_bc:<10.6f} {dT_dt:<12.6e} {info['newton_iter']:<7} "
              f"{info['gmres_info']['iterations']:<7}")
        
        # Detailed diagnostics at specific steps
        if step in diagnostic_steps:
            print("\n" + "+"*80)
            print(f"DETAILED DIAGNOSTICS AT STEP {step}")
            print("+"*80)
            
            print_newton_diagnostics(solver, info, step)
            print_bc_diagnostics(solver, solver.t, T_bc, chi, diff_funcs, energy_edges, n_groups)
            print_energy_balance(solver, T_bc, chi, sigma_funcs, diff_funcs, energy_edges, n_groups)
            
            # Check if we're at steady state
            if abs(dT_dt) < 1e-10:
                print("\n*** STEADY STATE REACHED ***")
                print(f"dT/dt = {dT_dt:.6e} keV/ns (< 1e-10)")
                print(f"Final T = {solver.T[0]:.8f} keV")
                print(f"T_bc = {T_bc:.8f} keV")
                print(f"Difference: T - T_bc = {solver.T[0] - T_bc:.8e} keV")
                print(f"Relative error: (T - T_bc)/T_bc = {(solver.T[0] - T_bc)/T_bc:.8e}")
                break
            
            print("+"*80 + "\n")
        
        solver.advance_time()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL STATE SUMMARY")
    print("="*80)
    print(f"Total steps: {step}")
    print(f"Final time: {solver.t:.4f} ns")
    print(f"Final T: {solver.T[0]:.8f} keV")
    print(f"T_bc: {T_bc:.8f} keV")
    print(f"Difference: {solver.T[0] - T_bc:.8e} keV")
    print(f"Relative error: {(solver.T[0] - T_bc)/T_bc:.8e}")
    
    # One more detailed energy balance
    print_energy_balance(solver, T_bc, chi, sigma_funcs, diff_funcs, energy_edges, n_groups)
    
    return solver


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Marshak Wave Steady State')
    parser.add_argument('--precond', action='store_true', 
                       help='Use LMFG preconditioner')
    parser.add_argument('--groups', type=int, default=10,
                       help='Number of energy groups (default: 10)')
    
    args = parser.parse_args()
    
    solver = run_marshak_wave_debug(
        use_preconditioner=args.precond,
        n_groups=args.groups
    )
