#!/usr/bin/env python3
"""
Test 1: Use a LARGE time step to reach equilibrium quickly
Test 2: Check what κ actually is at pseudo-equilibrium
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def run_test(dt, label):
    print(f"\n{'='*80}")
    print(f"TEST: {label}")
    print(f"Δt = {dt} ns")
    print(f"{'='*80}\n")
    
    # Simple setup
    r_min, r_max = 0.0, 1.0e-3
    n_cells = 1
    rho, cv = 1.0, 0.05
    
    n_groups = 5
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    
    # Constant opacities
    sigma_values = [100.0] * n_groups
    
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        sig = sigma_values[g]
        sigma_funcs.append(lambda T, r, s=sig: s)
        diff_funcs.append(lambda T, r, s=sig: C_LIGHT / (3.0 * s))
    
    # Chi
    from planck_integrals import Bg_multigroup
    B_g = Bg_multigroup(energy_edges, 0.05)
    chi = B_g / B_g.sum()
    
    # Reflecting BCs
    def bc(phi, r):
        return 0.0, 1.0, 0.0
    
    left_bcs = [bc] * n_groups
    right_bcs = [bc] * n_groups
    
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
        emission_fractions=chi, rho=rho, cv=cv
    )
    
    solver._debug_update_T = False
    
    T_init = 0.025
    T_rad = 0.05
    
    solver.T[:] = T_init
    solver.T_old[:] = T_init
    solver.E_r[:] = A_RAD * T_rad**4
    solver.E_r_old[:] = solver.E_r.copy()
    solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
    solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad**4
    
    # Expected equilibrium
    E_total_init = T_init * cv * rho + solver.E_r[0]
    from scipy.optimize import brentq
    T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total_init, 0.001, 1.0)
    E_r_eq = A_RAD * T_eq**4
    
    print(f"Initial: T = {T_init:.6f} keV, E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"Expected equilibrium: T_eq = {T_eq:.6f} keV, E_r_eq = {E_r_eq:.6e} GJ/cm³")
    print()
    
    # Take one step and examine κ
    print("Taking one time step...")
    
    # We need to capture κ from the Newton iteration
    # Monkey-patch to capture κ values
    captured_kappa = []
    old_compute = solver.compute_radiation_energy_from_kappa
    
    def compute_with_capture(kappa, T_star, xi_g_list):
        captured_kappa.append(kappa.copy())
        return old_compute(kappa, T_star, xi_g_list)
    
    solver.compute_radiation_energy_from_kappa = compute_with_capture
    solver.step()
    solver.compute_radiation_energy_from_kappa = old_compute
    
    T_final = solver.T[0]
    E_r_final = solver.E_r[0]
    T_rad_final = (E_r_final / A_RAD) ** 0.25
    
    print(f"\nAfter 1 step:")
    print(f"  T_mat = {T_final:.8f} keV")
    print(f"  T_rad = {T_rad_final:.8f} keV")
    print(f"  |T_mat - T_rad| = {abs(T_final - T_rad_final):.6e} keV")
    print(f"  E_r = {E_r_final:.10e} GJ/cm³")
    print()
    
    print(f"Comparison to equilibrium:")
    print(f"  T_eq = {T_eq:.8f} keV")
    print(f"  T_mat error = {abs(T_final - T_eq):.6e} keV")
    print(f"  E_r_eq = {E_r_eq:.10e} GJ/cm³")
    print(f"  E_r error = {abs(E_r_final - E_r_eq):.6e} GJ/cm³")
    print(f"  E_r relative error = {abs(E_r_final - E_r_eq) / E_r_eq:.6e}")
    print()
    
    # Examine κ
    if captured_kappa:
        kappa_initial = captured_kappa[0][0]
        kappa_final = captured_kappa[-1][0]
        
        print(f"κ values during Newton iteration:")
        print(f"  Initial guess: κ = {kappa_initial:.6e} GJ/(cm³·ns)")
        print(f"  Final converged: κ = {kappa_final:.6e} GJ/(cm³·ns)")
        print()
        
        # Compute expected absorption and emission at final state
        absorption_total = 0.0
        emission_total = 0.0
        for g in range(n_groups):
            E_r_g = E_r_final * chi[g]
            absorption_g = C_LIGHT * sigma_values[g] * E_r_g
            emission_g = C_LIGHT * sigma_values[g] * chi[g] * A_RAD * T_final**4
            absorption_total += absorption_g
            emission_total += emission_g
        
        net_expected = absorption_total - emission_total
        
        print(f"At final state:")
        print(f"  Total absorption = {absorption_total:.6e} GJ/(cm³·ns)")
        print(f"  Total emission = {emission_total:.6e} GJ/(cm³·ns)")
        print(f"  Net (abs - em) = {net_expected:.6e} GJ/(cm³·ns)")
        print(f"  κ from solver = {kappa_final:.6e} GJ/(cm³·ns)")
        print(f"  Ratio κ/net = {kappa_final / net_expected if abs(net_expected) > 1e-20 else 'inf'}")
        print()
        
        # Check: is κ close to zero or close to emission?
        print(f"  |κ| / emission = {abs(kappa_final) / emission_total:.6e}")
        print(f"  At true equilibrium, κ should be ~ 0, not equal to emission!")
        print(f"  (Absorption and emission are individually large but balanced)")
    
    return T_final, T_eq, E_r_final, E_r_eq

# Test with small and large time steps
print("="*80)
print("TESTING TIME STEP EFFECTS ON EQUILIBRATION")
print("="*80)

T1, Teq1, Er1, Ereq1 = run_test(dt=0.01, label="Small Δt = 0.01 ns")
T2, Teq2, Er2, Ereq2 = run_test(dt=1.0, label="Medium Δt = 1.0 ns") 
T3, Teq3, Er3, Ereq3 = run_test(dt=100.0, label="Large Δt = 100 ns")

print("\n" + "="*80)
print("SUMMARY: Effect of Δt on equilibration")
print("="*80)
print(f"\nΔt = 0.01 ns:  E_r error = {abs(Er1 - Ereq1) / Ereq1:.2%}")
print(f"Δt = 1.0 ns:   E_r error = {abs(Er2 - Ereq2) / Ereq2:.2%}")
print(f"Δt = 100 ns:   E_r error = {abs(Er3 - Ereq3) / Ereq3:.2%}")
print()
print("Larger Δt should give better equilibration in one step! ✓")
