#!/usr/bin/env python3
"""
Test if the 0D problem converges to the correct equilibrium.

At equilibrium:
- T_mat = T_rad (material and radiation at same temperature)
- E_r = a·T⁴ (radiation energy density matches blackbody at temperature T)
- No net energy exchange (absorption = emission)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def compute_radiation_temperature(E_r_total):
    """Compute effective radiation temperature from total energy density"""
    return (E_r_total / A_RAD) ** 0.25

def test_equilibrium_convergence(n_groups=5):
    """Run 0D problem to equilibrium and verify final state"""
    
    print(f"{'='*80}")
    print(f"0D EQUILIBRIUM CONVERGENCE TEST")
    print(f"{'='*80}")
    print()
    
    # Domain
    r_min = 0.0
    r_max = 1.0e-3  # 1 mm single cell
    n_cells = 1
    dt = 0.01  # ns
    
    # Material
    rho = 1.0  # g/cm³
    cv = 0.05 / rho  # Specific heat
    
    # Energy group edges
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    
    # Constant opacities
    sigma_values = []
    for g in range(n_groups):
        E_mid = np.sqrt(energy_edges[g] * energy_edges[g+1])
        sigma_g = 100.0 * (E_mid**(-2.0))
        sigma_values.append(sigma_g)
    
    # Create opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        sigma_g_fixed = sigma_values[g]
        
        def make_opacity_func(sigma_fixed):
            def opacity_func(T, r):
                return sigma_fixed
            return opacity_func
        
        def make_diffusion_func(sigma_fixed):
            def diffusion_func(T, r):
                return C_LIGHT / (3.0 * sigma_fixed)
            return diffusion_func
        
        sigma_funcs.append(make_opacity_func(sigma_g_fixed))
        diff_funcs.append(make_diffusion_func(sigma_g_fixed))
    
    # Emission fractions - use Planck-weighted
    from planck_integrals import Bg_multigroup
    T_ref = 0.05  # keV
    B_g_ref = Bg_multigroup(energy_edges, T_ref)
    chi = B_g_ref / B_g_ref.sum()
    
    # Reflecting BCs
    def left_bc(phi, r):
        return 0.0, 1.0, 0.0
    
    def right_bc(phi, r):
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [left_bc] * n_groups
    right_bc_funcs = [right_bc] * n_groups
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
        energy_edges=energy_edges, geometry='planar', dt=dt,
        diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs, right_bc_funcs=right_bc_funcs,
        emission_fractions=chi, rho=rho, cv=cv
    )
    
    # Disable debug output
    solver._debug_update_T = False
    
    # Initial conditions - OUT OF EQUILIBRIUM
    T_init = 0.025  # keV
    T_radiation_equivalent = 0.05  # Radiation from hotter source
    
    solver.T[:] = T_init
    solver.T_old[:] = T_init
    solver.E_r[:] = A_RAD * T_radiation_equivalent**4
    solver.E_r_old[:] = solver.E_r.copy()
    solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
    solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_radiation_equivalent**4
    
    E_total_initial = solver.T[0] * cv * rho + solver.E_r[0]
    T_rad_init = compute_radiation_temperature(solver.E_r[0])
    
    print(f"INITIAL STATE:")
    print(f"  Material temperature:  T_mat = {T_init:.6f} keV")
    print(f"  Radiation temperature: T_rad = {T_rad_init:.6f} keV")
    print(f"  Material energy density: e_mat = {solver.T[0] * cv * rho:.6e} GJ/cm³")
    print(f"  Radiation energy density: E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"  Total energy: {E_total_initial:.6e} GJ/cm³")
    print()
    
    # Compute expected equilibrium
    print(f"EXPECTED EQUILIBRIUM:")
    print(f"  In an isolated system (Neumann BCs), total energy is conserved.")
    print(f"  At equilibrium: T_mat = T_rad = T_eq")
    print(f"  And: E_total = c_v·ρ·T_eq + a·T_eq⁴")
    print()
    
    # Solve for equilibrium temperature
    # E_total = c_v·ρ·T + a·T⁴
    # This is a polynomial equation in T
    def energy_balance(T):
        return cv * rho * T + A_RAD * T**4 - E_total_initial
    
    from scipy.optimize import brentq
    T_eq_expected = brentq(energy_balance, 0.001, 1.0)
    E_r_eq_expected = A_RAD * T_eq_expected**4
    e_mat_eq_expected = cv * rho * T_eq_expected
    
    print(f"  Expected T_eq = {T_eq_expected:.6f} keV")
    print(f"  Expected E_r_eq = {E_r_eq_expected:.6e} GJ/cm³")
    print(f"  Expected e_mat_eq = {e_mat_eq_expected:.6e} GJ/cm³")
    print(f"  Total: {e_mat_eq_expected + E_r_eq_expected:.6e} GJ/cm³")
    print()
    
    # Run to equilibrium
    print(f"{'='*80}")
    print(f"RUNNING TO EQUILIBRIUM")
    print(f"{'='*80}")
    print()
    
    max_steps = 100  # Just a few steps to see what's happening
    check_interval = 1  # Check every step
    tolerance = 1e-5
    
    print(f"Step     T_mat [keV]  T_rad [keV]  |T_mat-T_rad|  E_total [GJ/cm³]  ΔE_total")
    print(f"-" * 100)
    
    for step in range(max_steps):
        T_old_step = solver.T[0]
        E_r_old_step = solver.E_r[0]
        
        solver.step(verbose=False)
        
        if step < 20 or step % check_interval == 0:
            T_mat = solver.T[0]
            T_rad = compute_radiation_temperature(solver.E_r[0])
            E_total = T_mat * cv * rho + solver.E_r[0]
            delta_E = E_total - E_total_initial
            T_diff = abs(T_mat - T_rad)
            
            print(f"{step:5d}    {T_mat:.8f}   {T_rad:.8f}   {T_diff:.3e}    {E_total:.10e}  {delta_E:.3e}")
            
            # Check for convergence
            if T_diff < tolerance * T_mat:
                print()
                print(f"Converged at step {step}!")
                break
        
        # Safety check for divergence
        if solver.T[0] > 10.0 or solver.T[0] < 0:
            print(f"\nDIVERGED at step {step}!")
            break
    
    # Final state
    print()
    print(f"{'='*80}")
    print(f"FINAL STATE (after {step} steps)")
    print(f"{'='*80}")
    
    T_mat_final = solver.T[0]
    T_rad_final = compute_radiation_temperature(solver.E_r[0])
    E_total_final = T_mat_final * cv * rho + solver.E_r[0]
    
    print(f"  Material temperature:  T_mat = {T_mat_final:.8f} keV")
    print(f"  Radiation temperature: T_rad = {T_rad_final:.8f} keV")
    print(f"  Temperature difference: |T_mat - T_rad| = {abs(T_mat_final - T_rad_final):.3e} keV")
    print()
    print(f"  Material energy density: e_mat = {T_mat_final * cv * rho:.10e} GJ/cm³")
    print(f"  Radiation energy density: E_r = {solver.E_r[0]:.10e} GJ/cm³")
    print(f"  Total energy: {E_total_final:.10e} GJ/cm³")
    print()
    
    # Compare to expected
    print(f"COMPARISON TO EXPECTED EQUILIBRIUM:")
    print(f"-" * 80)
    print(f"  Expected T_eq = {T_eq_expected:.8f} keV")
    print(f"  Actual T_mat  = {T_mat_final:.8f} keV")
    print(f"  Error in T    = {abs(T_mat_final - T_eq_expected):.3e} keV")
    print(f"  Relative err  = {abs(T_mat_final - T_eq_expected) / T_eq_expected:.3e}")
    print()
    
    print(f"  Expected E_total = {E_total_initial:.10e} GJ/cm³")
    print(f"  Actual E_total   = {E_total_final:.10e} GJ/cm³")
    print(f"  Energy error     = {abs(E_total_final - E_total_initial):.3e} GJ/cm³")
    print(f"  Relative err     = {abs(E_total_final - E_total_initial) / E_total_initial:.3e}")
    print()
    
    # Test pass/fail
    T_rel_error = abs(T_mat_final - T_eq_expected) / T_eq_expected
    T_mat_rad_diff = abs(T_mat_final - T_rad_final) / T_mat_final
    E_conservation_error = abs(E_total_final - E_total_initial) / E_total_initial
    
    print(f"{'='*80}")
    print(f"TEST RESULTS:")
    print(f"{'='*80}")
    
    tolerance_T = 0.01  # 1% for temperature
    tolerance_E = 1e-6  # Much stricter for energy conservation
    tolerance_eq = 1e-6  # T_mat should equal T_rad very precisely
    
    passed = True
    
    if T_rel_error < tolerance_T:
        print(f"✓ Converged to correct equilibrium temperature (error: {T_rel_error:.2e} < {tolerance_T})")
    else:
        print(f"✗ Did NOT converge to correct equilibrium temperature (error: {T_rel_error:.2e} >= {tolerance_T})")
        passed = False
    
    if T_mat_rad_diff < tolerance_eq:
        print(f"✓ Material and radiation temperatures match (diff: {T_mat_rad_diff:.2e} < {tolerance_eq})")
    else:
        print(f"✗ Material and radiation temperatures do NOT match (diff: {T_mat_rad_diff:.2e} >= {tolerance_eq})")
        passed = False
    
    if E_conservation_error < tolerance_E:
        print(f"✓ Energy is conserved (error: {E_conservation_error:.2e} < {tolerance_E})")
    else:
        print(f"✗ Energy is NOT conserved (error: {E_conservation_error:.2e} >= {tolerance_E})")
        passed = False
    
    print()
    if passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("The solver correctly converges to proper equilibrium!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("The solver is NOT converging to proper equilibrium!")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    test_equilibrium_convergence()
