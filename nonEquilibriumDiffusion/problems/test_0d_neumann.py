#!/usr/bin/env python3
"""
Test 0D-like problem with Neumann BCs (zero flux/reflecting boundaries).
Single cell, constant opacities, no diffusion, no sources.
Only material-radiation coupling should occur.

With T and E_r out of equilibrium initially, we should see:
  dT/dt = (Σ_g c·σ_a,g·E_r,g - Σ_g c·σ_a,g·χ_g·a·T^4) / (c_v·ρ)

This should match analytical calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def test_0d_neumann(n_groups=5):
    """Test 0D problem with reflecting boundaries"""
    
    print(f"{'='*80}")
    print(f"0D TEST WITH NEUMANN (REFLECTING) BOUNDARIES")
    print(f"{'='*80}")
    print(f"Configuration: {n_groups} groups, 1 cell, reflecting BCs")
    print(f"Purpose: Verify material-radiation coupling without boundary effects\n")
    
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
    
    # Constant opacities - choose different values per group
    sigma_values = []
    for g in range(n_groups):
        E_mid = np.sqrt(energy_edges[g] * energy_edges[g+1])
        # Use a simple power law for variety
        sigma_g = 100.0 * (E_mid**(-2.0))
        sigma_values.append(sigma_g)
    
    print(f"Opacities (cm⁻¹):")
    for g, sig in enumerate(sigma_values):
        print(f"  Group {g}: σ = {sig:.3e}")
    
    # Create constant opacity functions
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
    
    # Emission fractions - use Planck-weighted (NOT Rosseland!)
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    T_ref = 0.05  # keV
    # IMPORTANT: For emission, use Planck-weighted fractions from B_g, not Rosseland from dB/dT!
    from planck_integrals import Bg_multigroup
    B_g_ref = Bg_multigroup(energy_edges, T_ref)
    chi = B_g_ref / B_g_ref.sum()  # Planck-weighted
    
    print(f"\nEmission fractions χ (Planck-weighted):")
    print(f"  χ = {chi}")
    print(f"  Sum: {np.sum(chi):.6f} (should be 1.0)")
    
    # Reflecting BCs - zero flux on both sides
    def left_bc(phi, r):
        """Reflecting: ∇φ = 0, so A=0, B=1, C=0"""
        return 0.0, 1.0, 0.0
    
    def right_bc(phi, r):
        """Reflecting: ∇φ = 0"""
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
    
    # Enable debug output for temperature update
    solver._debug_update_T = True
    
    # Initial conditions - OUT OF EQUILIBRIUM
    T_init = 0.025  # keV
    T_radiation_equivalent = 0.05  # Set radiation as if from hotter source
    
    solver.T[:] = T_init
    solver.T_old[:] = T_init  # CRITICAL: Must initialize T_old to match initial conditions!
    solver.E_r[:] = A_RAD * T_radiation_equivalent**4  # Radiation "hotter" than material
    solver.E_r_old[:] = solver.E_r.copy()  # Initialize old radiation energy too
    solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
    solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_radiation_equivalent**4
    
    print(f"\n{'='*80}")
    print(f"INITIAL CONDITIONS (OUT OF EQUILIBRIUM)")
    print(f"{'='*80}")
    print(f"Material temperature: T = {T_init} keV")
    print(f"Radiation energy: E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"  (Equivalent to blackbody at T_rad = {T_radiation_equivalent} keV)")
    print(f"Equilibrium E_r at T_init would be: {A_RAD * T_init**4:.6e} GJ/cm³")
    print(f"  Ratio E_r / E_r,eq = {solver.E_r[0] / (A_RAD * T_init**4):.3f}")
    
    # Compute expected heating rate ANALYTICALLY
    print(f"\n{'='*80}")
    print(f"ANALYTICAL PREDICTION")
    print(f"{'='*80}")
    
    E_r_g = solver.E_r[0] * chi  # Energy by group
    
    total_absorption = 0.0
    total_emission = 0.0
    
    print(f"By-group analysis:")
    for g in range(n_groups):
        absorption_g = C_LIGHT * sigma_values[g] * E_r_g[g]
        emission_g = C_LIGHT * sigma_values[g] * chi[g] * A_RAD * T_init**4
        total_absorption += absorption_g
        total_emission += emission_g
        print(f"  Group {g}:")
        print(f"    E_r,g = {E_r_g[g]:.6e} GJ/cm³")
        print(f"    Absorption = c·σ·E_r = {absorption_g:.6e} GJ/(cm³·ns)")
        print(f"    Emission = c·σ·χ·a·T⁴ = {emission_g:.6e} GJ/(cm³·ns)")
        print(f"    Net = {absorption_g - emission_g:.6e} GJ/(cm³·ns)")
    
    net_heating = total_absorption - total_emission
    dT_dt_expected = net_heating / (cv * rho)
    delta_T_expected = dT_dt_expected * dt
    
    print(f"\nSummary:")
    print(f"  Total absorption = {total_absorption:.6e} GJ/(cm³·ns)")
    print(f"  Total emission = {total_emission:.6e} GJ/(cm³·ns)")
    print(f"  Net heating = {net_heating:.6e} GJ/(cm³·ns)")
    print(f"  c_v = {cv:.6e} GJ/(g·keV)")
    print(f"  ρ = {rho} g/cm³")
    print(f"  Expected dT/dt = {dT_dt_expected:.6e} keV/ns")
    print(f"  Expected ΔT after dt={dt} ns: {delta_T_expected:.6e} keV")
    print(f"  Expected T_new = {T_init + delta_T_expected:.6e} keV")
    
    # Take one time step
    print(f"\n{'='*80}")
    print(f"TAKING ONE TIME STEP")
    print(f"{'='*80}")
    
    solver.step(verbose=False)
    
    T_final = solver.T[0]
    E_r_final = solver.E_r[0]
    delta_T_actual = T_final - T_init
    delta_E_r = E_r_final - solver.E_r_old[0]
    
    print(f"\nResults:")
    print(f"  T_initial = {T_init:.6e} keV")
    print(f"  T_final = {T_final:.6e} keV")
    print(f"  ΔT_actual = {delta_T_actual:.6e} keV")
    print(f"  ΔT_expected = {delta_T_expected:.6e} keV")
    print(f"  Error = {abs(delta_T_actual - delta_T_expected):.6e} keV")
    print(f"  Relative error = {abs(delta_T_actual - delta_T_expected) / abs(delta_T_expected):.6e}")
    
    print(f"\n  E_r_initial = {solver.E_r_old[0]:.6e} GJ/cm³")
    print(f"  E_r_final = {E_r_final:.6e} GJ/cm³")
    print(f"  ΔE_r = {delta_E_r:.6e} GJ/cm³")
    
  # Check if test passes
    rel_error = abs(delta_T_actual - delta_T_expected) / abs(delta_T_expected)
    tolerance = 0.01  # 1% tolerance
    
    print(f"\n{'='*80}")
    if rel_error < tolerance:
        print(f"✓ TEST PASSED")
        print(f"  Material-radiation coupling works correctly!")
        print(f"  Relative error {rel_error:.6e} < {tolerance}")
    else:
        print(f"✗ TEST FAILED")
        print(f"  Material-radiation coupling has errors!")
        print(f"  Relative error {rel_error:.6e} >= {tolerance}")
    print(f"{'='*80}")
    
    return rel_error < tolerance

if __name__ == "__main__":
    success = test_0d_neumann(n_groups=5)
    sys.exit(0 if success else 1)
