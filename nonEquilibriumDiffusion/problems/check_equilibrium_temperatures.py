#!/usr/bin/env python3
"""
Check if temperatures match correctly at equilibrium.

At true equilibrium:
- T_mat should equal T_rad
- Both should equal the energy-conserving equilibrium temperature
- E_r should equal a·T^4 exactly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0  # Large timestep for equilibration
rho, cv = 1.0, 0.05

n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)

sigma_values = [100.0] * n_groups

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    sig = sigma_values[g]
    sigma_funcs.append(lambda T, r, s=sig: s)
    diff_funcs.append(lambda T, r, s=sig: C_LIGHT / (3.0 * s))

from planck_integrals import Bg_multigroup
B_g = Bg_multigroup(energy_edges, 0.05)
chi = B_g / B_g.sum()

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
T_rad_init = 0.05

solver.T[:] = T_init
solver.T_old[:] = T_init
solver.E_r[:] = A_RAD * T_rad_init**4
solver.E_r_old[:] = solver.E_r.copy()
solver.phi_g_fraction[:, :] = chi[:, np.newaxis]
solver.phi_g_stored[:, :] = chi[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

# Compute expected equilibrium
E_total_init = T_init * cv * rho + solver.E_r[0]
from scipy.optimize import brentq
T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total_init, 0.001, 1.0)
E_r_eq_expected = A_RAD * T_eq**4

print("="*80)
print("EQUILIBRIUM TEMPERATURE CHECK")
print("="*80)
print()

print("Initial state:")
print(f"  T_mat = {T_init:.10f} keV")
print(f"  E_r = {solver.E_r[0]:.15e} GJ/cm³")
print(f"  E_total = {E_total_init:.15e} GJ/cm³")
print()

print("Expected equilibrium (from energy conservation):")
print(f"  T_eq = {T_eq:.15f} keV")
print(f"  E_r_eq = a·T_eq⁴ = {E_r_eq_expected:.15e} GJ/cm³")
print(f"  E_mat_eq = cv·ρ·T_eq = {cv * rho * T_eq:.15e} GJ/cm³")
print(f"  E_total = {cv * rho * T_eq + E_r_eq_expected:.15e} GJ/cm³")
print()

# Take one step
solver.step()

T_mat = solver.T[0]
E_r = solver.E_r[0]
T_rad = (E_r / A_RAD) ** 0.25
E_mat = cv * rho * T_mat
E_total = E_mat + E_r

print("After one large timestep (dt = 100 ns):")
print(f"  T_mat = {T_mat:.15f} keV")
print(f"  E_r = {E_r:.15e} GJ/cm³")
print(f"  T_rad = (E_r/a)^(1/4) = {T_rad:.15f} keV")
print()

print("="*80)
print("TEMPERATURE CONSISTENCY CHECKS")
print("="*80)
print()

# Check 1: Does T_mat equal T_eq?
T_mat_error = abs(T_mat - T_eq)
print(f"1. Material temperature vs expected equilibrium:")
print(f"   T_mat = {T_mat:.15f} keV")
print(f"   T_eq  = {T_eq:.15f} keV")
print(f"   Error = {T_mat_error:.3e} keV")
print(f"   Relative error = {T_mat_error / T_eq:.3e}")
if T_mat_error < 1e-10 * T_eq:
    print(f"   ✓ T_mat matches expected equilibrium")
else:
    print(f"   ✗ T_mat does NOT match expected")
print()

# Check 2: Does E_r equal a·T_eq⁴?
E_r_expected_from_Teq = A_RAD * T_eq**4
E_r_error = abs(E_r - E_r_expected_from_Teq)
print(f"2. Radiation energy vs a·T_eq⁴:")
print(f"   E_r = {E_r:.15e} GJ/cm³")
print(f"   a·T_eq⁴ = {E_r_expected_from_Teq:.15e} GJ/cm³")
print(f"   Error = {E_r_error:.3e} GJ/cm³")
print(f"   Relative error = {E_r_error / E_r_expected_from_Teq:.3e}")
if E_r_error < 1e-10 * E_r_expected_from_Teq:
    print(f"   ✓ E_r matches a·T_eq⁴")
else:
    print(f"   ✗ E_r does NOT match a·T_eq⁴")
print()

# Check 3: Does T_rad equal T_mat?
T_diff = abs(T_mat - T_rad)
print(f"3. Material vs Radiation temperature:")
print(f"   T_mat = {T_mat:.15f} keV")
print(f"   T_rad = {T_rad:.15f} keV")
print(f"   Difference = {T_diff:.3e} keV")
print(f"   Relative difference = {T_diff / T_mat:.3e}")
if T_diff < 1e-8 * T_mat:
    print(f"   ✓ T_mat and T_rad are essentially equal")
else:
    print(f"   ✗ T_mat and T_rad differ significantly!")
print()

# Check 4: Energy conservation
E_total_error = abs(E_total - E_total_init)
print(f"4. Energy conservation:")
print(f"   E_total_initial = {E_total_init:.15e} GJ/cm³")
print(f"   E_total_final = {E_total:.15e} GJ/cm³")
print(f"   Error = {E_total_error:.3e} GJ/cm³")
print(f"   Relative error = {E_total_error / E_total_init:.3e}")
if E_total_error < 1e-10 * E_total_init:
    print(f"   ✓ Energy is conserved")
else:
    print(f"   ✗ Energy is NOT conserved!")
print()

# Check 5: What is E_r from a·T_mat⁴?
E_r_from_Tmat = A_RAD * T_mat**4
E_r_discrepancy = abs(E_r - E_r_from_Tmat)
print(f"5. E_r vs a·T_mat⁴:")
print(f"   E_r (from solver) = {E_r:.15e} GJ/cm³")
print(f"   a·T_mat⁴ = {E_r_from_Tmat:.15e} GJ/cm³")
print(f"   Discrepancy = {E_r_discrepancy:.3e} GJ/cm³")
print(f"   Relative discrepancy = {E_r_discrepancy / E_r:.3e}")
if E_r_discrepancy < 1e-8 * E_r:
    print(f"   ✓ E_r ≈ a·T_mat⁴ (equilibrium)")
else:
    print(f"   ✗ E_r ≠ a·T_mat⁴ (NOT at equilibrium!)")
print()

# Check 6: Are we at the correct equilibrium?
print("="*80)
print("FINAL VERDICT")
print("="*80)
print()

all_good = True
if T_mat_error >= 1e-10 * T_eq:
    print("✗ Material temperature does not match expected equilibrium")
    all_good = False
    
if E_total_error >= 1e-10 * E_total_init:
    print("✗ Total energy is not conserved")
    all_good = False

if E_r_discrepancy >= 1e-6 * E_r:
    print(f"⚠ E_r and a·T_mat⁴ differ by {E_r_discrepancy / E_r:.2%}")
    print(f"  This means radiation is not quite at blackbody for T_mat")
    print(f"  Possible causes:")
    print(f"    - φ_g distribution doesn't match equilibrium Planck distribution")
    print(f"    - Numerical precision limits")
    print(f"    - Need more timesteps to fully equilibrate")
    if E_r_discrepancy / E_r > 0.01:
        all_good = False

if all_good:
    print("✓✓✓ SYSTEM IS AT CORRECT EQUILIBRIUM ✓✓✓")
    print(f"    T_mat = T_rad = {T_mat:.10f} keV")
    print(f"    Energy is conserved")
    print(f"    E_r = a·T⁴ (within numerical precision)")
else:
    print("⚠⚠⚠ SYSTEM MAY NOT BE AT PERFECT EQUILIBRIUM ⚠⚠⚠")
    if E_r_discrepancy / E_r > 1e-6:
        print(f"\nThe {E_r_discrepancy / E_r:.2%} discrepancy in E_r suggests either:")
        print(f"  1. Need more time steps for full equilibration")
        print(f"  2. φ_g distribution hasn't fully relaxed to Planck distribution")
        print(f"  3. Numerical precision limits")
