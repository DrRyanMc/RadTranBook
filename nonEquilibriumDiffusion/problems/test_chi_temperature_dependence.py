#!/usr/bin/env python3
"""
Test if using correct χ_g for equilibrium temperature gives perfect equilibration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Setup
r_min, r_max = 0.0, 1.0e-3
n_cells = 1
dt = 100.0
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

# Initial conditions
T_init = 0.025
T_rad_init = 0.05

# Compute expected equilibrium temperature
E_r_init = A_RAD * T_rad_init**4
E_total_init = T_init * cv * rho + E_r_init
from scipy.optimize import brentq
T_eq = brentq(lambda T: cv * rho * T + A_RAD * T**4 - E_total_init, 0.001, 1.0)
E_r_eq = A_RAD * T_eq**4

print("="*80)
print("TEST: Effect of χ_g computed at correct equilibrium temperature")
print("="*80)
print(f"Expected equilibrium: T_eq = {T_eq:.10f} keV")
print(f"                      E_r_eq = {E_r_eq:.15e} GJ/cm³")
print()

# Test 1: χ_g at WRONG temperature (T_ref = 0.05 keV)
print("-"*80)
print("Test 1: χ_g computed at T_ref = 0.05 keV (WRONG)")
print("-"*80)

from planck_integrals import Bg_multigroup
B_g_wrong = Bg_multigroup(energy_edges, 0.05)
chi_wrong = B_g_wrong / B_g_wrong.sum()
print(f"χ_g at 0.05 keV: {chi_wrong}")

def bc(phi, r):
    return 0.0, 1.0, 0.0

left_bcs = [bc] * n_groups
right_bcs = [bc] * n_groups

solver1 = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi_wrong, rho=rho, cv=cv
)
solver1._debug_update_T = False

solver1.T[:] = T_init
solver1.T_old[:] = T_init
solver1.E_r[:] = E_r_init
solver1.E_r_old[:] = solver1.E_r.copy()
solver1.phi_g_fraction[:, :] = chi_wrong[:, np.newaxis]
solver1.phi_g_stored[:, :] = chi_wrong[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

solver1.step()

T_mat1 = solver1.T[0]
E_r1 = solver1.E_r[0]
T_rad1 = (E_r1 / A_RAD) ** 0.25
diff1 = abs(T_mat1 - T_rad1)

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat1:.15f} keV")
print(f"  T_rad = {T_rad1:.15f} keV")
print(f"  Difference = {diff1:.3e} keV ({diff1/T_mat1*100:.4f}%)")
print(f"  E_r / (a·T_mat⁴) = {E_r1 / (A_RAD * T_mat1**4):.10f}")

# Test 2: χ_g at CORRECT temperature (T_eq)
print()
print("-"*80)
print(f"Test 2: χ_g computed at T_eq = {T_eq:.5f} keV (CORRECT)")
print("-"*80)

B_g_correct = Bg_multigroup(energy_edges, T_eq)
chi_correct = B_g_correct / B_g_correct.sum()
print(f"χ_g at {T_eq:.5f} keV: {chi_correct}")

solver2 = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=r_min, r_max=r_max, n_cells=n_cells,
    energy_edges=energy_edges, geometry='planar', dt=dt,
    diffusion_coeff_funcs=diff_funcs, absorption_coeff_funcs=sigma_funcs,
    left_bc_funcs=left_bcs, right_bc_funcs=right_bcs,
    emission_fractions=chi_correct, rho=rho, cv=cv
)
solver2._debug_update_T = False

solver2.T[:] = T_init
solver2.T_old[:] = T_init
solver2.E_r[:] = E_r_init
solver2.E_r_old[:] = solver2.E_r.copy()
solver2.phi_g_fraction[:, :] = chi_correct[:, np.newaxis]
solver2.phi_g_stored[:, :] = chi_correct[:, np.newaxis] * A_RAD * C_LIGHT * T_rad_init**4

solver2.step()

T_mat2 = solver2.T[0]
E_r2 = solver2.E_r[0]
T_rad2 = (E_r2 / A_RAD) ** 0.25
diff2 = abs(T_mat2 - T_rad2)

print(f"\nAfter one large step:")
print(f"  T_mat = {T_mat2:.15f} keV")
print(f"  T_rad = {T_rad2:.15f} keV")
print(f"  Difference = {diff2:.3e} keV ({diff2/T_mat2*100:.4f}%)")
print(f"  E_r / (a·T_mat⁴) = {E_r2 / (A_RAD * T_mat2**4):.10f}")

# Compare
print()
print("="*80)
print("COMPARISON")
print("="*80)
print(f"χ_g at wrong T:   T_diff = {diff1:.3e} keV ({diff1/T_mat1*100:.4f}%)")
print(f"χ_g at correct T: T_diff = {diff2:.3e} keV ({diff2/T_mat2*100:.4f}%)")
print()
if diff2 < diff1 * 0.1:
    print("✓✓✓ Using correct χ_g DRAMATICALLY improves equilibration! ✓✓✓")
    print(f"    Improvement factor: {diff1/diff2:.1f}×")
else:
    print("⚠ Using correct χ_g did not significantly improve equilibration")
    print("   The issue may be something else...")
