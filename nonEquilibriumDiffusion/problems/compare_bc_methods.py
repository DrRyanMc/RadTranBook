#!/usr/bin/env python3
"""
Compare two methods for computing incoming BC values:
1. Marshak wave method: C = chi_g * (a*c*T^4)/2
2. Our method: C = 0.5 * (4π/c) * B_g

These should be equivalent!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD
from planck_integrals import Bg_multigroup

T = 0.025  # keV
n_groups = 5
energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)

print("="*80)
print("COMPARING BC VALUE COMPUTATION METHODS")
print("="*80)
print(f"Temperature: T = {T:.10f} keV")
print(f"Groups: {n_groups}")
print()

# Method 1: Marshak wave approach
print("-"*80)
print("Method 1: Marshak wave approach")
print("-"*80)
F_total = (A_RAD * C_LIGHT * T**4) / 2.0
print(f"F_total = (a*c*T^4)/2 = {F_total:.15e} GJ/(cm²·ns)")

# Get emission fractions
B_g = Bg_multigroup(energy_edges, T)
chi = B_g / B_g.sum()
print(f"\nEmission fractions χ_g:")
for g in range(n_groups):
    print(f"  Group {g}: χ_g = {chi[g]:.10e}")

BC_C_method1 = chi * F_total
print(f"\nBC C values (Method 1):")
for g in range(n_groups):
    print(f"  Group {g}: C = {BC_C_method1[g]:.15e} GJ/(cm²·ns)")
print(f"Sum: {BC_C_method1.sum():.15e}")

# Method 2: Our approach using B_g directly
print()
print("-"*80)
print("Method 2: Our approach using B_g")
print("-"*80)
phi_inc_g = (4.0 * np.pi / C_LIGHT) * B_g
print(f"φ_inc for each group:")
for g in range(n_groups):
    print(f"  Group {g}: φ_inc = {phi_inc_g[g]:.15e} GJ/(cm²·ns)")
print(f"Sum: {phi_inc_g.sum():.15e}")

BC_C_method2 = 0.5 * phi_inc_g
print(f"\nBC C values (Method 2):")
for g in range(n_groups):
    print(f"  Group {g}: C = {BC_C_method2[g]:.15e} GJ/(cm²·ns)")
print(f"Sum: {BC_C_method2.sum():.15e}")

# Compare
print()
print("="*80)
print("COMPARISON")
print("="*80)
print()
print("Group |  Method 1 (C)      |  Method 2 (C)      |  Ratio    |  Match?")
print("------+--------------------+--------------------+-----------+---------")
for g in range(n_groups):
    ratio = BC_C_method2[g] / BC_C_method1[g] if BC_C_method1[g] > 0 else 0
    match = "✓" if abs(ratio - 1.0) < 1e-10 else "✗"
    print(f"  {g}   | {BC_C_method1[g]:.12e} | {BC_C_method2[g]:.12e} | {ratio:.10f} | {match}")

print()
total_ratio = BC_C_method2.sum() / BC_C_method1.sum()
print(f"Total C (Method 1): {BC_C_method1.sum():.15e} GJ/(cm²·ns)")
print(f"Total C (Method 2): {BC_C_method2.sum():.15e} GJ/(cm²·ns)")
print(f"Ratio: {total_ratio:.15f}")
print()

if abs(total_ratio - 1.0) < 1e-10:
    print("✓✓✓ METHODS AGREE! Both give same total incoming flux ✓✓✓")
    print()
    print("The issue must be elsewhere:")
    print("  - How BC is applied during Newton iteration")
    print("  - Temperature dependence of D in Marshak BC")
    print("  - Interaction with time-dependent solver")
else:
    print("✗✗✗ METHODS DISAGREE! ✗✗✗")
    print()
    print("Possible causes:")
    print("  - B_g units or normalization issue")
    print("  - Different definition of flux vs scalar flux")
    print("  - Error in one of the formulations")

# Check what (4π/c)*B should equal
print()
print("="*80)
print("VERIFICATION: Total φ vs a*c*T⁴")
print("="*80)
total_phi= phi_inc_g.sum()
expected_phi = A_RAD * C_LIGHT * T**4
print(f"Σ φ_inc (from B_g):  {total_phi:.15e} GJ/(cm²·ns)")
print(f"a*c*T^4:            {expected_phi:.15e} GJ/(cm²·ns)")
print(f"Ratio:               {total_phi/expected_phi:.15f}")
print()
if abs(total_phi/expected_phi - 1.0) < 1e-6:
    print("✓ φ_inc correctly represents isotropic blackbody at T")
else:
    print("✗ φ_inc does NOT match expected a*c*T^4!")
    print("  This suggests B_g has wrong units or normalization")
