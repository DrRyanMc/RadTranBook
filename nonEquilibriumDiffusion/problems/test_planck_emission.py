#!/usr/bin/env python3
"""Test B_g and emission calculation."""

from planck_integrals import Bg
import numpy as np

# Test case
T = 0.025  # ke V
E_min, E_max = 1e-4, 8.706e-4
sigma_a_0 = 1.149e9  # cm^-1
chi_0 = 0.00893

# Constants
A_RAD = 0.01372
C_LIGHT = 29.9792458

print("="*60)
print("TESTING PLANCK INTEGRAL AND EMISSION")
print("="*60)
print(f"Group 0: [{E_min}, {E_max}] keV")
print(f"Temperature: {T} keV")
print(f"Opacity: σ = {sigma_a_0:.3e} cm^-1")
print(f"Rosseland fraction: χ = {chi_0}")
print()

# Get B_g from integral
B_g = Bg(E_min, E_max, T)
print(f"B_g from Planck integral: {B_g:.6e} GJ/(cm²·ns·sr)")
print()

# Compute emission using 4π·σ·B_g
emission_from_Bg = 4 * np.pi * sigma_a_0 * B_g
print(f"Emission = 4π·σ·B_g = {emission_from_Bg:.6e} GJ/(cm³·ns)")
print()

# Expected emission from c·σ·χ·a·T⁴
emission_expected = C_LIGHT * sigma_a_0 * chi_0 * A_RAD * T**4
print(f"Expected = c·σ·χ·a·T⁴ = {emission_expected:.6e} GJ/(cm³·ns)")
print()

print(f"Ratio (actual/expected): {emission_from_Bg / emission_expected:.6f}")
print(f"Discrepancy factor: {emission_expected / emission_from_Bg:.1f}x")
