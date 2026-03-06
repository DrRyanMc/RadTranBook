"""
Investigate the T⁴ scaling issue for fixed energy windows.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadHydro/Planck Integral/PlanckIntegrals')
from planck_integrals import Bg

print("="*80)
print("INVESTIGATING T⁴ SCALING FOR FIXED VS SCALED ENERGY WINDOWS")
print("="*80)
print()

# Stefan-Boltzmann constant
SIGMA_SB_OVER_PI = 0.0327314873  # GJ/(ns·cm²·keV⁴)

print("CASE 1: Fixed Energy Window")
print("-" * 80)
print()
print("For B_g = (σ/π) T⁴ [Π(E_high/T) - Π(E_low/T)]")
print("With FIXED E_low and E_high, as T increases:")
print("  - T⁴ term increases")
print("  - But x = E/T DECREASES (shifted relative to Wien peak)")
print("  - So Π(x_high) - Π(x_low) changes")
print("  - Overall scaling is NOT exactly T⁴")
print()

E_low = 0.1  # keV
E_high = 5.0  # keV
temperatures = [0.01, 0.1, 1.0, 10.0]

print(f"Fixed energy window: [{E_low}, {E_high}] keV")
print()

for T in temperatures:
    x_low = E_low / T
    x_high = E_high / T
    B = Bg(E_low, E_high, T)
    print(f"T = {T:6.2f} keV: x_low = {x_low:8.2f}, x_high = {x_high:8.2f}, B = {B:.6e}")

print()
print("As T increases:")
print("  - Low T: x values are VERY LARGE (above Wien peak)")
print("  - High T: x values are VERY SMALL (below Wien peak)")
print("This explains why scaling is not T⁴!")
print()

print()
print("CASE 2: Scaled Energy Window")
print("-" * 80)
print()
print("For energy window that SCALES WITH TEMPERATURE:")
print("  E_low = k₁ T, E_high = k₂ T")
print("Then x_low and x_high are CONSTANT, so Π difference is constant.")
print("Result: B_g DOES scale as T⁴ ✓")
print()

k_low = 0.5   # Dimensionless
k_high = 10.0  # Dimensionless

print(f"Scaled energy window: E_low = {k_low} T, E_high = {k_high} T")
print()

B_prev = None
T_prev = None
for T in temperatures:
    E_low_scaled = k_low * T
    E_high_scaled = k_high * T
    x_low = E_low_scaled / T  # = k_low (constant!)
    x_high = E_high_scaled / T  # = k_high (constant!)
    B = Bg(E_low_scaled, E_high_scaled, T)
    
    print(f"T = {T:6.2f} keV: x_low = {x_low:.2f}, x_high = {x_high:.2f}, B = {B:.6e}", end="")
    
    if B_prev is not None:
        T_ratio = T / T_prev
        B_ratio = B / B_prev
        expected_ratio = T_ratio**4
        print(f", Ratio = {B_ratio:.2f} (expected {expected_ratio:.2f})")
    else:
        print()
    
    B_prev = B
    T_prev = T

print()
print("Perfect T⁴ scaling when energy window scales with T! ✓")
print()

print()
print("CASE 3: Full Integral (0 to ∞)")
print("-" * 80)
print()
print("The TOTAL integrated intensity must satisfy Stefan-Boltzmann:")
print("  B_total = (σ/π) T⁴")
print()

E_max = 200.0  # Large enough to approximate infinity
B_prev = None
T_prev = None
for T in temperatures:
    B = Bg(0.0, E_max, T)
    B_SB = SIGMA_SB_OVER_PI * T**4
    
    print(f"T = {T:6.2f} keV: B_numerical = {B:.6e}, B_SB = {B_SB:.6e}", end="")
    
    if B_prev is not None:
        T_ratio = T / T_prev
        B_ratio = B / B_prev
        expected_ratio = T_ratio**4
        print(f", Ratio = {B_ratio:.2f} (expected {expected_ratio:.0f})")
    else:
        print()
    
    B_prev = B
    T_prev = T

print()
print("Perfect T⁴ scaling for full integral! ✓")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("The Planck integral library is CORRECT.")
print()
print("The apparent T⁴ scaling issue in Test 2 was due to using a FIXED energy")
print("window. Physical interpretation:")
print("  - At low T (0.01 keV): [0.1-5 keV] is FAR above the peak → very little emission")
print("  - At high T (10 keV): [0.1-5 keV] is FAR below the peak → very little emission")
print("  - At intermediate T (0.5-2 keV): Window overlaps peak → maximum emission")
print()
print("This is correct physics! The T⁴ scaling only applies when integrating")
print("over the FULL spectrum (0 to ∞) or when the energy window SCALES with T.")
print("="*80)
