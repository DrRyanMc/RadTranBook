"""
Debug harmonic mean implementation for homogeneous problems
"""
import numpy as np
import sys
sys.path.append('..')

# Physical constants
A_RAD = 0.01372
C_LIGHT = 29.9792458
RHO = 1.0

print("="*70)
print("Debug: Harmonic Mean for Homogeneous Materials")
print("="*70)

# Test case: uniform grid with temperature-dependent but spatially homogeneous opacity
T_left = 0.8  # keV
T_right = 0.6  # keV
T_avg = 0.5 * (T_left + T_right)  # 0.7 keV

# Marshak opacity: σ = 300 * T^(-3)
def marshak_opacity(T):
    return 300.0 * T**(-3)

# Cell centers and face
x_left = 0.005  # cm
x_right = 0.015  # cm
x_face = 0.010  # cm (midpoint)

dx_left = x_face - x_left  # 0.005
dx_right = x_right - x_face  # 0.005

print(f"\nTest setup:")
print(f"  T_left = {T_left} keV, T_right = {T_right} keV")
print(f"  T_avg = {T_avg} keV")
print(f"  x_left = {x_left} cm, x_face = {x_face} cm, x_right = {x_right} cm")
print(f"  dx_left = {dx_left} cm, dx_right = {dx_right} cm")

# OLD APPROACH: Evaluate at face location with face temperature
print(f"\n--- Old Approach ---")
sigma_face = marshak_opacity(T_avg)
D_old = 1.0 / (3.0 * sigma_face)  # no flux limiting
print(f"  σ_R(T_avg, x_face) = {sigma_face:.6f} cm⁻¹")
print(f"  D_face = {D_old:.6e} cm")

# NEW APPROACH: Harmonic mean evaluated at cell centers
print(f"\n--- New Approach (Harmonic Mean) ---")
# For homogeneous material, σ(T, x_left) = σ(T, x_right) = σ(T) (no x dependence)
sigma_left = marshak_opacity(T_avg)   # at x_left
sigma_right = marshak_opacity(T_avg)  # at x_right
print(f"  σ_R(T_avg, x_left) = {sigma_left:.6f} cm⁻¹")
print(f"  σ_R(T_avg, x_right) = {sigma_right:.6f} cm⁻¹")
print(f"  (These should be equal for homogeneous material)")

D_left = 1.0 / (3.0 * sigma_left)
D_right = 1.0 / (3.0 * sigma_right)
print(f"  D_left = {D_left:.6e} cm")
print(f"  D_right = {D_right:.6e} cm")

dx_total = dx_left + dx_right
D_harmonic = dx_total / (dx_left/D_left + dx_right/D_right)
print(f"  D_harmonic = {D_harmonic:.6e} cm")

print(f"\n--- Comparison ---")
print(f"  D_old = {D_old:.6e} cm")
print(f"  D_harmonic = {D_harmonic:.6e} cm")
print(f"  Difference: {abs(D_old - D_harmonic):.6e} cm")
print(f"  Relative difference: {abs(D_old - D_harmonic)/D_old*100:.6f}%")

if abs(D_old - D_harmonic)/D_old < 1e-10:
    print(f"  ✓ Harmonic mean equals old approach (as expected for homogeneous material)")
else:
    print(f"  ✗ WARNING: Harmonic mean differs from old approach!")

# Test with uniform grid
print(f"\n" + "="*70)
print("For homogeneous materials with uniform grid:")
print("  σ_left = σ_right = σ")
print("  D_left = D_right = D = 1/(3σ)")
print("  D_harmonic = (dx_left + dx_right) / (dx_left/D + dx_right/D)")
print("             = dx_total / (dx_total/D)")
print("             = D")
print("  ✓ Harmonic mean reduces to same value!")
print("="*70)
