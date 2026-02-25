"""
Analyze how diffusion coefficient is calculated on faces in twoDFV.py
"""
import numpy as np
import sys
sys.path.append('.')

# Physical constants (from twoDFV.py)
A_RAD = 0.01372
C_LIGHT = 29.9792458
RHO = 1.0

print("="*70)
print("Analysis: Diffusion Coefficient Calculation on Faces")
print("="*70)

print("\n--- Current Implementation ---")
print("For an interior face between cells i-1 and i:")
print("  1. T_face = 0.5 * (T[i-1] + T[i])  (average temperature)")
print("  2. x_face = x_faces[i]  (face coordinate)")
print("  3. sigma_R = rosseland_opacity(T_face, x_face, y_face)")
print("  4. Compute flux limiter λ(R) where R = |∇φ|/(σ_R·φ)")
print("  5. D_face = λ(R) / sigma_R")
print("  6. Use: coeff = A_face * D_face / dx")
print("     where dx = x_centers[i] - x_centers[i-1]")

print("\n--- Issue for Heterogeneous Materials ---")
print("Problem: Evaluating sigma_R at the FACE location (x_face)")
print("  - For a material interface at x=0.5:")
print("    * Left cell (x<0.5): has sigma_left")
print("    * Right cell (x>0.5): has sigma_right")
print("    * Face (x=0.5): sigma_R(T_face, 0.5, y) is ambiguous!")
print("  - The current approach doesn't properly account for")
print("    different material properties on either side")

print("\n--- Correct Approach for Heterogeneous Materials ---")
print("Should use harmonic mean:")
print("  1. T_avg = 0.5 * (T[i-1] + T[i])")
print("  2. Evaluate D on LEFT side at LEFT cell center:")
print("     sigma_left = rosseland_opacity(T_avg, x_centers[i-1], y_centers[j])")
print("     D_left = λ_left(R) / sigma_left")
print("  3. Evaluate D on RIGHT side at RIGHT cell center:")
print("     sigma_right = rosseland_opacity(T_avg, x_centers[i], y_centers[j])")
print("     D_right = λ_right(R) / sigma_right")
print("  4. Harmonic mean:")
print("     D_harmonic = (dx_left + dx_right) / (dx_left/D_left + dx_right/D_right)")
print("     where dx_left = x_face - x_centers[i-1]")
print("           dx_right = x_centers[i] - x_face")
print("  5. For uniform grid: dx_left = dx_right, so:")
print("     D_harmonic = 2 / (1/D_left + 1/D_right)")

print("\n--- Numerical Example ---")
# Example with discontinuous opacity
T_left = 1.0  # keV
T_right = 1.0 # keV
T_avg = 0.5 * (T_left + T_right)

sigma_left = 10.0  # high opacity (optically thick)
sigma_right = 1.0   # low opacity (optically thin)

D_left = 1.0 / (3.0 * sigma_left)
D_right = 1.0 / (3.0 * sigma_right)

print(f"Left cell:  σ_R = {sigma_left} cm⁻¹,  D = {D_left:.6f} cm")
print(f"Right cell: σ_R = {sigma_right} cm⁻¹, D = {D_right:.6f} cm")

# Current approach (evaluating at interface)
print(f"\nCurrent approach (ambiguous at interface):")
print(f"  If using left:  D_face ≈ {D_left:.6f} cm")
print(f"  If using right: D_face ≈ {D_right:.6f} cm")
print(f"  Depends on which side of x=0.5 is evaluated!")

# Correct approach (harmonic mean)
D_harmonic = 2.0 / (1.0/D_left + 1.0/D_right)
print(f"\nHarmonic mean (correct):")
print(f"  D_harmonic = 2/(1/{D_left:.6f} + 1/{D_right:.6f})")
print(f"             = {D_harmonic:.6f} cm")

print(f"\nNote: Harmonic mean is closer to the SMALLER value")
print(f"      This makes physical sense: the high-opacity region")
print(f"      (small D) dominates the resistance to diffusion")

print("\n--- Recommendation ---")
print("Modify get_diffusion_coefficient() to:")
print("  1. Accept T_avg, x_left, y_left, x_right, y_right")
print("  2. Compute D_left and D_right at respective cell centers")
print("  3. Return harmonic mean weighted by mesh spacing")

print("\nOR create a new method:")
print("  get_face_diffusion_harmonic(...) for heterogeneous problems")

print("\n" + "="*70)
