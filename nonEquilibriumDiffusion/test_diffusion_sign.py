"""Test to check if diffusion operator has correct sign"""
import numpy as np

# Simple test: 3 cells, uniform grid
# Cell 0: phi = 1.0
# Cell 1: phi = 2.0  (higher than neighbors)
# Cell 2: phi = 1.0

# With correct diffusion, cell 1 should decrease (smoothing)
# With wrong sign diffusion, cell 1 will increase (amplification)

# Discretization of -θ∇·D∇φ for backward Euler (θ=1)
# Current implementation:
#   diag[i] -= coeff_left + coeff_right  (negative)
#   sub[i-1] = +coeff_left                (positive)
#   super[i] = +coeff_right               (positive)

# For cell 1 (middle cell), assuming D=1, Δx=1, A=1, V=1:
# coeff_left = coeff_right = 1
# Matrix row for cell 1: [+1, -2, +1] for [φ_0, φ_1, φ_2]
# So: -∇·D∇φ_1 = +1*φ_0 - 2*φ_1 + 1*φ_2 = 1 - 4 + 1 = -2

# For standard diffusion equation ∂φ/∂t = ∇·D∇φ, we'd have:
# ∇·D∇φ_1 = -1*φ_0 + 2*φ_1 - 1*φ_2 = -1 + 4 - 1 = +2
# This is positive, which makes sense: flux INTO cell 1 from neighbors

# But our equation has ∂φ/∂t - ∇·D∇φ = RHS
# If we move -∇·D∇φ to LHS, we get: ∂φ/∂t = ∇·D∇φ + RHS
# Wait, that's the correct sign for diffusion!

# Let me think about this differently.
# The equation is: (φ^{n+1} - φ^n)/(c·Δt) - ∇·D∇φ^{n+1} = RHS
# Rearranging: φ^{n+1}/(c·Δt) - ∇·D∇φ^{n+1} = φ^n/(c·Δt) + RHS
# Or: [1/(c·Δt) - ∇·D∇]φ^{n+1} = φ^n/(c·Δt) + RHS

# For cell 1 with standard ∇·D∇φ = -φ_0 + 2φ_1 - φ_2:
# [1/(c·Δt) - (-φ_0 + 2φ_1 - φ_2)]φ^{n+1} = ...
# [1/(c·Δt)]φ_1^{n+1} - (-φ_0^{n+1} + 2φ_1^{n+1} - φ_2^{n+1}) = ...
# [1/(c·Δt)]φ_1^{n+1} + φ_0^{n+1} - 2φ_1^{n+1} + φ_2^{n+1} = ...
# φ_0^{n+1} + [1/(c·Δt) - 2]φ_1^{n+1} + φ_2^{n+1} = ...

# Matrix coefficients: [+1, 1/(c·Δt)-2, +1]
# This matches what the code has!

print("Standard ∇·D∇φ discretization:")
print("  For cell i: [-coeff_left, +(coeff_left+coeff_right), -coeff_right]")
print("  For φ = [1, 2, 1]: ∇·D∇φ_1 = -1 + 4 - 1 = +2 (flux IN)")
print()
print("Equation: ∂φ/∂t - ∇·D∇φ = RHS")
print("Rearranging: [1/Δt - ∇·D∇]φ^{n+1} = φ^n/Δt + RHS")
print()
print("For -∇·D∇φ in operator:")
print("  -∇·D∇ gives: [+coeff_left, -(coeff_left+coeff_right), +coeff_right]")
print("  This SHOULD be anti-diffusive... but equation requires it?")
print()
print("WAIT! Equation has a MINUS sign: ∂φ/∂t - ∇·D∇φ = RHS")
print("This is WRONG! For radiation diffusion, it should be:")
print("  ∂φ/∂t + ∇·D∇φ = RHS  (note the PLUS)")
print()
print("The issue is likely in the original equation formulation!")
