"""Test the diffusion operator stencil with a simple case"""
import numpy as np

# Test case: 3 cells with φ = [1, 2, 1]
# With proper diffusion, the center cell should have ∇·D∇φ > 0 (flux IN)
# This will cause φ_center to increase if ∂φ/∂t = ∇·D∇φ

# Current implementation constructs matrix for:
# (1/(c·dt)) * φ^{n+1} - ∇·D∇φ^{n+1} = RHS

# For middle cell (i=1) with uniform D=1, Δx=1, A=1, V=1, θ=1:
print("Test: 3 cells, φ = [1, 2, 1], uniform grid")
print("Current stencil for -∇·D∇φ in matrix:")
print("  sub[0] = +1 (coefficient of φ_0 in equation 1)")
print("  diag[1] = -(1+1) = -2 (coefficient of φ_1 in equation 1)")  
print("  super[1] = +1 (coefficient of φ_2 in equation 1)")
print()
print("Matrix equation 1:  +1*φ_0 - 2*φ_1 + 1*φ_2 + other_terms = RHS")
print("Diffusion contribution: +1*1 - 2*2 + 1*1 = 1 - 4 + 1 = -2")
print()
print("But standard ∇·D∇φ for φ=[1,2,1] should be:")
print("  ∇·D∇φ_1 = (flux_right - flux_left) / V")
print("  flux_right = -D*(φ_2 - φ_1)/Δx = -1*(1-2)/1 = +1")
print("  flux_left = -D*(φ_1 - φ_0)/Δx = -1*(2-1)/1 = -1")
print("  ∇·D∇φ_1 = (1 - (-1))/1 = +2 (flux INTO cell)")
print()
print("So -∇·D∇φ_1 = -2, which matches the stencil!")
print("The stencil appears CORRECT.")
print()
print("=" * 70)
print()

# But then why are we getting oscillations? Let me check if the explicit term has issues
print("Checking explicit diffusion term (for θ < 1)...")
print("Code has:")
print("  flux_left = -D * (φ_i - φ_{i-1}) / dr_left")
print("  explicit_diffusion += A_left * flux_left / V_i")
print("")
print("For i=1: flux_left = -1*(2-1)/1 = -1")
print("  explicit_diffusion += 1*(-1)/1 = -1")
print()
print("Then:")
print("  flux_right = -D * (φ_{i+1} - φ_i) / dr_right")
print("  explicit_diffusion -= A_right * flux_right / V_i")
print()
print("For i=1: flux_right = -1*(1-2)/1 = +1")
print("  explicit_diffusion -= 1*(1)/1 = -1")
print()
print("Total explicit_diffusion = -1 - 1 = -2")
print()
print("But ∇·D∇φ_1 should be +2, not -2!")
print("So the explicit diffusion term has the WRONG SIGN!")
print()
print("However, we're using θ=1 (Backward Euler), so explicit term isn't used.")
print()
print("=" * 70)
print()

# Let me check if there's an issue with how the matrix interacts with boundary conditions
print("Could the issue be in boundary condition application?")
print("Let me check apply_boundary_conditions_phi...")
