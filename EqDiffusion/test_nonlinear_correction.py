#!/usr/bin/env python3
"""
Test to verify the signs of nonlinear correction terms

Simple 3-cell problem with known gradient to check if nonlinear corrections
enhance or suppress diffusion correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT
)


def test_nonlinear_correction_sign():
    """
    Test nonlinear correction with simple 3-cell problem
    
    Setup:
    - Hot boundary on left (high Er)
    - Cold boundary on right (low Er)
    - Middle cell starts with intermediate value
    - Opacity σ ~ T^-3, so D ~ T^3 (D increases with temperature)
    
    Expected behavior:
    - Without nonlinear correction: standard diffusion
    - With nonlinear correction: should enhance diffusion since dD/dT > 0
      (hot regions have higher D, so flux should be larger)
    
    The nonlinear correction N^(k)[φ] = ∇·(D_E φ ∇E_r) should:
    - Be positive where ∇E_r is large (adds to diffusion)
    - Since D_E > 0 and ∇E_r points from hot to cold (negative gradient)
    - The correction should increase the effective diffusion coefficient
    """
    
    print("="*70)
    print("Test: Nonlinear Correction Sign Verification")
    print("="*70)
    
    # Material properties with strong temperature dependence
    def test_opacity(Er):
        """σ = 100 * T^-3"""
        T = temperature_from_Er(Er)
        T = max(T, 0.01)  # Floor for stability
        return 100.0 * T**(-3)
    
    def test_cv(T):
        return 0.1 / 1.0  # Simple constant
    
    def test_material_energy(T):
        return 0.1 * T
    
    # Boundary conditions: fixed values
    T_left = 1.0   # Hot left boundary
    T_right = 0.2  # Cold right boundary
    
    Er_left = A_RAD * T_left**4
    Er_right = A_RAD * T_right**4
    
    def left_bc(Er, x):
        return 1.0, 0.0, Er_left  # Dirichlet
    
    def right_bc(Er, x):
        return 1.0, 0.0, Er_right  # Dirichlet
    
    # Create solver with just 3 cells
    n_cells = 3
    
    print(f"\nProblem setup:")
    print(f"  Cells: {n_cells}")
    print(f"  Left BC:  T = {T_left:.3f} keV, Er = {Er_left:.4e}")
    print(f"  Right BC: T = {T_right:.3f} keV, Er = {Er_right:.4e}")
    print(f"  Opacity: σ = 100 * T^(-3) => D = c/(3σ) ~ T^3")
    print(f"  Expected: D_E > 0, so nonlinear term should enhance diffusion")
    
    # Test 1: WITHOUT nonlinear correction
    print("\n" + "-"*70)
    print("Test 1: Linear diffusion (no nonlinear correction)")
    print("-"*70)
    
    solver1 = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=n_cells, d=0,
        dt=0.1, max_newton_iter=20, newton_tol=1e-10,
        rosseland_opacity_func=test_opacity,
        specific_heat_func=test_cv,
        material_energy_func=test_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    solver1.use_nonlinear_correction = False
    solver1.max_newton_iter_per_step = 1
    
    # Initial condition: linear interpolation
    def initial_Er(r):
        return Er_left + (Er_right - Er_left) * r / 1.0
    
    solver1.set_initial_condition(initial_Er)
    
    Er_before_1 = solver1.Er.copy()
    print(f"\nBefore time step: Er = {Er_before_1}")
    
    solver1.time_step(n_steps=1, verbose=False)
    
    Er_after_1 = solver1.Er.copy()
    print(f"After time step:  Er = {Er_after_1}")
    
    delta_Er_1 = Er_after_1 - Er_before_1
    print(f"Change (ΔEr):    ΔEr = {delta_Er_1}")
    print(f"Middle cell changed by: {delta_Er_1[n_cells//2]:.4e}")
    
    # Test 2: WITH nonlinear correction
    print("\n" + "-"*70)
    print("Test 2: With nonlinear correction")
    print("-"*70)
    
    solver2 = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=n_cells, d=0,
        dt=0.1, max_newton_iter=20, newton_tol=1e-10,
        rosseland_opacity_func=test_opacity,
        specific_heat_func=test_cv,
        material_energy_func=test_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    solver2.use_nonlinear_correction = True
    solver2.use_secant_derivative = False
    solver2.max_newton_iter_per_step = 1
    
    solver2.set_initial_condition(initial_Er)
    
    Er_before_2 = solver2.Er.copy()
    print(f"\nBefore time step: Er = {Er_before_2}")
    
    solver2.time_step(n_steps=1, verbose=False)
    
    Er_after_2 = solver2.Er.copy()
    print(f"After time step:  Er = {Er_after_2}")
    
    delta_Er_2 = Er_after_2 - Er_before_2
    print(f"Change (ΔEr):    ΔEr = {delta_Er_2}")
    print(f"Middle cell changed by: {delta_Er_2[n_cells//2]:.4e}")
    
    # Compare the results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nChange in middle cell:")
    print(f"  Without nonlinear: {delta_Er_1[n_cells//2]:.4e}")
    print(f"  With nonlinear:    {delta_Er_2[n_cells//2]:.4e}")
    print(f"  Difference:        {(delta_Er_2[n_cells//2] - delta_Er_1[n_cells//2]):.4e}")
    
    # Analyze the effect
    print("\nAnalysis:")
    print(f"  D_E > 0 (since D ~ T^3 and T increases with Er)")
    print(f"  Gradient ∇Er < 0 (Er decreases from left to right)")
    
    # For the middle cell receiving flux from both sides
    # The nonlinear correction N = ∇·(D_E φ ∇Er) with D_E > 0
    # Should increase the effective diffusion
    
    if abs(delta_Er_2[n_cells//2]) > abs(delta_Er_1[n_cells//2]):
        print(f"\n  ✓ PASS: Nonlinear correction INCREASES diffusion (as expected)")
        print(f"    |ΔEr_with| > |ΔEr_without|")
        print(f"    This is CORRECT since D_E > 0 should enhance diffusion")
    else:
        print(f"\n  ✗ FAIL: Nonlinear correction DECREASES diffusion (unexpected!)")
        print(f"    |ΔEr_with| < |ΔEr_without|")
        print(f"    This suggests the sign might be WRONG")
    
    # Additional check: look at the flux balance
    print("\n" + "-"*70)
    print("Detailed flux analysis:")
    print("-"*70)
    
    for i in range(n_cells):
        T_i = temperature_from_Er(Er_before_1[i])
        D_i = solver1.get_diffusion_coefficient(Er_before_1[i])
        print(f"  Cell {i}: T = {T_i:.3f} keV, D = {D_i:.4e} cm²/ns")
    
    return solver1, solver2


def test_simple_gradient():
    """
    Even simpler test: 2-cell problem with fixed boundaries
    Just check the sign of the nonlinear contribution
    """
    print("\n\n" + "="*70)
    print("Test 2: Simple 2-Cell Gradient Test")
    print("="*70)
    
    # Create minimal solver
    n_cells = 2
    
    def test_opacity(Er):
        T = temperature_from_Er(Er)
        T = max(T, 0.01)
        return 10.0 * T**(-2)  # D ~ T^2
    
    T_left = 1.0
    T_right = 0.5
    Er_left = A_RAD * T_left**4
    Er_right = A_RAD * T_right**4
    
    def left_bc(Er, x):
        return 1.0, 0.0, Er_left
    
    def right_bc(Er, x):
        return 1.0, 0.0, Er_right
    
    print(f"\n2-cell problem with gradient:")
    print(f"  Cell 0 (left):  Er = {Er_left:.4e}")
    print(f"  Cell 1 (right): Er = {Er_right:.4e}")
    print(f"  ∇Er < 0 (decreases left to right)")
    print(f"  D ~ T^2, so D_E > 0")
    
    # Without nonlinear
    solver1 = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=n_cells, d=0,
        dt=0.05, max_newton_iter=10, newton_tol=1e-10,
        rosseland_opacity_func=test_opacity,
        left_bc_func=left_bc, right_bc_func=right_bc
    )
    solver1.use_nonlinear_correction = False
    solver1.max_newton_iter_per_step = 1
    
    Er_init = np.array([Er_left, Er_right])
    solver1.Er = Er_init.copy()
    solver1.Er_old = Er_init.copy()
    
    print(f"\nInitial: {solver1.Er}")
    solver1.time_step(n_steps=1, verbose=False)
    delta1 = solver1.Er - Er_init
    print(f"Without NL: {solver1.Er}")
    print(f"Change: {delta1}")
    
    # With nonlinear
    solver2 = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=n_cells, d=0,
        dt=0.05, max_newton_iter=10, newton_tol=1e-10,
        rosseland_opacity_func=test_opacity,
        left_bc_func=left_bc, right_bc_func=right_bc
    )
    solver2.use_nonlinear_correction = True
    solver2.max_newton_iter_per_step = 1
    
    solver2.Er = Er_init.copy()
    solver2.Er_old = Er_init.copy()
    
    solver2.time_step(n_steps=1, verbose=False)
    delta2 = solver2.Er - Er_init
    print(f"\nWith NL: {solver2.Er}")
    print(f"Change: {delta2}")
    
    print(f"\nNonlinear effect: {delta2 - delta1}")
    print(f"\nIf D_E > 0 and gradient exists, nonlinear term should")
    print(f"enhance the diffusion (make changes larger in magnitude)")


if __name__ == "__main__":
    solver1, solver2 = test_nonlinear_correction_sign()
    test_simple_gradient()
    
    print("\n" + "="*70)
    print("Tests completed!")
    print("="*70)
