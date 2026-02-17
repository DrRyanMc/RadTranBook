#!/usr/bin/env python3
"""
Test nonlinear correction at boundaries with simple D ~ T case

Focus on what happens at the first interior cell next to a Dirichlet boundary
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT
)


def test_boundary_cell():
    """
    Simple test with few cells to see boundary behavior
    
    Setup:
    - Only 5 cells
    - Dirichlet BC on left (high T)
    - Zero flux on right
    - D ~ T (linear)
    - Initial condition: linear gradient
    
    Focus: What happens to cell 1 (first interior cell)?
    """
    print("="*70)
    print("Boundary Nonlinear Correction Test")
    print("="*70)
    
    # Material properties with D ~ T
    def linear_D_opacity(Er):
        """σ = 100/T, so D = c/(3σ) = cT/300 ~ T"""
        T = temperature_from_Er(Er)
        T = max(T, 0.01)
        return 100.0 / T
    
    def simple_cv(T):
        return 0.1 / 1.0
    
    def simple_energy(T):
        return 0.1 * T
    
    # Boundary conditions
    T_left = 1.0
    Er_left = A_RAD * T_left**4
    
    def left_bc_robin(Er, x):
        """Robin BC: similar to Marshak incoming flux
        A*Er + B*(dEr/dr) = C
        For incoming radiation: approximately Er = Er_left at boundary
        But let's use a Robin form: Er + D*(dEr/dr) = Er_left
        """
        D_bc = 0.01  # Small diffusive length scale
        return 1.0, D_bc, Er_left  # A=1, B=D_bc, C=Er_left
    
    def right_bc_zero_flux(Er, x):
        return 0.0, 1.0, 0.0  # Zero flux: dEr/dx = 0
    
    # Problem setup
    n_cells = 5
    r_min = 0.0
    r_max = 1.0
    dt = 0.01
    
    print(f"\nProblem setup:")
    print(f"  Cells: {n_cells}")
    print(f"  Domain: [{r_min}, {r_max}]")
    print(f"  Dt: {dt}")
    print(f"  Left BC: ROBIN (flux-like), target T = {T_left} keV")
    print(f"  Right BC: Zero flux")
    print(f"  D ~ T (linear)")
    
    # Initial condition: linear temperature profile
    def initial_Er_linear(r):
        T = T_left * (1.0 - 0.5 * r)  # Decreases linearly
        return A_RAD * T**4
    
    print("\n" + "-"*70)
    print("Test 1: WITHOUT nonlinear correction")
    print("-"*70)
    
    solver1 = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=10, newton_tol=1e-10,
        rosseland_opacity_func=linear_D_opacity,
        specific_heat_func=simple_cv,
        material_energy_func=simple_energy,
        left_bc_func=left_bc_robin,
        right_bc_func=right_bc_zero_flux
    )
    
    solver1.use_nonlinear_correction = False
    solver1.max_newton_iter_per_step = 1
    
    solver1.set_initial_condition(initial_Er_linear)
    
    r1, Er1_before = solver1.get_solution()
    T1_before = temperature_from_Er(Er1_before)
    
    print(f"\nBefore time step:")
    print(f"  Positions: {r1}")
    print(f"  T: {T1_before}")
    
    solver1.time_step(n_steps=1, verbose=False)
    
    r1, Er1_after = solver1.get_solution()
    T1_after = temperature_from_Er(Er1_after)
    
    print(f"\nAfter time step:")
    print(f"  T: {T1_after}")
    print(f"  ΔT: {T1_after - T1_before}")
    
    print(f"\nFocus on cell 1 (first interior):")
    print(f"  T before: {T1_before[1]:.6f}")
    print(f"  T after:  {T1_after[1]:.6f}")
    print(f"  Change:   {T1_after[1] - T1_before[1]:.6e}")
    
    print("\n" + "-"*70)
    print("Test 2: WITH nonlinear correction")
    print("-"*70)
    
    solver2 = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=10, newton_tol=1e-10,
        rosseland_opacity_func=linear_D_opacity,
        specific_heat_func=simple_cv,
        material_energy_func=simple_energy,
        left_bc_func=left_bc_robin,
        right_bc_func=right_bc_zero_flux
    )
    
    solver2.use_nonlinear_correction = True
    solver2.use_secant_derivative = False
    solver2.max_newton_iter_per_step = 1
    
    solver2.set_initial_condition(initial_Er_linear)
    
    r2, Er2_before = solver2.get_solution()
    T2_before = temperature_from_Er(Er2_before)
    
    print(f"\nBefore time step:")
    print(f"  T: {T2_before}")
    
    solver2.time_step(n_steps=1, verbose=False)
    
    r2, Er2_after = solver2.get_solution()
    T2_after = temperature_from_Er(Er2_after)
    
    print(f"\nAfter time step:")
    print(f"  T: {T2_after}")
    print(f"  ΔT: {T2_after - T2_before}")
    
    print(f"\nFocus on cell 1 (first interior):")
    print(f"  T before: {T2_before[1]:.6f}")
    print(f"  T after:  {T2_after[1]:.6f}")
    print(f"  Change:   {T2_after[1] - T2_before[1]:.6e}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nCell 0 (boundary - should be identical):")
    print(f"  Without NL: {T1_after[0]:.6f} (should equal {T_left:.6f})")
    print(f"  With NL:    {T2_after[0]:.6f} (should equal {T_left:.6f})")
    print(f"  Difference: {T2_after[0] - T1_after[0]:.2e}")
    
    print(f"\nCell 1 (first interior cell):")
    print(f"  Change without NL: {T1_after[1] - T1_before[1]:.6e}")
    print(f"  Change with NL:    {T2_after[1] - T2_before[1]:.6e}")
    print(f"  Difference:        {(T2_after[1] - T2_before[1]) - (T1_after[1] - T1_before[1]):.6e}")
    
    # Check for spikes
    spike_detected = False
    for i in range(n_cells):
        if abs(T2_after[i] - T1_after[i]) > 0.1:  # More than 0.1 keV difference
            print(f"\n  ⚠ WARNING: Large difference at cell {i}: {T2_after[i] - T1_after[i]:.4f} keV")
            spike_detected = True
    
    if not spike_detected:
        print(f"\n  ✓ No large spikes detected")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(r1, T1_before, 'ko--', label='Initial', markersize=8)
    ax.plot(r1, T1_after, 'bs-', label='After (no NL)', markersize=8)
    ax.plot(r2, T2_after, 'r^-', label='After (with NL)', markersize=8)
    ax.axvline(r1[0], color='gray', linestyle=':', alpha=0.5, label='Boundary')
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    change_no_nl = T1_after - T1_before
    change_with_nl = T2_after - T2_before
    ax.plot(r1, change_no_nl, 'bs-', label='No NL', markersize=8)
    ax.plot(r2, change_with_nl, 'r^-', label='With NL', markersize=8)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(r1[0], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature Change ΔT (keV)')
    ax.set_title('Effect of One Time Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boundary_nonlinear_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'boundary_nonlinear_test.png'")
    plt.show()
    
    return solver1, solver2


if __name__ == "__main__":
    solver1, solver2 = test_boundary_cell()
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)
