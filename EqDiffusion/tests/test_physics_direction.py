#!/usr/bin/env python3
"""
Physics Verification: Check if nonlinear corrections have correct sign

This script creates a simple test case to verify the physics direction
of the nonlinear corrections.
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
    C_LIGHT,
    RHO
)


def marshak_opacity(Er):
    """Temperature-dependent Rosseland opacity: σ_R = 3 * T^-3"""
    T = temperature_from_Er(Er)  # keV
    n = 3
    T_min = 0.05  # Minimum temperature to prevent overflow (keV)
    if T < T_min:
        T = T_min
    return 3.0 * T**(-n)


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)"""
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T):
    """Material energy density: e = c_v * T (volumetric)"""
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric * T


def test_physics_direction():
    """Test the physics direction with a simple step function"""
    
    print("="*70)
    print("PHYSICS DIRECTION VERIFICATION")
    print("="*70)
    
    # Simple step function test: hot left, cold right
    # Expected: nonlinear should make hot region diffuse faster
    
    # Domain setup
    r_min = 0.0
    r_max = 1.0
    n_cells = 50
    
    # Time parameters
    dt = 0.001
    t_final = 0.01  # Very short time to see initial effect
    
    print(f"Test setup:")
    print(f"  Domain: [0, 1] cm, {n_cells} cells")
    print(f"  Time: {t_final} ns, dt = {dt} ns")
    print(f"  Initial: Step function T(x<0.5) = 1 keV, T(x≥0.5) = 0.2 keV")
    
    # Test both linear and nonlinear
    results = {}
    
    for use_nl, name in [(False, "Linear"), (True, "Nonlinear")]:
        print(f"\n--- {name} Case ---")
        
        # Create solver
        solver = RadiationDiffusionSolver(
            r_min=r_min,
            r_max=r_max,
            n_cells=n_cells,
            d=0,
            dt=dt,
            max_newton_iter=20,
            newton_tol=1e-8,
            rosseland_opacity_func=marshak_opacity,
            specific_heat_func=marshak_specific_heat,
            material_energy_func=marshak_material_energy,
            left_bc_func=lambda Er, x: (0.0, 1.0, 0.0),  # Zero flux
            right_bc_func=lambda Er, x: (0.0, 1.0, 0.0)  # Zero flux
        )
        
        # Configure nonlinear corrections
        solver.use_nonlinear_correction = use_nl
        if use_nl:
            solver.use_secant_derivative = False
            solver.max_newton_iter_per_step = 20
            solver.nonlinear_skip_boundary_cells = 0
            solver.nonlinear_limiter = 0.5  # Moderate limiter
        
        # Step function initial condition
        def initial_Er(r):
            T_init = np.where(r < 0.5, 1.0, 0.2)  # Step function
            return A_RAD * T_init**4
        
        solver.set_initial_condition(initial_Er)
        
        # Get initial state
        r0, Er0 = solver.get_solution()
        T0 = temperature_from_Er(Er0)
        
        print(f"  Initial: max T = {T0.max():.3f}, min T = {T0.min():.3f}")
        
        # Evolve for short time
        n_steps = int(t_final / dt)
        for step in range(n_steps):
            solver.time_step(n_steps=1, verbose=False)
        
        # Get final state
        r_final, Er_final = solver.get_solution()
        T_final = temperature_from_Er(Er_final)
        
        print(f"  Final:   max T = {T_final.max():.3f}, min T = {T_final.min():.3f}")
        
        # Store results
        results[name] = {
            'r': r_final.copy(),
            'T_initial': np.interp(r_final, r0, T0),
            'T_final': T_final.copy(),
            'max_T': T_final.max(),
            'min_T': T_final.min()
        }
    
    # Analysis
    print(f"\n{'='*70}")
    print("PHYSICS ANALYSIS")
    print(f"{'='*70}")
    
    linear = results["Linear"]
    nonlinear = results["Nonlinear"]
    
    # Compare diffusion rates by looking at temperature spread
    linear_spread = linear['max_T'] - linear['min_T']
    nonlinear_spread = nonlinear['max_T'] - nonlinear['min_T']
    
    print(f"Temperature spread (max - min):")
    print(f"  Linear:    {linear_spread:.4f} keV")
    print(f"  Nonlinear: {nonlinear_spread:.4f} keV")
    print(f"  Difference: {nonlinear_spread - linear_spread:.4f} keV")
    
    # Expected physics: Higher T → Lower opacity → Higher diffusion → Faster heat transfer
    # This should REDUCE the temperature spread (more mixing)
    if nonlinear_spread < linear_spread:
        print(f"  ✓ CORRECT: Nonlinear reduces spread (enhances diffusion)")
        physics_correct = True
    else:
        print(f"  ✗ WRONG: Nonlinear increases spread (reduces diffusion)")
        physics_correct = False
    
    # Look at specific regions
    # Hot region should cool faster with nonlinear (enhanced outward diffusion)
    # Cold region should heat faster with nonlinear (enhanced inward diffusion)
    
    hot_region = linear['r'] < 0.3  # Left side
    cold_region = linear['r'] > 0.7  # Right side
    
    linear_hot_avg = np.mean(linear['T_final'][hot_region])
    nonlinear_hot_avg = np.mean(nonlinear['T_final'][hot_region])
    
    linear_cold_avg = np.mean(linear['T_final'][cold_region])
    nonlinear_cold_avg = np.mean(nonlinear['T_final'][cold_region])
    
    print(f"\nRegional analysis:")
    print(f"  Hot region (r < 0.3):")
    print(f"    Linear: {linear_hot_avg:.4f} keV")
    print(f"    Nonlinear: {nonlinear_hot_avg:.4f} keV")
    print(f"    Change: {nonlinear_hot_avg - linear_hot_avg:.4f} keV")
    
    print(f"  Cold region (r > 0.7):")
    print(f"    Linear: {linear_cold_avg:.4f} keV")
    print(f"    Nonlinear: {nonlinear_cold_avg:.4f} keV")
    print(f"    Change: {nonlinear_cold_avg - linear_cold_avg:.4f} keV")
    
    # Expected: hot region cools more, cold region heats more
    hot_cools_more = nonlinear_hot_avg < linear_hot_avg
    cold_heats_more = nonlinear_cold_avg > linear_cold_avg
    
    if hot_cools_more and cold_heats_more:
        print(f"  ✓ CORRECT: Enhanced mixing (hot cools more, cold heats more)")
        regional_correct = True
    else:
        print(f"  ✗ WRONG: Reduced mixing (opposite of expected)")
        regional_correct = False
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperature profiles
    ax1.plot(linear['r'], linear['T_initial'], 'k--', linewidth=1, label='Initial', alpha=0.5)
    ax1.plot(linear['r'], linear['T_final'], 'b-', linewidth=2, label='Linear final')
    ax1.plot(nonlinear['r'], nonlinear['T_final'], 'r-', linewidth=2, label='Nonlinear final')
    ax1.set_xlabel('Position r (cm)')
    ax1.set_ylabel('Temperature T (keV)')
    ax1.set_title(f'Temperature Profiles (t = {t_final} ns)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference plot
    T_diff = nonlinear['T_final'] - linear['T_final']
    ax2.plot(nonlinear['r'], T_diff, 'g-', linewidth=2, label='T_nonlinear - T_linear')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position r (cm)')
    ax2.set_ylabel('Temperature Difference (keV)')
    ax2.set_title('Nonlinear Effect on Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('physics_direction_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved as 'physics_direction_test.png'")
    plt.show()
    
    # Overall assessment
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")
    
    if physics_correct and regional_correct:
        print("✓ PHYSICS CORRECT: Nonlinear corrections enhance diffusion as expected")
    else:
        print("✗ PHYSICS INCORRECT: Nonlinear corrections have wrong sign or implementation error")
        print("  This suggests a bug in the implementation!")
    
    return physics_correct and regional_correct


if __name__ == "__main__":
    test_physics_direction()