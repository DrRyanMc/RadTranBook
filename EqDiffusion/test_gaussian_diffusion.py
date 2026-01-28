#!/usr/bin/env python3
"""
Test nonlinear correction with Gaussian initial condition
Away from boundaries to avoid BC complications

Setup:
- Gaussian temperature profile in the middle of domain
- Zero-flux boundaries on both sides
- Diffusion coefficient D ~ T (linear), so D_E > 0
- Nonlinear correction should enhance diffusion at the peak
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT
)


def test_gaussian_diffusion():
    """
    Test with Gaussian initial condition and D ~ T
    """
    print("="*70)
    print("Gaussian Diffusion Test")
    print("="*70)
    
    # Material properties with D ~ T (linear dependence)
    def linear_opacity(Er):
        """σ = 100 / T, so D = c/(3σ) = c*T / 300 ~ T"""
        T = temperature_from_Er(Er)
        T = max(T, 0.01)  # Floor
        return 100.0 / T
    
    def simple_cv(T):
        return 0.1 / 1.0  # Constant specific heat
    
    def simple_material_energy(T):
        return 0.1 * T
    
    # Zero-flux boundary conditions (symmetric)
    def zero_flux_left(Er, x):
        return 0.0, 1.0, 0.0  # dEr/dx = 0
    
    def zero_flux_right(Er, x):
        return 0.0, 1.0, 0.0  # dEr/dx = 0
    
    # Problem setup
    r_min = 0.0
    r_max = 1.0
    n_cells = 100
    dt = 0.01
    n_steps = 10
    
    print(f"\nProblem setup:")
    print(f"  Domain: [{r_min}, {r_max}]")
    print(f"  Cells: {n_cells}")
    print(f"  Time step: {dt}")
    print(f"  Number of steps: {n_steps}")
    print(f"  Opacity: σ = 100/T => D ~ T")
    print(f"  Boundary conditions: zero flux (symmetric)")
    print(f"  Initial condition: Gaussian centered at x=0.5")
    
    # Create Gaussian initial condition
    def gaussian_Er(r):
        r_center = 0.5
        width = 0.1
        T_peak = 1.0    # Peak temperature
        T_base = 0.2    # Background temperature
        
        T = T_base + (T_peak - T_base) * np.exp(-((r - r_center) / width)**2)
        return A_RAD * T**4
    
    # Test 1: WITHOUT nonlinear correction
    print("\n" + "-"*70)
    print("Test 1: Standard diffusion (no nonlinear correction)")
    print("-"*70)
    
    solver1 = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=20, newton_tol=1e-8,
        rosseland_opacity_func=linear_opacity,
        specific_heat_func=simple_cv,
        material_energy_func=simple_material_energy,
        left_bc_func=zero_flux_left,
        right_bc_func=zero_flux_right
    )
    
    solver1.use_nonlinear_correction = False
    solver1.max_newton_iter_per_step = 3
    
    solver1.set_initial_condition(gaussian_Er)
    
    r1, Er1_init = solver1.get_solution()
    T1_init = temperature_from_Er(Er1_init)
    
    print(f"Initial peak temperature: {T1_init.max():.4f} keV")
    
    solver1.time_step(n_steps=n_steps, verbose=False)
    
    r1, Er1_final = solver1.get_solution()
    T1_final = temperature_from_Er(Er1_final)
    
    print(f"Final peak temperature: {T1_final.max():.4f} keV")
    print(f"Peak decreased by: {T1_init.max() - T1_final.max():.4f} keV")
    
    # Test 2: WITH nonlinear correction
    print("\n" + "-"*70)
    print("Test 2: With nonlinear correction")
    print("-"*70)
    
    solver2 = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=20, newton_tol=1e-8,
        rosseland_opacity_func=linear_opacity,
        specific_heat_func=simple_cv,
        material_energy_func=simple_material_energy,
        left_bc_func=zero_flux_left,
        right_bc_func=zero_flux_right
    )
    
    solver2.use_nonlinear_correction = True
    solver2.use_secant_derivative = False
    solver2.max_newton_iter_per_step = 3
    
    solver2.set_initial_condition(gaussian_Er)
    
    r2, Er2_init = solver2.get_solution()
    T2_init = temperature_from_Er(Er2_init)
    
    print(f"Initial peak temperature: {T2_init.max():.4f} keV")
    
    solver2.time_step(n_steps=n_steps, verbose=False)
    
    r2, Er2_final = solver2.get_solution()
    T2_final = temperature_from_Er(Er2_final)
    
    print(f"Final peak temperature: {T2_final.max():.4f} keV")
    print(f"Peak decreased by: {T2_init.max() - T2_final.max():.4f} keV")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    peak_decrease_1 = T1_init.max() - T1_final.max()
    peak_decrease_2 = T2_init.max() - T2_final.max()
    
    print(f"\nPeak temperature decrease:")
    print(f"  Without nonlinear: {peak_decrease_1:.4f} keV")
    print(f"  With nonlinear:    {peak_decrease_2:.4f} keV")
    print(f"  Ratio (with/without): {peak_decrease_2/peak_decrease_1:.3f}")
    
    print(f"\nAnalysis:")
    print(f"  D ~ T (linear), so D_E > 0")
    print(f"  At the peak: T is highest, so D is largest")
    print(f"  Gradient: ∇T points away from peak (outward)")
    print(f"  Expected: Nonlinear correction should ENHANCE diffusion")
    print(f"           (peak should flatten faster)")
    
    if peak_decrease_2 > peak_decrease_1 * 1.05:  # At least 5% more
        print(f"\n  ✓ PASS: Nonlinear correction ENHANCES diffusion")
        print(f"    Peak decreases faster with nonlinear correction")
    elif abs(peak_decrease_2 - peak_decrease_1) < 0.01 * peak_decrease_1:
        print(f"\n  ? NEUTRAL: Nonlinear correction has minimal effect")
        print(f"    This might be OK if the effect is small")
    else:
        print(f"\n  ✗ FAIL: Nonlinear correction SUPPRESSES diffusion")
        print(f"    Peak decreases slower with nonlinear correction")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature profiles
    ax = axes[0, 0]
    ax.plot(r1, T1_init, 'b--', linewidth=2, label='Initial', alpha=0.7)
    ax.plot(r1, T1_final, 'b-', linewidth=2, label='Final (no NL)')
    ax.plot(r2, T2_final, 'r-', linewidth=2, label='Final (with NL)')
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference plot
    ax = axes[0, 1]
    ax.plot(r1, T2_final - T1_final, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('ΔT (with NL - without NL) (keV)')
    ax.set_title('Effect of Nonlinear Correction on Temperature')
    ax.grid(True, alpha=0.3)
    
    # Diffusion coefficient profiles
    ax = axes[1, 0]
    D1_init = np.array([solver1.get_diffusion_coefficient(Er1_init[i]) for i in range(len(Er1_init))])
    D1_final = np.array([solver1.get_diffusion_coefficient(Er1_final[i]) for i in range(len(Er1_final))])
    D2_final = np.array([solver2.get_diffusion_coefficient(Er2_final[i]) for i in range(len(Er2_final))])
    
    ax.plot(r1, D1_init, 'b--', linewidth=2, label='Initial', alpha=0.7)
    ax.plot(r1, D1_final, 'b-', linewidth=2, label='Final (no NL)')
    ax.plot(r2, D2_final, 'r-', linewidth=2, label='Final (with NL)')
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Diffusion Coefficient D (cm²/ns)')
    ax.set_title('Diffusion Coefficient D ~ T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Peak evolution
    ax = axes[1, 1]
    ax.bar(['No NL', 'With NL'], [peak_decrease_1, peak_decrease_2], 
           color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Peak Temperature Decrease (keV)')
    ax.set_title('Diffusion Enhancement by Nonlinear Correction')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gaussian_diffusion_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'gaussian_diffusion_test.png'")
    plt.show()
    
    return solver1, solver2


if __name__ == "__main__":
    solver1, solver2 = test_gaussian_diffusion()
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)
