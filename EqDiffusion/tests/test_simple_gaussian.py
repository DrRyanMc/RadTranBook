#!/usr/bin/env python3
"""
Simple Gaussian diffusion test with default material properties

This is a simpler test that uses:
- Constant opacity (constant D)
- Constant specific heat (default)
- Gaussian initial condition

While not exactly linear (e_mat = ρ*cv*T is linear in T, not in Er),
this should be well-behaved and we can check:
1. Total energy conservation (with zero-flux BCs)
2. Peak decreases
3. Width increases
4. No blow-up or instabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT
)


def test_simple_gaussian():
    """
    Test with Gaussian initial condition, constant opacity, constant cv
    """
    print("="*70)
    print("Simple Gaussian Diffusion Test")
    print("="*70)
    
    # Constant opacity
    sigma_R = 100.0  # cm^-1
    D = C_LIGHT / (3.0 * sigma_R)
    
    print(f"\nPhysical setup:")
    print(f"  Constant opacity: σ_R = {sigma_R} cm⁻¹")
    print(f"  Diffusion coefficient: D = c/(3σ_R) = {D:.6e} cm²/sh")
    print(f"  Material: default constant specific heat")
    
    def constant_opacity(Er):
        return sigma_R
    
    # Zero-flux boundaries
    def zero_flux_left(Er, x):
        return 0.0, 1.0, 0.0
    
    def zero_flux_right(Er, x):
        return 0.0, 1.0, 0.0
    
    # Domain
    r_min = 0.0
    r_max = 2.0
    n_cells = 100
    
    # Gaussian parameters
    x0 = 1.0
    sigma0 = 0.15
    T_peak = 0.5  # Modest peak temperature
    T_background = 0.01
    
    Er_peak = A_RAD * T_peak**4
    Er_background = A_RAD * T_background**4
    
    print(f"\nInitial Gaussian:")
    print(f"  Center: x0 = {x0} cm")
    print(f"  Width: σ0 = {sigma0} cm")
    print(f"  Peak T: {T_peak} keV")
    print(f"  Background T: {T_background} keV")
    
    # Time stepping
    dt = 0.01
    n_steps = 10
    t_final = dt * n_steps
    
    print(f"\nTime integration:")
    print(f"  dt = {dt} sh")
    print(f"  steps = {n_steps}")
    print(f"  t_final = {t_final} sh")
    
    # Create solver
    print("\n" + "-"*70)
    print("Creating solver...")
    print("-"*70)
    
    solver = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=20, newton_tol=1e-8,
        rosseland_opacity_func=constant_opacity,
        left_bc_func=zero_flux_left,
        right_bc_func=zero_flux_right
    )
    
    # Initial condition
    def gaussian_Er(r):
        T = T_background + (T_peak - T_background) * np.exp(-(r - x0)**2 / (2 * sigma0**2))
        return A_RAD * T**4
    
    solver.set_initial_condition(gaussian_Er)
    
    r, Er_init = solver.get_solution()
    T_init = temperature_from_Er(Er_init)
    
    # Compute total energy
    _, V_cells = solver.A_faces, solver.V_cells
    total_energy_init = np.sum(Er_init * V_cells)
    
    print(f"\nInitial state:")
    print(f"  Peak T: {T_init.max():.4f} keV")
    print(f"  Peak Er: {Er_init.max():.6e} GJ/cm³")
    print(f"  Total radiation energy: {total_energy_init:.6e} GJ")
    
    # Time step with monitoring
    print("\nTime stepping:")
    for step in range(n_steps):
        solver.time_step(n_steps=1, verbose=False)
        if step % 2 == 0 or step == n_steps - 1:
            r_temp, Er_temp = solver.get_solution()
            T_temp = temperature_from_Er(Er_temp)
            total_energy_temp = np.sum(Er_temp * V_cells)
            energy_change = (total_energy_temp - total_energy_init) / total_energy_init
            print(f"  Step {step+1:3d}: max T = {T_temp.max():.4f} keV, total E change = {energy_change*100:+.3f}%")
    
    # Final state
    r, Er_final = solver.get_solution()
    T_final = temperature_from_Er(Er_final)
    total_energy_final = np.sum(Er_final * V_cells)
    
    print(f"\nFinal state:")
    print(f"  Peak T: {T_final.max():.4f} keV")
    print(f"  Peak Er: {Er_final.max():.6e} GJ/cm³")
    print(f"  Total radiation energy: {total_energy_final:.6e} GJ")
    print(f"  Energy change: {(total_energy_final - total_energy_init)/total_energy_init * 100:.3f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    peak_decrease = T_init.max() - T_final.max()
    print(f"\nPeak temperature decreased by: {peak_decrease:.4f} keV ({peak_decrease/T_init.max()*100:.1f}%)")
    
    # Estimate width increase
    def estimate_width(r, T, threshold=0.5):
        """Estimate width at half maximum"""
        T_max = T.max()
        T_half = T_max * threshold
        indices = np.where(T > T_half)[0]
        if len(indices) > 1:
            return r[indices[-1]] - r[indices[0]]
        return 0.0
    
    width_init = estimate_width(r, T_init)
    width_final = estimate_width(r, T_final)
    
    print(f"Width at half-max increased from {width_init:.4f} to {width_final:.4f} cm")
    print(f"Width increase: {(width_final/width_init - 1)*100:.1f}%")
    
    # Check for sanity
    print("\n" + "-"*70)
    if T_final.max() < T_init.max() and T_final.max() > 0:
        print("✓ PASS: Peak decreased as expected")
    else:
        print("✗ FAIL: Peak behavior unexpected")
    
    if abs((total_energy_final - total_energy_init)/total_energy_init) < 0.05:
        print("✓ PASS: Energy approximately conserved (< 5% change)")
    else:
        print("✗ FAIL: Energy not conserved")
    
    if width_final > width_init:
        print("✓ PASS: Width increased as expected")
    else:
        print("✗ FAIL: Width didn't increase")
    print("-"*70)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(r, T_init, 'b--', linewidth=2, label=f't = 0', alpha=0.7)
    ax.plot(r, T_final, 'r-', linewidth=2, label=f't = {t_final} sh')
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(r, Er_init, 'b--', linewidth=2, label=f't = 0', alpha=0.7)
    ax.plot(r, Er_final, 'r-', linewidth=2, label=f't = {t_final} sh')
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Radiation Energy Er (GJ/cm³)')
    ax.set_title('Radiation Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_simple_gaussian.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_simple_gaussian.png'")
    plt.show()
    
    return solver


if __name__ == "__main__":
    solver = test_simple_gaussian()
    
    print("\n" + "="*70)
    print("Simple Gaussian test completed!")
    print("="*70)
