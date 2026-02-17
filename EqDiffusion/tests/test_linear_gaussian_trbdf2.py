#!/usr/bin/env python3
"""
Test with linear diffusion and analytical Gaussian solution using TR-BDF2

For the problem to be linear, we need:
1. Constant opacity σ_R → constant diffusion coefficient D = c/(3σ_R)
2. Material energy proportional to radiation energy: e_mat ∝ Er

For e_mat ∝ Er:
  - Since Er = a*T^4 where a = 7.5657e-15 GJ/(cm³·keV^4)
  - And e_mat = ρ*cv*T
  - We need: ρ*cv*T ∝ T^4
  - Therefore: cv ∝ T^3

With these conditions, the coupled diffusion equation becomes:
  ∂Er/∂t = D*∇²Er
  
which has analytical Gaussian solutions that spread with time.

For a 1D Gaussian initial condition:
  Er(x, 0) = A*exp(-(x-x0)²/(2σ0²))
  
The solution evolves as:
  Er(x, t) = A*σ0/σ(t) * exp(-(x-x0)²/(2σ(t)²))
  
where σ(t)² = σ0² + 2*D*t

This test uses TR-BDF2 time integration and verifies the solver correctly reproduces 
the analytical solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT
)


def analytical_gaussian_1d(x, x0, sigma0, D, t, amplitude, k_coupling=0.0):
    """
    Analytical solution for 1D diffusion equation with Gaussian initial condition.
    
    For ∂u/∂t = D*∇²u with u(x,0) = A*exp(-(x-x0)²/(2σ0²))
    
    With material coupling e_mat = k*Er, the effective equation is:
      (1+k) * ∂Er/∂t = D*∇²Er
    So the effective diffusion coefficient is D_eff = D/(1+k)
    
    The solution is:
      u(x,t) = A*σ0/σ(t) * exp(-(x-x0)²/(2σ(t)²))
    where:
      σ(t)² = σ0² + 2*D_eff*t = σ0² + 2*D*t/(1+k)
    
    The amplitude decreases as σ0/σ(t) to conserve total energy.
    """
    D_eff = D / (1.0 + k_coupling)  # Effective diffusion with material coupling
    sigma_t_sq = sigma0**2 + 2 * D_eff * t
    sigma_t = np.sqrt(sigma_t_sq)
    
    # Amplitude decreases to conserve integral
    amplitude_t = amplitude * sigma0 / sigma_t
    
    return amplitude_t * np.exp(-(x - x0)**2 / (2 * sigma_t_sq))


def test_linear_gaussian_trbdf2():
    """
    Test solver against analytical Gaussian diffusion solution using TR-BDF2.
    """
    print("="*70)
    print("Linear Gaussian Diffusion Test (TR-BDF2)")
    print("Analytical solution verification with TR-BDF2 time integration")
    print("="*70)
    
    # Problem setup for LINEAR diffusion
    # Need: cv ∝ T³ so that e_mat = ρ*cv*T ∝ T⁴ ∝ Er
    
    # Constant opacity
    sigma_R = 100.0  # cm^-1
    D = C_LIGHT / (3.0 * sigma_R)  # Constant diffusion coefficient
    
    print(f"\nPhysical setup:")
    print(f"  Constant opacity: σ_R = {sigma_R} cm⁻¹")
    print(f"  Diffusion coefficient: D = c/(3σ_R) = {D:.6e} cm²/sh")
    print(f"  Heat capacity: cv(T) = β*T³ for linearity")
    print(f"  Time integration: TR-BDF2")
    
    def constant_opacity(Er):
        """Constant opacity for constant D"""
        return sigma_R
    
    # For linear coupling: e_mat ∝ T⁴ ∝ Er
    # Choose coefficient to give reasonable values
    # Since Er = A_RAD*T⁴, we want e_mat = k*Er for some constant k
    # Use small k to avoid strong coupling that causes instability
    k_coupling = 1.00
    
    # For consistency with the solver's Newton method:
    # If e_mat = k*A_RAD*T⁴, then ρ*cv(T) = de_mat/dT = 4*k*A_RAD*T³
    # Note: ρ is absorbed into the cv definition for this test
    
    def cubic_cv(T):
        """cv ∝ T³ for linear e_mat-Er coupling
        
        Chosen so that e_mat = ∫ ρ*cv dT = k*A_RAD*T⁴
        Requires: ρ*cv = 4*k*A_RAD*T³
        """
        return 4.0 * k_coupling * A_RAD * T**3
    
    def linear_material_energy(T):
        """e_mat = k*A_RAD*T⁴ ∝ Er for linear coupling"""
        return k_coupling * A_RAD * T**4
    
    print(f"  Coupling strength: k = {k_coupling}")
    print(f"  At T=1 keV: cv = {cubic_cv(1.0):.6f}, e_mat = {linear_material_energy(1.0):.6e} GJ/cm³")
    
    # Domain and discretization
    r_min = 0.0
    r_max = 2.0
    n_cells = 200
    
    # Gaussian parameters
    x0 = 1.0  # Center
    sigma0 = 0.15  # Initial width
    T_peak = 1.0  # Peak temperature (keV)
    T_background = 0.1  # Small background to avoid zeros
    
    # Convert to Er
    Er_peak = A_RAD * T_peak**4
    Er_background = A_RAD * T_background**4
    amplitude = Er_peak - Er_background
    
    print(f"\nInitial Gaussian:")
    print(f"  Center: x0 = {x0} cm")
    print(f"  Width: σ0 = {sigma0} cm")
    print(f"  Peak temperature: {T_peak} keV")
    print(f"  Peak Er: {Er_peak:.6e} GJ/cm³")
    
    # Boundary conditions: zero flux (Neumann) - more natural for isolated pulse
    def left_bc(Er, x):
        """Zero flux BC: dEr/dx = 0"""
        return 0.0, 1.0, 0.0
    
    def right_bc(Er, x):
        """Zero flux BC: dEr/dx = 0"""
        return 0.0, 1.0, 0.0
    
    # Time stepping
    dt = 0.001  # Small time step (sh)
    n_steps = 500
    t_final = dt * n_steps
    
    print(f"\nTime integration:")
    print(f"  Time step: dt = {dt} sh")
    print(f"  Number of steps: {n_steps}")
    print(f"  Final time: t = {t_final} sh")
    
    # Expected spreading (accounting for material coupling)
    D_eff = D / (1.0 + k_coupling)
    sigma_final_sq = sigma0**2 + 2 * D_eff * t_final
    sigma_final = np.sqrt(sigma_final_sq)
    print(f"\nExpected spreading:")
    print(f"  Effective D = D/(1+k) = {D_eff:.6e} cm²/sh")
    print(f"  σ(t) = √(σ0² + 2*D_eff*t)")
    print(f"  σ(0) = {sigma0:.6f} cm")
    print(f"  σ({t_final}) = {sigma_final:.6f} cm")
    print(f"  Increase: {(sigma_final/sigma0 - 1)*100:.1f}%")
    
    # Create solver
    print("\n" + "-"*70)
    print("Running numerical simulation with TR-BDF2...")
    print("-"*70)
    
    solver = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=50, newton_tol=1e-6,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    solver.gamma_trbdf2 = 2-np.sqrt(2)/2  # Standard value for TR-BDF2
    solver.max_newton_iter_per_step = 50
    # Initial condition
    def gaussian_Er(r):
        return Er_background + amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))
    
    solver.set_initial_condition(gaussian_Er)
    
    # Get initial solution
    r, Er_init = solver.get_solution()
    T_init = temperature_from_Er(Er_init)
    
    print(f"Initial peak Er (numerical): {Er_init.max():.6e} GJ/cm³")
    print(f"Initial peak T (numerical): {T_init.max():.4f} keV")
    
    # Advance in time with TR-BDF2
    print("\nTime stepping with TR-BDF2:")
    for step in range(n_steps):
        solver.time_step_trbdf2(n_steps=1, verbose=False)
        if step % 10 == 0:
            r_temp, Er_temp = solver.get_solution()
            T_temp = temperature_from_Er(Er_temp)
            print(f"  Step {step:3d}/{n_steps}: max T = {T_temp.max():.4f} keV, max Er = {Er_temp.max():.4e} GJ/cm³")
            if Er_temp.max() > 100 * Er_peak:
                print(f"  ERROR: Solution blowing up, stopping early")
                break
    
    # Get final solution
    r, Er_final = solver.get_solution()
    T_final = temperature_from_Er(Er_final)
    
    print(f"Final peak Er (numerical): {Er_final.max():.6e} GJ/cm³")
    print(f"Final peak T (numerical): {T_final.max():.4f} keV")
    
    # Compute analytical solutions (including material coupling effect)
    Er_analytical_init = Er_background + analytical_gaussian_1d(r, x0, sigma0, D, 0.0, amplitude, k_coupling)
    Er_analytical_final = Er_background + analytical_gaussian_1d(r, x0, sigma0, D, t_final, amplitude, k_coupling)
    
    T_analytical_init = temperature_from_Er(Er_analytical_init)
    T_analytical_final = temperature_from_Er(Er_analytical_final)
    
    print(f"\nFinal peak Er (analytical): {Er_analytical_final.max():.6e} GJ/cm³")
    print(f"Final peak T (analytical): {T_analytical_final.max():.4f} keV")
    
    # Compute errors
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # L2 error in Er
    Er_error = Er_final - Er_analytical_final
    L2_error_Er = np.sqrt(np.mean(Er_error**2))
    L2_norm_Er = np.sqrt(np.mean(Er_analytical_final**2))
    relative_L2_Er = L2_error_Er / L2_norm_Er
    
    # Max error in Er
    max_error_Er = np.max(np.abs(Er_error))
    max_Er = np.max(Er_analytical_final)
    relative_max_Er = max_error_Er / max_Er
    
    print(f"\nRadiation energy Er:")
    print(f"  L2 error: {L2_error_Er:.6e} GJ/cm³")
    print(f"  Relative L2 error: {relative_L2_Er:.6e} ({relative_L2_Er*100:.4f}%)")
    print(f"  Max error: {max_error_Er:.6e} GJ/cm³")
    print(f"  Relative max error: {relative_max_Er:.6e} ({relative_max_Er*100:.4f}%)")
    
    # Temperature errors
    T_error = T_final - T_analytical_final
    L2_error_T = np.sqrt(np.mean(T_error**2))
    L2_norm_T = np.sqrt(np.mean(T_analytical_final**2))
    relative_L2_T = L2_error_T / L2_norm_T
    
    max_error_T = np.max(np.abs(T_error))
    max_T = np.max(T_analytical_final)
    relative_max_T = max_error_T / max_T
    
    print(f"\nTemperature T:")
    print(f"  L2 error: {L2_error_T:.6e} keV")
    print(f"  Relative L2 error: {relative_L2_T:.6e} ({relative_L2_T*100:.4f}%)")
    print(f"  Max error: {max_error_T:.6e} keV")
    print(f"  Relative max error: {relative_max_T:.6e} ({relative_max_T*100:.4f}%)")
    
    # Check if test passes
    tolerance = 0.02  # 2% relative error
    print(f"\n" + "-"*70)
    if relative_L2_Er < tolerance and relative_L2_T < tolerance:
        print(f"✓ TEST PASSED (TR-BDF2)")
        print(f"  Relative errors below {tolerance*100}% threshold")
    else:
        print(f"✗ TEST FAILED")
        print(f"  Relative errors exceed {tolerance*100}% threshold")
    print("-"*70)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Initial condition
    # Temperature - Initial
    ax = axes[0, 0]
    ax.plot(r, T_init, 'b-', linewidth=2, label='Numerical (TR-BDF2)')
    ax.plot(r, T_analytical_init, 'r--', linewidth=2, label='Analytical', alpha=0.7)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Initial Temperature (t = 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Er - Initial
    ax = axes[0, 1]
    ax.plot(r, Er_init, 'b-', linewidth=2, label='Numerical (TR-BDF2)')
    ax.plot(r, Er_analytical_init, 'r--', linewidth=2, label='Analytical', alpha=0.7)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Radiation Energy Er (GJ/cm³)')
    ax.set_title(f'Initial Er (t = 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error - Initial
    ax = axes[0, 2]
    ax.plot(r, Er_init - Er_analytical_init, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Error in Er (GJ/cm³)')
    ax.set_title('Initial Error (discretization)')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Final condition
    # Temperature - Final
    ax = axes[1, 0]
    ax.plot(r, T_final, 'b-', linewidth=2, label='Numerical (TR-BDF2)')
    ax.plot(r, T_analytical_final, 'r--', linewidth=2, label='Analytical', alpha=0.7)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Final Temperature (t = {t_final:.3f} sh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Er - Final
    ax = axes[1, 1]
    ax.plot(r, Er_final, 'b-', linewidth=2, label='Numerical (TR-BDF2)')
    ax.plot(r, Er_analytical_final, 'r--', linewidth=2, label='Analytical', alpha=0.7)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Radiation Energy Er (GJ/cm³)')
    ax.set_title(f'Final Er (t = {t_final:.3f} sh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error - Final
    ax = axes[1, 2]
    ax.plot(r, Er_error, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Error in Er (GJ/cm³)')
    ax.set_title(f'Final Error (L2 rel: {relative_L2_Er*100:.3f}%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_linear_gaussian_trbdf2.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_linear_gaussian_trbdf2.png'")
    plt.show()
    
    return solver, relative_L2_Er, relative_L2_T


if __name__ == "__main__":
    solver, rel_err_Er, rel_err_T = test_linear_gaussian_trbdf2()
    
    print("\n" + "="*70)
    print("Linear Gaussian diffusion test (TR-BDF2) completed!")
    print("="*70)
