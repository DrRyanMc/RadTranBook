#!/usr/bin/env python3
"""
Test 2D solver with linear diffusion and analytical Gaussian solution

This test verifies that the 2D solver reduces to the 1D analytical solution
when one dimension is made very large (so there's no variation in that direction).

We use:
- x-direction: small domain [0, 2] cm where Gaussian pulse evolves
- y-direction: large domain [0, 10^6] cm with reflecting BCs (no variation)

The y-direction should remain uniform, and the x-direction should match
the 1D analytical Gaussian diffusion solution.

For linear diffusion, we need:
1. Constant opacity σ_R → constant diffusion coefficient D = c/(3σ_R)
2. Material energy proportional to radiation energy: e_mat ∝ Er
   Requires cv ∝ T³

The analytical 1D Gaussian solution:
  Er(x, t) = A*σ0/σ(t) * exp(-(x-x0)²/(2σ(t)²))
where:
  σ(t)² = σ0² + 2*D_eff*t
  D_eff = D/(1+k)  (with material coupling k)
"""

import numpy as np
import matplotlib.pyplot as plt
from twoDFV import (
    RadiationDiffusionSolver2D,
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
    """
    D_eff = D / (1.0 + k_coupling)
    sigma_t_sq = sigma0**2 + 2 * D_eff * t
    sigma_t = np.sqrt(sigma_t_sq)
    
    # Amplitude decreases to conserve integral
    amplitude_t = amplitude * sigma0 / sigma_t
    
    return amplitude_t * np.exp(-(x - x0)**2 / (2 * sigma_t_sq))


def test_2d_linear_gaussian(direction='x'):
    """
    Test 2D solver against analytical 1D Gaussian diffusion solution.
    
    Parameters:
    -----------
    direction : str
        'x' to test x-direction (y large), 'y' to test y-direction (x large)
    """
    print("="*70)
    print(f"2D Linear Gaussian Diffusion Test ({direction.upper()}-direction)")
    print("Verification that 2D solver reduces to 1D analytical solution")
    print("="*70)
    
    # Physical setup for LINEAR diffusion
    sigma_R = 100.0  # cm^-1
    D = C_LIGHT / (3.0 * sigma_R)  # Constant diffusion coefficient
    
    print(f"\nPhysical setup:")
    print(f"  Constant opacity: σ_R = {sigma_R} cm⁻¹")
    print(f"  Diffusion coefficient: D = c/(3σ_R) = {D:.6e} cm²/sh")
    
    def constant_opacity(Er):
        """Constant opacity for constant D"""
        return sigma_R
    
    # Linear coupling: e_mat ∝ Er
    k_coupling = 1.0e-5  # Small coupling for stability
    
    def cubic_cv(T):
        """cv ∝ T³ for linear e_mat-Er coupling"""
        return 4.0 * k_coupling * A_RAD * T**3
    
    def linear_material_energy(T):
        """e_mat = k*A_RAD*T⁴ ∝ Er for linear coupling"""
        return k_coupling * A_RAD * T**4
    
    print(f"  Coupling strength: k = {k_coupling}")
    
    # Domain and discretization - depends on direction
    if direction == 'x':
        # x-direction: normal size where variation occurs
        coord1_min = 0.0
        coord1_max = 2.0
        n1_cells = 100
        
        # y-direction: VERY LARGE to suppress variation
        coord2_min = 0.0
        coord2_max = 1.0e6  # 10^6 cm - huge domain
        n2_cells = 3  # Only need a few cells since no variation expected
        
        # Gaussian parameters (in x-direction)
        pulse_center = 1.0  # Center
        
        print(f"\nDomain (Gaussian in x-direction):")
        print(f"  x-direction: [{coord1_min}, {coord1_max}] cm with {n1_cells} cells")
        print(f"  y-direction: [{coord2_min}, {coord2_max}] cm with {n2_cells} cells")
        print(f"  → y-domain is {coord2_max/coord1_max:.0e}x larger to suppress variation")
        
    else:  # direction == 'y'
        # x-direction: VERY LARGE to suppress variation
        coord1_min = 0.0
        coord1_max = 1.0e6  # 10^6 cm - huge domain
        n1_cells = 3  # Only need a few cells since no variation expected
        
        # y-direction: normal size where variation occurs
        coord2_min = 0.0
        coord2_max = 2.0
        n2_cells = 100
        
        # Gaussian parameters (in y-direction)
        pulse_center = 1.0  # Center
        
        print(f"\nDomain (Gaussian in y-direction):")
        print(f"  x-direction: [{coord1_min}, {coord1_max}] cm with {n1_cells} cells")
        print(f"  y-direction: [{coord2_min}, {coord2_max}] cm with {n2_cells} cells")
        print(f"  → x-domain is {coord1_max/coord2_max:.0e}x larger to suppress variation")
    
    # Gaussian parameters
    sigma0 = 0.15  # Initial width
    T_peak = 1.0  # Peak temperature (keV)
    T_background = 0.1  # Background temperature
    
    # Convert to Er
    Er_peak = A_RAD * T_peak**4
    Er_background = A_RAD * T_background**4
    amplitude = Er_peak - Er_background
    
    print(f"\nInitial Gaussian ({direction}-direction):")
    print(f"  Center: {direction}0 = {pulse_center} cm")
    print(f"  Width: σ0 = {sigma0} cm")
    print(f"  Peak temperature: {T_peak} keV")
    print(f"  Peak Er: {Er_peak:.6e} GJ/cm³")
    
    # Boundary conditions
    def left_bc(Er_boundary, coord1_val, coord2_val, geometry):
        """Zero flux at x_min"""
        return 0.0, 1.0, 0.0
    
    def right_bc(Er_boundary, coord1_val, coord2_val, geometry):
        """Zero flux at x_max"""
        return 0.0, 1.0, 0.0
    
    def bottom_bc(Er_boundary, coord1_val, coord2_val, geometry):
        """Zero flux at y_min (reflecting)"""
        return 0.0, 1.0, 0.0
    
    def top_bc(Er_boundary, coord1_val, coord2_val, geometry):
        """Zero flux at y_max (reflecting)"""
        return 0.0, 1.0, 0.0
    
    # Time stepping
    dt = 0.001  # Small time step (sh)
    n_steps = 100  # Total steps
    t_final = dt * n_steps
    
    print(f"\nTime integration:")
    print(f"  Time step: dt = {dt} sh")
    print(f"  Number of steps: {n_steps}")
    print(f"  Final time: t = {t_final} sh")
    
    # Expected spreading (accounting for material coupling)
    D_eff = D / (1.0 + k_coupling)
    sigma_final_sq = sigma0**2 + 2 * D_eff * t_final
    sigma_final = np.sqrt(sigma_final_sq)
    print(f"\nExpected spreading (1D analytical):")
    print(f"  Effective D = D/(1+k) = {D_eff:.6e} cm²/sh")
    print(f"  σ(0) = {sigma0:.6f} cm")
    print(f"  σ({t_final}) = {sigma_final:.6f} cm")
    print(f"  Increase: {(sigma_final/sigma0 - 1)*100:.1f}%")
    
    # Create 2D solver
    print("\n" + "-"*70)
    print("Running 2D numerical simulation...")
    print("-"*70)
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=coord1_min, coord1_max=coord1_max, n1_cells=n1_cells,
        coord2_min=coord2_min, coord2_max=coord2_max, n2_cells=n2_cells,
        geometry='cartesian',
        dt=dt, max_newton_iter=20, newton_tol=1e-6,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        theta=0.5,  # Crank-Nicolson
        left_bc_func=left_bc,
        right_bc_func=right_bc,
        bottom_bc_func=bottom_bc,
        top_bc_func=top_bc,
        use_jfnk=False  # Use direct solver for this test
    )
    
    # Initial condition: Gaussian in specified direction, uniform in other
    if direction == 'x':
        def gaussian_Er_2d(x, y):
            return Er_background + amplitude * np.exp(-(x - pulse_center)**2 / (2 * sigma0**2))
    else:  # direction == 'y'
        def gaussian_Er_2d(x, y):
            return Er_background + amplitude * np.exp(-(y - pulse_center)**2 / (2 * sigma0**2))
    
    solver.set_initial_condition(gaussian_Er_2d)
    
    # Get initial solution
    coord1_coords, coord2_coords, Er_init_2d = solver.get_solution()
    T_init_2d = temperature_from_Er(Er_init_2d)
    
    # Extract cross-section along the direction of variation
    if direction == 'x':
        j_mid = n2_cells // 2
        coord_1d = coord1_coords
        Er_init = Er_init_2d[:, j_mid]
        T_init = T_init_2d[:, j_mid]
        # Check uniformity in y-direction
        Er_variation = np.std(Er_init_2d, axis=1).max() / Er_init.max()
        uniform_dir = 'y'
    else:  # direction == 'y'
        i_mid = n1_cells // 2
        coord_1d = coord2_coords
        Er_init = Er_init_2d[i_mid, :]
        T_init = T_init_2d[i_mid, :]
        # Check uniformity in x-direction
        Er_variation = np.std(Er_init_2d, axis=0).max() / Er_init.max()
        uniform_dir = 'x'
    
    print(f"\nInitial peak Er (numerical 2D): {Er_init.max():.6e} GJ/cm³")
    print(f"Initial peak T (numerical 2D): {T_init.max():.4f} keV")
    
    # Check uniformity in the large direction
    print(f"Initial {uniform_dir}-variation (relative std): {Er_variation:.2e}")
    
    # Advance in time
    print("\nTime stepping:")
    for step in range(n_steps):
        solver.time_step(n_steps=1, verbose=False)
        if step % 20 == 0 or step == n_steps - 1:
            coord1_temp, coord2_temp, Er_temp_2d = solver.get_solution()
            if direction == 'x':
                Er_temp = Er_temp_2d[:, j_mid]
            else:
                Er_temp = Er_temp_2d[i_mid, :]
            T_temp = temperature_from_Er(Er_temp)
            print(f"  Step {step+1:3d}/{n_steps}: max T = {T_temp.max():.4f} keV, "
                  f"max Er = {Er_temp.max():.4e} GJ/cm³")
    
    # Get final solution
    coord1_coords, coord2_coords, Er_final_2d = solver.get_solution()
    T_final_2d = temperature_from_Er(Er_final_2d)
    
    # Extract cross-section
    if direction == 'x':
        Er_final = Er_final_2d[:, j_mid]
        T_final = T_final_2d[:, j_mid]
        # Check final uniformity
        Er_variation_final = np.std(Er_final_2d, axis=1).max() / Er_final.max()
    else:  # direction == 'y'
        Er_final = Er_final_2d[i_mid, :]
        T_final = T_final_2d[i_mid, :]
        # Check final uniformity
        Er_variation_final = np.std(Er_final_2d, axis=0).max() / Er_final.max()
    
    print(f"\nFinal peak Er (numerical 2D): {Er_final.max():.6e} GJ/cm³")
    print(f"Final peak T (numerical 2D): {T_final.max():.4f} keV")
    
    # Check final uniformity
    print(f"Final {uniform_dir}-variation (relative std): {Er_variation_final:.2e}")
    
    # Compute 1D analytical solutions
    Er_analytical_init = Er_background + analytical_gaussian_1d(
        coord_1d, pulse_center, sigma0, D, 0.0, amplitude, k_coupling)
    Er_analytical_final = Er_background + analytical_gaussian_1d(
        coord_1d, pulse_center, sigma0, D, t_final, amplitude, k_coupling)
    
    T_analytical_init = temperature_from_Er(Er_analytical_init)
    T_analytical_final = temperature_from_Er(Er_analytical_final)
    
    print(f"\nFinal peak Er (analytical 1D): {Er_analytical_final.max():.6e} GJ/cm³")
    print(f"Final peak T (analytical 1D): {T_analytical_final.max():.4f} keV")
    
    # Compute errors
    print("\n" + "="*70)
    print("ERROR ANALYSIS (2D vs 1D Analytical)")
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
        print(f"✓ TEST PASSED ({direction.upper()}-direction)")
        print(f"  Relative errors below {tolerance*100}% threshold")
        print(f"  2D solver correctly reduces to 1D analytical solution!")
    else:
        print(f"✗ TEST FAILED ({direction.upper()}-direction)")
        print(f"  Relative errors exceed {tolerance*100}% threshold")
    print("-"*70)
    
    # Plot results
    fig = plt.figure(figsize=(18, 10))
    
    coord_label = direction
    
    # Row 1: 1D cross-sections comparing 2D numerical vs 1D analytical
    # Temperature - Initial
    ax = plt.subplot(2, 3, 1)
    ax.plot(coord_1d, T_init, 'b-', linewidth=2, label='2D Numerical')
    ax.plot(coord_1d, T_analytical_init, 'r--', linewidth=2, label='1D Analytical', alpha=0.7)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Initial Temperature (t = 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Er - Initial
    ax = plt.subplot(2, 3, 2)
    ax.plot(coord_1d, Er_init, 'b-', linewidth=2, label='2D Numerical')
    ax.plot(coord_1d, Er_analytical_init, 'r--', linewidth=2, label='1D Analytical', alpha=0.7)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Radiation Energy Er (GJ/cm³)')
    ax.set_title(f'Initial Er (t = 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error - Initial
    ax = plt.subplot(2, 3, 3)
    ax.plot(coord_1d, Er_init - Er_analytical_init, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Error in Er (GJ/cm³)')
    ax.set_title('Initial Error (discretization)')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Final condition
    # Temperature - Final
    ax = plt.subplot(2, 3, 4)
    ax.plot(coord_1d, T_final, 'b-', linewidth=2, label='2D Numerical')
    ax.plot(coord_1d, T_analytical_final, 'r--', linewidth=2, label='1D Analytical', alpha=0.7)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Final Temperature (t = {t_final:.3f} sh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Er - Final
    ax = plt.subplot(2, 3, 5)
    ax.plot(coord_1d, Er_final, 'b-', linewidth=2, label='2D Numerical')
    ax.plot(coord_1d, Er_analytical_final, 'r--', linewidth=2, label='1D Analytical', alpha=0.7)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Radiation Energy Er (GJ/cm³)')
    ax.set_title(f'Final Er (t = {t_final:.3f} sh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error - Final
    ax = plt.subplot(2, 3, 6)
    ax.plot(coord_1d, Er_error, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'Position {coord_label} (cm)')
    ax.set_ylabel('Error in Er (GJ/cm³)')
    ax.set_title(f'Final Error (L2 rel: {relative_L2_Er*100:.3f}%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_2d_linear_gaussian_{direction}.png', dpi=150, bbox_inches='tight')
    print(f"\n1D cross-section plot saved as 'test_2d_linear_gaussian_{direction}.png'")
    
    # Additional 2D visualization showing uniformity in the other direction
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    X, Y = np.meshgrid(coord1_coords, coord2_coords, indexing='ij')
    
    # Initial 2D solution
    ax = axes[0]
    im = ax.pcolormesh(X, Y, Er_init_2d, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='$E_r$ (GJ/cm³)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Initial Er (2D view)')
    ax.set_aspect('equal')
    
    # Final 2D solution
    ax = axes[1]
    im = ax.pcolormesh(X, Y, Er_final_2d, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='$E_r$ (GJ/cm³)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'Final Er at t={t_final:.3f} sh (2D view)')
    ax.set_aspect('equal')
    
    # Uniformity check (should be near zero)
    ax = axes[2]
    if direction == 'x':
        variation = np.std(Er_final_2d, axis=1)  # Standard deviation in y at each x
        ax.plot(coord1_coords, variation / Er_final.max(), 'b-', linewidth=2)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('Relative std in y-direction')
        ax.set_title('Y-uniformity check (should be ~0)')
    else:  # direction == 'y'
        variation = np.std(Er_final_2d, axis=0)  # Standard deviation in x at each y
        ax.plot(coord2_coords, variation / Er_final.max(), 'b-', linewidth=2)
        ax.set_xlabel('y (cm)')
        ax.set_ylabel('Relative std in x-direction')
        ax.set_title('X-uniformity check (should be ~0)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'test_2d_linear_gaussian_full_{direction}.png', dpi=150, bbox_inches='tight')
    print(f"Full 2D visualization saved as 'test_2d_linear_gaussian_full_{direction}.png'")
    
    plt.show()
    
    return solver, relative_L2_Er, relative_L2_T


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing 2D solver in BOTH directions")
    print("="*70)
    
    # Test x-direction (Gaussian in x, y large)
    print("\n")
    solver_x, rel_err_Er_x, rel_err_T_x = test_2d_linear_gaussian(direction='x')
    
    # Test y-direction (Gaussian in y, x large)
    print("\n\n")
    solver_y, rel_err_Er_y, rel_err_T_y = test_2d_linear_gaussian(direction='y')
    
    # Summary
    print("\n" + "="*70)
    print("2D linear Gaussian diffusion test completed for BOTH directions!")
    print("="*70)
    print(f"\nX-direction test:")
    print(f"  Er:  {rel_err_Er_x*100:.4f}% (relative L2)")
    print(f"  T:   {rel_err_T_x*100:.4f}% (relative L2)")
    print(f"  Status: {'✓ PASSED' if rel_err_Er_x < 0.02 and rel_err_T_x < 0.02 else '✗ FAILED'}")
    
    print(f"\nY-direction test:")
    print(f"  Er:  {rel_err_Er_y*100:.4f}% (relative L2)")
    print(f"  T:   {rel_err_T_y*100:.4f}% (relative L2)")
    print(f"  Status: {'✓ PASSED' if rel_err_Er_y < 0.02 and rel_err_T_y < 0.02 else '✗ FAILED'}")
    
    print(f"\nOverall: {'✓ BOTH DIRECTIONS PASS' if all([rel_err_Er_x < 0.02, rel_err_T_x < 0.02, rel_err_Er_y < 0.02, rel_err_T_y < 0.02]) else '✗ AT LEAST ONE DIRECTION FAILED'}")
    print("="*70)
