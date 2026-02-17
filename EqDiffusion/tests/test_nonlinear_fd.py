#!/usr/bin/env python3
"""
Test nonlinear correction accuracy using finite differences

Verify that the linearization L^(k)[φ] + N^(k)[φ] correctly approximates
the full nonlinear operator ∇·(D(E_r^(k) + φ) ∇(E_r^(k) + φ))
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


def test_nonlinear_linearization():
    """
    Test that the linearized operator matches finite difference approximation
    
    For a perturbation φ, we should have:
    ∇·(D(E_r + ε*φ) ∇(E_r + ε*φ)) ≈ ∇·(D(E_r) ∇E_r) + ε * [L^(k)[φ] + N^(k)[φ]]
    
    where L^(k) is the linear diffusion part and N^(k) is the nonlinear correction
    """
    print("="*70)
    print("Finite Difference Test of Nonlinear Correction")
    print("="*70)
    
    # Simple material with D ~ T
    def linear_D_opacity(Er):
        T = temperature_from_Er(Er)
        T = max(T, 0.01)
        return 100.0 / T
    
    def simple_cv(T):
        return 0.1 / 1.0
    
    def simple_energy(T):
        return 0.1 * T
    
    # Zero flux boundaries
    def zero_flux_bc(Er, x):
        return 0.0, 1.0, 0.0
    
    # Small problem for clarity
    n_cells = 10
    r_min = 0.0
    r_max = 1.0
    dt = 0.01
    
    print(f"\nSetup:")
    print(f"  Cells: {n_cells}")
    print(f"  Domain: [{r_min}, {r_max}]")
    print(f"  D ~ T (linear)")
    
    # Create solver
    solver = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=1, newton_tol=1e-10,
        rosseland_opacity_func=linear_D_opacity,
        specific_heat_func=simple_cv,
        material_energy_func=simple_energy,
        left_bc_func=zero_flux_bc,
        right_bc_func=zero_flux_bc
    )
    
    # Base state: Gaussian profile
    def gaussian_Er(r):
        r_center = 0.5
        width = 0.2
        T_peak = 1.0
        T_base = 0.3
        T = T_base + (T_peak - T_base) * np.exp(-((r - r_center) / width)**2)
        return A_RAD * T**4
    
    solver.set_initial_condition(gaussian_Er)
    Er_base = solver.Er.copy()
    
    # Perturbation: small sinusoidal
    r = solver.r_centers
    phi = 0.01 * A_RAD * np.sin(2 * np.pi * r)
    
    print(f"\nBase state max T: {temperature_from_Er(Er_base).max():.3f} keV")
    print(f"Perturbation amplitude: {phi.max():.4e}")
    
    # Test for different epsilon values
    epsilons = [0.1, 0.01, 0.001, 0.0001]
    
    print("\n" + "-"*70)
    print("Testing linearization for different ε values:")
    print("-"*70)
    
    for eps in epsilons:
        # Perturbed state
        Er_pert = Er_base + eps * phi
        
        # We need to isolate just the SPATIAL diffusion operator
        # The full system includes time-stepping terms which confuse the test
        
        # Manually compute the spatial diffusion operator -∇·(D ∇E_r)
        # For base state
        flux_base = np.zeros(n_cells + 1)
        for i in range(1, n_cells):  # Interior faces
            dx = solver.r_centers[i] - solver.r_centers[i-1]
            D_face = solver.get_diffusion_coefficient(0.5 * (Er_base[i-1] + Er_base[i]))
            flux_base[i] = -D_face * (Er_base[i] - Er_base[i-1]) / dx
        
        # Divergence for each cell
        div_base = np.zeros(n_cells)
        for i in range(n_cells):
            dx_left = solver.r_centers[i] - solver.r_centers[i-1] if i > 0 else solver.r_centers[i] - solver.r_faces[i]
            dx_right = solver.r_centers[i+1] - solver.r_centers[i] if i < n_cells-1 else solver.r_faces[i+1] - solver.r_centers[i]
            dx_total = dx_left + dx_right
            div_base[i] = (flux_base[i+1] - flux_base[i]) / dx_total
        
        # For perturbed state
        flux_pert = np.zeros(n_cells + 1)
        for i in range(1, n_cells):
            dx = solver.r_centers[i] - solver.r_centers[i-1]
            D_face = solver.get_diffusion_coefficient(0.5 * (Er_pert[i-1] + Er_pert[i]))
            flux_pert[i] = -D_face * (Er_pert[i] - Er_pert[i-1]) / dx
        
        div_pert = np.zeros(n_cells)
        for i in range(n_cells):
            dx_left = solver.r_centers[i] - solver.r_centers[i-1] if i > 0 else solver.r_centers[i] - solver.r_faces[i]
            dx_right = solver.r_centers[i+1] - solver.r_centers[i] if i < n_cells-1 else solver.r_faces[i+1] - solver.r_centers[i]
            dx_total = dx_left + dx_right
            div_pert[i] = (flux_pert[i+1] - flux_pert[i]) / dx_total
        
        # Finite difference derivative
        dF_fd = (div_pert - div_base) / eps
        
        # Now compute linearized operators acting on phi
        # Linear part: -∇·(D(Er_base) ∇phi)
        flux_linear = np.zeros(n_cells + 1)
        for i in range(1, n_cells):
            dx = solver.r_centers[i] - solver.r_centers[i-1]
            D_face = solver.get_diffusion_coefficient(0.5 * (Er_base[i-1] + Er_base[i]))
            flux_linear[i] = -D_face * (phi[i] - phi[i-1]) / dx
        
        div_linear = np.zeros(n_cells)
        for i in range(n_cells):
            dx_left = solver.r_centers[i] - solver.r_centers[i-1] if i > 0 else solver.r_centers[i] - solver.r_faces[i]
            dx_right = solver.r_centers[i+1] - solver.r_centers[i] if i < n_cells-1 else solver.r_faces[i+1] - solver.r_centers[i]
            dx_total = dx_left + dx_right
            div_linear[i] = (flux_linear[i+1] - flux_linear[i]) / dx_total
        
        # Nonlinear correction: -∇·(D_E(Er_base) * phi * ∇Er_base)
        flux_nl = np.zeros(n_cells + 1)
        for i in range(1, n_cells):
            dx = solver.r_centers[i] - solver.r_centers[i-1]
            Er_face = 0.5 * (Er_base[i-1] + Er_base[i])
            phi_face = 0.5 * (phi[i-1] + phi[i])
            D_E_face = solver.get_diffusion_coefficient_derivative(Er_face)
            grad_Er_base = (Er_base[i] - Er_base[i-1]) / dx
            flux_nl[i] = -D_E_face * phi_face * grad_Er_base
        
        div_nl = np.zeros(n_cells)
        for i in range(n_cells):
            dx_left = solver.r_centers[i] - solver.r_centers[i-1] if i > 0 else solver.r_centers[i] - solver.r_faces[i]
            dx_right = solver.r_centers[i+1] - solver.r_centers[i] if i < n_cells-1 else solver.r_faces[i+1] - solver.r_centers[i]
            dx_total = dx_left + dx_right
            div_nl[i] = (flux_nl[i+1] - flux_nl[i]) / dx_total
        
        # Total linearized operator
        L_phi = div_linear
        LN_phi = div_linear + div_nl
        
        # Compare
        error_without_nl = np.linalg.norm(dF_fd - L_phi) / (np.linalg.norm(dF_fd) + 1e-14)
        error_with_nl = np.linalg.norm(dF_fd - LN_phi) / (np.linalg.norm(dF_fd) + 1e-14)
        
        print(f"\nε = {eps:.4f}:")
        print(f"  ||dF/dE||:                   {np.linalg.norm(dF_fd):.6e}")
        print(f"  Error without NL correction: {error_without_nl:.6e}")
        print(f"  Error with NL correction:    {error_with_nl:.6e}")
        if error_with_nl > 0:
            print(f"  Improvement ratio:           {error_without_nl / error_with_nl:.2f}x")
        
        if eps == 0.01:  # Save detailed info for one case
            # Look at specific cells
            print(f"\n  Detailed comparison at ε = {eps}:")
            print(f"  Cell  |  FD derivative  |  Linear only  |  With NL  |")
            print(f"  " + "-"*60)
            cells_to_check = [i for i in [0, n_cells//4, n_cells//2, 3*n_cells//4, n_cells-1] if i < len(dF_fd)]
            for i in cells_to_check:
                print(f"  {i:3d}   |  {dF_fd[i]:13.6e} | {L_phi[i]:13.6e} | {LN_phi[i]:13.6e} |")
    
    # Visualization
    print("\n" + "-"*70)
    print("Creating visualization...")
    print("-"*70)
    
    eps = 0.01
    Er_pert = Er_base + eps * phi
    
    # Recompute for visualization using manual operators
    # (already computed in the loop above, but let's be explicit)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Base state
    ax = axes[0, 0]
    T_base = temperature_from_Er(Er_base)
    ax.plot(r, T_base, 'b-', linewidth=2)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title('Base State')
    ax.grid(True, alpha=0.3)
    
    # Perturbation
    ax = axes[0, 1]
    ax.plot(r, phi, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Perturbation φ')
    ax.set_title(f'Perturbation (ε = {eps})')
    ax.grid(True, alpha=0.3)
    
    # Comparison of derivatives
    ax = axes[1, 0]
    ax.plot(r, dF_fd, 'ko-', label='FD derivative', markersize=6)
    ax.plot(r, L_phi, 'b^--', label='Linear only', markersize=5)
    ax.plot(r, LN_phi, 'rs-', label='With NL correction', markersize=5, alpha=0.7)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Operator action on φ')
    ax.set_title('Linearization Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error comparison
    ax = axes[1, 1]
    error_linear = np.abs(dF_fd - L_phi)
    error_nl = np.abs(dF_fd - LN_phi)
    ax.semilogy(r, error_linear, 'b^--', label='Error (linear only)', markersize=5)
    ax.semilogy(r, error_nl, 'rs-', label='Error (with NL)', markersize=5, alpha=0.7)
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Pointwise Linearization Error')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('nonlinear_correction_verification.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'nonlinear_correction_verification.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    print("\nThe nonlinear correction should reduce the error as ε → 0.")
    print("If the error WITH NL is smaller than WITHOUT NL, the correction is working.")
    print("If the errors are similar or NL is worse, there may be a sign or formula error.")
    
    return solver


if __name__ == "__main__":
    solver = test_nonlinear_linearization()
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)
