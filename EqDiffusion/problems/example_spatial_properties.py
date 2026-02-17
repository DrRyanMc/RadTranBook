#!/usr/bin/env python3
"""
Example: Spatially-Dependent Material Properties in 2D

Demonstrates how to use spatially-varying opacity, specific heat, and 
material energy density in the 2D radiation diffusion solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from twoDFV import RadiationDiffusionSolver2D, temperature_from_Er, A_RAD
from numba import njit


# =============================================================================
# EXAMPLE 1: Spatially-varying opacity (region with different opacities)
# =============================================================================

@njit
def opacity_two_region(Er, x, y):
    """
    Opacity that varies by region:
    - High opacity (σ = 1.0) in left half (x < 0.5)
    - Low opacity (σ = 0.1) in right half (x >= 0.5)
    """
    if x < 0.5:
        return 1.0  # High opacity region
    else:
        return 0.1  # Low opacity region


@njit
def opacity_radial_gradient(Er, x, y):
    """
    Opacity that decreases with distance from center:
    σ(r) = σ_min + (σ_max - σ_min) * exp(-r²/r₀²)
    """
    x0, y0 = 0.5, 0.5
    r_squared = (x - x0)**2 + (y - y0)**2
    r0_squared = 0.2**2
    
    sigma_min = 0.1
    sigma_max = 2.0
    
    return sigma_min + (sigma_max - sigma_min) * np.exp(-r_squared / r0_squared)


# =============================================================================
# EXAMPLE 2: Spatially-varying specific heat
# =============================================================================

@njit
def specific_heat_layered(T, x, y):
    """
    Specific heat that varies in horizontal layers:
    - Top layer (y > 0.7): c_v = 2.0
    - Middle layer (0.3 < y < 0.7): c_v = 1.0
    - Bottom layer (y < 0.3): c_v = 0.5
    """
    if y > 0.7:
        return 2.0
    elif y > 0.3:
        return 1.0
    else:
        return 0.5


@njit
def specific_heat_gradient(T, x, y):
    """
    Specific heat that varies linearly across domain:
    c_v(x) = 0.5 + 1.5*x
    """
    return 0.5 + 1.5 * x


# =============================================================================
# EXAMPLE 3: Combined spatially-varying properties
# =============================================================================

@njit
def material_energy_with_spatial_cv(T, x, y):
    """
    Material energy density using spatially-varying specific heat
    e(T, x, y) = ρ * c_v(T, x, y) * T
    """
    rho = 1.0
    cv = specific_heat_gradient(T, x, y)
    return rho * cv * T


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_two_region_opacity():
    """Test with two regions of different opacity"""
    print("="*60)
    print("Example 1: Two-Region Opacity")
    print("="*60)
    print("Left half (x < 0.5): σ = 1.0 (high opacity, slow diffusion)")
    print("Right half (x >= 0.5): σ = 0.1 (low opacity, fast diffusion)")
    print()
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=40,
        coord2_min=0.0, coord2_max=1.0, n2_cells=40,
        geometry='cartesian', dt=5e-4, max_newton_iter=10,
        rosseland_opacity_func=opacity_two_region
    )
    
    # Hot spot at center-left
    def initial_Er(x, y):
        x0, y0 = 0.3, 0.5
        sigma = 0.1
        return 0.1 + 2.0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    print("Time stepping (10 steps)...")
    solver.time_step(n_steps=10, verbose=False)
    
    # Plot result
    coord1, coord2, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    X, Y = np.meshgrid(coord1, coord2, indexing='ij')
    
    # Plot temperature
    im1 = ax1.pcolormesh(X, Y, T_2d, shading='auto', cmap='hot')
    ax1.axvline(x=0.5, color='cyan', linestyle='--', linewidth=2, label='Interface')
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.set_title('Temperature (keV)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='T (keV)')
    
    # Plot opacity field
    opacity_field = np.zeros_like(Er_2d)
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            opacity_field[i, j] = opacity_two_region(Er_2d[i, j], 
                                                      coord1[i], coord2[j])
    
    im2 = ax2.pcolormesh(X, Y, opacity_field, shading='auto', cmap='viridis')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Interface')
    ax2.set_xlabel('x (cm)', fontsize=12)
    ax2.set_ylabel('y (cm)', fontsize=12)
    ax2.set_title('Opacity σ_R (cm⁻¹)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='σ_R (cm⁻¹)')
    
    plt.suptitle('Two-Region Opacity Problem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('two_region_opacity.png', dpi=150, bbox_inches='tight')
    print("Saved: two_region_opacity.png")
    plt.show()
    
    return solver


def test_radial_opacity_gradient():
    """Test with radially-varying opacity"""
    print("\n" + "="*60)
    print("Example 2: Radial Opacity Gradient")
    print("="*60)
    print("Opacity decreases from center: σ(r) = 0.1 + 1.9*exp(-r²/r₀²)")
    print("High opacity at center slows diffusion there")
    print()
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=40,
        coord2_min=0.0, coord2_max=1.0, n2_cells=40,
        geometry='cartesian', dt=5e-4, max_newton_iter=10,
        rosseland_opacity_func=opacity_radial_gradient
    )
    
    # Hot spot at center
    def initial_Er(x, y):
        x0, y0 = 0.5, 0.5
        sigma = 0.08
        return 0.1 + 3.0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    solver.set_initial_condition(initial_Er)
    
    print("Time stepping (15 steps)...")
    solver.time_step(n_steps=15, verbose=False)
    
    # Plot result
    coord1, coord2, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    X, Y = np.meshgrid(coord1, coord2, indexing='ij')
    
    # Plot temperature
    im1 = ax1.pcolormesh(X, Y, T_2d, shading='auto', cmap='hot')
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.set_title('Temperature (keV)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='T (keV)')
    
    # Plot opacity field
    opacity_field = np.zeros_like(Er_2d)
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            opacity_field[i, j] = opacity_radial_gradient(Er_2d[i, j], 
                                                           coord1[i], coord2[j])
    
    im2 = ax2.pcolormesh(X, Y, opacity_field, shading='auto', cmap='viridis')
    ax2.set_xlabel('x (cm)', fontsize=12)
    ax2.set_ylabel('y (cm)', fontsize=12)
    ax2.set_title('Opacity σ_R (cm⁻¹)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='σ_R (cm⁻¹)')
    
    plt.suptitle('Radial Opacity Gradient Problem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('radial_opacity_gradient.png', dpi=150, bbox_inches='tight')
    print("Saved: radial_opacity_gradient.png")
    plt.show()
    
    return solver


def test_layered_specific_heat():
    """Test with layered specific heat"""
    print("\n" + "="*60)
    print("Example 3: Layered Specific Heat")
    print("="*60)
    print("Specific heat varies in horizontal layers:")
    print("  Top (y > 0.7): c_v = 2.0")
    print("  Middle (0.3 < y < 0.7): c_v = 1.0")
    print("  Bottom (y < 0.3): c_v = 0.5")
    print()
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=40,
        coord2_min=0.0, coord2_max=1.0, n2_cells=40,
        geometry='cartesian', dt=5e-4, max_newton_iter=10,
        specific_heat_func=specific_heat_layered,
        material_energy_func=material_energy_with_spatial_cv
    )
    
    # Hot spot at center
    def initial_Er(x, y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return 0.1 + 2.0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    solver.set_initial_condition(initial_Er)
    
    print("Time stepping (10 steps)...")
    solver.time_step(n_steps=10, verbose=False)
    
    # Plot result
    coord1, coord2, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    X, Y = np.meshgrid(coord1, coord2, indexing='ij')
    
    # Plot temperature
    im1 = ax1.pcolormesh(X, Y, T_2d, shading='auto', cmap='hot')
    ax1.axhline(y=0.3, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0.7, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('x (cm)', fontsize=12)
    ax1.set_ylabel('y (cm)', fontsize=12)
    ax1.set_title('Temperature (keV)', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='T (keV)')
    
    # Plot specific heat field
    cv_field = np.zeros_like(Er_2d)
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            T = temperature_from_Er(Er_2d[i, j])
            cv_field[i, j] = specific_heat_layered(T, coord1[i], coord2[j])
    
    im2 = ax2.pcolormesh(X, Y, cv_field, shading='auto', cmap='viridis')
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('x (cm)', fontsize=12)
    ax2.set_ylabel('y (cm)', fontsize=12)
    ax2.set_title('Specific Heat c_v', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='c_v (GJ/(g·keV))')
    
    plt.suptitle('Layered Specific Heat Problem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('layered_specific_heat.png', dpi=150, bbox_inches='tight')
    print("Saved: layered_specific_heat.png")
    plt.show()
    
    return solver


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all examples"""
    print("="*70)
    print("SPATIALLY-DEPENDENT MATERIAL PROPERTIES EXAMPLES")
    print("="*70)
    print()
    print("This demonstrates how to use spatially-varying:")
    print("  1. Opacity σ_R(Er, x, y)")
    print("  2. Specific heat c_v(T, x, y)")
    print("  3. Material energy density e(T, x, y)")
    print()
    
    # Run examples
    solver1 = test_two_region_opacity()
    solver2 = test_radial_opacity_gradient()
    solver3 = test_layered_specific_heat()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nKey points:")
    print("  • Material property functions now accept (value, coord1, coord2)")
    print("  • Use @njit decorator from numba for performance")
    print("  • Pass custom functions to solver via:")
    print("    - rosseland_opacity_func=your_opacity_function")
    print("    - specific_heat_func=your_heat_function")
    print("    - material_energy_func=your_energy_function")
    print()


if __name__ == "__main__":
    main()
