#!/usr/bin/env python3
"""
Test nonlinear correction implementation for Marshak wave

Verifies:
1. Finite difference approximation of D_E = dD/dEr is accurate
2. Nonlinear correction terms have reasonable magnitudes
3. Compare analytical vs numerical derivatives
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


# Marshak wave properties
def marshak_opacity(Er):
    """σ_R = 300 * T^-3"""
    T = temperature_from_Er(Er)
    T_min = 0.01
    if T < T_min:
        T = T_min
    return 300.0 * T**(-3)


def marshak_specific_heat(T):
    """c_v = 0.3 GJ/(cm^3·keV)"""
    cv_volumetric = 0.3
    return cv_volumetric / RHO


def marshak_material_energy(T):
    """e_m = c_v * T"""
    return 0.3 * T


def marshak_left_bc(Er, x):
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc


def marshak_right_bc(Er, x):
    return 0.0, 1.0, 0.0


# Analytical derivative for D(Er) with Marshak opacity
def analytical_D_derivative(Er):
    """
    For σ_R = 300 * T^-3 and D = c/(3*σ_R):
    
    D = c/(900 * T^-3) = (c/900) * T^3
    
    With T = (Er/a)^(1/4):
    D = (c/900) * (Er/a)^(3/4)
    
    dD/dEr = (c/900) * (3/4) * (1/a)^(3/4) * Er^(-1/4)
            = (c/1200) * a^(-3/4) * Er^(-1/4)
    """
    a = A_RAD
    T = temperature_from_Er(Er)
    
    # Protect against very low T
    T_min = 0.01
    if T < T_min:
        T = T_min
        Er = a * T_min**4
    
    D_E_analytical = (C_LIGHT / 1200.0) * (a**(-0.75)) * (Er**(-0.25))
    return D_E_analytical


def test_derivative_accuracy():
    """Compare analytical vs finite difference derivatives"""
    
    print("="*70)
    print("TESTING D_E = dD/dEr ACCURACY")
    print("="*70)
    
    # Test over range of Er values
    T_values = np.logspace(-2, 0, 50)  # 0.01 to 1 keV
    Er_values = A_RAD * T_values**4
    
    D_values = []
    D_E_analytical = []
    D_E_numerical = []
    relative_errors = []
    
    # Create a dummy solver just to use its derivative function
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=0.1, n_cells=10, d=0, dt=0.01,
        max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    for Er in Er_values:
        T = temperature_from_Er(Er)
        
        # Diffusion coefficient
        D = solver.get_diffusion_coefficient(Er)
        D_values.append(D)
        
        # Analytical derivative
        D_E_anal = analytical_D_derivative(Er)
        D_E_analytical.append(D_E_anal)
        
        # Numerical derivative (as computed by solver)
        D_E_num = solver.get_diffusion_coefficient_derivative(Er)
        D_E_numerical.append(D_E_num)
        
        # Relative error
        rel_err = abs(D_E_num - D_E_anal) / abs(D_E_anal) if D_E_anal != 0 else 0
        relative_errors.append(rel_err)
    
    D_values = np.array(D_values)
    D_E_analytical = np.array(D_E_analytical)
    D_E_numerical = np.array(D_E_numerical)
    relative_errors = np.array(relative_errors)
    
    # Print statistics
    print(f"\nDerivative accuracy over T ∈ [0.01, 1.0] keV:")
    print(f"  Max relative error: {relative_errors.max():.2e}")
    print(f"  Mean relative error: {relative_errors.mean():.2e}")
    print(f"  Median relative error: {np.median(relative_errors):.2e}")
    
    # Find worst cases
    worst_idx = np.argmax(relative_errors)
    print(f"\nWorst case:")
    print(f"  T = {T_values[worst_idx]:.4f} keV")
    print(f"  Er = {Er_values[worst_idx]:.4e} GJ/cm³")
    print(f"  D = {D_values[worst_idx]:.4e} cm")
    print(f"  D_E (analytical) = {D_E_analytical[worst_idx]:.4e}")
    print(f"  D_E (numerical) = {D_E_numerical[worst_idx]:.4e}")
    print(f"  Relative error = {relative_errors[worst_idx]:.2e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: D vs T
    ax = axes[0, 0]
    ax.loglog(T_values, D_values, 'b-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient D (cm)', fontsize=12)
    ax.set_title('Diffusion Coefficient D(T)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: D_E comparison
    ax = axes[0, 1]
    ax.loglog(T_values, np.abs(D_E_analytical), 'r-', linewidth=2, label='Analytical')
    ax.loglog(T_values, np.abs(D_E_numerical), 'b--', linewidth=2, label='Numerical (FD)')
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('|dD/dEr|', fontsize=12)
    ax.set_title('Derivative dD/dEr: Analytical vs Numerical', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Relative error
    ax = axes[1, 0]
    ax.semilogy(T_values, relative_errors, 'g-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Relative Error in dD/dEr', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.01, color='r', linestyle='--', alpha=0.5, label='1% error')
    ax.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='10% error')
    ax.legend(fontsize=10)
    
    # Plot 4: D_E * Er (characteristic nonlinear term)
    ax = axes[1, 1]
    nl_term_anal = np.abs(D_E_analytical * Er_values)
    nl_term_num = np.abs(D_E_numerical * Er_values)
    ax.loglog(T_values, nl_term_anal, 'r-', linewidth=2, label='Analytical')
    ax.loglog(T_values, D_values, 'k--', linewidth=1, alpha=0.5, label='D (for reference)')
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('|D_E × Er| (cm)', fontsize=12)
    ax.set_title('Nonlinear Term Magnitude: |D_E × Er|', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('test_nonlinear_derivatives.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved as 'test_nonlinear_derivatives.png'")
    plt.show()
    
    return relative_errors.max()


def test_nonlinear_correction_with_limiter():
    """Test nonlinear correction with different limiter values"""
    
    print("\n" + "="*70)
    print("TESTING NONLINEAR CORRECTION WITH LIMITER")
    print("="*70)
    
    # Test different limiter values
    limiter_values = [None, 1.0, 0.5, 0.1]  # None = no NL correction
    n_cells = 100
    dt_value = 0.01  # ns
    t_final = 1.0  # ns
    
    solutions = {}
    
    for limiter in limiter_values:
        if limiter is None:
            label = 'No NL correction'
            use_nl = False
            lim_val = 1.0
        else:
            label = f'Limiter = {limiter}'
            use_nl = True
            lim_val = limiter
        
        print(f"\n{label}:")
        print(f"  Running...", end='', flush=True)
        
        try:
            solver = RadiationDiffusionSolver(
                r_min=0.0,
                r_max=0.2,
                n_cells=n_cells,
                d=0,
                dt=dt_value,
                max_newton_iter=50,
                newton_tol=1e-8,
                rosseland_opacity_func=marshak_opacity,
                specific_heat_func=marshak_specific_heat,
                material_energy_func=marshak_material_energy,
                left_bc_func=marshak_left_bc,
                right_bc_func=marshak_right_bc
            )
            
            solver.use_nonlinear_correction = use_nl
            solver.nonlinear_limiter = lim_val
            solver.nonlinear_skip_boundary_cells = 2 if use_nl else 0
            
            # Initial condition
            T_init = 0.1
            solver.set_initial_condition(lambda r: np.full_like(r, A_RAD * T_init**4))
            
            # Evolve
            n_steps = int(t_final / dt_value)
            for _ in range(n_steps):
                solver.time_step(n_steps=1, verbose=False)
            
            r, Er = solver.get_solution()
            T = temperature_from_Er(Er)
            
            # Check for oscillations and negatives
            T_diff = np.diff(T)
            sign_changes = np.sum(np.diff(np.sign(T_diff)) != 0)
            min_Er = Er.min()
            max_T = T.max()
            
            solutions[label] = {'r': r, 'Er': Er, 'T': T}
            
            print(f" SUCCESS")
            print(f"    Sign changes in T: {sign_changes}")
            print(f"    Min Er: {min_Er:.4e}, Max T: {max_T:.4f} keV")
            
        except Exception as e:
            print(f" FAILED: {e}")
            solutions[label] = None
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['black', 'blue', 'green', 'red']
    linestyles = ['-', '-', '--', ':']
    
    for i, (label, sol) in enumerate(solutions.items()):
        if sol is not None:
            ax1.plot(sol['r'], sol['T'], color=colors[i], linestyle=linestyles[i],
                    linewidth=2, label=label)
            ax2.plot(sol['r'], sol['Er'], color=colors[i], linestyle=linestyles[i],
                    linewidth=2, label=label)
    
    ax1.set_xlabel('Position r (cm)', fontsize=12)
    ax1.set_ylabel('Temperature T (keV)', fontsize=12)
    ax1.set_title('Temperature Profiles with Different Limiters', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    ax2.set_xlabel('Position r (cm)', fontsize=12)
    ax2.set_ylabel('Radiation Energy Er (GJ/cm³)', fontsize=12)
    ax2.set_title('Radiation Energy Profiles with Different Limiters', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.015])
    
    plt.tight_layout()
    plt.savefig('test_nonlinear_limiter.png', dpi=150, bbox_inches='tight')
    print(f"\nLimiter comparison saved as 'test_nonlinear_limiter.png'")
    plt.show()
    """Test nonlinear correction by comparing solutions with different mesh/time refinements"""
    
    print("\n" + "="*70)
    print("TESTING NONLINEAR CORRECTION WITH GRID REFINEMENT")
    print("="*70)
    
    # Run with different resolutions
    n_cells_values = [50, 100, 200]
    dt_value = 0.01  # ns
    t_final = 1.0  # ns
    
    solutions = {}
    
    for use_nl in [False, True]:
        nl_str = 'with_NL' if use_nl else 'no_NL'
        solutions[nl_str] = []
        
        print(f"\nRunning {'with' if use_nl else 'without'} nonlinear correction:")
        
        for n_cells in n_cells_values:
            print(f"  n_cells = {n_cells}...", end='', flush=True)
            
            solver = RadiationDiffusionSolver(
                r_min=0.0,
                r_max=0.2,
                n_cells=n_cells,
                d=0,
                dt=dt_value,
                max_newton_iter=50,
                newton_tol=1e-8,
                rosseland_opacity_func=marshak_opacity,
                specific_heat_func=marshak_specific_heat,
                material_energy_func=marshak_material_energy,
                left_bc_func=marshak_left_bc,
                right_bc_func=marshak_right_bc
            )
            
            solver.use_nonlinear_correction = use_nl
            solver.nonlinear_skip_boundary_cells = 2 if use_nl else 0
            
            # Initial condition
            T_init = 0.1
            solver.set_initial_condition(lambda r: np.full_like(r, A_RAD * T_init**4))
            
            # Evolve
            n_steps = int(t_final / dt_value)
            for _ in range(n_steps):
                solver.time_step(n_steps=1, verbose=False)
            
            r, Er = solver.get_solution()
            T = temperature_from_Er(Er)
            
            solutions[nl_str].append({'r': r, 'Er': Er, 'T': T, 'n_cells': n_cells})
            
            # Check for oscillations
            T_diff = np.diff(T)
            sign_changes = np.sum(np.diff(np.sign(T_diff)) != 0)
            print(f" done. Sign changes in T: {sign_changes}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '--']
    
    for idx, nl_str in enumerate(['no_NL', 'with_NL']):
        ax = axes[idx]
        
        for i, sol in enumerate(solutions[nl_str]):
            label = f"n={sol['n_cells']}"
            ax.plot(sol['r'], sol['T'], color=colors[i], linewidth=2, label=label)
        
        ax.set_xlabel('Position r (cm)', fontsize=12)
        ax.set_ylabel('Temperature T (keV)', fontsize=12)
        title = 'Without Nonlinear Correction' if nl_str == 'no_NL' else 'With Nonlinear Correction'
        ax.set_title(f'{title}\nGrid Refinement Test', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('test_nonlinear_refinement.png', dpi=150, bbox_inches='tight')
    print(f"\nRefinement comparison saved as 'test_nonlinear_refinement.png'")
    plt.show()


if __name__ == "__main__":
    # Test 1: Derivative accuracy
    max_error = test_derivative_accuracy()
    
    # Test 2: Nonlinear correction with different limiters
    test_nonlinear_correction_with_limiter()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Maximum relative error in dD/dEr: {max_error:.2e}")
    
    if np.isnan(max_error):
        print("⚠ Derivative calculation failed for very low Er values")
        print("  This indicates finite difference breaks down at low temperatures")
    elif max_error < 0.01:
        print("✓ Derivative approximation is EXCELLENT (< 1% error)")
    elif max_error < 0.1:
        print("✓ Derivative approximation is GOOD (< 10% error)")
    else:
        print("⚠ Derivative approximation has significant error (> 10%)")
    
    print("\nThe limiter prevents nonlinear correction from dominating the linear term.")
    print("Smaller limiter values (e.g., 0.1) make the method more stable but less accurate.")
    print("Limiter = 1.0 means NL correction ≤ linear term (recommended starting point).")
    print("="*70)
