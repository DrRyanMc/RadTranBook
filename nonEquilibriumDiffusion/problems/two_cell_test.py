#!/usr/bin/env python3
"""
Two-Cell Diffusion Test
Simple test with one hot cell and one cold cell to verify diffusion behavior
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    NonEquilibriumRadiationDiffusionSolver,
    A_RAD, 
    C_LIGHT,
    RHO
)

# Material properties (same as Zeldovich wave)
def test_rosseland_opacity(T):
    """σ_R = 300 * T^-3"""
    n = 3
    T_min = 0.001
    T_safe = max(T, T_min)
    return 300.0 * T_safe**(-n)

def test_planck_opacity(T):
    """σ_P = 300 * T^-3"""
    return test_rosseland_opacity(T)

def test_specific_heat(T):
    """c_v = 3e-6 GJ/(cm^3·keV)"""
    cv_volumetric = 3e-6
    return cv_volumetric / RHO

def test_material_energy(T):
    """e = ρ·c_v·T"""
    cv_volumetric = 3e-6
    return RHO * cv_volumetric * T

def test_inverse_material_energy(e):
    """T from e"""
    cv_volumetric = 3e-6
    return e / (RHO * cv_volumetric)

# Reflecting BCs
def test_left_bc(phi, x):
    return 0.0, 1.0, 0.0

def test_right_bc(phi, x):
    return 0.0, 1.0, 0.0

def run_two_cell_test():
    """Run two-cell test with one hot, one cold"""
    
    print("="*80)
    print("TWO-CELL DIFFUSION TEST")
    print("="*80)
    print("Setup: Cell 0 hot (T=5 keV), Cell 1 cold (T=0.1 keV)")
    print("Expected: Heat should diffuse from cell 0 to cell 1")
    print("="*80)
    
    # Two cells
    r_min = 0.0
    r_max = 0.2  # 2 cells of 0.1 cm each
    n_cells = 2
    dt = 1e-8  # Extremely small timestep
    
    # Create solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        d=0,  # planar
        dt=dt,
        max_newton_iter=20,
        newton_tol=1e-8,
        rosseland_opacity_func=test_rosseland_opacity,
        planck_opacity_func=test_planck_opacity,
        specific_heat_func=test_specific_heat,
        material_energy_func=test_material_energy,
        inverse_material_energy_func=test_inverse_material_energy,
        left_bc_func=test_left_bc,
        right_bc_func=test_right_bc,
        theta=1.0
    )
    
    # Initial condition: hot and cold in equilibrium
    T_init = np.array([5.0, 0.1])  # keV
    phi_init = A_RAD * C_LIGHT * T_init**4
    
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    print("\nInitial conditions:")
    r, phi, T = solver.get_solution()
    for i in range(n_cells):
        D = solver.get_diffusion_coefficient(T[i])
        sigma = test_rosseland_opacity(T[i])
        print(f"  Cell {i}: r={r[i]:.3f} cm, T={T[i]:.4f} keV, φ={phi[i]:.4e} GJ/cm³")
        print(f"           σ_R={sigma:.4e} cm⁻¹, D={D:.4e} cm²/ns")
    
    # Check face diffusion coefficient
    T_face_avg = 0.5 * (T[0] + T[1])
    D_face_avg = solver.get_diffusion_coefficient(T_face_avg)
    D0 = solver.get_diffusion_coefficient(T[0])
    D1 = solver.get_diffusion_coefficient(T[1])
    D_face_harmonic = 2.0 * D0 * D1 / (D0 + D1)
    
    print(f"\nFace diffusion coefficient (between cells):")
    print(f"  T_face (arithmetic avg) = {T_face_avg:.4f} keV")
    print(f"  D_face (from avg T) = {D_face_avg:.4e} cm²/ns")
    print(f"  D_0 = {D0:.4e} cm²/ns")
    print(f"  D_1 = {D1:.4e} cm²/ns")
    print(f"  D_face (harmonic avg) = {D_face_harmonic:.4e} cm²/ns")
    print(f"  Ratio D_face_avg / D_face_harmonic = {D_face_avg / D_face_harmonic:.4f}")
    
    # Compute expected flux
    dx = (r_max - r_min) / n_cells
    flux_expected_avg = -D_face_avg * (phi[1] - phi[0]) / dx
    flux_expected_harmonic = -D_face_harmonic * (phi[1] - phi[0]) / dx
    print(f"\nExpected flux (from cell 0 to cell 1):")
    print(f"  Using D_avg: {flux_expected_avg:.4e} GJ/(cm²·ns)")
    print(f"  Using D_harmonic: {flux_expected_harmonic:.4e} GJ/(cm²·ns)")
    print(f"  (Negative means flowing to the right)")
    
    # Take ONE timestep with verbose output
    print("\n" + "="*80)
    print("TAKING ONE TIMESTEP")
    print("="*80)
    
    # Manually step through Newton to see what's happening
    print("\nManually examining Newton iteration:")
    phi_star = phi.copy()
    T_star = T.copy()
    
    # Assemble and examine the phi equation
    A_phi, rhs_phi = solver.assemble_phi_equation(
        phi_star, T_star, phi, T, theta=1.0, source=None)
    
    print("\nPhi equation system (BEFORE boundary conditions):")
    print(f"  Matrix diagonal: {A_phi['diag']}")
    print(f"  Matrix sub: {A_phi['sub']}")
    print(f"  Matrix super: {A_phi['super']}")
    print(f"  RHS: {rhs_phi}")
    
    solver.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
    
    print("\nPhi equation system (AFTER boundary conditions):")
    print(f"  Matrix diagonal: {A_phi['diag']}")
    print(f"  Matrix sub: {A_phi['sub']}")
    print(f"  Matrix super: {A_phi['super']}")
    print(f"  RHS: {rhs_phi}")
    
    # Now run the actual timestep
    print("\nRunning actual timestep with DETAILED Newton output:")
    
    # Manually do Newton loop with full diagnostics
    phi_prev = phi.copy()
    T_prev = T.copy()
    phi_star = phi_prev.copy()
    T_star = T_prev.copy()
    
    for k in range(5):  # Just do 5 iterations
        print(f"\n--- Newton iteration {k+1} ---")
        print(f"  T_star = {T_star}")
        print(f"  phi_star = {phi_star}")
        
        # Solve phi equation
        A_phi, rhs_phi = solver.assemble_phi_equation(
            phi_star, T_star, phi_prev, T_prev, theta=1.0, source=None)
        solver.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
        from oneDFV import solve_tridiagonal
        phi_np1 = solve_tridiagonal(A_phi, rhs_phi)
        
        print(f"  phi_np1 (from phi eqn) = {phi_np1}")
        
        # Solve T equation
        T_np1 = solver.solve_T_equation(phi_np1, T_star, phi_prev, T_prev, theta=1.0)
        
        print(f"  T_np1 (from T eqn) = {T_np1}")
        
        # Compute residuals
        r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
        r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
        print(f"  Residuals: r_φ={r_phi:.4e}, r_T={r_T:.4e}")
        
        # Update
        phi_star = phi_np1.copy()
        T_star = T_np1.copy()
    
    # Reset and run with solver's method
    print("\n" + "="*80)
    print("Now running with solver.time_step():")
    solver.phi = phi.copy()
    solver.T = T.copy()
    solver.phi_old = phi.copy()
    solver.T_old = T.copy()
    solver.time_step(n_steps=1, verbose=True)
    
    print("\n" + "="*80)
    print("AFTER ONE TIMESTEP")
    print("="*80)
    
    r, phi_new, T_new = solver.get_solution()
    for i in range(n_cells):
        Delta_T = T_new[i] - T[i]
        Delta_phi = phi_new[i] - phi[i]
        print(f"  Cell {i}: T={T_new[i]:.4f} keV (ΔT={Delta_T:+.4f}), φ={phi_new[i]:.4e} GJ/cm³ (Δφ={Delta_phi:+.4e})")
    
    print("\nPhysical interpretation:")
    if T_new[0] < T[0]:
        print("  ✓ Cell 0 cooled down (correct - losing energy)")
    else:
        print("  ✗ Cell 0 heated up (WRONG!)")
    
    if T_new[1] > T[1]:
        print("  ✓ Cell 1 heated up (correct - gaining energy)")
    else:
        print("  ✗ Cell 1 cooled down (WRONG!)")
    
    # Energy conservation check
    E_init = np.sum(phi * solver.V_cells / C_LIGHT) + np.sum(test_material_energy(T) * solver.V_cells)
    E_final = np.sum(phi_new * solver.V_cells / C_LIGHT) + np.sum(test_material_energy(T_new) * solver.V_cells)
    E_error = abs(E_final - E_init) / E_init * 100
    print(f"\nEnergy conservation:")
    print(f"  Initial: {E_init:.6e} GJ")
    print(f"  Final:   {E_final:.6e} GJ")
    print(f"  Error:   {E_error:.4e}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature
    ax1.plot([r[0], r[0]], [0, T[0]], 'b-o', linewidth=2, markersize=8, label='Initial')
    ax1.plot([r[1], r[1]], [0, T[1]], 'b-o', linewidth=2, markersize=8)
    ax1.plot([r[0], r[0]], [0, T_new[0]], 'r-s', linewidth=2, markersize=8, label='After 1 step')
    ax1.plot([r[1], r[1]], [0, T_new[1]], 'r-s', linewidth=2, markersize=8)
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Temperature (keV)', fontsize=12)
    ax1.set_title('Temperature', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phi
    ax2.plot([r[0], r[0]], [0, phi[0]], 'b-o', linewidth=2, markersize=8, label='Initial')
    ax2.plot([r[1], r[1]], [0, phi[1]], 'b-o', linewidth=2, markersize=8)
    ax2.plot([r[0], r[0]], [0, phi_new[0]], 'r-s', linewidth=2, markersize=8, label='After 1 step')
    ax2.plot([r[1], r[1]], [0, phi_new[1]], 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Position (cm)', fontsize=12)
    ax2.set_ylabel('φ (GJ/cm³)', fontsize=12)
    ax2.set_title('Radiation Energy Density', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('two_cell_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'two_cell_test.png'")
    plt.close()

if __name__ == "__main__":
    run_two_cell_test()
