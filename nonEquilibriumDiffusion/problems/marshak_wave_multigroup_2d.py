#!/usr/bin/env python3
"""
Marshak Wave Problem - 2D Multigroup Version
Tests 2D solver against self-similar 1D solution with small y-domain

Problem setup:
- Two energy groups with same absorption coefficient
- Left boundary: Robin BC with A=0.5, B=D, C=F_g (Marshak BC)
- Right/bottom/top boundaries: Neumann BC (zero flux)
- Material opacity: σ = 300·T^-3 cm⁻¹ (constant per group, same for both)
- Heat capacity: c_v = 0.3 GJ/(cm³·keV)
- Domain: x ∈ [0, 0.1] cm, y ∈ [0, 0.005] cm (small to test 1D agreement)
- Plot solutions at specified times and compare with self-similar profile
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver_2d import MultigroupDiffusionSolver2D, C_LIGHT, A_RAD

# Physical constants
RHO = 1.0  # g/cm³

# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES
# =============================================================================

def marshak_opacity(T, x, y):
    """Opacity with power-law temperature dependence: σ = 300·T^-3
    
    Parameters:
    -----------
    T : float or ndarray
        Temperature in keV
    x, y : float or ndarray
        Position (unused for homogeneous material)
    
    Returns:
    --------
    float or ndarray
        Absorption coefficient in cm⁻¹
    """
    T_safe = 0.01
    if np.isscalar(T):
        T = max(T, T_safe)
        return 300.0 * T**(-3)
    else:
        T_arr = np.atleast_1d(T).copy()
        T_arr[T_arr < T_safe] = T_safe
        return 300.0 * T_arr**(-3)


def marshak_diffusion_coeff(T, x, y):
    """Diffusion coefficient: D = 1/(3σ_R)
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    x, y : float
        Position (cm) - unused for homogeneous material
    
    Returns:
    --------
    D : float
        Diffusion coefficient (cm²/ns)
    """
    sigma_R = marshak_opacity(T, x, y)
    return 1.0 / (3.0 * sigma_R)


# =============================================================================
# SELF-SIMILAR SOLUTION (from 1D problem)
# =============================================================================

def compute_self_similar_temp(r, t, xi_max=1.11305, omega=0.05989):
    """Compute self-similar temperature profile for Marshak wave
    
    T(r,t) = [max(0, (1 - xi/xi_max)(1 + omega*xi/xi_max))]^(1/6)
    where xi = r / sqrt(K_const * t)
    
    Parameters:
    -----------
    r : float or ndarray
        Radial position (cm)
    t : float
        Time (ns)
    xi_max : float
        Self-similar coordinate at wave front
    omega : float
        Self-similar shape parameter
    
    Returns:
    --------
    T : float or ndarray
        Temperature (keV)
    """
    # Scaling constant from Marshak problem
    K_const = 8.0 * A_RAD * C_LIGHT / ((4.0 + 3.0) * 3.0 * 300.0 * RHO * 0.3)
    
    if t < 1e-10:
        return np.ones_like(r) * 0.01  # Avoid division by zero
    
    xi = r / np.sqrt(K_const * t)
    
    r_arr = np.atleast_1d(r)
    T = np.zeros_like(r_arr)
    
    mask = xi < xi_max
    T[mask] = np.power((1.0 - xi[mask] / xi_max) * (1.0 + omega * xi[mask] / xi_max), 1.0 / 6.0)
    T[~mask] = 0.01
    
    if np.isscalar(r):
        return float(T[0])
    return T


# =============================================================================
# 2D MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_2d(use_preconditioner=False, n_cells_x=200, n_cells_y=20):
    """Run 2D multigroup Marshak wave simulation
    
    Parameters:
    -----------
    use_preconditioner : bool
        Use preconditioning in GMRES
    n_cells_x : int
        Number of cells in x-direction (radial)
    n_cells_y : int
        Number of cells in y-direction (transverse, small)
    """
    
    print("="*80)
    print("Marshak Wave Problem - 2D Multigroup Version")
    print("="*80)
    print("Test: 2D solver on quasi-1D domain (small y-extent)")
    print("Expected: Agreement with self-similar 1D solution")
    print("="*80)
    print(f"Mesh: {n_cells_x} × {n_cells_y} = {n_cells_x * n_cells_y} cells")
    print("Material properties:")
    print("  Opacity: σ = 300·T^-3 cm⁻¹ (both groups)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm³·keV)")
    print("  Density: ρ = 1.0 g/cm³")
    print("  Left BC: Marshak (Robin) at T = 1 keV")
    print("  Right/top/bottom BC: Neumann (zero flux)")
    if use_preconditioner:
        print("  GMRES: Using preconditioning")
    else:
        print("  GMRES: No preconditioning")
    print("="*80)
    
    # Problem setup
    n_groups = 2
    
    # Domain: quasi-1D (thin in y)
    x_min, x_max = 0.0, 0.5  # cm (large domain like 1D test)
    y_min, y_max = 0.0, 0.1  # cm (thin strip)
    
    # Energy group structure (keV)
    energy_edges = np.array([0.1, 2.0, 50.0])
    
    # Time stepping
    dt = 0.001  # ns (smaller for accuracy)
    target_times = [0.01, 0.05, 0.1]  # ns - run to meaningful times
    
    # Material properties
    rho = RHO
    cv = 0.3 / rho  # GJ/(g·keV)
    
    # Boundary conditions
    T_bc = 1.0  # keV (blackbody temperature)
    
    # Compute emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    sigma_a_ref = marshak_opacity(T_bc, 0.0, 0.0)
    sigma_a_groups = np.array([sigma_a_ref, sigma_a_ref])
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc,
                                                sigma_a_groups=sigma_a_groups)
    
    print(f"\nEnergy group edges: {energy_edges} keV")
    print(f"Emission fractions at T = {T_bc} keV: {chi}")
    
    # Incoming flux for boundary condition
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    F_g_values = [chi[g] * F_total for g in range(n_groups)]
    
    # Robin BC parameters
    sigma_bc = marshak_opacity(T_bc, 0.0, 0.0)
    BC_A = 0.5
    BC_B = 1.0 / (3.0 * sigma_bc)
    
    print(f"\nBoundary condition (Robin type - Marshak):")
    print(f"  A = {BC_A}, B = {BC_B:.6e} cm")
    print(f"  F_total(T_b) = {F_total:.6e} GJ/(cm²·ns)")
    for g in range(n_groups):
        print(f"  Group {g}: C = F_g = {F_g_values[g]:.6e} GJ/(cm²·ns)")
    
    # Boundary condition functions (for 2D solver)
    def make_left_bc_func(C_val):
        """Robin BC at x=0 (incoming radiation)"""
        def left_bc(phi, pos, t):
            return BC_A, BC_B, C_val
        return left_bc
    
    def zero_flux_bc(phi, pos, t):
        """Neumann BC: ∇φ = 0 (zero flux)"""
        return 0.0, 1.0, 0.0
    
    # Create BC function lists for each group
    left_bc_funcs = [make_left_bc_func(F_g_values[g]) for g in range(n_groups)]
    right_bc_funcs = [zero_flux_bc] * n_groups
    bottom_bc_funcs = [zero_flux_bc] * n_groups
    top_bc_funcs = [zero_flux_bc] * n_groups
    
    boundary_funcs = {
        'left': left_bc_funcs,
        'right': right_bc_funcs,
        'bottom': bottom_bc_funcs,
        'top': top_bc_funcs
    }
    
    # Create solver
    print(f"\nInitializing 2D multigroup solver...")
    solver = MultigroupDiffusionSolver2D(
        n_groups=n_groups,
        x_min=x_min, x_max=x_max, nx_cells=n_cells_x,
        y_min=y_min, y_max=y_max, ny_cells=n_cells_y,
        energy_edges=energy_edges,
        geometry='cartesian',
        dt=dt,
        diffusion_coeff_funcs=[marshak_diffusion_coeff] * n_groups,
        absorption_coeff_funcs=[marshak_opacity] * n_groups,
        boundary_funcs=boundary_funcs,
        rho=rho,
        cv=cv,
        max_newton_iter=5,
        newton_tol=1e-6
    )
    
    # Scaling constant for wave propagation
    K_const = 8.0 * A_RAD * C_LIGHT / ((4.0 + 3.0) * 3.0 * 300.0 * rho * cv)
    
    # Initialize from self-similar solution at t=0.1 ns
    # This matches the 1D test approach
    t_init = 0.1  # ns
    
    x_centers = solver.x_centers
    T_init_1d = compute_self_similar_temp(x_centers, t_init)
    
    # Expand to 2D (same profile in all y)
    T_init = np.tile(T_init_1d, (solver.ny_cells, 1)).T
    
    solver.T = T_init.flatten()
    solver.T_old = solver.T.copy()
    solver.E_r = A_RAD * solver.T**4
    solver.E_r_old = solver.E_r.copy()
    
    # Initialize phi groups from E_r
    from multigroup_diffusion_solver import Bg_multigroup
    B_g_init = Bg_multigroup(energy_edges, np.mean(T_init.flatten()))
    chi_init = B_g_init / np.sum(B_g_init)
    
    for g in range(n_groups):
        solver.phi_g_stored[g, :] = chi_init[g] * solver.E_r * C_LIGHT
    
    solver.t = t_init
    
    print(f"Initial condition: Self-similar solution at t = {t_init:.2e} ns")
    print(f"  T_max = {T_init.max():.6f} keV, T_min = {T_init.min():.6f} keV")
    print(f"  E_r_max = {solver.E_r.max():.6e} GJ/cm³")
    print(f"  Wave front position ≈ {np.sqrt(K_const * t_init) * 1.11305:.6f} cm")
    
    # Time evolution
    print("\nTime evolution:")
    print(f"{'Step':<6} {'Time':<10} {'T_max':<12} {'T_min':<12} {'E_r_max':<15} {'Newton':<8}")
    print("-" * 75)
    
    solutions = {}
    step = 0
    current_time = 0.0
    
    # Run to final time
    final_time = 5.0  # ns (match 1D test time range)
    max_steps = 5000
    
    for step in range(max_steps):
        # Perform one time step
        info = solver.step(verbose=False, gmres_tol=1e-6, gmres_maxiter=200,
                          use_preconditioner=use_preconditioner)
        
        current_time = solver.t
        
        # Print progress every N steps
        if step % 10 == 0:
            print(f"{step:<6} {current_time:<10.4f} {solver.T.max():<12.6f} {solver.T.min():<12.6f} {solver.E_r.max():<15.6e} {info['newton_iterations']:<8}")
        
        # Stop at final time
        if current_time >= final_time:
            print(f"{step:<6} {current_time:<10.4f} {solver.T.max():<12.6f} {solver.T.min():<12.6f} {solver.E_r.max():<15.6e} {info['newton_iterations']:<8}")
            print("\nFinal time reached!")
            break
    
    print("="*75)
    print("2D Marshak wave simulation completed!")
    
    # Generate plots comparing with 1D self-similar solution
    plot_results_2d(solver, energy_edges, RHO, cv)
    
    return solver


def plot_results_2d(solver, energy_edges, rho, cv):
    """Generate plots comparing 2D solution with self-similar profile
    
    Parameters:
    -----------
    solver : MultigroupDiffusionSolver2D
        The solver object with final solution
    energy_edges : ndarray
        Energy group edges
    rho : float
        Material density
    cv : float
        Heat capacity
    """
    
    print("\nGenerating plots...")
    
    # Extract transverse average (should be constant for 1D problem)
    T_2d = solver.T.reshape(solver.nx_cells, solver.ny_cells)
    T_1d = T_2d.mean(axis=1)  # Average over y
    
    E_r_2d = solver.E_r.reshape(solver.nx_cells, solver.ny_cells)
    E_r_1d = E_r_2d.mean(axis=1)
    
    # Compute self-similar profile at current time
    K_const = 8.0 * A_RAD * C_LIGHT / ((4.0 + 3.0) * 3.0 * 300.0 * rho * cv)
    T_selfsim = compute_self_similar_temp(solver.x_centers, solver.t)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot temperature
    ax1.plot(solver.x_centers, T_1d, 'b.-', label='2D solver (y-average)', linewidth=2, markersize=6)
    ax1.plot(solver.x_centers, T_selfsim, 'r--', label='Self-similar profile', linewidth=2)
    ax1.set_xlabel('Position x (cm)', fontsize=12)
    ax1.set_ylabel('Temperature T (keV)', fontsize=12)
    ax1.set_title(f'Temperature Profile at t = {solver.t:.4f} ns', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, solver.x_max])
    ax1.set_ylim([0, 1.2])
    
    # Plot radiation energy
    E_r_selfsim = A_RAD * T_selfsim**4
    ax2.semilogy(solver.x_centers, E_r_1d, 'b.-', label='2D solver (y-average)', linewidth=2, markersize=6)
    ax2.semilogy(solver.x_centers, E_r_selfsim, 'r--', label='Self-similar profile', linewidth=2)
    ax2.set_xlabel('Position x (cm)', fontsize=12)
    ax2.set_ylabel('Radiation Energy $E_r$ (GJ/cm³)', fontsize=12)
    ax2.set_title(f'Radiation Energy at t = {solver.t:.4f} ns', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, solver.x_max])
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'marshak_wave_2d_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    # Compute and print errors
    error_T = np.linalg.norm(T_1d - T_selfsim) / np.linalg.norm(T_selfsim)
    error_E_r = np.linalg.norm(E_r_1d - E_r_selfsim) / np.linalg.norm(E_r_selfsim)
    
    print(f"\nComparison with self-similar solution at t = {solver.t:.4f} ns:")
    print(f"  Relative error in T: {error_T:.3e}")
    print(f"  Relative error in E_r: {error_E_r:.3e}")
    
    # Check transverse uniformity (verify it's quasi-1D)
    T_variation = np.std(T_2d, axis=1)
    T_max = np.max(T_2d, axis=1)
    transverse_variation = np.max(T_variation / (T_max + 1e-10))
    
    print(f"\nTransverse uniformity check (2D effect on quasi-1D problem):")
    print(f"  Max relative variation in y-direction: {transverse_variation:.3e}")
    if transverse_variation < 1e-3:
        print("  ✓ Excellent transverse uniformity (confirms quasi-1D behavior)")
    elif transverse_variation < 1e-2:
        print("  ✓ Good transverse uniformity")
    else:
        print("  ⚠ Noticeable transverse variation")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="2D Marshak Wave Multigroup Solver")
    parser.add_argument('--precond', action='store_true', help='Use GMRES preconditioning')
    parser.add_argument('--nx', type=int, default=40, help='Number of cells in x')
    parser.add_argument('--ny', type=int, default=4, help='Number of cells in y')
    
    args = parser.parse_args()
    
    solver = run_marshak_wave_2d(use_preconditioner=args.precond, 
                                  n_cells_x=args.nx, n_cells_y=args.ny)
