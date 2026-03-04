#!/usr/bin/env python3
"""
Temperature Bump Test - Multigroup
Closed system with reflecting boundaries and initial temperature bump.

Setup:
- Material: Same as Marshak wave (σ = 300 T^-3, c_v = 0.3 GJ/(cm³·keV))
- Boundaries: Reflecting (zero flux) on both sides
- Initial: T_r = T (radiation in equilibrium with material)
- Profile: 1 keV bump in center, 0.1 keV elsewhere
- Energy should be strictly conserved in this closed system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD
from planck_integrals import Bg_multigroup

# Physical constants
RHO = 1.0  # g/cm³

def marshak_opacity(T, r):
    """Temperature-dependent opacity: σ = 300 * T^-3"""
    T_min = 0.01  # Minimum temperature to prevent overflow (keV)
    T_safe = max(T, T_min)
    return 300.0 * T_safe**(-3.0)

def marshak_diffusion_coeff(T, r):
    """Diffusion coefficient: D = c/(3σ_R)"""
    sigma_R = marshak_opacity(T, r)
    return C_LIGHT / (3.0 * sigma_R)

def run_bump_test():
    """Run temperature bump test with multigroup solver"""
    
    print("="*80)
    print("Temperature Bump Test - Multigroup (2 Groups)")
    print("="*80)
    print("Setup:")
    print("  Material: σ = 300 * T^-3 (cm^-1), c_v = 0.3 GJ/(cm³·keV)")
    print("  Boundaries: Reflecting (zero flux) on both sides")
    print("  Initial: T_r = T, with 1 keV bump in center, 0.1 keV elsewhere")
    print("  This is a closed system - energy must be conserved!")
    print("="*80)
    
    # Problem setup
    n_groups = 3
    r_min = 0.0      # cm
    r_max = 1.0      # cm
    n_cells = 40     # Reasonable resolution
    
    # Energy group structure (keV)
    energy_edges = np.array([1e-8,0.01, 2.0, 10.0])
    
    # Time stepping parameters
    dt = 1/30/20       # ns
    n_steps = 50     # Run for 50 timesteps
    
    # Material properties
    rho = RHO  # g/cm³
    cv = 0.03 / rho  # GJ/(g·keV) - specific heat per unit mass
    
    # Define boundary condition functions: reflecting (zero flux, Neumann BC)
    def reflecting_bc(phi, r):
        """Reflecting BC: ∇φ = 0 (zero flux)"""
        return 0.0, 1.0, 0.0
    
    # Create solver with reflecting boundaries
    print(f"\nInitializing multigroup solver with {n_cells} cells...")
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=[marshak_diffusion_coeff] * n_groups,
        absorption_coeff_funcs=[marshak_opacity] * n_groups,
        left_bc_funcs=[reflecting_bc] * n_groups,
        right_bc_funcs=[reflecting_bc] * n_groups,
        rho=rho,
        cv=cv
    )
    
    # Initial condition: Temperature bump in center
    r = solver.r_centers
    T_init = np.full(n_cells, 0.1)  # Base temperature 0.1 keV
    
    # Add bump in center (between 0.4 and 0.6 cm)
    center_mask = (r >= 0.4) & (r <= 0.6)
    T_init[center_mask] = 1.0  # 1 keV bump
    
    # Set radiation in equilibrium with material: E_r = a*T^4
    E_r_init = A_RAD * T_init**4
    
    # Compute equilibrium group fractions: φ_g/φ_total = B_g(T)/Σ B_g'(T)
    phi_g_fraction_init = np.zeros((n_groups, n_cells))
    for i in range(n_cells):
        Bg_T = Bg_multigroup(energy_edges, T_init[i])  # Planck integrals for each group
        B_total = np.sum(Bg_T)
        phi_g_fraction_init[:, i] = Bg_T / B_total if B_total > 0 else 1.0 / n_groups
    
    solver.T = T_init.copy()
    solver.E_r = E_r_init.copy()
    solver.phi_g_fraction = phi_g_fraction_init.copy()
    
    # Initialize "old" values for first timestep
    solver.T_old = solver.T.copy()
    solver.E_r_old = solver.E_r.copy()
    
    # Compute initial total energy (should be conserved)
    e_mat_init = rho * cv * solver.T
    E_total_init = np.sum(solver.E_r * solver.solvers[0].V_cells) + \
                   np.sum(e_mat_init * solver.solvers[0].V_cells)
    
    print(f"\nInitial conditions:")
    print(f"  Temperature range: {T_init.min():.3f} - {T_init.max():.3f} keV")
    print(f"  E_r range: {E_r_init.min():.6e} - {E_r_init.max():.6e} GJ/cm³")
    print(f"  Total energy: E_total = {E_total_init:.6e} GJ")
    print(f"  Opacity range: {marshak_opacity(T_init.min(), 0):.2e} - {marshak_opacity(T_init.max(), 0):.2e} cm⁻¹")
    
    # Storage for analysis
    time_history = [0.0]
    T_max_history = [T_init.max()]
    T_min_history = [T_init.min()]
    E_total_history = [E_total_init]
    energy_error_history = [0.0]
    
    # Time evolution
    print(f"\nTime evolution:")
    print(f"{'Step':<6} {'Time':<10} {'T_max':<10} {'T_min':<10} {'ΔE/E_0':<15} {'Newton':<8} {'Conv':<4}")
    print("-" * 80)
    
    for step in range(1, n_steps + 1):
        # Take timestep
        verbose = (step <= 2)  # Verbose on first 2 steps
        info = solver.step(max_newton_iter=10, newton_tol=1e-10,
                          gmres_tol=1e-10, gmres_maxiter=200,
                          verbose=verbose)
        
        solver.advance_time()
        
        # Compute energy conservation
        e_mat = rho * cv * solver.T
        E_total = np.sum(solver.E_r * solver.solvers[0].V_cells) + \
                  np.sum(e_mat * solver.solvers[0].V_cells)
        energy_error = (E_total - E_total_init) / E_total_init
        
        # Store history
        time_history.append(step * dt)
        T_max_history.append(solver.T.max())
        T_min_history.append(solver.T.min())
        E_total_history.append(E_total)
        energy_error_history.append(energy_error)
        
        # Print progress
        if step % 5 == 0 or step <= 3:
            converged_str = "✓" if info.get('converged', False) else "✗"
            n_iter = info.get('newton_iter', info.get('n_newton_iter', '?'))
            print(f"{step:<6} {step*dt:<10.4f} {solver.T.max():<10.6f} "
                  f"{solver.T.min():<10.6f} {energy_error:<15.6e} "
                  f"{n_iter:<8} {converged_str}")
    
    # Final analysis
    print("\n" + "="*80)
    print("Temperature Bump Test Completed!")
    print("="*80)
    
    E_total_final = E_total_history[-1]
    abs_error = E_total_final - E_total_init
    rel_error = abs_error / E_total_init
    max_error = max(energy_error_history, key=abs)
    
    print(f"\nFinal state at t = {time_history[-1]:.4f} ns:")
    print(f"  Temperature range: {solver.T.min():.6f} - {solver.T.max():.6f} keV")
    print(f"  E_r range: {solver.E_r.min():.6e} - {solver.E_r.max():.6e} GJ/cm³")
    
    print(f"\nEnergy conservation (closed system):")
    print(f"  Initial total: E_0 = {E_total_init:.10e} GJ")
    print(f"  Final total:   E_f = {E_total_final:.10e} GJ")
    print(f"  Absolute error: ΔE = {abs_error:.10e} GJ")
    print(f"  Relative error: ΔE/E_0 = {rel_error:.10e}")
    print(f"  Max deviation:  max|ΔE/E_0| = {abs(max_error):.10e}")
    
    tolerance = 1e-10
    if abs(rel_error) < tolerance:
        print(f"  ✓ Energy conserved to tolerance ({tolerance})")
    else:
        print(f"  ✗ WARNING: Energy conservation error exceeds tolerance!")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature profiles
    ax = axes[0, 0]
    ax.plot(r, T_init, 'k--', label='Initial', linewidth=2)
    ax.plot(solver.r_centers, solver.T, 'r-', label='Final', linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature evolution
    ax = axes[0, 1]
    time_arr = np.array(time_history)
    ax.plot(time_arr, T_max_history, 'r-', label='T_max', linewidth=2)
    ax.plot(time_arr, T_min_history, 'b-', label='T_min', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total energy
    ax = axes[1, 0]
    ax.plot(time_arr, E_total_history, 'k-', linewidth=2)
    ax.axhline(E_total_init, color='gray', linestyle=':', label='Initial')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Total Energy (GJ)')
    ax.set_title('Total Energy (Should Be Constant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Relative energy error
    ax = axes[1, 1]
    ax.plot(time_arr, np.array(energy_error_history) * 100, 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Relative error ΔE/E_0 (%)')
    ax.set_title('Energy Conservation Error (Closed System)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_multigroup_bump.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_multigroup_bump.png'")

if __name__ == "__main__":
    run_bump_test()
