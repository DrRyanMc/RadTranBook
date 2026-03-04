#!/usr/bin/env python3
"""
Marshak Wave Problem - 1-Group (Gray) Diffusion

This is a standard test problem for radiation diffusion:
- Temperature-dependent opacity: σ = 300 * T^-3
- Hot blackbody boundary condition on the left
- Cold initial condition
- Zero flux boundary on the right

With 1 group, GMRES should converge in 1-2 iterations since the operator
is essentially scalar.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Physical constants
RHO = 1.0  # g/cm³

# =============================================================================
# Material properties
# =============================================================================

def marshak_opacity(T, r):
    """Temperature-dependent opacity: σ = 300 * T^-3
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    r : float
        Position (cm) - not used, but needed for solver interface
    
    Returns:
    --------
    sigma : float
        Opacity (cm^-1)
    """
    T_min = 0.01  # Minimum temperature to prevent overflow (keV)
    T_safe = max(T, T_min)
    return 300.0 * T_safe**(-3.0)


def marshak_diffusion_coeff(T, r):
    """Diffusion coefficient: D = c/(3σ_R)
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    r : float
        Position (cm)
    
    Returns:
    --------
    D : float
        Diffusion coefficient (cm)
    """
    sigma_R = marshak_opacity(T, r)
    return C_LIGHT / (3.0 * sigma_R)


def run_marshak_wave_1group():
    """Run 1-group Marshak wave problem"""
    
    print("="*80)
    print("Marshak Wave Problem - 1-Group (Gray Diffusion)")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_a = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("="*80)
    
    # Problem setup
    n_groups = 1  # Gray diffusion
    r_min = 0.0      # cm
    r_max = 0.1      # cm
    n_cells = 50     # Good resolution
    
    # Energy group structure (keV) - single group covering full spectrum
    energy_edges = np.array([0.01, 10.0])
    
    # Time stepping parameters
    dt = 0.001       # ns
    target_times = [0.05]  # ns - just 5 timesteps for quick test
    
    # Material properties
    rho = RHO  # g/cm³
    # Use slightly large heat capacity to keep T nearly constant, but not exactly 1
    # (cv_large means f ≈ 1, but not exactly 1.0 which would cause ill-conditioning)
    cv = 0.3 / rho  # GJ/(g·keV) - 10 billion times normal
    
    # Boundary conditions
    T_bc = 1.0  # keV (blackbody temperature at left boundary)
    
    # Use a RADIATION Robin BC that's compatible with κ-reduction:
    # The idea: couple φ at boundary to external blackbody temperature
    # φ_exterior = a·c·T_bc^4
    # Use: φ + (D/h)·∇φ = φ_exterior
    # where h is a "heat transfer coefficient" (we use D itself for simplicity)
    # This gives: A=1, B=D, C=φ_exterior
    
    # For constant opacity, we can use a fixed D at boundary
    sigma_R_bc = marshak_opacity(T_bc, 0.0)
    D_bc_fixed = C_LIGHT / (3.0 * sigma_R_bc)  # Diffusion coeff at T_bc
    phi_exterior = A_RAD * C_LIGHT * T_bc**4
    
    # Robin BC: φ + D·∇φ = φ_exterior gives:
    # A = 1, B = D_bc_fixed, C = phi_exterior
    A_robin = 1.0
    B_robin = D_bc_fixed
    C_robin = phi_exterior
    
    # Define boundary condition functions (return A, B, C for Robin BC: A·φ + B·∇φ = C)
    def left_bc_func(phi, r):
        """Radiation Robin BC at left: φ + D·∇φ = a·c·T_bc^4"""
        return A_robin, B_robin, C_robin
    
    def right_bc_func(phi, r):
        """Neumann BC at right: ∇φ = 0 (zero flux)"""
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [left_bc_func]
    right_bc_funcs = [right_bc_func]
    
    print(f"\nEnergy group edges: {energy_edges} keV")
    print(f"Robin BC (radiation): A={A_robin}, B={B_robin:.6e}, C={C_robin:.6e}")
    print(f"  This couples interior φ to external blackbody at T={T_bc} keV")
    
    # Create solver
    print(f"\nInitializing 1-group solver with {n_cells} cells...")
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=[marshak_diffusion_coeff],
        absorption_coeff_funcs=[marshak_opacity],
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    # Initial condition: cold material
    T_init = 0.1  # keV
    solver.T = np.full(n_cells, T_init)
    solver.T_old = solver.T.copy()
    solver.E_r = np.full(n_cells, A_RAD * T_init**4)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    sigma_init = marshak_opacity(T_init, 0.0)
    print(f"Initial conditions: T = {T_init} keV, E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"Initial opacity: σ = {sigma_init:.2e} cm⁻¹")
    print(f"GMRES should converge in 1-2 iterations (1 group = scalar problem)")
    
    # Time evolution
    print("\nTime evolution:")
    print(f"{'Step':<6} {'Time':<10} {'T_max':<12} {'T_min':<12} {'E_r_max':<15} {'Newton':<8} {'GMRES':<8} {'Conv':<4}")
    print("-" * 90)
    
    current_time = 0.0
    solutions = []
    step_count = 0
    
    for target_time in target_times:
        while current_time < target_time:
            # Take timestep - use only 1 Newton iteration
            verbose_this_step = (step_count < 3)  # Verbose on first 3 steps
            info = solver.step(max_newton_iter=1, newton_tol=1e-6,
                              gmres_tol=1e-6, gmres_maxiter=50,
                              verbose=verbose_this_step)
            
            solver.advance_time()
            current_time += dt
            step_count += 1
            
            # Print progress
            converged_str = "✓" if info.get('converged', False) else "✗"
            n_newton = info.get('newton_iter', info.get('n_newton_iter', '?'))
            n_gmres = info.get('gmres_iter', '?')
            
            if step_count % 10 == 0 or step_count <= 5 or abs(current_time - target_time) < 1e-10:
                print(f"{step_count:<6} {current_time:<10.4f} {solver.T.max():<12.6f} "
                      f"{solver.T.min():<12.6f} {solver.E_r.max():<15.6e} "
                      f"{n_newton:<8} {n_gmres:<8} {converged_str}")
        
        # Store solution at this target time
        solutions.append({
            'time': current_time,
            'r': solver.r_centers.copy(),
            'T': solver.T.copy(),
            'E_r': solver.E_r.copy(),
            'kappa': solver.kappa.copy()
        })
        
        print(f"\nSolution stored at t = {current_time:.4f} ns")
    
    print("\n" + "="*80)
    print("Marshak Wave Simulation Completed!")
    print("="*80)
    
    return solutions, solver, energy_edges


def plot_results(solutions, solver, energy_edges):
    """Plot simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature profiles
    ax = axes[0, 0]
    for sol in solutions:
        ax.plot(sol['r'], sol['T'], label=f"t = {sol['time']:.2f} ns", linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Radiation energy density
    ax = axes[0, 1]
    for sol in solutions:
        ax.semilogy(sol['r'], sol['E_r'], label=f"t = {sol['time']:.2f} ns", linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('E_r (GJ/cm³)')
    ax.set_title('Radiation Energy Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absorption rate κ
    ax = axes[1, 0]
    for sol in solutions:
        ax.semilogy(sol['r'], np.maximum(sol['kappa'], 1e-10), 
                   label=f"t = {sol['time']:.2f} ns", linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('κ = σ*φ (GJ/(cm³·ns))')
    ax.set_title('Absorption Rate Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Wave front position vs time
    ax = axes[1, 1]
    times = [sol['time'] for sol in solutions]
    # Define wave front as position where T reaches 0.5 keV
    front_positions = []
    for sol in solutions:
        idx = np.where(sol['T'] > 0.5)[0]
        if len(idx) > 0:
            front_positions.append(sol['r'][idx[-1]])
        else:
            front_positions.append(0.0)
    
    ax.plot(times, front_positions, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Wave Front Position (cm)')
    ax.set_title('Wave Front Propagation (T > 0.5 keV)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('marshak_wave_1group.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'marshak_wave_1group.png'")


def main():
    """Main execution"""
    print("\nRunning 1-group (gray) Marshak wave simulation...")
    solutions, solver, energy_edges = run_marshak_wave_1group()
    plot_results(solutions, solver, energy_edges)


if __name__ == "__main__":
    main()
