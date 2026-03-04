#!/usr/bin/env python3
"""
Marshak Wave Problem - Multigroup Version
Classic radiative heat wave test problem with 2 energy groups

Problem setup:
- Two energy groups with same absorption coefficient
- Left boundary: incoming flux from blackbody at 1 keV
- Material opacity: σ_R = σ_P = σ_a = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Plot solutions at 1, 10, and 20 ns
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB

# Physical constants
RHO = 1.0  # g/cm³

# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES
# =============================================================================

def marshak_opacity(T, r):
    """Temperature-dependent opacity: σ = 10 * T^-3
    
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
    return 10.0 * T_safe**(-3.0)


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
        Diffusion coefficient (cm²/ns)
    """
    sigma_R = marshak_opacity(T, r)
    return C_LIGHT / (3.0 * sigma_R)


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_multigroup():
    """Run 2-group Marshak wave simulation"""
    
    print("="*80)
    print("Marshak Wave Problem - Multigroup (2 Groups)")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_a = 10 * T^-3 (cm^-1) for both groups")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("="*80)
    
    # Problem setup
    n_groups = 2
    r_min = 0.0      # cm
    r_max = 0.5      # cm
    n_cells = 20     # Reasonable resolution
    
    # Energy group structure (keV)
    # Split spectrum: low energy [0.01, 2.0] keV and high energy [2.0, 10.0] keV
    energy_edges = np.array([0.01, 2.0, 10.0])
    
    # Time stepping parameters
    dt = 0.01        # ns - larger timestep for efficiency
    target_times = [0.1, 0.5, 1.0]  # ns - run to meaningful times
    
    # Material properties
    rho = RHO  # g/cm³
    cv = 0.3 / rho  # GJ/(g·keV) - specific heat per unit mass
    
    # Boundary conditions
    T_bc = 1.0  # keV (blackbody temperature)
    
    # Compute emission fractions at T_bc to split boundary condition
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    
    # Both groups have same absorption coefficient, so provide it for proper weighting
    sigma_a_ref = marshak_opacity(T_bc, 0.0)
    sigma_a_groups = np.array([sigma_a_ref, sigma_a_ref])
    
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, 
                                                sigma_a_groups=sigma_a_groups)
    
    print(f"\nEnergy group edges: {energy_edges} keV")
    print(f"Emission fractions at T = {T_bc} keV: {chi}")
    
    # Total blackbody flux at boundary
    phi_bc_total = C_LIGHT * A_RAD * T_bc**4
    
    # Split by emission fractions
    phi_bc_groups = [chi[g] * phi_bc_total for g in range(n_groups)]
    
    print(f"Left boundary φ_bc,total = {phi_bc_total:.6e} GJ/cm³")
    print(f"Left boundary φ_bc per group: {phi_bc_groups}")
    
    # Define boundary condition functions
    def make_left_bc_func(phi_bc_val):
        """Create Dirichlet BC function for left boundary"""
        def left_bc(phi, r):
            return 1.0, 0.0, phi_bc_val
        return left_bc
    
    def right_bc_func(phi, r):
        """Neumann BC at right: ∇φ = 0 (zero flux)"""
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [make_left_bc_func(phi_bc_groups[g]) for g in range(n_groups)]
    right_bc_funcs = [right_bc_func] * n_groups
    
    # Create solver
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
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    # Initial condition: cold material (more realistic Marshak wave)
    T_init = 0.1  # keV (cold material, but opacity is now manageable)
    solver.T = np.full(n_cells, T_init)
    solver.T_old = solver.T.copy()
    solver.E_r = np.full(n_cells, A_RAD * T_init**4)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    sigma_init = marshak_opacity(T_init, 0.0)
    print(f"Initial conditions: T = {T_init} keV, E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"Initial opacity: σ = {sigma_init:.2e} cm⁻¹")
    
    # Time evolution
    print("\nTime evolution:")
    print(f"{'Step':<6} {'Time':<10} {'T_max':<12} {'T_min':<12} {'E_r_max':<15} {'Newton':<8} {'Conv':<4}")
    print("-" * 80)
    
    current_time = 0.0
    solutions = []
    step_count = 0
    
    for target_time in target_times:
        while current_time < target_time:
            # Take timestep
            verbose_this_step = (step_count < 3)  # Verbose on first 3 steps
            info = solver.step(max_newton_iter=10, newton_tol=1e-6,
                              gmres_tol=1e-4, gmres_maxiter=200,
                              verbose=verbose_this_step)
            
            solver.advance_time()
            current_time += dt
            step_count += 1
            
            # Print progress occasionally
            if step_count % 20 == 0 or step_count == 1:
                converged_str = "✓" if info['converged'] else "✗"
                print(f"{step_count:<6} {current_time:<10.4f} {solver.T.max():<12.6f} "
                      f"{solver.T.min():<12.6f} {solver.E_r.max():<15.6e} "
                      f"{info['newton_iter']:<8} {converged_str}")
        
        # Store solution at target time
        r = solver.r_centers.copy()
        T = solver.T.copy()
        E_r = solver.E_r.copy()
        T_rad = (E_r / A_RAD)**0.25
        
        # Compute individual group solutions for analysis
        solver.update_absorption_coefficients(T)
        solver.fleck_factor = solver.compute_fleck_factor(T)
        xi_g_list = [solver.compute_source_xi(g, T) for g in range(n_groups)]
        
        phi_groups = np.zeros((n_groups, n_cells))
        E_r_groups = np.zeros((n_groups, n_cells))
        for g in range(n_groups):
            phi_g = solver.compute_phi_g(g, solver.kappa, T, xi_g_list)
            phi_groups[g, :] = phi_g
            E_r_groups[g, :] = phi_g / C_LIGHT
        
        solutions.append({
            'time': current_time,
            'r': r,
            'T': T,
            'E_r': E_r,
            'T_rad': T_rad,
            'phi_groups': phi_groups,
            'E_r_groups': E_r_groups,
            'chi': solver.chi.copy()
        })
        
        print(f"  t = {current_time:.1f} ns:")
        print(f"    Material: max T = {T.max():.4f} keV, min T = {T.min():.4f} keV")
        print(f"    Radiation: max T_rad = {T_rad.max():.4f} keV, min T_rad = {T_rad.min():.4f} keV")
        print(f"    Max E_r = {E_r.max():.4e} GJ/cm³")
        for g in range(n_groups):
            print(f"    Group {g}: max E_r = {E_r_groups[g].max():.4e} GJ/cm³")
    
    return solutions, solver, energy_edges


def plot_marshak_wave_multigroup(solutions, energy_edges):
    """Plot multigroup Marshak wave solutions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'green', 'red']
    n_groups = len(energy_edges) - 1
    
    # Self-similar solution parameters
    xi_max = 1.11305
    omega = 0.05989
    self_similar = lambda xi: (xi < xi_max) * np.power((1 - xi/xi_max)*(1+omega*xi/xi_max), 1/6)
    xi_vals = np.linspace(0, xi_max, 200)
    K_const = 8*A_RAD*C_LIGHT/((4+3)*3*300*RHO*0.3)
    
    # Plot 1: Material temperature profiles
    ax = axes[0, 0]
    for sol, color in zip(solutions, colors):
        t = sol['time']
        r = sol['r']
        T = sol['T']
        ax.plot(r, T, color=color, linewidth=2, label=f't = {t:.0f} ns')
        
        # Self-similar solution
        r_ref = xi_vals * (K_const * t)**0.5
        T_ref = self_similar(xi_vals)
        ax.plot(r_ref, T_ref, color=color, linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('Material Temperature T (keV)', fontsize=11)
    ax.set_title('Material Temperature Profiles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, solutions[-1]['r'][-1])
    
    # Plot 2: Radiation temperature profiles
    ax = axes[0, 1]
    for sol, color in zip(solutions, colors):
        t = sol['time']
        r = sol['r']
        T_rad = sol['T_rad']
        ax.plot(r, T_rad, color=color, linewidth=2, label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('Radiation Temperature $T_{rad}$ (keV)', fontsize=11)
    ax.set_title('Radiation Temperature Profiles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, solutions[-1]['r'][-1])
    
    # Plot 3: Both temperatures comparison (last time)
    ax = axes[0, 2]
    sol = solutions[-1]
    t = sol['time']
    r = sol['r']
    T = sol['T']
    T_rad = sol['T_rad']
    
    ax.plot(r, T, 'b-', linewidth=2, label='Material T')
    ax.plot(r, T_rad, 'r--', linewidth=2, label='Radiation $T_{rad}$')
    
    # Self-similar solution
    r_ref = xi_vals * (K_const * t)**0.5
    T_ref = self_similar(xi_vals)
    ax.plot(r_ref, T_ref, 'k:', linewidth=2, alpha=0.7, label='Self-similar')
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('Temperature (keV)', fontsize=11)
    ax.set_title(f'Temperature Comparison at t = {t:.0f} ns', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, sol['r'][-1])
    
    # Plot 4: Total radiation energy density
    ax = axes[1, 0]
    for sol, color in zip(solutions, colors):
        t = sol['time']
        r = sol['r']
        E_r = sol['E_r']
        ax.plot(r, E_r, color=color, linewidth=2, label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('Total $E_r$ (GJ/cm³)', fontsize=11)
    ax.set_title('Total Radiation Energy Density', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, solutions[-1]['r'][-1])
    
    # Plot 5: Group-wise energy density (last time)
    ax = axes[1, 1]
    sol = solutions[-1]
    t = sol['time']
    r = sol['r']
    E_r_groups = sol['E_r_groups']
    
    group_colors = ['purple', 'orange']
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        ax.plot(r, E_r_groups[g], color=group_colors[g], linewidth=2, 
                label=f'Group {g} [{E_low:.2f}-{E_high:.1f} keV]')
    
    # Also plot total
    ax.plot(r, sol['E_r'], 'k--', linewidth=1.5, alpha=0.7, label='Total')
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('$E_{r,g}$ (GJ/cm³)', fontsize=11)
    ax.set_title(f'Group Energy Densities at t = {t:.0f} ns', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, sol['r'][-1])
    
    # Plot 6: Group fractions (last time)
    ax = axes[1, 2]
    sol = solutions[-1]
    t = sol['time']
    r = sol['r']
    E_r_groups = sol['E_r_groups']
    E_r_total = sol['E_r']
    
    for g in range(n_groups):
        fraction = E_r_groups[g] / (E_r_total + 1e-50)
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        ax.plot(r, fraction, color=group_colors[g], linewidth=2,
                label=f'Group {g} [{E_low:.2f}-{E_high:.1f} keV]')
    
    # Add horizontal line showing emission fractions
    chi = sol['chi']
    for g in range(n_groups):
        ax.axhline(chi[g], color=group_colors[g], linestyle=':', linewidth=1, 
                   alpha=0.5, label=f'χ_{g} = {chi[g]:.3f}')
    
    ax.set_xlabel('Position r (cm)', fontsize=11)
    ax.set_ylabel('Group Fraction $E_{r,g}/E_r$', fontsize=11)
    ax.set_title(f'Group Fractions at t = {t:.0f} ns', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(0, sol['r'][-1])
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('marshak_wave_multigroup.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'marshak_wave_multigroup.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run multigroup Marshak wave simulation
    print("\nRunning 2-group Marshak wave simulation...")
    solutions, solver, energy_edges = run_marshak_wave_multigroup()
    
    # Plot results
    print("\nPlotting multigroup Marshak wave results...")
    plot_marshak_wave_multigroup(solutions, energy_edges)
    
    print("\n" + "="*80)
    print("Multigroup Marshak wave simulation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
