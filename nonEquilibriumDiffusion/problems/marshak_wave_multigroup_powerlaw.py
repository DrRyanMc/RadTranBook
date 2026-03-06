#!/usr/bin/env python3
"""
Marshak Wave Problem - Multigroup Version with Power-Law Opacity

Problem setup:
- 10 energy groups with power-law absorption coefficient
- Opacity: σ_{a,g}(T,E) = 10.0 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
- For each group: use geometric mean of opacities at group boundaries
- Left boundary: Robin BC (Marshak boundary condition)
- Right boundary: Zero flux (Neumann BC)
- Heat capacity: c_v = 0.3 GJ/(cm³·keV)
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver_patched_lmfgk import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB, Bg_multigroup

# Physical constants
RHO = 1.0  # g/cm³

# =============================================================================
# POWER-LAW OPACITY FUNCTIONS
# =============================================================================

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """Power-law opacity at specific energy
    
    σ_a(T,E) = 10.0 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
    
    Parameters:
    -----------
    T : float or ndarray
        Temperature in keV
    E : float
        Photon energy in keV
    rho : float
        Density in g/cm³
    
    Returns:
    --------
    float or ndarray
        Absorption coefficient in cm⁻¹
    """
    T_safe = 1e-2  # Minimum temperature to avoid singularity
    T_use = np.maximum(T, T_safe)
    return np.minimum(1000.0* rho * ((T_use)**(-1/2)) * (E)**(-3.0),1e6)
    #return 300*T_use**(-3.0)
    #return 1e4*np.exp(-E/T_use)*(T_use)**(-3)


def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    """Create opacity function for a group using geometric mean at boundaries
    
    For group g with boundaries [E_low, E_high]:
    σ_{a,g}(T) = sqrt(σ_a(T, E_low) * σ_a(T, E_high))
    
    Parameters:
    -----------
    E_low : float
        Lower energy boundary (keV)
    E_high : float
        Upper energy boundary (keV)
    rho : float
        Density (g/cm³)
    
    Returns:
    --------
    function
        Opacity function σ_a(T, r) for this group
    """
    def opacity_func(T, r):
        """Group opacity using geometric mean of boundaries"""
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    """Create diffusion coefficient function for a group
    
    D_g(T) = 1/(3*σ_{a,g}(T))
    
    Parameters:
    -----------
    E_low : float
        Lower energy boundary (keV)
    E_high : float
        Upper energy boundary (keV)
    rho : float
        Density (g/cm³)
    
    Returns:
    --------
    function
        Diffusion coefficient D(T, r) for this group
    """
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    
    def diffusion_func(T, r):
        """Diffusion coefficient: D = 1/(3σ_R)"""
        sigma = opacity_func(T, r)
        return C_LIGHT / (3.0 * sigma)
    
    return diffusion_func


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_multigroup_powerlaw(use_preconditioner=False, n_groups=10, 
                                         time_dependent_bc=True):
    """Run multigroup Marshak wave simulation with power-law opacity
    
    Parameters:
    -----------
    use_preconditioner : bool
        Use LMFG (Linear Multifrequency Gray) preconditioning for GMRES
    n_groups : int
        Number of energy groups (default: 10)
    time_dependent_bc : bool
        If True, boundary temperature varies with time (default: False)
    """
    
    print("="*80)
    print(f"Marshak Wave Problem - Multigroup ({n_groups} Groups) with Power-Law Opacity")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_a(T,E) = 10.0 * ρ * T^{-1/2} * E^{-3} (cm^-1)")
    print("  Group opacity: geometric mean at group boundaries")
    print("  Density: ρ = 1.0 g/cm³")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Marshak boundary condition")
    print("  Right BC: Zero flux")
    if use_preconditioner:
        print("  GMRES: Using LMFG preconditioning")
    else:
        print("  GMRES: No preconditioning")
    print("="*80)
    
    # Problem setup
    r_min = 0.0      # cm
    r_max = 10.0      # cm
    n_cells = 50     # Reasonable resolution
    
    # Energy group structure (keV) - logarithmically spaced
    # Range from 1e-4 keV to 25.0 keV
    energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), n_groups + 1)
    
    # Time stepping parameters
    dt = 0.01        # ns - timestep
    target_times = [1.0]#, 5.0,20.]#, 10.0, 20.0]  # ns
    
    # Material properties
    rho = RHO  # g/cm³
    cv = 0.05 / rho  # GJ/(g·keV) - realistic value
    
    # Boundary conditions
    T_bc = 0.5  # keV (blackbody temperature at left boundary)
    
    # Create opacity and diffusion coefficient functions for each group
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, rho))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, rho))
    
    # For incoming blackbody BC, use Planck-group fractions (not sigma-weighted dB/dT fractions)
    B_g_bc = Bg_multigroup(energy_edges, T_bc)
    chi = B_g_bc / B_g_bc.sum()
    
    print(f"\nEnergy group edges (keV): {energy_edges}")
    print(f"Emission fractions at T = {T_bc} keV:")
    for g in range(n_groups):
        sigma_g = sigma_funcs[g](T_bc, 0.0)
        print(f"  Group {g:2d} [{energy_edges[g]:6.3f}, {energy_edges[g+1]:6.3f}] keV: "
              f"χ = {chi[g]:.6f}, σ_a = {sigma_g:.3e} cm^-1")
    
    # Compute incoming flux for boundary condition
    # For blackbody: F = (a*c*T^4)/2
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    F_g_values = [chi[g] * F_total for g in range(n_groups)]
    
    # Robin BC parameters: A*phi + B*dphi/dr = C
    # For Marshak BC in this solver: phi_g/2 + (2D_g)*dphi_g/dr = F_in,g
    # Compute group-dependent diffusion coefficient for BC (at left boundary, T ≈ T_bc)
    BC_A = 0.5
    BC_B_values = []
    for g in range(n_groups):
        D_g_bc = diff_funcs[g](T_bc, 0.0)
        BC_B_values.append(2.0 * D_g_bc)
    BC_C_values = F_g_values.copy()
    
    # Create mutable container for time-dependent BCs
    # The BC functions will reference this, and we can update it during time stepping
    bc_params = {
        'B_values': BC_B_values.copy(),
        'C_values': BC_C_values.copy()
    }
    
    print(f"\nBoundary condition (Marshak):")
    print(f"  A = {BC_A}")
    print(f"  B_g = D_g(T_bc): group-dependent diffusion coefficients")
    print(f"  F_total(T_b) = {F_total:.6e} GJ/(cm²·ns)")
    if time_dependent_bc:
        print(f"  Time-dependent BC: T_bc will ramp from 0.05 keV to 0.5 keV over 20 ns")
    for g in range(n_groups):
        print(f"    Group {g:2d}: B = {BC_B_values[g]:.6e} cm, C = {BC_C_values[g]:.6e} GJ/(cm²·ns)")
    
    # Define boundary condition functions
    # These reference bc_params dict, which we can update during time stepping
    def make_left_bc_func(group_idx):
        """Create Robin BC function for left boundary that references mutable bc_params"""
        def left_bc(phi, r):
            return BC_A, bc_params['B_values'][group_idx], bc_params['C_values'][group_idx]
        return left_bc
    
    def right_bc_func(phi, r):
        """Neumann BC at right: ∇φ = 0"""
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [make_left_bc_func(g) for g in range(n_groups)]
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
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    # Initial condition: cold everywhere
    r_centers = solver.r_centers
    T_init = 0.05 * np.ones(n_cells)  # Cold initial state
    #T_init[(r_centers > .25*r_max) & (r_centers < .75*r_max)] = 0.5  # Hot region in the middle
    solver.T = T_init.copy()
    solver.T_old = solver.T.copy()
    solver.E_r = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    # Update time
    solver.t = 0.0
    current_time = 0.0
    
    print(f"Initial conditions: Cold start")
    print(f"  T_init = {T_init.max():.6f} keV (cold)")
    print(f"  E_r_init = {solver.E_r.max():.6e} GJ/cm³")
    
    # Time evolution
    print("\nTime evolution:")
    print(f"{'Step':<6} {'Time(ns)':<10} {'T_max':<12} {'T_min':<12} {'E_r_max':<15} {'Newton':<8} {'GMRES':<8}")
    print("-" * 85)
    
    solutions = []
    step_count = 0
    target_idx = 0
    
    # Time stepping loop
    max_steps = 2500
    while current_time < target_times[-1] and step_count < max_steps:
        # For implicit methods, BC should be evaluated at the NEW time (t + dt)
        bc_time = solver.t + solver.dt
        
        # Update boundary conditions if time-dependent
        if time_dependent_bc:
            # Ramp T_bc from 0.05 keV to 1.0 keV over 1 ns
            T_bc_start = 1.0  # keV
            T_bc_end = 1.0     # keV
            t_ramp = 1.0      # ns
            if bc_time < t_ramp:
                T_bc_current = T_bc_start + (T_bc_end - T_bc_start) * (bc_time / t_ramp)
            else:
                T_bc_current = T_bc_end
            
            # Recompute incoming spectrum fractions from Planck group integrals
            B_g_current = Bg_multigroup(energy_edges, T_bc_current)
            chi_current = B_g_current / B_g_current.sum()
            
            # Recompute group-dependent BC parameters
            F_total_current = (A_RAD * C_LIGHT * T_bc_current**4) / 2.0
            
            # Temperature for evaluating diffusion coefficient: average of boundary and first cell
            T_avg_bc =T_bc_current # 0.5 * (T_bc_current + solver.T[0])
            
            for g in range(n_groups):
                # Update diffusion coefficient (evaluated at average temperature)
                B_g = 2.0 * diff_funcs[g](T_avg_bc, 0.0)
                # Update incoming flux (temperature AND spectrum-dependent)
                C_g = chi_current[g] * F_total_current
                
                # CRITICAL: Directly update the BC function in the solver object!
                # The solver's internal DiffusionOperatorSolver1D objects need the new BC
                def make_updated_bc(B_val, C_val):
                    def left_bc(phi, r):
                        return BC_A, B_val, C_val
                    return left_bc
                
                solver.solvers[g].left_bc_func = make_updated_bc(B_g, C_g)
            
            # Debug: print BC update occasionally
            if step_count % 10 == 0:  # Print every 10 steps instead of 100
                print(f"  [BC update] t={bc_time:.4f} ns, T_bc={T_bc_current:.4f} keV, "
                      f"F_total={F_total_current:.6e}")
        
        # Newton iteration
        info = solver.step(
            max_newton_iter=5,
            newton_tol=1e-6,
            gmres_tol=1e-6,
            gmres_maxiter=200,
            use_preconditioner=use_preconditioner,
            max_relative_change=1.0,
            verbose=False
        )
        
        step_count += 1
        # Update current_time after the step (for printing and saving)
        current_time = solver.t
        
        # Print progress
        if step_count % 5 == 0 or (target_idx < len(target_times) and 
                                    abs(current_time - target_times[target_idx]) < 0.5*dt):
            gmres_iter = info['gmres_info']['iterations']
            print(f"{step_count:<6} {current_time:<10.4f} {solver.T.max():<12.6f} "
                  f"{solver.T.min():<12.6f} {solver.E_r.max():<15.6e} "
                  f"{info['newton_iter']:<8} {gmres_iter:<8}")
        
        # Save solution at target times
        if target_idx < len(target_times) and current_time >= target_times[target_idx]:
            r = solver.r_centers.copy()
            T = solver.T.copy()
            E_r = solver.E_r.copy()
            T_rad = (E_r / A_RAD)**0.25
            
            # Compute group-wise solutions
            solver.update_absorption_coefficients(T)
            solver.fleck_factor = solver.compute_fleck_factor(T)
            xi_g_list = [solver.compute_source_xi(g, T, solver.t) for g in range(n_groups)]
            
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
                'E_r_groups': E_r_groups
            })
            
            print(f"\n>>> Saved solution at t = {current_time:.3f} ns")
            print(f"    T: max = {T.max():.6f} keV, min = {T.min():.6f} keV")
            print(f"    E_r: max = {E_r.max():.6e} GJ/cm³\n")
            
            target_idx += 1
        
        # Advance to next time step
        solver.advance_time()
    
    print("\nSimulation complete!")
    print(f"Total steps: {step_count}")
    print(f"Final time: {current_time:.3f} ns")
    
    # =============================================================================
    # PLOTTING
    # =============================================================================
    
    print("\nGenerating plots...")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green']
    
    for idx, sol in enumerate(solutions):
        color = colors[idx]
        label = f"t = {sol['time']:.1f} ns"
        
        # Temperature profile
        axes[0].plot(sol['r'], sol['T'], color=color, linewidth=2, label=label)
        
        # Radiation temperature
        axes[1].plot(sol['r'], sol['T_rad'], color=color, linewidth=2, label=label)
        
        # Total radiation energy
        axes[2].semilogy(sol['r'], sol['E_r'], color=color, linewidth=2, label=label)
    
    # Format Temperature plot
    axes[0].set_xlabel('Position (cm)', fontsize=12)
    axes[0].set_ylabel('Material Temperature (keV)', fontsize=12)
    axes[0].set_title('Material Temperature Profile', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Format Radiation Temperature plot
    axes[1].set_xlabel('Position (cm)', fontsize=12)
    axes[1].set_ylabel('Radiation Temperature (keV)', fontsize=12)
    axes[1].set_title('Radiation Temperature Profile', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Format Radiation Energy plot
    axes[2].set_xlabel('Position (cm)', fontsize=12)
    axes[2].set_ylabel('Radiation Energy (GJ/cm³)', fontsize=12)
    axes[2].set_title('Radiation Energy Profile', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    precond_str = "with_precond" if use_preconditioner else "no_precond"
    bc_str = "_timeBC" if time_dependent_bc else ""
    filename = f"marshak_wave_multigroup_powerlaw_{n_groups}g_{precond_str}{bc_str}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Show plot
    plt.show()
    
    return solver, solutions


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ========================================================================
    # EASY CONFIGURATION (for running from IDE)
    # Uncomment these lines to override command-line args
    # ========================================================================
    # USE_PRECONDITIONER = False
    # NUM_GROUPS = 10
    # TIME_DEPENDENT_BC = True
    # ========================================================================
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Marshak Wave - Multigroup with Power-Law Opacity')
    parser.add_argument('--precond', action='store_true', 
                       help='Use LMFG preconditioner')
    parser.add_argument('--groups', type=int, default=10,
                       help='Number of energy groups (default: 10)')
    parser.add_argument('--no-time-bc', action='store_true',
                       help='Disable time-dependent boundary condition (default: enabled)')
    
    args = parser.parse_args()
    
    # Override with hardcoded values if defined above
    use_precond = locals().get('USE_PRECONDITIONER', args.precond)
    n_groups = locals().get('NUM_GROUPS', args.groups)
    time_bc = locals().get('TIME_DEPENDENT_BC', not args.no_time_bc)
    
    solver, solutions = run_marshak_wave_multigroup_powerlaw(
        use_preconditioner=use_precond,
        n_groups=n_groups,
        time_dependent_bc=time_bc
    )
