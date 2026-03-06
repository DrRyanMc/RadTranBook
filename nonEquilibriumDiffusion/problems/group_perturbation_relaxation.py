#!/usr/bin/env python3
"""
Gaussian Temperature Relaxation Problem - Multigroup with Power-Law Opacity

Problem setup:
- 10 energy groups with power-law absorption coefficient
- Opacity: σ_{a,g}(T,E) = 100,000 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
- Initial condition: Gaussian temperature profile with peak T = 0.5 keV
- Material and radiation in local equilibrium everywhere
- Background temperature: T = 0.05 keV
- Boundary conditions: Neumann (zero flux) on both sides
- Time evolution: dt = 0.01 ns, run until t = 0.1 ns
- Watch the hot Gaussian diffuse and cool
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver_patched_lmfgk import (
    MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, Bg_multigroup, flux_limiter_larsen
)

# Physical constants
RHO = 2.0  # g/cm³

# =============================================================================
# POWER-LAW OPACITY FUNCTIONS (same as Marshak problem)
# =============================================================================

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """Power-law opacity at specific energy
    
    σ_a(T,E) = 100,000 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
    """
    T_safe = 1e-2  # Minimum temperature to avoid singularity
    T_use = np.maximum(T, T_safe)
    return np.minimum(10.0 * rho * (T_use)**(-0.5) * E**(-3.0), 1e24)


def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    """Create opacity function for a group using geometric mean at boundaries"""
    def opacity_func(T, r):
        """Group opacity using geometric mean of boundaries"""
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    """Create diffusion coefficient function for a group"""
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    
    def diffusion_func(T, r):
        """Diffusion coefficient: D = c/(3σ_R)"""
        sigma = opacity_func(T, r)
        return C_LIGHT / (3.0 * sigma)
    
    return diffusion_func


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_gaussian_temperature_relaxation(use_preconditioner=False):
    """Run relaxation problem with Gaussian temperature profile in equilibrium
    
    Parameters:
    -----------
    use_preconditioner : bool
        Use LMFG (Linear Multifrequency Gray) preconditioning for GMRES
    """
    
    print("="*80)
    print("Gaussian Temperature Relaxation - Multigroup with Power-Law Opacity")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_a(T,E) = 100,000 * ρ * T^{-1/2} * E^{-3} (cm^-1)")
    print("  Group opacity: geometric mean at group boundaries")
    print(f"  Density: ρ = {RHO} g/cm³")
    print("  Heat capacity: c_v = 0.05 GJ/(g·keV)")
    print("Boundary conditions:")
    print("  Left BC: Marshak (zero incoming radiation)")
    print("  Right BC: Marshak (zero incoming radiation)")
    print("Initial conditions:")
    print("  Gaussian temperature profile: peak T = 0.5 keV, background T = 0.05 keV")
    print("  Material-radiation local equilibrium everywhere")
    print("Time stepping:")
    
    # Problem setup
    r_min = 0.0      # cm
    r_max = 2.0      # cm
    n_cells = 100     # Spatial resolution
    
    # Energy group structure (keV) - logarithmically spaced
    # Range from 1e-4 keV to 25.0 keV
    n_groups = 10
    energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), n_groups + 1)
    
    # Time stepping parameters (adaptive)
    dt = 1e-4         # ns - initial timestep
    dt_max = 0.01      # ns - maximum timestep
    dt_growth = 1.1   # Growth factor per step
    t_final = 0.1     # ns - final time
    
    print(f"  Adaptive: dt starts at {dt} ns, increases by {dt_growth}x per step, max = {dt_max} ns")
    if use_preconditioner:
        print("  GMRES: Using LMFG preconditioning")
    else:
        print("  GMRES: No preconditioning")
    print("="*80)
    # Material properties
    rho = RHO        # g/cm³
    cv = 0.05  # GJ/(g·keV)
    
    # Initial equilibrium temperature
    T_0 = 0.5  # keV
    
    # Create opacity and diffusion coefficient functions for each group
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, rho))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, rho))
    
    print(f"\nEnergy group edges (keV): {energy_edges}")
    print(f"Equilibrium temperature T_0 = {T_0} keV")
    
    # Marshak boundary conditions with zero incoming radiation
    # BC form: (1/2)φ + 2D·∇φ = 0  =>  A·φ + B·∇φ = C with A=0.5, B=2D, C=0

    T_background = 0.1  # Background temperature (keV)
    left_bc_funcs = []
    right_bc_funcs = []
    # For incoming blackbody BC, use Planck-group fractions (not sigma-weighted dB/dT fractions)
    B_g_bc = Bg_multigroup(energy_edges, T_background)
    chi = B_g_bc / B_g_bc.sum()
    F_total = (A_RAD * C_LIGHT * T_background**4) / 2.0
    F_g_values = [chi[g] * F_total for g in range(n_groups)]
    for g in range(n_groups):
        D_func = diff_funcs[g]
        
        def make_marshak_bc(D_func_bound):
            def marshak_bc(phi, r):
                """Marshak BC: (1/2)φ + 2D·∇φ = 0"""
                # Need to evaluate at boundary - use a representative T
                T_boundary = T_0  # Will be updated during solve
                D = D_func_bound(T_boundary, r)
                return 0.5, 2.0 * D, F_g_values[g]
            return marshak_bc
        
        left_bc_funcs.append(make_marshak_bc(D_func))
        right_bc_funcs.append(make_marshak_bc(D_func))
    
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
        flux_limiter_funcs=flux_limiter_larsen,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    # ==========================================================================
    # INITIAL CONDITIONS: Gaussian Temperature Profile in Equilibrium
    # ==========================================================================
    
    r_centers = solver.r_centers
    n_cells = len(r_centers)
    
    # Material temperature: Gaussian profile with peak at T_0
    r_center = 0.5 * (r_min + r_max)  # Center of domain
    sigma_gaussian = 0.25  # Width of Gaussian (cm)
    
    # Create Gaussian temperature profile: T(r) = T_bg + (T_0 - T_bg) * exp(-(r-r_c)^2/σ^2)
    gaussian = np.exp(-((r_centers - r_center) / sigma_gaussian)**2)
    T_init = T_background + (T_0 - T_background) * gaussian
    
    print(f"\nInitial Gaussian temperature profile:")
    print(f"  Peak temperature (center): T_max = {T_0} keV")
    print(f"  Background temperature: T_min = {T_background} keV")
    print(f"  Center position: r = {r_center} cm")
    print(f"  Gaussian width: σ = {sigma_gaussian} cm")
    
    # Initialize group energy densities in LOCAL equilibrium at each position
    E_r_groups_init = np.zeros((n_groups, n_cells))
    E_r_init = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Local equilibrium radiation energy at this temperature
        E_r_local = A_RAD * T_init[i]**4
        E_r_init[i] = E_r_local
        
        # Compute local Planck fractions at this temperature
        B_g_local = Bg_multigroup(energy_edges, T_init[i])
        chi_local = B_g_local / B_g_local.sum()
        
        # Distribute radiation energy among groups according to local Planck fractions
        for g in range(n_groups):
            E_r_groups_init[g, i] = chi_local[g] * E_r_local
    
    print(f"\nRadiation energy ranges:")
    print(f"  Total E_r: min = {E_r_init.min():.6e}, max = {E_r_init.max():.6e} GJ/cm³")
    for g in range(n_groups):
        print(f"  Group {g}: min = {E_r_groups_init[g].min():.6e}, "
              f"max = {E_r_groups_init[g].max():.6e} GJ/cm³")
    
    # Store fractional distribution (for source term computation)
    phi_total_init = E_r_init * C_LIGHT
    phi_g_fraction_init = np.zeros((n_groups, n_cells))
    for g in range(n_groups):
        phi_g_fraction_init[g, :] = E_r_groups_init[g, :] / (E_r_init + 1e-100)
    
    # Set initial conditions in solver
    solver.T = T_init.copy()
    solver.T_old = solver.T.copy()
    solver.E_r = E_r_init.copy()
    solver.E_r_old = solver.E_r.copy()
    solver.phi_g_fraction = phi_g_fraction_init.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    # Initialize phi_g_stored from energy densities
    for g in range(n_groups):
        solver.phi_g_stored[g, :] = E_r_groups_init[g, :] * C_LIGHT
    
    # Update time
    solver.t = 0.0
    current_time = 0.0
    
    # ==========================================================================
    # TIME EVOLUTION
    # ==========================================================================
    
    print("\nTime evolution:")
    print(f"{'Step':<6} {'Time(ns)':<10} {'dt(ns)':<10} {'T_max':<12} {'T_min':<12} "
          f"{'E_r_max':<15} {'Newton':<8} {'GMRES':<8}")
    print("-" * 95)
    
    solutions = []
    step_count = 0
    
    # Save initial state
    solutions.append({
        'time': 0.0,
        'r': r_centers.copy(),
        'T': T_init.copy(),
        'E_r': E_r_init.copy(),
        'E_r_groups': E_r_groups_init.copy()
    })
    
    # Time stepping loop with adaptive dt
    max_steps = 10000  # Safety limit
    current_dt = dt
    while current_time < t_final and step_count < max_steps:
        # Newton iteration
        info = solver.step(
            max_newton_iter=10,
            newton_tol=1e-6,
            gmres_tol=1e-18,
            gmres_maxiter=200,
            use_preconditioner=use_preconditioner,
            max_relative_change=1.0,
            verbose=False
        )
        
        step_count += 1
        current_time = solver.t
        
        # Print progress every step
        if step_count % 1 == 0:
            gmres_iter = info['gmres_info']['iterations']
            print(f"{step_count:<6} {current_time:<10.4f} {current_dt:<10.3e} {solver.T.max():<12.6f} "
                  f"{solver.T.min():<12.6f} {solver.E_r.max():<15.6e} "
                  f"{info['newton_iter']:<8} {gmres_iter:<8}")
        
        # Save solution every step or at final time
        if step_count % 1 == 0 or current_time >= t_final:
            # Compute all group energies
            solver.update_absorption_coefficients(solver.T)
            solver.fleck_factor = solver.compute_fleck_factor(solver.T)
            xi_g_list = [solver.compute_source_xi(g, solver.T, solver.t) 
                         for g in range(n_groups)]
            
            E_r_groups_current = np.zeros((n_groups, n_cells))
            for g in range(n_groups):
                phi_g = solver.compute_phi_g(g, solver.kappa, solver.T, xi_g_list)
                E_r_groups_current[g, :] = phi_g / C_LIGHT
            
            solutions.append({
                'time': current_time,
                'r': r_centers.copy(),
                'T': solver.T.copy(),
                'E_r': solver.E_r.copy(),
                'E_r_groups': E_r_groups_current.copy()
            })
        
        # Advance to next time step
        solver.advance_time()
        
        # Increase timestep adaptively
        current_dt = min(current_dt * dt_growth, dt_max)
        solver.dt = current_dt
    
    print("\nSimulation complete!")
    print(f"Total steps: {step_count}")
    print(f"Final time: {current_time:.4f} ns")
    
    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    
    print("\nGenerating plots...")
    
    # Select solutions to plot: initial condition + ~4 later snapshots
    if len(solutions) <= 5:
        plot_indices = range(len(solutions))
    else:
        # Always include initial condition, then sample later times
        later_indices = list(range(1, len(solutions)))[::max(1, (len(solutions)-1)//4)]
        plot_indices = [0] + later_indices
    
    solutions_to_plot = [solutions[i] for i in plot_indices]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Material temperature vs time
    ax = axes[0, 0]
    for sol in solutions_to_plot:
        ax.plot(sol['r'], sol['T'], linewidth=2, 
                label=f"t = {sol['time']:.3f} ns")
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Material Temperature (keV)', fontsize=12)
    ax.set_title('Material Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Radiation temperature vs time
    ax = axes[0, 1]
    for sol in solutions_to_plot:
        T_rad = (sol['E_r'] / A_RAD)**0.25
        ax.plot(sol['r'], T_rad, linewidth=2, label=f"t = {sol['time']:.3f} ns")
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Radiation Temperature (keV)', fontsize=12)
    ax.set_title('Radiation Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total radiation energy vs time
    ax = axes[1, 0]
    for sol in solutions_to_plot:
        ax.semilogy(sol['r'], sol['E_r'], linewidth=2, 
                    label=f"t = {sol['time']:.3f} ns")
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Total Radiation Energy (GJ/cm³)', fontsize=12)
    ax.set_title('Total Radiation Energy Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Radiation spectrum at domain center
    ax = axes[1, 1]
    center_idx = n_cells // 2
    
    # Compute group center energies for x-axis
    energy_centers = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    
    # Plot spectrum at different times
    for sol in solutions_to_plot:
        # Actual radiation spectrum
        E_r_spectrum = sol['E_r_groups'][:, center_idx]
        line, = ax.loglog(energy_centers, E_r_spectrum, linewidth=2, marker='o',
                          label=f"t = {sol['time']:.3f} ns")
        
        # Planckian at local radiation temperature using same group structure
        E_r_center = sol['E_r'][center_idx]
        T_rad_center = (E_r_center / A_RAD)**0.25
        B_g_planck = Bg_multigroup(energy_edges, T_rad_center)
        # Normalize the same way as initialization
        chi_planck = B_g_planck / B_g_planck.sum()
        E_r_planck = chi_planck * E_r_center
        
        # Plot Planckian as dashed line with same color
        ax.loglog(energy_centers, E_r_planck, linewidth=1.5, 
                  linestyle='--', color=line.get_color(), alpha=0.7)
    
    ax.set_xlabel('Photon Energy (keV)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density (GJ/cm³)', fontsize=12)
    ax.set_title('Radiation Spectrum at Domain Center', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    precond_str = "with_precond" if use_preconditioner else "no_precond"
    filename = f"gaussian_temperature_relaxation_{precond_str}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    plt.show()
    
    return solver, solutions


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gaussian Temperature Relaxation - Multigroup Non-Equilibrium Diffusion'
    )
    parser.add_argument('--precond', action='store_true', 
                       help='Use LMFG preconditioner')
    
    args = parser.parse_args()
    
    solver, solutions = run_gaussian_temperature_relaxation(
        use_preconditioner=args.precond
    )
