#!/usr/bin/env python3
"""
2D Multigroup Marshak Wave with Power-Law Opacity

Tests directional symmetry by running wave in both x and y directions
with 10 energy groups and power-law opacity.

Problem setup:
- Wave propagates in one direction (x or y)
- Perpendicular direction has reflecting boundaries (zero flux)
- Left boundary: incoming flux from blackbody at 1 keV
- Right boundary: zero flux (Neumann)
- Material opacity: σ_a(T,E) = 100,000 * ρ * T^{-1/2} * E^{-3} (cm^-1)
- 10 logarithmically-spaced energy groups from 0.0001 to 25 keV
- Heat capacity: c_v = 0.05 GJ/(g·keV)
- Density: ρ = 10.0 g/cm³

The two runs (x-direction vs y-direction) should give identical results
when compared along their respective propagation directions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multigroup_diffusion_solver_2d import (
    MultigroupDiffusionSolver2D, C_LIGHT, A_RAD, 
    Bg_multigroup, dBgdT_multigroup
)

# Physical constants
RHO = 10.0  # g/cm³

# =============================================================================
# POWER-LAW OPACITY FUNCTIONS
# =============================================================================

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """Power-law opacity at specific energy
    
    σ_a(T,E) = 100,000 cm⁻¹ * (ρ/(g/cm³)) * (T/keV)^{-1/2} * (E/keV)^{-3}
    """
    T_safe = 1e-2  # Minimum temperature to avoid singularity
    T_use = np.maximum(T, T_safe)
    return np.minimum(100000.0 * rho * (T_use)**(-0.5) * E**(-3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    """Create opacity function for a group using geometric mean at boundaries"""
    def opacity_func(T, x, y):
        """Group opacity using geometric mean of boundaries"""
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func


def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    """Create diffusion coefficient function for a group"""
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    
    def diffusion_func(T, x, y):
        """Diffusion coefficient: D = c/(3σ_R)"""
        sigma = opacity_func(T, x, y)
        return C_LIGHT / (3.0 * sigma)
    
    return diffusion_func


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def make_marshak_bc_func(T_bc, E_low, E_high, rho=1.0):
    """Create Marshak boundary condition function for a group
    
    Marshak BC: (1/2)φ + 2D·∇φ = F_in
    
    where F_in = c * B_g(T_bc) is the incoming radiation flux for this group
    
    Robin BC form: A·φ + B·∇φ = C
    So: A = 0.5, B = 2D_g(T_bc), C = c * B_g(T_bc)
    """
    D_func = make_powerlaw_diffusion_func(E_low, E_high, rho)
    
    def marshak_bc(phi, pos, t):
        """Marshak BC at T_bc"""
        # Diffusion coefficient at boundary
        x, y = pos
        D = D_func(T_bc, x, y)
        
        # Incoming group flux from Planck spectrum
        B_g = Bg_multigroup(np.array([E_low, E_high]), T_bc)[0]
        F_in = C_LIGHT * B_g
        
        return 0.5, 2.0 * D, F_in
    
    return marshak_bc


def bc_zero_flux(phi, pos, t):
    """Zero flux boundary (Neumann): 0·φ + 1·∇φ = 0"""
    return 0.0, 1.0, 0.0


def bc_reflecting(phi, pos, t):
    """Reflecting boundary (zero flux, same as bc_zero_flux)"""
    return 0.0, 1.0, 0.0


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_2d(direction='x'):
    """Run 2D multigroup Marshak wave with wave propagating in x or y direction
    
    Parameters:
    -----------
    direction : str
        'x' for wave propagating in x-direction, 'y' for y-direction
    
    Returns:
    --------
    solutions : list of dict
        Solutions at target times
    solver : MultigroupDiffusionSolver2D
        The solver object
    """
    
    print("="*80)
    print(f"2D Multigroup Marshak Wave - Wave in {direction.upper()}-direction")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_a(T,E) = 100,000 * ρ * T^{-1/2} * E^{-3} (cm^-1)")
    print("  Heat capacity: c_v = 0.05 GJ/(g·keV)")
    print(f"  Density: ρ = {RHO} g/cm³")
    print(f"  Wave direction: {direction}")
    print("="*80)
    
    # Energy group structure (keV) - logarithmically spaced
    n_groups = 10
    energy_edges = np.logspace(np.log10(1e-4), np.log10(25.0), n_groups + 1)
    
    print(f"\nEnergy group edges (keV): {energy_edges}")
    
    # Create opacity and diffusion coefficient functions for each group
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, RHO))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, RHO))
    
    # Create Marshak BC functions for each group
    T_bc = 0.1  # keV - boundary temperature
    left_bc_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        left_bc_funcs.append(make_marshak_bc_func(T_bc, E_low, E_high, RHO))
    
    # Problem setup
    # Wave propagation direction: 0 to 6 cm, reasonable mesh
    # Perpendicular direction: small domain with reflecting BCs, 1 cell
    
    if direction == 'x':
        # Wave propagates in x-direction (left to right)
        x_min, x_max = 0.0, 6.0
        y_min, y_max = 0.0, 0.1  # Narrow in y
        nx_cells = 50
        ny_cells = 3  # 1 cell perpendicular for true 1D behavior
        
        # Boundary conditions
        boundary_funcs = {
            'left': left_bc_funcs,        # x=0: Marshak BC for each group
            'right': [bc_zero_flux] * n_groups,  # x=6: zero flux
            # 'left': [bc_reflecting] * n_groups,        # x=0: Marshak BC for each group
            # 'right': [bc_reflecting] * n_groups,  # x=6: zero flux
            'bottom': [bc_reflecting] * n_groups,  # y=0: reflecting
            'top': [bc_reflecting] * n_groups      # y=0.1: reflecting
        }
        
    elif direction == 'y':
        # Wave propagates in y-direction (bottom to top)
        x_min, x_max = 0.0, 0.1  # Narrow in x
        y_min, y_max = 0.0, 6.0
        nx_cells = 3  # 1 cell perpendicular for true 1D behavior
        ny_cells = 50
        
        # Boundary conditions
        boundary_funcs = {
            'left': [bc_reflecting] * n_groups,   # x=0: reflecting
            'right': [bc_reflecting] * n_groups,  # x=0.1: reflecting
            'bottom': left_bc_funcs,              # y=0: Marshak BC for each group
            'top': [bc_zero_flux] * n_groups      # y=6: zero flux
        }
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'x' or 'y'")
    
    # Time stepping parameters
    dt = 0.01  # ns
    target_times = [0.1] #[1.0, 3.0, 5.0]  # ns
    
    # Material properties
    rho = RHO  # g/cm³
    cv = 0.05  # GJ/(g·keV)
    
    # Create solver
    print(f"\nInitializing 2D multigroup solver with {nx_cells} × {ny_cells} = {nx_cells*ny_cells} cells...")
    print(f"  Groups: {n_groups}")
    print(f"  Total unknowns: {n_groups * nx_cells * ny_cells}")
    
    solver = MultigroupDiffusionSolver2D(
        n_groups=n_groups,
        x_min=x_min,
        x_max=x_max,
        nx_cells=nx_cells,
        y_min=y_min,
        y_max=y_max,
        ny_cells=ny_cells,
        energy_edges=energy_edges,
        geometry='cartesian',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        boundary_funcs=boundary_funcs,
        rho=rho,
        cv=cv,
        max_newton_iter=5,  # Increased from 5 - first step often needs more iterations
        newton_tol=1e-6,
        theta=1.0
    )
    
    # Initial condition: cold material in local equilibrium
    T_init = 0.05  # keV
    print(f"\nInitial conditions: T = {T_init} keV (local equilibrium)")
    
    # Initialize group energy densities in equilibrium
    E_r_init = A_RAD * T_init**4
    B_g_init = Bg_multigroup(energy_edges, T_init)
    chi_init = B_g_init / B_g_init.sum()
    E_r_groups_init = chi_init * E_r_init
    
    # Set initial conditions in solver
    solver.T = np.full(solver.n_total, T_init)
    solver.T_old = solver.T.copy()
    solver.E_r = np.full(solver.n_total, E_r_init)
    solver.E_r_old = solver.E_r.copy()
    
    for g in range(n_groups):
        solver.phi_g_stored[g, :] = E_r_groups_init[g] * C_LIGHT
    
    solver.t = 0.0
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    
    # Save initial state
    x_centers = solver.x_centers.copy()
    y_centers = solver.y_centers.copy()
    T_2d = solver.T.reshape(nx_cells, ny_cells).copy()
    E_r_2d = solver.E_r.reshape(nx_cells, ny_cells).copy()
    E_r_groups_3d = np.zeros((n_groups, nx_cells, ny_cells))
    for g in range(n_groups):
        E_r_groups_3d[g, :, :] = (solver.phi_g_stored[g, :] / C_LIGHT).reshape(nx_cells, ny_cells)
    
    T_rad_2d = (E_r_2d / A_RAD)**0.25
    
    solutions.append({
        'time': 0.0,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'T_2d': T_2d,
        'E_r_2d': E_r_2d,
        'T_rad_2d': T_rad_2d,
        'E_r_groups_3d': E_r_groups_3d
    })
    
    print(f"\nInitial state:")
    print(f"  Material: T = {T_2d.min():.4f} to {T_2d.max():.4f} keV")
    print(f"  Radiation: T_rad = {T_rad_2d.min():.4f} to {T_rad_2d.max():.4f} keV")
    print(f"  E_r = {E_r_2d.min():.4e} to {E_r_2d.max():.4e} GJ/cm³")
    
    step_count = 0  # Track time steps for periodic output
    
    for target_time in target_times:
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time:
                temp_dt = target_time - current_time
                solver.dt = temp_dt
            
            step_count += 1
            
            # Take timestep using step method (verbose only for first step)
            info = solver.step(
                verbose=(step_count == 1),
                gmres_tol=1e-6,  # Relaxed from 1e-8: cycling behavior indicates ~1e-6 is best achievable
                gmres_maxiter=200,
                use_preconditioner=False
            )
            
            current_time = solver.t
            
            # Print summary on first step and every 10 time steps
            if step_count == 1 or step_count % 10 == 0:
                T_min = solver.T.min()
                T_max = solver.T.max()
                newton_iters = info.get('newton_iterations', 0)
                total_gmres_iters = info.get('total_gmres_iterations', 0)
                print(f"  Step {step_count}: t={current_time:.4e} ns, T=[{T_min:.4f}, {T_max:.4f}] keV, "
                      f"Newton={newton_iters}, Total GMRES={total_gmres_iters}")
            
            # # Check for convergence failure
            # if not info['converged']:
            #     print(f"\n*** WARNING: Newton iteration did not converge at t={current_time:.4e} ns ***")
            #     print(f"    Final residual: {info.get('final_residual', 'N/A')}")
            #     print(f"    Iterations: {info['newton_iterations']}")
                
            #     # Check for negative or NaN temperatures
            #     if np.any(solver.T < 0) or np.any(np.isnan(solver.T)):
            #         print(f"    ERROR: Negative or NaN temperatures detected!")
            #         print(f"    Temperature range: [{solver.T.min():.4f}, {solver.T.max():.4f}] keV")
            #         raise RuntimeError("Solver diverged: negative or NaN temperatures")
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        x_centers = solver.x_centers.copy()
        y_centers = solver.y_centers.copy()
        T_2d = solver.T.reshape(nx_cells, ny_cells).copy()
        E_r_2d = solver.E_r.reshape(nx_cells, ny_cells).copy()
        
        # Extract group energies
        E_r_groups_3d = np.zeros((n_groups, nx_cells, ny_cells))
        for g in range(n_groups):
            E_r_groups_3d[g, :, :] = (solver.phi_g_stored[g, :] / C_LIGHT).reshape(nx_cells, ny_cells)
        
        T_rad_2d = (E_r_2d / A_RAD)**0.25
        
        solutions.append({
            'time': current_time,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'T_2d': T_2d,
            'E_r_2d': E_r_2d,
            'T_rad_2d': T_rad_2d,
            'E_r_groups_3d': E_r_groups_3d
        })
        
        print(f"  t = {current_time:.1f} ns:")
        print(f"    Material: max T = {T_2d.max():.4f} keV, min T = {T_2d.min():.4f} keV")
        print(f"    Radiation: max T_rad = {T_rad_2d.max():.4f} keV")
    
    return solutions, solver, direction


def extract_1d_profile(solutions, direction):
    """Extract 1D profile along wave direction from middle of perpendicular direction
    
    Parameters:
    -----------
    solutions : list of dict
        2D solutions at different times
    direction : str
        'x' or 'y'
    
    Returns:
    --------
    profiles : list of dict
        1D profiles along wave direction
    """
    profiles = []
    
    for sol in solutions:
        if direction == 'x':
            # Extract 1D profile in x (only 1 cell in y)
            r = sol['x_centers']
            T = sol['T_2d'][:, 0]
            T_rad = sol['T_rad_2d'][:, 0]
            E_r = sol['E_r_2d'][:, 0]
            E_r_groups = sol['E_r_groups_3d'][:, :, 0]  # (n_groups, nx_cells)
        else:  # direction == 'y'
            # Extract 1D profile in y (only 1 cell in x)
            r = sol['y_centers']
            T = sol['T_2d'][0, :]
            T_rad = sol['T_rad_2d'][0, :]
            E_r = sol['E_r_2d'][0, :]
            E_r_groups = sol['E_r_groups_3d'][:, 0, :]  # (n_groups, ny_cells)
        
        profiles.append({
            'time': sol['time'],
            'r': r,
            'T': T,
            'T_rad': T_rad,
            'E_r': E_r,
            'E_r_groups': E_r_groups
        })
    
    return profiles


def plot_comparison(profiles_x, profiles_y, energy_edges):
    """Plot comparison of x-direction and y-direction Marshak waves
    
    Parameters:
    -----------
    profiles_x : list of dict
        1D profiles from x-direction wave
    profiles_y : list of dict
        1D profiles from y-direction wave
    energy_edges : ndarray
        Energy group edges (keV)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'red', 'green']
    times = [prof['time'] for prof in profiles_x]
    
    for idx, (prof_x, prof_y, color) in enumerate(zip(profiles_x, profiles_y, colors)):
        t = prof_x['time']
        
        # Plot 1: Material temperature - X direction
        ax = axes[0, 0]
        ax.plot(prof_x['r'], prof_x['T'], color=color, linewidth=2, 
                linestyle='-', label=f't = {t:.0f} ns (X)')
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material T: X-direction Wave', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 2: Material temperature - Y direction
        ax = axes[0, 1]
        ax.plot(prof_y['r'], prof_y['T'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (Y)')
        ax.set_xlabel('Position y (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material T: Y-direction Wave', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 3: Comparison - overlay both directions
        ax = axes[0, 2]
        ax.plot(prof_x['r'], prof_x['T'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (X)', alpha=0.7)
        ax.plot(prof_y['r'], prof_y['T'], color=color, linewidth=2,
                linestyle='--', label=f't = {t:.0f} ns (Y)', alpha=0.7)
        ax.set_xlabel('Position (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('X vs Y Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        
        # Plot 4: Radiation temperature - X direction
        ax = axes[1, 0]
        ax.plot(prof_x['r'], prof_x['T_rad'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (X)')
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('$T_{rad}$ (keV)', fontsize=11)
        ax.set_title('Radiation T: X-direction Wave', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 5: Radiation temperature - Y direction
        ax = axes[1, 1]
        ax.plot(prof_y['r'], prof_y['T_rad'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (Y)')
        ax.set_xlabel('Position y (cm)', fontsize=11)
        ax.set_ylabel('$T_{rad}$ (keV)', fontsize=11)
        ax.set_title('Radiation T: Y-direction Wave', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 6: Difference between X and Y (last time only)
        if idx == len(profiles_x) - 1:
            ax = axes[1, 2]
            
            # Interpolate to common grid for difference
            r_common = np.linspace(0, 6.0, 200)
            T_x_interp = np.interp(r_common, prof_x['r'], prof_x['T'])
            T_y_interp = np.interp(r_common, prof_y['r'], prof_y['T'])
            T_diff = T_x_interp - T_y_interp
            
            ax.plot(r_common, T_diff, 'k-', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Position (cm)', fontsize=11)
            ax.set_ylabel('$T_X - T_Y$ (keV)', fontsize=11)
            ax.set_title(f'Difference at t = {t:.0f} ns', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.suptitle('2D Multigroup Marshak Wave: Directional Symmetry Test', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('marshak_wave_multigroup_powerlaw_2d_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'marshak_wave_multigroup_powerlaw_2d_comparison.png'")


def compute_differences(profiles_x, profiles_y):
    """Compute quantitative differences between x and y direction results"""
    print("\n" + "="*80)
    print("Quantitative Comparison: X-direction vs Y-direction")
    print("="*80)
    
    for prof_x, prof_y in zip(profiles_x, profiles_y):
        t = prof_x['time']
        
        # Interpolate to common grid
        r_common = np.linspace(0, 6.0, 200)
        T_x_interp = np.interp(r_common, prof_x['r'], prof_x['T'])
        T_y_interp = np.interp(r_common, prof_y['r'], prof_y['T'])
        
        # Compute differences
        abs_diff = np.abs(T_x_interp - T_y_interp)
        rel_diff = abs_diff / (np.maximum(T_x_interp, T_y_interp) + 1e-10)
        
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        print(f"\nt = {t:.1f} ns:")
        print(f"  Max absolute difference: {max_abs_diff:.6e} keV")
        print(f"  Mean absolute difference: {mean_abs_diff:.6e} keV")
        print(f"  Max relative difference: {max_rel_diff:.6e}")
        print(f"  Mean relative difference: {mean_rel_diff:.6e}")
        
        # Check if differences are small (validation)
        if max_abs_diff < 1e-3 and max_rel_diff < 0.01:
            print(f"  ✓ X and Y directions agree well")
        else:
            print(f"  ✗ WARNING: Significant differences detected")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("2D MULTIGROUP MARSHAK WAVE - DIRECTIONAL SYMMETRY TEST")
    print("="*80)
    print("\nThis test runs the multigroup Marshak wave in both X and Y directions")
    print("with reflecting boundaries in the perpendicular direction.")
    print("The results should be identical, demonstrating directional symmetry.")
    print("="*80)
    
    # Energy group structure
    n_groups = 10
    energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), n_groups + 1)
    
    # Run in X direction
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "PART 1: X-DIRECTION WAVE" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    solutions_x, solver_x, _ = run_marshak_wave_2d(direction='x')
    profiles_x = extract_1d_profile(solutions_x, 'x')
    
    # Run in Y direction
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "PART 2: Y-DIRECTION WAVE" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    solutions_y, solver_y, _ = run_marshak_wave_2d(direction='y')
    profiles_y = extract_1d_profile(solutions_y, 'y')
    
    # Compare results
    compute_differences(profiles_x, profiles_y)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(profiles_x, profiles_y, energy_edges)
    
    print("\n" + "="*80)
    print("2D multigroup Marshak wave directional symmetry test completed!")
    print("="*80)


if __name__ == "__main__":
    main()
