"""
Example: 2D Heterogeneous Material Problem
============================================

This example demonstrates how to use spatially-varying material properties
and time-dependent boundary conditions using the enhanced twoDFV.py solver.

Features demonstrated:
1. Spatially-varying opacity (heterogeneous material)
2. Spatially-varying heat capacity
3. Time-dependent boundary conditions (pulsed source)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Physical constants (from twoDFV.py)
A_RAD = 0.01372  # Radiation constant (GJ/cm³/keV⁴)
C_LIGHT = 29.9792458  # Speed of light (cm/ns)
RHO = 1.0  # Density (g/cm³)

def main():
    print("="*70)
    print("2D Heterogeneous Material Problem")
    print("="*70)
    
    # Problem parameters
    nx, ny = 30, 30
    x_min, x_max = 0.0, 1.0  # cm
    y_min, y_max = 0.0, 1.0  # cm
    
    dt = 0.001  # ns
    t_final = 0.1  # ns
    n_steps = int(t_final / dt)
    
    print(f"\nGrid: {nx} × {ny} cells")
    print(f"Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}] cm")
    print(f"Time: Δt = {dt} ns, t_final = {t_final} ns, n_steps = {n_steps}")
    
    # Define heterogeneous material properties
    # Region 1 (left half, x < 0.5): high opacity, low heat capacity
    # Region 2 (right half, x >= 0.5): low opacity, high heat capacity
    
    SIGMA_LEFT = 10.0  # cm⁻¹ (optically thick)
    SIGMA_RIGHT = 1.0   # cm⁻¹ (optically thin)
    CV_LEFT = 0.01      # GJ/cm³/keV (low heat capacity)
    CV_RIGHT = 0.1      # GJ/cm³/keV (high heat capacity)
    
    def rosseland_opacity(T, x, y):
        """Piecewise constant Rosseland opacity"""
        if x < 0.5:
            return SIGMA_LEFT
        else:
            return SIGMA_RIGHT
    
    def planck_opacity(T, x, y):
        """Piecewise constant Planck opacity"""
        if x < 0.5:
            return SIGMA_LEFT
        else:
            return SIGMA_RIGHT
    
    def specific_heat(T, x, y):
        """Piecewise constant specific heat"""
        if x < 0.5:
            return CV_LEFT / RHO
        else:
            return CV_RIGHT / RHO
    
    def material_energy(T, x, y):
        """Material energy density: e = ρ·c_v·T"""
        cv = specific_heat(T, x, y)
        return RHO * cv * T
    
    def inverse_material_energy(e, x, y):
        """Inverse: T from e"""
        cv = specific_heat(0.0, x, y)  # T doesn't matter for constant c_v
        return e / (RHO * cv)
    
    # Define time-dependent boundary condition (pulsed source)
    T_SOURCE = 2.0  # keV (hot source)
    T_PULSE_END = 0.05  # ns (pulse duration)
    
    def left_boundary(phi, pos, t, boundary='left', geometry='cartesian'):
        """Time-dependent Dirichlet BC on left boundary (pulsed source)"""
        # Robin BC: A*φ + B*(n·∇φ) = C
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet
        
        # Pulse: hot for t < T_PULSE_END, then turn off
        if t < T_PULSE_END:
            T_bc = T_SOURCE
        else:
            T_bc = 0.1  # Background temperature
        
        phi_bc = A_RAD * T_bc**4 * C_LIGHT
        C_bc = phi_bc
        return A_bc, B_bc, C_bc
    
    def zero_flux_boundary(phi, pos, t, boundary='right', geometry='cartesian'):
        """Zero flux boundary (reflecting)"""
        A_bc = 0.0
        B_bc = 1.0  # Neumann
        C_bc = 0.0  # Zero flux
        return A_bc, B_bc, C_bc
    
    boundary_funcs = {
        'left': left_boundary,
        'right': zero_flux_boundary,
        'bottom': zero_flux_boundary,
        'top': zero_flux_boundary
    }
    
    print("\nMaterial properties:")
    print(f"  Left region (x < 0.5):")
    print(f"    σ_P = σ_R = {SIGMA_LEFT} cm⁻¹ (optically thick)")
    print(f"    C_v = {CV_LEFT} GJ/cm³/keV (low heat capacity)")
    print(f"  Right region (x >= 0.5):")
    print(f"    σ_P = σ_R = {SIGMA_RIGHT} cm⁻¹ (optically thin)")
    print(f"    C_v = {CV_RIGHT} GJ/cm³/keV (high heat capacity)")
    
    print("\nBoundary conditions:")
    print(f"  Left: Time-dependent Dirichlet (pulsed source)")
    print(f"    T = {T_SOURCE} keV for t < {T_PULSE_END} ns")
    print(f"    T = 0.1 keV for t >= {T_PULSE_END} ns")
    print(f"  Right, Bottom, Top: Zero flux (reflecting)")
    
    # Initialize solver
    print("\nInitializing solver...")
    solver = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=x_min, x_max=x_max, nx_cells=nx,
        y_min=y_min, y_max=y_max, ny_cells=ny,
        geometry='cartesian', dt=dt,
        rosseland_opacity_func=rosseland_opacity,
        planck_opacity_func=planck_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        inverse_material_energy_func=inverse_material_energy,
        boundary_funcs=boundary_funcs,
        theta=1.0  # Implicit Euler
    )
    
    # Set initial conditions (cold start)
    T_init = 0.1  # keV
    phi_init = A_RAD * T_init**4 * C_LIGHT
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    print(f"\nInitial conditions: T = {T_init} keV (uniform)")
    
    # Time evolution
    print("\nTime evolution...")
    
    # Store snapshots at different times
    times = [0.0, 0.025, 0.05, 0.075, 0.1]
    snapshots = {'t': [], 'T': [], 'phi': []}
    
    step = 0
    while solver.current_time < t_final - 0.5*dt:
        # Save snapshot before time step
        if any(abs(solver.current_time - t_snap) < 0.5*dt for t_snap in times):
            snapshots['t'].append(solver.current_time)
            snapshots['T'].append(solver.get_T_2d().copy())
            snapshots['phi'].append(solver.get_phi_2d().copy())
            print(f"  Snapshot at t = {solver.current_time:.4f} ns")
        
        # Take time step
        solver.time_step(n_steps=1, verbose=False)
        step += 1
        
        if step % 10 == 0:
            T_max = np.max(solver.T)
            T_min = np.min(solver.T)
            print(f"  Step {step}/{n_steps}: t = {solver.current_time:.4f} ns, "
                  f"T ∈ [{T_min:.3f}, {T_max:.3f}] keV")
    
    # Final snapshot
    snapshots['t'].append(solver.current_time)
    snapshots['T'].append(solver.get_T_2d().copy())
    snapshots['phi'].append(solver.get_phi_2d().copy())
    print(f"  Final snapshot at t = {solver.current_time:.4f} ns")
    
    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, len(snapshots['t']), figsize=(16, 7))
    
    x_centers = solver.x_centers
    y_centers = solver.y_centers
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    for i, (t, T_2d, phi_2d) in enumerate(zip(snapshots['t'], snapshots['T'], snapshots['phi'])):
        # Plot temperature
        ax = axes[0, i]
        im = ax.contourf(X, Y, T_2d, levels=20, cmap='hot')
        ax.axvline(x=0.5, color='white', linestyle='--', linewidth=1, alpha=0.7, label='Material interface')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f't = {t:.3f} ns')
        ax.set_aspect('equal')
        if i == 0:
            ax.text(0.02, 0.98, 'Temperature (keV)', transform=ax.transAxes,
                   verticalalignment='top', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        plt.colorbar(im, ax=ax)
        
        # Plot radiation temperature
        ax = axes[1, i]
        T_rad = (phi_2d / (A_RAD * C_LIGHT))**0.25
        im = ax.contourf(X, Y, T_rad, levels=20, cmap='plasma')
        ax.axvline(x=0.5, color='white', linestyle='--', linewidth=1, alpha=0.7, label='Material interface')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_aspect('equal')
        if i == 0:
            ax.text(0.02, 0.98, 'Radiation Temperature (keV)', transform=ax.transAxes,
                  verticalalignment='top', fontsize=10, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('heterogeneous_2d_solution.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: heterogeneous_2d_solution.png")
    
    # Plot centerline profiles at final time
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Centerline along x (at y = 0.5)
    j_center = ny // 2
    T_final = snapshots['T'][-1]
    phi_final = snapshots['phi'][-1]
    T_rad_final = (phi_final / (A_RAD * C_LIGHT))**0.25
    
    ax = axes[0]
    ax.plot(x_centers, T_final[:, j_center], 'b-', linewidth=2, label='Material T')
    ax.plot(x_centers, T_rad_final[:, j_center], 'r--', linewidth=2, label='Radiation T')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(0.25, ax.get_ylim()[1]*0.9, 'High σ\nLow $C_v$', ha='center', fontsize=10)
    ax.text(0.75, ax.get_ylim()[1]*0.9, 'Low σ\nHigh $C_v$', ha='center', fontsize=10)
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title(f'Centerline Profile at t = {snapshots["t"][-1]:.3f} ns', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Time evolution at specific points
    ax = axes[1]
    # Left region (x=0.25, y=0.5)
    i_left = nx // 4
    T_left = [T_2d[i_left, j_center] for T_2d in snapshots['T']]
    # Right region (x=0.75, y=0.5)
    i_right = 3 * nx // 4
    T_right = [T_2d[i_right, j_center] for T_2d in snapshots['T']]
    
    ax.plot(snapshots['t'], T_left, 'bo-', linewidth=2, label=f'x=0.25 (high σ, low $C_v$)')
    ax.plot(snapshots['t'], T_right, 'rs-', linewidth=2, label=f'x=0.75 (low σ, high $C_v$)')
    ax.axvline(x=T_PULSE_END, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Pulse ends')
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Material Temperature (keV)', fontsize=12)
    ax.set_title('Temperature Evolution at Two Points', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heterogeneous_2d_profiles.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: heterogeneous_2d_profiles.png")
    
    print("\n" + "="*70)
    print("Simulation complete!")
    print("="*70)
    print(f"\nKey observations:")
    print(f"  - Left region (high opacity): radiation diffuses slowly")
    print(f"  - Right region (low opacity): radiation propagates faster")
    print(f"  - Pulsed boundary source turns off at t = {T_PULSE_END} ns")
    print(f"  - Different heat capacities affect temperature rise rates")

if __name__ == '__main__':
    main()
