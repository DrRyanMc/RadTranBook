#!/usr/bin/env python3
"""
Marshak Wave Problem - Smooth Initial Condition Version
Classic radiative heat wave test problem with smooth initial condition

Problem setup:
- Left boundary: incoming flux from blackbody at 1 keV
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Initial condition: T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .25)])/2
- For testing second-order accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver, 
    temperature_from_Er, 
    A_RAD, 
    C_LIGHT,
    RHO
)


# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES
# =============================================================================

def marshak_opacity(Er):
    """Temperature-dependent Rosseland opacity: σ_R = 300 * T^-3
    
    Parameters:
    -----------
    Er : float
        Radiation energy density (GJ/cm^3)
    
    Returns:
    --------
    sigma_R : float
        Rosseland opacity (cm^-1)
    """
    T = temperature_from_Er(Er)  # keV
    n = 3
    T_min = 0.05  # Minimum temperature to prevent overflow (keV)
    if T < T_min:
        T = T_min
    return 3.0 * T**(-n)


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)
    
    Note: This is volumetric heat capacity, but the solver expects
    specific heat per unit mass. We'll use c_v/ρ.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    cv : float
        Specific heat capacity per unit mass (GJ/(g·keV))
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T):
    """Material energy density: e = c_v * T (volumetric)
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    e : float
        Material energy density (GJ/cm^3)
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric * T


# =============================================================================
# MARSHAK WAVE BOUNDARY CONDITIONS
# =============================================================================

def marshak_left_bc(Er, x):
    """Left boundary: incoming flux from blackbody at T = 1 keV
    
    For a blackbody boundary, we use:
    - Incoming flux: F_in = σ_SB * T_bc^4
    - Boundary condition: F = F_in - D * dE_r/dx
    
    This gives Robin BC: A*E_r + B*(dE_r/dx) = C
    where we relate it to the flux condition.
    
    For radiation: F = -D * dE_r/dx at boundary
    Blackbody BC: F = (c/4) * E_r_bc - c/(4) * E_r (approximately)
    
    More accurately, use: E_r = a*T_bc^4 at boundary (Dirichlet)
    or use flux condition with D * dE_r/dx = c/4 * (E_r_bc - E_r)
    """
    T_bc = 1.0  # keV (blackbody temperature)
    Er_bc = A_RAD * T_bc**4  # Radiation energy density at T_bc
    
    # Use Dirichlet BC: E_r = Er_bc at boundary
    return 1.0, 0.0, Er_bc  # A*E_r + B*dE_r/dx = C -> E_r = Er_bc


def marshak_right_bc(Er, x):
    """Right boundary: zero incoming flux (or zero gradient)
    
    Use zero flux condition: dE_r/dx = 0
    This is Robin BC with: 0*E_r + 1*(dE_r/dx) = 0
    """
    return 0.0, 1.0, 0.0  # A, B, C: 0*E_r + 1*dE_r/dx = 0


# =============================================================================
# SMOOTH INITIAL CONDITION
# =============================================================================

def smooth_initial_temperature(x):
    """Smooth initial temperature profile for second-order accuracy testing
    
    T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .125)])/2
    
    This creates a smooth transition from T=1 keV at x=0 to T=0.2 keV at x=0.5
    
    Parameters:
    -----------
    x : array_like
        Position array (cm)
    
    Returns:
    --------
    T : array_like
        Temperature at each position (keV)
    """
    return 1.0 + (0.2 - 1.0) * (1.0 + np.tanh(50.0 * (x - 0.125))) / 2.0


# =============================================================================
# MARSHAK WAVE SIMULATION WITH SMOOTH INITIAL CONDITION
# =============================================================================

def run_marshak_wave_smooth():
    """Run Marshak wave simulation with smooth initial condition and plot results at specified times"""
    
    print("="*60)
    print("Marshak Wave Problem - Smooth Initial Condition")
    print("="*60)
    print("Material properties:")
    print("  Opacity: σ_R = 3 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("Initial condition:")
    print("  T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .25)])/2")
    print("="*60)
    
    # Problem setup
    r_min = 0.0    # cm
    r_max = 1.0   # cm (adjust as needed for wave propagation)
    n_cells = 400  # Fine mesh for capturing smooth profiles accurately
    
    # Time stepping parameters
    dt = 0.01  # ns (smaller time step for better stability)
    target_times = [1.0, 5.0]  # ns (shorter target times)
    
    # Create solver with Marshak wave properties
    solver = RadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=0,  # Planar geometry
        dt=dt,
        max_newton_iter=20,
        newton_tol=1e-8,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    # Solver configuration
    solver.max_newton_iter_per_step = 1  # Newton iterations per time step
    
    print(f"\nSolver configuration:")
    print(f"  Max Newton iterations: {solver.max_newton_iter_per_step}")
    
    # Smooth initial condition
    def initial_Er(r):
        T_init = smooth_initial_temperature(r)
        return A_RAD * T_init**4
    
    solver.set_initial_condition(initial_Er)
    
    # Display initial condition
    r_initial, Er_initial = solver.get_solution()
    T_initial = temperature_from_Er(Er_initial)
    print(f"\nInitial condition:")
    print(f"  Temperature range: {T_initial.min():.3f} to {T_initial.max():.3f} keV")
    print(f"  Energy density range: {Er_initial.min():.4e} to {Er_initial.max():.4e} GJ/cm^3")
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    
    # Store initial solution
    solutions.append((0.0, r_initial.copy(), Er_initial.copy(), T_initial.copy()))
    
    for target_time in target_times:
        step_count = 0
        damping_count = 0
        
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time:
                steps_needed = 1
                temp_dt = target_time - current_time
                solver.dt = temp_dt
            else:
                steps_needed = 1
            
            solver.time_step(n_steps=steps_needed, verbose=False)
            current_time += solver.dt
            step_count += 1
            
            # Count damping events (approximate from verbose output pattern)
            # Report progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Time: {current_time:.4f} ns (step {step_count})")
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        r, Er = solver.get_solution()
        T = temperature_from_Er(Er)
        solutions.append((current_time, r.copy(), Er.copy(), T.copy()))
        print(f"  t = {current_time:.1f} ns: max T = {T.max():.4f} keV, max E_r = {Er.max():.4e} GJ/cm^3")
    
    return solutions, solver


def plot_marshak_wave_smooth(solutions):
    """Plot Marshak wave solutions with smooth initial condition"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['black', 'blue', 'green', 'red']
    linestyles = ['-', '-', '-', '-']
    
    # Plot temperature profiles
    ax = axes[0]
    for i, (t, r, Er, T) in enumerate(solutions):
        ax.plot(r, T, color=colors[i], linewidth=2, linestyle=linestyles[i], 
                label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Marshak Wave (Smooth Initial Condition): Temperature Profiles', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, solutions[-1][1][-1])
    
    # Plot radiation energy density profiles
    ax = axes[1]
    for i, (t, r, Er, T) in enumerate(solutions):
        ax.plot(r, Er, color=colors[i], linewidth=2, linestyle=linestyles[i], 
                label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Marshak Wave (Smooth Initial Condition): Radiation Energy Density Profiles', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, solutions[-1][1][-1])
    
    plt.tight_layout()
    plt.savefig('marshak_wave_smooth.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'marshak_wave_smooth.png'")
    plt.show()


def plot_initial_condition():
    """Plot the smooth initial condition"""
    
    x = np.linspace(0, 0.5, 1000)
    T_smooth = smooth_initial_temperature(x)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(x, T_smooth, 'b-', linewidth=3, label='T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - 0.25)])/2')
    ax.set_xlabel('Position x (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Smooth Initial Temperature Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig('marshak_initial_condition_smooth.png', dpi=150, bbox_inches='tight')
    print("Initial condition plot saved as 'marshak_initial_condition_smooth.png'")
    plt.show()


def plot_material_properties():
    """Plot material properties as functions of temperature"""
    
    T_range = np.logspace(-2, 0.5, 200)  # 0.01 to ~3 keV
    Er_range = A_RAD * T_range**4
    
    sigma_R = np.array([marshak_opacity(Er) for Er in Er_range])
    D = C_LIGHT / (3.0 * sigma_R)  # Diffusion coefficient
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Opacity vs temperature
    ax = axes[0]
    ax.loglog(T_range, sigma_R, 'b-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Opacity $\\sigma_R$ (cm$^{-1}$)', fontsize=12)
    ax.set_title('Rosseland Opacity: $\\sigma_R = 300 \\cdot T^{-3}$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Diffusion coefficient vs temperature
    ax = axes[1]
    ax.loglog(T_range, D, 'r-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient $D$ (cm²/ns)', fontsize=12)
    ax.set_title('Diffusion Coefficient: $D = c/(3\\sigma_R)$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('marshak_material_properties_smooth.png', dpi=150, bbox_inches='tight')
    print("Material properties plot saved as 'marshak_material_properties_smooth.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Plot initial condition
    print("Plotting smooth initial condition...")
    #plot_initial_condition()
    
    # Plot material properties
    print("\nPlotting material properties...")
    #plot_material_properties()
    
    # Run Marshak wave simulation with smooth initial condition
    print("\nRunning Marshak wave simulation with smooth initial condition...")
    solutions, solver = run_marshak_wave_smooth()
    
    # Plot results
    print("\nPlotting Marshak wave results...")
    plot_marshak_wave_smooth(solutions)
    
    print("\n" + "="*60)
    print("Marshak wave simulation (smooth initial condition) completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()