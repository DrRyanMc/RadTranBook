#!/usr/bin/env python3
"""
Marshak Wave Problem with TR-BDF2 Time Integration
Classic radiative heat wave test problem

Problem setup:
- Left boundary: incoming flux from blackbody at 1 keV
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Time integration: TR-BDF2 (composite trapezoidal-BDF2 method)
- Plot solutions at 10, 30, and 50 ns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver, 
    temperature_from_Er, 
    A_RAD, 
    C_LIGHT
)
RHO = 1.0  # g/cm^3 (assumed constant density

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
    return 300.0 * T**(-n)


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
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_trbdf2():
    """Run Marshak wave simulation with TR-BDF2 and plot results at specified times"""
    
    print("="*60)
    print("Marshak Wave Problem (TR-BDF2)")
    print("="*60)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("  Time integration: TR-BDF2")
    print("="*60)
    
    # Problem setup
    r_min = 0.0    # cm
    r_max = 0.5   # cm (adjust as needed for wave propagation)
    n_cells = 100  # Fine mesh for capturing wave front
    
    # Time stepping parameters
    dt = 0.01  # ns (start with small time step)
    target_times = [1.0, 10.0, 20.0]  # ns
    
    # Create solver with Marshak wave properties
    solver = RadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=0,  # Planar geometry
        dt=dt,
        max_newton_iter=50,  # Upper limit on Newton iterations
        newton_tol=1e-8,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc,
        theta=1.0  # Will be overridden by TR-BDF2 stages
    )
    
    # Initial condition: cold material (low temperature)
    def initial_Er(r):
        T_init = 0.01  # keV (cold but not too small to avoid numerical issues)
        return np.full_like(r, A_RAD * T_init**4)
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution using TR-BDF2
    print("\nTime evolution with TR-BDF2:")
    current_time = 0.0
    solutions = []
    
    for target_time in target_times:
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time:
                steps_needed = 1
                temp_dt = target_time - current_time
                solver.dt = temp_dt
            else:
                steps_needed = 1
            
            # Use TR-BDF2 time stepping
            solver.time_step_trbdf2(n_steps=steps_needed, verbose=False)
            current_time += solver.dt
            print(f"  Time: {current_time:.3f} ns")
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        r, Er = solver.get_solution()
        T = temperature_from_Er(Er)
        solutions.append((current_time, r.copy(), Er.copy(), T.copy()))
        print(f"  t = {current_time:.1f} ns: max T = {T.max():.4f} keV, max E_r = {Er.max():.4e} GJ/cm^3")
    
    return solutions, solver


def plot_marshak_wave(solutions):
    """Plot Marshak wave solutions"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red']
    
    #analytic solution
    xi_max = 1.11305
    omega = 0.05989
    self_similar = lambda xi: (xi < xi_max) * np.power((1 - xi/xi_max)*(1+omega*xi/xi_max), 1/6)
    xi_vals = np.linspace(0, xi_max, 200)
    K_const = 8*A_RAD*C_LIGHT/((4+3)*3*300*RHO*0.3)
    
    # Plot temperature profiles
    ax = axes[0]
    for (t, r, Er, T), color in zip(solutions, colors):
        ax.plot(r, T, color=color, linewidth=2, label=f't = {t:.0f} ns (TR-BDF2)')
        #evaluate self-similar solution at this time
        r_ref = xi_vals * (K_const * t)**0.5
        T_ref = self_similar(xi_vals)
        ax.plot(r_ref, T_ref, color=color, linestyle='--', linewidth=1, label=f'SS t={t:.0f} ns')

    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Marshak Wave: Temperature Profiles (TR-BDF2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, solutions[-1][1][-1])
    
    # Plot radiation energy density profiles
    ax = axes[1]
    for (t, r, Er, T), color in zip(solutions, colors):
        ax.plot(r, Er, color=color, linewidth=2, label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Marshak Wave: Radiation Energy Density Profiles (TR-BDF2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, solutions[-1][1][-1])
    
    plt.tight_layout()
    plt.savefig('marshak_wave_trbdf2.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'marshak_wave_trbdf2.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run Marshak wave simulation with TR-BDF2
    print("\nRunning Marshak wave simulation with TR-BDF2...")
    solutions, solver = run_marshak_wave_trbdf2()
    
    # Plot results
    print("\nPlotting Marshak wave results...")
    plot_marshak_wave(solutions)
    
    print("\n" + "="*60)
    print("Marshak wave simulation (TR-BDF2) completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
