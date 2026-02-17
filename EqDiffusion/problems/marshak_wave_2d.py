#!/usr/bin/env python3
"""
2D Marshak Wave Problem - 1D physics in 2D geometry
Classic radiative heat wave test problem using 2D solver with Cartesian geometry

Problem setup:
- One z boundary (z_min): Dirichlet BC at T = 1 keV
- Other boundaries: reflecting (zero flux)
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Cartesian geometry with x dimension very large (effectively 1D in z)
- Plot solutions at specified times and compare with self-similar solution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from twoDFV import (
    RadiationDiffusionSolver2D,
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

def marshak_bc_left_x(Er_boundary, x_val, z_val, geometry='cartesian'):
    """Left x boundary: reflecting (zero flux)
    
    Robin BC: A*E_r + B*(dE_r/dx) = C
    Zero flux: 0*E_r + 1*dE_r/dx = 0
    """
    return 0.0, 1.0, 0.0  # Reflecting boundary


def marshak_bc_right_x(Er_boundary, x_val, z_val, geometry='cartesian'):
    """Right x boundary: reflecting (zero flux)"""
    return 0.0, 1.0, 0.0  # Reflecting boundary


def marshak_bc_bottom_z(Er_boundary, x_val, z_val, geometry='cartesian'):
    """Bottom z boundary (z_min): Dirichlet at T = 1 keV
    
    For a blackbody boundary at T_bc = 1 keV:
    E_r = a*T_bc^4 at boundary (Dirichlet)
    Robin BC: A*E_r + B*(dE_r/dz) = C  ->  1*E_r + 0*dE_r/dz = Er_bc
    """
    T_bc = 1.0  # keV (blackbody temperature)
    Er_bc = A_RAD * T_bc**4  # Radiation energy density at T_bc
    
    # Dirichlet BC: E_r = Er_bc
    return 1.0, 0.0, Er_bc  # A*E_r + B*dE_r/dz = C -> E_r = Er_bc


def marshak_bc_top_z(Er_boundary, x_val, z_val, geometry='cartesian'):
    """Top z boundary (z_max): reflecting (zero flux)"""
    return 0.0, 1.0, 0.0  # Reflecting boundary


# =============================================================================
# SELF-SIMILAR SOLUTION
# =============================================================================

def self_similar_solution(z, t):
    """
    Self-similar solution for Marshak wave
    
    For the parameters:
    - σ_R = 300 * T^-3
    - c_v = 0.3 GJ/(cm^3·keV)
    - n = 3 (opacity exponent)
    
    Self-similar variable: ξ = z / sqrt(K*t)
    where K = (8*a*c)/(7*3*σ_0*ρ*c_v) for n=3
    
    Temperature profile: T(ξ) = (1 - ξ/ξ_max)^(1/6) * (1 + ω*ξ/ξ_max)^(1/6)
    
    Parameters:
    -----------
    z : ndarray
        Position array (cm)
    t : float
        Time (ns)
    
    Returns:
    --------
    T_ss : ndarray
        Self-similar temperature profile (keV)
    """
    # Self-similar solution parameters for n=3
    xi_max = 1.11305  # Wave front position
    omega = 0.05989   # Correction parameter
    
    # Scaling constant K
    n = 3
    sigma_0 = 300.0
    cv_vol = 0.3
    K_const = (8 * A_RAD * C_LIGHT) / ((4 + n) * 3 * sigma_0 * RHO * cv_vol)
    
    # Self-similar variable
    xi = z / np.sqrt(K_const * t)
    
    # Temperature profile
    T_ss = np.zeros_like(z)
    mask = xi < xi_max
    T_ss[mask] = ((1 - xi[mask]/xi_max) * (1 + omega*xi[mask]/xi_max))**(1.0/6.0)
    
    return T_ss


def plot_self_similar_reference(t_values, z_max=0.5):
    """Plot self-similar solution at specified times"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    z_vals = np.linspace(0, z_max, 200)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for t, color in zip(t_values, colors):
        T_ss = self_similar_solution(z_vals, t)
        ax.plot(z_vals, T_ss, color=color, linestyle='--', linewidth=2, 
                label=f'Self-similar t={t:.0f} ns')
    
    ax.set_xlabel('Position z (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Marshak Wave: Self-Similar Solution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, z_max)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('marshak_wave_2d_selfsimilar.png', dpi=150, bbox_inches='tight')
    print("Self-similar solution plot saved as 'marshak_wave_2d_selfsimilar.png'")


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_2d():
    """Run 2D Marshak wave simulation (effectively 1D in z direction)"""
    
    print("="*70)
    print("2D Marshak Wave Problem (1D physics)")
    print("="*70)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Bottom z BC: Dirichlet at T = 1 keV")
    print("  Other BCs: Zero flux (reflecting)")
    print("="*70)
    
    # Problem setup
    # x direction: very large to make it effectively 1D
    x_min = 0.0
    x_max = 100.0  # Large x extent
    n_x_cells = 6  # Few cells in x (homogeneous solution expected)
    
    # z direction: this is the physical direction
    z_min = 0.0
    z_max = 0.2  # cm (adjust for wave propagation)
    n_z_cells = 200  # Fine mesh for capturing wave front
    
    # Time stepping parameters
    dt = 0.01  # ns (start with small time step)
    target_times = [1.0, 3.0,10.0] #, 10.0, 20.0]  # ns
    
    # Create 2D solver
    solver = RadiationDiffusionSolver2D(
        coord1_min=x_min,
        coord1_max=x_max,
        n1_cells=n_x_cells,
        coord2_min=z_min,
        coord2_max=z_max,
        n2_cells=n_z_cells,
        geometry='cartesian',
        dt=dt,
        max_newton_iter=20,
        newton_tol=1e-8,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_bc_left_x,
        right_bc_func=marshak_bc_right_x,
        bottom_bc_func=marshak_bc_bottom_z,
        top_bc_func=marshak_bc_top_z,
        theta=1.0,  # Implicit Euler for stability
        use_jfnk=False  # Use direct solver
    )
    
    # Initial condition: cold material
    def initial_Er(x, z):
        T_init = 0.005  # keV (cold but not too small)
        return A_RAD * T_init**4
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    
    for target_time in target_times:
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time:
                temp_dt = target_time - current_time
                solver.dt = temp_dt
            else:
                temp_dt = dt
            
            # Take one time step
            try:
                solver.time_step(verbose=False)
                current_time += temp_dt
                print(f"  Time: {current_time:.3f} ns")
            except Exception as e:
                print(f"  Error at t={current_time:.3f} ns: {e}")
                break
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        # Extract 1D profile along center line (middle x value)
        Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
        i_center = solver.n1_cells // 2
        Er_1d = Er_2d[i_center, :]
        T_1d = temperature_from_Er(Er_1d)
        z_centers = solver.coord2_centers
        
        solutions.append((current_time, z_centers.copy(), Er_1d.copy(), T_1d.copy()))
        print(f"  t = {current_time:.1f} ns: max T = {T_1d.max():.4f} keV")
    
    return solutions, solver


def plot_marshak_wave_2d(solutions):
    """Plot 2D Marshak wave solutions and comparison with self-similar"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red']
    
    # Plot temperature profiles
    ax = axes[0]
    for (t, z, Er, T), color in zip(solutions, colors):
        ax.plot(z, T, color=color, linewidth=2, label=f'Numerical t = {t:.0f} ns')
        
        # Overlay self-similar solution
        T_ss = self_similar_solution(z, t)
        ax.plot(z, T_ss, color=color, linestyle='--', linewidth=1, 
                alpha=0.7, label=f'Self-similar t = {t:.0f} ns')
    
    ax.set_xlabel('Position z (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('2D Marshak Wave: Temperature Profiles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xlim(0, solutions[-1][1][-1])
    ax.set_ylim(0, 1.1)
    
    # Plot radiation energy density profiles
    ax = axes[1]
    for (t, z, Er, T), color in zip(solutions, colors):
        ax.plot(z, Er, color=color, linewidth=2, label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position z (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('2D Marshak Wave: Radiation Energy Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(0, solutions[-1][1][-1])
    
    plt.tight_layout()
    plt.savefig('marshak_wave_2d.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'marshak_wave_2d.png'")


def plot_2d_heatmap(solver, time_label):
    """Plot 2D heatmap of temperature distribution"""
    
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    T_2d = temperature_from_Er(Er_2d)
    
    X, Z = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.pcolormesh(X, Z, T_2d, shading='auto', cmap='hot')
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_title(f'Temperature Distribution at t = {time_label}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (keV)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'marshak_wave_2d_heatmap_{time_label}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved as 'marshak_wave_2d_heatmap_{time_label}.png'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Plot self-similar reference solution
    print("\nGenerating self-similar reference solution...")
    plot_self_similar_reference([1.0, 10.0, 20.0, 30.0, 50.0])
    
    # Run 2D Marshak wave simulation
    print("\nRunning 2D Marshak wave simulation...")
    solutions, solver = run_marshak_wave_2d()
    
    # Plot results
    print("\nPlotting 2D Marshak wave results...")
    plot_marshak_wave_2d(solutions)
    
    # Plot 2D heatmaps at final time
    if solutions:
        final_time = solutions[-1][0]
        print(f"\nGenerating 2D heatmap at t = {final_time:.1f} ns...")
        plot_2d_heatmap(solver, f"{final_time:.0f}ns")
    
    print("\n" + "="*70)
    print("2D Marshak wave simulation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
