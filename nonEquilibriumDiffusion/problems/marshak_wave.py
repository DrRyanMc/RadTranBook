#!/usr/bin/env python3
"""
Marshak Wave Problem for Non-Equilibrium Radiation Diffusion
Classic radiative heat wave test problem

Problem setup:
- Left boundary: incoming flux from blackbody at 1 keV
- Material opacity: σ_R = σ_P = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Plot solutions at 1, 10, and 20 ns
"""

import sys
import os
# Add parent directory to path to import oneDFV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import NonEquilibriumRadiationDiffusionSolver
from numba import njit

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³

# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES
# =============================================================================

def marshak_opacity(T):
    """Temperature-dependent opacity: σ = 300 * T^-3
    
    Parameters:
    -----------
    T : float or array
        Temperature (keV)
    
    Returns:
    --------
    sigma : float or array
        Opacity (cm^-1)
    """
    n = 3
    T_min = 0.01  # Minimum temperature to prevent overflow (keV)
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)


def marshak_rosseland_opacity(T):
    """Rosseland opacity: σ_R = 300 * T^-3"""
    return marshak_opacity(T)


def marshak_planck_opacity(T):
    """Planck opacity: σ_P = 300 * T^-3 (same as Rosseland for this problem)"""
    return marshak_opacity(T)


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV) = 0.3/ρ GJ/(g·keV)
    
    Parameters:
    -----------
    T : float or array
        Temperature (keV)
    
    Returns:
    --------
    cv : float or array
        Specific heat capacity per unit mass (GJ/(g·keV))
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T):
    """Material energy density: e = ρ·c_v·T
    
    Parameters:
    -----------
    T : float or array
        Temperature (keV)
    
    Returns:
    --------
    e : float or array
        Material energy density (GJ/cm³)
    """
    cv_specific = marshak_specific_heat(T)
    return RHO * cv_specific * T

@njit
def marshak_inverse_material_energy(e):
    """Inverse: T from e = ρ·c_v·T => T = e/(ρ·c_v)
    
    Parameters:
    -----------
    e : float or array
        Material energy density (GJ/cm³)
    
    Returns:
    --------
    T : float or array
        Temperature (keV)
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return e / cv_volumetric


# =============================================================================
# MARSHAK WAVE BOUNDARY CONDITIONS
# =============================================================================

def marshak_left_bc(phi, x):
    """Left boundary: incoming flux from blackbody at T = 1 keV
    
    For a blackbody boundary, use Dirichlet condition:
    φ = c·E_r = c·a·T_bc^4 at boundary
    
    Returns Robin BC coefficients: A·φ + B·(dφ/dx) = C
    For Dirichlet: A=1, B=0, C=φ_bc
    """
    T_bc = 1.0  # keV (blackbody temperature)
    phi_bc = C_LIGHT * A_RAD * T_bc**4  # φ = c·E_r at T_bc
    
    return 1.0, 0.0, phi_bc  # A·φ + B·dφ/dx = C -> φ = φ_bc


def marshak_right_bc(phi, x):
    """Right boundary: zero incoming flux (zero gradient)
    
    Use zero flux condition: dφ/dx = 0
    This is Robin BC with: 0·φ + 1·(dφ/dx) = 0
    """
    return 0.0, 1.0, 0.0  # A, B, C: 0·φ + 1·dφ/dx = 0


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave():
    """Run Marshak wave simulation and plot results at specified times"""
    
    print("="*80)
    print("Marshak Wave Problem - Non-Equilibrium Radiation Diffusion")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_P = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Blackbody at T = 1 keV")
    print("  Right BC: Zero flux")
    print("="*80)
    
    # Problem setup
    r_min = 0.0      # cm
    r_max = 0.5      # cm (adjust as needed for wave propagation)
    n_cells = 100    # Fine mesh for capturing wave front
    
    # Time stepping parameters
    dt = 0.01        # ns (start with small time step)
    target_times = [1.0, 10.0, 20.0]  # ns
    
    # Create solver with Marshak wave properties
    print(f"\nInitializing solver with {n_cells} cells...")
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=0,  # Planar geometry
        dt=dt,
        max_newton_iter=20,
        newton_tol=1e-8,
        rosseland_opacity_func=marshak_rosseland_opacity,
        planck_opacity_func=marshak_planck_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        inverse_material_energy_func=marshak_inverse_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc,
        theta=1.0  # Backward Euler for stability
    )
    
    # Initial condition: cold material (low temperature)
    T_init = 0.01  # keV (cold but not too small to avoid numerical issues)
    phi_init = C_LIGHT * A_RAD * T_init**4
    solver.phi = np.full(n_cells, phi_init)
    solver.T = np.full(n_cells, T_init)
    
    print(f"Initial conditions: T = {T_init} keV, φ = {phi_init:.6e} GJ/cm³")
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    
    for target_time in target_times:
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            steps_to_target = int(np.ceil((target_time - current_time) / dt))
            
            if current_time + dt > target_time:
                # Last step - adjust dt to hit target exactly
                temp_dt = target_time - current_time
                solver.dt = temp_dt
                steps_needed = 1
            else:
                steps_needed = 1
            
            # Take timestep
            solver.time_step(n_steps=steps_needed, verbose=False)
            current_time += solver.dt
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        r = solver.r_centers.copy()
        phi = solver.phi.copy()
        T = solver.T.copy()
        Er = phi / C_LIGHT
        T_rad = (Er / A_RAD)**0.25
        
        solutions.append({
            'time': current_time,
            'r': r,
            'phi': phi,
            'Er': Er,
            'T': T,
            'T_rad': T_rad
        })
        
        print(f"  t = {current_time:.1f} ns:")
        print(f"    Material: max T = {T.max():.4f} keV, min T = {T.min():.4f} keV")
        print(f"    Radiation: max T_rad = {T_rad.max():.4f} keV, min T_rad = {T_rad.min():.4f} keV")
        print(f"    Max φ = {phi.max():.4e} GJ/cm³")
    
    return solutions, solver


def plot_marshak_wave(solutions):
    """Plot Marshak wave solutions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'green', 'red']
    
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
        ax.plot(r, T, color=color, linewidth=2, label=f't = {t:.0f} ns (material)')
        
        # Self-similar solution
        r_ref = xi_vals * (K_const * t)**0.5
        T_ref = self_similar(xi_vals)
        ax.plot(r_ref, T_ref, color=color, linestyle='--', linewidth=1, alpha=0.7, label=f't = {t:.0f} ns (SS)')
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Material Temperature T (keV)', fontsize=12)
    ax.set_title('Material Temperature Profiles', fontsize=14, fontweight='bold')
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
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Temperature $T_{rad}$ (keV)', fontsize=12)
    ax.set_title('Radiation Temperature Profiles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, solutions[-1]['r'][-1])
    
    # Plot 3: Both temperatures comparison (last time)
    ax = axes[1, 0]
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
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title(f'Temperature Comparison at t = {t:.0f} ns', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, sol['r'][-1])
    
    # Plot 4: Radiation energy density profiles
    ax = axes[1, 1]
    for sol, color in zip(solutions, colors):
        t = sol['time']
        r = sol['r']
        Er = sol['Er']
        ax.plot(r, Er, color=color, linewidth=2, label=f't = {t:.0f} ns')
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Radiation Energy Density Profiles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, solutions[-1]['r'][-1])
    
    plt.tight_layout()
    plt.savefig('marshak_wave_noneq.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'marshak_wave_noneq.png'")


def plot_material_properties():
    """Plot material properties as functions of temperature"""
    
    T_range = np.logspace(-2, 0.5, 200)  # 0.01 to ~3 keV
    
    sigma_R = marshak_rosseland_opacity(T_range)
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
    plt.savefig('marshak_material_properties_noneq.png', dpi=150, bbox_inches='tight')
    print("Material properties plot saved as 'marshak_material_properties_noneq.png'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Plot material properties
    print("\nPlotting material properties...")
    plot_material_properties()
    
    # Run Marshak wave simulation
    print("\nRunning Marshak wave simulation...")
    solutions, solver = run_marshak_wave()
    
    # Plot results
    print("\nPlotting Marshak wave results...")
    plot_marshak_wave(solutions)
    
    print("\n" + "="*80)
    print("Marshak wave simulation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
