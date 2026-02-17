#!/usr/bin/env python3
"""
Zeldovich Wave Problem
Radiative heat wave with initial energy pulse in first cell

Problem setup:
- Left boundary: reflecting (zero flux)
- Initial condition: cold everywhere except first cell with E_r = 1/dx
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Plot solutions at multiple times
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from oneDFV import (
    RadiationDiffusionSolver, 
    temperature_from_Er, 
    A_RAD, 
    C_LIGHT
)

from plotfuncs import *
# Add path to import analytical solution
sys.path.insert(0, str(Path.home() / "Dropbox/Apps/Overleaf/RadTranBook/img/equilibriumDiffusion"))
from zeldovich import T_of_r_t

RHO = 1.0  # g/cm^3 (assumed constant density)

# =============================================================================
# ZELDOVICH WAVE MATERIAL PROPERTIES (same as Marshak)
# =============================================================================

def zeldovich_opacity(Er):
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


def zeldovich_specific_heat(T):
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
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def zeldovich_material_energy(T):
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
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return cv_volumetric * T
# def temperature_from_Er(Er):
#     """Convert radiation energy density to temperature: T = (E_r/a)^(1/4)"""
#     return Er/3000

# =============================================================================
# ZELDOVICH WAVE BOUNDARY CONDITIONS
# =============================================================================

def zeldovich_left_bc(Er, x):
    """Left boundary: reflecting (zero flux)
    
    Use zero flux condition: dE_r/dx = 0
    This is Robin BC with: A*E_r + B*(dE_r/dx) = C
    For zero flux: A=0, B=1, C=0
    """
    return 0.0, 1.0, 0.0  # A, B, C: 0*E_r + 1*dE_r/dx = 0


def zeldovich_right_bc(Er, x):
    """Right boundary: zero incoming flux (or zero gradient)
    
    Use zero flux condition: dE_r/dx = 0
    This is Robin BC with: 0*E_r + 1*(dE_r/dx) = 0
    """
    return 0.0, 1.0, 0.0  # A, B, C: 0*E_r + 1*dE_r/dx = 0


# =============================================================================
# ZELDOVICH WAVE SIMULATION
# =============================================================================

def run_zeldovich_wave(d=0,N=1, use_TRBDF2=False):
    """Run Zeldovich wave simulation and plot results at specified times"""
    
    print("="*60)
    print("Zeldovich Wave Problem")
    print("="*60)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Left BC: Reflecting (zero flux)")
    print("  Right BC: Zero flux")
    print("  Initial condition: E_r = 1/dx in first cell, cold elsewhere")
    print("="*60)
    
    # Problem setup
    r_min = 0.0    # cm
    r_max = 3    # cm (adjust as needed for wave propagation)
    n_cells = 100  # Fine mesh for capturing wave front
    
    # Time stepping parameters
    dt = 0.01  # ns (start with small time step)
    target_times = [0.1,0.3,1.0,3.0]#,.1,1.0]  # ns
    # Create solver with Zeldovich wave properties
    solver = RadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=d,  # Planar geometry
        dt=dt,
        max_newton_iter=20,  # Upper limit on Newton iterations
        newton_tol=1e-8,
        rosseland_opacity_func=zeldovich_opacity,
        specific_heat_func=zeldovich_specific_heat,
        material_energy_func=zeldovich_material_energy,
        left_bc_func=zeldovich_left_bc,
        right_bc_func=zeldovich_right_bc,
        theta=1.0  # Time discretization: 1.0=implicit Euler, 0.5=Crank-Nicolson
    )
    
    # Initial condition: cold material except first cell with energy pulse
    def initial_Er(r):
        T_cold = 0.01  # keV (cold but not too small to avoid numerical issues)
        Er_cold = A_RAD * T_cold**4
        
        # Create array of cold values
        Er_init = np.full_like(r, Er_cold)
        
        # Set first cell to have energy = 1/dx
        dx = (r_max - r_min) / n_cells
        factor = 1.0
        if (d>0):
            factor = 2.0
        Etot = factor*(1.0 + 0.000010988*0 + 0*A_RAD*10.988**4)

        Er_init[0] = 0.4*Etot/solver.V_cells[0] # Energy = 1/dx in first cell (at r=0)
        Er_init[1] = (0.05+0.00625)*Etot/solver.V_cells[1]
        Er_init[2] = 0.025*Etot/solver.V_cells[2]
        Er_init[3] = 0.0125*Etot/solver.V_cells[3]
        Er_init[4] = 0.00625*Etot/solver.V_cells[4] 
        #Er_init[2] = 0.05*Etot
        print(f"Setting first cell E_r = {Er_init[0]:.4e} GJ/cm³ (1/dx)")

        #print initial radiation and material energy in first cell
        T_first = temperature_from_Er(Er_init[0])
        print(f"First cell initial temperature = {T_first:.4f} keV")
        print(f"First cell initial material energy = {zeldovich_material_energy(T_first):.4e} GJ/cm³")
        print(f"initial radiation energy density in first cell = {Er_init[n_cells//2]:.4e} GJ/cm³")
        total_energy = np.sum(Er_init) * (r_max - r_min) + np.sum(zeldovich_material_energy(temperature_from_Er(Er_init))) * (r_max - r_min)
        print("\n" + "="*70 + "\n" + f"  t = {0.0:.3f} ns: total energy = {total_energy:.4e} GJ")
        return Er_init
    
    solver.set_initial_condition(initial_Er)
    
    # Print initial condition info
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    total_energy = np.sum(Er) * (r[1]-r[0]) + np.sum(zeldovich_material_energy(T)) * (r[1]-r[0])
    print(f"  t = {0.0:.3f} ns: total energy = {total_energy:.4e} GJ")
    dx = (r_max - r_min) / n_cells
    print(f"\nInitial condition:")
    print(f"  Cell width: dx = {dx:.4f} cm")
    print(f"  First cell E_r = {Er[0]:.4e} GJ/cm³ (= 1/dx)")
    print(f"  First cell T = {T[0]:.4f} keV")
    print(f"  Background E_r = {Er[-1]:.4e} GJ/cm³")
    print(f"  Background T = {T[-1]:.4f} keV")
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    step_count = 0
    target_theta = solver.theta
    for target_time in target_times:
        while current_time < target_time:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time:
                steps_needed = 1
                temp_dt = target_time - current_time
                solver.dt = temp_dt
            else:
                steps_needed = 1
            if use_TRBDF2 and step_count>1:
                # Use TR-BDF2 time stepping
                solver.time_step_trbdf2(n_steps=steps_needed, verbose=False)
            else:
                #if step count is zero make sure theta=1.0 for first step
                if step_count==0:
                    solver.theta=1.0
                else:
                    solver.theta=target_theta
                solver.time_step(n_steps=steps_needed, verbose=False)
            current_time += solver.dt
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
            step_count += 1
        
        # Store solution at target time
        r, Er = solver.get_solution()
        T = temperature_from_Er(Er)
        #check energy conservation
        total_energy = np.sum(Er*solver.V_cells) + np.sum(zeldovich_material_energy(T)*solver.V_cells)
        print(f"  t = {current_time:.3f} ns: total energy = {total_energy:.4e} GJ")
        solutions.append((current_time, r.copy(), Er.copy(), T.copy()))
        print(f"  t = {current_time:.1f} ns: max T = {T.max():.4f} keV, max E_r = {Er.max():.4e} GJ/cm^3, total energy = {total_energy:.4e} GJ")
    
    return solutions, solver


def plot_zeldovich_wave(solutions, include_analytical=True, N=1):
    """Plot Zeldovich wave solutions comparing numerical and analytical
    
    Parameters:
    -----------
    solutions : list of tuples
        List of (time, r, Er, T) numerical solutions
    include_analytical : bool
        If True, overlay analytical self-similar solution
    N : int
        Spatial dimension (1=planar, 2=cylindrical, 3=spherical)
    """
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 8/1.518))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot temperature profiles
    ax = axes
    for i, (t, r, Er, T) in enumerate(solutions):
        color = colors[i % len(colors)]
        # Numerical solution
        ax.plot(r, T, color=color, linewidth=2, linestyle='-', 
                label=f't = {t:.1f} ns (Numerical)', marker='o', markersize=3, markevery=10)
        
        # Analytical solution (planar geometry, N=1)
        if include_analytical:
            try:
                T_analytical, R_front = T_of_r_t(r, t, N=N)
                ax.plot(r, T_analytical, color=color, linewidth=1.5, linestyle='--', 
                        alpha=0.7, label=f't = {t:.1f} ns (Analytical)')
                # Mark wave front
                ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
            except Exception as e:
                print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    #ax.set_title('Zeldovich Wave: Temperature Profiles (Numerical vs Analytical)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0, solutions[-1][1][-1])
    
    # # Plot radiation energy density profiles
    # ax = axes[1]
    # for i, (t, r, Er, T) in enumerate(solutions):
    #     color = colors[i % len(colors)]
    #     # Numerical solution
    #     ax.plot(r, Er, color=color, linewidth=2, linestyle='-', 
    #             label=f't = {t:.1f} ns (Numerical)', marker='o', markersize=3, markevery=10)
        
    #     # Analytical solution (convert T to Er)
    #     if include_analytical:
    #         try:
    #             T_analytical, R_front = T_of_r_t(r, t, N=N)
    #             Er_analytical = A_RAD * T_analytical**4
    #             ax.plot(r, Er_analytical, color=color, linewidth=1.5, linestyle='--', 
    #                     alpha=0.7, label=f't = {t:.1f} ns (Analytical)')
    #             # Mark wave front
    #             ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
    #         except Exception as e:
    #             print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    # ax.set_xlabel('Position r (cm)', fontsize=12)
    # ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    # ax.set_title('Zeldovich Wave: Radiation Energy Density Profiles (Numerical vs Analytical)', fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=9, loc='best')
    # ax.set_xlim(0, solutions[-1][1][-1])
    
    plt.tight_layout()
    d = N-1
    #include d value in filename
    show(f'zeldovich_wave_d{d}.pdf')
    #plt.savefig('zeldovich_wave.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'zeldovich_wave_d{d}.pdf'")



def plot_material_properties():
    """Plot material properties as functions of temperature"""
    
    T_range = np.logspace(-2, 0.5, 200)  # 0.01 to ~3 keV
    Er_range = A_RAD * T_range**4
    
    sigma_R = np.array([zeldovich_opacity(Er) for Er in Er_range])
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
    plt.savefig('zeldovich_material_properties.png', dpi=150, bbox_inches='tight')
    print("Material properties plot saved as 'zeldovich_material_properties.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run Zeldovich wave simulation
    print("\nRunning Zeldovich wave simulation...")

    d = 2  # Geometry dimension (0=planar, 1=cylindrical, 2=spherical)
    N = d + 1  # Spatial dimension (0=planar, 1=cylindrical, 2=spherical)
    solutions, solver = run_zeldovich_wave(d=d,N=N, use_TRBDF2=True)
    
    # Plot results
    print("\nPlotting Zeldovich wave results...")
    plot_zeldovich_wave(solutions, include_analytical=True,N=N)
    
    print("\n" + "="*60)
    print("Zeldovich wave simulation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
