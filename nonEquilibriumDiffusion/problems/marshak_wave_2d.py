#!/usr/bin/env python3
"""
2D Marshak Wave Problem for Non-Equilibrium Radiation Diffusion
Tests directional symmetry by running wave in both x and y directions

Problem setup:
- Wave propagates in one direction (x or y)
- Perpendicular direction has reflecting boundaries (zero flux)
- Left boundary: incoming flux from blackbody at 1 keV
- Right boundary: zero flux
- Material opacity: σ_R = σ_P = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
- Plot solutions at 1, 10, and 20 ns

The two runs (x-direction vs y-direction) should give identical results
when compared along their respective propagation directions.
"""

import sys
import os
# Add parent directory to path to import twoDFV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

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


def marshak_rosseland_opacity(T, x, y):
    """Rosseland opacity: σ_R = 300 * T^-3"""
    return marshak_opacity(T)


def marshak_planck_opacity(T, x, y):
    """Planck opacity: σ_P = 300 * T^-3 (same as Rosseland for this problem)"""
    return marshak_opacity(T)


def marshak_specific_heat(T, x, y):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV) = 0.3/ρ GJ/(g·keV)
    
    Parameters:
    -----------
    T : float or array
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm) - not used for homogeneous problem
    
    Returns:
    --------
    cv : float or array
        Specific heat capacity per unit mass (GJ/(g·keV))
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T, x, y):
    """Material energy density: e = ρ·c_v·T
    
    Parameters:
    -----------
    T : float or array
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm) - not used for homogeneous problem
    
    Returns:
    --------
    e : float or array
        Material energy density (GJ/cm³)
    """
    cv_specific = marshak_specific_heat(T, x, y)
    return RHO * cv_specific * T


def marshak_inverse_material_energy(e, x, y):
    """Inverse: T from e = ρ·c_v·T => T = e/(ρ·c_v)
    
    Parameters:
    -----------
    e : float or array
        Material energy density (GJ/cm³)
    x, y : float
        Spatial coordinates (cm) - not used for homogeneous problem
    
    Returns:
    --------
    T : float or array
        Temperature (keV)
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return e / cv_volumetric


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def bc_blackbody_incoming(phi, pos, t, boundary='left', geometry='cartesian'):
    """Blackbody boundary at T = 1 keV (Dirichlet)
    
    φ = c·a·T_bc^4 at boundary
    Robin BC: A·φ + B·(n·∇φ) = C
    For Dirichlet: A=1, B=0, C=φ_bc
    """
    T_bc = 1.0  # keV
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc


def bc_zero_flux(phi, pos, t, boundary='left', geometry='cartesian'):
    """Zero flux boundary (Neumann)
    
    Robin BC: 0·φ + 1·(n·∇φ) = 0
    """
    return 0.0, 1.0, 0.0


def bc_reflecting(phi, pos, t, boundary='left', geometry='cartesian'):
    """Reflecting boundary (zero flux, same as bc_zero_flux)"""
    return 0.0, 1.0, 0.0


# =============================================================================
# MARSHAK WAVE SIMULATION
# =============================================================================

def run_marshak_wave_2d(direction='x', theta=1.0):
    """Run 2D Marshak wave simulation with wave propagating in x or y direction
    
    Parameters:
    -----------
    direction : str
        'x' for wave propagating in x-direction, 'y' for y-direction
    theta : float
        Time integration parameter: 1.0=Backward Euler, 0.5=Crank-Nicolson, 0.0=Forward Euler
    
    Returns:
    --------
    solutions : list of dict
        Solutions at target times
    solver : NonEquilibriumRadiationDiffusionSolver2D
        The solver object
    """
    
    print("="*80)
    print(f"2D Marshak Wave Problem - Wave in {direction.upper()}-direction")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_P = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print(f"  Wave direction: {direction}")
    if theta == 1.0:
        print("  Time integration: Backward Euler (theta=1.0)")
    elif theta == 0.5:
        print("  Time integration: Crank-Nicolson (theta=0.5)")
    else:
        print(f"  Time integration: theta={theta}")
    print("="*80)
    
    # Problem setup
    # Wave propagation direction: 0 to 0.5 cm, fine mesh
    # Perpendicular direction: smaller domain with reflecting BCs, coarse mesh
    
    if direction == 'x':
        # Wave propagates in x-direction (left to right)
        x_min, x_max = 0.0, 0.25
        y_min, y_max = 0.0, 0.1  # Narrow in y, just to make it 2D
        nx_cells = 50  # Coarser mesh for faster testing
        ny_cells = 1    # 1 cell perpendicular for true 1D behavior
        
        # Boundary conditions
        boundary_funcs = {
            'left': bc_blackbody_incoming,   # x=0: blackbody
            'right': bc_zero_flux,           # x=0.5: zero flux
            'bottom': bc_reflecting,         # y=0: reflecting
            'top': bc_reflecting             # y=0.1: reflecting
        }
        
    elif direction == 'y':
        # Wave propagates in y-direction (bottom to top)
        x_min, x_max = 0.0, 0.1  # Narrow in x
        y_min, y_max = 0.0, 0.25
        nx_cells = 1    # 1 cell perpendicular for true 1D behavior
        ny_cells = 50  # Coarser mesh for faster testing
        
        # Boundary conditions
        boundary_funcs = {
            'left': bc_reflecting,           # x=0: reflecting
            'right': bc_reflecting,          # x=0.1: reflecting
            'bottom': bc_blackbody_incoming, # y=0: blackbody
            'top': bc_zero_flux              # y=0.5: zero flux
        }
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'x' or 'y'")
    
    # Time stepping parameters
    dt = 0.02        # ns (slightly larger for faster testing)
    target_times = [1.0, 5.0]  # ns (fewer snapshots for faster testing)
    
    # Create solver
    print(f"\nInitializing 2D solver with {nx_cells} × {ny_cells} = {nx_cells*ny_cells} cells...")
    solver = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=x_min,
        x_max=x_max,
        nx_cells=nx_cells,
        y_min=y_min,
        y_max=y_max,
        ny_cells=ny_cells,
        geometry='cartesian',
        dt=dt,
        max_newton_iter=20,  # More iterations for temperature-dependent opacities
        newton_tol=1e-6,     # Slightly relaxed tolerance for faster convergence
        rosseland_opacity_func=marshak_rosseland_opacity,
        planck_opacity_func=marshak_planck_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        inverse_material_energy_func=marshak_inverse_material_energy,
        boundary_funcs=boundary_funcs,
        theta=theta  # Time integration parameter
    )
    
    # Initial condition: cold material
    T_init = 0.01  # keV
    phi_init = C_LIGHT * A_RAD * T_init**4
    
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    print(f"Initial conditions: T = {T_init} keV, φ = {phi_init:.6e} GJ/cm²")
    
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
        x_centers = solver.x_centers.copy()
        y_centers = solver.y_centers.copy()
        phi_2d = solver.get_phi_2d()
        T_2d = solver.get_T_2d()
        
        Er_2d = phi_2d / C_LIGHT
        T_rad_2d = (Er_2d / A_RAD)**0.25
        
        solutions.append({
            'time': current_time,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'phi_2d': phi_2d,
            'Er_2d': Er_2d,
            'T_2d': T_2d,
            'T_rad_2d': T_rad_2d
        })
        
        print(f"  t = {current_time:.1f} ns:")
        print(f"    Material: max T = {T_2d.max():.4f} keV, min T = {T_2d.min():.4f} keV")
        print(f"    Radiation: max T_rad = {T_rad_2d.max():.4f} keV, min T_rad = {T_rad_2d.min():.4f} keV")
        print(f"    Max φ = {phi_2d.max():.4e} GJ/cm²")
    
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
            # Extract 1D profile in x (only 1 cell in y, so just take that)
            r = sol['x_centers']
            T = sol['T_2d'][:, 0]  # Only cell in y
            T_rad = sol['T_rad_2d'][:, 0]
            phi = sol['phi_2d'][:, 0]
            Er = sol['Er_2d'][:, 0]
        else:  # direction == 'y'
            # Extract 1D profile in y (only 1 cell in x, so just take that)
            r = sol['y_centers']
            T = sol['T_2d'][0, :]  # Only cell in x
            T_rad = sol['T_rad_2d'][0, :]
            phi = sol['phi_2d'][0, :]
            Er = sol['Er_2d'][0, :]
        
        profiles.append({
            'time': sol['time'],
            'r': r,
            'T': T,
            'T_rad': T_rad,
            'phi': phi,
            'Er': Er
        })
    
    return profiles


def plot_comparison(profiles_x, profiles_y):
    """Plot comparison of x-direction and y-direction Marshak waves
    
    Parameters:
    -----------
    profiles_x : list of dict
        1D profiles from x-direction wave
    profiles_y : list of dict
        1D profiles from y-direction wave
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'red']  # Two times
    times = [prof['time'] for prof in profiles_x]
    
    # Self-similar solution parameters
    xi_max = 1.11305
    omega = 0.05989
    self_similar = lambda xi: (xi < xi_max) * np.power((1 - xi/xi_max)*(1+omega*xi/xi_max), 1/6)
    xi_vals = np.linspace(0, xi_max, 200)
    K_const = 8*A_RAD*C_LIGHT/((4+3)*3*300*RHO*0.3)
    
    for idx, (prof_x, prof_y, color) in enumerate(zip(profiles_x, profiles_y, colors)):
        t = prof_x['time']
        
        # Plot 1: Material temperature - X direction
        ax = axes[0, 0]
        ax.plot(prof_x['r'], prof_x['T'], color=color, linewidth=2, 
                linestyle='-', label=f't = {t:.0f} ns (X)')
        
        # Self-similar solution
        r_ref = xi_vals * (K_const * t)**0.5
        T_ref = self_similar(xi_vals)
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material T: X-direction Wave', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 2: Material temperature - Y direction
        ax = axes[0, 1]
        ax.plot(prof_y['r'], prof_y['T'],  color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (Y)')
        
        # Self-similar solution
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
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
        
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
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
            r_common = np.linspace(0, 0.5, 200)
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
    
    plt.suptitle('2D Marshak Wave: Directional Symmetry Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('marshak_wave_2d_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'marshak_wave_2d_comparison.png'")


def compute_differences(profiles_x, profiles_y):
    """Compute quantitative differences between x and y direction results
    
    Parameters:
    -----------
    profiles_x : list of dict
        1D profiles from x-direction wave
    profiles_y : list of dict
        1D profiles from y-direction wave
    
    Returns:
    --------
    differences : dict
        Statistics on differences
    """
    print("\n" + "="*80)
    print("Quantitative Comparison: X-direction vs Y-direction")
    print("="*80)
    
    for prof_x, prof_y in zip(profiles_x, profiles_y):
        t = prof_x['time']
        
        # Interpolate to common grid
        r_common = np.linspace(0, 0.5, 200)
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


def plot_theta_comparison(profiles_be, profiles_cn):
    """Plot comparison of Backward Euler vs Crank-Nicolson
    
    Parameters:
    -----------
    profiles_be : list of dict
        1D profiles from Backward Euler (theta=1.0)
    profiles_cn : list of dict
        1D profiles from Crank-Nicolson (theta=0.5)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'red']  # Two times
    times = [prof['time'] for prof in profiles_be]
    
    # Self-similar solution parameters
    xi_max = 1.11305
    omega = 0.05989
    self_similar = lambda xi: (xi < xi_max) * np.power((1 - xi/xi_max)*(1+omega*xi/xi_max), 1/6)
    xi_vals = np.linspace(0, xi_max, 200)
    K_const = 8*A_RAD*C_LIGHT/((4+3)*3*300*RHO*0.3)
    
    for idx, (prof_be, prof_cn, color) in enumerate(zip(profiles_be, profiles_cn, colors)):
        t = prof_be['time']
        
        # Plot 1: Material temperature - Backward Euler
        ax = axes[0, 0]
        ax.plot(prof_be['r'], prof_be['T'], color=color, linewidth=2, 
                linestyle='-', label=f't = {t:.0f} ns (BE)')
        
        # Self-similar solution
        r_ref = xi_vals * (K_const * t)**0.5
        T_ref = self_similar(xi_vals)
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material T: Backward Euler (θ=1.0)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 2: Material temperature - Crank-Nicolson
        ax = axes[0, 1]
        ax.plot(prof_cn['r'], prof_cn['T'],  color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (CN)')
        
        # Self-similar solution
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material T: Crank-Nicolson (θ=0.5)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 3: Comparison - overlay both methods
        ax = axes[0, 2]
        ax.plot(prof_be['r'], prof_be['T'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (BE)', alpha=0.7)
        ax.plot(prof_cn['r'], prof_cn['T'], color=color, linewidth=2,
                linestyle='--', label=f't = {t:.0f} ns (CN)', alpha=0.7)
        
        ax.plot(r_ref, T_ref, 'k:', linewidth=1.5, alpha=0.5, label='Self-similar' if idx == 0 else '')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('BE vs CN Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        
        # Plot 4: Radiation temperature - Backward Euler
        ax = axes[1, 0]
        ax.plot(prof_be['r'], prof_be['T_rad'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (BE)')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('$T_{rad}$ (keV)', fontsize=11)
        ax.set_title('Radiation T: Backward Euler', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 5: Radiation temperature - Crank-Nicolson
        ax = axes[1, 1]
        ax.plot(prof_cn['r'], prof_cn['T_rad'], color=color, linewidth=2,
                linestyle='-', label=f't = {t:.0f} ns (CN)')
        
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('$T_{rad}$ (keV)', fontsize=11)
        ax.set_title('Radiation T: Crank-Nicolson', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 6: Difference between BE and CN (last time only)
        if idx == len(profiles_be) - 1:
            ax = axes[1, 2]
            
            # Interpolate to common grid for difference
            r_common = np.linspace(0, 0.5, 200)
            T_be_interp = np.interp(r_common, prof_be['r'], prof_be['T'])
            T_cn_interp = np.interp(r_common, prof_cn['r'], prof_cn['T'])
            T_diff = T_be_interp - T_cn_interp
            
            ax.plot(r_common, T_diff, 'k-', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Position x (cm)', fontsize=11)
            ax.set_ylabel('$T_{BE} - T_{CN}$ (keV)', fontsize=11)
            ax.set_title(f'Difference at t = {t:.0f} ns', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.suptitle('2D Marshak Wave: Theta Method Comparison (BE vs CN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('marshak_wave_2d_theta_comparison.png', dpi=150, bbox_inches='tight')
    print("\nTheta comparison plot saved as 'marshak_wave_2d_theta_comparison.png'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(test_theta_comparison=True):
    """Main execution function
    
    Parameters:
    -----------
    test_theta_comparison : bool
        If True, compare Backward Euler vs Crank-Nicolson
        If False, run standard directional symmetry test
    """
    
    if test_theta_comparison:
        # Test both Backward Euler and Crank-Nicolson
        print("\n" + "="*80)
        print("2D MARSHAK WAVE - THETA COMPARISON TEST")
        print("="*80)
        print("\nThis test runs the Marshak wave with both Backward Euler (theta=1.0)")
        print("and Crank-Nicolson (theta=0.5) to compare convergence behavior.")
        print("="*80)
        
        # Run with Backward Euler (theta=1.0)
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*22 + "PART 1: BACKWARD EULER (θ=1.0)" + " "*25 + "║")
        print("╚" + "="*78 + "╝")
        solutions_be, solver_be, _ = run_marshak_wave_2d(direction='x', theta=1.0)
        profiles_be = extract_1d_profile(solutions_be, 'x')
        
        # Run with Crank-Nicolson (theta=0.5)
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*21 + "PART 2: CRANK-NICOLSON (θ=0.5)" + " "*25 + "║")
        print("╚" + "="*78 + "╝")
        solutions_cn, solver_cn, _ = run_marshak_wave_2d(direction='x', theta=0.5)
        profiles_cn = extract_1d_profile(solutions_cn, 'x')
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARING BACKWARD EULER vs CRANK-NICOLSON")
        print("="*80)
        compute_differences(profiles_be, profiles_cn)
        
        # Plot comparison
        print("\nGenerating theta comparison plots...")
        plot_theta_comparison(profiles_be, profiles_cn)
        
        print("\n" + "="*80)
        print("2D Marshak wave theta comparison test completed successfully!")
        print("="*80)
    else:
        # Standard directional symmetry test
        print("\n" + "="*80)
        print("2D MARSHAK WAVE - DIRECTIONAL SYMMETRY TEST")
        print("="*80)
        print("\nThis test runs the Marshak wave in both X and Y directions")
        print("with reflecting boundaries in the perpendicular direction.")
        print("The results should be identical, demonstrating directional symmetry.")
        print("="*80)
        
        # Run in X direction
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*25 + "PART 1: X-DIRECTION WAVE" + " "*29 + "║")
        print("╚" + "="*78 + "╝")
        solutions_x, solver_x, _ = run_marshak_wave_2d(direction='x', theta=1.0)
        profiles_x = extract_1d_profile(solutions_x, 'x')
        
        # Run in Y direction
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*25 + "PART 2: Y-DIRECTION WAVE" + " "*29 + "║")
        print("╚" + "="*78 + "╝")
        solutions_y, solver_y, _ = run_marshak_wave_2d(direction='y', theta=1.0)
        profiles_y = extract_1d_profile(solutions_y, 'y')
        
        # Compare results
        compute_differences(profiles_x, profiles_y)
        
        # Plot comparison
        print("\nGenerating comparison plots...")
        plot_comparison(profiles_x, profiles_y)
        
        print("\n" + "="*80)
        print("2D Marshak wave directional symmetry test completed successfully!")
        print("="*80)


if __name__ == "__main__":
    main()
