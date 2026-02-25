#!/usr/bin/env python3
"""
2D Cylindrical Zel'dovich Wave Problem - Point Source at Origin
Tests 2D cylindrical solver with radially symmetric point source

Problem setup:
- Cylindrical (r-z) geometry
- All boundaries: reflecting (zero flux)
- Initial condition: energy pulse at r≈0, z≈0 (approximate Dirac delta)
- Material opacity: σ_R = σ_P = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)
- Compare radial profiles ρ = √(r² + z²) with d=2 (spherical) self-similar solution

The 2D cylindrical solution should match the 1D spherical solution
when plotted as function of radius ρ from the origin.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import twoDFV
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twoDFV import (
    NonEquilibriumRadiationDiffusionSolver2D,
    A_RAD,
    C_LIGHT,
    RHO
)

# Try to import analytical solution
try:
    eq_utils_dir = os.path.join(parent_dir, "..", "EqDiffusion", "utils")
    sys.path.insert(0, eq_utils_dir)
    from zeldovich import T_of_r_t
    ANALYTICAL_AVAILABLE = True
except ImportError:
    print("Warning: Analytical solution module not found. Will plot numerical only.")
    ANALYTICAL_AVAILABLE = False

# =============================================================================
# ZELDOVICH WAVE MATERIAL PROPERTIES (same as 1D version)
# =============================================================================

def zeldovich_rosseland_opacity(T, x, y):
    """σ_R = 300 * T^-3"""
    n = 3
    T_min = 0.001
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)


def zeldovich_planck_opacity(T, x, y):
    """σ_P = 300 * T^-3"""
    return zeldovich_rosseland_opacity(T, x, y)


def zeldovich_specific_heat(T, x, y):
    """c_v = 3e-6 GJ/(cm^3·keV)"""
    cv_volumetric = 3e-6
    return cv_volumetric / RHO


def zeldovich_material_energy(T, x, y):
    """e = ρ·c_v·T"""
    cv_volumetric = 3e-6
    return RHO * cv_volumetric * T


def zeldovich_inverse_material_energy(e, x, y):
    """T from e"""
    cv_volumetric = 3e-6
    return e / (RHO * cv_volumetric)


# =============================================================================
# BOUNDARY CONDITIONS - all reflecting
# =============================================================================

def bc_reflecting(phi, pos, t, boundary='left', geometry='cylindrical'):
    """Reflecting boundary: zero flux (∇φ · n = 0)"""
    return 0.0, 1.0, 0.0


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_zeldovich_wave_2d():
    """Run 2D cylindrical Zel'dovich wave with point source at origin"""
    
    global ANALYTICAL_AVAILABLE
    
    print("="*80)
    print("2D Cylindrical Zel'dovich Wave - Point Source at Origin")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_P = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)")
    print("  All boundaries: Reflecting (zero flux)")
    print("  Initial condition: Energy pulse at r≈0, z≈0")
    print("  Geometry: Cylindrical (r-z)")
    print("="*80)
    
    # Domain setup
    # Small r to capture near-axis behavior, but r_min > 0 to avoid singularity
    r_min = 0.0001   # cm (small but non-zero)
    r_max = 3.0    # cm
    z_min = 0.0    # cm
    z_max = 3.0    # cm
    
    # Grid resolution (small for quick testing)
    nr_cells = 50
    nz_cells = 50
    
    # Time stepping
    dt_initial = 1e-6  # Larger initial dt for faster run
    dt_max = 1e-3
    dt = dt_initial
    target_times = [0.1, 0.3, 1.0, 3.0]  # ns (four snapshots)
    
    # All boundaries reflecting
    boundary_funcs = {
        'left': bc_reflecting,    # r = r_min
        'right': bc_reflecting,   # r = r_max
        'bottom': bc_reflecting,  # z = 0
        'top': bc_reflecting      # z = z_max
    }
    
    # Create solver
    print(f"\nInitializing 2D cylindrical solver with {nr_cells} × {nz_cells} = {nr_cells*nz_cells} cells...")
    solver = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=r_min,
        x_max=r_max,
        nx_cells=nr_cells,
        y_min=z_min,
        y_max=z_max,
        ny_cells=nz_cells,
        geometry='cylindrical',
        dt=dt,
        max_newton_iter=100,
        newton_tol=1e-6,
        rosseland_opacity_func=zeldovich_rosseland_opacity,
        planck_opacity_func=zeldovich_planck_opacity,
        specific_heat_func=zeldovich_specific_heat,
        material_energy_func=zeldovich_material_energy,
        inverse_material_energy_func=zeldovich_inverse_material_energy,
        boundary_funcs=boundary_funcs,
        theta=1.0
    )
    
    # Initial condition: approximate Dirac delta at origin
    # Put high temperature in small region near (r_min, z=0)
    t_init = 0.05  # ns - start from analytical solution
    
    print(f"\nSetting initial condition at t_init = {t_init} ns...")
    
    T_2d = np.zeros((nr_cells, nz_cells))
    phi_2d = np.zeros((nr_cells, nz_cells))
    
    if ANALYTICAL_AVAILABLE:
        # Initialize from spherical self-similar solution (d=2, N=3)
        print("  Using analytical self-similar solution for d=2 (spherical)")
        try:
            for i in range(nr_cells):
                for j in range(nz_cells):
                    r = solver.x_centers[i]
                    z = solver.y_centers[j]
                    rho = np.sqrt(r**2 + z**2)  # Spherical radius from origin
                    
                    # Get analytical temperature at this radius
                    T_analytical, _ = T_of_r_t(np.array([rho]), t_init, N=3)
                    T_min_floor = 0.01  # keV
                    T_2d[i, j] = max(T_analytical[0], T_min_floor)
                    phi_2d[i, j] = A_RAD * C_LIGHT * T_2d[i, j]**4
            
            print(f"  Successfully initialized from analytical solution")
            print(f"  T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
        except Exception as e:
            print(f"  Warning: Could not use analytical solution: {e}")
            print(f"  Falling back to hot spot initial condition")
            ANALYTICAL_AVAILABLE = False
            raise e #need the analytic solution to run this test
    
    if not ANALYTICAL_AVAILABLE:
        # Fallback: hot spot near origin
        T_cold = 0.01  # keV
        T_hot = 5.9    # keV
        
        for i in range(nr_cells):
            for j in range(nz_cells):
                r = solver.x_centers[i]
                z = solver.y_centers[j]
                rho = np.sqrt(r**2 + z**2)
                
                # Put energy in small region near origin
                if rho < 0.1:
                    T_2d[i, j] = T_hot
                else:
                    T_2d[i, j] = T_cold
                
                phi_2d[i, j] = A_RAD * C_LIGHT * T_2d[i, j]**4
        
        t_init = 0.0
        print(f"  Hot spot initial condition: T_hot = {T_hot} keV in ρ < 0.1 cm")
    
    solver.set_initial_condition(phi_init=phi_2d, T_init=T_2d)
    
    # Print initial info
    T_2d = solver.get_T_2d()
    phi_2d = solver.get_phi_2d()
    print(f"\nInitial condition:")
    print(f"  T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
    print(f"  φ range: [{phi_2d.min():.4e}, {phi_2d.max():.4e}] GJ/cm²")
    
    # Compute initial energy
    E_rad_init = np.sum(phi_2d * solver.V_cells.T) / C_LIGHT  # Convert φ to E_r
    E_mat_init = np.sum(zeldovich_material_energy(T_2d, 0, 0) * solver.V_cells.T)
    E_total_init = E_rad_init + E_mat_init
    print(f"\nInitial energy:")
    print(f"  Radiation: {E_rad_init:.6e} GJ")
    print(f"  Material:  {E_mat_init:.6e} GJ")
    print(f"  Total:     {E_total_init:.6e} GJ")
    
    # Time evolution
    print(f"\nTime evolution:")
    current_time = 0.0
    solutions = []
    step_count = 0
    
    for target_time_physical in target_times:
        target_time_numerical = target_time_physical - t_init
        
        if target_time_numerical <= 0:
            print(f"  Skipping t = {target_time_physical:.1f} ns (at or before initial time)")
            continue
        
        print(f"\nAdvancing to t = {target_time_physical:.1f} ns...")
        
        while current_time < target_time_numerical:
            # Adjust dt to hit target exactly
            if current_time + dt > target_time_numerical:
                dt = target_time_numerical - current_time
            
            solver.dt = dt
            
            # Store old values
            phi_old = solver.phi.copy()
            T_old = solver.T.copy()
            
            # Take timestep
            verbose_step = (step_count < 3 or step_count % 100 == 0)
            solver.time_step(n_steps=1, verbose=verbose_step)
            
            # Check changes for adaptive dt
            phi_change = np.max(np.abs(solver.phi - phi_old) / (np.abs(phi_old) + 1e-10))
            T_change = np.max(np.abs(solver.T - T_old) / (np.abs(T_old) + 1e-10))
            
            # Adaptive timestep
            if phi_change < 0.1 and T_change < 0.1 and dt < dt_max:
                dt_new = min(dt * 1.5, dt_max)
                if step_count % 100 == 0 and dt_new > dt:
                    print(f"    Step {step_count}: t={current_time:.6e} ns, increasing dt {dt:.2e} → {dt_new:.2e} ns")
                dt = dt_new
            elif phi_change > 0.5 or T_change > 0.5:
                dt = max(dt * 0.5, dt_initial)
                if verbose_step:
                    print(f"    Step {step_count}: Large changes, reducing dt to {dt:.2e} ns")
            
            current_time += solver.dt
            step_count += 1
        
        # Store solution
        physical_time = current_time + t_init
        T_2d = solver.get_T_2d()
        phi_2d = solver.get_phi_2d()
        
        solutions.append({
            'time': physical_time,
            'r_centers': solver.x_centers.copy(),
            'z_centers': solver.y_centers.copy(),
            'T_2d': T_2d.copy(),
            'phi_2d': phi_2d.copy()
        })
        
        # Energy conservation check
        E_rad = np.sum(phi_2d * solver.V_cells.T) / C_LIGHT
        E_mat = np.sum(zeldovich_material_energy(T_2d, 0, 0) * solver.V_cells.T)
        E_total = E_rad + E_mat
        E_error = abs(E_total - E_total_init) / E_total_init * 100
        
        print(f"  t = {physical_time:.1f} ns:")
        print(f"    T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
        print(f"    Total energy: {E_total:.6e} GJ (error: {E_error:.2e}%)")
    
    return solutions, solver


def extract_radial_profiles(solutions):
    """Extract 1D radial profiles from 2D solutions
    
    For each solution, extract T and φ as function of ρ = √(r² + z²)
    by sampling along various rays from origin
    """
    profiles = []
    
    for sol in solutions:
        t = sol['time']
        r_centers = sol['r_centers']
        z_centers = sol['z_centers']
        T_2d = sol['T_2d']
        phi_2d = sol['phi_2d']
        
        # Sample along multiple directions and collect (rho, T, phi) points
        rho_list = []
        T_list = []
        phi_list = []
        
        # Sample along z-axis (r=r_min, varying z)
        i_axis = 0  # Lowest r index (near axis)
        for j in range(len(z_centers)):
            r = r_centers[i_axis]
            z = z_centers[j]
            rho = np.sqrt(r**2 + z**2)
            rho_list.append(rho)
            T_list.append(T_2d[i_axis, j])
            phi_list.append(phi_2d[i_axis, j])
        
        # Sample along r-axis (varying r, z=0)
        j_plane = 0  # z=0 plane
        for i in range(len(r_centers)):
            r = r_centers[i]
            z = z_centers[j_plane]
            rho = np.sqrt(r**2 + z**2)
            rho_list.append(rho)
            T_list.append(T_2d[i, j_plane])
            phi_list.append(phi_2d[i, j_plane])
        
        # Sample along diagonal (r=z line)
        for i in range(len(r_centers)):
            r = r_centers[i]
            # Find closest z to r
            j = np.argmin(np.abs(z_centers - r))
            z = z_centers[j]
            rho = np.sqrt(r**2 + z**2)
            rho_list.append(rho)
            T_list.append(T_2d[i, j])
            phi_list.append(phi_2d[i, j])
        
        # Convert to arrays and sort by rho
        rho_array = np.array(rho_list)
        T_array = np.array(T_list)
        phi_array = np.array(phi_list)
        
        sort_idx = np.argsort(rho_array)
        rho_sorted = rho_array[sort_idx]
        T_sorted = T_array[sort_idx]
        phi_sorted = phi_array[sort_idx]
        
        # Average values at similar rho (for cleaner profile)
        rho_unique = []
        T_unique = []
        phi_unique = []
        
        n_bins = 50
        rho_bins = np.linspace(rho_sorted.min(), rho_sorted.max(), n_bins+1)
        
        for i in range(n_bins):
            mask = (rho_sorted >= rho_bins[i]) & (rho_sorted < rho_bins[i+1])
            if np.any(mask):
                rho_unique.append(np.mean(rho_sorted[mask]))
                T_unique.append(np.mean(T_sorted[mask]))
                phi_unique.append(np.mean(phi_sorted[mask]))
        
        profiles.append({
            'time': t,
            'rho': np.array(rho_unique),
            'T': np.array(T_unique),
            'phi': np.array(phi_unique)
        })
    
    return profiles


def plot_results(solutions, profiles):
    """Plot 2D solutions and radial profiles with analytical comparison"""
    
    global ANALYTICAL_AVAILABLE
    
    n_times = len(solutions)
    colors = ['blue', 'green', 'orange', 'red']
    
    # Figure 1: 2D temperature fields (2×2 grid for 4 times)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    axes1 = axes1.flatten()
    
    for idx, sol in enumerate(solutions):
        ax = axes1[idx]
        t = sol['time']
        r_centers = sol['r_centers']
        z_centers = sol['z_centers']
        T_2d = sol['T_2d']
        
        # Create mesh grid for plotting
        R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
        
        # Plot temperature field
        levels = np.linspace(T_2d.min(), T_2d.max(), 20)
        contour = ax.contourf(R, Z, T_2d, levels=levels, cmap='hot')
        ax.set_xlabel('r (cm)', fontsize=11)
        ax.set_ylabel('z (cm)', fontsize=11)
        ax.set_title(f't = {t:.1f} ns', fontsize=12, fontweight='bold')
        
        # Set explicit limits to show only first quadrant (r >= 0, z >= 0)
        ax.set_xlim(np.min(r_centers), r_centers[-1])
        ax.set_ylim(np.min(z_centers), z_centers[-1])
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('T (keV)', fontsize=10)
        
        # Add arc showing wave front from r-axis to z-axis (if analytical available)
        if ANALYTICAL_AVAILABLE:
            try:
                _, R_front = T_of_r_t(np.array([1.0]), t, N=3)
                # Draw arc in first quadrant only
                theta = np.linspace(0, np.pi/2, 100)
                r_arc = R_front * np.cos(theta)
                z_arc = R_front * np.sin(theta)
                ax.plot(r_arc, z_arc, 'c--', linewidth=2, alpha=0.8)
                ax.text(0.02, 0.98, f'Front Radius = {R_front:.2f} cm',
                       transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', color='cyan',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            except:
                pass
    
    plt.tight_layout()
    plt.savefig('zeldovich_2d_temperature_fields.png', dpi=150, bbox_inches='tight')
    print("\nTemperature fields plot saved as 'zeldovich_2d_temperature_fields.png'")
    plt.close()
    
    # Figure 2: Radial profiles with analytical comparison
    n_plot_rows = (n_times + 1) // 2  # At least 1 row
    fig2, axes2 = plt.subplots(n_plot_rows, 2, figsize=(14, 5*n_plot_rows))
    if n_times == 1:
        axes2 = axes2.reshape(1, 2)
    
    for idx, prof in enumerate(profiles):
        # Temperature profile
        row_idx = idx // 2 if n_times > 1 else 0
        col_idx = 0
        ax = axes2[row_idx, col_idx] if n_times > 1 else axes2[0, 0]
        t = prof['time']
        rho = prof['rho']
        T = prof['T']
        color = colors[idx]
        
        ax.plot(rho, T, 'o', color=color, markersize=4, alpha=0.6,
               label=f't = {t:.1f} ns (numerical)')
        
        # Analytical solution
        if ANALYTICAL_AVAILABLE:
            try:
                rho_analytical = np.linspace(rho.min(), rho.max(), 200)
                T_analytical, R_front = T_of_r_t(rho_analytical, t, N=3)
                ax.plot(rho_analytical, T_analytical, '-', color=color, linewidth=2,
                       alpha=0.8, label=f't = {t:.1f} ns (analytical d=2)')
                ax.axvline(R_front, color=color, linestyle=':', alpha=0.5)
            except Exception as e:
                print(f"Warning: Could not plot analytical for t={t}: {e}")
        
        ax.set_xlabel('ρ = √(r² + z²) (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Material Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Radiation temperature profiles
    for idx, prof in enumerate(profiles):
        row_idx = idx // 2 if n_times > 1 else 0
        col_idx = 1
        ax = axes2[row_idx, col_idx] if n_times > 1 else axes2[0, 1]
        t = prof['time']
        rho = prof['rho']
        phi = prof['phi']
        color = colors[idx]
        
        # Convert φ to T_rad
        Er = phi / C_LIGHT
        T_rad = (Er / A_RAD)**0.25
        
        ax.plot(rho, T_rad, 'o', color=color, markersize=4, alpha=0.6,
               label=f't = {t:.1f} ns')
        
        ax.set_xlabel('ρ = √(r² + z²) (cm)', fontsize=11)
        ax.set_ylabel('$T_{rad}$ (keV)', fontsize=11)
        ax.set_title('Radiation Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('2D Cylindrical Zel\'dovich Wave: Radial Profiles vs d=2 Self-Similar Solution',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('zeldovich_2d_radial_profiles.png', dpi=150, bbox_inches='tight')
    print("Radial profiles plot saved as 'zeldovich_2d_radial_profiles.png'")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run simulation
    solutions, solver = run_zeldovich_wave_2d()
    
    # Extract radial profiles
    print("\nExtracting radial profiles...")
    profiles = extract_radial_profiles(solutions)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(solutions, profiles)
    
    print("\n" + "="*80)
    print("2D Cylindrical Zel'dovich wave test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
