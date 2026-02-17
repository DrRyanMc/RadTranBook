#!/usr/bin/env python3
"""
Cylindrical (r-z) Zeldovich Wave Problem
Radiative heat wave with initial energy pulse at r=0

Problem setup:
- All boundaries: reflecting (zero flux)
- Initial condition: delta-function-like energy pulse at r=0 (axis)
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)
- Cylindrical (r-z) geometry with axial symmetry expected
- Compare radial profile with cylindrical 1D self-similar solution
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from twoDFV import (
    RadiationDiffusionSolver2D,
    temperature_from_Er,
    A_RAD,
    C_LIGHT,
    RHO
)

from plotfuncs import hide_spines, show
# Add path to import analytical solution
sys.path.insert(0, str(Path.home() / "Dropbox/Apps/Overleaf/RadTranBook/img/equilibriumDiffusion"))
from zeldovich import T_of_r_t

# =============================================================================
# ZELDOVICH WAVE MATERIAL PROPERTIES
# =============================================================================

def zeldovich_opacity(Er, r_val, z_val):
    """Temperature-dependent Rosseland opacity: σ_R = 300 * T^-3
    
    Parameters:
    -----------
    Er : float
        Radiation energy density (GJ/cm^3)
    r_val : float
        Radial coordinate (cm) - unused but required by interface
    z_val : float
        Axial coordinate (cm) - unused but required by interface
    
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


def zeldovich_specific_heat(T, r_val, z_val):
    """Specific heat: c_v = 3e-6 GJ/(cm^3·keV)
    
    Note: This is volumetric heat capacity, but the solver expects
    specific heat per unit mass. We'll use c_v/ρ.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    r_val : float
        Radial coordinate (cm) - unused but required by interface
    z_val : float
        Axial coordinate (cm) - unused but required by interface
    
    Returns:
    --------
    cv : float
        Specific heat capacity per unit mass (GJ/(g·keV))
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def zeldovich_material_energy(T, r_val, z_val):
    """Material energy density: e = c_v * T (volumetric)
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    r_val : float
        Radial coordinate (cm) - unused but required by interface
    z_val : float
        Axial coordinate (cm) - unused but required by interface
    
    Returns:
    --------
    e : float
        Material energy density (GJ/cm^3)
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return cv_volumetric * T


# =============================================================================
# ZELDOVICH WAVE BOUNDARY CONDITIONS (all reflecting)
# =============================================================================

def zeldovich_bc_reflecting(Er_boundary, coord1_val, coord2_val, geometry='cylindrical'):
    """Reflecting boundary: zero flux
    
    Robin BC: A*E_r + B*(dE_r/dn) = C
    Zero flux: 0*E_r + 1*dE_r/dn = 0
    """
    return 0.0, 1.0, 0.0  # Reflecting boundary


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_radial_profile_rz(Er_2d, r_centers, z_centers):
    """Extract radial profile from r-z solution
    
    Average temperature values at each radial position across all z values.
    
    Parameters:
    -----------
    Er_2d : ndarray (n_r_cells, n_z_cells)
        2D radiation energy density
    r_centers : ndarray
        r-coordinates of cell centers
    z_centers : ndarray
        z-coordinates of cell centers
    
    Returns:
    --------
    r_vals : ndarray
        Radial distances
    T_vals : ndarray
        Averaged temperature at each radial distance
    """
    n_r, n_z = Er_2d.shape
    
    # For each r position, average over all z positions
    r_vals = r_centers.copy()
    T_vals = np.zeros(n_r)
    
    for i in range(n_r):
        # Average temperature across all z at this r
        Er_at_r = Er_2d[i, :]
        T_at_r = temperature_from_Er(Er_at_r)
        T_vals[i] = np.mean(T_at_r)
    
    return r_vals, T_vals


# =============================================================================
# ZELDOVICH WAVE SIMULATION
# =============================================================================

def run_zeldovich_wave_rz():
    """Run cylindrical r-z Zeldovich wave simulation with axial source"""
    
    print("="*70)
    print("Cylindrical (r-z) Zeldovich Wave Problem")
    print("="*70)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)")
    print("  All BCs: Zero flux (reflecting)")
    print("  Initial condition: Delta-function energy pulse at r=0")
    print("="*70)
    
    # Problem setup
    # r direction (radial)
    r_min = 0.0
    r_max = 3.0
    n_r_cells = 100
    
    # z direction (axial)
    z_min = 0.0
    z_max = 1.0  # Shorter in z since we expect uniformity
    n_z_cells = 30
    
    # Time stepping parameters
    dt = 0.001  # ns (small time step for stability)
    target_times = [0.1, 0.3, 1.0, 3.0]  # ns
    
    # Create 2D solver with cylindrical geometry
    solver = RadiationDiffusionSolver2D(
        coord1_min=r_min,
        coord1_max=r_max,
        n1_cells=n_r_cells,
        coord2_min=z_min,
        coord2_max=z_max,
        n2_cells=n_z_cells,
        geometry='cylindrical',  # KEY: cylindrical geometry
        dt=dt,
        max_newton_iter=20,
        newton_tol=1e-8,
        rosseland_opacity_func=zeldovich_opacity,
        specific_heat_func=zeldovich_specific_heat,
        material_energy_func=zeldovich_material_energy,
        left_bc_func=zeldovich_bc_reflecting,   # r=r_min (r=0)
        right_bc_func=zeldovich_bc_reflecting,  # r=r_max
        bottom_bc_func=zeldovich_bc_reflecting, # z=z_min
        top_bc_func=zeldovich_bc_reflecting,    # z=z_max
        theta=1.0,  # Implicit Euler for stability
        use_jfnk=False  # Use direct solver
    )
    
    # Initial condition: cold material with delta-function pulse at r=0
    def initial_Er(r, z):
        T_cold = 0.01  # keV (cold but not too small)
        Er_cold = A_RAD * T_cold**4
        return Er_cold
    
    solver.set_initial_condition(initial_Er)
    
    # Add energy pulse to cells near r=0
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    
    # Total energy to deposit (similar to 1D case)
    E_total = 1.0  # Total energy in GJ
    
    # Deposit energy in cells near r=0 (first few radial cells, all z)
    # Use Gaussian-like distribution in r
    energy_cells = []
    weights = []
    
    # Consider first few radial cells
    n_r_init = min(5, solver.n1_cells)
    for i in range(n_r_init):
        for j in range(solver.n2_cells):
            r = solver.coord1_centers[i]
            weight = np.exp(-r**2 / 0.01)  # Gaussian weight in r
            energy_cells.append((i, j))
            weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights /= weights.sum()
    
    # Distribute energy
    for (i, j), weight in zip(energy_cells, weights):
        V_cell = solver.V_cells[i, j]
        Er_2d[i, j] += weight * E_total / V_cell
    
    solver.Er = Er_2d.flatten()
    solver.Er_old = solver.Er.copy()
    
    print(f"\nInitial condition:")
    print(f"  First cell (r=0) Er = {Er_2d[0, 0]:.4e} GJ/cm³")
    print(f"  First cell (r=0) T = {temperature_from_Er(Er_2d[0, 0]):.4f} keV")
    print(f"  Total initial energy = {E_total:.4e} GJ")
    
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
                if current_time % 0.1 < dt:  # Print every 0.1 ns
                    print(f"  Time: {current_time:.3f} ns")
            except Exception as e:
                print(f"  Error at t={current_time:.3f} ns: {e}")
                break
            
            # Restore dt if we temporarily changed it
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution at target time
        Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
        
        # Extract radial profile (average over z)
        r_vals, T_vals = extract_radial_profile_rz(Er_2d, solver.coord1_centers, solver.coord2_centers)
        
        # Also check energy conservation
        total_Er = np.sum(Er_2d * solver.V_cells)
        T_2d = temperature_from_Er(Er_2d)
        # Compute material energy with spatial coordinates
        mat_energy_2d = np.zeros_like(T_2d)
        for i in range(solver.n1_cells):
            for j in range(solver.n2_cells):
                mat_energy_2d[i, j] = zeldovich_material_energy(T_2d[i, j], 
                                                                 solver.coord1_centers[i],
                                                                 solver.coord2_centers[j])
        total_mat = np.sum(mat_energy_2d * solver.V_cells)
        total_energy = total_Er + total_mat
        
        solutions.append((current_time, r_vals.copy(), T_vals.copy(), Er_2d.copy()))
        print(f"  t = {current_time:.3f} ns: max T = {T_2d.max():.4f} keV, total energy = {total_energy:.4e} GJ")
    
    return solutions, solver


def plot_zeldovich_wave_rz(solutions):
    """Plot cylindrical r-z Zeldovich wave solutions and comparison with 1D analytical"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot temperature profiles (radial)
    ax = axes[0]
    for i, (t, r, T, Er_2d) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # r-z numerical solution (z-averaged)
        ax.plot(r, T, color=color, linewidth=2, linestyle='-', 
                label=f'r-z Numerical t = {t:.2f} ns', marker='o', markersize=3, markevery=5)
        
        # Cylindrical 1D analytical solution (N=2)
        try:
            T_analytical, R_front = T_of_r_t(r, t, N=2)
            ax.plot(r, T_analytical, color=color, linewidth=1.5, linestyle='--', 
                    alpha=0.7, label=f'1D Cylindrical t = {t:.2f} ns')
            # Mark wave front
            ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
        except Exception as e:
            print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    ax.set_xlabel('Radial Distance r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    ax.set_title('Cylindrical r-z Zeldovich Wave: Temperature vs. 1D Solution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0, solutions[-1][1][-1])
    
    # Plot radiation energy density profiles
    ax = axes[1]
    for i, (t, r, T, Er_2d) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # Convert temperature to Er for plotting
        Er_radial = A_RAD * T**4
        ax.plot(r, Er_radial, color=color, linewidth=2, linestyle='-',
                label=f'r-z t = {t:.2f} ns', marker='o', markersize=3, markevery=5)
        
        # Analytical Er
        try:
            T_analytical, R_front = T_of_r_t(r, t, N=2)
            Er_analytical = A_RAD * T_analytical**4
            ax.plot(r, Er_analytical, color=color, linewidth=1.5, linestyle='--', 
                    alpha=0.7, label=f'1D Cylindrical t = {t:.2f} ns')
        except:
            pass
    
    ax.set_xlabel('Radial Distance r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    ax.set_title('Cylindrical r-z Zeldovich Wave: Radiation Energy Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0, solutions[-1][1][-1])
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('zeldovich_wave_rz.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'zeldovich_wave_rz.png'")


def plot_publication_comparison(solver, solutions, times=[0.3, 1.0]):
    """Create publication-quality comparison plot with quadrants
    
    Single pcolormesh plot where quadrants show:
    - Top-left: Analytical at early time
    - Top-right: Numerical at early time
    - Bottom-left: Analytical at late time
    - Bottom-right: Numerical at late time
    
    Parameters:
    -----------
    solver : RadiationDiffusionSolver2D
        The solver object with grid information
    solutions : list
        List of (time, r, T, Er_2d) tuples from simulation
    times : list
        Two times to compare [early_time, late_time]
    """
    
    # Find solutions closest to requested times
    solution_dict = {t: (r, T, Er_2d) for t, r, T, Er_2d in solutions}
    selected_solutions = []
    for target_time in times:
        closest_time = min(solution_dict.keys(), key=lambda t: abs(t - target_time))
        r, T, Er_2d = solution_dict[closest_time]
        selected_solutions.append((closest_time, Er_2d))
    
    # Extract the two solution times
    time1, Er_2d_1 = selected_solutions[0]  # Early time
    time2, Er_2d_2 = selected_solutions[1]  # Late time
    
    # Convert to temperature
    T_num_1 = temperature_from_Er(Er_2d_1)
    T_num_2 = temperature_from_Er(Er_2d_2)
    
    # Compute analytical solutions
    T_ana_1 = np.zeros_like(T_num_1)
    T_ana_2 = np.zeros_like(T_num_2)
    
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            r = solver.coord1_centers[i]
            z = solver.coord2_centers[j]
            # For cylindrical r-z, we just use r for the analytical solution
            
            try:
                T_val, _ = T_of_r_t(np.array([r]), time1, N=2)
                T_ana_1[i, j] = T_val[0]
            except:
                T_ana_1[i, j] = 0.0
                
            try:
                T_val, _ = T_of_r_t(np.array([r]), time2, N=2)
                T_ana_2[i, j] = T_val[0]
            except:
                T_ana_2[i, j] = 0.0
    
    # Get dimensions
    n1, n2 = T_num_1.shape
    
    # Create composite array with quadrants
    T_composite = np.full((n1, n2), np.nan)
    
    # Fill quadrants:
    # Top-left: Analytical at early time (time1)
    T_composite[:n1//2, :n2//2] = T_ana_1[:n1//2, :n2//2]
    # Top-right: Numerical at early time (time1)
    T_composite[:n1//2, n2//2:] = T_num_1[:n1//2, n2//2:]
    # Bottom-left: Analytical at late time (time2)
    T_composite[n1//2:, :n2//2] = T_ana_2[n1//2:, :n2//2]
    # Bottom-right: Numerical at late time (time2)
    T_composite[n1//2:, n2//2:] = T_num_2[n1//2:, n2//2:]
    
    # Find global min/max for consistent color scale
    T_min = min(T_ana_1.min(), T_num_1.min(), T_ana_2.min(), T_num_2.min())
    T_max = max(T_ana_1.max(), T_num_1.max(), T_ana_2.max(), T_num_2.max())
    
    # Create extended coordinate arrays
    dr = solver.coord1_centers[1] - solver.coord1_centers[0]
    dz = solver.coord2_centers[1] - solver.coord2_centers[0]
    
    R_extended, Z_extended = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    import matplotlib.font_manager as fm
    font = None
    try:
        font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc', size = 12)
    except:
        pass

    # Plot composite with viridis colormap
    cmap = 'plasma'
    im = ax.pcolormesh(R_extended, Z_extended, T_composite, shading='auto', 
                       cmap=cmap, vmin=T_min, vmax=T_max)
    
    # Add labels and styling
    ax.set_xlabel('r (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_aspect('equal')
    
    # Add text labels in each quadrant
    r_mid = (solver.coord1_centers[0] + solver.coord1_centers[-1]) / 2
    z_mid = (solver.coord2_centers[0] + solver.coord2_centers[-1]) / 2
    
    # Text properties with white background box
    text_props = dict(fontsize=11, fontweight='bold', ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontproperties=font)
    
    # Add labels
    ax.text(r_mid * 0.5, z_mid * 0.3, "Self-Similar", **text_props)
    ax.text(r_mid * 1.5, z_mid * 0.3, "Numerical", **text_props)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (keV)', fontsize=12, fontweight='bold')
    
    # Draw dividing lines
    ax.axhline(z_mid, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(r_mid, color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    show('zeldovich_quadrant_comparison_rz.png', cbar_ax=cbar.ax)
    
    print("\nQuadrant comparison plot saved as 'zeldovich_quadrant_comparison_rz.png'")
    plt.show()


def plot_rz_heatmap(solver, time_label):
    """Plot r-z heatmap of temperature distribution"""
    
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    T_2d = temperature_from_Er(Er_2d)
    
    R, Z = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.pcolormesh(R, Z, T_2d, shading='auto', cmap='hot')
    ax.set_xlabel('r (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_title(f'Cylindrical r-z Zeldovich Wave: Temperature at t = {time_label}', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (keV)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'zeldovich_wave_rz_heatmap_{time_label}.png', dpi=150, bbox_inches='tight')
    print(f"Heatmap saved as 'zeldovich_wave_rz_heatmap_{time_label}.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run cylindrical r-z Zeldovich wave simulation
    print("\nRunning cylindrical r-z Zeldovich wave simulation...")
    solutions, solver = run_zeldovich_wave_rz()
    
    # Plot results
    print("\nPlotting r-z Zeldovich wave results...")
    plot_zeldovich_wave_rz(solutions)
    
    # Create publication-quality comparison plot
    print("\nGenerating publication comparison plot...")
    plot_publication_comparison(solver, solutions, times=[0.1, 1.0])
    
    # Plot r-z heatmaps at multiple times
    if solutions:
        for t, r, T, Er_2d in solutions:
            print(f"\nGenerating r-z heatmap at t = {t:.2f} ns...")
            # Temporarily set solver.Er to this solution for plotting
            solver.Er = Er_2d.flatten()
            plot_rz_heatmap(solver, f"{t:.2f}ns")
    
    print("\n" + "="*70)
    print("Cylindrical r-z Zeldovich wave simulation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
