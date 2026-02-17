#!/usr/bin/env python3
"""
Cylindrical (r-z) Zeldovich Wave Problem with Spherical Symmetry
Radiative heat wave with initial energy pulse at a point

Problem setup:
- All boundaries: reflecting (zero flux)
- Initial condition: delta-function-like energy pulse at (r=0, z=z_center)
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)
- Cylindrical (r-z) geometry with spherical wave expected
- Compare with spherical 1D self-similar solution (N=3)
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

def extract_spherical_profile_rz(Er_2d, r_centers, z_centers, z_source):
    """Extract spherical radial profile from r-z solution
    
    Compute distance from source point (r=0, z=z_source) as sqrt(r^2 + (z-z_source)^2)
    and bin/average temperature values by this radial distance.
    
    Parameters:
    -----------
    Er_2d : ndarray (n_r_cells, n_z_cells)
        2D radiation energy density
    r_centers : ndarray
        r-coordinates of cell centers
    z_centers : ndarray
        z-coordinates of cell centers
    z_source : float
        z-coordinate of source point
    
    Returns:
    --------
    R_vals : ndarray
        Spherical radial distances from source
    T_vals : ndarray
        Averaged temperature at each radial distance
    """
    n_r, n_z = Er_2d.shape
    
    # Compute spherical distance from source for each cell
    R_grid = np.zeros((n_r, n_z))
    for i in range(n_r):
        for j in range(n_z):
            r = r_centers[i]
            z = z_centers[j]
            R_grid[i, j] = np.sqrt(r**2 + (z - z_source)**2)
    
    # Flatten and sort
    R_flat = R_grid.flatten()
    Er_flat = Er_2d.flatten()
    
    sort_idx = np.argsort(R_flat)
    R_sorted = R_flat[sort_idx]
    Er_sorted = Er_flat[sort_idx]
    
    # Use adaptive binning: each bin has approximately the same number of cells
    n_bins = 150
    cells_per_bin = max(10, len(R_sorted) // n_bins)
    
    R_centers_out = []
    T_binned = []
    
    i = 0
    while i < len(R_sorted):
        i_end = min(i + cells_per_bin, len(R_sorted))
        
        # Average radius and temperature in this bin
        R_avg = np.mean(R_sorted[i:i_end])
        T_avg = np.mean(temperature_from_Er(Er_sorted[i:i_end]))
        
        R_centers_out.append(R_avg)
        T_binned.append(T_avg)
        
        i = i_end
    
    return np.array(R_centers_out), np.array(T_binned)


# =============================================================================
# ZELDOVICH WAVE SIMULATION
# =============================================================================

def run_zeldovich_wave_rz_spherical():
    """Run cylindrical r-z Zeldovich wave with spherical symmetry"""
    
    print("="*70)
    print("Cylindrical (r-z) Zeldovich Wave - Spherical Symmetry")
    print("="*70)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)")
    print("  All BCs: Zero flux (reflecting)")
    print("  Initial condition: Point source at (r=0, z=z_center)")
    print("  Comparison: Spherical 1D solution (N=3)")
    print("="*70)
    
    # Problem setup
    # r direction (radial)
    r_min = 0.0
    r_max = 2.0
    n_r_cells = 40
    
    # z direction (axial)
    z_min = 0.0
    z_max = 4.0  # Symmetric domain
    n_z_cells = 80
    
    # Source location
    z_source = (z_min + z_max) / 2.0  # Center of z domain
    
    # Time stepping parameters
    dt = 0.001  # ns (small time step for stability)
    target_times = [0.1, 0.3, 1.0]#, 1.0, 3.0]  # ns
    
    # Create 2D solver with cylindrical geometry
    solver = RadiationDiffusionSolver2D(
        coord1_min=r_min,
        coord1_max=r_max,
        n1_cells=n_r_cells,
        coord2_min=z_min,
        coord2_max=z_max,
        n2_cells=n_z_cells,
        geometry='cylindrical',
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
    
    # Initial condition: cold material
    def initial_Er(r, z):
        T_cold = 0.01  # keV (cold but not too small)
        Er_cold = A_RAD * T_cold**4
        return Er_cold
    
    solver.set_initial_condition(initial_Er)
    
    # Add energy pulse near source point (r=0, z=z_source)
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    
    # Total energy to deposit
    E_total = 1.0  # Total energy in GJ
    
    # Find cells near source point
    j_source = np.argmin(np.abs(solver.coord2_centers - z_source))
    
    # Deposit energy in cells near source with Gaussian distribution
    energy_cells = []
    weights = []
    
    for i in range(solver.n1_cells):
        for j in range(solver.n2_cells):
            r = solver.coord1_centers[i]
            z = solver.coord2_centers[j]
            # Spherical distance from source
            R = np.sqrt(r**2 + (z - z_source)**2)
            weight = np.exp(-R**2 / 0.01)  # Gaussian weight
            if weight > 1e-4:  # Only include significant contributions
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
    print(f"  Source location: (r=0, z={z_source:.3f})")
    print(f"  Source cell index: j={j_source}")
    print(f"  Source cell Er = {Er_2d[0, j_source]:.4e} GJ/cm³")
    print(f"  Source cell T = {temperature_from_Er(Er_2d[0, j_source]):.4f} keV")
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
        
        # Extract spherical radial profile
        R_vals, T_vals = extract_spherical_profile_rz(Er_2d, solver.coord1_centers, 
                                                       solver.coord2_centers, z_source)
        
        # Check energy conservation
        total_Er = np.sum(Er_2d * solver.V_cells)
        T_2d = temperature_from_Er(Er_2d)
        # Compute material energy with spatial coordinates
        mat_energy_2d = np.zeros_like(T_2d)
        for i in range(solver.n1_cells):
            for j in range(solver.n2_cells):
                r_val = solver.coord1_centers[i]
                z_val = solver.coord2_centers[j]
                mat_energy_2d[i, j] = zeldovich_material_energy(T_2d[i, j], r_val, z_val)
        total_mat = np.sum(mat_energy_2d * solver.V_cells)
        total_energy = total_Er + total_mat
        
        solutions.append((current_time, R_vals.copy(), T_vals.copy(), Er_2d.copy(), z_source))
        print(f"  t = {current_time:.3f} ns: max T = {T_2d.max():.4f} keV, total energy = {total_energy:.4e} GJ")
    
    return solutions, solver


def plot_zeldovich_wave_rz_spherical(solutions):
    """Plot spherical Zeldovich wave solutions in r-z geometry"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot temperature profiles (spherical radial)
    for i, (t, R, T, Er_2d, z_source) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # r-z numerical solution (spherical binning)
        ax.plot(R, T, color=color, linewidth=2, linestyle='-', 
                label=f'r-z Numerical t = {t:.2f} ns', marker='o', markersize=3, markevery=10)
        
        # Spherical 1D analytical solution (N=3)
        try:
            T_analytical, R_front = T_of_r_t(R, t, N=3)
            ax.plot(R, T_analytical, color=color, linewidth=1.5, linestyle='--', 
                    alpha=0.7, label=f'1D Spherical t = {t:.2f} ns')
            # Mark wave front
            ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
        except Exception as e:
            print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    ax.set_xlabel('Spherical Radial Distance R (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    #ax.set_title('r-z Zeldovich Wave: Temperature vs. Spherical 1D Solution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0.0, solutions[-1][1][-1])
    
    show('zeldovich_wave_rz_spherical.pdf')
    print("\nPlot saved as 'zeldovich_wave_rz_spherical.pdf'")


def plot_publication_comparison(solver, solutions, times=[0.1, 1.0]):
    """Create publication-quality quadrant comparison plot
    
    Parameters:
    -----------
    solver : RadiationDiffusionSolver2D
        Solver object with mesh information
    solutions : list
        List of (time, R, T, Er_2d, z_source) tuples
    times : list
        Two times for early/late comparison
    """
    # Find solutions closest to requested times
    sol_early = min(solutions, key=lambda s: abs(s[0] - times[0]))
    sol_late = min(solutions, key=lambda s: abs(s[0] - times[1]))
    
    t_early, _, _, Er_early, z_source_early = sol_early
    t_late, _, _, Er_late, z_source_late = sol_late
    
    # Compute temperature fields
    T_early_num = temperature_from_Er(Er_early)
    T_late_num = temperature_from_Er(Er_late)
    
    # Create analytical solutions on the same grid
    R_grid, Z_grid = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    # Compute spherical distance from source
    R_spherical_early = np.sqrt(R_grid**2 + (Z_grid - z_source_early)**2)
    R_spherical_late = np.sqrt(R_grid**2 + (Z_grid - z_source_late)**2)
    
    # Compute analytical temperatures
    T_early_ana, _ = T_of_r_t(R_spherical_early, t_early, N=3)
    T_late_ana, _ = T_of_r_t(R_spherical_late, t_late, N=3)
    
    #redefine R_grid and Z_grid for plotting, it needs to go from -max to max in r and 0 Z_max in z so the total size will be doubled in the r direction
    r_from_neg_to_pos = np.concatenate((-solver.coord1_centers[::-1], solver.coord1_centers))
    R_grid, Z_grid = np.meshgrid(r_from_neg_to_pos, solver.coord2_centers-np.mean(solver.coord2_centers), indexing='ij')

    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    
    # Determine common color scale
    vmin = 0.0
    vmax = max(T_early_num.max(), T_late_num.max(), T_early_ana.max(), T_late_ana.max())
    
    N_z = T_early_num.shape[1]
    combined_solutions = np.zeros((T_early_num.shape[0]*2, T_early_num.shape[1]))
    #now assemble the four quadrants
    #in the top left quadrant the r position must be backward from max to 0 and 
    combined_solutions[0:T_early_num.shape[0], N_z//2:] = T_early_ana[::-1,N_z//2:]
    #top right quadrant
    combined_solutions[T_late_num.shape[0]:, N_z//2:] = T_late_ana[:, N_z//2:]
    #bottom left quadrant both are backward
    combined_solutions[0:T_early_num.shape[0], 0:N_z//2] = T_early_num[::-1, 0:N_z//2]
    #bottom right, z backward
    combined_solutions[T_early_num.shape[0]:, 0:N_z//2] = T_late_num[:, 0:N_z//2]

    #flip up/down in second axis to match orientation
    combined_solutions = combined_solutions[:, ::-1]
    import matplotlib.font_manager as fm
    font = None
    try:
        font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc', size = 12)
    except:
        pass
    # Top left: Early analytical
    ax = axes
    im = ax.pcolormesh(R_grid, Z_grid, combined_solutions, shading='auto', 
                       cmap='plasma', vmin=vmin, vmax=vmax)
    
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_xlabel('r (cm)', fontsize=12)
    #set equal aspect ratio
    ax.set_aspect('equal') #, adjustable='box')
    #ax.set_title(f'Analytical t = {t_early:.2f} ns', fontsize=14)
    # Text properties with white background box
    text_props = dict(fontsize=11, fontweight='bold', ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontproperties=font)
    

    #add dashed circle to indicate wavefront at early time
    try:
        r_front_1 = T_of_r_t(np.array([0.0]), t_early, N=3)[1]
        circle_1 = plt.Circle((0, 0), r_front_1, fill=False, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle_1)
    except:
        pass
    #now a dashed circle for late time
    try:
        r_front_2 = T_of_r_t(np.array([0.0]), t_late, N=3)[1]
        circle_2 = plt.Circle((0, 0), r_front_2, fill=False, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle_2)
    except:
        pass

    
    #draw a horizontal line to separate quadrants
    ax.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)

    ax.text(-1.6, -1.8, "Self-Similar", **text_props)
    ax.text(-1.6, 1.8, "Numerical", **text_props)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label('Temperature (keV)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    show('zeldovich_quadrant_comparison_rz_spherical.png', cbar_ax=cbar.ax)
    print("\nQuadrant comparison plot saved as 'zeldovich_quadrant_comparison_rz_spherical.png'")


def plot_rz_spherical_heatmap(solver, time_label, z_source):
    """Plot r-z heatmap of temperature distribution with spherical contours"""
    
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    T_2d = temperature_from_Er(Er_2d)
    
    R, Z = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(R, Z, T_2d, shading='auto', cmap='hot')
    ax.set_xlabel('r (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_title(f'r-z Zeldovich Wave (Spherical): Temperature at t = {time_label}', 
                 fontsize=14, fontweight='bold')
    
    # Add spherical contours centered at (r=0, z=z_source)
    R_contours = [0.5, 1.0, 1.5, 2.0, 2.5]
    for R_c in R_contours:
        # For each contour, plot arc: r^2 + (z-z_source)^2 = R_c^2
        r_vals = np.linspace(0, min(R_c, solver.coord1_max), 100)
        z_upper = z_source + np.sqrt(np.maximum(0, R_c**2 - r_vals**2))
        z_lower = z_source - np.sqrt(np.maximum(0, R_c**2 - r_vals**2))
        
        ax.plot(r_vals, z_upper, 'w--', alpha=0.3, linewidth=1)
        ax.plot(r_vals, z_lower, 'w--', alpha=0.3, linewidth=1)
    
    # Mark source location
    ax.plot(0, z_source, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=0.5)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (keV)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'zeldovich_wave_rz_spherical_heatmap_{time_label}.png', dpi=150, bbox_inches='tight')
    print(f"Heatmap saved as 'zeldovich_wave_rz_spherical_heatmap_{time_label}.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run spherical Zeldovich wave in r-z geometry
    print("\nRunning spherical Zeldovich wave in r-z geometry...")
    solutions, solver = run_zeldovich_wave_rz_spherical()
    
    # Plot results
    print("\nPlotting r-z spherical Zeldovich wave results...")
    plot_zeldovich_wave_rz_spherical(solutions)
    
    # Plot publication comparison
    print("\nCreating publication quadrant comparison...")
    plot_publication_comparison(solver, solutions, times=[0.1, 1.0])
    
    # Plot r-z heatmaps at multiple times
    if solutions:
        for t, R, T, Er_2d, z_source in solutions:
            print(f"\nGenerating r-z heatmap at t = {t:.2f} ns...")
            # Temporarily set solver.Er to this solution for plotting
            solver.Er = Er_2d.flatten()
            plot_rz_spherical_heatmap(solver, f"{t:.2f}ns", z_source)
    
    print("\n" + "="*70)
    print("Spherical Zeldovich wave in r-z geometry completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
