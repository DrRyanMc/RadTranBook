#!/usr/bin/env python3
"""
2D Zeldovich Wave Problem
Radiative heat wave with initial energy pulse at center

Problem setup:
- All boundaries: reflecting (zero flux)
- Initial condition: delta-function-like energy pulse at center (x=0, z=0)
- Material opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)
- 2D Cartesian geometry with radial symmetry expected
- Compare center line (r from origin) with cylindrical 1D self-similar solution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
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
    """Specific heat: c_v = 3e-6 GJ/(cm^3·keV)
    
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


# =============================================================================
# ZELDOVICH WAVE BOUNDARY CONDITIONS (all reflecting)
# =============================================================================

def zeldovich_bc_reflecting(Er_boundary, coord1_val, coord2_val, geometry='cartesian'):
    """Reflecting boundary: zero flux
    
    Robin BC: A*E_r + B*(dE_r/dn) = C
    Zero flux: 0*E_r + 1*dE_r/dn = 0
    """
    return 0.0, 1.0, 0.0  # Reflecting boundary


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_radial_profile(Er_2d, coord1_centers, coord2_centers):
    """Extract radial profile from 2D solution
    
    For each radial distance from origin, average the temperature values
    at that approximate distance. Uses adaptive binning to ensure each
    bin has sufficient cells for averaging.
    
    Parameters:
    -----------
    Er_2d : ndarray (n1_cells, n2_cells)
        2D radiation energy density
    coord1_centers : ndarray
        x-coordinates of cell centers
    coord2_centers : ndarray
        z-coordinates of cell centers
    
    Returns:
    --------
    r_vals : ndarray
        Radial distances from origin
    T_vals : ndarray
        Averaged temperature at each radial distance
    """
    n1, n2 = Er_2d.shape
    
    # Compute radial distance for each cell
    r_grid = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            r_grid[i, j] = np.sqrt(coord1_centers[i]**2 + coord2_centers[j]**2)
    
    # Get unique radial distances (binned)
    r_flat = r_grid.flatten()
    Er_flat = Er_2d.flatten()
    
    # Sort by radius
    sort_idx = np.argsort(r_flat)
    r_sorted = r_flat[sort_idx]
    Er_sorted = Er_flat[sort_idx]
    
    # Use adaptive binning: each bin has approximately the same number of cells
    n_bins = 150
    cells_per_bin = max(10, len(r_sorted) // n_bins)  # At least 10 cells per bin
    
    r_centers = []
    T_binned = []
    
    i = 0
    while i < len(r_sorted):
        i_end = min(i + cells_per_bin, len(r_sorted))
        
        # Average radius and temperature in this bin
        r_avg = np.mean(r_sorted[i:i_end])
        T_avg = np.mean(temperature_from_Er(Er_sorted[i:i_end]))
        
        r_centers.append(r_avg)
        T_binned.append(T_avg)
        
        i = i_end
    
    return coord1_centers, temperature_from_Er(Er_2d)[:,n2//2]#np.array(r_centers), np.array(T_binned)


# =============================================================================
# ZELDOVICH WAVE SIMULATION
# =============================================================================

def run_zeldovich_wave_2d():
    """Run 2D Zeldovich wave simulation with central delta-function source"""
    
    print("="*70)
    print("2D Zeldovich Wave Problem")
    print("="*70)
    print("Material properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)")
    print("  All BCs: Zero flux (reflecting)")
    print("  Initial condition: Delta-function energy pulse at center")
    print("="*70)
    
    # Problem setup - symmetric domain centered at origin
    # x direction
    x_min = -2.0
    x_max = 2.0
    n_x_cells = 80
    
    # z direction
    z_min = -2.0
    z_max = 2.0
    n_z_cells = 80
    
    # Time stepping parameters
    dt = 0.001  # ns (small time step for stability)
    target_times = [0.1, 0.3, 1.0]#, 3.0]  # ns
    if n_z_cells > 30:
        target_times = [0.1, 0.3, 1.0]  # high res means run to later times
    
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
        rosseland_opacity_func=zeldovich_opacity,
        specific_heat_func=zeldovich_specific_heat,
        material_energy_func=zeldovich_material_energy,
        left_bc_func=zeldovich_bc_reflecting,
        right_bc_func=zeldovich_bc_reflecting,
        bottom_bc_func=zeldovich_bc_reflecting,
        top_bc_func=zeldovich_bc_reflecting,
        theta=1.0,  # Implicit Euler for stability
        use_jfnk=False  # Use direct solver
    )
    
    # Initial condition: cold material with delta-function pulse at center
    def initial_Er(x, z):
        T_cold = 0.01  # keV (cold but not too small)
        Er_cold = A_RAD * T_cold**4
        return Er_cold
    
    solver.set_initial_condition(initial_Er)
    
    # Add energy pulse to central cells (near origin)
    # Find cells closest to origin
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    
    # Find center indices
    i_center = np.argmin(np.abs(solver.coord1_centers))
    j_center = np.argmin(np.abs(solver.coord2_centers))
    
    # Total energy to deposit (similar to 1D case)
    E_total = 1.0  # Total energy in GJ
    
    # Deposit energy in central cell and neighbors (Gaussian-like distribution)
    energy_cells = []
    weights = []
    for di in range(-2, 3):
        for dj in range(-2, 3):
            i = i_center + di
            j = j_center + dj
            if 0 <= i < solver.n1_cells and 0 <= j < solver.n2_cells:
                r_dist = np.sqrt((solver.coord1_centers[i])**2 + (solver.coord2_centers[j])**2)
                weight = np.exp(-r_dist**2 / 0.01)  # Gaussian weight
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
    print(f"  Center cell Er = {Er_2d[i_center, j_center]:.4e} GJ/cm³")
    print(f"  Center cell T = {temperature_from_Er(Er_2d[i_center, j_center]):.4f} keV")
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
        
        # Extract radial profile
        r_vals, T_vals = extract_radial_profile(Er_2d, solver.coord1_centers, solver.coord2_centers)
        
        # Also check energy conservation
        total_Er = np.sum(Er_2d * solver.V_cells)
        T_2d = temperature_from_Er(Er_2d)
        total_mat = np.sum(zeldovich_material_energy(T_2d) * solver.V_cells)
        total_energy = total_Er + total_mat
        
        solutions.append((current_time, r_vals.copy(), T_vals.copy(), Er_2d.copy()))
        print(f"  t = {current_time:.3f} ns: max T = {T_2d.max():.4f} keV, total energy = {total_energy:.4e} GJ")
    
    return solutions, solver


def plot_zeldovich_wave_2d(solutions):
    """Plot 2D Zeldovich wave solutions and comparison with cylindrical 1D analytical"""
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 8 / 1.518))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot temperature profiles (radial from center)
    ax = axes
    for i, (t, r, T, Er_2d) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # 2D numerical solution (radial average)
        ax.plot(r, T, color=color, linewidth=2, linestyle='-', 
                label=f'2D Numerical t = {t:.2f} ns', marker='o', markersize=3, markevery=5)
        
        # Cylindrical 1D analytical solution (N=2)
        try:
            T_analytical, R_front = T_of_r_t(r, t, N=2)
            ax.plot(r, T_analytical, color=color, linewidth=1.5, linestyle='--', 
                    alpha=0.7, label=f'1D Cylindrical t = {t:.2f} ns')
            # Mark wave front
            ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
        except Exception as e:
            print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    ax.set_xlabel('Radial Distance (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    #ax.set_title('2D Zeldovich Wave: Temperature vs. Cylindrical 1D Solution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0, solutions[-1][1][-1])
    
    # # Plot radiation energy density profiles
    # ax = axes[1]
    # for i, (t, r, T, Er_2d) in enumerate(solutions):
    #     color = colors[i % len(colors)]
        
    #     # Convert temperature to Er for plotting
    #     Er_radial = A_RAD * T**4
    #     ax.plot(r, Er_radial, color=color, linewidth=2, linestyle='-',
    #             label=f'2D t = {t:.2f} ns', marker='o', markersize=3, markevery=5)
        
    #     # Analytical Er
    #     try:
    #         T_analytical, R_front = T_of_r_t(r, t, N=2)
    #         Er_analytical = A_RAD * T_analytical**4
    #         ax.plot(r, Er_analytical, color=color, linewidth=1.5, linestyle='--', 
    #                 alpha=0.7, label=f'1D Cylindrical t = {t:.2f} ns')
    #     except:
    #         pass
    
    # ax.set_xlabel('Radial Distance from Center r (cm)', fontsize=12)
    # ax.set_ylabel('Radiation Energy Density $E_r$ (GJ/cm³)', fontsize=12)
    # ax.set_title('2D Zeldovich Wave: Radiation Energy Density', fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=9, loc='best')
    # ax.set_xlim(0, solutions[-1][1][-1])
    # ax.set_yscale('log')
    
    plt.tight_layout()
    show('zeldovich_wave_2d_xy.pdf')
    print("\nPlot saved as 'zeldovich_wave_2d.pdf'")


def plot_2d_heatmap(solver, time_label):
    """Plot 2D heatmap of temperature distribution"""
    
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    T_2d = temperature_from_Er(Er_2d)
    
    X, Z = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.pcolormesh(X, Z, T_2d, shading='auto', cmap='hot')
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_title(f'2D Zeldovich Wave: Temperature at t = {time_label}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Add circle contours to show radial symmetry
    r_contours = [0.5, 1.0, 1.5, 2.0]
    for r in r_contours:
        circle = plt.Circle((0, 0), r, fill=False, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature (keV)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'zeldovich_wave_2d_heatmap_{time_label}.png', dpi=150, bbox_inches='tight')
    print(f"Heatmap saved as 'zeldovich_wave_2d_heatmap_{time_label}.png'")
    plt.show()


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
            x = solver.coord1_centers[i]
            z = solver.coord2_centers[j]
            r = np.sqrt(x**2 + z**2)
            
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
    
    # Create composite array (2*n1 x 2*n2) with white separators
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
    dx = solver.coord1_centers[1] - solver.coord1_centers[0]
    dz = solver.coord2_centers[1] - solver.coord2_centers[0]
    
    x_extended = np.concatenate([
        solver.coord1_centers,
        [solver.coord1_centers[-1] + dx],  # separator
        solver.coord1_centers + solver.coord1_centers[-1] + 2*dx
    ])
    
    z_extended = np.concatenate([
        solver.coord2_centers,
        [solver.coord2_centers[-1] + dz],  # separator
        solver.coord2_centers + solver.coord2_centers[-1] + 2*dz
    ])
    
    X_extended, Z_extended = np.meshgrid(solver.coord1_centers, solver.coord2_centers, indexing='ij')
    
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
    im = ax.pcolormesh(X_extended, Z_extended, T_composite, shading='auto', 
                       cmap=cmap, vmin=T_min, vmax=T_max)
    
    # Add labels and styling
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('z (cm)', fontsize=12)
    ax.set_aspect('equal')
    
    # Add text labels in each quadrant
    x_mid = (solver.coord1_centers[0] + solver.coord1_centers[-1]) / 2
    z_mid = (solver.coord2_centers[0] + solver.coord2_centers[-1]) / 2
    x_offset = solver.coord1_centers[-1] + 2*dx
    z_offset = solver.coord2_centers[-1] + 2*dz
    
    # Text properties with white background box
    text_props = dict(fontsize=11, fontweight='bold', ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontproperties=font)
    
    # # Top-left: Analytical early
    ax.text(-1.6, -1.8, "Self-Similar", **text_props)
    ax.text(-1.6, 1.8, "Numerical", **text_props)

    #add dashed circle to indicate wavefront at early time
    try:
        r_front_1 = T_of_r_t(np.array([0.0]), time1, N=2)[1]
        circle_1 = plt.Circle((0, 0), r_front_1, fill=False, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle_1)
    except:
        pass
    #now a dashed circle for late time
    try:
        r_front_2 = T_of_r_t(np.array([0.0]), time2, N=2)[1]
        circle_2 = plt.Circle((0, 0), r_front_2, fill=False, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle_2)
    except:
        pass
    
    # # Top-right: Numerical early
    # ax.text(x_mid + x_offset, solver.coord2_centers[-1]*0.95, 
    #         f'Numerical\nt = {time1:.2f} ns', **text_props)
    
    # # Bottom-left: Analytical late
    # ax.text(x_mid, z_mid + z_offset, 
    #         f'Analytical\nt = {time2:.2f} ns', **text_props)
    
    # # Bottom-right: Numerical late
    # ax.text(x_mid + x_offset, z_mid + z_offset, 
    #         f'Numerical\nt = {time2:.2f} ns', **text_props)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (keV)', fontsize=12, fontweight='bold')
    #draw a horizontal line to separate quadrants
    ax.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.tight_layout()
    show('zeldovich_quadrant_comparison.png',cbar_ax=cbar.ax)
    
    print("\nQuadrant comparison plot saved as 'zeldovich_quadrant_comparison.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run 2D Zeldovich wave simulation
    print("\nRunning 2D Zeldovich wave simulation...")
    solutions, solver = run_zeldovich_wave_2d()
    
    # Plot results
    print("\nPlotting 2D Zeldovich wave results...")
    plot_zeldovich_wave_2d(solutions)
    
    # Create publication-quality comparison plot
    print("\nGenerating publication comparison plot...")
    plot_publication_comparison(solver, solutions, times=[0.1, 1.0])
    
    # Plot 2D heatmaps at multiple times
    if solutions:
        for t, r, T, Er_2d in solutions:
            print(f"\nGenerating 2D heatmap at t = {t:.2f} ns...")
            # Temporarily set solver.Er to this solution for plotting
            solver.Er = Er_2d.flatten()
            plot_2d_heatmap(solver, f"{t:.2f}ns")
    
    print("\n" + "="*70)
    print("2D Zeldovich wave simulation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
