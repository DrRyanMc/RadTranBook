#!/usr/bin/env python3
"""
Marshak Wave with Refined Zone Test

Tests custom face arrays for mesh refinement in the Marshak wave problem.
The wave propagates through a refined region to verify the implementation
handles non-uniform grids with custom face arrays correctly.

Setup:
- Wave propagates in x-direction (1D slab with one cell in y)
- Refined mesh near left boundary where wave enters
- Compare uniform vs refined mesh solutions using both x_stretch and custom faces
"""

import sys
import os
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
# MESH GENERATION
# =============================================================================

def generate_custom_refined_faces(x_min, x_max, n_cells, refine_region=(0.0, 0.1), 
                                   n_refine_cells=20):
    """
    Generate custom face positions with refinement in specified region
    
    Parameters:
    -----------
    x_min, x_max : float
        Domain boundaries
    n_cells : int
        Total number of cells
    refine_region : tuple
        (x_start, x_end) of region to refine
    n_refine_cells : int
        Number of cells in refined region
    
    Returns:
    --------
    x_faces : ndarray
        Custom face positions
    """
    x_refine_start, x_refine_end = refine_region
    
    # Calculate cells outside refined region
    n_outside = n_cells - n_refine_cells
    
    # Check if refinement region is at the start
    if x_refine_start == x_min:
        # Refined region at left, coarse on right
        x_faces_refine = np.linspace(x_refine_start, x_refine_end, n_refine_cells + 1)
        x_faces_coarse = np.linspace(x_refine_end, x_max, n_outside + 1)[1:]
        x_faces = np.concatenate([x_faces_refine, x_faces_coarse])
    elif x_refine_end == x_max:
        # Coarse on left, refined at right
        x_faces_coarse = np.linspace(x_min, x_refine_start, n_outside + 1)
        x_faces_refine = np.linspace(x_refine_start, x_refine_end, n_refine_cells + 1)[1:]
        x_faces = np.concatenate([x_faces_coarse, x_faces_refine])
    else:
        # Refined in middle
        n_left = max(1, n_outside // 2)
        n_right = n_outside - n_left
        
        x_faces_left = np.linspace(x_min, x_refine_start, n_left + 1)
        x_faces_refine = np.linspace(x_refine_start, x_refine_end, n_refine_cells + 1)[1:]
        x_faces_right = np.linspace(x_refine_end, x_max, n_right + 1)[1:]
        
        x_faces = np.concatenate([x_faces_left, x_faces_refine, x_faces_right])
    
    return x_faces


# =============================================================================
# MATERIAL PROPERTIES (same as standard Marshak wave)
# =============================================================================

def marshak_opacity(T):
    """σ = 300 * T^-3"""
    n = 3
    T_min = 0.01
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)


def marshak_rosseland_opacity(T, x, y):
    return marshak_opacity(T)


def marshak_planck_opacity(T, x, y):
    return marshak_opacity(T)


def marshak_specific_heat(T, x, y):
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO


def marshak_material_energy(T, x, y):
    cv_specific = marshak_specific_heat(T, x, y)
    return RHO * cv_specific * T


def marshak_inverse_material_energy(e, x, y):
    cv_volumetric = 0.3
    return e / cv_volumetric


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def bc_blackbody_incoming(phi, pos, t, boundary='left', geometry='cartesian'):
    """Blackbody at T = 1 keV"""
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc


def bc_zero_flux(phi, pos, t, boundary='right', geometry='cartesian'):
    """Zero flux boundary"""
    return 0.0, 1.0, 0.0


# =============================================================================
# RUN MARSHAK WAVE WITH GIVEN MESH
# =============================================================================

def run_marshak_wave(nx_cells=50, x_stretch=None, x_faces=None, y_faces=None, label='uniform'):
    """Run Marshak wave with specified mesh
    
    Parameters:
    -----------
    nx_cells : int
        Number of cells in x-direction (ignored if x_faces provided)
    x_stretch : float or None
        Stretching factor (1.0 = uniform, >1.0 = refined near x_min)
        Ignored if x_faces is provided
    x_faces : ndarray or None
        Custom face positions in x-direction
    y_faces : ndarray or None
        Custom face positions in y-direction (optional)
    label : str
        Label for this run
    """
    print("="*80)
    print(f"Marshak Wave - {label} mesh")
    if x_faces is not None:
        print(f"  Using custom face arrays with {len(x_faces)-1} cells in x")
    else:
        print(f"  nx_cells = {nx_cells}, x_stretch = {x_stretch}")
    print("="*80)
    
    # Domain and boundary conditions
    x_min, x_max = 0.0, 0.5
    y_min, y_max = 0.0, 0.1
    ny_cells = 1
    
    boundary_funcs = {
        'left': bc_blackbody_incoming,
        'right': bc_zero_flux,
        'bottom': bc_zero_flux,
        'top': bc_zero_flux
    }
    
    # Time stepping
    dt = 0.02  # ns
    target_times = [1.0, 5.0]
    
    # Create solver with specified mesh
    if x_faces is not None:
        # Use custom face arrays
        if y_faces is None:
            y_faces = np.linspace(y_min, y_max, ny_cells + 1)
        
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_faces=x_faces,
            y_faces=y_faces,
            geometry='cartesian', dt=dt,
            max_newton_iter=20,
            newton_tol=1e-6,
            rosseland_opacity_func=marshak_rosseland_opacity,
            planck_opacity_func=marshak_planck_opacity,
            specific_heat_func=marshak_specific_heat,
            material_energy_func=marshak_material_energy,
            inverse_material_energy_func=marshak_inverse_material_energy,
            boundary_funcs=boundary_funcs,
            theta=1.0
        )
    else:
        # Use standard grid with stretching
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_min=x_min, x_max=x_max, nx_cells=nx_cells,
            y_min=y_min, y_max=y_max, ny_cells=ny_cells,
            geometry='cartesian', dt=dt,
            x_stretch=x_stretch if x_stretch is not None else 1.0, 
            y_stretch=1.0,
            max_newton_iter=20,
            newton_tol=1e-6,
            rosseland_opacity_func=marshak_rosseland_opacity,
            planck_opacity_func=marshak_planck_opacity,
            specific_heat_func=marshak_specific_heat,
            material_energy_func=marshak_material_energy,
            inverse_material_energy_func=marshak_inverse_material_energy,
            boundary_funcs=boundary_funcs,
            theta=1.0
        )
    
    # Initial condition
    T_init = 0.01
    phi_init = C_LIGHT * A_RAD * T_init**4
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    # Time evolution
    print("\nTime evolution:")
    current_time = 0.0
    solutions = []
    
    for target_time in target_times:
        while current_time < target_time:
            if current_time + dt > target_time:
                temp_dt = target_time - current_time
                solver.dt = temp_dt
                steps_needed = 1
            else:
                steps_needed = 1
            
            solver.time_step(n_steps=steps_needed, verbose=False)
            current_time += solver.dt
            
            if solver.dt != dt:
                solver.dt = dt
        
        # Store solution
        T_2d = solver.get_T_2d()
        phi_2d = solver.get_phi_2d()
        
        solutions.append({
            'time': current_time,
            'x': solver.x_centers.copy(),
            'T': T_2d[:, 0],
            'phi': phi_2d[:, 0]
        })
        
        print(f"  t = {current_time:.1f} ns:")
        print(f"    T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
    
    return solutions, solver


# =============================================================================
# COMPARISON AND PLOTTING
# =============================================================================

def plot_mesh_comparison(solutions_uniform, solutions_refined, solver_uniform, solver_refined):
    """Compare uniform vs refined mesh solutions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red']
    
    # Get mesh info
    x_uniform = solver_uniform.x_centers
    x_refined = solver_refined.x_centers
    
    for idx, (sol_u, sol_r, color) in enumerate(zip(solutions_uniform, solutions_refined, colors)):
        t = sol_u['time']
        
        # Temperature comparison
        ax = axes[0, 0]
        ax.plot(sol_u['x'], sol_u['T'], 'o-', color=color, alpha=0.6, 
                label=f't = {t:.0f} ns (uniform)', markersize=4)
        ax.plot(sol_r['x'], sol_r['T'], 's-', color=color, alpha=0.6,
                label=f't = {t:.0f} ns (refined)', markersize=3)
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Temperature Profiles', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Difference
        ax = axes[0, 1]
        # Interpolate to common grid for comparison
        x_common = np.linspace(0, 0.5, 200)
        T_u_interp = np.interp(x_common, sol_u['x'], sol_u['T'])
        T_r_interp = np.interp(x_common, sol_r['x'], sol_r['T'])
        diff = T_u_interp - T_r_interp
        ax.plot(x_common, diff, color=color, linewidth=2, label=f't = {t:.0f} ns')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T_uniform - T_refined (keV)', fontsize=11)
        ax.set_title('Temperature Difference', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # Mesh visualization
    ax = axes[1, 0]
    x_u_faces = solver_uniform.x_faces
    x_r_faces = solver_refined.x_faces
    y_pos_u = np.ones_like(x_u_faces)
    y_pos_r = np.ones_like(x_r_faces) * 0.5
    ax.scatter(x_u_faces, y_pos_u, marker='|', s=100, c='blue', label='Uniform mesh')
    ax.scatter(x_r_faces, y_pos_r, marker='|', s=100, c='red', label='Refined mesh')
    ax.set_ylim(0, 1.5)
    ax.set_xlabel('Position x (cm)', fontsize=11)
    ax.set_ylabel('Mesh type', fontsize=11)
    ax.set_title('Mesh Comparison', fontsize=12, fontweight='bold')
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['Refined', 'Uniform'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Cell size distribution
    ax = axes[1, 1]
    dx_uniform = np.diff(x_u_faces)
    dx_refined = np.diff(x_r_faces)
    x_mid_u = 0.5 * (x_u_faces[:-1] + x_u_faces[1:])
    x_mid_r = 0.5 * (x_r_faces[:-1] + x_r_faces[1:])
    ax.semilogy(x_mid_u, dx_uniform, 'o-', color='blue', label='Uniform', markersize=4)
    ax.semilogy(x_mid_r, dx_refined, 's-', color='red', label='Refined', markersize=3)
    ax.set_xlabel('Position x (cm)', fontsize=11)
    ax.set_ylabel('Cell width Δx (cm)', fontsize=11)
    ax.set_title('Cell Size Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.suptitle('Marshak Wave: Uniform vs Refined Mesh', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('marshak_wave_refined_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'marshak_wave_refined_comparison.png'")


def compute_error_metrics(solutions_uniform, solutions_refined):
    """Compute quantitative error metrics"""
    print("\n" + "="*80)
    print("Error Metrics: Uniform vs Refined Mesh")
    print("="*80)
    
    for sol_u, sol_r in zip(solutions_uniform, solutions_refined):
        t = sol_u['time']
        
        # Interpolate to common fine grid
        x_common = np.linspace(0, 0.5, 500)
        T_u_interp = np.interp(x_common, sol_u['x'], sol_u['T'])
        T_r_interp = np.interp(x_common, sol_r['x'], sol_r['T'])
        
        # Compute errors
        abs_diff = np.abs(T_u_interp - T_r_interp)
        rel_diff = abs_diff / (np.maximum(T_u_interp, T_r_interp) + 1e-10)
        
        max_abs = np.max(abs_diff)
        mean_abs = np.mean(abs_diff)
        max_rel = np.max(rel_diff)
        mean_rel = np.mean(rel_diff)
        
        print(f"\nt = {t:.1f} ns:")
        print(f"  Max absolute difference: {max_abs:.6e} keV")
        print(f"  Mean absolute difference: {mean_abs:.6e} keV")
        print(f"  Max relative difference: {max_rel:.6e}")
        print(f"  Mean relative difference: {mean_rel:.6e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MARSHAK WAVE WITH REFINED MESH TEST")
    print("="*80)
    print("\nThis test compares uniform and refined meshes using both")
    print("x_stretch parameter and custom face arrays to verify the")
    print("implementation handles non-uniform grids correctly.")
    print("="*80)
    
    # Run with uniform mesh
    nx_cells = 50
    solutions_uniform, solver_uniform = run_marshak_wave(
        nx_cells=nx_cells, x_stretch=1.0, label='Uniform'
    )
    
    # Run with refined mesh using x_stretch
    # A stretch factor of 1.05-1.08 gives good refinement
    solutions_stretch, solver_stretch = run_marshak_wave(
        nx_cells=nx_cells, x_stretch=1.06, label='Refined (x_stretch)'
    )
    
    # Run with custom face arrays
    # Generate custom refined faces with high resolution near left boundary
    x_faces_custom = generate_custom_refined_faces(
        x_min=0.0, x_max=0.5, n_cells=nx_cells,
        refine_region=(0.0, 0.15),  # Refine first 15 cm where wave enters
        n_refine_cells=30  # 60% of cells in refined region
    )
    y_faces_custom = np.linspace(0.0, 0.1, 2)  # Single cell in y
    solutions_custom, solver_custom = run_marshak_wave(
        x_faces=x_faces_custom, y_faces=y_faces_custom, 
        label='Refined (custom faces)'
    )
    
    # Compare all solutions
    print("\n" + "="*80)
    print("Error Metrics: Uniform vs x_stretch Refined")
    print("="*80)
    compute_error_metrics(solutions_uniform, solutions_stretch)
    
    print("\n" + "="*80)
    print("Error Metrics: Uniform vs Custom Faces Refined")
    print("="*80)
    compute_error_metrics(solutions_uniform, solutions_custom)
    
    print("\n" + "="*80)
    print("Error Metrics: x_stretch vs Custom Faces")
    print("="*80)
    compute_error_metrics(solutions_stretch, solutions_custom)
    
    # Plot comparison (uniform vs custom faces)
    plot_mesh_comparison(solutions_uniform, solutions_custom, 
                        solver_uniform, solver_custom)
    
    # Additional plot comparing x_stretch vs custom faces
    plot_mesh_comparison_3way(solutions_uniform, solutions_stretch, solutions_custom,
                             solver_uniform, solver_stretch, solver_custom)
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


def plot_mesh_comparison_3way(sol_uniform, sol_stretch, sol_custom,
                              solver_uniform, solver_stretch, solver_custom):
    """Compare all three mesh approaches"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red']
    
    for idx, color in enumerate(colors):
        t = sol_uniform[idx]['time']
        
        # Temperature comparison
        ax = axes[0, 0]
        ax.plot(sol_uniform[idx]['x'], sol_uniform[idx]['T'], 'o-', color=color, 
                alpha=0.5, label=f't={t:.0f} ns (uniform)', markersize=4)
        ax.plot(sol_stretch[idx]['x'], sol_stretch[idx]['T'], 's-', color=color,
                alpha=0.5, label=f't={t:.0f} ns (x_stretch)', markersize=3)
        ax.plot(sol_custom[idx]['x'], sol_custom[idx]['T'], '^-', color=color,
                alpha=0.5, label=f't={t:.0f} ns (custom)', markersize=3)
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Temperature Profiles', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1)
        
        # Zoom in on wave front
        ax = axes[0, 1]
        x_zoom_max = 0.2  # Focus on first 20 cm
        mask_u = sol_uniform[idx]['x'] <= x_zoom_max
        mask_s = sol_stretch[idx]['x'] <= x_zoom_max
        mask_c = sol_custom[idx]['x'] <= x_zoom_max
        ax.plot(sol_uniform[idx]['x'][mask_u], sol_uniform[idx]['T'][mask_u], 
                'o-', color=color, alpha=0.5, label=f't={t:.0f} ns (uniform)', markersize=6)
        ax.plot(sol_stretch[idx]['x'][mask_s], sol_stretch[idx]['T'][mask_s], 
                's-', color=color, alpha=0.5, label=f't={t:.0f} ns (x_stretch)', markersize=5)
        ax.plot(sol_custom[idx]['x'][mask_c], sol_custom[idx]['T'][mask_c], 
                '^-', color=color, alpha=0.5, label=f't={t:.0f} ns (custom)', markersize=5)
        ax.set_xlabel('Position x (cm)', fontsize=11)
        ax.set_ylabel('T (keV)', fontsize=11)
        ax.set_title('Wave Front Detail', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # Mesh visualization
    ax = axes[1, 0]
    x_u_faces = solver_uniform.x_faces
    x_s_faces = solver_stretch.x_faces
    x_c_faces = solver_custom.x_faces
    ax.scatter(x_u_faces, np.ones_like(x_u_faces) * 1.0, marker='|', s=100, 
              c='blue', label='Uniform', alpha=0.7)
    ax.scatter(x_s_faces, np.ones_like(x_s_faces) * 0.6, marker='|', s=100, 
              c='green', label='x_stretch', alpha=0.7)
    ax.scatter(x_c_faces, np.ones_like(x_c_faces) * 0.2, marker='|', s=100, 
              c='red', label='Custom faces', alpha=0.7)
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel('Position x (cm)', fontsize=11)
    ax.set_ylabel('Mesh type', fontsize=11)
    ax.set_title('Mesh Comparison', fontsize=12, fontweight='bold')
    ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_yticklabels(['Custom', 'x_stretch', 'Uniform'])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Cell size distribution (zoom on refined region)
    ax = axes[1, 1]
    dx_uniform = np.diff(x_u_faces)
    dx_stretch = np.diff(x_s_faces)
    dx_custom = np.diff(x_c_faces)
    x_mid_u = 0.5 * (x_u_faces[:-1] + x_u_faces[1:])
    x_mid_s = 0.5 * (x_s_faces[:-1] + x_s_faces[1:])
    x_mid_c = 0.5 * (x_c_faces[:-1] + x_c_faces[1:])
    
    ax.semilogy(x_mid_u, dx_uniform, 'o-', color='blue', label='Uniform', markersize=4)
    ax.semilogy(x_mid_s, dx_stretch, 's-', color='green', label='x_stretch', markersize=3)
    ax.semilogy(x_mid_c, dx_custom, '^-', color='red', label='Custom faces', markersize=3)
    ax.set_xlim(0, 0.25)  # Focus on refined region
    ax.set_xlabel('Position x (cm)', fontsize=11)
    ax.set_ylabel('Cell width Δx (cm)', fontsize=11)
    ax.set_title('Cell Size Distribution (refined region)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.suptitle('Marshak Wave: Mesh Refinement Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('marshak_wave_refined_3way_comparison.png', dpi=150, bbox_inches='tight')
    print("\n3-way comparison plot saved as 'marshak_wave_refined_3way_comparison.png'")


if __name__ == "__main__":
    main()
