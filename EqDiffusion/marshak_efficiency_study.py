#!/usr/bin/env python3
"""
Efficiency Study for Marshak Wave Problem

Compares different solution methods:
1. One iteration without nonlinear corrections
2. One iteration with nonlinear corrections
3. Fully converged without nonlinear corrections
4. Fully converged with nonlinear corrections

Metrics:
- Error relative to highly converged reference solution
- Number of linear solves
- Efficiency (error vs computational cost)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT,
    RHO
)


# =============================================================================
# MARSHAK WAVE SETUP (same as marshak_wave.py)
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
    T = (temperature_from_Er(np.abs(Er)))  # keV
    T_min = 0.05  # Minimum temperature to prevent overflow (keV)
    if T < T_min:
        T = T_min
    return 300.0 * T**(-3)


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)
    
    Note: This returns cv per unit MASS. The solver multiplies by RHO internally.
    So we return cv_volumetric / RHO to get the correct volumetric heat capacity.
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def marshak_material_energy(T):
    """Material energy density: e_m = c_v * T"""
    return marshak_specific_heat(T) * T


def marshak_left_bc(Er, x):
    """Left boundary: Dirichlet BC at T = 1 keV"""
    T_bc = 1.0  # keV
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc  # Dirichlet


def marshak_right_bc(Er, x):
    """Right boundary: zero flux"""
    return 0.0, 1.0, 0.0


# =============================================================================
# REFERENCE SOLUTION
# =============================================================================

def generate_reference_solution(t_final=0.1, save_path='marshak_reference.pkl'):
    """Generate highly converged reference solution with dt = 1e-5 ns"""
    
    print("="*70)
    print("GENERATING REFERENCE SOLUTION")
    print("="*70)
    
    # Problem parameters
    r_min = 0.0
    r_max = 0.2  # cm
    n_cells = 50  # Fine mesh
    dt_ref = 1e-5  # ns - very small time step
    T_init = 0.1  # keV
    
    print(f"Domain: [{r_min}, {r_max}] cm with {n_cells} cells")
    print(f"Time step: dt = {dt_ref} ns")
    print(f"Final time: {t_final} ns")
    print(f"Total steps: {int(t_final/dt_ref)}")
    
    # Create solver
    solver = RadiationDiffusionSolver(
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        d=0,  # Planar
        dt=dt_ref,
        max_newton_iter=50,
        newton_tol=1e-10,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    # Enable nonlinear corrections for reference (most accurate)
    solver.use_nonlinear_correction = True
    solver.use_secant_derivative = False
    solver.nonlinear_skip_boundary_cells = 0
    
    # Initial condition
    def initial_Er(r):
        return np.full_like(r, A_RAD * T_init**4)
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    n_steps = int(t_final / dt_ref)
    print(f"\nEvolving solution for {n_steps} steps...")
    
    # Report every 10% progress
    report_interval = max(1, n_steps // 10)
    
    import time
    start_time = time.time()
    
    for step in range(n_steps):
        solver.time_step(n_steps=1, verbose=False)
        
        if (step + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            progress = 100 * (step + 1) / n_steps
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (n_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"  Progress: {progress:.0f}% ({step+1}/{n_steps}), "
                  f"{steps_per_sec:.1f} steps/s, ETA: {eta:.1f}s")
    
    # Get final solution
    r_ref, Er_ref = solver.get_solution()
    T_ref = temperature_from_Er(Er_ref)
    
    print(f"\nReference solution complete!")
    print(f"  Max T = {T_ref.max():.4f} keV")
    print(f"  Max E_r = {Er_ref.max():.4e} GJ/cm^3")
    
    # Save reference solution
    reference_data = {
        'r': r_ref,
        'Er': Er_ref,
        'T': T_ref,
        'dt': dt_ref,
        't_final': t_final,
        'n_cells': n_cells,
        'r_min': r_min,
        'r_max': r_max
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(reference_data, f)
    
    print(f"\nReference solution saved to {save_path}")
    
    return reference_data


def load_or_generate_reference(t_final=0.1, save_path='marshak_reference.pkl'):
    """Load reference solution if exists, otherwise generate it"""
    
    if os.path.exists(save_path):
        print(f"Loading reference solution from {save_path}...")
        with open(save_path, 'rb') as f:
            reference_data = pickle.load(f)
        print(f"  Reference: {reference_data['n_cells']} cells, "
              f"dt = {reference_data['dt']} ns, t_final = {reference_data['t_final']} ns")
        # Check if reference matches requested final time
        if abs(reference_data['t_final'] - t_final) > 1e-10:
            print(f"  Warning: Reference t_final = {reference_data['t_final']} doesn't match requested {t_final}")
            print(f"  Regenerating reference solution...")
            return generate_reference_solution(t_final, save_path)
        return reference_data
    else:
        print(f"Reference solution not found. Generating...")
        return generate_reference_solution(t_final, save_path)


# =============================================================================
# RUN METHOD WITH SPECIFIC PARAMETERS
# =============================================================================

def run_method(dt, use_nonlinear, max_newton_iter, newton_tol, t_final=0.1, n_cells=50):
    """
    Run Marshak wave with specified method parameters
    
    Returns:
        r, Er, T, n_linear_solves
    """
    
    # Problem parameters
    r_min = 0.0
    r_max = 0.2  # cm
    T_init = 0.1  # keV
    
    # Create solver
    solver = RadiationDiffusionSolver(
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        d=0,
        dt=dt,
        max_newton_iter=max_newton_iter,
        newton_tol=newton_tol,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    solver.use_nonlinear_correction = use_nonlinear
    solver.use_secant_derivative = False
    solver.nonlinear_skip_boundary_cells = 0
    
    # Initial condition
    def initial_Er(r):
        return np.full_like(r, A_RAD * T_init**4)
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution - count linear solves
    n_steps = int(t_final / dt)
    n_linear_solves = 0
    
    for step in range(n_steps):
        # Count Newton iterations in this step
        Er_prev = solver.Er.copy()
        
        # Manually perform Newton iterations to count them
        Er_k = solver.Er.copy()
        for k in range(max_newton_iter):
            # Each Newton iteration = one linear solve
            A_tri, rhs = solver.assemble_system(
                Er_k, Er_prev,
                solver.use_nonlinear_correction,
                solver.use_secant_derivative
            )
            solver.apply_boundary_conditions(A_tri, rhs, Er_k, solver.use_nonlinear_correction)
            
            from oneDFV import solve_tridiagonal
            Er_new = solve_tridiagonal(A_tri, rhs)
            n_linear_solves += 1
            
            # Check convergence
            norm_k = np.linalg.norm(Er_k)
            if norm_k < 1e-14:
                norm_k = 1.0
            residual = np.linalg.norm(Er_new - Er_k) / norm_k
            
            if residual < newton_tol:
                Er_k = Er_new
                break
            
            Er_k = Er_new.copy()
        
        solver.Er = Er_k
        solver.Er_old = Er_prev.copy()
    
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    
    return r, Er, T, n_linear_solves


# =============================================================================
# PLOT SOLUTION
# =============================================================================

def plot_solution(r, Er, T, title="Solution", reference=None):
    """
    Plot radiation energy and temperature profiles
    
    Parameters:
    -----------
    r : array
        Position (cm)
    Er : array
        Radiation energy density (GJ/cm³)
    T : array
        Temperature (keV)
    title : str
        Plot title
    reference : dict, optional
        Reference solution with keys 'r', 'Er', 'T'
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot radiation energy
    ax1.plot(r, Er, 'b-', linewidth=2, label='Solution')
    if reference is not None:
        ax1.plot(reference['r'], reference['Er'], 'r--', linewidth=1.5, 
                alpha=0.7, label='Reference')
    ax1.set_xlabel('Position r (cm)', fontsize=12)
    ax1.set_ylabel('Radiation Energy $E_r$ (GJ/cm³)', fontsize=12)
    ax1.set_title(f'{title}: Radiation Energy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature
    ax2.plot(r, T, 'b-', linewidth=2, label='Solution')
    if reference is not None:
        ax2.plot(reference['r'], reference['T'], 'r--', linewidth=1.5,
                alpha=0.7, label='Reference')
    ax2.set_xlabel('Position r (cm)', fontsize=12)
    ax2.set_ylabel('Temperature T (keV)', fontsize=12)
    ax2.set_title(f'{title}: Temperature', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with sanitized filename
    filename = title.replace(' ', '_').replace(',', '').lower() + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Solution plot saved as '{filename}'")
    plt.show()


# =============================================================================
# COMPUTE ERROR RELATIVE TO REFERENCE
# =============================================================================

def compute_error(r_test, Er_test, r_ref, Er_ref):
    """
    Compute L2 error relative to reference solution
    Interpolate if grids don't match
    """
    
    if len(r_test) == len(r_ref) and np.allclose(r_test, r_ref):
        # Same grid
        Er_test_interp = Er_test
    else:
        # Interpolate test solution to reference grid
        Er_test_interp = np.interp(r_ref, r_test, Er_test)
    
    # L2 relative error
    error_abs = np.linalg.norm(Er_test_interp - Er_ref)
    ref_norm = np.linalg.norm(Er_ref)
    error_rel = error_abs / ref_norm if ref_norm > 0 else error_abs
    
    return error_rel


# =============================================================================
# EFFICIENCY STUDY
# =============================================================================

def run_efficiency_study(t_final=0.1):
    """Run complete efficiency study"""
    
    print("\n" + "="*70)
    print("MARSHAK WAVE EFFICIENCY STUDY")
    print(f"Final time: {t_final} ns")
    print("="*70)
    
    # Load or generate reference solution
    reference = load_or_generate_reference(t_final=t_final)
    r_ref = reference['r']
    Er_ref = reference['Er']
    
    # Time steps to test
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    # Methods to compare
    methods = [
        {
            'name': 'One iter, no NL',
            'use_nonlinear': False,
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'color': 'blue',
            'marker': 'o'
        },
        {
            'name': 'One iter, with NL',
            'use_nonlinear': True,
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'color': 'green',
            'marker': 's'
        },
        {
            'name': 'Converged, no NL',
            'use_nonlinear': False,
            'max_newton_iter': 50,
            'newton_tol': 1e-8,
            'color': 'red',
            'marker': '^'
        },
        {
            'name': 'Converged, with NL',
            'use_nonlinear': True,
            'max_newton_iter': 50,
            'newton_tol': 1e-8,
            'color': 'purple',
            'marker': 'D'
        }
    ]
    
    # Storage for results
    results = {method['name']: {'dt': [], 'error': [], 'n_solves': []} 
               for method in methods}
    
    # Run study
    print("\n" + "="*70)
    print("RUNNING METHODS")
    print("="*70)
    
    for method in methods:
        print(f"\n{method['name']}:")
        print("-" * 50)
        
        for i, dt in enumerate(dt_values):
            print(f"  [{i+1}/{len(dt_values)}] dt = {dt:.4f} ns...", end='', flush=True)
            
            import time
            start = time.time()
            
            r, Er, T, n_solves = run_method(
                dt=dt,
                use_nonlinear=method['use_nonlinear'],
                max_newton_iter=method['max_newton_iter'],
                newton_tol=method['newton_tol'],
                t_final=t_final
            )
            
            elapsed = time.time() - start
            error = compute_error(r, Er, r_ref, Er_ref)
            
            results[method['name']]['dt'].append(dt)
            results[method['name']]['error'].append(error)
            results[method['name']]['n_solves'].append(n_solves)
            
            print(f" error = {error:.3e}, n_solves = {n_solves}, time = {elapsed:.2f}s")
            
            # Plot solution for converged methods with smallest dt
            if 'Converged' in method['name'] and dt == min(dt_values):
                print(f"  Plotting {method['name']} solution...")
                plot_solution(r, Er, T, 
                            title=f"{method['name']}, dt={dt} ns",
                            reference=reference)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_results(results, methods)
    
    return results, reference


def plot_results(results, methods):
    """Generate the three comparison plots"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Error vs Δt
    ax = axes[0]
    for method in methods:
        name = method['name']
        dt_vals = results[name]['dt']
        errors = results[name]['error']
        
        ax.loglog(dt_vals, errors, 
                 marker=method['marker'], 
                 color=method['color'],
                 label=name,
                 markersize=8,
                 linewidth=2)
    
    # Add reference lines
    dt_arr = np.array([min(results[methods[0]['name']]['dt']), 
                       max(results[methods[0]['name']]['dt'])])
    ax.loglog(dt_arr, 1e-2 * dt_arr, 'k--', alpha=0.3, linewidth=1, label='O(Δt)')
    ax.loglog(dt_arr, 1e-1 * dt_arr**2, 'k:', alpha=0.3, linewidth=1, label='O(Δt²)')
    
    ax.set_xlabel('Time step Δt (ns)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Error vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Number of linear solves vs Δt
    ax = axes[1]
    for method in methods:
        name = method['name']
        dt_vals = results[name]['dt']
        n_solves = results[name]['n_solves']
        
        ax.loglog(dt_vals, n_solves,
                 marker=method['marker'],
                 color=method['color'],
                 label=name,
                 markersize=8,
                 linewidth=2)
    
    ax.set_xlabel('Time step Δt (ns)', fontsize=12)
    ax.set_ylabel('Number of Linear Solves', fontsize=12)
    ax.set_title('Computational Cost vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()  # Larger dt on left
    
    # Plot 3: Error vs number of linear solves (efficiency plot)
    ax = axes[2]
    for method in methods:
        name = method['name']
        n_solves = results[name]['n_solves']
        errors = results[name]['error']
        
        ax.loglog(n_solves, errors,
                 marker=method['marker'],
                 color=method['color'],
                 label=name,
                 markersize=8,
                 linewidth=2)
    
    ax.set_xlabel('Number of Linear Solves', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Efficiency: Error vs Cost', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('marshak_efficiency_study.png', dpi=200, bbox_inches='tight')
    print(f"\nPlots saved as 'marshak_efficiency_study.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Use short time for quick testing - change to 10.0 for full study
    t_final = 1.0  # ns
    
    results, reference = run_efficiency_study(t_final=t_final)
    
    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print("="*70)
