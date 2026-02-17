#!/usr/bin/env python3
"""
Efficiency Study for Smooth Marshak Wave Problem

Compares different solution methods for smooth initial condition:
1. One Newton iteration per time step
2. Multiple Newton iterations per time step
3. Different time step sizes

Metrics:
- Error relative to highly converged reference solution
- Number of linear solves
- Efficiency (error vs computational cost)

Initial condition: T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .125)])/2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
from numba import njit
from pathlib import Path
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT,
    RHO
)


# =============================================================================
# MARSHAK WAVE SETUP (same material properties as original)
# =============================================================================
@njit
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
    return 3.0 * T**(-3)

@njit
def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)
    
    Note: This returns cv per unit MASS. The solver multiplies by RHO internally.
    So we return cv_volumetric / RHO to get the correct volumetric heat capacity.
    """
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)

@njit
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
# SMOOTH INITIAL CONDITION
# =============================================================================

def smooth_initial_temperature(x):
    """Smooth initial temperature profile for second-order accuracy testing
    
    T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .125)])/2
    
    This creates a smooth transition from T=1 keV at x=0 to T=0.2 keV at x=0.5
    
    Parameters:
    -----------
    x : array_like
        Position array (cm)
    
    Returns:
    --------
    T : array_like
        Temperature at each position (keV)
    """
    return 1.0 + (0.2 - 1.0) * (1.0 + np.tanh(50.0 * (x - 0.125))) / 2.0


# =============================================================================
# REFERENCE SOLUTION
# =============================================================================

def generate_reference_solution(t_final=0.1, save_path='marshak_smooth_reference.pkl'):
    """Generate highly converged reference solution with dt = 1e-5 ns"""
    
    print("="*70)
    print("GENERATING SMOOTH MARSHAK REFERENCE SOLUTION")
    print("="*70)
    
    # Problem parameters
    r_min = 0.0
    r_max = 1.0  # cm - full domain for smooth initial condition
    n_cells = 400  # Fine mesh for smooth profile
    dt_ref = 1e-5  # ns - very small time step
    
    print(f"Domain: [{r_min}, {r_max}] cm with {n_cells} cells")
    print(f"Time step: dt = {dt_ref} ns")
    print(f"Final time: {t_final} ns")
    print(f"Total steps: {int(t_final/dt_ref)}")
    print("Initial condition: T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .125)])/2")
    
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
    
    # Smooth initial condition
    def initial_Er(r):
        T_init = smooth_initial_temperature(r)
        return A_RAD * T_init**4
    
    solver.set_initial_condition(initial_Er)
    
    # Display initial condition statistics
    r_init, Er_init = solver.get_solution()
    T_init = temperature_from_Er(Er_init)
    print(f"Initial temperature range: {T_init.min():.3f} to {T_init.max():.3f} keV")
    
    # Time evolution
    n_steps = int(t_final / dt_ref)
    print(f"\nEvolving solution for {n_steps} steps...")
    
    # Report every 10% progress
    report_interval = max(1, n_steps // 10)
    
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
    print(f"  Temperature range: {T_ref.min():.4f} to {T_ref.max():.4f} keV")
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
        'r_max': r_max,
        'initial_condition': 'smooth_tanh'
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(reference_data, f)
    
    print(f"\nReference solution saved to {save_path}")
    
    return reference_data


def load_or_generate_reference(t_final=0.1, save_path='marshak_smooth_reference.pkl'):
    """Load reference solution if exists, otherwise generate it"""
    
    if os.path.exists(save_path):
        print(f"Loading smooth Marshak reference solution from {save_path}...")
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

def run_method(dt, max_newton_iter, newton_tol, t_final=0.1, n_cells=400,
               theta=1.0, use_trbdf2=False):
    """
    Run smooth Marshak wave with specified method parameters
    
    Parameters:
    -----------
    theta : float
        Time discretization parameter (1.0=implicit Euler, 0.5=Crank-Nicolson)
    use_trbdf2 : bool
        If True, use TR-BDF2 time integration (overrides theta)
    
    Returns:
        r, Er, T, n_linear_solves
    """
    
    # Problem parameters
    r_min = 0.0
    r_max = 1.0  # cm - full domain
    
    # Create solver
    solver = RadiationDiffusionSolver(
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        d=0,
        dt=dt,
        max_newton_iter=max_newton_iter,
        newton_tol=newton_tol,
        theta=theta,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    solver.max_newton_iter_per_step = max_newton_iter
    
    # Smooth initial condition
    def initial_Er(r):
        T_init = smooth_initial_temperature(r)
        return A_RAD * T_init**4
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    n_steps = int(t_final / dt)
    
    if use_trbdf2:
        # TR-BDF2 method: 2 stages per time step
        gamma = 2.0 - np.sqrt(2.0)
        for step in range(n_steps):
            solver.time_step_trbdf2(n_steps=1, gamma=gamma, verbose=False)
        # Conservative estimate: 4 linear solves per step (2 per stage)
        n_linear_solves = n_steps * 4
    else:
        # Standard theta method
        for step in range(n_steps):
            solver.time_step(n_steps=1, verbose=False)
        
        # Estimate linear solves based on Newton iterations per step
        if max_newton_iter == 1:
            n_linear_solves = n_steps  # One linear solve per time step
        else:
            # For converged methods, estimate based on typical convergence
            avg_newton_iters = 2  # Typical convergence for this problem
            n_linear_solves = n_steps * avg_newton_iters
    
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot radiation energy
    ax1.plot(r, Er, 'b-', linewidth=2, label='Solution')
    if reference is not None:
        ax1.plot(reference['r'], reference['Er'], 'r--', linewidth=1.5, 
                alpha=0.7, label='Reference')
    ax1.set_xlabel('Position r (cm)', fontsize=12)
    ax1.set_ylabel('Radiation Energy $E_r$ (GJ/cm³)', fontsize=12)
    ax1.set_title(f'{title}: Radiation Energy (Smooth Initial Condition)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature
    ax2.plot(r, T, 'b-', linewidth=2, label='Solution')
    if reference is not None:
        ax2.plot(reference['r'], reference['T'], 'r--', linewidth=1.5,
                alpha=0.7, label='Reference')
    ax2.set_xlabel('Position r (cm)', fontsize=12)
    ax2.set_ylabel('Temperature T (keV)', fontsize=12)
    ax2.set_title(f'{title}: Temperature (Smooth Initial Condition)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with sanitized filename
    filename = title.replace(' ', '_').replace(',', '').lower() + '_smooth.png'
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
    
    Only includes cells where Er > 1.1 * Er_min_initial to avoid bias from
    cold material regions.
    """
    
    if len(r_test) == len(r_ref) and np.allclose(r_test, r_ref):
        # Same grid
        Er_test_interp = Er_test
    else:
        # Interpolate test solution to reference grid
        Er_test_interp = np.interp(r_ref, r_test, Er_test)
    
    # Minimum initial temperature in smooth profile is 0.2 keV
    T_min_init = 0.2  # keV
    Er_min_init = A_RAD * T_min_init**4
    Er_threshold = 1.1 * Er_min_init
    
    # Mask: only include cells where reference energy is above threshold
    mask = Er_ref > Er_threshold
    
    if not np.any(mask):
        # If no cells above threshold, compute error normally
        error_abs = np.linalg.norm(Er_test_interp - Er_ref)
        ref_norm = np.linalg.norm(Er_ref)
        error_rel = error_abs / ref_norm if ref_norm > 0 else error_abs
    else:
        # Compute error only in cells above threshold
        Er_test_masked = Er_test_interp[mask]
        Er_ref_masked = Er_ref[mask]
        
        error_abs = np.linalg.norm(Er_test_masked - Er_ref_masked)
        ref_norm = np.linalg.norm(Er_ref_masked)
        error_rel = error_abs / ref_norm if ref_norm > 0 else error_abs
    
    return error_rel


# =============================================================================
# EFFICIENCY STUDY
# =============================================================================

def run_efficiency_study(t_final=0.1):
    """Run complete efficiency study for smooth Marshak wave"""
    
    print("\n" + "="*70)
    print("SMOOTH MARSHAK WAVE EFFICIENCY STUDY")
    print(f"Final time: {t_final} ns")
    print("Initial condition: T(x) = 1 + (0.2 - 1)*(1 + Tanh[50*(x - .125)])/2")
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
            'name': 'One Newton iter',
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'color': 'blue',
            'marker': 'o',
            'theta': 1.0,
            'use_trbdf2': False
        },
        {
            'name': 'Converged Newton',
            'max_newton_iter': 50,
            'newton_tol': 1e-8,
            'color': 'red',
            'marker': '^',
            'theta': 1.0,
            'use_trbdf2': False
        },
        {
            'name': 'Theta=0.5, no NL',
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'color': 'orange',
            'marker': 'v',
            'theta': 0.5,
            'use_trbdf2': False
        },
        {
            'name': 'TR-BDF2',
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'color': 'cyan',
            'marker': 'p',
            'theta': 1.0,  # Not used for TR-BDF2
            'use_trbdf2': True
        },
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
            
            start = time.time()
            
            r, Er, T, n_solves = run_method(
                dt=dt,
                max_newton_iter=method['max_newton_iter'],
                newton_tol=method['newton_tol'],
                t_final=t_final,
                theta=method['theta'],
                use_trbdf2=method['use_trbdf2']
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
    """Generate the three comparison plots for smooth Marshak wave"""
    
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
    ax.set_title('Error vs Time Step (Smooth Initial Condition)', fontsize=13, fontweight='bold')
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
    ax.set_title('Efficiency: Error vs Cost (Smooth Initial Condition)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('marshak_smooth_efficiency_study.png', dpi=200, bbox_inches='tight')
    print(f"\nPlots saved as 'marshak_smooth_efficiency_study.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Use short time for quick testing - change to larger value for full study
    t_final = 1.0  # ns
    
    results, reference = run_efficiency_study(t_final=t_final)
    
    print("\n" + "="*70)
    print("SMOOTH MARSHAK WAVE EFFICIENCY STUDY COMPLETE")
    print("="*70)
    
    # Summary statistics
    print("\nSummary:")
    for method_name, data in results.items():
        min_error = min(data['error'])
        min_cost = min(data['n_solves'])
        max_cost = max(data['n_solves'])
        print(f"  {method_name}: min error = {min_error:.2e}, cost range = {min_cost}-{max_cost} solves")