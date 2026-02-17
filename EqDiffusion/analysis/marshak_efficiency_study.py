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
from plotfuncs import *


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
# REFERENCE SOLUTION (Self-Similar Analytical Solution)
# =============================================================================

def generate_self_similar_solution(t_final=0.1, r_max=0.2, n_cells=400):
    """Generate analytical self-similar solution for Marshak wave
    
    Self-similar solution parameters from Marshak wave theory:
    - xi_max = 1.11305 (maximum extent of similarity variable)
    - omega = 0.05989 (correction parameter)
    - T(xi) = [(1 - xi/xi_max)*(1 + omega*xi/xi_max)]^(1/6)
    - xi = r / sqrt(K*t)
    - K = 8*a*c / [(n+4)*3*sigma_0*rho*cv]
    
    For this problem:
    - n = 3 (opacity exponent: sigma_R = 300*T^-3)
    - sigma_0 = 300 cm^-1
    - cv = 0.3 GJ/(cm^3*keV)
    - rho = 1.0 g/cm^3
    """
    
    print("="*70)
    print("GENERATING SELF-SIMILAR REFERENCE SOLUTION")
    print("="*70)
    
    # Self-similar solution parameters
    xi_max = 1.11305
    omega = 0.05989
    
    # Diffusion constant K
    n = 3  # opacity exponent
    sigma_0 = 300.0  # cm^-1
    cv = 0.3  # GJ/(cm^3*keV)
    K_const = 8 * A_RAD * C_LIGHT / ((n + 4) * 3 * sigma_0 * RHO * cv)
    
    print(f"K constant: {K_const:.6e} cm^2/ns")
    print(f"Wave front at t={t_final} ns: r ~ {xi_max * np.sqrt(K_const * t_final):.4f} cm")
    
    # Problem parameters
    r_min = 0.0
    
    print(f"Domain: [{r_min}, {r_max}] cm with {n_cells} cells")
    print(f"Final time: t = {t_final} ns")
    print()
    
    # Create spatial grid
    r = np.linspace(r_min, r_max, n_cells)
    
    # Self-similar solution function
    def self_similar_T(xi):
        """Temperature as function of similarity variable xi"""
        return np.where(
            xi < xi_max,
            np.power((1 - xi/xi_max) * (1 + omega*xi/xi_max), 1.0/6.0),
            0.0  # Beyond wave front
        )
    #load in data from MarshakP04573.csv, first column is xi/xi_max, third column is thf
    loaded_table = np.loadtxt(Path(__file__).parent / 'MarshakP04573.csv', delimiter=',', skiprows=0)
    xi_table = loaded_table[:,0]*xi_max
    thf_table = loaded_table[:,2]

    #use scipy interpolate for better accuracy
    from scipy.interpolate import interp1d
    T_interp = interp1d(xi_table, thf_table, kind='linear', fill_value=0.0, bounds_error=False)
    # Compute similarity variable xi = r / sqrt(K*t)
    xi = r / np.sqrt(K_const * t_final)
    
    # Evaluate temperature from self-similar solution
    T_ref = T_interp(xi)
    
    # Convert to radiation energy density
    Er_ref = A_RAD * T_ref**4
    
    print(f"Self-similar solution generated!")
    print(f"  Max T = {T_ref.max():.4f} keV")
    print(f"  Max Er = {Er_ref.max():.4e} GJ/cm^3")
    print(f"  Wave front location: {xi_max * np.sqrt(K_const * t_final):.4f} cm")
    
    # Package reference data
    reference_data = {
        'r': r,
        'Er': Er_ref,
        'T': T_ref,
        't_final': t_final,
        'n_cells': n_cells,
        'r_min': r_min,
        'r_max': r_max,
        'K_const': K_const,
        'xi_max': xi_max,
        'type': 'self_similar'
    }
    
    return reference_data


def load_or_generate_reference(t_final=0.1, r_max=0.2, n_cells=400):
    """Generate self-similar analytical reference solution
    
    Note: This now always generates the analytical solution on-the-fly
    since it's very fast and always exact (no need to cache).
    """
    
    print(f"Generating self-similar analytical reference solution...")
    return generate_self_similar_solution(t_final=t_final, r_max=r_max, n_cells=n_cells)


# =============================================================================
# RUN METHOD WITH SPECIFIC PARAMETERS
# =============================================================================

def run_method(dt, max_newton_iter, newton_tol, t_final=0.1, n_cells=400, 
               theta=1.0, use_trbdf2=False):
    """
    Run Marshak wave with specified method parameters
    
    Parameters:
    -----------
    dt : float
        Time step size (ns)
    max_newton_iter : int
        Maximum Newton iterations per time step
    newton_tol : float
        Newton convergence tolerance
    theta : float
        Time discretization parameter (1.0=implicit Euler, 0.5=Crank-Nicolson)
    use_trbdf2 : bool
        If True, use TR-BDF2 time integration (overrides theta)
    
    Returns:
        r, Er, T, n_linear_solves
    """
    
    # Problem parameters
    r_min = 0.0
    r_max = 0.2  # cm
    T_init = 0.01  # keV
    
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
    
    # Initial condition
    def initial_Er(r):
        return np.full_like(r, A_RAD * T_init**4)
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution - count linear solves
    n_steps = int(t_final / dt)
    n_linear_solves = 0
    
    if use_trbdf2:
        # TR-BDF2 method: 2 stages per time step, count iterations manually
        gamma = 2.0 - np.sqrt(2.0)
        original_dt = solver.dt
        
        for step in range(n_steps):
            Er_n = solver.Er.copy()
            
            # Stage 1: Trapezoidal rule to intermediate point
            solver.dt = gamma * original_dt
            Er_k = Er_n.copy()
            
            for k in range(max_newton_iter):
                # Assemble with theta=0.5 (trapezoidal)
                A_tri, rhs = solver.assemble_system(Er_k, Er_n, theta=0.5)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                
                from oneDFV import solve_tridiagonal
                Er_new = solve_tridiagonal(A_tri, rhs)
                n_linear_solves += 1
                
                # Check convergence (after first iteration)
                if k > 0:
                    norm_k = np.linalg.norm(Er_k)
                    if norm_k < 1e-14:
                        norm_k = 1.0
                    residual = np.linalg.norm(Er_new - Er_k) / norm_k
                    if residual < newton_tol:
                        Er_k = Er_new
                        break
                Er_k = Er_new.copy()
            
            Er_intermediate = Er_k
            
            # Stage 2: BDF2 from t^n and t^{n+gamma} to t^{n+1}
            solver.dt = original_dt
            Er_k = Er_intermediate.copy()
            
            for k in range(max_newton_iter):
                # Assemble BDF2 system
                A_tri, rhs = solver.assemble_system_bdf2(Er_k, Er_n, Er_intermediate, gamma)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                
                from oneDFV import solve_tridiagonal
                Er_new = solve_tridiagonal(A_tri, rhs)
                n_linear_solves += 1
                
                # Check convergence (after first iteration)
                if k > 0:
                    norm_k = np.linalg.norm(Er_k)
                    if norm_k < 1e-14:
                        norm_k = 1.0
                    residual = np.linalg.norm(Er_new - Er_k) / norm_k
                    if residual < newton_tol:
                        Er_k = Er_new
                        break
                Er_k = Er_new.copy()
            
            solver.Er = Er_k
            solver.Er_old = Er_n.copy()
        
        solver.dt = original_dt
    else:
        # Standard theta method
        for step in range(n_steps):
            # Count Newton iterations in this step
            Er_prev = solver.Er.copy()
            
            # Manually perform Newton iterations to count them
            Er_k = solver.Er.copy()
            for k in range(max_newton_iter):
                # Each Newton iteration = one linear solve
                A_tri, rhs = solver.assemble_system(Er_k, Er_prev, theta=solver.theta)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                
                from oneDFV import solve_tridiagonal
                Er_new = solve_tridiagonal(A_tri, rhs)
                n_linear_solves += 1
                
                # Check convergence (requires at least 2 iterations to verify)
                if k > 0:  # Only check after first iteration
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
    
    Only includes cells where Er > 1.1 * Er_initial to avoid bias from
    cold material in front of the wave.
    """
    
    if len(r_test) == len(r_ref) and np.allclose(r_test, r_ref):
        # Same grid
        Er_test_interp = Er_test
    else:
        # Interpolate test solution to reference grid
        Er_test_interp = np.interp(r_ref, r_test, Er_test)
    
    # Initial energy density (T_init = 0.05 keV from run_method)
    T_init = 0.05  # keV
    Er_init = A_RAD * T_init**4
    Er_threshold = 1.1 * Er_init
    
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

def run_efficiency_study(t_final=0.1, n_cells_min=40):
    """Run complete efficiency study
    
    Parameters:
    -----------
    t_final : float
        Final simulation time (ns)
    n_cells_min : int
        Minimum number of cells (used for largest dt)
    
    Note: n_cells scales with dt to maintain constant dx/(c*dt) ratio
    """
    
    print("\n" + "="*70)
    print("MARSHAK WAVE EFFICIENCY STUDY")
    print(f"Final time: {t_final} ns")
    print(f"Minimum cells: {n_cells_min}")
    print("="*70)
    
    # Domain parameters
    r_min = 0.0
    r_max = 0.2  # cm
    
    # Time steps to test
    dt_values = [0.1, 0.05, 0.01, 0.005]#, 0.0005]
    
    # Calculate n_cells for each dt to maintain constant dx/(c*dt)
    # For the largest dt, use n_cells_min
    # For smaller dt: n_cells = n_cells_min * (dt_max / dt)
    dt_max = max(dt_values)
    n_cells_dict = {}
    for dt in dt_values:
        n_cells = int(n_cells_min * dt_max / dt)
        n_cells_dict[dt] = n_cells
    
    print("\nSpatial/temporal resolution scaling:")
    print("  (maintaining constant dx/(c*dt) ratio)")
    for dt in dt_values:
        dx = r_max / n_cells_dict[dt]
        ratio = dx / (C_LIGHT * dt)
        print(f"  dt = {dt:6.4f} ns: n_cells = {n_cells_dict[dt]:4d}, "
              f"dx = {dx:.6f} cm, dx/(c*dt) = {ratio:.6f}")
    
    # Generate reference solution with finest resolution
    n_cells_ref = max(n_cells_dict.values())
    print(f"\nGenerating reference with n_cells = {n_cells_ref}")
    reference = load_or_generate_reference(t_final=t_final, r_max=r_max, n_cells=n_cells_ref)
    r_ref = reference['r']
    Er_ref = reference['Er']
    
    # Methods to compare
    # Style convention: converged = solid line + unfilled marker, one iter = dashed line + filled marker
    methods = [
        {
            'name': 'Implicit Euler (one iter)',
            'max_newton_iter': 1,
            'newton_tol': 1e-6,
            'theta': 1.0,
            'use_trbdf2': False,
            'color': 'blue',
            'marker': 'o',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'Implicit Euler (converged)',
            'max_newton_iter': 20,
            'newton_tol': 1e-6,
            'theta': 1.0,
            'use_trbdf2': False,
            'color': 'blue',
            'marker': 'o',
            'linestyle': '-',
            'fillstyle': 'none'
        },
        {
            'name': 'Crank-Nicolson (one iter)',
            'max_newton_iter': 1,
            'newton_tol': 1e-8,
            'theta': 0.5,
            'use_trbdf2': False,
            'color': 'green',
            'marker': 's',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'Crank-Nicolson (converged)',
            'max_newton_iter': 20,
            'newton_tol': 1e-8,
            'theta': 0.5,
            'use_trbdf2': False,
            'color': 'green',
            'marker': 's',
            'linestyle': '-',
            'fillstyle': 'none'
        },
        {
            'name': 'TR-BDF2 (one iter)',
            'max_newton_iter': 1,
            'newton_tol': 1e-8,
            'theta': 1.0,  # Ignored for TR-BDF2
            'use_trbdf2': True,
            'color': 'red',
            'marker': '^',
            'linestyle': '--',
            'fillstyle': 'full'
        },
        {
            'name': 'TR-BDF2 (converged)',
            'max_newton_iter': 20,
            'newton_tol': 1e-8,
            'theta': 1.0,  # Ignored for TR-BDF2
            'use_trbdf2': True,
            'color': 'red',
            'marker': '^',
            'linestyle': '-',
            'fillstyle': 'none'
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
            n_cells = n_cells_dict[dt]  # Use scaled n_cells for this dt
            print(f"  [{i+1}/{len(dt_values)}] dt = {dt:.4f} ns, n_cells = {n_cells}...", 
                  end='', flush=True)
            
            import time
            start = time.time()
            
            r, Er, T, n_solves = run_method(
                dt=dt,
                max_newton_iter=method['max_newton_iter'],
                newton_tol=method['newton_tol'],
                theta=method['theta'],
                use_trbdf2=method['use_trbdf2'],
                t_final=t_final, n_cells=n_cells
            )
            
            elapsed = time.time() - start
            error = compute_error(r, Er, r_ref, Er_ref)
            
            results[method['name']]['dt'].append(dt)
            results[method['name']]['error'].append(error)
            results[method['name']]['n_solves'].append(n_solves)
            
            print(f" error = {error:.3e}, n_solves = {n_solves}, time = {elapsed:.2f}s")
            
            # Plot solution for converged methods with smallest dt, make case of name all lowercase
            if 'converged' in method['name'].lower() and dt == min(dt_values):
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
    """Generate the three comparison plots as separate figures"""
    
    # =========================================================================
    # Figure 1: Error vs Δt
    # =========================================================================
    fig1 = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    for method in methods:
        name = method['name']
        dt_vals = results[name]['dt']
        errors = results[name]['error']
        
        # Only show converged methods in legend, remove "(converged)" text
        if 'converged' in name.lower():
            legend_label = name.replace(' (converged)', '')
        else:
            legend_label = '_nolegend_'  # Hide from legend
        
        ax.loglog(dt_vals, errors, 
                 marker=method['marker'], 
                 color=method['color'],
                 linestyle=method['linestyle'],
                 fillstyle=method['fillstyle'],
                 label=legend_label,
                 markersize=8,
                 linewidth=2,
                 markeredgewidth=1.5)
    
    # Add reference lines
    dt_arr = np.array([min(results[methods[0]['name']]['dt']), 
                       max(results[methods[0]['name']]['dt'])])
    #get max error from implicit euler one step and have that 1/2 that be the reference line at max dt
    max_error = max(results['Implicit Euler (one iter)']['error'])
    ax.loglog(dt_arr, 0.5 * max_error * dt_arr / dt_arr.max(), 'k--', alpha=0.3, linewidth=1, label=r'O($\Delta t$)')
    #now for dt^2 line use 1/2 of trbdf2 converged at max dt
    max_error_dt2 = max(results['TR-BDF2 (converged)']['error'])
    #ax.loglog(dt_arr, 0.5 * max_error_dt2 * (dt_arr / dt_arr.max())**2, 'k:', alpha=0.3, linewidth=1, label=r'O($\Delta t^2$)')
    #ax.loglog(dt_arr, 1e-2 * dt_arr, 'k--', alpha=0.3, linewidth=1, label='O(Δt)')
    #ax.loglog(dt_arr, 1e-1 * dt_arr**2, 'k:', alpha=0.3, linewidth=1, label='O(Δt²)')
    
    ax.set_xlabel(r'Time step $\Delta t$ (ns)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    #no titles
    #ax.set_title('Error vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    show('marshak_error_vs_dt_Ti05.pdf')
    print(f"\nFigure 1 saved as 'marshak_error_vs_dt_Ti05.pdf'")
    plt.close(fig1)
    
    # =========================================================================
    # Figure 2: Number of linear solves vs Δt
    # =========================================================================
    fig2 = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    for method in methods:
        name = method['name']
        dt_vals = results[name]['dt']
        n_solves = results[name]['n_solves']
        
        # Only show converged methods in legend, remove "(converged)" text
        if 'converged' in name.lower():
            legend_label = name.replace(' (converged)', '')
        else:
            legend_label = '_nolegend_'  # Hide from legend
        
        ax.loglog(dt_vals, n_solves,
                 marker=method['marker'],
                 color=method['color'],
                 linestyle=method['linestyle'],
                 fillstyle=method['fillstyle'],
                 label=legend_label,
                 markersize=8,
                 linewidth=2,
                 markeredgewidth=1.5)
    
    ax.set_xlabel(r'Time step $\Delta t$ (ns)', fontsize=12)
    ax.set_ylabel('Number of Linear Solves', fontsize=12)
    #no titles
    #ax.set_title('Computational Cost vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()  # Larger dt on left
    
    plt.tight_layout()
    show('marshak_cost_vs_dt_Ti05.pdf')
    print(f"Figure 2 saved as 'marshak_cost_vs_dt_Ti05.pdf'")
    plt.close(fig2)
    
    # =========================================================================
    # Figure 3: Error vs number of linear solves (efficiency plot)
    # =========================================================================
    fig3 = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    for method in methods:
        name = method['name']
        n_solves = results[name]['n_solves']
        errors = results[name]['error']
        
        # Only show converged methods in legend, remove "(converged)" text
        if 'converged' in name.lower():
            legend_label = name.replace(' (converged)', '')
        else:
            legend_label = '_nolegend_'  # Hide from legend
        
        ax.loglog(n_solves, errors,
                 marker=method['marker'],
                 color=method['color'],
                 linestyle=method['linestyle'],
                 fillstyle=method['fillstyle'],
                 label=legend_label,
                 markersize=8,
                 linewidth=2,
                 markeredgewidth=1.5)
    
    ax.set_xlabel('Number of Linear Solves', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    #no titles
    #ax.set_title('Efficiency: Error vs Cost', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    show('marshak_efficiency_Ti05.pdf')
    print(f"Figure 3 saved as 'marshak_efficiency_Ti05.pdf'")
    plt.close(fig3)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Use short time for quick testing - change to 10.0 for full study
    t_final = 10.0  # ns
    n_cells_min = 20  # Minimum cells (for largest dt)
    results, reference = run_efficiency_study(t_final=t_final, n_cells_min=n_cells_min)
    
    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print("="*70)
