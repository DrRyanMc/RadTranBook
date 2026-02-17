#!/usr/bin/env python3
"""
Efficiency Study for Linear Gaussian Diffusion Problem

Compares different time integration methods on a linear diffusion problem
with analytical Gaussian solution.

Problem:
- Constant opacity σ_R = 100 cm^-1
- Diffusion coefficient D = c/(3σ_R)
- Linear coupling: cv ∝ T^3 so that e_mat ∝ Er
- Analytical solution: Gaussian spreading with σ(t)² = σ0² + 2*D_eff*t

Methods compared:
1. Implicit Euler (θ=1.0) - one iteration vs converged
2. Crank-Nicolson (θ=0.5) - one iteration vs converged  
3. TR-BDF2 - one iteration vs converged

Metrics:
- Error relative to analytical solution
- Number of linear solves
- Efficiency (error vs computational cost)
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time_module
from oneDFV import (
    RadiationDiffusionSolver,
    temperature_from_Er,
    A_RAD,
    C_LIGHT,
    solve_tridiagonal
)
from plotfuncs import *

# =============================================================================
# LINEAR GAUSSIAN PROBLEM SETUP
# =============================================================================

# Physical parameters
SIGMA_R = 100.0  # cm^-1, constant opacity
D = C_LIGHT / (3.0 * SIGMA_R)  # Diffusion coefficient
K_COUPLING = 1.0e-5  # Weak material coupling for linearity
D_EFF = D / (1.0 + K_COUPLING)  # Effective diffusion with coupling

# Gaussian parameters
X0 = 2.0  # Center (cm)
SIGMA0 = 0.15  # Initial width (cm)
T_PEAK = 1.0  # Peak temperature (keV)
T_BACKGROUND = 0.1  # Background temperature (keV)

# Convert to Er
ER_PEAK = A_RAD * T_PEAK**4
ER_BACKGROUND = A_RAD * T_BACKGROUND**4
AMPLITUDE = ER_PEAK - ER_BACKGROUND


def constant_opacity(Er):
    """Constant opacity for linear diffusion"""
    return SIGMA_R


def cubic_cv(T):
    """cv ∝ T^3 for linear e_mat-Er coupling"""
    return 4.0 * K_COUPLING * A_RAD * T**3


def linear_material_energy(T):
    """e_mat = k*A_RAD*T^4 ∝ Er for linear coupling"""
    return K_COUPLING * A_RAD * T**4


def left_bc(Er, x):
    """Zero flux BC: dEr/dx = 0"""
    return 0.0, 1.0, 0.0


def right_bc(Er, x):
    """Zero flux BC: dEr/dx = 0"""
    return 0.0, 1.0, 0.0


# =============================================================================
# ANALYTICAL SOLUTION
# =============================================================================

def analytical_gaussian_1d(x, t):
    """
    Analytical solution for Gaussian diffusion
    
    Er(x,t) = Er_bg + A*σ0/σ(t) * exp(-(x-x0)²/(2σ(t)²))
    where σ(t)² = σ0² + 2*D_eff*t
    """
    sigma_t_sq = SIGMA0**2 + 2 * D_EFF * t
    sigma_t = np.sqrt(sigma_t_sq)
    
    # Amplitude decreases to conserve integral
    amplitude_t = AMPLITUDE * SIGMA0 / sigma_t
    
    return ER_BACKGROUND + amplitude_t * np.exp(-(x - X0)**2 / (2 * sigma_t_sq))


def generate_analytical_reference(t_final, r_max=4.0, n_cells=400):
    """Generate analytical reference solution"""
    
    print("="*70)
    print("GENERATING ANALYTICAL REFERENCE SOLUTION")
    print("="*70)
    
    r_min = 0.0
    r = np.linspace(r_min, r_max, n_cells)
    
    # Compute analytical solution at final time
    Er_ref = analytical_gaussian_1d(r, t_final)
    T_ref = temperature_from_Er(Er_ref)
    
    # Expected width
    sigma_final = np.sqrt(SIGMA0**2 + 2 * D_EFF * t_final)
    
    print(f"Problem parameters:")
    print(f"  Opacity: σ_R = {SIGMA_R} cm^-1")
    print(f"  Diffusion: D = {D:.6e} cm²/sh")
    print(f"  Effective D: D_eff = {D_EFF:.6e} cm²/sh")
    print(f"  Initial width: σ0 = {SIGMA0} cm")
    print(f"  Final width: σ({t_final}) = {sigma_final:.6f} cm")
    print(f"  Spreading: {(sigma_final/SIGMA0 - 1)*100:.1f}%")
    print()
    print(f"Domain: [{r_min}, {r_max}] cm with {n_cells} cells")
    print(f"Final time: t = {t_final} sh")
    print(f"  Max T = {T_ref.max():.4f} keV")
    print(f"  Max Er = {Er_ref.max():.4e} GJ/cm³")
    
    reference_data = {
        'r': r,
        'Er': Er_ref,
        'T': T_ref,
        't_final': t_final,
        'n_cells': n_cells,
        'r_min': r_min,
        'r_max': r_max,
        'sigma_final': sigma_final,
        'type': 'analytical'
    }
    
    return reference_data


# =============================================================================
# RUN METHOD WITH SPECIFIC PARAMETERS
# =============================================================================

def run_method(dt, max_newton_iter, newton_tol, t_final=0.5, n_cells=200,
               theta=1.0, use_trbdf2=False):
    """
    Run linear Gaussian diffusion with specified method parameters
    
    Returns:
        r, Er, T, n_linear_solves
    """
    
    # Domain parameters
    r_min = 0.0
    r_max = 4.0  # cm
    
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
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    # Initial condition: Gaussian
    def gaussian_Er(r):
        return ER_BACKGROUND + AMPLITUDE * np.exp(-(r - X0)**2 / (2 * SIGMA0**2))
    
    solver.set_initial_condition(gaussian_Er)
    
    # Time evolution
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
        # Standard theta method - manually count Newton iterations
        for step in range(n_steps):
            Er_prev = solver.Er.copy()
            Er_k = solver.Er.copy()
            
            for k in range(max_newton_iter):
                # Each Newton iteration = one linear solve
                A_tri, rhs = solver.assemble_system(Er_k, Er_prev, theta=solver.theta)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                
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
                alpha=0.7, label='Analytical')
    ax1.set_xlabel('Position x (cm)', fontsize=12)
    ax1.set_ylabel('Radiation Energy $E_r$ (GJ/cm³)', fontsize=12)
    ax1.set_title(f'{title}: Radiation Energy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot temperature
    ax2.plot(r, T, 'b-', linewidth=2, label='Solution')
    if reference is not None:
        ax2.plot(reference['r'], reference['T'], 'r--', linewidth=1.5,
                alpha=0.7, label='Analytical')
    ax2.set_xlabel('Position x (cm)', fontsize=12)
    ax2.set_ylabel('Temperature T (keV)', fontsize=12)
    ax2.set_title(f'{title}: Temperature', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with sanitized filename
    filename = title.replace(' ', '_').replace(',', '').lower() + '.pdf'
    show(filename)
    #plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Solution plot saved as '{filename}'")
    #plt.show()


# =============================================================================
# COMPUTE ERROR RELATIVE TO ANALYTICAL SOLUTION
# =============================================================================

def compute_error(r_test, Er_test, r_ref, Er_ref):
    """
    Compute L2 error relative to analytical solution
    
    Only includes cells where Er > 1.1 * Er_background to focus on
    the active diffusion region.
    """
    
    if len(r_test) == len(r_ref) and np.allclose(r_test, r_ref):
        # Same grid
        Er_test_interp = Er_test
    else:
        # Interpolate test solution to reference grid
        Er_test_interp = np.interp(r_ref, r_test, Er_test)
    
    # Threshold to focus on active region
    Er_threshold = 1.1 * ER_BACKGROUND
    
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

def run_efficiency_study(t_final=0.5, n_cells_min=20):
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
    print("LINEAR GAUSSIAN DIFFUSION EFFICIENCY STUDY")
    print(f"Final time: {t_final} ns")
    print(f"Minimum cells: {n_cells_min}")
    print("="*70)
    
    # Domain parameters
    r_min = 0.0
    r_max = 4.0  # cm
    
    # Time steps to test (ns)
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    # Calculate n_cells for each dt to maintain constant dx/(c*dt)
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
        print(f"  dt = {dt:6.4f} sh: n_cells = {n_cells_dict[dt]:4d}, "
              f"dx = {dx:.6f} cm, dx/(c*dt) = {ratio:.6f}")
    
    # Generate analytical reference solution with finest resolution
    n_cells_ref = max(n_cells_dict.values())
    print(f"\nGenerating analytical reference with n_cells = {n_cells_ref}")
    reference = generate_analytical_reference(t_final=t_final, r_max=r_max, n_cells=n_cells_ref)
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
            'max_newton_iter': 50,
            'newton_tol': 1e-8,
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
            'max_newton_iter': 50,
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
            'max_newton_iter': 50,
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
            print(f"  [{i+1}/{len(dt_values)}] dt = {dt:.4f} sh, n_cells = {n_cells}...", 
                  end='', flush=True)
            
            start = time_module.time()
            
            r, Er, T, n_solves = run_method(
                dt=dt,
                max_newton_iter=method['max_newton_iter'],
                newton_tol=method['newton_tol'],
                theta=method['theta'],
                use_trbdf2=method['use_trbdf2'],
                t_final=t_final, n_cells=n_cells
            )
            
            elapsed = time_module.time() - start
            error = compute_error(r, Er, r_ref, Er_ref)
            
            results[method['name']]['dt'].append(dt)
            results[method['name']]['error'].append(error)
            results[method['name']]['n_solves'].append(n_solves)
            
            print(f" error = {error:.3e}, n_solves = {n_solves}, time = {elapsed:.2f}s")
            #print total energy to verify conservation
            total_energy = np.sum(Er*(r_max - r_min)/len(r))  + np.sum(linear_material_energy(temperature_from_Er(Er)) * (r_max - r_min)/len(r))  # GJ/cm^2
            print(f"    Total Radiation Energy = {total_energy:.6e} GJ/cm")
            
            # Plot solution for converged methods with smallest dt
            if 'converged' in method['name'].lower() and dt == min(dt_values):
                print(f"  Plotting {method['name']} solution...")
                plot_solution(r, Er, T, 
                            title=f"{method['name']}, dt={dt} sh",
                            reference=reference)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
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
    #get max error from implicit euler converged and have that 1/2 that be the reference line at max dt
    max_error = max(results['Implicit Euler (converged)']['error'])
    ax.loglog(dt_arr, 0.5 * max_error * dt_arr / dt_arr.max(), 'k--', alpha=0.3, linewidth=1, label=r'O($\Delta t$)')
    #now for dt^2 line use 1/2 of trbdf2 converged at max dt
    max_error_dt2 = max(results['TR-BDF2 (converged)']['error'])
    ax.loglog(dt_arr, 0.5 * max_error_dt2 * (dt_arr / dt_arr.max())**2, 'k:', alpha=0.3, linewidth=1, label=r'O($\Delta t^2$)')
    
    ax.set_xlabel(r'Time step $\Delta t$ (ns)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    #no titles for figures in publications
    #ax.set_title('Error vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    show('linear_gaussian_error_vs_dt.pdf')
    print(f"\nFigure 1 saved as 'linear_gaussian_error_vs_dt.png'")
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
    #no titles for figures in publications
    #ax.set_title('Computational Cost vs Time Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()  # Larger dt on left
    
    plt.tight_layout()
    show('linear_gaussian_cost_vs_dt.pdf')
    print(f"Figure 2 saved as 'linear_gaussian_cost_vs_dt.pdf'")
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
    #no titles for figures in publications
    #ax.set_title('Efficiency: Error vs Cost', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    show('linear_gaussian_efficiency.pdf')
    print(f"Figure 3 saved as 'linear_gaussian_efficiency.pdf   '")
    plt.close(fig3)
    plt.show()
    
    return results, reference


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    t_final = 0.5  # ns
    n_cells_min = 40  # Minimum cells (for largest dt)
    results, reference = run_efficiency_study(t_final=t_final, n_cells_min=n_cells_min)
    
    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print("="*70)
