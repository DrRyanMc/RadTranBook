#!/usr/bin/env python3
"""
Efficiency Study for Linear Gaussian Diffusion Problem in Spherical Geometry

Compares different time integration methods on a linear diffusion problem
with analytical Gaussian-like solution in spherical coordinates (d=2).

Problem:
- Constant opacity σ_R = 100 cm^-1
- Diffusion coefficient D = c/(3σ_R)
- Linear coupling: cv ∝ T^3 so that e_mat ∝ Er
- Spherical geometry: ∇·(D∇Er) in spherical coordinates

Methods compared:
1. Implicit Euler (θ=1.0) - one iteration vs converged
2. Crank-Nicolson (θ=0.5) - one iteration vs converged  
3. TR-BDF2 - one iteration vs converged

Metrics:
- Error relative to analytical solution
- Number of linear solves
- Efficiency (error vs computational cost)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
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
R0 = 0.0  # Center at origin for true spherical symmetry
SIGMA0 = 0.5  # Initial width (cm)
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
    """Zero flux BC at inner boundary: dEr/dr = 0"""
    return 0.0, 1.0, 0.0


def right_bc(Er, x):
    """Zero flux BC at outer boundary: dEr/dr = 0"""
    return 0.0, 1.0, 0.0


# =============================================================================
# ANALYTICAL SOLUTION (Spherical Geometry Approximation)
# =============================================================================

def analytical_gaussian_spherical(r, t):
    """
    Analytical solution for Gaussian diffusion in spherical geometry centered at origin
    
    Er(r,t) = Er_bg + A*σ0/σ(t) * exp(-r²/(2σ(t)²))
    where σ(t)² = σ0² + 2*D_eff*t
    
    For radially symmetric diffusion from the origin.
    """
    sigma_t_sq = SIGMA0**2 + 2 * D_EFF * t
    sigma_t = np.sqrt(sigma_t_sq)
    
    # Amplitude decreases to conserve integral
    amplitude_t = AMPLITUDE * SIGMA0 / sigma_t
    
    return ER_BACKGROUND + amplitude_t * np.exp(-r**2 / (2 * sigma_t_sq))


def cell_average_analytical(r_faces, t, n_quad=10):
    """
    Compute cell-averaged analytical solution for spherical geometry
    
    For each cell i, computes:
        <Er>_i = (1/V_i) ∫∫∫ Er(r) dV
               = (1/V_i) ∫[r_{i-1/2} to r_{i+1/2}] Er(r) * 4πr² dr
    
    Parameters:
    -----------
    r_faces : array
        Face positions (length n_cells+1)
    t : float
        Time at which to evaluate
    n_quad : int
        Number of quadrature points per cell
        
    Returns:
    --------
    Er_avg : array
        Cell-averaged values (length n_cells)
    """
    from scipy import integrate
    
    n_cells = len(r_faces) - 1
    Er_avg = np.zeros(n_cells)
    
    for i in range(n_cells):
        r_left = r_faces[i]
        r_right = r_faces[i+1]
        
        # Volume of spherical shell
        V_i = (4.0 * np.pi / 3.0) * (r_right**3 - r_left**3)
        
        # Integrate Er(r) * 4π*r² over the cell
        def integrand(r):
            return analytical_gaussian_spherical(r, t) * 4.0 * np.pi * r**2
        
        integral, _ = integrate.quad(integrand, r_left, r_right, limit=50)
        Er_avg[i] = integral / V_i
    
    return Er_avg


def generate_analytical_reference(t_final, r_min=0.0, r_max=4.0, n_cells=400):
    """Generate analytical reference solution"""
    
    print("="*70)
    print("GENERATING ANALYTICAL REFERENCE SOLUTION (SPHERICAL)")
    print("="*70)
    
    # Create a temporary solver just to get the correct cell centers and faces
    # This ensures the analytical solution uses the exact same grid as numerical
    temp_solver = RadiationDiffusionSolver(
        r_min=r_min, r_max=r_max, n_cells=n_cells, d=2, dt=0.001
    )
    r = temp_solver.r_centers
    
    # Compute CELL-AVERAGED analytical solution at final time
    # This is critical for finite volume comparison!
    Er_ref = cell_average_analytical(temp_solver.r_faces, t_final)
    T_ref = temperature_from_Er(Er_ref)
    
    # Expected width
    sigma_final = np.sqrt(SIGMA0**2 + 2 * D_EFF * t_final)
    
    print(f"Problem parameters:")
    print(f"  Opacity: σ_R = {SIGMA_R} cm^-1")
    print(f"  Diffusion: D = {D:.6e} cm²/sh")
    print(f"  Effective D: D_eff = {D_EFF:.6e} cm²/sh")
    print(f"  Initial width: σ0 = {SIGMA0} cm")
    print(f"  Center: r0 = {R0} cm (at origin)")
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
        'type': 'analytical_spherical'
    }
    
    return reference_data


# =============================================================================
# RUN METHOD WITH SPECIFIC PARAMETERS
# =============================================================================

def run_method(method_name, theta, max_newton_iter, n_cells, dt, t_final, 
               use_trbdf2=False, newton_tol=1e-8, r_min=0.0, r_max=4.0):
    """
    Run one method configuration and return results
    
    Parameters:
    -----------
    method_name : str
        Descriptive name for the method
    theta : float
        Time discretization parameter (ignored if use_trbdf2=True)
    max_newton_iter : int
        Maximum Newton iterations per time step
    n_cells : int
        Number of spatial cells
    dt : float
        Time step size (sh)
    t_final : float
        Final simulation time (sh)
    use_trbdf2 : bool
        Use TR-BDF2 instead of theta method
    newton_tol : float
        Newton convergence tolerance
    r_min, r_max : float
        Domain boundaries
    """
    
    print(f"\nRunning: {method_name}")
    print(f"  n_cells={n_cells}, dt={dt:.6f}, max_iter={max_newton_iter}")
    
    # Create solver with spherical geometry (d=2)
    solver = RadiationDiffusionSolver(
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        d=2,  # Spherical geometry
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
    
    # Initial condition: Cell-averaged Gaussian centered at origin
    # Must use cell averages to match the finite volume discretization!
    Er_init = cell_average_analytical(solver.r_faces, t=0)
    solver.Er = Er_init
    solver.Er_old = Er_init.copy()
    
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
                
                Er_k = Er_new
            
            Er_intermediate = Er_k
            
            # Stage 2: BDF2 from t^n and t^{n+γ} to t^{n+1}
            solver.dt = original_dt
            Er_k = Er_intermediate.copy()
            
            for k in range(max_newton_iter):
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
                
                Er_k = Er_new
            
            solver.Er = Er_k
            solver.Er_old = Er_n
        
        # Restore dt
        solver.dt = original_dt
        
    else:
        # Theta method
        for step in range(n_steps):
            Er_prev = solver.Er.copy()
            Er_k = Er_prev.copy()
            
            for k in range(max_newton_iter):
                A_tri, rhs = solver.assemble_system(Er_k, Er_prev, theta=theta)
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
                
                Er_k = Er_new
            
            solver.Er = Er_k
            solver.Er_old = Er_prev
    
    # Get final solution
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    
    print(f"  Completed: {n_linear_solves} linear solves")
    
    results = {
        'method_name': method_name,
        'r': r,
        'Er': Er,
        'T': T,
        'n_linear_solves': n_linear_solves,
        'n_cells': n_cells,
        'dt': dt,
        'theta': theta,
        'use_trbdf2': use_trbdf2,
        'max_newton_iter': max_newton_iter
    }
    
    return results


# =============================================================================
# COMPUTE ERRORS
# =============================================================================

def compute_error(numerical_results, reference_data):
    """Compute L2 error relative to reference solution"""
    
    r_num = numerical_results['r']
    Er_num = numerical_results['Er']
    
    # Get the solver object to access r_faces
    # Recreate solver to get face positions
    temp_solver = RadiationDiffusionSolver(
        r_min=reference_data['r_min'],
        r_max=reference_data['r_max'],
        n_cells=len(r_num),
        d=2,
        dt=0.001
    )
    
    # Compute cell-averaged analytical solution (not point values!)
    t_final = reference_data['t_final']
    Er_analytical = cell_average_analytical(temp_solver.r_faces, t_final)
    
    # L2 relative error
    error = np.linalg.norm(Er_num - Er_analytical) / np.linalg.norm(Er_analytical)
    
    return error


# =============================================================================
# MAIN EFFICIENCY STUDY
# =============================================================================

def efficiency_study(t_final=0.3, r_min=0.0, r_max=4.0, n_cells=800):
    """
    Run efficiency study comparing methods
    
    Parameters:
    -----------
    t_final : float
        Final time (sh)
    r_min, r_max : float
        Domain boundaries (cm)
    n_cells : int
        Number of cells (use more to reduce spatial error)
    """
    
    print("="*70)
    print("LINEAR GAUSSIAN EFFICIENCY STUDY - SPHERICAL GEOMETRY (d=2)")
    print("="*70)
    
    # Generate analytical reference
    reference_data = generate_analytical_reference(t_final, r_min, r_max, 400)
    
    # Time step sizes to test
    dt_values = [0.01, 0.005, 0.002, 0.001]
    
    # Methods to compare
    methods = [
        # Implicit Euler (θ=1)
        {'name': 'Impl Euler (1 iter)', 'theta': 1.0, 'max_iter': 1, 'trbdf2': False},
        {'name': 'Impl Euler (conv)', 'theta': 1.0, 'max_iter': 20, 'trbdf2': False},
        
        # Crank-Nicolson (θ=0.5)
        {'name': 'Crank-Nicolson (1 iter)', 'theta': 0.5, 'max_iter': 1, 'trbdf2': False},
        {'name': 'Crank-Nicolson (conv)', 'theta': 0.5, 'max_iter': 20, 'trbdf2': False},
        
        # TR-BDF2
        {'name': 'TR-BDF2 (1 iter)', 'theta': None, 'max_iter': 1, 'trbdf2': True},
        {'name': 'TR-BDF2 (conv)', 'theta': None, 'max_iter': 20, 'trbdf2': True},
    ]
    
    # Run all combinations
    all_results = []
    
    for dt in dt_values:
        print(f"\n{'='*70}")
        print(f"Time step: dt = {dt} sh")
        print(f"{'='*70}")
        
        for method in methods:
            result = run_method(
                method_name=method['name'],
                theta=method['theta'] if method['theta'] is not None else 1.0,
                max_newton_iter=method['max_iter'],
                n_cells=n_cells,
                dt=dt,
                t_final=t_final,
                use_trbdf2=method['trbdf2'],
                r_min=r_min,
                r_max=r_max
            )
            
            # Compute error
            error = compute_error(result, reference_data)
            result['error'] = error
            result['dt'] = dt
            
            print(f"    Error: {error:.4e}")
            
            all_results.append(result)
    
    return all_results, reference_data


# =============================================================================
# PLOTTING
# =============================================================================

def plot_efficiency_curves(all_results, reference_data):
    """Plot error vs work and convergence curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group results by method
    method_groups = {}
    for result in all_results:
        name = result['method_name']
        if name not in method_groups:
            method_groups[name] = []
        method_groups[name].append(result)
    
    # Colors and markers
    colors = {
        'Impl Euler (1 iter)': 'C0',
        'Impl Euler (conv)': 'C0',
        'Crank-Nicolson (1 iter)': 'C1',
        'Crank-Nicolson (conv)': 'C1',
        'TR-BDF2 (1 iter)': 'C2',
        'TR-BDF2 (conv)': 'C2',
    }
    
    markers = {
        '1 iter': 'o',
        'conv': 's',
    }
    
    # Plot 1: Error vs linear solves (efficiency)
    ax1 = axes[0]
    for name, results in method_groups.items():
        results = sorted(results, key=lambda x: x['n_linear_solves'])
        solves = [r['n_linear_solves'] for r in results]
        errors = [r['error'] for r in results]
        
        marker = 'o' if '1 iter' in name else 's'
        linestyle = '--' if '1 iter' in name else '-'
        
        ax1.loglog(solves, errors, marker=marker, linestyle=linestyle,
                   color=colors[name], label=name, markersize=8, linewidth=2)
    
    ax1.set_xlabel('Number of Linear Solves', fontsize=12)
    ax1.set_ylabel('Relative L2 Error', fontsize=12)
    ax1.set_title('Efficiency: Error vs Computational Cost (Spherical)', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error vs dt (convergence order)
    ax2 = axes[1]
    for name, results in method_groups.items():
        results = sorted(results, key=lambda x: x['dt'], reverse=True)
        dts = [r['dt'] for r in results]
        errors = [r['error'] for r in results]
        
        marker = 'o' if '1 iter' in name else 's'
        linestyle = '--' if '1 iter' in name else '-'
        
        ax2.loglog(dts, errors, marker=marker, linestyle=linestyle,
                   color=colors[name], label=name, markersize=8, linewidth=2)
    
    # Reference slopes
    dt_ref = np.array([0.001, 0.01])
    ax2.loglog(dt_ref, 5e-3 * dt_ref**1, 'k:', alpha=0.5, label='1st order')
    ax2.loglog(dt_ref, 5e-4 * dt_ref**2, 'k--', alpha=0.5, label='2nd order')
    
    ax2.set_xlabel('Time Step Δt (sh)', fontsize=12)
    ax2.set_ylabel('Relative L2 Error', fontsize=12)
    ax2.set_title('Convergence Order (Spherical)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_gaussian_efficiency_spherical.png', dpi=300, bbox_inches='tight')
    print("\nSaved plot: linear_gaussian_efficiency_spherical.png")
    plt.show()


def plot_solution_comparison(all_results, reference_data):
    """Plot solution profiles"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Er and T profiles for finest resolution
    r_ref = reference_data['r']
    Er_ref = reference_data['Er']
    T_ref = reference_data['T']
    
    # Get converged results with finest dt
    finest_dt = min([r['dt'] for r in all_results])
    converged_results = [r for r in all_results if r['dt'] == finest_dt and 'conv' in r['method_name']]
    
    # Er plot
    ax1 = axes[0]
    ax1.plot(r_ref, Er_ref, 'k-', linewidth=2, label='Analytical')
    for result in converged_results:
        ax1.plot(result['r'], result['Er'], '--', linewidth=1.5, label=result['method_name'])
    ax1.set_xlabel('Radius r (cm)', fontsize=12)
    ax1.set_ylabel('Er (GJ/cm³)', fontsize=12)
    ax1.set_title(f'Radiation Energy (t={reference_data["t_final"]} sh, Spherical)', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # T plot
    ax2 = axes[1]
    ax2.plot(r_ref, T_ref, 'k-', linewidth=2, label='Analytical')
    for result in converged_results:
        ax2.plot(result['r'], result['T'], '--', linewidth=1.5, label=result['method_name'])
    ax2.set_xlabel('Radius r (cm)', fontsize=12)
    ax2.set_ylabel('Temperature (keV)', fontsize=12)
    ax2.set_title(f'Temperature (t={reference_data["t_final"]} sh, Spherical)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_gaussian_solution_spherical.png', dpi=300, bbox_inches='tight')
    print("Saved plot: linear_gaussian_solution_spherical.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution"""
    
    # Run efficiency study
    all_results, reference_data = efficiency_study(
        t_final=0.01,
        r_min=0.0,  # Start from origin for true spherical symmetry
        r_max=4.0,
        n_cells=201  # Use more cells to reduce spatial error
    )
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_efficiency_curves(all_results, reference_data)
    plot_solution_comparison(all_results, reference_data)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find most efficient method (smallest error for given work)
    finest_dt = min([r['dt'] for r in all_results])
    finest_results = [r for r in all_results if r['dt'] == finest_dt]
    
    print(f"\nResults at finest time step (dt={finest_dt}):")
    for result in sorted(finest_results, key=lambda x: x['error']):
        print(f"  {result['method_name']:30s}: error={result['error']:.4e}, solves={result['n_linear_solves']}")
    
    print("\n" + "="*70)
    print("Study complete!")
    print("="*70)


if __name__ == "__main__":
    main()
