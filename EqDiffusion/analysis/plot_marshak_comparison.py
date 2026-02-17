#!/usr/bin/env python3
"""
Compare analytical self-similar Marshak solution with numerical solutions
using Implicit Euler, Crank-Nicolson, and TR-BDF2 methods
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from oneDFV import RadiationDiffusionSolver, solve_tridiagonal
from plotfuncs import show

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_LIGHT = 299792458e-7  # cm/sh (speed of light in cm per shake)
A_RAD = 0.01372  # GJ/(cm³·keV⁴) (radiation constant)
RHO = 1.0  # g/cm³


def temperature_from_Er(Er):
    """Convert radiation energy density to temperature"""
    return (Er / A_RAD)**(1.0/4.0)


# =============================================================================
# MARSHAK WAVE PROBLEM SETUP
# =============================================================================

def marshak_opacity(Er):
    """Rosseland mean opacity: σ_R = 300*T^-3 cm^-1"""
    T = temperature_from_Er(Er)
    return 300.0 / T**3


def marshak_specific_heat(T):
    """Specific heat: c_v = 0.3 GJ/(cm³·keV)"""
    return 0.3


def marshak_material_energy(T):
    """Material energy density: e_mat = c_v * T"""
    return marshak_specific_heat(T) * T


def marshak_left_bc(Er, r):
    """Left boundary (r=0): T^4 = 1 keV^4 (T_boundary = 1 keV)"""
    T_boundary = 1.0  # keV
    Er_boundary = A_RAD * T_boundary**4
    # Robin BC: A*Er + B*dEr/dr = C
    # For Dirichlet: A=1, B=0, C=Er_boundary
    return 1.0, 0.0, Er_boundary


def marshak_right_bc(Er, r):
    """Right boundary: zero flux dEr/dr = 0"""
    # Robin BC: A*Er + B*dEr/dr = C
    # For zero flux: A=0, B=1, C=0
    return 0.0, 1.0, 0.0


# =============================================================================
# SELF-SIMILAR ANALYTICAL SOLUTION
# =============================================================================

def generate_self_similar_solution(t, r_max=0.2, n_cells=400):
    """Generate analytical self-similar solution at time t"""
    
    # Self-similar solution parameters
    xi_max = 1.11305
    omega = 0.05989
    
    # Diffusion constant K
    n = 3  # opacity exponent
    sigma_0 = 300.0  # cm^-1
    cv = 0.3  # GJ/(cm^3*keV)
    K_const = 8 * A_RAD * C_LIGHT / ((n + 4) * 3 * sigma_0 * RHO * cv)
    
    # Create spatial grid
    r_min = 0.0
    r = np.linspace(r_min, r_max, n_cells)
    
    # Load tabulated self-similar solution
    loaded_table = np.loadtxt(Path(__file__).parent / 'MarshakP04573.csv', 
                              delimiter=',', skiprows=0)
    xi_table = loaded_table[:,0] * xi_max
    thf_table = loaded_table[:,2]
    
    # Interpolate
    from scipy.interpolate import interp1d
    T_interp = interp1d(xi_table, thf_table, kind='linear', 
                       fill_value=0.0, bounds_error=False)
    
    # Compute similarity variable xi = r / sqrt(K*t)
    xi = r / np.sqrt(K_const * t)
    
    # Evaluate temperature from self-similar solution
    T = T_interp(xi)
    
    # Convert to radiation energy density
    Er = A_RAD * T**4
    
    return r, Er, T


# =============================================================================
# RUN NUMERICAL SOLUTION
# =============================================================================

def run_marshak(method_name, dt, t_final, n_cells=400, max_newton_iter=50, 
                newton_tol=1e-8, theta=1.0, use_trbdf2=False):
    """Run Marshak wave with specified method"""
    
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
    
    # Time evolution
    n_steps = int(t_final / dt)
    
    if use_trbdf2:
        # TR-BDF2 method
        gamma = 2.0 - np.sqrt(2.0)
        original_dt = solver.dt
        
        for step in range(n_steps):
            Er_n = solver.Er.copy()
            
            # Stage 1: Trapezoidal rule to intermediate point
            solver.dt = gamma * original_dt
            Er_k = Er_n.copy()
            
            for k in range(max_newton_iter):
                A_tri, rhs = solver.assemble_system(Er_k, Er_n, theta=0.5)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                Er_new = solve_tridiagonal(A_tri, rhs)
                
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
            
            # Stage 2: BDF2
            solver.dt = original_dt
            Er_k = Er_intermediate.copy()
            
            for k in range(max_newton_iter):
                A_tri, rhs = solver.assemble_system_bdf2(Er_k, Er_n, Er_intermediate, gamma)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                Er_new = solve_tridiagonal(A_tri, rhs)
                
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
            Er_prev = solver.Er.copy()
            Er_k = solver.Er.copy()
            
            for k in range(max_newton_iter):
                A_tri, rhs = solver.assemble_system(Er_k, Er_prev, theta=solver.theta)
                solver.apply_boundary_conditions(A_tri, rhs, Er_k)
                Er_new = solve_tridiagonal(A_tri, rhs)
                
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
            solver.Er_old = Er_prev.copy()
    
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    
    return r, Er, T


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("MARSHAK WAVE COMPARISON: ANALYTICAL vs NUMERICAL")
    print("="*70)
    
    # Simulation parameters
    times = [1.0, 10.0]  # ns
    dt = 0.01  # ns
    n_cells = 50
    
    # Methods to compare - both converged and 2-iteration versions
    methods = [
        {'name': 'Implicit Euler (converged)', 'theta': 1.0, 'use_trbdf2': False, 
         'max_newton_iter': 50, 'color': 'blue', 'marker': 'o', 'fillstyle': 'none'},
        {'name': 'Implicit Euler (2 iter)', 'theta': 1.0, 'use_trbdf2': False, 
         'max_newton_iter': 1, 'color': 'blue', 'marker': 'o', 'fillstyle': 'full'},
        {'name': 'Crank-Nicolson (converged)', 'theta': 0.5, 'use_trbdf2': False,
         'max_newton_iter': 50, 'color': 'green', 'marker': 's', 'fillstyle': 'none'},
        {'name': 'Crank-Nicolson (2 iter)', 'theta': 0.5, 'use_trbdf2': False,
         'max_newton_iter': 1, 'color': 'green', 'marker': 's', 'fillstyle': 'full'},
        {'name': 'TR-BDF2 (converged)', 'theta': 1.0, 'use_trbdf2': True,
         'max_newton_iter': 50, 'color': 'red', 'marker': '^', 'fillstyle': 'none'},
        {'name': 'TR-BDF2 (2 iter)', 'theta': 1.0, 'use_trbdf2': True,
         'max_newton_iter': 1, 'color': 'red', 'marker': '^', 'fillstyle': 'full'}
    ]
    
    # Create single figure for both times
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # Time-specific line styles: solid for t=1.0, dashed for t=10.0
    time_styles = {1.0: '-', 10.0: '-'}
    
    # Run simulations for each time
    for t_final in times:
        print(f"\n{'='*70}")
        print(f"TIME: t = {t_final} ns")
        print(f"{'='*70}")
        
        # Generate analytical solution
        print(f"\nGenerating analytical solution...")
        r_anal, Er_anal, T_anal = generate_self_similar_solution(t_final, n_cells=n_cells)
        print(f"  Max T = {T_anal.max():.4f} keV")
        
        # Plot analytical solution (black, different style for each time)
        linestyle = time_styles[t_final]
        ax1.plot(r_anal, Er_anal, 'k', linestyle=linestyle, linewidth=2.0, 
                alpha=0.7, zorder=10)
        ax2.plot(r_anal, T_anal, 'k', linestyle=linestyle, linewidth=2.0,
                alpha=0.7, zorder=10)
        
        # Run and plot numerical solutions
        for method in methods:
            print(f"\nRunning {method['name']}...")
            r, Er, T = run_marshak(
                method['name'], 
                dt=dt, 
                t_final=t_final, 
                n_cells=n_cells,
                max_newton_iter=method['max_newton_iter'],
                theta=method['theta'],
                use_trbdf2=method['use_trbdf2']
            )
            
            # Compute error
            Er_interp = np.interp(r_anal, r, Er)
            error = np.linalg.norm(Er_interp - Er_anal) / np.linalg.norm(Er_anal)
            print(f"  Max T = {T.max():.4f} keV")
            print(f"  Error = {error:.3e}")
            
            # Plot with markers only (no connecting lines for clarity)
            markevery = 2 if t_final == 1.0 else 2
            #if converged, make line solid, else dashed
            if 'converged' in method['name']:
                linestyle = '-'
            else:
                linestyle = '--'
            ax1.plot(r[Er>np.min(Er)], Er[Er>np.min(Er)], linestyle=linestyle, 
                    color=method['color'], marker=method['marker'],
                    fillstyle=method['fillstyle'],
                    markevery=markevery, markersize=4, alpha=0.7,
                    markeredgewidth=1.0)
            ax2.plot(r[T>np.min(T)], T[T>np.min(T)], linestyle=linestyle,
                    color=method['color'], marker=method['marker'],
                    fillstyle=method['fillstyle'],
                    markevery=markevery, markersize=4, alpha=0.7,
                    markeredgewidth=1.0)
    
    # Add text annotations to indicate times
    # Place annotations near the peak of each curve
    ax1.text(0.150, 0.004, 't = 10.0 ns', fontsize=10, bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))
    ax1.text(0.05, 0.002, 't = 1.0 ns', fontsize=10, bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))
    
    ax2.text(0.150, 0.56, 't = 10.0 ns', fontsize=10, bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))
    ax2.text(0.05, 0.42, 't = 1.0 ns', fontsize=10, bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))
    
    # Create custom legend for methods (not times)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2, label='Self-Similar'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markeredgecolor='blue', fillstyle='none', markeredgewidth=1.5,
               markersize=6, label='Implicit Euler'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markeredgecolor='green', fillstyle='none', markeredgewidth=1.5,
               markersize=6, label='Crank-Nicolson'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markeredgecolor='red', fillstyle='none', markeredgewidth=1.5,
               markersize=6, label='TR-BDF2')
    ]
    
    # Format plots
    ax1.set_xlabel('Position r (cm)')
    ax1.set_ylabel('Radiation Energy $E_r$ (GJ/cm$^3$)')
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Position r (cm)')
    ax2.set_ylabel('Temperature T (keV)')
    #ax2.legend(handles=legend_elements, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'marshak_comparison.pdf'
    show(filename)
    print(f"\nFigure saved: {filename}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
