#!/usr/bin/env python3
"""
Resolution Study: Does T converge to T_bc as we increase number of zones?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD, SIGMA_SB

# =============================================================================
# OPACITY FUNCTIONS
# =============================================================================

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    T_safe = 1e-2
    T_use = np.maximum(T, T_safe)
    return 10.0 * rho * ((T_use)**(-1/2)) * (E)**(-3.0)

def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    def opacity_func(T, r):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func

def make_powerlaw_diffusion_func(E_low, E_high, rho=1.0):
    opacity_func = make_powerlaw_opacity_func(E_low, E_high, rho)
    def diffusion_func(T, r):
        sigma = opacity_func(T, r)
        return 1.0 / (3.0 * sigma)
    return diffusion_func

# =============================================================================
# SINGLE TEST RUN
# =============================================================================

def run_to_steady_state(n_cells, n_groups=5, T_bc=0.05, max_time=10.0, verbose=False):
    """Run single configuration to near steady state"""
    
    # Setup
    r_min = 0.0
    r_max = 1.0e-3
    
    energy_edges = np.logspace(np.log10(1e-4), np.log10(5.0), n_groups + 1)
    
    dt = 0.01
    
    rho = 1.0
    cv = 0.05 / rho
    # T_bc is now a parameter
    
    # Create opacity functions
    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g+1]
        sigma_funcs.append(make_powerlaw_opacity_func(E_low, E_high, rho))
        diff_funcs.append(make_powerlaw_diffusion_func(E_low, E_high, rho))
    
    # Emission fractions
    from multigroup_diffusion_solver import compute_emission_fractions_from_edges
    
    sigma_a_groups = np.zeros(n_groups)
    for g in range(n_groups):
        sigma_a_groups[g] = sigma_funcs[g](T_bc, 0.0)
    
    chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc, 
                                                sigma_a_groups=sigma_a_groups)
    
    # Boundary conditions
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    BC_A = 0.5
    
    def make_left_bc_func(group_idx):
        def left_bc(phi, r):
            T_avg = 0.5 * (T_bc + solver.T[0])
            D_g = diff_funcs[group_idx](T_avg, 0.0)
            C_g = chi[group_idx] * F_total
            return BC_A, D_g, C_g
        return left_bc
    
    def right_bc_func(phi, r):
        return 0.0, 1.0, 0.0
    
    left_bc_funcs = [make_left_bc_func(g) for g in range(n_groups)]
    right_bc_funcs = [right_bc_func] * n_groups
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        emission_fractions=chi,
        rho=rho,
        cv=cv
    )
    
    # Initial condition - start closer to T_bc for faster convergence
    T_init = 0.5 * T_bc  # Start at half the boundary temperature
    solver.T = T_init * np.ones(n_cells)
    solver.T_old = solver.T.copy()
    solver.E_r = A_RAD * T_init**4 * np.ones(n_cells)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    solver.t = 0.0
    
    if verbose:
        print(f"\nRunning: n_cells = {n_cells}")
        print(f"  Δx = {(r_max-r_min)/n_cells:.6e} cm")
        print(f"  T_bc = {T_bc} keV")
    
    # Evolve to steady state
    step = 0
    T_history = [solver.T.copy()]
    t_history = [0.0]
    
    prev_T_max = solver.T.max()
    converged = False
    
    while solver.t < max_time and step < 1000:
        info = solver.step(
            max_newton_iter=50,
            newton_tol=1e-8,
            gmres_tol=1e-8,
            gmres_maxiter=200,
            use_preconditioner=False,
            verbose=False
        )
        
        step += 1
        
        # Check for convergence
        dT_max = abs(solver.T.max() - prev_T_max)
        if dT_max < 1e-8 and step > 20:
            converged = True
            if verbose:
                print(f"  Converged at step {step}, t = {solver.t:.4f} ns")
            break
        
        prev_T_max = solver.T.max()
        
        if step % 50 == 0:
            T_history.append(solver.T.copy())
            t_history.append(solver.t)
            if verbose:
                print(f"  Step {step}: t = {solver.t:.4f}, T_max = {solver.T.max():.6f}, T_min = {solver.T.min():.6f}")
        
        solver.advance_time()
    
    # Final state
    T_final = solver.T.copy()
    E_r_final = solver.E_r.copy()
    r_centers = solver.r_centers.copy()
    
    # Compute radiation temps
    T_rad_final = (E_r_final / A_RAD)**0.25
    
    # Compute average temperatures
    T_avg = np.mean(T_final)
    T_rad_avg = np.mean(T_rad_final)
    
    # Compute cell-wise optical depth
    Δx = (r_max - r_min) / n_cells
    tau_groups = []
    for g in range(n_groups):
        sigma_g = sigma_funcs[g](T_bc, 0.0)
        tau_g = sigma_g * Δx
        tau_groups.append(tau_g)
    
    if verbose:
        print(f"  Final: T_avg = {T_avg:.6f} keV (T_bc = {T_bc:.6f})")
        print(f"  T/T_bc = {T_avg/T_bc:.6f}")
        print(f"  T_rad_avg = {T_rad_avg:.6f} keV")
        print(f"  Optical depths τ = σ*Δx:")
        for g in range(n_groups):
            print(f"    Group {g}: τ = {tau_groups[g]:.6e}")
    
    return {
        'n_cells': n_cells,
        'Δx': Δx,
        'T_final': T_final,
        'T_rad_final': T_rad_final,
        'T_avg': T_avg,
        'T_rad_avg': T_rad_avg,
        'r_centers': r_centers,
        'converged': converged,
        'final_time': solver.t,
        'steps': step,
        'tau_groups': tau_groups,
        'T_history': T_history,
        't_history': t_history
    }


# =============================================================================
# RESOLUTION STUDY
# =============================================================================

def run_resolution_study():
    """Run with increasing number of cells"""
    
    print("="*80)
    print("RESOLUTION STUDY: Does T → T_bc as n_cells increases?")
    print("="*80)
    
    T_bc = 0.2  # keV - HIGHER DRIVE TEMPERATURE
    n_groups = 5
    
    print(f"\nDrive temperature: T_bc = {T_bc} keV")
    print(f"Number of energy groups: {n_groups}\n")
    
    # Test with increasing resolution (simplified for speed)
    n_cells_list = [1, 2, 5, 10]
    
    results = []
    
    print(f"\n{'n_cells':<10} {'Δx(cm)':<12} {'T_avg':<12} {'T/T_bc':<10} {'T_rad_avg':<12} {'Converged':<12} {'Steps':<8}")
    print("-"*90)
    
    for n_cells in n_cells_list:
        result = run_to_steady_state(n_cells, n_groups=n_groups, T_bc=T_bc, max_time=2.0, verbose=False)
        results.append(result)
        
        T_avg = result['T_avg']
        T_ratio = T_avg / T_bc
        T_rad_avg = result['T_rad_avg']
        converged_str = "Yes" if result['converged'] else f"No ({result['final_time']:.2f}ns)"
        
        print(f"{n_cells:<10} {result['Δx']:<12.6e} {T_avg:<12.8f} {T_ratio:<10.6f} {T_rad_avg:<12.8f} {converged_str:<12} {result['steps']:<8}")
    
    # Summary
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    print(f"\nT_bc = {T_bc} keV (target)")
    print(f"\n{'n_cells':<10} {'T/T_bc':<12} {'Error %':<12} {'Max τ':<12} {'Min τ':<12}")
    print("-"*60)
    
    for result in results:
        T_ratio = result['T_avg'] / T_bc
        error_pct = 100 * abs(T_ratio - 1.0)
        tau_max = max(result['tau_groups'])
        tau_min = min(result['tau_groups'])
        
        print(f"{result['n_cells']:<10} {T_ratio:<12.6f} {error_pct:<12.2f} {tau_max:<12.6e} {tau_min:<12.6e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: T/T_bc vs n_cells
    ax = axes[0, 0]
    n_cells_arr = np.array([r['n_cells'] for r in results])
    T_ratios = np.array([r['T_avg'] / T_bc for r in results])
    
    ax.semilogx(n_cells_arr, T_ratios, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='T = T_bc (target)')
    ax.set_xlabel('Number of Cells', fontsize=12)
    ax.set_ylabel('T_avg / T_bc', fontsize=12)
    ax.set_title('Temperature Convergence vs Resolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Temperature profiles for different resolutions
    ax = axes[0, 1]
    for i, result in enumerate(results):
        if result['n_cells'] in [1, 5, 20, 100]:
            ax.plot(result['r_centers']*1e3, result['T_final'], 'o-', 
                   label=f"n={result['n_cells']}", markersize=4)
    
    ax.axhline(y=T_bc, color='r', linestyle='--', linewidth=2, label='T_bc')
    ax.set_xlabel('Position (mm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Temperature Profiles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error vs resolution
    ax = axes[1, 0]
    errors = np.array([100 * abs(r['T_avg']/T_bc - 1.0) for r in results])
    Δx_arr = np.array([r['Δx'] for r in results])
    
    ax.loglog(Δx_arr, errors, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Cell Size Δx (cm)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Convergence Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: T vs T_rad for different resolutions
    ax = axes[1, 1]
    for i, result in enumerate(results):
        if result['n_cells'] in [1, 5, 20, 100]:
            ax.plot(result['r_centers']*1e3, result['T_final'], 'o-', 
                   label=f"T (n={result['n_cells']})", markersize=4)
            ax.plot(result['r_centers']*1e3, result['T_rad_final'], 's--', 
                   label=f"T_rad (n={result['n_cells']})", markersize=4, alpha=0.7)
    
    ax.set_xlabel('Position (mm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Material vs Radiation Temperature', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resolution_study_marshak_wave.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: resolution_study_marshak_wave.png")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = run_resolution_study()
