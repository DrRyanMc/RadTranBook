#!/usr/bin/env python3
"""
1-Group 0-D Equilibration Test

Simplest possible test: 1 group, 1 cell, closed system.
Cold material (T=0.4 keV) + hot radiation (T_r=1.0 keV).
Material should heat up, radiation should cool down.
Total energy E_total = E_r + ρ*c_v*T must be conserved.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D

# Physical constants
C_LIGHT = 2.99792458e1  # cm/ns
A_RAD = 0.01372  # GJ/(cm³·keV⁴)

def run_1group_0d_test():
    """Run 1-group 0-D equilibration test."""
    
    print("="*80)
    print("1-Group 0-D Equilibration Test")
    print("="*80)
    
    # Parameters
    n_groups = 1
    sigma_a = 5.0  # cm^-1 (constant opacity)
    C_v = 0.01  # GJ/(g·keV)
    rho = 1.0  # g/cm³
    
    # Energy edges (very wide range to approximate gray)
    energy_edges = np.array([0.01, 10.0])  # keV
    
    print(f"Parameters:")
    print(f"  Groups: {n_groups}")
    print(f"  σ_a = {sigma_a} cm^-1")
    print(f"  C_v = {C_v} GJ/(g·keV)")
    print(f"  ρ = {rho} g/cm³")
    print(f"  Energy edges: {energy_edges} keV")
    
    # Create solver with minimal spatial structure
    # Use 1 cell with very large diffusion to make it effectively 0-D
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=1.0,
        n_cells=1,
        energy_edges=energy_edges,
        geometry='planar',
        dt=0.01,  # ns
        diffusion_coeff_funcs=[lambda T, r: 1e10],  # Very large D
        absorption_coeff_funcs=[lambda T, r: sigma_a],
        left_bc='neumann',   # Zero flux (closed system)
        right_bc='neumann',  # Zero flux (closed system)
        left_bc_values=[0.0],
        right_bc_values=[0.0],
        rho=rho,
        cv=C_v
    )
    
    # Initial conditions
    T_init = 0.4  # keV (cold material)
    T_rad_init = 1.0  # keV (hot radiation)
    
    solver.T = np.array([T_init])
    solver.E_r = np.array([A_RAD * T_rad_init**4])
    
    # Initialize "old" values for first timestep
    solver.T_old = solver.T.copy()
    solver.E_r_old = solver.E_r.copy()
    
    print(f"\nInitial conditions:")
    print(f"  Material temperature: T = {T_init} keV")
    print(f"  Radiation temperature: T_r = {T_rad_init} keV")
    print(f"  Radiation energy: E_r = {solver.E_r[0]:.6e} GJ/cm³")
    print(f"  Expected equilibrium: T_eq ≈ {T_rad_init:.3f} keV (hot radiation heats cold material)")
    
    # Initial energy budget
    e_mat_init = rho * C_v * solver.T[0]
    E_r_init = solver.E_r[0]
    E_total_init = e_mat_init + E_r_init
    
    print(f"\nInitial energy budget:")
    print(f"  Material energy: e = {e_mat_init:.6e} GJ/cm³")
    print(f"  Radiation energy: E_r = {E_r_init:.6e} GJ/cm³")
    print(f"  Total energy: E_total = {E_total_init:.6e} GJ/cm³")
    
    # Storage for history
    n_steps = 10
    time_history = np.zeros(n_steps + 1)
    T_history = np.zeros(n_steps + 1)
    T_rad_history = np.zeros(n_steps + 1)
    E_r_history = np.zeros(n_steps + 1)
    e_mat_history = np.zeros(n_steps + 1)
    E_total_history = np.zeros(n_steps + 1)
    energy_error_history = np.zeros(n_steps + 1)
    
    # Store initial state
    time_history[0] = 0.0
    T_history[0] = solver.T[0]
    T_rad_history[0] = (solver.E_r[0] / A_RAD)**0.25
    E_r_history[0] = solver.E_r[0]
    e_mat_history[0] = e_mat_init
    E_total_history[0] = E_total_init
    energy_error_history[0] = 0.0
    
    # Time stepping
    print(f"\nStep   Time       T (keV)      T_rad (keV)  E_r (GJ/cm³)    ΔE/E_0       Newton  Conv")
    print("-"*80)
    
    for step in range(1, n_steps + 1):
        # Take a time step
        info = solver.step(
            max_newton_iter=10,
            newton_tol=1e-8,
            gmres_tol=1e-6,
            gmres_maxiter=200,
            verbose=False
        )
        
        # Advance time (store current as old for next step)
        solver.advance_time()
        
        # Compute current state
        time_history[step] = step * solver.dt
        T_history[step] = solver.T[0]
        E_r_history[step] = solver.E_r[0]
        
        # Check for negative E_r
        if E_r_history[step] < 0:
            print(f"\n*** ERROR: E_r went negative! E_r = {E_r_history[step]:.6e}")
            T_rad_history[step] = 0
        else:
            T_rad_history[step] = (E_r_history[step] / A_RAD)**0.25
            
        e_mat_history[step] = rho * C_v * solver.T[0]
        E_total_history[step] = E_r_history[step] + e_mat_history[step]
        energy_error_history[step] = (E_total_history[step] - E_total_init) / E_total_init
        
        conv_symbol = "✓" if info.get('converged', False) else "✗"
        newton_iters = info.get('n_newton_iter', info.get('newton_iterations', '?'))
        
        print(f"{step:<6} {time_history[step]:<10.4f} {T_history[step]:<12.6f} "
              f"{T_rad_history[step]:<12.6f} {E_r_history[step]:<15.6e} "
              f"{energy_error_history[step]:<12.6e} {newton_iters:<7} {conv_symbol}")
    
    # Final summary
    print("\n" + "="*80)
    print("1-Group 0-D test completed!")
    print("="*80)
    
    T_final = T_history[-1]
    T_rad_final = T_rad_history[-1]
    E_r_final = E_r_history[-1]
    e_mat_final = e_mat_history[-1]
    E_total_final = E_total_history[-1]
    
    print(f"\nFinal state at t = {time_history[-1]:.2f} ns:")
    print(f"  Material temperature: T = {T_final:.6f} keV")
    print(f"  Radiation temperature: T_r = {T_rad_final:.6f} keV")
    print(f"  ΔT = T - T_r = {T_final - T_rad_final:.6e} keV")
    print(f"  Radiation energy: E_r = {E_r_final:.6e} GJ/cm³")
    print(f"  Material energy: e = {e_mat_final:.6e} GJ/cm³")
    
    # Energy conservation check
    abs_error = E_total_final - E_total_init
    rel_error = abs_error / E_total_init
    max_deviation = np.max(np.abs(E_total_history - E_total_init))
    max_rel_deviation = max_deviation / E_total_init
    
    print(f"\nEnergy conservation:")
    print(f"  Initial total: E_0 = {E_total_init:.10e} GJ/cm³")
    print(f"  Final total:   E_f = {E_total_final:.10e} GJ/cm³")
    print(f"  Absolute error: ΔE = {abs_error:.10e} GJ/cm³")
    print(f"  Relative error: ΔE/E_0 = {rel_error:.10e}")
    print(f"  Max deviation: {max_deviation:.10e} GJ/cm³ ({max_rel_deviation:.10e} relative)")
    
    if abs(rel_error) > 1e-6:
        print(f"  ✗ WARNING: Significant energy conservation error!")
    else:
        print(f"  ✓ Energy conserved to tolerance")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature evolution
    ax = axes[0, 0]
    ax.plot(time_history, T_history, 'b-', label='Material T', linewidth=2)
    ax.plot(time_history, T_rad_history, 'r--', label='Radiation T_r', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Temperature (keV)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy components
    ax = axes[0, 1]
    ax.plot(time_history, E_r_history, 'r-', label='Radiation E_r', linewidth=2)
    ax.plot(time_history, e_mat_history, 'b-', label='Material e', linewidth=2)
    ax.plot(time_history, E_total_history, 'k--', label='Total', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Energy density (GJ/cm³)')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total energy (zoomed)
    ax = axes[1, 0]
    ax.plot(time_history, E_total_history, 'k-', linewidth=2)
    ax.axhline(E_total_init, color='gray', linestyle=':', label='Initial')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Total energy (GJ/cm³)')
    ax.set_title('Total Energy (Should Be Constant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Relative energy error
    ax = axes[1, 1]
    ax.plot(time_history, energy_error_history * 100, 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Relative error ΔE/E_0 (%)')
    ax.set_title('Energy Conservation Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_multigroup_1group.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_multigroup_1group.png'")

if __name__ == "__main__":
    run_1group_0d_test()
