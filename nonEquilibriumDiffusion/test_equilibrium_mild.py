#!/usr/bin/env python3
"""
Test equilibrium problem with MILD non-equilibrium: Infinite medium starting closer to equilibrium

This version uses initial conditions that are closer to equilibrium to test solver stability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid seg faults
import matplotlib.pyplot as plt
from oneDFV import (
    NonEquilibriumRadiationDiffusionSolver,
    A_RAD, C_LIGHT, RHO
)


def main():
    print("="*70)
    print("Non-Equilibrium Infinite Medium Test (Mild Initial Conditions)")
    print("="*70)
    
    # Physical parameters
    C_V = 0.01  # GJ/cm³/keV (specific heat capacity)
    SIGMA_P = 10.0  # cm⁻¹ (Planck opacity)
    SIGMA_R = 10.0  # cm⁻¹ (Rosseland opacity)
    
    # Initial conditions - CLOSER TO EQUILIBRIUM
    T_init = 0.9  # keV (material temperature)
    T_rad_init = 1.0  # keV (radiation temperature) - only 10% difference
    phi_init = A_RAD * C_LIGHT * T_rad_init**4  # φ = acT_rad⁴
    
    # Time stepping
    dt = 0.01  # ns
    t_final = 0.1  # ns
    n_steps = int(t_final / dt)
    
    print(f"\nPhysical parameters:")
    print(f"  C_v = {C_V} GJ/cm³/keV")
    print(f"  σ_P = {SIGMA_P} cm⁻¹")
    print(f"  σ_R = {SIGMA_R} cm⁻¹")
    print(f"\nInitial conditions (mild non-equilibrium):")
    print(f"  T_mat = {T_init} keV")
    print(f"  T_rad = {T_rad_init} keV")
    print(f"  Relative difference: {abs(T_rad_init-T_init)/T_init*100:.1f}%")
    print(f"  φ = {phi_init:.6e} GJ/cm²")
    print(f"\nTime stepping:")
    print(f"  Δt = {dt} ns")
    print(f"  t_final = {t_final} ns")
    print(f"  n_steps = {n_steps}")
    
    # Define material properties
    def specific_heat(T):
        """Constant specific heat per unit mass: c_v = C_v / ρ"""
        return C_V / RHO
    
    def material_energy(T):
        """Linear material energy: e(T) = C_v * T"""
        return C_V * T
    
    def planck_opacity(T):
        """Constant Planck opacity"""
        return SIGMA_P
    
    def rosseland_opacity(T):
        """Constant Rosseland opacity"""
        return SIGMA_R
    
    # Define reflecting boundary conditions (zero flux)
    def reflecting_bc_left(phi, x):
        """Left reflecting boundary: ∇φ · n = 0"""
        A_bc = 0.0
        B_bc = 1.0
        C_bc = 0.0
        return A_bc, B_bc, C_bc
    
    def reflecting_bc_right(phi, x):
        """Right reflecting boundary: ∇φ · n = 0"""
        A_bc = 0.0
        B_bc = 1.0
        C_bc = 0.0
        return A_bc, B_bc, C_bc
    
    # Create solver
    print("\n" + "="*70)
    print("Initializing solver...")
    print("="*70)
    
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0,
        r_max=1.0,
        n_cells=3,  # Even fewer cells
        d=0,  # Cartesian geometry
        dt=dt,
        max_newton_iter=50,
        newton_tol=1e-8,
        rosseland_opacity_func=rosseland_opacity,
        planck_opacity_func=planck_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        left_bc_func=reflecting_bc_left,
        right_bc_func=reflecting_bc_right,
        theta=1.0  # Implicit Euler
    )
    
    # Set initial conditions
    print(f"\nSetting initial conditions...")
    solver.set_initial_condition(
        phi_init=phi_init,
        T_init=T_init
    )
    
    # Store solution history
    times = [0.0]
    phi_history = [solver.phi.copy()]
    T_history = [solver.T.copy()]
    
    # Time evolution
    print("\n" + "="*70)
    print("Time evolution...")
    print("="*70)
    
    for step in range(n_steps):
        t = (step + 1) * dt
        print(f"\nTime step {step+1}/{n_steps}, t = {t:.4f} ns")
        
        solver.time_step(n_steps=1, verbose=True)
        
        # Check for negative temperatures (unphysical)
        if np.any(solver.T < 0) or np.any(solver.phi < 0):
            print(f"\n*** WARNING: Negative values detected at step {step+1}! ***")
            print(f"  T_min = {solver.T.min():.6e} keV")
            print(f"  φ_min = {solver.phi.min():.6e} GJ/cm²")
            print(f"  Solver appears unstable - stopping.")
            break
        
        # Store solution
        times.append(t)
        phi_history.append(solver.phi.copy())
        T_history.append(solver.T.copy())
        
        # Print current state
        mid_cell = solver.n_cells // 2
        T_current = solver.T[mid_cell]
        phi_current = solver.phi[mid_cell]
        T_rad_current = (phi_current / (A_RAD * C_LIGHT))**0.25
        
        print(f"  Middle cell: T_mat = {T_current:.6f} keV, T_rad = {T_rad_current:.6f} keV")
        print(f"  All cells T_mat: {solver.T}")
    
    # Analyze results
    print("\n" + "="*70)
    print("Results Analysis")
    print("="*70)
    
    mid_cell = solver.n_cells // 2
    
    print(f"\nSpatial uniformity check at final time:")
    T_final = solver.T
    phi_final = solver.phi
    print(f"  T range: [{T_final.min():.6f}, {T_final.max():.6f}] keV")
    print(f"  T std dev: {T_final.std():.6e} keV")
    print(f"  φ range: [{phi_final.min():.6e}, {phi_final.max():.6e}] GJ/cm²")
    print(f"  φ std dev: {phi_final.std():.6e} GJ/cm²")
    
    # Evolution of middle cell
    T_mid_history = np.array([T_history[i][mid_cell] for i in range(len(times))])
    phi_mid_history = np.array([phi_history[i][mid_cell] for i in range(len(times))])
    T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
    
    print(f"\nMiddle cell evolution:")
    print(f"  Initial: T_mat = {T_mid_history[0]:.6f} keV, T_rad = {T_rad_history[0]:.6f} keV")
    print(f"  Final:   T_mat = {T_mid_history[-1]:.6f} keV, T_rad = {T_rad_history[-1]:.6f} keV")
    print(f"  Difference at final time: ΔT = {abs(T_mid_history[-1] - T_rad_history[-1]):.6e} keV")
    
    # Check energy conservation
    E_r_history = phi_mid_history / C_LIGHT
    e_mat_history = C_V * T_mid_history
    E_total_history = E_r_history + e_mat_history
    
    print(f"\nEnergy conservation check:")
    print(f"  Initial total energy: {E_total_history[0]:.6e} GJ/cm³")
    print(f"  Final total energy:   {E_total_history[-1]:.6e} GJ/cm³")
    print(f"  Relative change:      {abs(E_total_history[-1]-E_total_history[0])/E_total_history[0]:.6e}")
    
    # Plotting
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Temperature evolution
    ax = axes[0, 0]
    ax.plot(times, T_mid_history, 'b-o', linewidth=2, markersize=6, label='T_mat')
    ax.plot(times, T_rad_history, 'r-s', linewidth=2, markersize=6, label='T_rad')
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Temperature Evolution (Middle Cell)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 2: Temperature difference
    ax = axes[0, 1]
    temp_diff = T_rad_history - T_mid_history
    ax.plot(times, temp_diff, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('T_rad - T_mat (keV)', fontsize=12)
    ax.set_title('Non-Equilibrium Measure', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy evolution
    ax = axes[1, 0]
    ax.plot(times, E_r_history, 'b-o', linewidth=2, markersize=6, label='E_r')
    ax.plot(times, e_mat_history, 'r-s', linewidth=2, markersize=6, label='e_mat')
    ax.plot(times, E_total_history, 'k-^', linewidth=2, markersize=6, label='E_total')
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Energy Density (GJ/cm³)', fontsize=12)
    ax.set_title('Energy Evolution (Middle Cell)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 4: Total energy conservation
    ax = axes[1, 1]
    rel_energy_change = (E_total_history - E_total_history[0]) / E_total_history[0]
    ax.plot(times, rel_energy_change, 'ko-', linewidth=2, markersize=6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Relative Change in Total Energy', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.suptitle('Non-Equilibrium Infinite Medium Test (Mild)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_equilibrium_mild_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: test_equilibrium_mild_results.png")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)


if __name__ == "__main__":
    main()
