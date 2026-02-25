#!/usr/bin/env python3
"""
Test 0-D (single cell) equilibration problem

This eliminates all spatial effects to test if the temporal coupling works correctly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oneDFV import (
    NonEquilibriumRadiationDiffusionSolver,
    A_RAD, C_LIGHT, RHO
)


def main():
    print("="*70)
    print("0-D (Single Cell) Equilibration Test")
    print("="*70)
    
    # Physical parameters
    C_V = 0.01  # GJ/cm³/keV
    SIGMA_P = 10.0  # cm⁻¹
    SIGMA_R = 10.0  # cm⁻¹
    
    # Initial conditions - start closer to equilibrium
    T_init = 0.95  # keV
    T_rad_init = 1.0  # keV
    phi_init = A_RAD * C_LIGHT * T_rad_init**4
    
    # Time stepping - use very small time step
    dt = 0.001  # ns
    t_final = 0.01  # ns
    n_steps = int(t_final / dt)
    
    print(f"\nPhysical parameters:")
    print(f"  C_v = {C_V} GJ/cm³/keV")
    print(f"  σ_P = {SIGMA_P} cm⁻¹")
    print(f"\nInitial conditions:")
    print(f"  T_mat = {T_init} keV")
    print(f"  T_rad = {T_rad_init} keV")
    print(f"  φ = {phi_init:.6e} GJ/cm²")
    print(f"\nTime stepping:")
    print(f"  Δt = {dt} ns")
    print(f"  t_final = {t_final} ns")
    print(f"  n_steps = {n_steps}")
    
    # Material properties
    def specific_heat(T):
        return C_V / RHO
    
    def material_energy(T):
        return C_V * T
    
    def planck_opacity(T):
        return SIGMA_P
    
    def rosseland_opacity(T):
        return SIGMA_R
    
    # Dummy BCs (won't matter for single cell with reflecting...)
    def dummy_bc(phi, x):
        return 0.0, 1.0, 0.0
    
    # Create solver with SINGLE CELL
    print("\n" + "="*70)
    print("Initializing solver with 1 cell (0-D problem)...")
    print("="*70)
    
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0,
        r_max=1.0,
        n_cells=1,  # SINGLE CELL
        d=0,
        dt=dt,
        max_newton_iter=50,
        newton_tol=1e-10,
        rosseland_opacity_func=rosseland_opacity,
        planck_opacity_func=planck_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        left_bc_func=dummy_bc,
        right_bc_func=dummy_bc,
        theta=1.0
    )
    
    # Set initial conditions
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    # Store solution history
    times = [0.0]
    phi_history = [solver.phi[0]]
    T_history = [solver.T[0]]
    
    # Time evolution
    print("\n" + "="*70)
    print("Time evolution...")
    print("="*70)
    
    for step in range(n_steps):
        t = (step + 1) * dt
        
        solver.time_step(n_steps=1, verbose=False)
        
        # Check for negative/NaN values
        if np.any(solver.T < 0) or np.any(solver.phi < 0) or np.any(np.isnan(solver.T)) or np.any(np.isnan(solver.phi)):
            print(f"\n*** ERROR at step {step+1}! ***")
            print(f"  T = {solver.T[0]:.6e} keV")
            print(f"  φ = {solver.phi[0]:.6e} GJ/cm²")
            print("  Solver appears unstable!")
            break
        
        # Store solution
        times.append(t)
        phi_history.append(solver.phi[0])
        T_history.append(solver.T[0])
        
        # Print periodically
        if (step + 1) % max(1, n_steps // 5) == 0:
            T_current = solver.T[0]
            phi_current = solver.phi[0]
            T_rad_current = (phi_current / (A_RAD * C_LIGHT))**0.25
            print(f"t = {t:.4f} ns: T_mat = {T_current:.6f} keV, T_rad = {T_rad_current:.6f} keV")
    
    # Convert to arrays
    times = np.array(times)
    phi_history = np.array(phi_history)
    T_history = np.array(T_history)
    T_rad_history = (phi_history / (A_RAD * C_LIGHT))**0.25
    
    # Analysis
    print("\n" + "="*70)
    print("Results Analysis")
    print("="*70)
    
    print(f"\nEvolution:")
    print(f"  Initial: T_mat = {T_history[0]:.6f} keV, T_rad = {T_rad_history[0]:.6f} keV")
    print(f"  Final:   T_mat = {T_history[-1]:.6f} keV, T_rad = {T_rad_history[-1]:.6f} keV")
    print(f"  Difference: ΔT = {abs(T_history[-1] - T_rad_history[-1]):.6e} keV")
    
    # Energy conservation
    E_r_history = phi_history / C_LIGHT
    e_mat_history = C_V * T_history
    E_total = E_r_history + e_mat_history
    
    print(f"\nEnergy conservation:")
    print(f"  Initial: {E_total[0]:.6e} GJ/cm³")
    print(f"  Final:   {E_total[-1]:.6e} GJ/cm³")
    print(f"  Relative change: {abs(E_total[-1] - E_total[0]) / E_total[0]:.6e}")
    print(f"  Max deviation: {(E_total.max() - E_total.min()) / E_total[0]:.6e}")
    
    # Expected equilibrium
    E_total_eq = E_total[0]
    # At equilibrium: E_r = aT⁴, e = C_v*T
    # E_total = aT⁴ + C_v*T
    # This is nonlinear; for rough estimate:
    T_eq_approx = T_rad_history[-1]  # Should converge to something between initial values
    print(f"\nApproximate equilibrium temperature: ~{T_eq_approx:.6f} keV")
    
    # Plotting
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Temperature evolution
    ax = axes[0, 0]
    ax.plot(times * 1000, T_history, 'b-o', linewidth=2, markersize=4, label='T_mat')
    ax.plot(times * 1000, T_rad_history, 'r-s', linewidth=2, markersize=4, label='T_rad')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Temperature Evolution (0-D)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 2: Temperature difference
    ax = axes[0, 1]
    temp_diff = T_rad_history - T_history
    ax.plot(times * 1000, temp_diff, 'g-o', linewidth=2, markersize=4)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('T_rad - T_mat (keV)', fontsize=12)
    ax.set_title('Non-Equilibrium Measure', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy components
    ax = axes[1, 0]
    ax.plot(times * 1000, E_r_history, 'b-o', linewidth=2, markersize=4, label='E_r')
    ax.plot(times * 1000, e_mat_history, 'r-s', linewidth=2, markersize=4, label='e_mat')
    ax.plot(times * 1000, E_total, 'k-^', linewidth=2, markersize=4, label='E_total')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Energy Density (GJ/cm³)', fontsize=12)
    ax.set_title('Energy Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 4: Energy conservation
    ax = axes[1, 1]
    rel_energy = (E_total - E_total[0]) / E_total[0]
    ax.plot(times * 1000, rel_energy, 'ko-', linewidth=2, markersize=4)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Relative Energy Change', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.suptitle('0-D Equilibration Test', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_0d_equilibration.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: test_0d_equilibration.png")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)


if __name__ == "__main__":
    main()
