#!/usr/bin/env python3
"""
Test equilibrium problem: Infinite medium with non-equilibrium initial conditions

Physical setup:
- Infinite medium (modeled with reflecting boundaries)
- Cartesian geometry (d=0)
- Material: C_v = 0.01 GJ/cm³/keV
- Planck opacity: σ_P = 10 cm⁻¹
- Initial conditions: T = 0.4 keV, φ = ac(1.0 keV)⁴ (non-equilibrium)
- Time step: 0.01 ns
- Final time: 0.06 ns

The system should evolve towards equilibrium where T_rad = T_mat.
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
    print("Non-Equilibrium Infinite Medium Test")
    print("="*70)
    
    # Physical parameters
    C_V = 0.01  # GJ/cm³/keV (specific heat capacity)
    SIGMA_P = 5.0  # cm⁻¹ (Planck opacity)
    SIGMA_R = 5.0  # cm⁻¹ (Rosseland opacity, assuming same as Planck)
    
    # Initial conditions
    T_init = 0.4  # keV (material temperature)
    T_rad_init = 1.0  # keV (radiation temperature)
    phi_init = A_RAD * C_LIGHT * T_rad_init**4  # φ = acT_rad⁴
    
    # Time stepping
    dt = 0.01  # ns (same as reference equilibrationTest.py)
    t_final = 0.06  # ns
    n_steps = int(t_final / dt)
    
    print(f"\nPhysical parameters:")
    print(f"  C_v = {C_V} GJ/cm³/keV")
    print(f"  σ_P = {SIGMA_P} cm⁻¹")
    print(f"  σ_R = {SIGMA_R} cm⁻¹")
    print(f"\nInitial conditions:")
    print(f"  T_mat = {T_init} keV")
    print(f"  T_rad = {T_rad_init} keV (from φ)")
    print(f"  φ = {phi_init:.6e} GJ/cm²")
    print(f"\nTime stepping:")
    print(f"  Δt = {dt} ns")
    print(f"  t_final = {t_final} ns")
    print(f"  n_steps = {n_steps}")
    
    # Define material properties
    def specific_heat(T):
        """Constant specific heat per unit mass"""
        # Note: e(T) = C_v * T where C_v is per unit volume
        # So ρ * c_v = C_v, thus c_v = C_v / ρ
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
        # Robin BC: A*φ + B*(n·∇φ) = C
        # For reflecting: 0*φ + 1*(n·∇φ) = 0
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
    
    # Test with multiple time integration methods
    methods = [
        ('Implicit Euler', 1.0, False),
        ('Crank-Nicolson', 0.5, False),
        ('TR-BDF2', None, True),  # TR-BDF2 uses its own time stepper
    ]
    
    results = {}
    
    for method_name, theta_value, is_trbdf2 in methods:
        print("\n" + "="*70)
        print(f"Testing with {method_name}" + (f" (θ = {theta_value})" if theta_value is not None else ""))
        print("="*70)
        
        # Create solver
        print("\nInitializing solver...")
        
        solver = NonEquilibriumRadiationDiffusionSolver(
            r_min=0.0,
            r_max=1.0,  # Domain size doesn't matter for infinite medium
            n_cells=5,  # Just a few cells for infinite medium
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
            theta=theta_value if theta_value is not None else 1.0  # Dummy value for TR-BDF2
        )
        
        # Set initial conditions
        print(f"Setting initial conditions...")
        solver.set_initial_condition(
            phi_init=phi_init,  # Uniform φ corresponding to T_rad = 1.0 keV
            T_init=T_init       # Uniform T = 0.4 keV
        )
        
        # Store solution history
        times = [0.0]
        phi_history = [solver.phi.copy()]
        T_history = [solver.T.copy()]
        
        # Time evolution
        print("\nTime evolution...")
        
        print_interval = max(1, n_steps // 10)  # Print ~10 times during evolution
        
        for step in range(n_steps):
            t = (step + 1) * dt
            verbose = (step % print_interval == 0 or step == n_steps - 1)
            
            if verbose:
                print(f"  Time step {step+1}/{n_steps}, t = {t:.4f} ns")
            
            # Use appropriate time stepper
            if is_trbdf2:
                solver.time_step_trbdf2(n_steps=1, verbose=verbose)
            else:
                solver.time_step(n_steps=1, verbose=verbose)
            
            # Check for negative temperatures (unphysical)
            if np.any(solver.T < 0):
                print(f"\n*** WARNING: Negative temperatures detected at step {step+1}! ***")
                print(f"  T_min = {solver.T.min():.6e} keV")
                print(f"  This indicates Newton solver instability.")
                print(f"  Consider: smaller time step, different initial conditions, or solver parameters")
                break
            
            # Store solution
            times.append(t)
            phi_history.append(solver.phi.copy())
            T_history.append(solver.T.copy())
            
            # Print current state (use middle cell as representative)
            if verbose:
                mid_cell = solver.n_cells // 2
                T_current = solver.T[mid_cell]
                phi_current = solver.phi[mid_cell]
                T_rad_current = (phi_current / (A_RAD * C_LIGHT))**0.25
                
                print(f"    Middle cell: T_mat = {T_current:.6f} keV, T_rad = {T_rad_current:.6f} keV")
        
        # Store results for this method
        results[method_name] = {
            'times': times,
            'phi_history': phi_history,
            'T_history': T_history,
            'theta': theta_value,
            'final_T': solver.T.copy(),
            'final_phi': solver.phi.copy()
        }
    
    # Analyze and compare results
    print("\n" + "="*70)
    print("Results Comparison")
    print("="*70)
    
    for method_name, result_data in results.items():
        print(f"\n{method_name} (θ = {result_data['theta']}):")
        
        mid_cell = 2  # Middle of 5 cells
        times = result_data['times']
        T_history = result_data['T_history']
        phi_history = result_data['phi_history']
        
        T_mid_history = np.array([T_history[i][mid_cell] for i in range(len(times))])
        phi_mid_history = np.array([phi_history[i][mid_cell] for i in range(len(times))])
        T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
        
        print(f"  Initial: T_mat = {T_mid_history[0]:.6f} keV, T_rad = {T_rad_history[0]:.6f} keV")
        print(f"  Final:   T_mat = {T_mid_history[-1]:.6f} keV, T_rad = {T_rad_history[-1]:.6f} keV")
        print(f"  Difference: ΔT = {abs(T_mid_history[-1] - T_rad_history[-1]):.6e} keV")
        
        # Check spatial uniformity
        T_final = result_data['final_T']
        phi_final = result_data['final_phi']
        print(f"  Spatial uniformity:")
        print(f"    T range: [{T_final.min():.6f}, {T_final.max():.6f}] keV")
        print(f"    T std dev: {T_final.std():.6e} keV")
        
        # Energy conservation check
        E_r_history = phi_mid_history / C_LIGHT
        e_mat_history = C_V * T_mid_history
        E_total_history = E_r_history + e_mat_history
        
        print(f"  Energy conservation:")
        print(f"    Initial: {E_total_history[0]:.6e} GJ/cm³")
        print(f"    Final:   {E_total_history[-1]:.6e} GJ/cm³")
        print(f"    Relative change: {abs(E_total_history[-1] - E_total_history[0]) / E_total_history[0]:.6e}")
    
    # Plotting
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {'Implicit Euler': 'blue', 'Crank-Nicolson': 'red', 'TR-BDF2': 'green'}
    markers = {'Implicit Euler': 'o', 'Crank-Nicolson': 's', 'TR-BDF2': '^'}
    
    for method_name, result_data in results.items():
        mid_cell = 2
        times = np.array(result_data['times'])
        T_history = result_data['T_history']
        phi_history = result_data['phi_history']
        
        T_mid_history = np.array([T_history[i][mid_cell] for i in range(len(times))])
        phi_mid_history = np.array([phi_history[i][mid_cell] for i in range(len(times))])
        T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
        
        color = colors[method_name]
        marker = markers[method_name]
        
        # Plot 1: Material temperature evolution
        ax = axes[0, 0]
        ax.plot(times, T_mid_history, color=color, marker=marker, linewidth=2, 
               markersize=4, markevery=max(1, len(times)//20), label=method_name)
        ax.plot(times, T_rad_history, "--",color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//20), label=method_name)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_mat (keV)', fontsize=11)
        ax.set_title('Material Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85,1.0)
        ax.legend(fontsize=10)
        
        # Plot 2: Radiation temperature evolution
        ax = axes[0, 1]
        ax.plot(times, T_rad_history, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//20), label=method_name)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_rad (keV)', fontsize=11)
        ax.set_title('Radiation Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 3: Temperature difference
        ax = axes[0, 2]
        temp_diff = T_rad_history - T_mid_history
        ax.plot(times, temp_diff, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//20), label=method_name)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_rad - T_mat (keV)', fontsize=11)
        ax.set_title('Non-Equilibrium Measure', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 4: Energy components
        ax = axes[1, 0]
        E_r_history = phi_mid_history / C_LIGHT
        e_mat_history = C_V * T_mid_history
        E_total = E_r_history + e_mat_history
        ax.plot(times, E_r_history, color=color, linestyle='--', linewidth=2, 
                alpha=0.7, label=f'{method_name} E_r')
        ax.plot(times, e_mat_history, color=color, linestyle=':', linewidth=2,
                alpha=0.7, label=f'{method_name} e_mat')
        ax.plot(times, E_total, color=color, linestyle='-', linewidth=2.5,
                marker=marker, markersize=4, markevery=max(1, len(times)//20),
                label=f'{method_name} E_total')
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('Energy Density (GJ/cm³)', fontsize=11)
        ax.set_title('Energy Components', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        
        # Plot 5: Energy conservation
        ax = axes[1, 1]
        rel_energy_change = (E_total - E_total[0]) / E_total[0]
        ax.plot(times, rel_energy_change, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//20), label=method_name)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('Relative Energy Change', fontsize=11)
        ax.set_title('Energy Conservation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Plot 6: Final spatial profile
        ax = axes[1, 2]
        r = np.linspace(0, 1, 5)
        T_final = result_data['final_T']
        ax.plot(r, T_final, color=color, marker=marker, linewidth=2,
                markersize=8, label=method_name)
        ax.set_xlabel('Position (cm)', fontsize=11)
        ax.set_ylabel('T_mat (keV)', fontsize=11)
        ax.set_title('Final Spatial Profile', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.suptitle('Non-Equilibrium Infinite Medium: Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_equilibrium_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: test_equilibrium_results.png")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
