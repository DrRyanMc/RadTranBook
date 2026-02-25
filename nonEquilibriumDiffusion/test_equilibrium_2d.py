#!/usr/bin/env python3
"""
Test 2D equilibrium problem: Infinite medium with non-equilibrium initial conditions

Physical setup:
- Infinite medium (modeled with reflecting boundaries on all sides)
- Cartesian (x-y) geometry
- Material: C_v = 0.01 GJ/cm³/keV
- Planck opacity: σ_P = 5.0 cm⁻¹
- Rosseland opacity: σ_R = 5.0 cm⁻¹
- Initial conditions: T = 0.4 keV, φ = ac(1.0 keV)⁴ (non-equilibrium)
- Time step: 0.01 ns
- Final time: 0.06 ns

The system should evolve towards equilibrium where T_rad = T_mat.
Since the medium is infinite (reflecting BCs) and starts uniformly,
the solution should remain spatially uniform.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import twoDFV
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twoDFV import (
    NonEquilibriumRadiationDiffusionSolver2D,
    A_RAD, C_LIGHT, RHO
)


def main():
    print("="*70)
    print("2D Non-Equilibrium Infinite Medium Test")
    print("="*70)
    
    # Physical parameters
    C_V = 0.01  # GJ/cm³/keV (specific heat capacity per unit volume)
    SIGMA_P = 5.0  # cm⁻¹ (Planck opacity)
    SIGMA_R = 5.0  # cm⁻¹ (Rosseland opacity)
    
    # Initial conditions
    T_init = 0.4  # keV (material temperature)
    T_rad_init = 1.0  # keV (radiation temperature)
    phi_init = A_RAD * C_LIGHT * T_rad_init**4  # φ = acT_rad⁴
    
    # Time stepping
    dt = 0.01  # ns
    t_final = 0.06  # ns
    n_steps = int(t_final / dt)
    
    # Spatial grid (small since solution should be uniform)
    nx = 5
    ny = 5
    
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
    print(f"\nSpatial grid:")
    print(f"  nx × ny = {nx} × {ny} cells")
    
    # Define material properties
    def specific_heat(T, x, y):
        """Constant specific heat per unit mass"""
        return C_V / RHO
    
    def material_energy(T, x, y):
        """Linear material energy: e(T) = C_v * T"""
        return C_V * T
    
    def inverse_material_energy(e, x, y):
        """Inverse: T from e"""
        return e / C_V
    
    def planck_opacity(T, x, y):
        """Constant Planck opacity"""
        return SIGMA_P
    
    def rosseland_opacity(T, x, y):
        """Constant Rosseland opacity"""
        return SIGMA_R
    
    # Define reflecting boundary conditions (zero flux) for all boundaries
    def reflecting_bc(phi, pos, t, boundary='left', geometry='cartesian'):
        """Reflecting boundary: ∇φ · n = 0"""
        # Robin BC: A*φ + B*(n·∇φ) = C
        # For reflecting: 0*φ + 1*(n·∇φ) = 0
        A_bc = 0.0
        B_bc = 1.0
        C_bc = 0.0
        return A_bc, B_bc, C_bc
    
    # Boundary functions for all sides
    boundary_funcs = {
        'left': reflecting_bc,
        'right': reflecting_bc,
        'bottom': reflecting_bc,
        'top': reflecting_bc
    }
    
    # Test with multiple time integration methods
    methods = [
        ('Implicit Euler', 1.0, False),
        ('Crank-Nicolson', 0.5, False),
        ('TR-BDF2', None, True),
    ]
    
    results = {}
    
    for method_name, theta_value, is_trbdf2 in methods:
        print("\n" + "="*70)
        print(f"Testing with {method_name}" + (f" (θ = {theta_value})" if theta_value is not None else ""))
        print("="*70)
        
        # Create solver
        print("\nInitializing 2D solver...")
        
        solver = NonEquilibriumRadiationDiffusionSolver2D(
            x_min=0.0,
            x_max=1.0,
            nx_cells=nx,
            y_min=0.0,
            y_max=1.0,
            ny_cells=ny,
            geometry='cartesian',
            dt=dt,
            max_newton_iter=50,
            newton_tol=1e-8,
            rosseland_opacity_func=rosseland_opacity,
            planck_opacity_func=planck_opacity,
            specific_heat_func=specific_heat,
            material_energy_func=material_energy,
            inverse_material_energy_func=inverse_material_energy,
            boundary_funcs=boundary_funcs,
            theta=theta_value if theta_value is not None else 1.0
        )
        
        # Set uniform initial conditions
        print(f"Setting uniform initial conditions...")
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
        
        print_interval = max(1, n_steps // 10)
        
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
            
            # Check for negative temperatures
            if np.any(solver.T < 0):
                print(f"\n*** WARNING: Negative temperatures detected at step {step+1}! ***")
                print(f"  T_min = {solver.T.min():.6e} keV")
                break
            
            # Store solution
            times.append(t)
            phi_history.append(solver.phi.copy())
            T_history.append(solver.T.copy())
            
            # Print current state (use center cell as representative)
            if verbose:
                mid_i = nx // 2
                mid_j = ny // 2
                mid_idx = mid_i + mid_j * nx
                T_current = solver.T[mid_idx]
                phi_current = solver.phi[mid_idx]
                T_rad_current = (phi_current / (A_RAD * C_LIGHT))**0.25
                
                print(f"    Center cell: T_mat = {T_current:.6f} keV, T_rad = {T_rad_current:.6f} keV")
        
        # Store results for this method
        results[method_name] = {
            'times': times,
            'phi_history': phi_history,
            'T_history': T_history,
            'theta': theta_value,
            'final_T': solver.T.copy(),
            'final_phi': solver.phi.copy(),
            'final_T_2d': solver.get_T_2d(),
            'final_phi_2d': solver.get_phi_2d()
        }
    
    # Analyze and compare results
    print("\n" + "="*70)
    print("Results Comparison")
    print("="*70)
    
    mid_i = nx // 2
    mid_j = ny // 2
    mid_idx = mid_i + mid_j * nx
    
    for method_name, result_data in results.items():
        print(f"\n{method_name}" + (f" (θ = {result_data['theta']})" if result_data['theta'] is not None else "") + ":")
        
        times = result_data['times']
        T_history = result_data['T_history']
        phi_history = result_data['phi_history']
        
        T_mid_history = np.array([T_history[i][mid_idx] for i in range(len(times))])
        phi_mid_history = np.array([phi_history[i][mid_idx] for i in range(len(times))])
        T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
        
        print(f"  Initial: T_mat = {T_mid_history[0]:.6f} keV, T_rad = {T_rad_history[0]:.6f} keV")
        print(f"  Final:   T_mat = {T_mid_history[-1]:.6f} keV, T_rad = {T_rad_history[-1]:.6f} keV")
        print(f"  Difference: |ΔT| = {abs(T_mid_history[-1] - T_rad_history[-1]):.6e} keV")
        
        # Check spatial uniformity
        T_final = result_data['final_T']
        phi_final = result_data['final_phi']
        print(f"  Spatial uniformity:")
        print(f"    T range: [{T_final.min():.6f}, {T_final.max():.6f}] keV")
        print(f"    T std dev: {T_final.std():.6e} keV")
        print(f"    φ range: [{phi_final.min():.6e}, {phi_final.max():.6e}] GJ/cm²")
        print(f"    φ std dev: {phi_final.std():.6e} GJ/cm²")
        
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
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'Implicit Euler': 'blue', 'Crank-Nicolson': 'red', 'TR-BDF2': 'green'}
    markers = {'Implicit Euler': 'o', 'Crank-Nicolson': 's', 'TR-BDF2': '^'}
    
    for method_name, result_data in results.items():
        times = np.array(result_data['times'])
        T_history = result_data['T_history']
        phi_history = result_data['phi_history']
        
        T_mid_history = np.array([T_history[i][mid_idx] for i in range(len(times))])
        phi_mid_history = np.array([phi_history[i][mid_idx] for i in range(len(times))])
        T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
        
        color = colors[method_name]
        marker = markers[method_name]
        
        # Plot 1: Material temperature evolution
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(times, T_mid_history, color=color, marker=marker, linewidth=2, 
               markersize=4, markevery=max(1, len(times)//10), label=method_name)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_mat (keV)', fontsize=11)
        ax.set_title('Material Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 2: Radiation temperature evolution
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(times, T_rad_history, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//10), label=method_name)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_rad (keV)', fontsize=11)
        ax.set_title('Radiation Temperature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 3: Temperature difference
        ax = fig.add_subplot(gs[0, 2])
        temp_diff = T_rad_history - T_mid_history
        ax.plot(times, temp_diff, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//10), label=method_name)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T_rad - T_mat (keV)', fontsize=11)
        ax.set_title('Non-Equilibrium Measure', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 4: Energy components
        ax = fig.add_subplot(gs[1, 0])
        E_r_history = phi_mid_history / C_LIGHT
        e_mat_history = C_V * T_mid_history
        E_total = E_r_history + e_mat_history
        ax.plot(times, E_r_history, color=color, linestyle='--', linewidth=2, 
                alpha=0.7, label=f'{method_name} E_r')
        ax.plot(times, e_mat_history, color=color, linestyle=':', linewidth=2,
                alpha=0.7, label=f'{method_name} e_mat')
        ax.plot(times, E_total, color=color, linestyle='-', linewidth=2.5,
                marker=marker, markersize=4, markevery=max(1, len(times)//10),
                label=f'{method_name} E_total')
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('Energy Density (GJ/cm³)', fontsize=11)
        ax.set_title('Energy Components', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        
        # Plot 5: Energy conservation
        ax = fig.add_subplot(gs[1, 1])
        rel_energy_change = (E_total - E_total[0]) / E_total[0]
        ax.plot(times, rel_energy_change, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//10), label=method_name)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('Relative Energy Change', fontsize=11)
        ax.set_title('Energy Conservation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Plot 6: Spatial uniformity (std dev over time)
        ax = fig.add_subplot(gs[1, 2])
        T_std_history = [T_history[i].std() for i in range(len(times))]
        ax.plot(times, T_std_history, color=color, marker=marker, linewidth=2,
                markersize=4, markevery=max(1, len(times)//10), label=method_name)
        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('T std dev (keV)', fontsize=11)
        ax.set_title('Spatial Uniformity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Plot 7-9: Final spatial profiles for each method
    for idx, (method_name, result_data) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, idx])
        
        T_final_2d = result_data['final_T_2d']
        
        # Create a heatmap
        im = ax.imshow(T_final_2d.T, origin='lower', cmap='hot', 
                      extent=[0, 1, 0, 1], aspect='auto')
        ax.set_xlabel('x (cm)', fontsize=11)
        ax.set_ylabel('y (cm)', fontsize=11)
        ax.set_title(f'{method_name}: Final T_mat', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('T (keV)', fontsize=10)
    
    plt.suptitle('2D Non-Equilibrium Infinite Medium: Method Comparison', 
                 fontsize=14, fontweight='bold')
    
    output_file = 'test_equilibrium_2d_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    plt.close()
    
    # Validation summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    print("\nExpected behavior:")
    print("  1. T_mat and T_rad should converge towards each other")
    print("  2. Solution should remain spatially uniform (low std dev)")
    print("  3. Total energy should be conserved")
    print("  4. All methods should give similar results")
    
    print("\nObserved behavior:")
    all_passed = True
    
    for method_name, result_data in results.items():
        times = result_data['times']
        T_history = result_data['T_history']
        phi_history = result_data['phi_history']
        
        T_mid_history = np.array([T_history[i][mid_idx] for i in range(len(times))])
        phi_mid_history = np.array([phi_history[i][mid_idx] for i in range(len(times))])
        T_rad_history = (phi_mid_history / (A_RAD * C_LIGHT))**0.25
        
        # Check convergence
        final_temp_diff = abs(T_rad_history[-1] - T_mid_history[-1])
        initial_temp_diff = abs(T_rad_history[0] - T_mid_history[0])
        convergence_ok = final_temp_diff < initial_temp_diff
        
        # Check spatial uniformity
        T_final = result_data['final_T']
        uniformity_ok = T_final.std() < 1e-4
        
        # Check energy conservation
        E_r_history = phi_mid_history / C_LIGHT
        e_mat_history = C_V * T_mid_history
        E_total = E_r_history + e_mat_history
        energy_conservation_ok = abs(E_total[-1] - E_total[0]) / E_total[0] < 1e-6
        
        print(f"\n  {method_name}:")
        print(f"    Convergence: {'✓' if convergence_ok else '✗'} "
              f"(ΔT: {initial_temp_diff:.4f} → {final_temp_diff:.6f} keV)")
        print(f"    Uniformity: {'✓' if uniformity_ok else '✗'} "
              f"(std dev: {T_final.std():.3e} keV)")
        print(f"    Energy conservation: {'✓' if energy_conservation_ok else '✗'} "
              f"(relative change: {abs(E_total[-1] - E_total[0]) / E_total[0]:.3e})")
        
        if not (convergence_ok and uniformity_ok and energy_conservation_ok):
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All validation checks PASSED")
    else:
        print("✗ Some validation checks FAILED")
    print("="*70)
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
