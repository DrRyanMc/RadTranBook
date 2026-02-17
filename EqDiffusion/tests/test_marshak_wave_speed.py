#!/usr/bin/env python3
"""
Marshak Wave Front Speed Test

Test if nonlinear corrections affect wave front speed correctly.
For Marshak wave: higher T → lower opacity → higher diffusion → faster wave front.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    RadiationDiffusionSolver, 
    temperature_from_Er, 
    A_RAD, 
    C_LIGHT,
    RHO
)


def marshak_opacity(Er):
    """Temperature-dependent Rosseland opacity: σ_R = 3 * T^-3"""
    T = temperature_from_Er(Er)
    T_min = 0.05
    if T < T_min:
        T = T_min
    return 3.0 * T**(-3)


def marshak_specific_heat(T):
    cv_volumetric = 0.3
    return cv_volumetric / RHO


def marshak_material_energy(T):
    cv_volumetric = 0.3
    return cv_volumetric * T


def marshak_left_bc(Er, x):
    """Hot boundary at T = 1 keV"""
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc


def marshak_right_bc(Er, x):
    """Zero flux"""
    return 0.0, 1.0, 0.0


def find_wave_front_position(r, T, threshold_T=0.6):
    """Find position where T drops to threshold (wave front)"""
    if T.max() < threshold_T:
        return 0.0  # No wave front found
    
    # Find first position where T drops below threshold
    indices = np.where(T >= threshold_T)[0]
    if len(indices) == 0:
        return 0.0
    
    return r[indices[-1]]  # Rightmost position above threshold


def test_marshak_wave_speed():
    """Test Marshak wave front propagation speed"""
    
    print("="*70)
    print("MARSHAK WAVE FRONT SPEED TEST")
    print("="*70)
    
    # Problem setup - focus on early wave propagation
    r_min = 0.0
    r_max = 0.3  # Smaller domain to focus on wave front
    n_cells = 100
    dt = 0.0001  # Small time step for accuracy
    t_final = 0.05  # Longer time to see wave propagation
    
    print(f"Test setup:")
    print(f"  Domain: [0, 0.3] cm, {n_cells} cells")
    print(f"  Time: {t_final} ns, dt = {dt} ns")
    print(f"  Expected: Nonlinear wave should propagate faster")
    
    results = {}
    
    for use_nl, name in [(False, "Linear"), (True, "Nonlinear")]:
        print(f"\n--- {name} Case ---")
        
        # Create solver
        solver = RadiationDiffusionSolver(
            r_min=r_min,
            r_max=r_max,
            n_cells=n_cells,
            d=0,
            dt=dt,
            max_newton_iter=20,
            newton_tol=1e-8,
            rosseland_opacity_func=marshak_opacity,
            specific_heat_func=marshak_specific_heat,
            material_energy_func=marshak_material_energy,
            left_bc_func=marshak_left_bc,
            right_bc_func=marshak_right_bc
        )
        
        # Configure nonlinear corrections
        solver.use_nonlinear_correction = use_nl
        if use_nl:
            solver.use_secant_derivative = False
            solver.max_newton_iter_per_step = 20
            solver.nonlinear_skip_boundary_cells = 1  # Skip boundary cells
            solver.nonlinear_limiter = 0.3  # Moderate limiter
        
        # Cold initial condition (like original Marshak wave)
        def initial_Er(r):
            T_init = 0.1  # keV - cold material
            return np.full_like(r, A_RAD * T_init**4)
        
        solver.set_initial_condition(initial_Er)
        
        # Track wave front position over time
        times = []
        front_positions = []
        
        current_time = 0.0
        n_steps = int(t_final / dt)
        
        # Sample every 10 steps for analysis
        sample_interval = max(1, n_steps // 20)
        
        for step in range(n_steps):
            solver.time_step(n_steps=1, verbose=False)
            current_time += dt
            
            if step % sample_interval == 0:
                r, Er = solver.get_solution()
                T = temperature_from_Er(Er)
                
                # Find wave front position (where T > 0.6 keV)
                front_pos = find_wave_front_position(r, T, threshold_T=0.6)
                
                times.append(current_time)
                front_positions.append(front_pos)
                
                if step % (n_steps // 4) == 0:
                    print(f"  t = {current_time:.4f} ns: front at r = {front_pos:.4f} cm, max T = {T.max():.3f} keV")
        
        # Final state
        r_final, Er_final = solver.get_solution()
        T_final = temperature_from_Er(Er_final)
        final_front_pos = find_wave_front_position(r_final, T_final, threshold_T=0.6)
        
        print(f"  Final: front at r = {final_front_pos:.4f} cm, max T = {T_final.max():.3f} keV")
        
        results[name] = {
            'r': r_final,
            'T': T_final,
            'times': np.array(times),
            'front_positions': np.array(front_positions),
            'final_front_pos': final_front_pos
        }
    
    # Analysis
    print(f"\n{'='*70}")
    print("WAVE SPEED ANALYSIS")
    print(f"{'='*70}")
    
    linear_front = results["Linear"]['final_front_pos']
    nonlinear_front = results["Nonlinear"]['final_front_pos']
    
    print(f"Final wave front positions:")
    print(f"  Linear:    {linear_front:.6f} cm")
    print(f"  Nonlinear: {nonlinear_front:.6f} cm")
    print(f"  Difference: {nonlinear_front - linear_front:.6f} cm")
    
    # Expected: nonlinear should penetrate further (positive difference)
    if nonlinear_front > linear_front:
        print(f"  ✓ CORRECT: Nonlinear wave propagates faster (penetrates further)")
        speed_correct = True
    else:
        print(f"  ✗ WRONG: Nonlinear wave propagates slower")
        speed_correct = False
    
    # Compute average wave speeds
    linear_times = results["Linear"]['times']
    linear_pos = results["Linear"]['front_positions']
    nonlinear_times = results["Nonlinear"]['times']
    nonlinear_pos = results["Nonlinear"]['front_positions']
    
    # Linear fit to get average speed (skip early times where front position is 0)
    linear_valid = linear_pos > 0.001
    nonlinear_valid = nonlinear_pos > 0.001
    
    if np.sum(linear_valid) > 2 and np.sum(nonlinear_valid) > 2:
        linear_speed = np.polyfit(linear_times[linear_valid], linear_pos[linear_valid], 1)[0]
        nonlinear_speed = np.polyfit(nonlinear_times[nonlinear_valid], nonlinear_pos[nonlinear_valid], 1)[0]
        
        print(f"\nAverage wave speeds:")
        print(f"  Linear:    {linear_speed:.6f} cm/ns")
        print(f"  Nonlinear: {nonlinear_speed:.6f} cm/ns")
        print(f"  Ratio:     {nonlinear_speed/linear_speed:.6f}")
        
        if nonlinear_speed > linear_speed:
            print(f"  ✓ CORRECT: Nonlinear wave is faster")
        else:
            print(f"  ✗ WRONG: Nonlinear wave is slower")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Temperature profiles
    ax = axes[0]
    ax.plot(results["Linear"]['r'], results["Linear"]['T'], 'b-', linewidth=2, label='Linear')
    ax.plot(results["Nonlinear"]['r'], results["Nonlinear"]['T'], 'r-', linewidth=2, label='Nonlinear')
    ax.axhline(0.6, color='k', linestyle='--', alpha=0.5, label='Front threshold')
    ax.set_xlabel('Position r (cm)')
    ax.set_ylabel('Temperature T (keV)')
    ax.set_title(f'Final Temperature Profiles (t = {t_final} ns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Wave front position vs time
    ax = axes[1]
    if len(results["Linear"]['times']) > 0:
        ax.plot(results["Linear"]['times'], results["Linear"]['front_positions'], 'b-', linewidth=2, label='Linear')
        ax.plot(results["Nonlinear"]['times'], results["Nonlinear"]['front_positions'], 'r-', linewidth=2, label='Nonlinear')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Wave Front Position (cm)')
    ax.set_title('Wave Front Propagation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature difference
    ax = axes[2]
    if len(results["Linear"]['r']) == len(results["Nonlinear"]['r']):
        T_diff = results["Nonlinear"]['T'] - results["Linear"]['T']
        ax.plot(results["Linear"]['r'], T_diff, 'g-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position r (cm)')
        ax.set_ylabel('T_nonlinear - T_linear (keV)')
        ax.set_title('Nonlinear Effect on Temperature')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('marshak_wave_speed_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved as 'marshak_wave_speed_test.png'")
    plt.show()
    
    return speed_correct


if __name__ == "__main__":
    test_marshak_wave_speed()