#!/usr/bin/env python3
"""
Example: Time-Dependent Boundary Conditions

Demonstrates how to use time-dependent boundary conditions in the 2D solver.
Example problem: Heat source that ramps up over time at the left boundary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from twoDFV import RadiationDiffusionSolver2D, temperature_from_Er, A_RAD

# =============================================================================
# TIME-DEPENDENT BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def bc_left_ramping_source(Er_boundary, coord1_val, coord2_val, geometry, time=0.0):
    """
    Left boundary with time-dependent source that ramps up
    
    Temperature ramps from T=0.1 keV to T=0.5 keV over first 2 ns,
    then stays constant
    """
    # Ramp up source temperature over time
    t_ramp = 2.0  # ns
    if time < t_ramp:
        # Linear ramp
        T_source = 0.1 + (0.5 - 0.1) * (time / t_ramp)
    else:
        # Constant after ramp
        T_source = 0.5  # keV
    
    Er_source = A_RAD * T_source**4
    
    # Dirichlet boundary: Er = Er_source
    return 1.0, 0.0, Er_source


def bc_left_pulsed_source(Er_boundary, coord1_val, coord2_val, geometry, time=0.0):
    """
    Left boundary with pulsed source (sinusoidal)
    
    Temperature varies sinusoidally: T = T_avg + T_amplitude * sin(2π * time / period)
    """
    T_avg = 0.3  # keV (average temperature)
    T_amplitude = 0.1  # keV (amplitude of oscillation)
    period = 5.0  # ns (period of oscillation)
    
    T_source = T_avg + T_amplitude * np.sin(2.0 * np.pi * time / period)
    T_source = max(0.05, T_source)  # Keep positive
    
    Er_source = A_RAD * T_source**4
    
    # Dirichlet boundary: Er = Er_source
    return 1.0, 0.0, Er_source


def bc_left_step_function(Er_boundary, coord1_val, coord2_val, geometry, time=0.0):
    """
    Left boundary with step function at t=1 ns
    
    Low temperature before t=1 ns, high temperature after
    """
    if time < 1.0:
        T_source = 0.1  # keV (low)
    else:
        T_source = 0.5  # keV (high)
    
    Er_source = A_RAD * T_source**4
    
    # Dirichlet boundary: Er = Er_source
    return 1.0, 0.0, Er_source


def bc_right_open(Er_boundary, coord1_val, coord2_val, geometry, time=0.0):
    """Right boundary: Open (low constant value)"""
    return 1.0, 0.0, A_RAD * 0.01**4  # T = 0.01 keV


def bc_reflecting(Er_boundary, coord1_val, coord2_val, geometry, time=0.0):
    """Reflecting boundary (Neumann with zero flux)"""
    return 0.0, 1.0, 0.0


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_time_dependent_example(bc_type='ramping'):
    """
    Run example with time-dependent boundary conditions
    
    Parameters:
    -----------
    bc_type : str
        Type of time-dependent BC: 'ramping', 'pulsed', or 'step'
    """
    print("="*70)
    print(f"TIME-DEPENDENT BOUNDARY CONDITIONS EXAMPLE: {bc_type.upper()}")
    print("="*70)
    
    # Select boundary condition function
    if bc_type == 'ramping':
        left_bc = bc_left_ramping_source
    elif bc_type == 'pulsed':
        left_bc = bc_left_pulsed_source
    elif bc_type == 'step':
        left_bc = bc_left_step_function
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")
    
    # Create solver
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0,
        coord1_max=2.0,
        n1_cells=40,
        coord2_min=0.0,
        coord2_max=1.0,
        n2_cells=20,
        geometry='cartesian',
        dt=0.01,  # ns
        max_newton_iter=20,
        newton_tol=1e-6,
        left_bc_func=left_bc,
        right_bc_func=bc_right_open,
        bottom_bc_func=bc_reflecting,
        top_bc_func=bc_reflecting,
        theta=1.0,
        use_jfnk=False
    )
    
    # Set initial condition
    Er_init = A_RAD * 0.05**4  # Low background
    solver.set_initial_condition(Er_init)
    
    # Time evolution
    t_final = 5.0  # ns
    n_steps = int(t_final / solver.dt)
    
    # Storage for time history
    times = [0.0]
    left_boundary_temps = [0.05]  # Initial temperature
    center_temps = []
    
    # Get center cell index
    i_center = solver.n1_cells // 2
    j_center = solver.n2_cells // 2
    Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
    T_2d = temperature_from_Er(Er_2d)
    center_temps.append(T_2d[i_center, j_center])
    
    print(f"\nRunning for {n_steps} steps to t = {t_final} ns...")
    
    for step in range(n_steps):
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{n_steps}, t = {solver.current_time:.3f} ns")
        
        # Take time step
        Er_prev = solver.Er.copy()
        solver.Er = solver.newton_step_direct(Er_prev, verbose=False)
        solver.Er_old = Er_prev.copy()
        solver.current_time += solver.dt
        
        # Record temperatures
        times.append(solver.current_time)
        
        # Left boundary temperature (from BC function)
        A_bc, B_bc, C_bc = left_bc(0.0, 0.0, 0.5, 'cartesian', solver.current_time)
        if abs(A_bc) > 1e-14:
            Er_left = C_bc / A_bc
            T_left = temperature_from_Er(Er_left)
            left_boundary_temps.append(T_left)
        
        # Center temperature
        Er_2d = solver.Er.reshape((solver.n1_cells, solver.n2_cells))
        T_2d = temperature_from_Er(Er_2d)
        center_temps.append(T_2d[i_center, j_center])
    
    # Convert to arrays
    times = np.array(times)
    left_boundary_temps = np.array(left_boundary_temps)
    center_temps = np.array(center_temps)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Temperature history
    ax = axes[0]
    ax.plot(times, left_boundary_temps, 'b-', linewidth=2, label='Left Boundary (source)')
    ax.plot(times, center_temps, 'r--', linewidth=2, label='Center Point')
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title(f'Temperature History - {bc_type.capitalize()} BC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Final temperature profile
    ax = axes[1]
    x_centers, y_centers, Er_2d = solver.get_solution()
    T_2d = temperature_from_Er(Er_2d)
    
    # Plot x-slice at y=0.5
    j_mid = solver.n2_cells // 2
    ax.plot(x_centers, T_2d[:, j_mid], 'ko-', linewidth=2, markersize=4)
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title(f'Final Temperature Profile (t = {solver.current_time:.3f} ns)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'time_dependent_bc_{bc_type}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.close()
    
    print("\n" + "="*70)
    print("COMPLETED")
    print("="*70)
    
    return solver


if __name__ == "__main__":
    # Run examples with different time-dependent boundary conditions
    
    # Example 1: Ramping source
    print("\n\nExample 1: RAMPING SOURCE\n")
    solver1 = run_time_dependent_example('ramping')
    
    # Example 2: Pulsed source
    print("\n\nExample 2: PULSED SOURCE\n")
    solver2 = run_time_dependent_example('pulsed')
    
    # Example 3: Step function
    print("\n\nExample 3: STEP FUNCTION\n")
    solver3 = run_time_dependent_example('step')
