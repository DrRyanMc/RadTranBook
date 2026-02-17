#!/usr/bin/env python3
"""
Debug why Marshak wave doesn't propagate with theta < 1
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import RadiationDiffusionSolver, A_RAD, temperature_from_Er

# Marshak wave parameters
SIGMA_R = 300.0  # Correct value from marshak_wave.py
C_V = 0.3

def marshak_opacity(Er):
    """σ_R = 300 * T^-3"""
    T = temperature_from_Er(Er)
    T_min = 0.001
    if T < T_min:
        T = T_min
    return SIGMA_R / T**3

def marshak_cv(T):
    return C_V

def marshak_material_energy(T):
    return C_V * T

def marshak_left_bc(Er, r):
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def marshak_right_bc(Er, r):
    return 1.0, -1.0, 0.0

# Test with different theta values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

theta_values = [1.0, 0.75, 0.5]
colors = ['red', 'blue', 'green']
t_final = 5.0  # ns (same as marshak_wave.py)
dt = 0.01

for theta, color in zip(theta_values, colors):
    print(f"\n{'='*70}")
    print(f"Testing with theta = {theta}")
    print('='*70)
    
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=0.3, n_cells=100, d=0,  # Same domain as marshak_wave.py
        dt=dt, theta=theta,
        max_newton_iter=20, newton_tol=1e-6,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_cv,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    # Cold initial condition
    T_init = 0.01
    Er_init = A_RAD * T_init**4
    solver.set_initial_condition(Er_init)
    
    # Track solution at multiple times
    n_steps = int(t_final / dt)
    snapshot_steps = [0, n_steps//4, n_steps//2, n_steps]
    
    for i, step in enumerate(snapshot_steps):
        if step > 0:
            steps_to_take = step - (snapshot_steps[i-1] if i > 0 else 0)
            solver.time_step(n_steps=steps_to_take, verbose=False)
        
        r, Er = solver.get_solution()
        T = np.array([temperature_from_Er(er) for er in Er])
        
        t = step * dt
        alpha = 0.3 + 0.7 * (i / len(snapshot_steps))
        
        if i < 3:
            axes[0, 0].plot(r, T, color=color, alpha=alpha, 
                           label=f'θ={theta}, t={t:.2f}' if i == 2 else None)
        
    # Final profile
    print(f"\nFinal results (t={t_final}):")
    print(f"  T at x=0: {T[0]:.4f} keV")
    print(f"  T at x=1: {T[20]:.4f} keV")
    print(f"  T at x=2: {T[40]:.4f} keV")
    print(f"  Max T: {T.max():.4f} keV")
    print(f"  Wave penetration (T>0.5): {r[T>0.5].max() if np.any(T>0.5) else 0:.4f} cm")
    
    axes[0, 1].plot(r, T, 'o-', color=color, label=f'θ={theta}', markersize=3)

# Plot styling
axes[0, 0].set_xlabel('Position (cm)')
axes[0, 0].set_ylabel('Temperature (keV)')
axes[0, 0].set_title('Temperature Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim([0, 0.25])

axes[0, 1].set_xlabel('Position (cm)')
axes[0, 1].set_ylabel('Temperature (keV)')
axes[0, 1].set_title(f'Final Temperature Profile (t={t_final})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, 0.25])

# Test 2: Look at energy balance
print(f"\n{'='*70}")
print("Energy balance check")
print('='*70)

for theta in [1.0, 0.5]:
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=5.0, n_cells=50, d=0,
        dt=0.02, theta=theta,
        max_newton_iter=20, newton_tol=1e-6,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_cv,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    solver.set_initial_condition(Er_init)
    
    times = []
    total_energies = []
    
    for step in range(30):
        r, Er = solver.get_solution()
        times.append(step * solver.dt)
        total_energies.append(np.sum(Er * solver.V_cells))
        solver.time_step(n_steps=1, verbose=False)
    
    axes[1, 0].plot(times, total_energies, 'o-', label=f'θ={theta}', markersize=4)
    
    print(f"\nθ={theta}:")
    print(f"  Initial energy: {total_energies[0]:.4e}")
    print(f"  Final energy: {total_energies[-1]:.4e}")
    print(f"  Energy increase: {total_energies[-1] - total_energies[0]:.4e}")

axes[1, 0].set_xlabel('Time (ns)')
axes[1, 0].set_ylabel('Total Radiation Energy')
axes[1, 0].set_title('Energy Evolution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Test 3: Check what happens at boundary
print(f"\n{'='*70}")
print("Boundary behavior check")
print('='*70)

for theta in [1.0, 0.5]:
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=2.0, n_cells=20, d=0,
        dt=0.05, theta=theta,
        max_newton_iter=20, newton_tol=1e-6,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_cv,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    solver.set_initial_condition(Er_init)
    
    r, Er_before = solver.get_solution()
    T_before = np.array([temperature_from_Er(er) for er in Er_before])
    
    solver.time_step(n_steps=1, verbose=True)
    
    r, Er_after = solver.get_solution()
    T_after = np.array([temperature_from_Er(er) for er in Er_after])
    
    print(f"\nθ={theta}:")
    print(f"  T[0] before: {T_before[0]:.6f}, after: {T_after[0]:.6f}, change: {T_after[0]-T_before[0]:.6e}")
    print(f"  T[1] before: {T_before[1]:.6f}, after: {T_after[1]:.6f}, change: {T_after[1]-T_before[1]:.6e}")
    print(f"  T[2] before: {T_before[2]:.6f}, after: {T_after[2]:.6f}, change: {T_after[2]-T_before[2]:.6e}")

axes[1, 1].text(0.5, 0.5, 'See terminal output\nfor detailed diagnostics', 
                ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('debug_marshak_theta.png', dpi=150)
print(f"\n{'='*70}")
print("Plot saved as 'debug_marshak_theta.png'")
print('='*70)
plt.show()
