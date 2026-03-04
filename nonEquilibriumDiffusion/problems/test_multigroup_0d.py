#!/usr/bin/env python3
"""
0-D Multigroup Test Problem
Simple equilibration between radiation and material with 2 energy groups

Problem setup:
- Single cell (0-D, no spatial variation)
- σ_a = 5 cm^-1 for both groups
- C_v = 0.01 GJ/(g·keV)
- T_init = 0.4 keV (material)
- T_r_init = 1.0 keV (radiation)
- dt = 0.01 ns
- 10 timesteps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

print("="*80)
print("0-D Multigroup Equilibration Test")
print("="*80)

# Problem parameters
n_groups = 2
n_cells = 1  # 0-D: single cell
r_min = 0.0
r_max = 1.0  # Arbitrary for 0-D

# Material properties
rho = 1.0  # g/cm³
cv = 0.01  # GJ/(g·keV)
sigma_a = 5.0  # cm^-1

# Constant properties
def const_diffusion(T, r):
    """Diffusion coefficient (large for 0-D to minimize spatial effects)"""
    return 1e10  # Very large D makes it effectively 0-D

def const_absorption(T, r):
    """Absorption coefficient"""
    return sigma_a

# Energy group edges (keV)
energy_edges = np.array([0.01, 2.0, 10.0])

# Boundary conditions (not important for 0-D)
left_bc_values = [0.0] * n_groups
right_bc_values = [0.0] * n_groups

print(f"Parameters:")
print(f"  Groups: {n_groups}")
print(f"  σ_a = {sigma_a} cm^-1 (both groups)")
print(f"  C_v = {cv} GJ/(g·keV)")
print(f"  ρ = {rho} g/cm³")
print(f"  Energy edges: {energy_edges} keV")

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=0.01,  # ns
    diffusion_coeff_funcs=[const_diffusion] * n_groups,
    absorption_coeff_funcs=[const_absorption] * n_groups,
    left_bc='neumann',  # Zero flux
    right_bc='neumann',
    left_bc_values=left_bc_values,
    right_bc_values=right_bc_values,
    rho=rho,
    cv=cv
)

# Initial conditions
T_init = 0.4  # keV (material)
T_r_init = 1.0  # keV (radiation)

solver.T = np.array([T_init])
solver.T_old = solver.T.copy()
solver.E_r = np.array([A_RAD * T_r_init**4])  # E_r = a*T_r^4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print(f"\nInitial conditions:")
print(f"  Material temperature: T = {T_init} keV")
print(f"  Radiation temperature: T_r = {T_r_init} keV")
print(f"  Radiation energy: E_r = {solver.E_r[0]:.6e} GJ/cm³")
print(f"  Expected equilibrium: T_eq ≈ {T_r_init:.3f} keV (hot radiation heats cold material)")

# Storage for plotting
times = [0.0]
T_history = [T_init]
T_r_history = [T_r_init]
E_r_history = [solver.E_r[0]]

# Energy conservation tracking
e_mat_init = rho * cv * T_init  # Material energy density
E_total_init = solver.E_r[0] + e_mat_init  # Total energy density
e_mat_history = [e_mat_init]
E_total_history = [E_total_init]

print(f"\nInitial energy budget:")
print(f"  Material energy: e = {e_mat_init:.6e} GJ/cm³")
print(f"  Radiation energy: E_r = {solver.E_r[0]:.6e} GJ/cm³")
print(f"  Total energy: E_total = {E_total_init:.6e} GJ/cm³")

# Time evolution
print(f"\n{'Step':<6} {'Time':<10} {'T (keV)':<12} {'T_rad (keV)':<12} {'E_r (GJ/cm³)':<15} {'ΔE/E_0':<12} {'Newton':<8} {'Conv':<4}")
print("-" * 95)

n_steps = 10
for step in range(n_steps):
    # Take timestep with verbose output on first step
    info = solver.step(max_newton_iter=10, newton_tol=1e-8,
                      gmres_tol=1e-6, gmres_maxiter=200,
                      verbose=(step == 0))
    
    solver.advance_time()
    
    # Compute radiation temperature
    T_rad = (solver.E_r[0] / A_RAD)**0.25
    
    # Compute material energy and total energy
    e_mat = rho * cv * solver.T[0]
    E_total = solver.E_r[0] + e_mat
    
    # Energy conservation error
    dE_rel = (E_total - E_total_init) / E_total_init
    
    # Store for plotting
    times.append((step + 1) * 0.01)
    T_history.append(solver.T[0])
    T_r_history.append(T_rad)
    E_r_history.append(solver.E_r[0])
    e_mat_history.append(e_mat)
    E_total_history.append(E_total)
    
    # Print progress
    conv_str = "✓" if info['converged'] else "✗"
    print(f"{step+1:<6} {times[-1]:<10.4f} {solver.T[0]:<12.6f} {T_rad:<12.6f} "
          f"{solver.E_r[0]:<15.6e} {dE_rel:<12.6e} {info['newton_iter']:<8} {conv_str}")

print("\n" + "="*80)
print("0-D test completed!")
print("="*80)

# Final state
print(f"\nFinal state at t = {times[-1]:.2f} ns:")
print(f"  Material temperature: T = {solver.T[0]:.6f} keV")
print(f"  Radiation temperature: T_r = {T_r_history[-1]:.6f} keV")
print(f"  ΔT = T - T_r = {solver.T[0] - T_r_history[-1]:.6e} keV")
print(f"  Radiation energy: E_r = {solver.E_r[0]:.6e} GJ/cm³")
print(f"  Material energy: e = {e_mat_history[-1]:.6e} GJ/cm³")

# Energy conservation check
print(f"\nEnergy conservation:")
print(f"  Initial total: E_0 = {E_total_init:.10e} GJ/cm³")
print(f"  Final total:   E_f = {E_total_history[-1]:.10e} GJ/cm³")
print(f"  Absolute error: ΔE = {E_total_history[-1] - E_total_init:.10e} GJ/cm³")
print(f"  Relative error: ΔE/E_0 = {(E_total_history[-1] - E_total_init)/E_total_init:.10e}")

max_dE = max(abs(E - E_total_init) for E in E_total_history)
max_dE_rel = max_dE / E_total_init
print(f"  Max deviation: {max_dE:.10e} GJ/cm³ ({max_dE_rel:.10e} relative)")

if max_dE_rel < 1e-10:
    print("  ✓ Energy is conserved to machine precision!")
elif max_dE_rel < 1e-6:
    print("  ✓ Energy is well conserved")
else:
    print("  ✗ WARNING: Significant energy conservation error!")

# Check group distribution
solver.update_absorption_coefficients(solver.T)
solver.fleck_factor = solver.compute_fleck_factor(solver.T)
xi_g_list = [solver.compute_source_xi(g, solver.T) for g in range(n_groups)]

print(f"\nGroup analysis:")
print(f"  Emission fractions χ: {solver.chi}")
for g in range(n_groups):
    phi_g = solver.compute_phi_g(g, solver.kappa, solver.T, xi_g_list)
    E_r_g = phi_g[0] / C_LIGHT
    fraction = E_r_g / solver.E_r[0]
    print(f"  Group {g}: E_r = {E_r_g:.6e} GJ/cm³ ({100*fraction:.2f}% of total)")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temperature evolution
ax = axes[0, 0]
ax.plot(times, T_history, 'b-', linewidth=2, marker='o', markersize=4, label='Material T')
ax.plot(times, T_r_history, 'r--', linewidth=2, marker='s', markersize=4, label='Radiation $T_{rad}$')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Temperature (keV)', fontsize=12)
ax.set_title('0-D Multigroup Equilibration', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Energy evolution
ax = axes[0, 1]
ax.plot(times, E_r_history, 'g-', linewidth=2, marker='o', markersize=4, label='$E_r$')
ax.plot(times, e_mat_history, 'orange', linewidth=2, marker='s', markersize=4, label='$e_{mat}$')
ax.plot(times, E_total_history, 'k--', linewidth=2, marker='^', markersize=4, label='$E_{total}$')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Energy Density (GJ/cm³)', fontsize=12)
ax.set_title('Energy Components', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Energy conservation error
ax = axes[1, 0]
dE_rel_history = [(E - E_total_init)/E_total_init for E in E_total_history]
ax.plot(times, dE_rel_history, 'r-', linewidth=2, marker='o', markersize=4)
ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('$(E_{total} - E_0) / E_0$', fontsize=12)
ax.set_title('Relative Energy Conservation Error', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Group distribution over time
ax = axes[1, 1]
group_fractions = []
for i, t in enumerate(times[1:], 1):  # Skip initial
    T = T_history[i]
    solver.T = np.array([T])
    solver.update_absorption_coefficients(solver.T)
    solver.fleck_factor = solver.compute_fleck_factor(solver.T)
    xi_g_list = [solver.compute_source_xi(g, solver.T) for g in range(n_groups)]
    
    fractions = []
    for g in range(n_groups):
        phi_g = solver.compute_phi_g(g, solver.kappa, solver.T, xi_g_list)
        E_r_g = phi_g[0] / C_LIGHT
        fractions.append(E_r_g / E_r_history[i])
    group_fractions.append(fractions)

group_fractions = np.array(group_fractions)
for g in range(n_groups):
    ax.plot(times[1:], group_fractions[:, g], linewidth=2, marker='o', markersize=3,
            label=f'Group {g} [{energy_edges[g]:.2f}-{energy_edges[g+1]:.1f} keV]')
    ax.axhline(solver.chi[g], color=f'C{g}', linestyle=':', linewidth=1, alpha=0.5,
               label=f'χ_{g} = {solver.chi[g]:.3f}')

ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('Group Fraction', fontsize=12)
ax.set_title('Group Energy Fractions', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('test_multigroup_0d.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'test_multigroup_0d.png'")
plt.show()
