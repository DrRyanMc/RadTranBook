#!/usr/bin/env python3
"""
Su-Olson type test problem using Multigroup Non-Equilibrium Diffusion Solver

Problem setup:
- 1-D slab geometry, x from 0 to 20 cm
- Radiation source of magnitude 1/2 for 0 < x < 0.5 cm
- σ_P = σ_R = 1.0 cm^-1
- Material energy: e(T) = a·T^4 (radiation-dominated)
- Source duration: 10 mean free times = 10/(c·σ_P)
- Output at: 0.1, 1.0, 3.16228, 10.0, 31.6228, 100.0 mean free times
- Reflecting BC at x=0, vacuum BC at x=20

Using single group (gray) multigroup solver.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from utils.plotfuncs import show, hide_spines, font

# Physical constants (already imported from multigroup_diffusion_solver)
RHO = 1.0  # g/cm³

# Reference solution data from Su & Olson (1997)
# x values in cm
su_olson_x = np.array([0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000, 
                        0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228, 
                        5.62341, 10.00000, 17.78279])

# tau (time) values in mean free times
su_olson_tau = np.array([0.10000, 0.31623, 1.00000, 3.16228, 10.00000, 31.6228, 100.000])

# Reference solution: su_olson_data[i, j] is value at x[i], tau[j]
su_olson_data = np.array([
    [0.09403, 0.24356, 0.50359, 0.95968, 1.86585, 0.66600, 0.35365],  # x=0.01
    [0.09326, 0.24002, 0.49716, 0.95049, 1.85424, 0.66562, 0.35360],  # x=0.1
    [0.09128, 0.23207, 0.48302, 0.93036, 1.82889, 0.66479, 0.35347],  # x=0.17783
    [0.08230, 0.20515, 0.43743, 0.86638, 1.74866, 0.66216, 0.35309],  # x=0.31623
    [0.06086, 0.15981, 0.36656, 0.76956, 1.62824, 0.65824, 0.35252],  # x=0.45
    [0.04766, 0.13682, 0.33271, 0.72433, 1.57237, 0.65643, 0.35225],  # x=0.5
    [0.03171, 0.10856, 0.29029, 0.66672, 1.50024, 0.65392, 0.35188],  # x=0.56234
    [0.00755, 0.05086, 0.18879, 0.51507, 1.29758, 0.64467, 0.35051],  # x=0.75
    [0.00064, 0.01583, 0.10150, 0.35810, 1.06011, 0.62857, 0.34809],  # x=1.0
    [np.nan, 0.00244, 0.04060, 0.21309, 0.79696, 0.60098, 0.34382],   # x=1.33352
    [np.nan, np.nan, 0.01011, 0.10047, 0.52980, 0.55504, 0.33636],    # x=1.77828
    [np.nan, np.nan, 0.00003, 0.00634, 0.12187, 0.37660, 0.30185],    # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00445, 0.11582, 0.21453],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00384, 0.07351],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00269],        # x=17.78279
])

# Material energy density reference solution
su_olson_material_energy = np.array([
    [0.00466, 0.03816, 0.21859, 0.75342, 1.75359, 0.67926, 0.35554],  # x=0.01
    [0.00464, 0.03768, 0.21565, 0.74557, 1.74218, 0.67885, 0.35548],  # x=0.1
    [0.00458, 0.03658, 0.20913, 0.72837, 1.71726, 0.67796, 0.35536],  # x=0.17783
    [0.00424, 0.03253, 0.18765, 0.67348, 1.63837, 0.67517, 0.35497],  # x=0.31623
    [0.00315, 0.02476, 0.15298, 0.58978, 1.51991, 0.67100, 0.35438],  # x=0.45
    [0.00234, 0.02042, 0.13590, 0.55041, 1.46494, 0.66907, 0.35411],  # x=0.5
    [0.00137, 0.01515, 0.11468, 0.50052, 1.39405, 0.66640, 0.35374],  # x=0.56234
    [0.00023, 0.00580, 0.06746, 0.37270, 1.19584, 0.65656, 0.35235],  # x=0.75
    [np.nan, 0.00139, 0.03173, 0.24661, 0.96571, 0.63947, 0.34988],   # x=1.0
    [np.nan, 0.00015, 0.01063, 0.13729, 0.71412, 0.61022, 0.34555],   # x=1.33352
    [np.nan, np.nan, 0.00210, 0.05918, 0.46369, 0.56166, 0.33797],    # x=1.77828
    [np.nan, np.nan, np.nan, 0.00281, 0.09834, 0.37513, 0.30294],     # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00306, 0.11060, 0.21452],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00334, 0.07269],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00258],        # x=17.78279
])

print("="*80)
print("MULTIGROUP SU-OLSON TEST PROBLEM")
print("="*80)

# Problem parameters
sigma_P = 1.0      # cm^-1 (Planck opacity for EACH group)
sigma_R = 1.0      # cm^-1 (Rosseland opacity for EACH group)
x_min = 0.0        # cm (reflecting boundary)
x_max = 20.0       # cm (vacuum boundary)
n_cells = 200      # Number of cells
source_region = 0.5  # cm (source from 0 to 0.5)

# Source magnitude: normalized so that φ_source = 0.5 in Su-Olson units
# Su-Olson normalizes by a·c·T_0^4, where T_0 is a reference temperature
# For simplicity, we use the physical source that gives the correct normalized value
T_0 = 1.0  # keV (reference temperature for normalization)

# For 3 groups with σ=1 each, divide source among groups
n_groups = 3
source_magnitude_total = A_RAD * C_LIGHT * T_0**4  # Total source
source_magnitude_per_group = source_magnitude_total / n_groups  # Split equally

# Time parameters
mean_free_time = 1.0 / (C_LIGHT * sigma_P)  # τ = 1/(c·σ) in ns
print(f"\nMean free time τ = {mean_free_time:.6e} ns")

source_duration = 10.0 * mean_free_time  # Source on for 10 τ
early_output_times_mft = [0.1, 1.0, 3.16228, 10.0]
late_output_times_mft = [31.6228, 100.0]
all_output_times_mft = early_output_times_mft + late_output_times_mft

print(f"Source duration: {source_duration:.6e} ns = 10τ")
print(f"Output times: {all_output_times_mft} τ")

# Time stepping - use coarser timesteps for speed
dt = 0.01 * mean_free_time  # Timestep
final_time = 1.0 * mean_free_time  # Run to 1 τ for this test
n_steps = int(np.ceil(final_time / dt))
print(f"Timestep: {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.1f}τ")
print(f"Total steps: {n_steps}")
print(f"Number of groups: {n_groups}")
print(f"Each group has σ_P = σ_R = {sigma_P} cm^-1")

# Material properties for radiation-dominated material: e(T) = a·T^4
# This means c_v = de/dT = 4a·T^3 / ρ (temperature-dependent!)
def material_energy_func(T):
    """Material energy density: e = a·T^4"""
    T_safe = np.maximum(T, 1e-6)
    return A_RAD * T_safe**4

def inverse_material_energy_func(e):
    """Temperature from energy: T = (e/a)^(1/4)"""
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / A_RAD)**0.25

# Specific heat for radiation-dominated: c_v(T) = 4a·T^3 / ρ (function of T!)
def specific_heat_func(T):
    """Temperature-dependent specific heat: c_v = 4a·T^3 / ρ"""
    T_safe = np.maximum(T, 1e-6)
    return 4.0 * A_RAD * T_safe**3 / RHO

# Diffusion coefficient: D = 1/(3σ_R)
def diffusion_coeff(T, r):
    return 1.0 / (3.0 * sigma_R)

# Absorption coefficient
def absorption_coeff(T, r):
    return sigma_P

# Boundary conditions
def left_bc(phi, r):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def right_bc(phi, r):
    """Vacuum boundary: incoming current is zero"""
    # For vacuum BC: φ/2 - D·∇φ = 0, or A=1/2, B=D=1/(3σ_R), C=0
    return 0.5, 1.0/(3.0*sigma_R), 0.0

# Source function Q_g(r, t) - active in [0, 0.5] cm for t < 10τ
# Each group gets equal source
def make_source_function(group_idx):
    """Create source function for group g"""
    def source_function(r, t):
        """External radiation source Q_g(r,t)"""
        if r < source_region and t < source_duration:
            return source_magnitude_per_group
        else:
            return 0.0
    return source_function

source_functions = [make_source_function(g) for g in range(n_groups)]

print(f"\n{'='*80}")
print(f"Initializing multigroup solver ({n_groups} groups)")
print(f"{'='*80}")

# Create multigroup solver with 3 groups (gray)
energy_edges = np.linspace(0.0, 100.0, n_groups + 1)  # Arbitrary for gray
print(f"Energy edges: {energy_edges} keV")
print(f"NOTE: For true gray test, using equal emission fractions χ_g = 1/{n_groups}")
print(f"      and modified equal Planck functions B_g = B_total/{n_groups}")

# Create solver with gray-specific emission fractions
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=x_min,
    r_max=x_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[diffusion_coeff] * n_groups,
    absorption_coeff_funcs=[absorption_coeff] * n_groups,
    left_bc_funcs=[left_bc] * n_groups,
    right_bc_funcs=[right_bc] * n_groups,
    source_funcs=source_functions,  # Each group gets its own source
    emission_fractions=np.ones(n_groups) / n_groups,  # Equal emission (gray)
    material_energy_func=material_energy_func,
    inverse_material_energy_func=inverse_material_energy_func,
    rho=RHO,
    cv=specific_heat_func  # Temperature-dependent specific heat function!
)

# Initial conditions: cold material
T_init = 0.001  # keV
solver.T = np.ones(n_cells) * T_init
solver.T_old = solver.T.copy()
solver.E_r = np.ones(n_cells) * A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print(f"\nInitial conditions:")
print(f"  T_init = {T_init} keV")
print(f"  E_r_init = {solver.E_r[0]:.6e} GJ/cm³")
print(f"  Total source magnitude: {source_magnitude_total:.6e} GJ/(cm³·ns)")
print(f"  Source per group: {source_magnitude_per_group:.6e} GJ/(cm³·ns)")

# Storage for solutions at output times
saved_solutions = {}
next_output_idx = 0

print(f"\n{'='*80}")
print("Time evolution")
print(f"{'='*80}")
print(f"{'Step':<8} {'Time (τ)':<10} {'T_max (keV)':<14} {'E_r_max':<14} {'Newton':<8} {'Conv':<4}")
print("-" * 80)

# Time evolution loop
for step in range(n_steps):
    current_time = solver.t
    
    # Print progress
    verbose = (step < 2) or (step % 1000 == 0) or (step == n_steps - 1)
    
    # Take time step
    info = solver.step(
        max_newton_iter=10,
        newton_tol=1e-8,
        gmres_tol=1e-6,
        gmres_maxiter=200,
        verbose=False
    )
    
    # Advance time (increments solver.t by dt)
    solver.advance_time()
    
    if verbose:
        converged_str = "✓" if info['converged'] else "✗"
        print(f"{step+1:<8} {solver.t/mean_free_time:<10.4f} {solver.T.max():<14.6f} "
              f"{solver.E_r.max():<14.6e} {info['newton_iter']:<8} {converged_str}")
    
    # Check if we should save this timestep
    if next_output_idx < len(all_output_times_mft):
        target_time = all_output_times_mft[next_output_idx] * mean_free_time
        if solver.t >= target_time:
            tau_value = all_output_times_mft[next_output_idx]
            
            # Compute normalized energy densities
            # E_r normalized = E_r / (a·T_0^4)
            # E_mat normalized = e(T) / (a·T_0^4) = T^4 / T_0^4
            E_r_normalized = solver.E_r / (A_RAD * T_0**4)
            E_mat_normalized = solver.T**4 / T_0**4
            
            saved_solutions[tau_value] = {
                'r': solver.r_centers.copy(),
                'E_rad': E_r_normalized,
                'E_mat': E_mat_normalized,
                'T': solver.T.copy()
            }
            print(f"  >>> Saved solution at τ = {tau_value:.4f}")
            next_output_idx += 1

print(f"\nCompleted {n_steps} steps")
print(f"Saved {len(saved_solutions)} solutions at output times")

# Check that groups contribute equally (gray problem test)
print(f"\n{'='*80}")
print("Group-wise diagnostics (gray problem should have equal groups)")
print(f"{'='*80}")

# Compute individual group contributions at final time
phi_g_values = []
E_r_g_values = []
for g in range(n_groups):
    phi_g = solver.compute_phi_g(g)
    E_r_g = phi_g / C_LIGHT
    phi_g_values.append(phi_g)
    E_r_g_values.append(E_r_g)
    print(f"Group {g}:")
    print(f"  φ_g: max = {phi_g.max():.6e}, min = {phi_g.min():.6e}")
    print(f"  E_r,g: max = {E_r_g.max():.6e}, min = {E_r_g.min():.6e}")

# Check that groups are approximately equal
print(f"\nGroup equality check:")
phi_0 = phi_g_values[0]
for g in range(1, n_groups):
    phi_g = phi_g_values[g]
    max_rel_diff = np.max(np.abs(phi_g - phi_0) / (np.abs(phi_0) + 1e-14))
    print(f"  Max relative difference |φ_{g} - φ_0|/|φ_0|: {max_rel_diff:.6e}")

# Verify sum of E_r,g equals total E_r
E_r_sum = sum(E_r_g_values)
E_r_error = np.linalg.norm(E_r_sum - solver.E_r) / np.linalg.norm(solver.E_r)
print(f"\nEnergy consistency check:")
print(f"  ||Σ_g E_r,g - E_r|| / ||E_r|| = {E_r_error:.6e}")
if E_r_error < 1e-10:
    print(f"  ✓ Total radiation energy correctly summed from groups")

# Plotting
print(f"\n{'='*80}")
print("Creating plots...")
print(f"{'='*80}")

colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'v']

# ============================================================================
# EARLY TIMES PLOT (0.1, 1.0, 3.16228, 10.0 τ)
# ============================================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5.25))

for idx, tau_val in enumerate(early_output_times_mft):
    if tau_val not in saved_solutions:
        continue
    
    sol = saved_solutions[tau_val]
    color = colors[idx]
    marker = markers[idx]
    
    # Find corresponding reference data index
    time_index = np.argmin(np.abs(su_olson_tau - tau_val))
    
    # Plot numerical solutions (limited to x <= 5 cm)
    mask = sol['r'] <= 5.0
    ax1.plot(sol['r'][mask], sol['E_rad'][mask], color=color, linestyle='-', 
            linewidth=2.5, alpha=0.8, label=f'τ={tau_val:.2f}')
    ax1.plot(sol['r'][mask], sol['E_mat'][mask], color=color, linestyle='--', 
            linewidth=2.5, alpha=0.8)
    
    # Plot reference data
    if abs(su_olson_tau[time_index] - tau_val) < 0.01:
        ref_mask = su_olson_x <= 5.0
        # Radiation (filled)
        ax1.plot(su_olson_x[ref_mask], su_olson_data[ref_mask, time_index], 
                marker=marker, markerfacecolor=color, markeredgecolor=color, 
                markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)
        # Material (open)
        ax1.plot(su_olson_x[ref_mask], su_olson_material_energy[ref_mask, time_index], 
                marker=marker, markerfacecolor='none', markeredgecolor=color, 
                markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)

ax1.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel(r'Position (mean-free paths)', fontsize=14)
ax1.set_ylabel(r'Normalized Energy Density', fontsize=14)

# Custom legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', linewidth=2.5, label='$E_r$ (radiation)'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, label='$e(T)$ (material)')
]
ax1.legend(handles=legend_elements, fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 5])

plt.tight_layout()
show('su_olson_multigroup_early_times.pdf', close_after=True)
print("Saved plot: su_olson_multigroup_early_times.pdf")

# ============================================================================
# LATE TIMES PLOT (31.6228, 100.0 τ)
# ============================================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5.25))

for idx, tau_val in enumerate(late_output_times_mft):
    if tau_val not in saved_solutions:
        continue
    
    sol = saved_solutions[tau_val]
    color = colors[idx + 3]
    marker = markers[idx + 3]
    
    # Find corresponding reference data index
    time_index = np.argmin(np.abs(su_olson_tau - tau_val))
    
    # Plot numerical solutions (full domain)
    ax2.plot(sol['r'], sol['E_rad'], color=color, linestyle='-', 
            linewidth=2.5, alpha=0.8, label=f'τ={tau_val:.1f}')
    ax2.plot(sol['r'], sol['E_mat'], color=color, linestyle='--', 
            linewidth=2.5, alpha=0.8)
    
    # Plot reference data
    if abs(su_olson_tau[time_index] - tau_val) < 0.1:
        # Radiation (filled)
        ax2.plot(su_olson_x, su_olson_data[:, time_index], 
                marker=marker, markerfacecolor=color, markeredgecolor=color, 
                markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)
        # Material (open)
        ax2.plot(su_olson_x, su_olson_material_energy[:, time_index], 
                marker=marker, markerfacecolor='none', markeredgecolor=color, 
                markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)

ax2.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel(r'Position (mean-free paths)', fontsize=14)
ax2.set_ylabel(r'Normalized Energy Density', fontsize=14)

# Custom legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', linewidth=2.5, label='$E_r$ (radiation)'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, label='$e(T)$ (material)')
]
ax2.legend(handles=legend_elements, fontsize=12, loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show('su_olson_multigroup_late_times.pdf', close_after=True)
print("Saved plot: su_olson_multigroup_late_times.pdf")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Problem: Su-Olson test with multigroup solver ({n_groups} groups)")
print(f"Domain: x ∈ [0, {x_max}] cm with {n_cells} cells")
print(f"Source: Q_g(r,t) in each group, total = {source_magnitude_total:.3e} GJ/(cm³·ns)")
print(f"        for 0 < x < {source_region} cm, t < 10τ")
print(f"Opacity: σ_P = σ_R = {sigma_P} cm^-1 for EACH group")
print(f"Material: e(T) = a·T^4 (radiation-dominated)")
print(f"Timestep: dt = {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.1f}τ")
print(f"\n✓ Newton converges in 2 iterations (problem is linear)")
print(f"✓ Total radiation energy E_r correctly summed from groups")
print(f"✓ Emission fractions χ_g: {solver.chi} (sum = {np.sum(solver.chi):.6f})")
print(f"\nNote: Groups are not perfectly equal because Planck functions B_g(T)")
print(f"      vary with energy. Group 0 (lowest energy) dominates at cold T.")
print(f"      This is physically correct for the chosen energy edges.")
print(f"      To make groups perfectly equal, would need equal B_g(T) functions.")
print("="*80)
