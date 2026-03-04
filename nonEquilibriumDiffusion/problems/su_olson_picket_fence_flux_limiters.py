#!/usr/bin/env python3
"""
Su-Olson Picket Fence with Flux Limiter Comparison

Compares different flux limiters for the two-group picket fence problem:
- Group 1: σ₁ = 2/11 cm⁻¹ (optically thin)
- Group 2: σ₂ = 20/11 cm⁻¹ (optically thick)
- Equal Planck functions: B_g = 0.5 * (a*c*T⁴)/(4π)
- Source: Q_g = 0.5 in each group for x < 0.5 cm
- Material: e(T) = a*T⁴, C_v = 4*a*T³

Flux limiters tested:
- Levermore-Pomraning
- Larsen n=2
- Sum
- Max

Runs to τ=1.0 and compares with transport solutions from Table 4.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D,
    C_LIGHT, A_RAD,
    flux_limiter_levermore_pomraning,
    flux_limiter_larsen,
    flux_limiter_sum,
    flux_limiter_max
)

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from utils.plotfuncs import show, hide_spines, font

# Transport reference data from Table 4 of Su & Olson (1997)
# Format: x, tau=0.1, tau=0.3, tau=1.0, tau=3.0
transport_U1 = np.array([
    [0.00, 0.04956, 0.14632, 0.39890, 0.65095],
    [0.10, 0.04956, 0.14632, 0.39418, 0.64570],
    [0.30, 0.04956, 0.14181, 0.35349, 0.60076],
    [0.45, 0.04578, 0.10753, 0.28118, 0.52264],
    [0.50, 0.02478, 0.07316, 0.23277, 0.47187],
    [0.55, 0.00378, 0.03879, 0.18410, 0.42049],
    [0.75, np.nan, 0.00105, 0.09013, 0.31167],
    [1.00, np.nan, np.nan, 0.03332, 0.22950],
    [1.35, np.nan, np.nan, 0.00250, 0.15410],
    [1.80, np.nan, np.nan, np.nan, 0.09082],
    [2.35, np.nan, np.nan, np.nan, 0.04035],
    [3.15, np.nan, np.nan, np.nan, 0.00316],
])

transport_U2 = np.array([
    [0.00, 0.04585, 0.11858, 0.26676, 0.51786],
    [0.10, 0.04585, 0.11858, 0.26401, 0.51216],
    [0.30, 0.04585, 0.11563, 0.23839, 0.46275],
    [0.45, 0.04254, 0.08956, 0.18496, 0.37580],
    [0.50, 0.02293, 0.05929, 0.14353, 0.32037],
    [0.55, 0.00331, 0.02902, 0.10206, 0.26455],
    [0.75, np.nan, 0.00066, 0.03736, 0.15017],
    [1.00, np.nan, np.nan, 0.01016, 0.07857],
    [1.35, np.nan, np.nan, 0.00054, 0.03199],
    [1.80, np.nan, np.nan, np.nan, 0.00963],
    [2.35, np.nan, np.nan, np.nan, 0.00182],
    [3.15, np.nan, np.nan, np.nan, 0.00004],
])

transport_V = np.array([
    [0.00, 0.00458, 0.03511, 0.23884, 0.81005],
    [0.10, 0.00458, 0.03511, 0.23680, 0.80144],
    [0.30, 0.00458, 0.03489, 0.21640, 0.72556],
    [0.45, 0.00446, 0.02893, 0.16752, 0.58601],
    [0.50, 0.00229, 0.01756, 0.12369, 0.49152],
    [0.55, 0.00012, 0.00617, 0.07986, 0.39651],
    [0.75, np.nan, 0.00002, 0.02270, 0.21608],
    [1.00, np.nan, np.nan, 0.00427, 0.11109],
    [1.35, np.nan, np.nan, 0.00007, 0.04604],
    [1.80, np.nan, np.nan, np.nan, 0.01513],
    [2.35, np.nan, np.nan, np.nan, 0.00336],
    [3.15, np.nan, np.nan, np.nan, 0.00007],
])

print("="*80)
print("SU-OLSON PICKET FENCE WITH FLUX LIMITERS")
print("="*80)

# Problem parameters
n_groups = 2
x_min = 0.0
x_max = 10.0
n_cells = 200
source_region = 0.5  # cm

# Opacities for picket fence (cm⁻¹)
sigma_1 = 2.0 / 11.0   # Group 1: optically thin
sigma_2 = 20.0 / 11.0  # Group 2: optically thick

print(f"\nProblem setup:")
print(f"  Groups: {n_groups}")
print(f"  σ₁ = {sigma_1:.6f} cm⁻¹ (optically thin)")
print(f"  σ₂ = {sigma_2:.6f} cm⁻¹ (optically thick)")
print(f"  Domain: [{x_min}, {x_max}] cm with {n_cells} cells")

# Time parameters
sigma_avg = (sigma_1 + sigma_2) / 2.0
mean_free_time = 1.0 / (C_LIGHT * sigma_avg)
print(f"\nMean free time τ = {mean_free_time:.6e} ns")

# Run to τ=1.0 only
output_times_mft = [0.1, 0.3, 1.0]
dt = 0.01 * mean_free_time
final_time = 1.0 * mean_free_time
n_steps = int(np.ceil(final_time / dt))

print(f"Timestep: {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.1f}τ")
print(f"Total steps: {n_steps}")
print(f"Output times: {output_times_mft} τ")

# Reference temperature
T_h = 1.0  # keV

# Custom Planck functions for picket fence
def planck_picket(T):
    """B_g = 0.5 * (a*c*T⁴)/(4π)"""
    T_safe = np.maximum(T, 1e-6)
    return 0.5 * (A_RAD * C_LIGHT * T_safe**4) / (4.0 * np.pi)

def dplanck_dT_picket(T):
    """dB_g/dT = 2*a*c*T³/(4π)"""
    T_safe = np.maximum(T, 1e-6)
    return 2.0 * (A_RAD * C_LIGHT * T_safe**3) / (4.0 * np.pi)

# Material EOS
def material_energy(T):
    T_safe = np.maximum(T, 1e-6)
    return A_RAD * T_safe**4

def inverse_material_energy(e):
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / A_RAD)**0.25

def specific_heat(T):
    T_safe = np.maximum(T, 1e-6)
    return 4.0 * A_RAD * T_safe**3

# Rosseland opacities for flux limiters
def rosseland_1(T, r):
    return sigma_1

def rosseland_2(T, r):
    return sigma_2

# Absorption coefficients
def absorption_1(T, r):
    return sigma_1

def absorption_2(T, r):
    return sigma_2

# Boundary conditions
def left_bc(phi, r):
    """Reflecting boundary"""
    return 0.0, 1.0, 0.0

def right_bc(phi, r):
    """Vacuum boundary"""
    return 0.5, 1.0/(3.0*sigma_avg), 0.0

# Source functions
source_normalized = 0.5
source_magnitude = source_normalized * A_RAD * C_LIGHT * T_h**4

def source_1(r, t):
    if r < source_region:
        return source_magnitude
    return 0.0

def source_2(r, t):
    if r < source_region:
        return source_magnitude
    return 0.0

# Emission fractions
emission_fractions = np.array([0.5, 0.5])

# Define flux limiters to test
flux_limiters = {
    'Levermore-Pomraning': flux_limiter_levermore_pomraning,
    'Larsen n=2': flux_limiter_larsen,
    'Sum': flux_limiter_sum,
    'Max': flux_limiter_max
}

print(f"\nTesting {len(flux_limiters)} flux limiters")

# Storage for results
results = {}

# Energy edges (arbitrary since using custom Planck functions)
energy_edges = np.array([0.0, 50.0, 100.0])

print(f"\n{'='*80}")
print("Running with different flux limiters...")
print(f"{'='*80}")

# Run with each flux limiter
for limiter_name, limiter_func in flux_limiters.items():
    print(f"\n{'-'*60}")
    print(f"Running: {limiter_name}")
    print(f"{'-'*60}")
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=x_min,
        r_max=x_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=None,  # Let flux limiter create D = λ/σ_R
        absorption_coeff_funcs=[absorption_1, absorption_2],
        left_bc_funcs=[left_bc, left_bc],
        right_bc_funcs=[right_bc, right_bc],
        source_funcs=[source_1, source_2],
        emission_fractions=emission_fractions,
        planck_funcs=planck_picket,
        dplanck_dT_funcs=dplanck_dT_picket,
        material_energy_func=material_energy,
        inverse_material_energy_func=inverse_material_energy,
        cv=specific_heat,
        rho=1.0,
        flux_limiter_funcs=[limiter_func] * n_groups,
        rosseland_opacity_funcs=[rosseland_1, rosseland_2]
    )
    
    # Initial conditions
    T_init = 0.001  # keV
    solver.T = np.ones(n_cells) * T_init
    solver.T_old = solver.T.copy()
    solver.E_r = np.ones(n_cells) * A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    # Time evolution
    saved_solutions = {}
    next_output_idx = 0
    
    for step in range(n_steps):
        # Take time step
        info = solver.step(
            max_newton_iter=10,
            newton_tol=1e-8,
            gmres_tol=1e-6,
            gmres_maxiter=200,
            verbose=False
        )
        
        # Advance time
        solver.advance_time()
        
        # Progress
        verbose = (step < 2) or (step % 1000 == 0) or (step == n_steps - 1)
        if verbose:
            print(f"  Step {step+1}/{n_steps}: t={solver.t/mean_free_time:.3f}τ, "
                  f"T_max={solver.T.max():.6f} keV, E_r_max={solver.E_r.max():.6e}")
        
        # Save at output times
        if next_output_idx < len(output_times_mft):
            target_time = output_times_mft[next_output_idx] * mean_free_time
            if solver.t >= target_time:
                tau_value = output_times_mft[next_output_idx]
                
                # Compute group radiation energies
                phi_1 = solver.compute_phi_g(0)
                phi_2 = solver.compute_phi_g(1)
                E_r_1 = phi_1 / C_LIGHT
                E_r_2 = phi_2 / C_LIGHT
                
                # Normalize by a*T_h^4
                norm = A_RAD * T_h**4
                U1 = E_r_1 / norm
                U2 = E_r_2 / norm
                V = solver.T**4 / T_h**4
                
                saved_solutions[tau_value] = {
                    'r': solver.r_centers.copy(),
                    'U1': U1,
                    'U2': U2,
                    'V': V,
                    'T': solver.T.copy()
                }
                print(f"  >>> Saved solution at τ = {tau_value:.2f}")
                next_output_idx += 1
    
    results[limiter_name] = saved_solutions
    print(f"  Completed {limiter_name}")

print(f"\n{'='*80}")
print("Creating comparison plots...")
print(f"{'='*80}")

# Plot settings
colors_time = ['blue', 'green', 'red']
flux_limiter_colors = {
    'Levermore-Pomraning': 'red',
    'Larsen n=2': 'green',
    'Sum': 'black',
    'Max': 'purple'
}
flux_limiter_linestyles = {
    'Levermore-Pomraning': '--',
    'Larsen n=2': '-.',
    'Sum': '-',
    'Max': (0, (3, 1, 1, 1))
}

# Create three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax_U1, ax_U2, ax_V = axes

for idx, tau_val in enumerate(output_times_mft):
    time_color = colors_time[idx]
    
    # Find corresponding time index in transport data
    tau_transport = [0.1, 0.3, 1.0, 3.0]
    tau_idx = tau_transport.index(tau_val) if tau_val in tau_transport else None
    
    # Plot flux limiter results
    for limiter_name, limiter_results in results.items():
        if tau_val not in limiter_results:
            continue
        
        sol = limiter_results[tau_val]
        limiter_color = flux_limiter_colors[limiter_name]
        limiter_linestyle = flux_limiter_linestyles[limiter_name]
        
        mask = sol['r'] <= 5.0
        
        # U1
        ax_U1.plot(sol['r'][mask], sol['U1'][mask], color=limiter_color,
                  linestyle=limiter_linestyle, linewidth=2.0, alpha=0.7)
        
        # U2
        ax_U2.plot(sol['r'][mask], sol['U2'][mask], color=limiter_color,
                  linestyle=limiter_linestyle, linewidth=2.0, alpha=0.7)
        
        # V
        ax_V.plot(sol['r'][mask], sol['V'][mask], color=limiter_color,
                 linestyle=limiter_linestyle, linewidth=2.0, alpha=0.7)
    
    # Plot transport reference data
    if tau_idx is not None:
        ref_mask = transport_U1[:, 0] <= 5.0
        
        # U1 transport
        ax_U1.plot(transport_U1[ref_mask, 0], transport_U1[ref_mask, tau_idx+1],
                  marker='s', markerfacecolor=time_color, markeredgecolor='black',
                  markersize=6, markeredgewidth=1.5, linestyle='', alpha=0.8, zorder=10)
        
        # U2 transport
        ax_U2.plot(transport_U2[ref_mask, 0], transport_U2[ref_mask, tau_idx+1],
                  marker='s', markerfacecolor=time_color, markeredgecolor='black',
                  markersize=6, markeredgewidth=1.5, linestyle='', alpha=0.8, zorder=10)
        
        # V transport
        ax_V.plot(transport_V[ref_mask, 0], transport_V[ref_mask, tau_idx+1],
                 marker='s', markerfacecolor=time_color, markeredgecolor='black',
                 markersize=6, markeredgewidth=1.5, linestyle='', alpha=0.8, zorder=10)

# Configure U1 plot
ax_U1.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_U1.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_U1.set_ylabel(r'$U_1$ (Group 1 Radiation)', fontsize=12)
ax_U1.set_title('Group 1: σ₁ = 2/11 (Thin)', fontsize=13)
ax_U1.set_xscale('log')
ax_U1.set_yscale('log')
ax_U1.grid(True, alpha=0.3, which='both')
ax_U1.set_xlim([0.05, 5.0])
ax_U1.set_ylim([1e-3, 1e0])

# Create legend for U1
legend_elements = []
for limiter_name in flux_limiters.keys():
    legend_elements.append(
        Line2D([0], [0], color=flux_limiter_colors[limiter_name],
               linestyle=flux_limiter_linestyles[limiter_name],
               linewidth=2.0, label=limiter_name)
    )
legend_elements.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=6, linestyle='', label='Transport')
)
ax_U1.legend(handles=legend_elements, fontsize=9, loc='best')

# Configure U2 plot
ax_U2.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_U2.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_U2.set_ylabel(r'$U_2$ (Group 2 Radiation)', fontsize=12)
ax_U2.set_title('Group 2: σ₂ = 20/11 (Thick)', fontsize=13)
ax_U2.set_xscale('log')
ax_U2.set_yscale('log')
ax_U2.grid(True, alpha=0.3, which='both')
ax_U2.set_xlim([0.05, 5.0])
ax_U2.set_ylim([1e-4, 1e0])

# Configure V plot
ax_V.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_V.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_V.set_ylabel(r'$V$ (Material Energy)', fontsize=12)
ax_V.set_title('Material Energy Density', fontsize=13)
ax_V.set_xscale('log')
ax_V.set_yscale('log')
ax_V.grid(True, alpha=0.3, which='both')
ax_V.set_xlim([0.05, 5.0])
ax_V.set_ylim([1e-5, 1e0])

plt.tight_layout()
show('su_olson_picket_fence_flux_limiters.pdf', close_after=True)
print("Saved plot: su_olson_picket_fence_flux_limiters.pdf")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Picket Fence with flux limiters: Two-group problem")
print(f"  Group 1 (thin):  σ₁ = {sigma_1:.6f} cm⁻¹")
print(f"  Group 2 (thick): σ₂ = {sigma_2:.6f} cm⁻¹")
print(f"  Tested {len(flux_limiters)} flux limiters")
print(f"  Ran to τ = {max(output_times_mft):.1f}")
print(f"\n✓ Comparison with transport solutions from Su & Olson (1997) Table 4")
print(f"✓ Lines show flux limiter solutions, markers show transport reference")
print("="*80)
