#!/usr/bin/env python3
"""
Su-Olson Picket Fence Benchmark (Case A)

Two-group problem with:
- Group 1: σ₁ = 2/11 cm⁻¹ (optically thin group)
- Group 2: σ₂ = 20/11 cm⁻¹ (optically thick group)
- Equal Planck functions: B_g = 0.5 * (a*c*T⁴)/(4π) for both groups
- Source: Q_g = 0.5 in each group for x < 0.5 cm
- Material: e(T) = a*T⁴, C_v = 4*a*T³
- Reflecting BC at x=0, vacuum BC at x=20

Reference data from Table 2 of Su & Olson (1997).
Results are normalized by a*T_h⁴ where T_h is a reference temperature.
Time is normalized by τ = c*t.
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

# Reference solution data from Su & Olson (1997) Table 2
# Format: x, tau=0.1, tau=0.3, tau=1.0, tau=3.0
ref_data_U1 = np.array([
    [0.00, 0.03873, 0.08265, 0.16857, 0.30028],
    [0.20, 0.03649, 0.07923, 0.16426, 0.29531],
    [0.40, 0.02951, 0.06891, 0.15134, 0.28045],
    [0.50, 0.02402, 0.06109, 0.14164, 0.26934],
    [0.60, 0.01840, 0.05282, 0.13115, 0.25719],
    [0.80, 0.01027, 0.03884, 0.11187, 0.23393],
    [1.00, 0.00535, 0.02790, 0.09478, 0.21218],
    [1.25, 0.00214, 0.01785, 0.07633, 0.18718],
    [1.50, 0.00075, 0.01098, 0.06081, 0.16454],
    [1.75, 0.00023, 0.00649, 0.04793, 0.14418],
    [2.00, 0.00007, 0.00368, 0.03735, 0.12593],
])

ref_data_U2 = np.array([
    [0.00, 0.04578, 0.11394, 0.23845, 0.44678],
    [0.20, 0.04501, 0.10820, 0.22479, 0.42434],
    [0.40, 0.03684, 0.08416, 0.17908, 0.35415],
    [0.50, 0.02293, 0.05926, 0.14007, 0.29871],
    [0.60, 0.00902, 0.03433, 0.10051, 0.24141],
    [0.80, 0.00084, 0.00994, 0.04968, 0.15585],
    [1.00, 0.00004, 0.00232, 0.02364, 0.09963],
    [1.25, 0.00000, 0.00028, 0.00882, 0.05695],
    [1.50, 0.00000, 0.00003, 0.00323, 0.03321],
    [1.75, 0.00000, 0.00001, 0.00125, 0.02023],
    [2.00, 0.00000, 0.00000, 0.00057, 0.01312],
])

ref_data_V = np.array([
    [0.00, 0.00452, 0.03326, 0.20363, 0.66656],
    [0.20, 0.00447, 0.03200, 0.19265, 0.63301],
    [0.40, 0.00380, 0.02552, 0.15342, 0.52613],
    [0.50, 0.00229, 0.01736, 0.11740, 0.43986],
    [0.60, 0.00077, 0.00919, 0.08102, 0.35113],
    [0.80, 0.00011, 0.00258, 0.03862, 0.22372],
    [1.00, 0.00003, 0.00087, 0.01903, 0.14403],
    [1.25, 0.00001, 0.00035, 0.00882, 0.08608],
    [1.50, 0.00000, 0.00018, 0.00489, 0.05476],
    [1.75, 0.00000, 0.00009, 0.00314, 0.03754],
    [2.00, 0.00000, 0.00005, 0.00218, 0.02760],
])

print("="*80)
print("SU-OLSON PICKET FENCE BENCHMARK (Case A)")
print("="*80)

# Problem parameters
n_groups = 2
x_min = 0.0
x_max = 10.0
n_cells = 100
source_region = 0.5  # cm

# Opacities for picket fence (cm⁻¹)
sigma_1 = 2.0 / 11.0   # Group 1: optically thin
sigma_2 = 20.0 / 11.0  # Group 2: optically thick

print(f"\nProblem setup:")
print(f"  Groups: {n_groups}")
print(f"  σ₁ = {sigma_1:.6f} cm⁻¹ (optically thin)")
print(f"  σ₂ = {sigma_2:.6f} cm⁻¹ (optically thick)")
print(f"  Domain: [{x_min}, {x_max}] cm with {n_cells} cells")
print(f"  Source region: x < {source_region} cm")

# Time parameters (use weighted average opacity for time scale)
sigma_avg = (sigma_1 + sigma_2) / 2.0
mean_free_time = 1.0 / (C_LIGHT * sigma_avg)
print(f"\nTime scale (using average σ):")
print(f"  σ_avg = {sigma_avg:.6f} cm⁻¹")
print(f"  Mean free time τ = 1/(c·σ_avg) = {mean_free_time:.6e} ns")

# Time stepping
dt = 0.01 * mean_free_time
final_time = 3.0 * mean_free_time  # Run to τ=3.0
n_steps = int(np.ceil(final_time / dt))
output_times_mft = [0.1, 0.3, 1.0, 3.0]

print(f"\nTime stepping:")
print(f"  dt = {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"  Final time: {final_time:.6e} ns = {final_time/mean_free_time:.1f}τ")
print(f"  Total steps: {n_steps}")
print(f"  Output times: {output_times_mft} τ")

# Reference temperature for normalization
T_h = 1.0  # keV (reference temperature)

# Custom Planck functions: B_g = 0.5 * (a*c*T⁴)/(4π) for both groups
def planck_picket(T):
    """Equal Planck function for picket fence: B_g = 0.5 * (a*c*T⁴)/(4π)"""
    T_safe = np.maximum(T, 1e-6)
    return 0.5 * (A_RAD * C_LIGHT * T_safe**4) / (4.0 * np.pi)

# Custom Planck derivatives: dB_g/dT = 2*a*c*T³/(4π)
def dplanck_dT_picket(T):
    """Derivative: dB_g/dT = 2*a*c*T³/(4π)"""
    T_safe = np.maximum(T, 1e-6)
    return 2.0 * (A_RAD * C_LIGHT * T_safe**3) / (4.0 * np.pi)

print(f"\nPlanck functions:")
print(f"  B_g(T) = 0.5 * (a*c*T⁴)/(4π) for both groups")
print(f"  dB_g/dT = 2*a*c*T³/(4π) for both groups")

# Material EOS: e(T) = a*T⁴
def material_energy(T):
    T_safe = np.maximum(T, 1e-6)
    return A_RAD * T_safe**4

def inverse_material_energy(e):
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / A_RAD)**0.25

# Specific heat: C_v = 4*a*T³
def specific_heat(T):
    T_safe = np.maximum(T, 1e-6)
    return 4.0 * A_RAD * T_safe**3

print(f"\nMaterial properties:")
print(f"  e(T) = a*T⁴")
print(f"  C_v(T) = 4*a*T³")

# Diffusion coefficients: D_g = 1/(3*σ_g)
def diffusion_1(T, r):
    return 1.0 / (3.0 * sigma_1)

def diffusion_2(T, r):
    return 1.0 / (3.0 * sigma_2)

# Absorption coefficients
def absorption_1(T, r):
    return sigma_1

def absorption_2(T, r):
    return sigma_2

# Boundary conditions
def left_bc(phi, r):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0

def right_bc(phi, r):
    """Vacuum boundary"""
    return 0.5, 1.0/(3.0*sigma_avg), 0.0

# Source functions: Q_g in physical units
# Normalized source q_g = 0.5, so Q_g = q_g * a * c * T_h^4
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

print(f"\nSources:")
print(f"  Normalized: q₁ = q₂ = {source_normalized} for x < {source_region} cm")
print(f"  Physical: Q₁ = Q₂ = {source_magnitude:.6e} GJ/(cm³·ns) for x < {source_region} cm")

# Emission fractions: χ_g = 0.5 for both groups (equal Planck functions)
emission_fractions = np.array([0.5, 0.5])

print(f"\n{'='*80}")
print("Initializing solver with custom Planck functions")
print(f"{'='*80}")

# Energy edges (arbitrary since we're using custom Planck functions)
energy_edges = np.array([0.0, 50.0, 100.0])

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=x_min,
    r_max=x_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[diffusion_1, diffusion_2],
    absorption_coeff_funcs=[absorption_1, absorption_2],
    left_bc_funcs=[left_bc, left_bc],
    right_bc_funcs=[right_bc, right_bc],
    source_funcs=[source_1, source_2],
    emission_fractions=emission_fractions,
    planck_funcs=planck_picket,  # Single function for both groups
    dplanck_dT_funcs=dplanck_dT_picket,  # Single function for both groups
    material_energy_func=material_energy,
    inverse_material_energy_func=inverse_material_energy,
    cv=specific_heat,
    rho=1.0
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

# Storage for solutions
saved_solutions = {}
next_output_idx = 0

print(f"\n{'='*80}")
print("Time evolution")
print(f"{'='*80}")
print(f"{'Step':<8} {'Time (τ)':<10} {'T_max (keV)':<14} {'E_r_max':<14} {'Newton':<8}")
print("-" * 70)

# Time evolution
for step in range(n_steps):
    verbose = (step < 2) or (step % 1000 == 0) or (step == n_steps - 1)
    
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
    
    if verbose:
        print(f"{step+1:<8} {solver.t/mean_free_time:<10.4f} {solver.T.max():<14.6f} "
              f"{solver.E_r.max():<14.6e} {info['newton_iter']:<8}")
    
    # Save solutions at output times
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
            V = solver.T**4 / T_h**4  # Material energy normalized
            
            saved_solutions[tau_value] = {
                'r': solver.r_centers.copy(),
                'U1': U1,
                'U2': U2,
                'V': V,
                'T': solver.T.copy()
            }
            print(f"  >>> Saved solution at τ = {tau_value:.2f}")
            next_output_idx += 1

print(f"\nCompleted {n_steps} steps")
print(f"Saved {len(saved_solutions)} solutions at output times")

# Plotting
print(f"\n{'='*80}")
print("Creating comparison plots...")
print(f"{'='*80}")

colors = ['blue', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'D']

# Create three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax_U1, ax_U2, ax_V = axes

for idx, tau_val in enumerate(output_times_mft):
    if tau_val not in saved_solutions:
        continue
    
    sol = saved_solutions[tau_val]
    color = colors[idx]
    marker = markers[idx]
    
    # Plot U1 (Group 1 radiation)
    mask = sol['r'] <= 3.5
    ax_U1.plot(sol['r'][mask], sol['U1'][mask], color=color, linestyle='-', 
              linewidth=2.5, alpha=0.8, label=f'τ={tau_val:.1f}')
    
    # Plot reference data for U1
    ref_mask_U1 = ref_data_U1[:, 0] <= 3.5
    tau_idx = list(output_times_mft).index(tau_val)
    ax_U1.plot(ref_data_U1[ref_mask_U1, 0], ref_data_U1[ref_mask_U1, tau_idx+1],
              marker=marker, markerfacecolor=color, markeredgecolor=color,
              markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)
    
    # Plot U2 (Group 2 radiation)
    ax_U2.plot(sol['r'][mask], sol['U2'][mask], color=color, linestyle='-',
              linewidth=2.5, alpha=0.8, label=f'τ={tau_val:.1f}')
    
    # Plot reference data for U2
    ref_mask_U2 = ref_data_U2[:, 0] <= 3.5
    ax_U2.plot(ref_data_U2[ref_mask_U2, 0], ref_data_U2[ref_mask_U2, tau_idx+1],
              marker=marker, markerfacecolor=color, markeredgecolor=color,
              markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)
    
    # Plot V (Material energy)
    ax_V.plot(sol['r'][mask], sol['V'][mask], color=color, linestyle='-',
             linewidth=2.5, alpha=0.8, label=f'τ={tau_val:.1f}')
    
    # Plot reference data for V
    ref_mask_V = ref_data_V[:, 0] <= 3.5
    ax_V.plot(ref_data_V[ref_mask_V, 0], ref_data_V[ref_mask_V, tau_idx+1],
             marker=marker, markerfacecolor=color, markeredgecolor=color,
             markersize=7, markeredgewidth=1.5, linestyle='', alpha=0.6)

# Configure U1 plot
ax_U1.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_U1.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_U1.set_ylabel(r'$U_1$ (Group 1 Radiation)', fontsize=12)
ax_U1.set_title('Group 1: σ₁ = 2/11 (Thin)', fontsize=13)
ax_U1.legend(fontsize=10, loc='best')
ax_U1.grid(True, alpha=0.3)
ax_U1.set_xlim([0, 3.5])

# Configure U2 plot
ax_U2.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_U2.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_U2.set_ylabel(r'$U_2$ (Group 2 Radiation)', fontsize=12)
ax_U2.set_title('Group 2: σ₂ = 20/11 (Thick)', fontsize=13)
ax_U2.legend(fontsize=10, loc='best')
ax_U2.grid(True, alpha=0.3)
ax_U2.set_xlim([0, 3.5])

# Configure V plot
ax_V.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax_V.set_xlabel('Position (mean-free paths)', fontsize=12)
ax_V.set_ylabel(r'$V$ (Material Energy)', fontsize=12)
ax_V.set_title('Material Energy Density', fontsize=13)
ax_V.legend(fontsize=10, loc='best')
ax_V.grid(True, alpha=0.3)
ax_V.set_xlim([0, 3.5])

plt.tight_layout()
show('su_olson_picket_fence.pdf', close_after=True)
print("Saved plot: su_olson_picket_fence.pdf")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Picket Fence Benchmark (Case A): Two-group problem")
print(f"  Group 1 (thin):  σ₁ = {sigma_1:.6f} cm⁻¹")
print(f"  Group 2 (thick): σ₂ = {sigma_2:.6f} cm⁻¹")
print(f"  Equal Planck functions: B_g = 0.5*(a*c*T⁴)/(4π)")
print(f"  Source: Q_g = {source_magnitude} for x < {source_region} cm")
print(f"  Domain: [{x_min}, {x_max}] cm with {n_cells} cells")
print(f"\nResults normalized by a*T_h⁴ with T_h = {T_h} keV")
print(f"Time normalized by τ = c*t")
print(f"\n✓ Comparison with Su & Olson (1997) Table 2")
print(f"✓ Lines show numerical solution, markers show reference data")
print("="*80)
