#!/usr/bin/env python3
"""
Su-Olson Picket Fence Benchmark (Case A) - Multigroup IMC Version

Two-group problem with:
- Group 1: σ₁ = 2/11 cm⁻¹ (optically thin group)
- Group 2: σ₂ = 20/11 cm⁻¹ (optically thick group)
- Equal Planck functions: B_g = 0.5 * (a*c*T⁴)/(4π) for both groups
  (This means b_g = 0.5 emission fraction for each group)
- Source: Q_g = 0.5 in each group for x < 0.5 cm
- Material: e(T) = a*T⁴, C_v = 4*a*T³
- Reflecting BC at x=0, vacuum BC at x=20

This implements the same benchmark as in nonEquilibriumDiffusion/problems/su_olson_picket_fence.py
but using the multigroup IMC method instead of diffusion.

Reference data from Table 2 of Su & Olson (1997).
Note: Table 2 has different values than Table 4 (which contains transport solutions)
"""

import numpy as np
import sys

# Import multigroup IMC
import sys
import os

# Add parent directory to path to import MG_IMC package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from MG_IMC import (
    run_simulation,
    C_LIGHT as __c,
    A_RAD as __a,
)

# Add utils to path for plotting
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from plotfuncs import show
# Try to import plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, skipping plots")
    PLOTTING_AVAILABLE = False

# Reference solution data from Su & Olson (1997) Table 4
# Format: x, tau=0.1, tau=0.3, tau=1.0, tau=3.0
ref_data_U1 = np.array([
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

ref_data_U2 = np.array([
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

ref_data_V = np.array([
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
print("SU-OLSON PICKET FENCE BENCHMARK (Case A) - Multigroup IMC")
print("="*80)

# Problem parameters
n_groups = 2
x_min = 0.0
x_max = 3.6  # Extend beyond 2.0 to see full behavior
nx = 130
ny = 1  # 1D problem in 2D code

# Opacities for picket fence (cm⁻¹)
sigma_1 = 2.0 / 11.0   # Group 1: optically thin
sigma_2 = 20.0 / 11.0  # Group 2: optically thick

print(f"\nProblem setup:")
print(f"  Groups: {n_groups}")
print(f"  σ₁ = {sigma_1:.6f} cm⁻¹ (optically thin)")
print(f"  σ₂ = {sigma_2:.6f} cm⁻¹ (optically thick)")
print(f"  Domain: [{x_min}, {x_max}] cm with {nx} cells")
print(f"  Source region: x < 0.5 cm")

# Create spatial mesh
x_edges = np.linspace(x_min, x_max, nx + 1)
y_edges = np.array([0.0, 1.0])  # Dummy for 1D

# Energy group edges (arbitrary since we'll use custom emission)
# Group 0: low energy (thin), Group 1: high energy (thick)
energy_edges = np.array([0.1, 1.0, 10.0])  # keV

# Reference temperature for normalization
T_h = 1.0  # keV

# Material properties matching Su-Olson benchmark
# Material: e(T) = a*T⁴
def material_energy(T):
    T_safe = np.maximum(T, 1e-6)
    return __a * T_safe**4

def inverse_material_energy(e):
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / __a)**0.25

# Specific heat: C_v = 4*a*T³
def specific_heat(T):
    T_safe = np.maximum(T, 1e-6)
    return 4.0 * __a * T_safe**3

print(f"\nMaterial properties:")
print(f"  e(T) = a*T⁴")
print(f"  C_v(T) = 4*a*T³")
print(f"  Reference temperature: T_h = {T_h} keV")

# Opacity functions - constant opacities independent of temperature
def sigma_a_group_0(T):
    """Group 1 (thin): σ₁ = 2/11 cm⁻¹"""
    return sigma_1 * np.ones_like(T)

def sigma_a_group_1(T):
    """Group 2 (thick): σ₂ = 20/11 cm⁻¹"""
    return sigma_2 * np.ones_like(T)

sigma_a_funcs = [sigma_a_group_0, sigma_a_group_1]

print(f"\nOpacities:")
print(f"  Group 0 (thin):  σ₁ = {sigma_1:.6f} cm⁻¹ (constant)")
print(f"  Group 1 (thick): σ₂ = {sigma_2:.6f} cm⁻¹ (constant)")

# Source configuration
# Q_g = 0.5 * a * c * T_h^4 for x < 0.5 cm
source_region = 0.5  # cm
source_normalized = 0.5
source_magnitude = source_normalized * __a * __c * T_h**4

# Create source array: (n_groups, nx, ny)
source = np.zeros((n_groups, nx, ny))
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
mask = x_centers < source_region

for g in range(n_groups):
    source[g, mask, 0] = source_magnitude

print(f"\nSources:")
print(f"  Normalized: q₁ = q₂ = {source_normalized}")
print(f"  Physical: Q₁ = Q₂ = {source_magnitude:.6e} GJ/(cm³·ns)")
print(f"  Source region: x < {source_region} cm")
print(f"  Number of source cells: {np.sum(mask)}")

# Initial conditions - cold material
T_init = 0.001  # keV (matching diffusion version)
Tinit = np.full((nx, ny), T_init)
Tr_init = np.full((nx, ny), T_init)

print(f"\nInitial conditions:")
print(f"  T_init = {T_init} keV")

# Boundary conditions
# Left: reflecting (x=0)
# Right: vacuum (x=x_max)
# Top/Bottom: reflecting (for 1D slab)
T_boundary = (0.0, 0.0, 0.0, 0.0)  # No boundary emission
reflect = (True, False, True, True)  # Reflect left, top, bottom; vacuum right

print(f"\nBoundary conditions:")
print(f"  Left (x=0):  Reflecting")
print(f"  Right (x={x_max}): Vacuum")
print(f"  Top/Bottom: Reflecting (1D)")

# Time parameters
sigma_avg = (sigma_1 + sigma_2) / 2.0
mean_free_time = 1.0 / (__c * sigma_avg)

print(f"\nTime scale:")
print(f"  σ_avg = {sigma_avg:.6f} cm⁻¹")
print(f"  Mean free time τ = 1/(c·σ_avg) = {mean_free_time:.6e} ns")

# Particle counts - need substantial particles for accuracy
Ntarget = 200000     # Material emission particles per step
Nboundary = 0        # No boundary emission
Nsource = 200000      # Source particles per step
Nmax = 400000        # Maximum census particles

print(f"\nParticle counts:")
print(f"  Ntarget (material emission): {Ntarget}")
print(f"  Nsource (fixed source): {Nsource}")
print(f"  Nmax (census limit): {Nmax}")

# Time stepping
dt = 0.005 * mean_free_time
output_times_tau = [0.1, 0.3, 1.0, 3.0]
output_times = [tau * mean_free_time for tau in output_times_tau]
final_time = max(output_times)

print(f"\nTime stepping:")
print(f"  dt = {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"  Final time: {final_time:.6e} ns = {final_time/mean_free_time:.1f}τ")
print(f"  Output times: {output_times_tau} τ")

# Output frequency
n_steps = int(np.ceil(final_time / dt))
output_freq = max(1, n_steps // 100)

print(f"  Total steps: {n_steps}")
print(f"  Output frequency: every {output_freq} steps")

print(f"\n{'='*80}")
print("Running Multigroup IMC Simulation")
print(f"{'='*80}\n")

# Emission fractions for picket fence: equal for both groups (not Planck integrals!)
emission_fractions = np.array([0.5, 0.5])

print(f"\nEmission configuration:")
print(f"  Using equal emission fractions: b_1 = b_2 = 0.5")
print(f"  (Override Planck integrals for picket fence problem)\n")

# Run simulation
history, final_state = run_simulation(
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nsource=Nsource,
    Nmax=Nmax,
    Tinit=Tinit,
    Tr_init=Tr_init,
    T_boundary=T_boundary,
    dt=dt,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=material_energy,
    inv_eos=inverse_material_energy,
    cv=specific_heat,
    source=source,
    final_time=final_time,
    reflect=reflect,
    output_freq=output_freq,
    theta=1.0,
    use_scalar_intensity_Tr=False,  # Use particle binning for better accuracy
    Ntarget_ic=Ntarget,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
    emission_fractions=emission_fractions,  # Equal emission (picket fence)
)

print(f"\n{'='*80}")
print("Simulation Complete")
print(f"{'='*80}")

# Extract solutions at output times
print(f"\nExtracting solutions at output times...")

saved_solutions = {}
for i, (tau_val, target_time) in enumerate(zip(output_times_tau, output_times)):
    # Find closest history entry
    closest_idx = None
    min_diff = float('inf')
    
    for j, info in enumerate(history):
        time_diff = abs(info['time'] - target_time)
        if time_diff < min_diff:
            min_diff = time_diff
            closest_idx = j
    
    if closest_idx is not None:
        info = history[closest_idx]
        
        # Extract radiation energy by group
        E_rad_by_group = info['radiation_energy_by_group']  # (n_groups, nx, ny)
        
        # Normalize by a*T_h^4
        norm = __a * T_h**4
        U1 = E_rad_by_group[0, :, 0] / norm
        U2 = E_rad_by_group[1, :, 0] / norm
        
        # Material energy: V = T^4 / T_h^4
        T = info['temperature']
        V = (T[:, 0]**4) / (T_h**4)
        
        saved_solutions[tau_val] = {
            'x': x_centers.copy(),
            'U1': U1,
            'U2': U2,
            'V': V,
            'T': T[:, 0].copy(),
            'time': info['time'],
        }
        
        actual_tau = info['time'] / mean_free_time
        print(f"  τ = {tau_val:.1f} (target), τ = {actual_tau:.3f} (actual), "
              f"time = {info['time']:.6e} ns, N_particles = {info['N_particles']}")
        
        # Debug: print actual values and energy distribution
        tau_idx = [0.1, 0.3, 1.0, 3.0].index(tau_val)
        ref_u1_x0 = ref_data_U1[0, tau_idx + 1]
        ref_u2_x0 = ref_data_U2[0, tau_idx + 1]
        print(f"    U1 at x=0: {U1[0]:.6f} (ref: {ref_u1_x0:.5f})")
        print(f"    U2 at x=0: {U2[0]:.6f} (ref: {ref_u2_x0:.5f})")
        print(f"    E_rad_1 at x=0: {E_rad_by_group[0, 0, 0]:.6e} GJ/cm³")
        print(f"    E_rad_2 at x=0: {E_rad_by_group[1, 0, 0]:.6e} GJ/cm³")
        
        # Check total energy in each group
        total_E1 = np.sum(E_rad_by_group[0, :, :])
        total_E2 = np.sum(E_rad_by_group[1, :, :])
        print(f"    Total E_rad_1 (all cells): {total_E1:.6e} GJ")
        print(f"    Total E_rad_2 (all cells): {total_E2:.6e} GJ")
        print(f"    Ratio E2/E1: {total_E2/(total_E1+1e-30):.3f}")
        
        # Check transport statistics
        profiling = info['profiling']
        events = profiling['transport_events']
        print(f"\n    Transport events:")
        print(f"      Total events: {events['total']}")
        print(f"      Boundary crossings: {events['boundary_crossings']}")
        print(f"      Weight floor kills: {events['weight_floor_kills']}")
        print(f"      Census events: {events['census_events']}")

# Plotting
if PLOTTING_AVAILABLE and len(saved_solutions) > 0:
    print(f"\n{'='*80}")
    print("Creating comparison plots...")
    print(f"{'='*80}")
    
    colors_time = ['blue', 'green', 'red', 'purple']

    plot_specs = [
        ('U1', ref_data_U1, r'$U_1$ (Group 1 Radiation)', 'Group 1: σ₁ = 2/11 (Thin)', (1e-3, 1e0), 'test_su_olson_picket_fence_imc_U1.pdf'),
        ('U2', ref_data_U2, r'$U_2$ (Group 2 Radiation)', 'Group 2: σ₂ = 20/11 (Thick)', (1e-4, 1e0), 'test_su_olson_picket_fence_imc_U2.pdf'),
        ('V', ref_data_V, r'$V$ (Material Energy)', 'Material Energy Density', (1e-5, 1e0), 'test_su_olson_picket_fence_imc_V.pdf'),
    ]

    for quantity_key, ref_data, y_label, title, y_limits, output_file in plot_specs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        for idx, tau_val in enumerate(output_times_tau):
            if tau_val not in saved_solutions:
                continue

            sol = saved_solutions[tau_val]
            color = colors_time[idx]

            y = sol[quantity_key]
            mask = (sol['x'] >= 0.05) & (sol['x'] <= 5.0) & (y > 0.0)
            ax.plot(sol['x'][mask], y[mask], color=color, linestyle='-', linewidth=2.5, alpha=0.8,
                    label=f'IMC τ={tau_val:.1f}')

            if tau_val in [0.1, 0.3, 1.0, 3.0]:
                tau_idx = [0.1, 0.3, 1.0, 3.0].index(tau_val)
                ref_mask = (~np.isnan(ref_data[:, tau_idx + 1])) & (ref_data[:, 0] >= 0.05) & (ref_data[:, tau_idx + 1] > 0.0)
                ax.plot(ref_data[ref_mask, 0], ref_data[ref_mask, tau_idx + 1],
                        marker='s', markerfacecolor=color, markeredgecolor='black',
                        markersize=6, markeredgewidth=1.5, linestyle='', alpha=0.8,
                        label=f'Ref τ={tau_val:.1f}')

        ax.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel('Position (mean-free paths)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        #ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0.05, 5.0])
        ax.set_ylim(y_limits)

        fig.tight_layout()
        show(output_file, close_after=True)
        

# Print summary statistics
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Su-Olson Picket Fence Benchmark (Case A) - Multigroup IMC")
print(f"\nProblem setup:")
print(f"  Group 1 (thin):  σ₁ = {sigma_1:.6f} cm⁻¹")
print(f"  Group 2 (thick): σ₂ = {sigma_2:.6f} cm⁻¹")
print(f"  Material: e(T) = a*T⁴,  C_v = 4*a*T³")
print(f"  Source: Q_g = 0.5*a*c*T_h⁴ for x < {source_region} cm, both groups")
print(f"  Domain: [{x_min}, {x_max}] cm with {nx} cells")
print(f"\nParticle statistics:")
print(f"  Material emission: {Ntarget} particles/step")
print(f"  Source: {Nsource} particles/step")
print(f"  Final census: {len(final_state.weights)} particles")

if len(saved_solutions) > 0:
    print(f"\nResults at output times:")
    for tau_val in output_times_tau:
        if tau_val in saved_solutions:
            sol = saved_solutions[tau_val]
            T_max = np.max(sol['T'])
            U1_max = np.max(sol['U1'])
            U2_max = np.max(sol['U2'])
            V_max = np.max(sol['V'])
            print(f"  τ = {tau_val:.1f}:  T_max = {T_max:.4f} keV,  "
                  f"U1_max = {U1_max:.5f},  U2_max = {U2_max:.5f},  V_max = {V_max:.5f}")

print(f"\n✓ Reference data from Su & Olson (1997) Table 2")
print(f"✓ Results normalized by a*T_h⁴ with T_h = {T_h} keV")
print(f"✓ Time normalized by mean free time τ = 1/(c*σ_avg)")

if PLOTTING_AVAILABLE:
    print(f"✓ Comparison plot shows IMC (solid lines) vs Reference (hollow markers)")
else:
    print("  (Plotting skipped - matplotlib not available)")

print("="*80)
