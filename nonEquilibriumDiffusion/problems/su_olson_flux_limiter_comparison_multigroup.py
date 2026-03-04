#!/usr/bin/env python3
"""
Su-Olson test problem comparing different flux limiters (MULTIGROUP VERSION)

This script runs the Su-Olson problem with multiple flux limiters and compares results:
- Standard (λ=1/3) - no limiting
- Levermore-Pomraning
- Larsen
- Sum
- Max

Problem setup:
- 1-D slab geometry, x from 0 to 20 cm
- Radiation source of magnitude 1/2 for 0 < x < 0.5 cm
- σ_P = σ_R = 1.0 cm^-1 for each group
- Material energy: e(T) = a·T^4 (radiation-dominated)
- Source duration: 10 mean free times = 10/(c·σ_P)
- Output at: 0.1, 1.0, 3.16228, 10.0, 31.6228, and 100.0 mean free times
- Reflecting BC at x=0, vacuum BC at x=20
- 3 groups with equal emission fractions (gray approximation)
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
    flux_limiter_standard,
    flux_limiter_levermore_pomraning,
    flux_limiter_larsen,
    flux_limiter_sum,
    flux_limiter_max
)

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from utils.plotfuncs import show, hide_spines, font

# Physical constants
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

# Material energy density reference solution: su_olson_material_energy[i, j] is value at x[i], tau[j]
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

# Transport solution for radiation energy density (normalized): transport_rad_energy[i, j] is value at x[i], tau[j]
transport_rad_energy = np.array([
    [0.09531, 0.27526, 0.64308, 1.20052, 2.23575, 0.69020, 0.35720],  # x=0.01
    [0.09531, 0.27526, 0.63585, 1.18869, 2.21944, 0.68974, 0.35714],  # x=0.10
    [0.09532, 0.27527, 0.61958, 1.16190, 2.18344, 0.68878, 0.35702],  # x=0.17783
    [0.09529, 0.26262, 0.56187, 1.07175, 2.06448, 0.68569, 0.35664],  # x=0.31623
    [0.08823, 0.20312, 0.44711, 0.90951, 1.86072, 0.68111, 0.35599],  # x=0.45
    [0.04765, 0.13762, 0.35801, 0.79902, 1.73178, 0.67908, 0.35574],  # x=0.50
    [0.00375, 0.06277, 0.25374, 0.66678, 1.57496, 0.67619, 0.35538],  # x=0.56234
    [np.nan, 0.00280, 0.11430, 0.44675, 1.27398, 0.66548, 0.35393],   # x=0.75
    [np.nan, np.nan, 0.03648, 0.27540, 0.98782, 0.64691, 0.35141],    # x=1.0
    [np.nan, np.nan, 0.00291, 0.14531, 0.70822, 0.61538, 0.34697],    # x=1.33352
    [np.nan, np.nan, np.nan, 0.05968, 0.45016, 0.56353, 0.33924],     # x=1.77828
    [np.nan, np.nan, np.nan, 0.00123, 0.09673, 0.36965, 0.30346],     # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00375, 0.10830, 0.21382],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00390, 0.07200],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00272],        # x=17.78279
])

# Transport solution for material energy density (normalized): transport_mat_energy[i, j] is value at x[i], tau[j]
transport_mat_energy = np.array([
    [0.00468, 0.04093, 0.27126, 0.94670, 2.11186, 0.70499, 0.35914],  # x=0.01
    [0.00468, 0.04093, 0.26839, 0.93712, 2.09585, 0.70452, 0.35908],  # x=0.10
    [0.00468, 0.04093, 0.26261, 0.91525, 2.06052, 0.70348, 0.35895],  # x=0.17783
    [0.00468, 0.04032, 0.23978, 0.84082, 1.94365, 0.70020, 0.35854],  # x=0.31623
    [0.00455, 0.03314, 0.18826, 0.70286, 1.74291, 0.69532, 0.35793],  # x=0.45
    [0.00234, 0.02046, 0.14187, 0.60492, 1.61536, 0.69308, 0.35766],  # x=0.50
    [0.00005, 0.00635, 0.08838, 0.48843, 1.46027, 0.68994, 0.35728],  # x=0.56234
    [np.nan, 0.00005, 0.03014, 0.30656, 1.16591, 0.67850, 0.35581],   # x=0.75
    [np.nan, np.nan, 0.00625, 0.17519, 0.88992, 0.65868, 0.35326],    # x=1.0
    [np.nan, np.nan, 0.00017, 0.08352, 0.62521, 0.62507, 0.34875],    # x=1.33352
    [np.nan, np.nan, np.nan, 0.02935, 0.38688, 0.57003, 0.34086],     # x=1.77828
    [np.nan, np.nan, np.nan, 0.00025, 0.07642, 0.36727, 0.30517],     # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00253, 0.10312, 0.21377],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.00342, 0.07122],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.00261],        # x=17.78279
])


# Problem parameters
sigma_P = 1.0      # cm^-1 (Planck opacity for EACH group)
sigma_R = 1.0      # cm^-1 (Rosseland opacity for EACH group)
x_min = 0.0        # cm (reflecting boundary)
x_max = 20.0       # cm
n_cells = 400      # Number of cells
source_region = 0.5  # cm (source from 0 to 0.5)

# Source magnitude: For 3 groups with σ=1 each, divide source among groups
T_0 = 1.0  # keV (reference temperature for normalization)
n_groups = 3
source_magnitude_total = A_RAD * C_LIGHT * T_0**4
source_magnitude_per_group = source_magnitude_total / n_groups

# Time parameters
mean_free_time = 1.0 / (C_LIGHT * sigma_P)  # τ = 1/(c·σ) in ns
print(f"Mean free time τ = {mean_free_time:.6f} ns")

source_duration = 10.0 * mean_free_time  # Source on for 10 τ

# Define output times for plots
early_output_times_mft = [0.1, 1.0] #, 3.16228, 10.0]  # Early times in mean free times
late_output_times_mft = []   # Late times in mean free times  (disabled for speed)
all_output_times_mft = early_output_times_mft + late_output_times_mft

print(f"Source duration: {source_duration:.6f} ns = 10τ")
print(f"Early output times: {early_output_times_mft} τ")
print(f"Late output times: {late_output_times_mft} τ")

# Time stepping - run to last output time
dt = 0.01 * mean_free_time  # Larger timestep for speed
final_time = max(all_output_times_mft) * mean_free_time if all_output_times_mft else mean_free_time
n_steps = int(np.ceil(final_time / dt))

print(f"Timestep: {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.3f}τ")
print(f"Total steps: {n_steps}")
print(f"Number of groups: {n_groups}")

# Material properties for radiation-dominated material
def material_energy_func(T):
    """Material energy density: e = a·T^4"""
    T_safe = np.maximum(T, 1e-6)
    return A_RAD * T_safe**4

def inverse_material_energy_func(e):
    """Temperature from energy: T = (e/a)^(1/4)"""
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / A_RAD)**0.25

def specific_heat_func(T):
    """Temperature-dependent specific heat: c_v = 4a·T^3 / ρ"""
    T_safe = np.maximum(T, 1e-6)
    return 4.0 * A_RAD * T_safe**3 / RHO

# Diffusion coefficient: D = 1/(3σ_R) (for no flux limiting)
def diffusion_coeff_standard(T, r):
    return 1.0 / (3.0 * sigma_R)

# Rosseland opacity (for flux limiters that need it)
def rosseland_opacity(T, r):
    return sigma_R

# Absorption coefficient
def absorption_coeff(T, r):
    return sigma_P

# Boundary conditions
def left_bc(phi, r):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def right_bc(phi, r):
    """Vacuum boundary: incoming current is zero"""
    return 0.5, 1.0/(3.0*sigma_R), 0.0  # A=1/2, B=1/(3σ_R), C=0

# Source function Q_g(r, t) - active in [0, 0.5] cm for t < 10τ
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

# Define flux limiters to test
flux_limiters = {
    'Levermore-Pomraning': flux_limiter_levermore_pomraning,
    'Larsen n=2': flux_limiter_larsen,
    'Sum': flux_limiter_sum,
    'Max': flux_limiter_max
}

# Test that flux limiters are actually different functions
print("\nVerifying flux limiters are distinct:")
R_test = np.array([0.1, 1.0, 10.0])
for name, func in flux_limiters.items():
    lambda_vals = np.array([func(R) for R in R_test])
    print(f"  {name:25s}: λ(R={R_test[0]})={lambda_vals[0]:.4f}, λ(R={R_test[1]})={lambda_vals[1]:.4f}, λ(R={R_test[2]})={lambda_vals[2]:.4f}")

# Wrap flux limiters to track R values
flux_limiters_wrapped = {}
R_values_tracker = {}  # Track R values for each limiter

for name, func in flux_limiters.items():
    R_values_tracker[name] = []
    
    def make_wrapper(original_func, tracker_list, limiter_name=name):
        def wrapper(R):
            if len(tracker_list) < 10:  # Only track first 10 to avoid memory issues
                tracker_list.append(R)
                if len(tracker_list) == 1:
                    print(f"  [{limiter_name}] First R value: {R:.6f}")
            return original_func(R)
        return wrapper
    
    flux_limiters_wrapped[name] = make_wrapper(func, R_values_tracker[name])

# Storage for results
results = {}

print(f"\n{'='*80}")
print("Running Su-Olson problem with different flux limiters (MULTIGROUP)")
print(f"{'='*80}")

# Energy edges (arbitrary for gray approximation)
energy_edges = np.linspace(0.0, 100.0, n_groups + 1)

# Run with each flux limiter
for limiter_name, limiter_func in flux_limiters_wrapped.items():
    print(f"\n{'-'*80}")
    print(f"Running with {limiter_name} flux limiter...")
    print(f"{'-'*80}")
    
    # Create solver
    # When using flux limiters, don't pass diffusion_coeff_funcs - let the limiter wrapping create them
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=x_min,
        r_max=x_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=None,  # Let flux limiter wrapping create D = λ/σ_R
        absorption_coeff_funcs=[absorption_coeff] * n_groups,
        left_bc_funcs=[left_bc] * n_groups,
        right_bc_funcs=[right_bc] * n_groups,
        source_funcs=source_functions,
        emission_fractions=np.ones(n_groups) / n_groups,  # Equal emission (gray)
        material_energy_func=material_energy_func,
        inverse_material_energy_func=inverse_material_energy_func,
        cv=specific_heat_func,  # Temperature-dependent c_v
        flux_limiter_funcs=[limiter_func] * n_groups,  # Apply same limiter to all groups
        rosseland_opacity_funcs=[rosseland_opacity] * n_groups
    )
    
    # Initial conditions
    T_init = 0.001  # keV
    solver.T = np.ones(n_cells) * T_init
    solver.T_old = solver.T.copy()
    solver.E_r = np.ones(n_cells) * A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    # Time evolution with saving at output times
    saved_solutions = {}  # Dictionary to store solutions at specific times
    next_output_idx = 0
    
    # Track phi statistics
    phi_max_history = []
    
    for step in range(n_steps):
        current_time = solver.t
        
        # Take Newton step
        info = solver.step(
            max_newton_iter=10,
            newton_tol=1e-6,
            gmres_tol=1e-6,
            gmres_maxiter=200,
            verbose=False
        )
        
        # Advance time
        solver.advance_time()
        
        # Track phi statistics
        phi_max_history.append(np.max(solver.phi_g_stored))
        
        # Print progress
        verbose = (step < 2) or (step % 10 == 0) or (step == n_steps - 1)
        if verbose:
            print(f"  Step {step+1}/{n_steps}: t = {solver.t/mean_free_time:.3f}τ, "
                  f"max(phi_stored) = {phi_max_history[-1]:.3e}, max(E_r) = {solver.E_r.max():.3e}")
        
        # Check if we should save this timestep (when we cross an output time)
        if next_output_idx < len(all_output_times_mft):
            target_time = all_output_times_mft[next_output_idx] * mean_free_time
            if solver.t >= target_time:
                tau_value = all_output_times_mft[next_output_idx]
                
                # Compute normalized energy densities
                # E_r normalized = E_r / (a·T_0^4)
                # φ/ac normalized = E_r/a = (E_r / a) / (c·T_0^4) = E_r / (a·c·T_0^4)
                # But Su-Olson uses φ/(a·c·T_0^4) = E_r / (a·T_0^4)
                E_r_normalized = solver.E_r / (A_RAD * T_0**4)
                E_mat_normalized = solver.T**4 / T_0**4
                
                saved_solutions[tau_value] = {
                    'r': solver.r_centers.copy(),
                    'phi': solver.E_r.copy() * C_LIGHT,  # Convert E_r to φ
                    'T': solver.T.copy(),
                    'E_rad': E_r_normalized,
                    'E_mat': E_mat_normalized
                }
                print(f"    Saved solution at τ = {tau_value:.4f}")
                next_output_idx += 1
    
    # Store results for this limiter
    results[limiter_name] = saved_solutions
    
    print(f"  Completed {limiter_name}: saved {len(saved_solutions)} time snapshots")

# Print R value statistics
print(f"\n{'='*80}")
print("Flux limiter R value statistics:")
print(f"{'='*80}")
for name in R_values_tracker.keys():
    R_vals = np.array(R_values_tracker[name])
    if len(R_vals) > 0:
        print(f"  {name:25s}: min={np.min(R_vals):.6f}, max={np.max(R_vals):.6f}, "
              f"mean={np.mean(R_vals):.6f}, median={np.median(R_vals):.6f}")
        print(f"  {'':25s}  # calls: {len(R_vals)}")
    else:
        print(f"  {name:25s}: No R values recorded!")

# Check that solutions are actually different
print(f"\n{'='*80}")
print("Verifying solutions are different between flux limiters:")
print(f"{'='*80}")
limiter_names = list(results.keys())
for i, name1 in enumerate(limiter_names):
    for name2 in limiter_names[i+1:]:
        for tau_val in results[name1].keys():
            if tau_val in results[name2]:
                sol1 = results[name1][tau_val]
                sol2 = results[name2][tau_val]
                max_diff_E_rad = np.max(np.abs(sol1['E_rad'] - sol2['E_rad']))
                max_diff_T = np.max(np.abs(sol1['T'] - sol2['T']))
                print(f"  τ={tau_val:<6.2f} {name1:25s} vs {name2:20s}: "
                      f"ΔE_rad={max_diff_E_rad:.6e}, ΔT={max_diff_T:.6e}")

# Plotting comparisons
print(f"\n{'='*80}")
print("Creating comparison plots...")
print(f"{'='*80}")

# Define colors for different times
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Use consistent markers: circles for diffusion, squares for transport
marker_diffusion = 'o'
marker_transport = 's'

# Define colors and line styles for different flux limiters
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
    'Max': (0, (3, 1, 1, 1))  # densely dashdotted
}

# ============================================================================
# EARLY TIMES PLOT (0.1, 1.0, 3.16228, 10.0 τ)
# Radiation energy, x=0 to 10 cm
# ============================================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5.25))

# Plot all flux limiters
for limiter_name, limiter_results in results.items():
    limiter_color = flux_limiter_colors[limiter_name]
    limiter_linestyle = flux_limiter_linestyles[limiter_name]
    
    for idx, tau_val in enumerate(early_output_times_mft):
        if tau_val not in limiter_results:
            continue
        
        sol = limiter_results[tau_val]
        time_color = colors[idx]
        
        # Plot numerical radiation energy with limiter-specific style
        mask = sol['r'] <= 10.0
        ax1.plot(sol['r'][mask], sol['E_rad'][mask], color=limiter_color, 
                linestyle=limiter_linestyle, linewidth=2.0, alpha=0.7)

# Plot reference data (transport solution) with consistent markers
for idx, tau_val in enumerate(early_output_times_mft):
    time_color = colors[idx]
    time_index = np.argmin(np.abs(su_olson_tau - tau_val))
    
    if abs(su_olson_tau[time_index] - tau_val) < 0.01:
        ref_mask = su_olson_x <= 10.0
        
        # Transport: Radiation energy reference (filled squares)
        ax1.plot(su_olson_x[ref_mask], transport_rad_energy[ref_mask, time_index], 
                marker=marker_transport, markerfacecolor=time_color, markeredgecolor='black', 
                markersize=6, markeredgewidth=1.5, linestyle='', alpha=0.8, zorder=10)

ax1.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel('Position (mean-free path)', fontsize=14)
ax1.set_ylabel(r'Radiation Energy Density $\frac{\phi}{a c T_0^4}$', fontsize=14)
ax1.set_yscale('log')
ax1.set_xscale('log')

# Create custom legend with flux limiters and reference symbols
legend_elements = []
# Add flux limiters
for limiter_name in ['Levermore-Pomraning', 'Larsen n=2', 'Sum', 'Max']:
    legend_elements.append(
        Line2D([0], [0], color=flux_limiter_colors[limiter_name], 
               linestyle=flux_limiter_linestyles[limiter_name], 
               linewidth=2.0, label=limiter_name)
    )
# Add reference data markers
legend_elements.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black',
           markersize=6, linestyle='', label='Transport ref')
)
ax1.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)

ax1.grid(True, alpha=0.3)
ax1.set_xlim([.2, 8])
ax1.set_ylim([1e-3, 3e0])

plt.tight_layout()
show('su_olson_flux_limiter_multigroup_early_times.pdf', close_after=True)
print("Saved plot to 'su_olson_flux_limiter_multigroup_early_times.pdf'")

# ============================================================================
# LATE TIMES PLOT (31.6228, 100.0 τ)
# Radiation energy, full domain
# ============================================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5.25))

# Plot all flux limiters
for limiter_name, limiter_results in results.items():
    limiter_color = flux_limiter_colors[limiter_name]
    limiter_linestyle = flux_limiter_linestyles[limiter_name]
    
    for idx, tau_val in enumerate(late_output_times_mft):
        if tau_val not in limiter_results:
            continue
        
        sol = limiter_results[tau_val]
        time_color = colors[idx + 3]  # Use different colors
        
        # Plot numerical radiation energy with limiter-specific style (full domain)
        ax2.plot(sol['r'], sol['E_rad'], color=limiter_color, 
                linestyle=limiter_linestyle, linewidth=2.0, alpha=0.7)

# Plot reference data (transport solution) with consistent markers
for idx, tau_val in enumerate(late_output_times_mft):
    time_color = colors[idx + 3]
    time_index = np.argmin(np.abs(su_olson_tau - tau_val))
    
    if abs(su_olson_tau[time_index] - tau_val) < 0.1:
        # Transport: Radiation energy reference (filled squares)
        ax2.plot(su_olson_x, transport_rad_energy[:, time_index], 
                marker=marker_transport, markerfacecolor=time_color, markeredgecolor='black', 
                markersize=6, markeredgewidth=1.5, linestyle='', alpha=1.0, zorder=10)

ax2.axvline(x=source_region, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Position (mean-free path)', fontsize=14)
ax2.set_ylabel(r'Radiation Energy Density $\frac{\phi}{a c T_0^4}$', fontsize=14)
ax2.set_yscale('log')
ax2.set_xscale('log')

# Create custom legend with flux limiters and reference symbols
legend_elements = []
# Add flux limiters
for limiter_name in ['Levermore-Pomraning', 'Larsen n=2', 'Sum', 'Max']:
    legend_elements.append(
        Line2D([0], [0], color=flux_limiter_colors[limiter_name], 
               linestyle=flux_limiter_linestyles[limiter_name], 
               linewidth=2.0, label=limiter_name)
    )
# Add reference data markers
legend_elements.append(
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black',
           markersize=6, linestyle='', label='Transport ref')
)
ax2.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)
ax2.set_xlim(left=0.2, right=2e1)
ax2.set_ylim(bottom=1e-3, top=1e0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show('su_olson_flux_limiter_multigroup_late_times.pdf', close_after=True)
print("Saved plot to 'su_olson_flux_limiter_multigroup_late_times.pdf'")

print(f"\n{'='*80}")
print("Comparison complete!")
print(f"{'='*80}")
