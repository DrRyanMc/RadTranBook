"""
Bennett test problem comparing different flux limiters

This script runs the Bennett problem with multiple flux limiters and compares results:
- Levermore-Pomraning
- Larsen
- Sum
- Max

Problem setup:
- 1-D slab geometry, x from 0 to 20 cm
- Radiation source of magnitude 1/2 for 0 < x < 0.5 cm
- σ_P = σ_R = 1.0 cm^-1
- Material energy: e(T) = Cv·ρ·T with CONSTANT Cv = 0.03 GJ/(g·keV)
- Source duration: 10 mean free times = 10/(c·σ_P)
- Output at: 0.1, 1.0, 3.16228, 10.0, 31.6228, and 100.0 mean free times
- Reflecting BC at x=0, vacuum BC at x=20
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from oneDFV import (NonEquilibriumRadiationDiffusionSolver,
                    flux_limiter_standard,
                    flux_limiter_levermore_pomraning,
                    flux_limiter_larsen,
                    flux_limiter_sum,
                    flux_limiter_max, 
                    flux_limiter_standard)

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.plotfuncs import show, hide_spines, font

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³
CV_CONST = 0.03    # GJ/(g·keV) - CONSTANT specific heat for Bennett problem

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
# Bennett problem transport solution
transport_rad_energy = np.array([
    [0.095162, 0.271108, 0.563683, 0.765084, 1.96832, 0.267247, 0.085108],  # x=0.01
    [0.095162, 0.271108, 0.557609, 0.756116, 1.950367, 0.266877, 0.085054],  # x=0.10
    [0.095162, 0.271108, 0.543861, 0.736106, 1.910675, 0.266071, 0.084937],  # x=0.17783
    [0.095162, 0.258592, 0.495115, 0.668231, 1.779896, 0.263527, 0.084565],  # x=0.31623
    [0.08809, 0.199962, 0.396442, 0.543721, 1.558248, 0.259729, 0.084008],   # x=0.45
    [0.047581, 0.135554, 0.316071, 0.453151, 1.420865, 0.257976, 0.08375],   # x=0.50
    [0.00376, 0.061935, 0.222261, 0.349209, 1.252213, 0.255538, 0.083392],   # x=0.56234
    [np.nan, 0.002788, 0.102348, 0.21078, 0.908755, 0.246543, 0.082061],     # x=0.75
    [np.nan, np.nan, 0.034228, 0.124305, 0.562958, 0.230831, 0.079715],      # x=1.0
    [np.nan, np.nan, 0.002864, 0.067319, 0.27752, 0.203718, 0.075591],       # x=1.33352
    [np.nan, np.nan, np.nan, 0.031357, 0.120054, 0.158039, 0.068419],        # x=1.77828
    [np.nan, np.nan, np.nan, 0.001057, 0.013737, 0.022075, 0.036021],        # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.000413, 0.000814, 0.001068],          # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 5e-06, 5e-06],                  # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],                # x=17.78279
])

# Transport solution for material energy density (normalized): transport_mat_energy[i, j] is value at x[i], tau[j]
# Bennett problem transport solution
transport_mat_energy = np.array([
    [0.004837, 0.045121, 0.354022, 1.613529, 2.57461, 1.592549, 1.190296],   # x=0.01
    [0.004837, 0.045121, 0.350958, 1.601467, 2.568476, 1.591998, 1.190108],  # x=0.10
    [0.004837, 0.045121, 0.343803, 1.573757, 2.554747, 1.590795, 1.189698],  # x=0.17783
    [0.004837, 0.044507, 0.316063, 1.47078, 2.507772, 1.586979, 1.188398],   # x=0.31623
    [0.004705, 0.036765, 0.249325, 1.238666, 2.421019, 1.581228, 1.186445],  # x=0.45
    [0.002419, 0.022562, 0.183937, 1.025219, 2.361647, 1.578549, 1.185538],  # x=0.50
    [5.1e-05, 0.006779, 0.108887, 0.759317, 2.280932, 1.5748, 1.184271],     # x=0.56234
    [np.nan, 6.4e-05, 0.034842, 0.416175, 2.069946, 1.56071, 1.179537],      # x=0.75
    [np.nan, np.nan, 0.006872, 0.214491, 1.68516, 1.535052, 1.171036],       # x=1.0
    [np.nan, np.nan, 0.000168, 0.094966, 1.028758, 1.487096, 1.155611],      # x=1.33352
    [np.nan, np.nan, np.nan, 0.032116, 0.471906, 1.391456, 1.127131],        # x=1.77828
    [np.nan, np.nan, np.nan, 0.000196, 0.049604, 0.471468, 0.954827],        # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.001163, 0.019493, 0.082189],          # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan, 0.000113, 0.000487],            # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],                # x=17.78279
])


# Problem parameters
sigma_P = 1.0      # cm^-1 (Planck and Rosseland opacity)
x_min = 0.0        # cm (reflecting boundary)
x_max = 20.0       # cm
n_cells = 400      # Number of cells
source_region = 0.5  # cm (source from 0 to 0.5)
source_magnitude = A_RAD * C_LIGHT  # Half the equilibrium at T=1 keV

# Time parameters
mean_free_time = 1.0 / (C_LIGHT * sigma_P)  # τ = 1/(c·σ) in ns
print(f"Bennett problem with constant Cv = {CV_CONST} GJ/(g·keV)")
print(f"Mean free time τ = {mean_free_time:.6f} ns")

source_duration = 10.0 * mean_free_time  # Source on for 10 τ

# Define output times for plots
early_output_times_mft = [0.1, 1.0, 3.16228, 10.0]  # Early times in mean free times
late_output_times_mft = [31.6228, 100.0]   # Late times in mean free times
all_output_times_mft = early_output_times_mft + late_output_times_mft

print(f"Source duration: {source_duration:.6f} ns = 10τ")
print(f"Early output times: {early_output_times_mft} τ")
print(f"Late output times: {late_output_times_mft} τ")

# Time stepping - run to last output time
dt = 0.005 * mean_free_time  # Timestep
final_time = max(all_output_times_mft) * mean_free_time
n_steps = int(np.ceil(final_time / dt))

print(f"Timestep: {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.3f}τ")
print(f"Total steps: {n_steps}")

# Material properties for Bennett problem with CONSTANT specific heat
def bennett_opacity(T):
    """Planck opacity - constant"""
    return sigma_P

def bennett_rosseland_opacity(T):
    """Rosseland opacity - same as Planck for this test"""
    return sigma_P

def bennett_specific_heat(T):
    """Specific heat for Bennett problem: c_v = CONSTANT = 0.03 GJ/(g·keV)"""
    # Return constant value, independent of temperature
    if np.isscalar(T):
        return CV_CONST
    else:
        return np.full_like(T, CV_CONST)

def bennett_energy(T):
    """Material energy density for Bennett problem: e = Cv·ρ·T (linear in T)"""
    T_safe = np.maximum(T, 1e-6)
    return CV_CONST * RHO * T_safe

def bennett_inverse_energy(e):
    """Inverse: T from e for Bennett problem: T = e/(Cv·ρ)"""
    e_safe = np.maximum(e, 1e-30)
    return e_safe / (CV_CONST * RHO)

# Boundary conditions
def reflecting_bc(phi, x):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def vacuum_bc(phi, x):
    """Vacuum boundary: incoming current is zero"""
    return 0.5, 1.0/(3.0*sigma_P), 0.0  # A=1/2, B=1/(3*sigma_R), C=0

# Define flux limiters to test
flux_limiters = {
    'Levermore-Pomraning': flux_limiter_levermore_pomraning,
    'Larsen n=2': flux_limiter_larsen,
    'Sum': flux_limiter_sum,
    'Max': flux_limiter_max,
    'None': flux_limiter_standard  # Use standard diffusion (no limiter) for comparison
}

# Storage for results
results = {}

print(f"\n{'='*80}")
print("Running Bennett problem with different flux limiters")
print(f"{'='*80}")

# Run with each flux limiter
for limiter_name, limiter_func in flux_limiters.items():
    print(f"\n{'-'*80}")
    print(f"Running with {limiter_name} flux limiter...")
    print(f"{'-'*80}")
    
    # Create solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=x_min, r_max=x_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=bennett_rosseland_opacity,
        planck_opacity_func=bennett_opacity,
        specific_heat_func=bennett_specific_heat,
        material_energy_func=bennett_energy,
        inverse_material_energy_func=bennett_inverse_energy,
        left_bc_func=reflecting_bc,
        right_bc_func=vacuum_bc,
        theta=1.0,
        flux_limiter_func=limiter_func
    )
    
    # Determine which cells have the source
    source_mask = solver.r_centers < source_region
    
    # Initial conditions
    T_init = 0.001  # keV
    phi_init = A_RAD * C_LIGHT * T_init**4
    solver.phi = np.ones(n_cells) * phi_init
    solver.T = np.ones(n_cells) * T_init
    
    # Source strength
    source_strength = np.zeros(n_cells)
    source_strength[source_mask] = source_magnitude
    
    def source_function(t):
        """Return source array at time t"""
        if t <= source_duration:
            return source_strength.copy()
        else:
            return np.zeros(n_cells)
    
    # Time evolution with saving at output times
    saved_solutions = {}  # Dictionary to store solutions at specific times
    next_output_idx = 0
    current_time = 0.0
    
    for step in range(n_steps):
        current_time_before = current_time
        current_time = (step + 1) * dt
        
        active_source = source_function(current_time_before)
        
        # Print progress
        verbose = (step < 2) or (step % 1000 == 0) or (step == n_steps - 1)
        if verbose:
            print(f"  Step {step+1}/{n_steps}: t = {current_time/mean_free_time:.3f}τ")
        
        solver.time_step(n_steps=1, source=active_source, verbose=False)
        
        # Check if we should save this timestep (when we cross an output time)
        if next_output_idx < len(all_output_times_mft):
            target_time = all_output_times_mft[next_output_idx] * mean_free_time
            if current_time >= target_time:
                tau_value = all_output_times_mft[next_output_idx]
                saved_solutions[tau_value] = {
                    'r': solver.r_centers.copy(),
                    'phi': solver.phi.copy(),
                    'T': solver.T.copy(),
                    'E_rad': solver.phi.copy() / (A_RAD * C_LIGHT),
                    'E_mat': bennett_energy(solver.T.copy()) / (A_RAD * C_LIGHT)  # Normalized material energy
                }
                print(f"    Saved solution at τ = {tau_value:.4f}")
                next_output_idx += 1
    
    # Store results for this limiter
    results[limiter_name] = saved_solutions
    
    print(f"  Completed {limiter_name}: saved {len(saved_solutions)} time snapshots")

# Plotting comparisons
print(f"\n{'='*80}")
print("Creating comparison plots...")
print(f"{'='*80}")

# Define colors for different times
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Use consistent markers: circles for diffusion, squares for transport (all times)
marker_diffusion = 'o'
marker_transport = 's'

# Define colors and line styles for different flux limiters
flux_limiter_colors = {
    'Levermore-Pomraning': 'red',
    'Larsen n=2': 'green',
    'Sum': 'black',
    'Max': 'purple',
    'None': 'blue'
}

flux_limiter_linestyles = {
    'Levermore-Pomraning': '--',
    'Larsen n=2': '-.',
    'Sum': '-',
    'Max': (0, (3, 1, 1, 1)),  # densely dashdotted
    'None': ':'  # dotted for no limiter
}

# ============================================================================
# EARLY TIMES PLOT (0.1, 1.0, 3.16228, 10.0 τ)
# Both radiation and material energy, x=0 to 10 cm
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

# Plot reference data (transport) with consistent markers
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
for limiter_name in ['Levermore-Pomraning', 'Larsen n=2', 'Sum', 'Max', 'None']:
    legend_elements.append(
        Line2D([0], [0], color=flux_limiter_colors[limiter_name], 
               linestyle=flux_limiter_linestyles[limiter_name], 
               linewidth=2.0, label=limiter_name)
    )
# Add reference data markers
legend_elements.extend([
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black',
           markersize=6, linestyle='', label='Transport ref')
])
ax1.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)

ax1.grid(True, alpha=0.3)
ax1.set_xlim([.2, 8])
ax1.set_ylim([1e-3, 3e0])

plt.tight_layout()
show('bennett_flux_limiter_early_times.pdf', close_after=True)
print("Saved plot to 'bennett_flux_limiter_early_times.pdf'")

# ============================================================================
# LATE TIMES PLOT (31.6228, 100.0 τ)
# Both radiation and material energy, full domain
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

# Plot reference data (transport) with consistent markers
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
for limiter_name in ['Levermore-Pomraning', 'Larsen n=2', 'Sum', 'Max', 'None']:
    legend_elements.append(
        Line2D([0], [0], color=flux_limiter_colors[limiter_name], 
               linestyle=flux_limiter_linestyles[limiter_name], 
               linewidth=2.0, label=limiter_name)
    )
# Add reference data markers
legend_elements.extend([
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black',
           markersize=6, linestyle='', label='Transport ref')
])
ax2.legend(handles=legend_elements, fontsize=10, loc='best', ncol=1)
ax2.set_xlim(left=0.2, right=2e1)
ax2.set_ylim(bottom=1e-3, top=1e0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show('bennett_flux_limiter_late_times.pdf', close_after=True)
print("Saved plot to 'bennett_flux_limiter_late_times.pdf'")

print(f"\n{'='*80}")
print("Comparison complete!")
print(f"{'='*80}")
