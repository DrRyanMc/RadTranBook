"""
Su-Olson type test problem for non-equilibrium radiation diffusion
using TR-BDF2 time discretization

Problem setup:
- 1-D slab geometry, x from 0 to 12 cm
- Radiation source of magnitude 1/2 for 0 < x < 0.5 cm
- σ_P = σ_R = 1.0 cm^-1
- Material energy: e(T) = a·T^4 (radiation-dominated)
- Source duration: 10 mean free times = 10/(c·σ_P)
- Output at: 0.1, 1.0, and 10.0 mean free times
- Reflecting BC at x=0, vacuum BC at x=12
- Time discretization: TR-BDF2 (two-stage, second-order)
"""
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³

# Reference solution data from Su & Olson (1997)
# x values in cm
su_olson_x = np.array([0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000, 
                        0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228, 
                        5.62341, 10.00000, 17.78279])

# tau (time) values in mean free times
su_olson_tau = np.array([0.10000, 0.31623, 1.00000, 3.16228, 10.00000])

# Reference solution: su_olson_data[i, j] is value at x[i], tau[j]
# NaN for missing data points
su_olson_data = np.array([
    [0.09403, 0.24356, 0.50359, 0.95968, 1.86585],  # x=0.01
    [0.09326, 0.24002, 0.49716, 0.95049, 1.85424],  # x=0.1
    [0.09128, 0.23207, 0.48302, 0.93036, 1.82889],  # x=0.17783
    [0.08230, 0.20515, 0.43743, 0.86638, 1.74866],  # x=0.31623
    [0.06086, 0.15981, 0.36656, 0.76956, 1.62824],  # x=0.45
    [0.04766, 0.13682, 0.33271, 0.72433, 1.57237],  # x=0.5
    [0.03171, 0.10856, 0.29029, 0.66672, 1.50024],  # x=0.56234
    [0.00755, 0.05086, 0.18879, 0.51507, 1.29758],  # x=0.75
    [0.00064, 0.01583, 0.10150, 0.35810, 1.06011],  # x=1.0
    [np.nan, 0.00244, 0.04060, 0.21309, 0.79696],   # x=1.33352
    [np.nan, np.nan, 0.01011, 0.10047, 0.52980],    # x=1.77828
    [np.nan, np.nan, 0.00003, 0.00634, 0.12187],    # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00445],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan],       # x=17.78279
])

# Material energy density reference solution: su_olson_material_energy[i, j] is value at x[i], tau[j]
su_olson_material_energy = np.array([
    [0.00466, 0.03816, 0.21859, 0.75342, 1.75359],  # x=0.01
    [0.00464, 0.03768, 0.21565, 0.74557, 1.74218],  # x=0.1
    [0.00458, 0.03658, 0.20913, 0.72837, 1.71726],  # x=0.17783
    [0.00424, 0.03253, 0.18765, 0.67348, 1.63837],  # x=0.31623
    [0.00315, 0.02476, 0.15298, 0.58978, 1.51991],  # x=0.45
    [0.00234, 0.02042, 0.13590, 0.55041, 1.46494],  # x=0.5
    [0.00137, 0.01515, 0.11468, 0.50052, 1.39405],  # x=0.56234
    [0.00023, 0.00580, 0.06746, 0.37270, 1.19584],  # x=0.75
    [np.nan, 0.00139, 0.03173, 0.24661, 0.96571],   # x=1.0
    [np.nan, 0.00015, 0.01063, 0.13729, 0.71412],   # x=1.33352
    [np.nan, np.nan, 0.00210, 0.05918, 0.46369],    # x=1.77828
    [np.nan, np.nan, np.nan, 0.00281, 0.09834],     # x=3.16228
    [np.nan, np.nan, np.nan, np.nan, 0.00306],      # x=5.62341
    [np.nan, np.nan, np.nan, np.nan, np.nan],       # x=10.0
    [np.nan, np.nan, np.nan, np.nan, np.nan],       # x=17.78279
])

# Problem parameters
sigma_P = 1.0      # cm^-1 (Planck and Rosseland opacity)
x_min = 0.0        # cm (reflecting boundary)
x_max = 12.0       # cm
n_cells = 120      # Number of cells
source_region = 0.5  # cm (source from 0 to 0.5)
# Source magnitude: For φ equation, source is in units of GJ/(cm^3·ns)
source_magnitude = A_RAD * C_LIGHT  # Half the equilibrium at T=1 keV

# Time parameters
mean_free_time = 1.0 / (C_LIGHT * sigma_P)  # τ = 1/(c·σ) in ns
print(f"Mean free time τ = {mean_free_time:.6f} ns")

source_duration = 10.0 * mean_free_time  # Source on for 10 τ
output_times_mft = [0.1, 1.0, 10.0]  # Output times in mean free times
output_times = [t * mean_free_time for t in output_times_mft]

print(f"Source duration: {source_duration:.6f} ns = 10τ")
print(f"Output times: {[f'{t:.6f} ns' for t in output_times]}")

# Time stepping - run to 10 mean free times
dt = 0.01 * mean_free_time  # Timestep
n_steps = 1000  # 1000 steps to reach 10 mean free times
final_time = n_steps * dt
print(f"Timestep: {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: {final_time:.6e} ns = {final_time/mean_free_time:.3f}τ")
print(f"Total steps: {n_steps}")

# Material properties for radiation-dominated material
# e(T) = a·T^4, so C_v = de/dT = 4a·T^3
def radiation_dominated_opacity(T):
    """Planck opacity - constant"""
    return sigma_P

def radiation_dominated_rosseland_opacity(T):
    """Rosseland opacity - same as Planck for this test"""
    return sigma_P

def radiation_dominated_specific_heat(T):
    """Specific heat for e = a·T^4: c_v = 4a·T^3/ρ"""
    T_safe = np.maximum(T, 1e-6)  # Avoid division by zero
    return (4.0 * A_RAD * T_safe**3) / RHO

def radiation_dominated_energy(T):
    """Material energy density for radiation-dominated: e = a·T^4"""
    T_safe = np.maximum(T, 1e-6)  # Avoid numerical issues
    return A_RAD * T_safe**4

def radiation_dominated_inverse_energy(e):
    """Inverse: T from e for radiation-dominated: T = (e/a)^(1/4)"""
    e_safe = np.maximum(e, 1e-30)  # Avoid negative/zero
    return (e_safe / A_RAD)**0.25

# Boundary conditions
def reflecting_bc(phi, x):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def vacuum_bc(phi, x):
    """Vacuum boundary: incoming current is zero"""
    return 0.5, 1.0/(3.0*sigma_P), 0.0  # A=1/2, B=1/(3*sigma_R), C=0

# Create solver (TR-BDF2 doesn't need theta parameter)
print(f"\nInitializing solver with {n_cells} cells...")
solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=x_min, r_max=x_max, n_cells=n_cells, d=0,  # Planar geometry
    dt=dt, max_newton_iter=10, newton_tol=1e-8,
    rosseland_opacity_func=radiation_dominated_rosseland_opacity,
    planck_opacity_func=radiation_dominated_opacity,
    specific_heat_func=radiation_dominated_specific_heat,
    material_energy_func=radiation_dominated_energy,
    inverse_material_energy_func=radiation_dominated_inverse_energy,
    left_bc_func=reflecting_bc,
    right_bc_func=vacuum_bc
)

# Determine which cells have the source
source_mask = solver.r_centers < source_region
print(f"Source region: {np.sum(source_mask)} cells (x < {source_region} cm)")

# Initial conditions (start with finite temperature to avoid stiffness)
T_init = 0.001  # keV
phi_init = A_RAD * C_LIGHT * T_init**4  # Corresponding radiation
solver.phi = np.ones(n_cells) * phi_init
solver.T = np.ones(n_cells) * T_init

print(f"Initial conditions: T = {T_init} keV, φ = {phi_init:.6e} GJ/cm³")
print(f"Source magnitude: {source_magnitude:.6e} GJ/(cm³·ns)")
print(f"Ratio of source to initial φ/dt: {source_magnitude / (phi_init/dt):.3f}")

print(f"\n{'='*80}")
print("TR-BDF2 Time Integration")
print(f"{'='*80}")

# TR-BDF2 parameter
Lambda = 2.0 - np.sqrt(2.0)  # ≈ 0.586
print(f"TR-BDF2 parameter Λ = {Lambda:.6f}")

# Determine source strength - spatial distribution
source_strength = np.zeros(n_cells)
source_strength[source_mask] = source_magnitude

# Create time-dependent source function
def source_function(t):
    """Return source array at time t"""
    if t <= source_duration:
        return source_strength.copy()
    else:
        return np.zeros(n_cells)

print(f"Source active in {np.sum(source_mask)} cells")
print(f"Source duration: {source_duration:.6e} ns = {source_duration/mean_free_time:.1f}τ")

# Time evolution loop using built-in solver.time_step_trbdf2() with source
current_time = 0.0
for step in range(n_steps):
    current_time_before = current_time
    current_time = (step + 1) * dt
    
    # Determine active source for this timestep
    active_source = source_function(current_time_before)
    
    # Print progress every 100 steps
    verbose = (step < 2) or (step % 100 == 0) or (step == n_steps - 1)
    
    if verbose:
        print(f"\nStep {step+1}/{n_steps}: t = {current_time:.6e} ns = {current_time/mean_free_time:.3f}τ")
    
    # Take single TR-BDF2 timestep with source
    solver.time_step_trbdf2(n_steps=1, Lambda=Lambda, source=active_source, verbose=verbose)

print(f"\n{'='*80}")
print(f"Completed {n_steps} steps to t = {current_time:.6e} ns = {current_time/mean_free_time:.3f}τ")
print(f"{'='*80}")

print(f"\nResults at t = {current_time/mean_free_time:.1f}τ:")
print(f"  In source region (x < {source_region} cm):")
print(f"    Mean T = {np.mean(solver.T[source_mask]):.6f} keV")
print(f"    Mean T_rad = {np.mean((solver.phi[source_mask]/(A_RAD*C_LIGHT))**0.25):.6f} keV")
print(f"  Outside source region:")
print(f"    Mean T = {np.mean(solver.T[~source_mask]):.6f} keV")
T_rad_outside = solver.phi[~source_mask] / (A_RAD * C_LIGHT)
T_rad_outside_pos = T_rad_outside[T_rad_outside > 0]
if len(T_rad_outside_pos) > 0:
    print(f"    Mean T_rad = {np.mean(T_rad_outside_pos**0.25):.6f} keV")

# Plotting
print(f"\n{'='*80}")
print("Creating plots...")
print(f"{'='*80}")

# Calculate radiation temperature
T_rad = np.zeros_like(solver.phi)
positive_mask = solver.phi > 0
T_rad[positive_mask] = (solver.phi[positive_mask] / (A_RAD * C_LIGHT))**0.25

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Figure out what time from the table we are at
su_olson_tau_ns = su_olson_tau * mean_free_time
time_index = np.argmin(np.abs(su_olson_tau_ns - current_time))

# Plot 1: Radiation energy density φ (full domain)
ax = axes[0, 0]
ax.plot(solver.r_centers, solver.phi/(A_RAD * C_LIGHT), 'r-', linewidth=2, label='Numerical (TR-BDF2)')
ax.plot(su_olson_x, su_olson_data[:, time_index], 'ko', label=f'Su-Olson Ref at {su_olson_tau[time_index]:.2f}τ')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=source_region, color='red', linestyle='--', alpha=0.5, label='Source boundary')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Radiation Energy Density φ/c (GJ/cm³)', fontsize=12)
ax.set_title(f'Radiation Energy Density at t = {current_time/mean_free_time:.1f}τ', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Material energy density (full domain)
ax = axes[0, 1]
ax.plot(solver.r_centers, solver.T**4, 'b-', linewidth=2, label='Numerical (TR-BDF2)')
ax.plot(su_olson_x, su_olson_material_energy[:, time_index], 'ko', label=f'Su-Olson Ref at {su_olson_tau[time_index]:.2f}τ')
ax.axvline(x=source_region, color='red', linestyle='--', alpha=0.5, label='Source boundary')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Material Energy Density e (GJ/cm³)', fontsize=12)
ax.set_title(f'Material Energy Density at t = {current_time/mean_free_time:.1f}τ', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Radiation (zoomed to first 5 cm)
ax = axes[1, 0]
ax.plot(solver.r_centers, solver.phi/(A_RAD * C_LIGHT), 'r-', linewidth=2, label='φ/c')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=source_region, color='red', linestyle='--', alpha=0.5, label='Source boundary')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Radiation Energy Density φ/c (GJ/cm³)', fontsize=12)
ax.set_title('φ/c (zoomed to first 5 cm)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])

# Plot 4: Log scale (full domain)
ax = axes[1, 1]
ax.semilogy(solver.r_centers, solver.phi/(A_RAD * C_LIGHT), 'r-', linewidth=2, label='Numerical')
ax.semilogy(su_olson_x, su_olson_data[:, time_index], 'ko', label='Su-Olson Ref')
ax.axvline(x=source_region, color='red', linestyle='--', alpha=0.5, label='Source boundary')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Radiation Energy Density φ/c (GJ/cm³)', fontsize=12)
ax.set_title('φ/c (log scale)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([1e-4, 10])

plt.tight_layout()
plt.savefig('su_olson_trbdf2.png', dpi=150, bbox_inches='tight')
print("Saved plot to su_olson_trbdf2.png")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Problem: Su-Olson test with TR-BDF2 time discretization")
print(f"Domain: x ∈ [0, {x_max}] cm with {n_cells} cells")
print(f"Source: magnitude {source_magnitude:.6e} GJ/(cm³·ns) for 0 < x < {source_region} cm")
print(f"Opacity: σ_P = {sigma_P} cm^-1")
print(f"Material: e(T) = a·T^4 (radiation-dominated)")
print(f"Time discretization: TR-BDF2 (two-stage, second-order)")
print(f"TR-BDF2 parameter: Λ = {Lambda:.6f}")
print(f"Timestep: dt = {dt:.6e} ns = {dt/mean_free_time:.3f}τ")
print(f"Final time: t = {final_time:.6e} ns = {final_time/mean_free_time:.3f}τ")
print(f"Total timesteps: {n_steps}")
print(f"\nFinal results at t = {current_time/mean_free_time:.1f}τ:")
print(f"  In source region: T ~ {np.mean(solver.T[source_mask]):.4f} keV")
print(f"  Outside source:   T ~ {np.mean(solver.T[~source_mask]):.4f} keV")
