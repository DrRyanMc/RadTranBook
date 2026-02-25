"""
Convergence study for numerical time integration methods

This script compares the temporal convergence rates of three time integration schemes:
- Backward Euler (θ=1.0, first-order accurate)
- Crank-Nicolson (θ=0.5, second-order accurate)
- TR-BDF2 (second-order accurate)

Problem setup: Su-Olson type test problem
- 1-D slab geometry, x from 0 to 20 cm
- Radiation source of magnitude 1/2 for 0 < x < 0.5 cm
- σ_P = σ_R = 1.0 cm^-1
- Material energy: e(T) = a·T^4 (radiation-dominated)
- Source duration: 10 mean free times = 10/(c·σ_P)
- Convergence study at τ = 1.0 mean free time
- Reflecting BC at x=0, vacuum BC at x=20
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Add project root to path for shared utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.plotfuncs import show, hide_spines, font

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³

# Problem parameters
sigma_P = 1.0      # cm^-1 (Planck and Rosseland opacity)
x_min = 0.0        # cm (reflecting boundary)
x_max = 2.0       # cm
source_region = 0.5  # cm (source from 0 to 0.5)
source_magnitude = A_RAD * C_LIGHT  # Half the equilibrium at T=1 keV

# Time parameters
mean_free_time = 1.0 / (C_LIGHT * sigma_P)  # τ = 1/(c·σ) in ns
source_duration = 10.0 * mean_free_time  # Source on for 10 τ
target_time = 1.0 * mean_free_time  # Study convergence at τ = 1.0

# CFL-based refinement
# For diffusion: CFL = D * dt / dx^2, where D ~ c/(3*sigma)
# To maintain constant CFL: n_cells ~ 1/sqrt(dt)
base_n_cells = 50      # Base number of cells
base_dt = mean_free_time / 10.0  # Base timestep
CFL_number = C_LIGHT / (3.0 * sigma_P) * base_dt / ((x_max - x_min) / base_n_cells)**2

print(f"{'='*80}")
print("TEMPORAL CONVERGENCE STUDY (Constant CFL)")
print(f"{'='*80}")
print(f"Mean free time τ = {mean_free_time:.6f} ns")
print(f"Source duration: 10.0 τ")
print(f"Target time for convergence study: 1.0 τ")
print(f"Domain: [{x_min}, {x_max}] cm")
print(f"CFL number: {CFL_number:.4f}")
print(f"Base configuration: {base_n_cells} cells, Δt = {base_dt/mean_free_time:.6f}τ")

# Material properties for Su-Olson problem
def radiation_dominated_opacity(T):
    """Planck opacity - constant"""
    return sigma_P

def radiation_dominated_rosseland_opacity(T):
    """Rosseland opacity - same as Planck for this test"""
    return sigma_P

def radiation_dominated_specific_heat(T):
    """Specific heat: c_v = 4aT³/ρ for radiation-dominated material"""
    T_safe = np.maximum(T, 1e-6)
    return (4.0 * A_RAD * T_safe**3) / RHO

def radiation_dominated_energy(T):
    """Material energy density: e = aT⁴ for radiation-dominated material"""
    T_safe = np.maximum(T, 1e-6)
    return A_RAD * T_safe**4

def radiation_dominated_inverse_energy(e):
    """Inverse: T from e for radiation-dominated material"""
    e_safe = np.maximum(e, 1e-30)
    return (e_safe / A_RAD)**0.25

# Boundary conditions
def reflecting_bc(phi, x):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def vacuum_bc(phi, x):
    """Vacuum boundary: incoming current is zero"""
    return 0.5, 1.0/(3.0*sigma_P), 0.0  # A=1/2, B=1/(3*sigma_R), C=0


def run_simulation(dt, n_cells, method='theta', theta=1.0, verbose=False):
    """
    Run simulation to target_time with given timestep and method
    
    Parameters:
    -----------
    dt : float
        Timestep size
    n_cells : int
        Number of spatial cells
    method : str
        'theta' for θ-method or 'trbdf2' for TR-BDF2
    theta : float
        θ parameter for θ-method (0.5=Crank-Nicolson, 1.0=Backward Euler)
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    r_centers : array
        Cell centers
    phi : array
        Radiation energy density at target time
    T : array
        Material temperature at target time
    """
    # Create solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=x_min, r_max=x_max, n_cells=n_cells, d=0,
        dt=dt, max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=radiation_dominated_rosseland_opacity,
        planck_opacity_func=radiation_dominated_opacity,
        specific_heat_func=radiation_dominated_specific_heat,
        material_energy_func=radiation_dominated_energy,
        inverse_material_energy_func=radiation_dominated_inverse_energy,
        left_bc_func=reflecting_bc,
        right_bc_func=vacuum_bc,
        theta=theta
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
    
    # Time evolution - ensure we stop exactly at target_time
    current_time = 0.0
    step_count = 0
    
    while current_time < target_time:
        # Determine timestep size (adjust last step to hit target_time exactly)
        if current_time + dt > target_time:
            actual_dt = target_time - current_time
            # Need to temporarily change solver dt for this last step
            original_dt = solver.dt
            solver.dt = actual_dt
        else:
            actual_dt = dt
        
        active_source = source_function(current_time)
        
        if verbose and (step_count % 100 == 0):
            print(f"    Step {step_count+1}: t = {current_time/mean_free_time:.4f}τ, dt = {actual_dt/mean_free_time:.6f}τ")
        
        # Use appropriate time stepping method
        if method == 'trbdf2':
            solver.time_step_trbdf2(n_steps=1, source=active_source, verbose=False)
        else:
            solver.time_step(n_steps=1, source=active_source, verbose=False)
        
        current_time += actual_dt
        step_count += 1
        
        # Restore original dt if we changed it
        if actual_dt != dt:
            solver.dt = original_dt
    
    if verbose:
        print(f"    Final time: {current_time/mean_free_time:.6f}τ ({step_count} steps)")
    
    return solver.r_centers.copy(), solver.phi.copy(), solver.T.copy()


def compute_errors(r, phi, T, r_ref, phi_ref, T_ref):
    """
    Compute L1, L2, and L-infinity errors against reference solution
    
    Parameters:
    -----------
    r : array
        Cell centers for numerical solution
    phi : array
        Numerical radiation energy density
    T : array
        Numerical temperature
    r_ref : array
        Cell centers for reference solution
    phi_ref : array
        Reference radiation energy density
    T_ref : array
        Reference temperature
        
    Returns:
    --------
    errors : dict
        Dictionary with error norms for phi and T
    """
    # Interpolate numerical solution onto reference mesh
    phi_interp = np.interp(r_ref, r, phi)
    T_interp = np.interp(r_ref, r, T)
    
    # Compute errors for radiation field
    phi_error = np.abs(phi_interp - phi_ref)
    phi_L1 = np.mean(phi_error)
    phi_L2 = np.sqrt(np.mean(phi_error**2))
    phi_Linf = np.max(phi_error)
    
    # Compute errors for temperature
    T_error = np.abs(T_interp - T_ref)
    T_L1 = np.mean(T_error)
    T_L2 = np.sqrt(np.mean(T_error**2))
    T_Linf = np.max(T_error)
    
    return {
        'phi_L1': phi_L1,
        'phi_L2': phi_L2,
        'phi_Linf': phi_Linf,
        'T_L1': T_L1,
        'T_L2': T_L2,
        'T_Linf': T_Linf
    }


# Define range of timesteps and corresponding mesh sizes for constant CFL
n_refinements = 4
dt_factors = 2.0**np.arange(n_refinements)  # [1, 2, 4, 8, 16, 32]
timesteps = base_dt / dt_factors

# Calculate n_cells for each timestep to maintain constant CFL
# n_cells ~ 1/sqrt(dt) to keep CFL constant
n_cells_list = []
for dt in timesteps:
    # From CFL = D * dt / dx^2 with constant CFL:
    # dx = sqrt(D * dt / CFL)
    # n_cells = (x_max - x_min) / dx
    D_effective = C_LIGHT / (3.0 * sigma_P)
    dx = np.sqrt(D_effective * dt / CFL_number)
    n_cells_i = int(np.ceil((x_max - x_min) / dx))
    n_cells_list.append(n_cells_i)

# Generate or load reference solution with very fine timestep and mesh
print(f"\n{'-'*80}")

# Check for old reference files with different naming scheme
old_ref_pattern = "convergence_reference_dt"
script_dir = os.path.dirname(os.path.abspath(__file__))
old_files = [f for f in os.listdir(script_dir) if f.startswith(old_ref_pattern) and not "tau" in f]
if old_files:
    print(f"WARNING: Found {len(old_files)} old reference file(s) with outdated naming:")
    for f in old_files:
        print(f"  - {f}")
    print("Consider deleting these to avoid confusion.")
    print(f"{'-'*80}")

dt_ref = mean_free_time / 10000.0  # Very fine timestep
# Calculate corresponding mesh size to maintain same CFL
D_effective = C_LIGHT / (3.0 * sigma_P)
dx_ref = np.sqrt(D_effective * dt_ref / CFL_number)
n_cells_ref = int(np.ceil((x_max - x_min) / dx_ref))

# Define reference solution filename based on parameters
# Include target_time to avoid using cached solutions from different runs
ref_filename = f"convergence_reference_t{target_time/mean_free_time:.1f}tau_dt{dt_ref/mean_free_time:.6f}_ncells{n_cells_ref}.npz"
ref_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ref_filename)

# Try to load existing reference solution
if os.path.exists(ref_filepath):
    print(f"Loading existing reference solution from {ref_filename}...")
    print(f"{'-'*80}")
    ref_data = np.load(ref_filepath)
    r_ref = ref_data['r']
    phi_ref = ref_data['phi']
    T_ref = ref_data['T']
    print(f"Reference solution loaded: {len(r_ref)} cells, Δt = {dt_ref/mean_free_time:.6f}τ")
else:
    print(f"Generating reference solution with very fine timestep...")
    print(f"{'-'*80}")
    print(f"Reference: Δt = {dt_ref/mean_free_time:.6f}τ, {n_cells_ref} cells, Δx = {dx_ref:.6f} cm (~{int(target_time/dt_ref)} steps)")
    print(f"Using Backward Euler (theta=1.0) for stability with fine timestep")
    
    r_ref, phi_ref, T_ref = run_simulation(dt_ref, n_cells_ref, method='TR-BDF2',verbose=False)
    
    # Save reference solution for future use
    np.savez(ref_filepath, r=r_ref, phi=phi_ref, T=T_ref)
    print(f"Reference solution computed and saved to {ref_filename}")

print(f"\n{'-'*80}")
print(f"Testing {n_refinements} refinement levels (constant CFL = {CFL_number:.4f}):")
for i, (dt, n_cells_i) in enumerate(zip(timesteps, n_cells_list)):
    n_steps = int(np.ceil(target_time / dt))
    dx = (x_max - x_min) / n_cells_i
    n_source_cells = int(source_region / dx)
    print(f"  {i+1}. Δt = {dt/mean_free_time:.6f}τ, {n_cells_i} cells, Δx = {dx:.4f} cm (~{n_source_cells} cells in source)")
print(f"{'-'*80}")

# Storage for results
methods = {
    'Backward Euler': {'method': 'theta', 'theta': 1.0, 'color': 'blue', 'marker': 'o'},
    'Crank-Nicolson': {'method': 'theta', 'theta': 0.5, 'color': 'red', 'marker': 's'},
    'TR-BDF2': {'method': 'trbdf2', 'theta': 1.0, 'color': 'green', 'marker': '^'}
}

results = {}

# Run convergence study for each method
for method_name, method_params in methods.items():
    print(f"\n{'='*80}")
    print(f"Testing {method_name}")
    print(f"{'='*80}")
    
    errors_list = []
    
    for i, (dt, n_cells_i) in enumerate(zip(timesteps, n_cells_list)):
        n_steps = int(np.ceil(target_time / dt))
        dx = (x_max - x_min) / n_cells_i
        print(f"\n  Run {i+1}/{n_refinements}: Δt = {dt/mean_free_time:.6f}τ, {n_cells_i} cells, Δx = {dx:.4f} cm")
        
        r, phi, T = run_simulation(
            dt, 
            n_cells_i,
            method=method_params['method'], 
            theta=method_params['theta'],
            verbose=False
        )
        
        errors = compute_errors(r, phi, T, r_ref, phi_ref, T_ref)
        errors_list.append(errors)
        
        print(f"    φ errors: L1={errors['phi_L1']:.4e}, L2={errors['phi_L2']:.4e}, L∞={errors['phi_Linf']:.4e}")
        print(f"    T errors: L1={errors['T_L1']:.4e}, L2={errors['T_L2']:.4e}, L∞={errors['T_Linf']:.4e}")
    
    results[method_name] = {
        'timesteps': timesteps,
        'errors': errors_list,
        'params': method_params
    }

# Compute convergence rates
print(f"\n{'='*80}")
print("CONVERGENCE RATES")
print(f"{'='*80}")

for method_name, data in results.items():
    print(f"\n{method_name}:")
    errors = data['errors']
    
    # Compute rates between consecutive refinements
    phi_L2_errors = np.array([e['phi_L2'] for e in errors])
    T_L2_errors = np.array([e['T_L2'] for e in errors])
    
    # Compute convergence rate using last few points (avoiding very small dt where roundoff dominates)
    if len(phi_L2_errors) >= 3:
        # Use middle points to avoid initial transients and final roundoff
        idx1, idx2 = len(phi_L2_errors)//3, 2*len(phi_L2_errors)//3
        dt_ratio = timesteps[idx1] / timesteps[idx2]
        
        phi_rate = np.log(phi_L2_errors[idx1] / phi_L2_errors[idx2]) / np.log(dt_ratio)
        T_rate = np.log(T_L2_errors[idx1] / T_L2_errors[idx2]) / np.log(dt_ratio)
        
        print(f"  φ L2 convergence rate: {phi_rate:.2f}")
        print(f"  T L2 convergence rate: {T_rate:.2f}")

# Plotting
print(f"\n{'='*80}")
print("Creating convergence plots...")
print(f"{'='*80}")

# Create two subplots: one for φ, one for T
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Normalized timestep values
dt_normalized = timesteps / mean_free_time

# Plot phi errors
ax = axes[0]
for method_name, data in results.items():
    params = data['params']
    errors = data['errors']
    phi_L2 = np.array([e['phi_L2'] for e in errors])
    
    ax.loglog(dt_normalized, phi_L2, 
             marker=params['marker'], color=params['color'], 
             linewidth=2, markersize=8, label=method_name)

# Add reference lines for convergence orders
dt_mid = dt_normalized[len(dt_normalized)//2]
error_mid = 1e-3  # Reference error level
ax.loglog(dt_normalized, error_mid * (dt_normalized/dt_mid)**1, 
         'k--', alpha=0.3, linewidth=1, label='1st order')
ax.loglog(dt_normalized, error_mid * (dt_normalized/dt_mid)**2, 
         'k:', alpha=0.3, linewidth=1, label='2nd order')

ax.set_xlabel(r'Timestep $\Delta t$ (mean free times)', fontsize=12)
ax.set_ylabel(r'$L^2$ Error in $\phi$', fontsize=12)
ax.set_title('Radiation Energy Density Convergence', fontsize=13)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Plot T errors
ax = axes[1]
for method_name, data in results.items():
    params = data['params']
    errors = data['errors']
    T_L2 = np.array([e['T_L2'] for e in errors])
    
    ax.loglog(dt_normalized, T_L2, 
             marker=params['marker'], color=params['color'], 
             linewidth=2, markersize=8, label=method_name)

# Add reference lines
error_mid = 1e-3  # Reference error level
ax.loglog(dt_normalized, error_mid * (dt_normalized/dt_mid)**1, 
         'k--', alpha=0.3, linewidth=1, label='1st order')
ax.loglog(dt_normalized, error_mid * (dt_normalized/dt_mid)**2, 
         'k:', alpha=0.3, linewidth=1, label='2nd order')

ax.set_xlabel(r'Timestep $\Delta t$ (mean free times)', fontsize=12)
ax.set_ylabel(r'$L^2$ Error in $T$', fontsize=12)
ax.set_title('Material Temperature Convergence', fontsize=13)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
show('convergence_study.png', close_after=True)
print("Saved plot to 'convergence_study.pdf'")

print(f"\n{'='*80}")
print("Convergence study complete!")
print(f"{'='*80}")
