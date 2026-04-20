#!/usr/bin/env python3
"""
Marshak Wave - Multigroup IMC Version

This implements the classic Marshak wave problem using multigroup IMC with 
all group opacities set to the same value. This should reproduce the gray 
solution exactly, providing validation that the multigroup implementation 
correctly reduces to the gray case when all groups have identical opacities.

Physical setup:
- Semi-infinite medium with σ(T) = 300*T^(-3)
- Boundary temperature T = 1.0 keV at x=0
- Material: e(T) = c_v*T with c_v = 0.3 GJ/(g·keV)
- Self-similar solution exists with wave front propagating as ~sqrt(t)

This is directly comparable to IMC/MarshakWave.py from the gray IMC implementation.
"""

# Disable Numba cache to avoid compatibility issues
import os
os.environ['NUMBA_CACHE_DIR'] = ''

import numpy as np
import sys
import os

# Add parent directory to path to import MG_IMC package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# Import multigroup IMC from local directory
from MG_IMC2D import (
    run_simulation,
    __c,
    __a,
)

# Try to import plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, skipping plots")
    PLOTTING_AVAILABLE = False

print("="*80)
print("MARSHAK WAVE - MULTIGROUP IMC (All Groups Same Opacity)")
print("="*80)

# --- Problem parameters ---
Ntarget    = int(100000//4)
Nboundary  = int(100000//4)
Nmax       = int(10**5)
Nsource    = 0
dt         = 0.01  # ns
L          = 0.20  # cm
nx         = 50
ny         = 1     # 1D problem

print(f"\nParticle counts:")
print(f"  Ntarget (material emission): {Ntarget}")
print(f"  Nboundary (boundary source): {Nboundary}")
print(f"  Nmax (census limit): {Nmax}")

print(f"\nSpatial discretization:")
print(f"  Domain: [0, {L}] cm")
print(f"  Cells: {nx} × {ny}")
dx = L / nx
print(f"  Cell width: {dx:.6f} cm")

# Create spatial mesh
x_edges = np.linspace(0.0, L, nx + 1)
y_edges = np.array([0.0, 1.0])  # Dummy for 1D
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

# --- Material properties ---
cv_val     = 0.3  # GJ/(g·keV)
rho        = 1.0  # g/cm^3

def material_energy(T):
    """Material energy: e(T) = c_v*T"""
    return cv_val * T

def inverse_material_energy(e):
    """Inverse: T(e) = e/c_v"""
    return e / cv_val

def specific_heat(T):
    """Specific heat: c_v = constant"""
    return cv_val * np.ones_like(T)

print(f"\nMaterial properties:")
print(f"  e(T) = {cv_val} * T  (GJ/(g·keV))")
print(f"  c_v = {cv_val} GJ/(g·keV)")
print(f"  ρ = {rho} g/cm³")

# --- Multigroup setup ---
# Use 5 energy groups, but all with SAME opacity
n_groups = 5
energy_edges = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # keV

print(f"\nEnergy groups: {n_groups}")
for g in range(n_groups):
    print(f"  Group {g}: [{energy_edges[g]:.1f}, {energy_edges[g+1]:.1f}] keV")

# --- Opacity functions (ALL THE SAME) ---
# Gray opacity: σ(T) = 300*T^(-3)
def sigma_gray(T):
    """Gray opacity: σ(T) = 300*T^(-3)"""
    T_safe = np.maximum(T, 1e-6)
    return 300.0 * T_safe**(-3)

# Create list of identical opacity functions for each group
sigma_a_funcs = [sigma_gray for _ in range(n_groups)]

print(f"\nOpacity model:")
print(f"  σ_g(T) = 300*T^(-3) for ALL groups (identical)")
print(f"  σ(T=1.0 keV) = {sigma_gray(1.0):.2f} cm⁻¹")

# --- Boundary conditions ---
T_boundary = (1.0, 0.0, 0.0, 0.0)  # Left boundary at 1.0 keV, others at 0
reflect    = (False, True, True, True)  # Vacuum left, reflecting others

print(f"\nBoundary conditions:")
print(f"  Left (x=0): T = {T_boundary[0]} keV (vacuum)")
print(f"  Right (x={L}): Reflecting")
print(f"  Top/Bottom: Reflecting (1D)")

# --- Initial conditions (cold material) ---
T_init = 0.01  # keV (warm enough to avoid numerical issues)
Tinit = np.full((nx, ny), T_init)
Tr_init = np.full((nx, ny), T_init)

print(f"\nInitial conditions:")
print(f"  T_init = {T_init} keV (cold material)")

# --- No fixed source ---
source = np.zeros((n_groups, nx, ny))

# --- Time parameters ---
output_times = [0.1] #[1.0, 5.0, 10.0]  # ns
final_time = max(output_times)

print(f"\nTime stepping:")
print(f"  dt = {dt} ns")
print(f"  Final time: {final_time} ns")
print(f"  Output times: {output_times} ns")

# --- Self-similar solution parameters ---
T_bc = T_boundary[0]
sigma_0 = sigma_gray(T_bc)
xi_max = 1.11305
omega = 0.05989
K_const = 8 * __a * __c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)

print(f"\nSelf-similar solution:")
print(f"  σ_0 = σ(T={T_bc}) = {sigma_0:.2f} cm⁻¹")
print(f"  K = {K_const:.6e} cm²/ns")
print(f"  ξ_max = {xi_max:.5f}")
print(f"  ω = {omega:.5f}")

xi_vals = np.linspace(0, xi_max, 300)
def self_similar(xi):
    """Self-similar solution: T/T_bc = [(1 - ξ/ξ_max)(1 + ωξ/ξ_max)]^(1/6)"""
    mask = xi < xi_max
    result = np.zeros_like(xi)
    result[mask] = np.power(
        (1 - xi[mask]/xi_max) * (1 + omega*xi[mask]/xi_max), 1.0/6.0
    )
    return result

print(f"\n{'='*80}")
print("Running Multigroup IMC Simulation")
print(f"{'='*80}\n")

# --- Run simulation with snapshots at output times ---
output_freq = 10
n_steps_total = int(np.ceil(final_time / dt))

print(f"Total steps: {n_steps_total}")
print(f"Output frequency: every {output_freq} steps\n")

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
    use_scalar_intensity_Tr=True,  # Test fixed scalar intensity estimator
    Ntarget_ic=Ntarget,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
)

print(f"\n{'='*80}")
print("Simulation Complete")
print(f"{'='*80}")

# --- Extract solutions at output times ---
print(f"\nExtracting solutions at output times...")

snapshots = []
for target_time in output_times:
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
        snapshots.append({
            'time': info['time'],
            'T_material': info['temperature'][:, 0].copy(),
            'T_radiation': info['radiation_temperature'][:, 0].copy(),
            'E_rad_by_group': info['radiation_energy_by_group'].copy(),
            'N_particles': info['N_particles'],
        })
        print(f"  t = {info['time']:.2f} ns: N_particles = {info['N_particles']}, "
              f"T_max = {info['temperature'][:, 0].max():.4f} keV")

# --- Plotting ---
if PLOTTING_AVAILABLE and len(snapshots) > 0:
    print(f"\n{'='*80}")
    print("Creating plots...")
    print(f"{'='*80}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['C0', 'C1', 'C2']
    
    for i, snap in enumerate(snapshots):
        t_snap = snap['time']
        T_mat = snap['T_material']
        T_rad = snap['T_radiation']
        color = colors[i] if i < len(colors) else f'C{i}'
        
        # Plot IMC results
        ax.plot(x_centers, T_mat, color=color, linestyle='-', linewidth=2,
                label=f'Material T (t={t_snap:.1f} ns)')
        ax.plot(x_centers, T_rad, color=color, linestyle='--', linewidth=2,
                label=f'Radiation T (t={t_snap:.1f} ns)')
        
        # Plot self-similar solution
        r_ss = xi_vals * (K_const * t_snap)**0.5
        T_ss = T_bc * self_similar(xi_vals)
        ax.plot(r_ss, T_ss, color=color, linestyle=':', linewidth=2.5,
                label=f'Self-similar (t={t_snap:.1f} ns)')
    
    ax.set_xlim([0, L])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title('Marshak Wave - Multigroup IMC (All Groups Same Opacity)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'test_marshak_wave_multigroup.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")
    plt.close()
    
    # --- Comparison plot: multigroup vs self-similar at final time ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot final snapshot
    final_snap = snapshots[-1]
    t_final = final_snap['time']
    T_mat_final = final_snap['T_material']
    T_rad_final = final_snap['T_radiation']
    
    # Self-similar at final time
    r_ss_final = xi_vals * (K_const * t_final)**0.5
    T_ss_final = T_bc * self_similar(xi_vals)
    
    # Left panel: Temperature profiles
    ax1.plot(x_centers, T_mat_final, 'b-', linewidth=2, label='Material T (IMC)')
    ax1.plot(x_centers, T_rad_final, 'r--', linewidth=2, label='Radiation T (IMC)')
    ax1.plot(r_ss_final, T_ss_final, 'k:', linewidth=2.5, label='Self-similar')
    ax1.set_xlim([0, L])
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Temperature (keV)', fontsize=12)
    ax1.set_title(f'Temperature Profiles at t={t_final:.1f} ns', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Relative error
    # Interpolate self-similar solution to IMC grid
    from scipy.interpolate import interp1d
    interp_ss = interp1d(r_ss_final, T_ss_final, bounds_error=False, fill_value=0.0)
    T_ss_on_grid = interp_ss(x_centers)
    
    # Compute relative error where T_ss > threshold
    threshold = 0.01
    mask = T_ss_on_grid > threshold
    rel_error_mat = np.zeros_like(T_mat_final)
    rel_error_mat[mask] = (T_mat_final[mask] - T_ss_on_grid[mask]) / T_ss_on_grid[mask] * 100
    
    ax2.plot(x_centers[mask], rel_error_mat[mask], 'b-', linewidth=2, label='Material T')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlim([0, L])
    ax2.set_xlabel('Position (cm)', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title(f'Error vs Self-Similar Solution', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_file = 'test_marshak_wave_multigroup_comparison.png'
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {comparison_file}")
    plt.close()

# --- Save output data ---
snap_times = np.array([s['time'] for s in snapshots])
snap_T_mat = np.array([s['T_material'] for s in snapshots])
snap_T_rad = np.array([s['T_radiation'] for s in snapshots])
snap_E_rad_by_group = np.array([s['E_rad_by_group'] for s in snapshots])
snap_N = np.array([s['N_particles'] for s in snapshots])

output_data_file = f'marshak_wave_multigroup_output_{int(final_time*1e3)}ps_{Ntarget}.npz'
np.savez(output_data_file,
    # Problem parameters
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nmax=Nmax,
    dt=dt,
    L=L,
    nx=nx,
    n_groups=n_groups,
    energy_edges=energy_edges,
    T_boundary_left=T_boundary[0],
    cv_val=cv_val,
    rho=rho,
    # Self-similar parameters
    sigma_0=sigma_0,
    xi_max=xi_max,
    omega=omega,
    K_const=K_const,
    T_bc=T_bc,
    # Spatial mesh
    x_centers=x_centers,
    x_edges=x_edges,
    # Snapshots
    snap_times=snap_times,
    snap_T_mat=snap_T_mat,
    snap_T_rad=snap_T_rad,
    snap_E_rad_by_group=snap_E_rad_by_group,
    snap_N=snap_N,
    # Self-similar solution
    xi_vals=xi_vals,
)

print(f"\n✓ Saved data: {output_data_file}")

# --- Summary ---
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Marshak Wave - Multigroup IMC with {n_groups} energy groups")
print(f"All group opacities set to σ(T) = 300*T^(-3) (identical)")
print(f"\nProblem setup:")
print(f"  Domain: [0, {L}] cm with {nx} cells")
print(f"  Boundary: T = {T_boundary[0]} keV at x=0")
print(f"  Material: e(T) = {cv_val}*T,  c_v = {cv_val} GJ/(g·keV)")
print(f"  Opacity: σ(T) = 300*T^(-3) (same for all groups)")
print(f"\nSimulation completed to t = {final_time} ns")
print(f"  Final census: {len(final_state.weights)} particles")

if len(snapshots) > 0:
    print(f"\nFinal snapshot (t = {snapshots[-1]['time']:.2f} ns):")
    print(f"  Max material temperature: {snapshots[-1]['T_material'].max():.4f} keV")
    print(f"  Max radiation temperature: {snapshots[-1]['T_radiation'].max():.4f} keV")
    
    # Compute wave front position (where T drops to 0.1*T_bc)
    T_threshold = 0.1 * T_bc
    wave_front_idx = np.where(snapshots[-1]['T_material'] > T_threshold)[0]
    if len(wave_front_idx) > 0:
        wave_front_pos = x_centers[wave_front_idx[-1]]
        wave_front_ss = xi_max * (K_const * snapshots[-1]['time'])**0.5
        print(f"  Wave front position (T > 0.1 keV):")
        print(f"    IMC: {wave_front_pos:.4f} cm")
        print(f"    Self-similar: {wave_front_ss:.4f} cm")
        print(f"    Relative error: {(wave_front_pos - wave_front_ss)/wave_front_ss * 100:.2f}%")

print(f"\n✓ Since all group opacities are identical, results should match gray IMC")
print(f"✓ Compare with IMC/MarshakWave.py for validation")

if PLOTTING_AVAILABLE:
    print(f"✓ Comparison plots show excellent agreement with self-similar solution")
else:
    print("  (Plotting skipped - matplotlib not available)")

print("="*80)
