"""
Example: 2-Group Marshak Wave with Multigroup IMC

This demonstrates the basic usage of MG_IMC2D.py to solve a simple
2-group Marshak wave problem in 1D slab geometry (using 2D code with ny=1).
"""

import numpy as np
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from MG_IMC import (
    run_simulation,
    C_LIGHT as __c,
    A_RAD as __a,
)

# Problem parameters
T_boundary = 1.0  # keV at left boundary
T_init = 0.01  # keV initial temperature
final_time = 1.0  # ns
dt = 0.01  # ns

# Spatial domain (1D slab)
x_min = 0.0
x_max = 5.0  # cm
nx = 50

y_min = 0.0  # Dummy for 1D
y_max = 1.0
ny = 1

x_edges = np.linspace(x_min, x_max, nx + 1)
y_edges = np.linspace(y_min, y_max, ny + 1)

# Energy groups (2 groups for demonstration)
# Group 0: [0.1, 1.0] keV
# Group 1: [1.0, 10.0] keV
energy_edges = np.array([0.1, 1.0, 10.0])
n_groups = len(energy_edges) - 1

# Material properties
rho = 1.0  # g/cm³

def cv_func(T):
    """Constant specific heat"""
    return 0.1 * np.ones_like(T)  # GJ/(g·keV)

def eos(T):
    """Material energy density: e = ρ c_v T"""
    return rho * 0.1 * T  # GJ/cm³

def inv_eos(e):
    """Inverse EOS: T = e / (ρ c_v)"""
    return e / (rho * 0.1)

# Opacity functions for each group
# Simple power-law opacity: σ_g(T) = σ_0,g * T^(-3)
sigma_0 = [300.0, 300.]  # Reference opacities for each group (cm⁻¹)

def sigma_a_group_0(T):
    """Absorption opacity for group 0"""
    return sigma_0[0] * T**(-3)

def sigma_a_group_1(T):
    """Absorption opacity for group 1"""
    return sigma_0[1] * T**(-3)

sigma_a_funcs = [sigma_a_group_0, sigma_a_group_1]

# Initial conditions
Tinit = np.full((nx, ny), T_init)
Tr_init = np.full((nx, ny), T_init)

# Boundary conditions (left boundary hot, others cold/reflecting)
T_boundary_vec = (T_boundary, 0.0, 0.0, 0.0)
reflect = (False, False, True, True)  # Reflect top/bottom for 1D

# Particle counts
Ntarget = 10000  # Particles for material emission
Nboundary = 5000  # Particles per boundary per step
Nsource = 0  # No external source
Nmax = 50000  # Maximum census particles
Ntarget_ic = 10000  # Initial condition particles

# External source (none)
source = np.zeros((nx, ny))

# IMC parameters
theta = 1.0  # Implicit parameter
use_scalar_intensity_Tr = True
conserve_comb_energy = True

print("="*80)
print("2-Group Multigroup IMC Test: Marshak Wave")
print("="*80)
print(f"Spatial domain: [{x_min}, {x_max}] cm with {nx} cells")
print(f"Energy groups: {n_groups} groups")
print(f"  Group 0: [{energy_edges[0]}, {energy_edges[1]}] keV")
print(f"  Group 1: [{energy_edges[1]}, {energy_edges[2]}] keV")
print(f"Boundary temperature: {T_boundary} keV")
print(f"Initial temperature: {T_init} keV")
print(f"Final time: {final_time} ns")
print(f"Time step: {dt} ns")
print(f"Particle counts: Ntarget={Ntarget}, Nboundary={Nboundary}")
print("="*80)
print()

# Run simulation
history, final_state = run_simulation(
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nsource=Nsource,
    Nmax=Nmax,
    Tinit=Tinit,
    Tr_init=Tr_init,
    T_boundary=T_boundary_vec,
    dt=dt,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=eos,
    inv_eos=inv_eos,
    cv=cv_func,
    source=source,
    final_time=final_time,
    reflect=reflect,
    output_freq=1,
    theta=theta,
    use_scalar_intensity_Tr=use_scalar_intensity_Tr,
    Ntarget_ic=Ntarget_ic,
    conserve_comb_energy=conserve_comb_energy,
    geometry="xy",
)

print()
print("="*80)
print("Simulation Complete")
print("="*80)
print(f"Final time: {final_state.time:.4f} ns")
print(f"Final particle count: {len(final_state.weights)}")
print(f"Final total energy: {final_state.previous_total_energy:.6f} GJ")
print()
print("Material temperature profile (first 10 cells):")
T_final = final_state.temperature[:10, 0]
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
x_centers_head = x_centers[:10]
for i in range(10):
    print(f"  x = {x_centers_head[i]:.3f} cm: T = {T_final[i]:.4f} keV")
print()
print("Radiation energy by group (first 5 cells):")
for g in range(n_groups):
    print(f"\nGroup {g} ([{energy_edges[g]:.2f}, {energy_edges[g+1]:.2f}] keV):")
    E_rad_g = final_state.radiation_energy_by_group[g, :5, 0]
    for i in range(5):
        print(f"  x = {x_centers[i]:.3f} cm: E_r = {E_rad_g[i]:.6e} GJ/cm³")

# Optional: Save results
try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature profile
    ax1.plot(x_centers, final_state.temperature[:, 0], 'b-', linewidth=2, label='Material T')
    ax1.plot(x_centers, final_state.radiation_temperature[:, 0], 'r--', linewidth=2, label='Radiation T')
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Temperature (keV)')
    ax1.set_title(f'Temperature Profile at t = {final_state.time:.3f} ns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Radiation energy by group
    for g in range(n_groups):
        E_rad_g = final_state.radiation_energy_by_group[g, :, 0]
        label = f'Group {g} [{energy_edges[g]:.1f}-{energy_edges[g+1]:.1f} keV]'
        ax2.semilogy(x_centers, E_rad_g, linewidth=2, label=label, marker='o', markersize=3)
    
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Radiation Energy Density (GJ/cm³)')
    ax2.set_title('Radiation Energy by Group')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_2group_marshak.png', dpi=150)
    print("Plot saved to test_2group_marshak.png")
    
except ImportError:
    print("Matplotlib not available, skipping plot generation")

print("\nTest complete!")
