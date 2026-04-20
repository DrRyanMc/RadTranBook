#!/usr/bin/env python3
"""Quick test to verify use_scalar_intensity_Tr fix"""

import os
os.environ['NUMBA_CACHE_DIR'] = ''

import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MG_IMC2D import run_simulation, __c, __a

print("Testing use_scalar_intensity_Tr fix...")
print("="*60)

# Simple 2-group problem - just run a few steps
x_edges = np.linspace(0.0, 0.1, 20)
y_edges = np.array([0.0, 0.01])
energy_edges = np.array([0.1, 1.0, 5.0])  # 2 groups

nx, ny = len(x_edges) - 1, len(y_edges) - 1
n_groups = len(energy_edges) - 1

# Simple constant opacity functions
sigma_0 = 100.0
sigma_a_funcs = [lambda T, i=i: sigma_0 for i in range(n_groups)]

# Initial cold material
T_init = 0.01  # keV
Tinit = np.full((nx, ny), T_init)
Tr_init = np.full((nx, ny), T_init)
T_boundary = [1.0, 0.0, 0.0, 0.0]  # Left boundary hot

c_v = 0.3  # GJ/(g·keV)
material_energy = lambda T: c_v * T
inverse_material_energy = lambda e: e / c_v
specific_heat = lambda T: c_v  # Constant cv

# Boundary source
def source(t, dt):
    return ("boundary", None)

Ntarget = 10000
Nboundary = 10000
Nsource = 0
Nmax = 40000

print(f"Domain: {nx}x{ny} cells, {n_groups} groups")
print(f"Running 5 timesteps with Ntarget={Ntarget}...")
print()

history, final_state = run_simulation(
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nsource=Nsource,
    Nmax=Nmax,
    Tinit=Tinit,
    Tr_init=Tr_init,
    T_boundary=T_boundary,
    dt=0.01,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=material_energy,
    inv_eos=inverse_material_energy,
    cv=specific_heat,
    source=source,
    final_time=0.05,  # 5 steps
    reflect=(False, True, True, True),
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=True,
    Ntarget_ic=Ntarget,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=100000,
)

print()
print("="*60)
print("ENERGY CONSERVATION CHECK")
print("="*60)

# Extract energy history from history
times = [h['time'] for h in history]
total_internal = [h['total_internal_energy'] for h in history]
total_radiation = [h['total_radiation_energy'] for h in history]
N_particles = [h['N_particles'] for h in history]
total_energy = [total_internal[i] + total_radiation[i] for i in range(len(times))]

print(f"{'Time':>10} {'N':>10} {'Total E':>12} {'Internal E':>12} {'Rad E':>12}")
for i in range(len(times)):
    print(f"{times[i]:10.6f} {N_particles[i]:10d} {total_energy[i]:12.6f} "
          f"{total_internal[i]:12.6f} {total_radiation[i]:12.6f}")

# Check for negative energies
if any(e < 0 for e in total_internal):
    print("\n❌ FAILED: Negative internal energy detected!")
    sys.exit(1)

# Check particle conservation
initial_N = N_particles[0]
final_N = N_particles[-1]
particle_loss = (initial_N - final_N) / initial_N if initial_N > 0 else 0
if particle_loss > 0.5:
    print(f"\n❌ FAILED: Massive particle loss: {particle_loss*100:.1f}%")
    sys.exit(1)

# Check energy monotonicity (total energy should increase or stay constant with boundary source)
energy_increase = total_energy[-1] - total_energy[0]
if energy_increase < -1e-6:  # Allow small numerical noise
    print(f"\n❌ FAILED: Total energy decreased significantly!")
    sys.exit(1)

print("\n✓ PASSED: use_scalar_intensity_Tr feature working correctly!")
print(f"  - No negative energies")
print(f"  - Particle loss: {particle_loss*100:.1f}%")
print(f"  - Energy increase: {energy_increase:.6f} GJ")
print()
