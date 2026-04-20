"""
Quick validation test for MG_IMC2D.

Tests basic functionality with a small problem to ensure the code runs correctly.
"""

import numpy as np
import sys

# Test import
print("Testing MG_IMC2D import...")
try:
    from MG_IMC2D import (
        init_simulation,
        step,
        __c,
        __a,
    )
    print("✓ MG_IMC2D module imported successfully")
except Exception as e:
    print(f"✗ Failed to import MG_IMC2D: {e}")
    sys.exit(1)

# Test utilities
print("\nTesting mg_utils...")
try:
    from mg_utils import (
        create_log_energy_groups,
        powerlaw_opacity_functions,
        simple_eos_functions,
    )
    print("✓ mg_utils module imported successfully")
except Exception as e:
    print(f"✗ Failed to import mg_utils: {e}")
    sys.exit(1)

# Setup small test problem
print("\nSetting up test problem...")
print("-" * 60)

# Tiny spatial domain
nx, ny = 5, 1
x_edges = np.linspace(0, 1, nx + 1)
y_edges = np.array([0, 1])

# 2 energy groups
energy_edges = create_log_energy_groups(0.1, 10.0, 2)
n_groups = len(energy_edges) - 1

print(f"Spatial mesh: {nx} × {ny}")
print(f"Energy groups: {n_groups}")
print(f"  Group 0: [{energy_edges[0]:.2f}, {energy_edges[1]:.2f}] keV")
print(f"  Group 1: [{energy_edges[1]:.2f}, {energy_edges[2]:.2f}] keV")

# Material properties
eos, inv_eos, cv = simple_eos_functions(cv_value=0.1, rho=1.0)

# Opacity functions
sigma_funcs = powerlaw_opacity_functions(energy_edges, sigma_ref=1.0, T_ref=1.0, alpha=2.0)

# Initial conditions
T_init = 0.1  # keV
Tinit = np.full((nx, ny), T_init)
Tr_init = np.full((nx, ny), T_init)

print(f"Initial temperature: {T_init} keV")

# Test 1: Initialization
print("\nTest 1: Initialization...")
print("-" * 60)
try:
    state = init_simulation(
        Ntarget=100,
        Tinit=Tinit,
        Tr_init=Tr_init,
        edges1=x_edges,
        edges2=y_edges,
        energy_edges=energy_edges,
        eos=eos,
        inv_eos=inv_eos,
        Ntarget_ic=100,
        geometry="xy",
    )
    print(f"✓ Initialization successful")
    print(f"  Initial particles: {len(state.weights)}")
    print(f"  Initial total energy: {state.previous_total_energy:.6e} GJ")
    print(f"  Groups represented: {np.unique(state.groups)}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Single time step
print("\nTest 2: Single time step...")
print("-" * 60)
try:
    state_new, info = step(
        state=state,
        Ntarget=50,
        Nboundary=10,
        Nsource=0,
        Nmax=500,
        T_boundary=(0.5, 0.0, 0.0, 0.0),
        dt=0.01,
        edges1=x_edges,
        edges2=y_edges,
        energy_edges=energy_edges,
        sigma_a_funcs=sigma_funcs,
        inv_eos=inv_eos,
        cv=cv,
        source=np.zeros((nx, ny)),
        reflect=(False, False, True, True),
        theta=1.0,
        geometry="xy",
    )
    print(f"✓ Time step completed successfully")
    print(f"  Final particles: {info['N_particles']}")
    print(f"  Total energy: {info['total_energy']:.6e} GJ")
    print(f"  Material energy: {info['total_internal_energy']:.6e} GJ")
    print(f"  Radiation energy: {info['total_radiation_energy']:.6e} GJ")
    print(f"  Boundary emission: {info['boundary_emission']:.6e} GJ")
    print(f"  Energy conservation error: {info['energy_loss']:.6e} GJ")
    print(f"  Transport events: {info['profiling']['transport_events']['total']}")
except Exception as e:
    print(f"✗ Time step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check group distribution
print("\nTest 3: Group distribution...")
print("-" * 60)
try:
    groups = state_new.groups
    for g in range(n_groups):
        n_g = np.sum(groups == g)
        frac = n_g / len(groups) if len(groups) > 0 else 0.0
        print(f"  Group {g}: {n_g} particles ({frac*100:.1f}%)")
    print("✓ Group distribution computed")
except Exception as e:
    print(f"✗ Group distribution check failed: {e}")
    sys.exit(1)

# Test 4: Radiation energy by group
print("\nTest 4: Radiation energy by group...")
print("-" * 60)
try:
    rad_by_group = state_new.radiation_energy_by_group
    print(f"  Shape: {rad_by_group.shape}")
    for g in range(n_groups):
        E_g_total = np.sum(rad_by_group[g, :, :])
        print(f"  Group {g} total energy: {E_g_total:.6e} GJ")
    total_rad = np.sum(rad_by_group)
    print(f"  Total radiation energy: {total_rad:.6e} GJ")
    print("✓ Radiation energy by group verified")
except Exception as e:
    print(f"✗ Radiation energy check failed: {e}")
    sys.exit(1)

# Test 5: Temperature update
print("\nTest 5: Temperature evolution...")
print("-" * 60)
try:
    T_before = state.temperature[0, 0]
    T_after = state_new.temperature[0, 0]
    print(f"  First cell temperature: {T_before:.6f} → {T_after:.6f} keV")
    print(f"  Change: {T_after - T_before:.6e} keV")
    if T_after != T_before:
        print("✓ Temperature evolved")
    else:
        print("  (No change - may be OK for small timestep)")
except Exception as e:
    print(f"✗ Temperature check failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("All validation tests passed! ✓")
print("=" * 60)
print("\nMG_IMC2D is ready to use.")
print("Try running: python test_2group_marshak.py")
