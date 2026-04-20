#!/usr/bin/env python3
"""
0-D Energy Conservation Diagnostic Test

This test checks fundamental energy conservation properties for multigroup IMC
in a 0-D infinite medium (single cell with all reflecting boundaries).

Checks:
1. Is sum(b_g * sigma_g) = sigma_P? (Planck mean opacity)
2. Does sum of emitted particle weights = V*a*c*T^4*dt*sigma_P? 
3. With reflecting boundaries, is total energy exactly conserved?
"""

import os
os.environ['NUMBA_CACHE_DIR'] = ''

import numpy as np
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from MG_IMC2D import __c, __a
from planck_integrals import Bg

print("="*80)
print("0-D ENERGY CONSERVATION DIAGNOSTIC TEST")
print("="*80)

# --- Problem setup ---
# Simple 0-D problem: single cell, gray opacities
sigma_val = 1.0  # cm^-1, same for all groups
cv_val = 0.01    # GJ/(g*keV)

# Energy groups
energy_edges = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # keV
n_groups = len(energy_edges) - 1

# Initial conditions
T = 0.5   # keV (material temperature)
Tr = 1.0  # keV (radiation temperature)
dt = 0.025  # ns

# Cell geometry 
L = 0.1  # cm (cell size)
V = L * 1.0  # cm^3 (volume, with unit width in y)

print(f"\nProblem setup:")
print(f"  Temperature: T = {T} keV")
print(f"  Radiation temperature: Tr = {Tr} keV")
print(f"  Cell volume: V = {V} cm³")
print(f"  Time step: dt = {dt} ns")
print(f"  Opacity: σ = {sigma_val} cm⁻¹ (all groups)")
print(f"  Groups: {n_groups}")
for g in range(n_groups):
    print(f"    Group {g}: [{energy_edges[g]:.1f}, {energy_edges[g+1]:.1f}] keV")

# --- CHECK 1: Planck mean opacity ---
print(f"\n{'='*80}")
print("CHECK 1: Planck Mean Opacity")
print(f"{'='*80}")

# Compute b_g for each group
b_g = np.zeros(n_groups)
for g in range(n_groups):
    E_low = energy_edges[g]
    E_high = energy_edges[g + 1]
    b_g[g] = Bg(E_low, E_high, T)

# Verify b_g sums to 1
sum_bg = np.sum(b_g)
print(f"\nPlanck fractions b_g at T = {T} keV:")
for g in range(n_groups):
    print(f"  Group {g}: b_{g} = {b_g[g]:.6f}")
print(f"\nSum of b_g = {sum_bg:.10f}  (should be 1.0)")

if not np.isclose(sum_bg, 1.0, rtol=1e-6):
    print(f"⚠ WARNING: b_g does not sum to 1!")
else:
    print(f"✓ b_g sums to 1.0")

# Compute Planck mean opacity
sigma_g = np.full(n_groups, sigma_val)
sigma_P = np.sum(b_g * sigma_g)

print(f"\nPlanck mean opacity:")
print(f"  σ_P = Σ(b_g × σ_g) = {sigma_P:.6f} cm⁻¹")
print(f"  Expected (gray): {sigma_val:.6f} cm⁻¹")

if np.isclose(sigma_P, sigma_val, rtol=1e-6):
    print(f"✓ σ_P matches gray opacity")
else:
    print(f"❌ ERROR: σ_P does not match!")

# --- CHECK 2: Emission accounting ---
print(f"\n{'='*80}")
print("CHECK 2: Material Emission Accounting")
print(f"{'='*80}")

# Fleck factor (for theta=1.0)
theta = 1.0
beta = 4.0 * __a * T**3 / cv_val
f = 1.0 / (1.0 + theta * beta * sigma_P * __c * dt)
f = np.clip(f, 0.0, 1.0)

print(f"\nFleck factor calculation:")
print(f"  β = 4aT³/c_v = {beta:.6e}")
print(f"  f = 1/(1 + θβσ_Pct) = {f:.6f}")

# Effective opacity after Fleck factor
sigma_a_eff = sigma_val * f
sigma_s_eff = sigma_val * (1.0 - f)

print(f"\nEffective opacities:")
print(f"  σ_a (absorption) = {sigma_a_eff:.6f} cm⁻¹")
print(f"  σ_s (scattering) = {sigma_s_eff:.6f} cm⁻¹")
print(f"  σ_a + σ_s = {sigma_a_eff + sigma_s_eff:.6f} cm⁻¹")

# Total emission energy (gray formula)
E_emit_gray = __a * __c * T**4 * sigma_val * dt * V
E_emit_effective = __a * __c * T**4 * sigma_a_eff * dt * V

print(f"\nMaterial emission energy:")
print(f"  E_emit (if σ_true used) = acT⁴σΔtV = {E_emit_gray:.6e} GJ")
print(f"  E_emit (if σ_a=fσ used) = acT⁴(fσ)ΔtV = {E_emit_effective:.6e} GJ")

# Emission by group
emission_by_group = sigma_a_eff * b_g * __a * __c * T**4 * dt * V
total_emission = np.sum(emission_by_group)

print(f"\nEmission by group:")
for g in range(n_groups):
    print(f"  Group {g}: {emission_by_group[g]:.6e} GJ  ({100*b_g[g]:.2f}%)")
print(f"\nTotal emission = {total_emission:.6e} GJ")

if np.isclose(total_emission, E_emit_effective, rtol=1e-10):
    print(f"✓ Sum of group emissions matches expected")
else:
    print(f"❌ ERROR: Emission accounting mismatch!")
    print(f"   Difference: {abs(total_emission - E_emit_effective):.6e} GJ")

# --- CHECK 3: Run simulation and verify energy conservation ---
print(f"\n{'='*80}")
print("CHECK 3: Energy Conservation in Simulation")
print(f"{'='*80}")

from MG_IMC2D import run_simulation

# Mesh setup
x_edges = np.array([0.0, L])
y_edges = np.array([0.0, 1.0])  # Unit width
nx, ny = 1, 1

# Initial conditions
Tinit = np.full((nx, ny), T)
Trinit = np.full((nx, ny), Tr)
T_boundary = [0.0, 0.0, 0.0, 0.0]  # No boundary sources

# Opacity functions (gray)
sigma_a_funcs = [lambda T_val, s=sigma_val: s + 0*T_val for _ in range(n_groups)]

# Material properties
eos = lambda T_val: cv_val * T_val
inv_eos = lambda e: e / cv_val
cv_func = lambda T_val: cv_val

# No sources
def source_func(t, dt):
    return ("boundary", None)

# Simulation parameters
Ntarget = 1000
Nboundary = 0
Nsource = 0
Nmax = 10**6
n_steps = 3  # Just a few steps
final_time = dt * n_steps

print(f"\nRunning {n_steps} timesteps...")
print(f"  Ntarget = {Ntarget}")
print(f"  All boundaries reflecting (closed system)")
print(f"  use_scalar_intensity_Tr = False (use particle binning)")

history, final_state = run_simulation(
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nsource=Nsource,
    Nmax=Nmax,
    Tinit=Tinit,
    Tr_init=Trinit,
    T_boundary=T_boundary,
    dt=dt,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=eos,
    inv_eos=inv_eos,
    cv=cv_func,
    source=source_func,
    final_time=final_time,
    reflect=(True, True, True, True),  # All reflecting
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=False,  # Use particle binning for accuracy
    Ntarget_ic=Ntarget,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
)

print(f"\n{'Time':>10} {'N':>8} {'E_total':>12} {'E_internal':>12} {'E_rad':>12} {'Lost':>12}")
print("-"*80)

E_initial = None
for i, h in enumerate(history):
    t = h['time']
    N = h['N_particles']
    E_int = h['total_internal_energy']
    E_rad = h['total_radiation_energy']
    E_total = E_int + E_rad
    E_lost = h.get('energy_loss', 0.0)
    
    if E_initial is None:
        E_initial = E_total
    
    print(f"{t:10.6f} {N:8d} {E_total:12.6e} {E_int:12.6e} {E_rad:12.6e} {E_lost:12.6e}")

E_final = history[-1]['total_internal_energy'] + history[-1]['total_radiation_energy']
E_change = E_final - E_initial
E_change_pct = 100 * E_change / E_initial

print(f"\n{'='*80}")
print("ENERGY CONSERVATION SUMMARY")
print(f"{'='*80}")
print(f"Initial total energy:  {E_initial:.10e} GJ")
print(f"Final total energy:    {E_final:.10e} GJ")
print(f"Change in energy:      {E_change:.10e} GJ  ({E_change_pct:+.6f}%)")
print(f"Lost energy (cumulative): {history[-1].get('energy_loss', 0.0):.10e} GJ")

# Energy conservation tolerance
tol = 1e-10  # Absolute tolerance in GJ
if abs(E_change) < tol:
    print(f"\n✓ PASSED: Energy conserved within tolerance {tol:.1e} GJ")
else:
    print(f"\n❌ FAILED: Energy not conserved!")
    print(f"   Tolerance: {tol:.1e} GJ")
    print(f"   Actual change: {abs(E_change):.1e} GJ")

# Additional diagnostics
print(f"\n{'='*80}")
print("ADDITIONAL DIAGNOSTICS")
print(f"{'='*80}")

# Check if energy is monotonically increasing
energies = [h['total_internal_energy'] + h['total_radiation_energy'] for h in history]
is_increasing = all(energies[i+1] >= energies[i] for i in range(len(energies)-1))
if is_increasing:
    print("⚠ Total energy is monotonically increasing (should be constant)")
else:
    print("Total energy has fluctuations (Monte Carlo noise expected)")

# Check particle count
N_initial = history[0]['N_particles']
N_final = history[-1]['N_particles']
print(f"\nParticle evolution:")
print(f"  Initial: {N_initial}")
print(f"  Final:   {N_final}")
print(f"  Growth:  {N_final - N_initial} particles")

# Material emission check
print(f"\nExpected emission per step: {E_emit_effective:.6e} GJ")
print(f"Observed energy increase: {E_change / n_steps:.6e} GJ per step")

if abs(E_change / n_steps - 0.0) < tol:
    print("✓ No energy change (as expected for closed system)")
else:
    ratio = (E_change / n_steps) / E_emit_effective
    print(f"⚠ Energy increasing at {ratio:.2%} of emission rate")

print()
