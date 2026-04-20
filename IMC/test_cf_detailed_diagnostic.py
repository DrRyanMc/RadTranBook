#!/usr/bin/env python3
"""Detailed diagnostic to understand CF absorption behavior"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import IMC2D_CarterForest as imc2d_cf

# Simple Marshak wave material model
CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4) ** -3
def eos(T): return CV_VAL * T
def inv_eos(u): return u / CV_VAL
def cv(T): return 0.0 * T + CV_VAL

# Small mesh
n = 20
x_edges = np.linspace(0.0, 0.1, n + 1)
y_edges = np.linspace(0.0, 0.1, n + 1)

Tinit = np.full((n, n), 1e-4)
Trinit = np.full((n, n), 1e-4)
source = np.zeros((n, n))

print('[CF Detailed Diagnostic]')
print('='*60)

state = imc2d_cf.init_simulation(
    5000, Tinit, Trinit, x_edges, y_edges, eos, inv_eos, geometry='xy'
)

cell_vol = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

print(f'Initial radiation energy: {np.sum(state.weights):.6e} GJ')
print(f'Initial material energy:  {np.sum(state.internal_energy) * cell_vol:.6e} GJ')

# Run one step
T_boundary = (1.0, 0.0, 0.0, 0.0)
reflect = (False, True, True, True)
dt = 0.01

state, info = imc2d_cf.step(
    state, 5000, 2000, 0, 20000, T_boundary, dt, 
    x_edges, y_edges, sigma_a_f, inv_eos, cv, source,
    reflect=reflect, geometry='xy', max_events_per_particle=1000
)

print(f'\nAfter step:')
print(f'  Boundary emission:       {info["boundary_emission"]:.6e} GJ')
print(f'  Boundary loss:           {info["boundary_loss"]:.6e} GJ')
print(f'  Final radiation energy:  {info["total_radiation_energy"]:.6e} GJ')
print(f'  Final material energy:   {info["total_internal_energy"]:.6e} GJ')
print(f'  Energy conservation:     {info["energy_loss"]:.6e} GJ')

# Check transport statistics
prof = info.get('profiling', {})
tevents = prof.get('transport_events', {})
print(f'\nTransport Events:')
print(f'  Total events:            {tevents.get("total", 0)}')
print(f'  Boundary crossings:      {tevents.get("boundary_crossings", 0)}')
print(f'  Absorption+continue:     {tevents.get("absorption_continue_events", 0)}')
print(f'  Absorption+capture:      {tevents.get("absorption_capture_events", 0)}')
print(f'  Census events:           {tevents.get("census_events", 0)}')

# Calculate what SHOULD have been emitted by material
sigma_a = sigma_a_f(state.temperature)
beta = 4.0 * imc2d_cf.__a * np.maximum(state.temperature, 1e-12) ** 3 / cv(state.temperature)
rate = imc2d_cf.__c * sigma_a * beta

expected_emission = np.zeros_like(state.temperature)
mask = (rate > 1e-12) & (beta > 1e-15)
expected_emission[mask] = imc2d_cf.__a * state.temperature[mask]**4 * (1.0 - np.exp(-rate[mask] * dt)) / beta[mask]
expected_emission[~mask] = imc2d_cf.__a * state.temperature[~mask]**4 * imc2d_cf.__c * sigma_a[~mask] * dt

total_expected_emission = np.sum(expected_emission) * cell_vol

print(f'\nMaterial Emission Analysis:')
print(f'  Expected material emission: {total_expected_emission:.6e} GJ')
print(f'  T_material average:         {state.temperature.mean():.6f} keV')

ratio_captured = tevents.get("absorption_capture_events", 0) / max(tevents.get("absorption_continue_events", 0) + tevents.get("absorption_capture_events", 0), 1)
print(f'\nCapture Ratio:')
print(f'  Fraction of absorptions that capture: {ratio_captured:.4f}')
print(f'  (Should be positive if material is heating)')
