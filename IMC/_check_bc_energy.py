#!/usr/bin/env python3
"""Check boundary energy scaling: 1D slab vs 2D xy per unit transverse area."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC1D as imc1d
import IMC2D as imc2d

CV = 0.3
def sigma_a(T): return 300*np.maximum(T, 1e-4)**-3
def eos(T):     return CV*T
def inv_eos(u): return u/CV
def cv(T):      return 0*T+CV

L = 0.2; n = 60; dt = 0.01; T_bc = 1.0

# ── 1D slab ────────────────────────────────────────────────────────────────
mesh_1d = np.column_stack([np.linspace(0,L,n+1)[:-1], np.linspace(0,L,n+1)[1:]])
s1d = imc1d.init_simulation(5000, np.full(n,1e-4), np.full(n,1e-4),
                             mesh_1d, eos, inv_eos, geometry='slab')
_, info1d = imc1d.step(s1d, 5000, 12000, 0, 80000, (T_bc, 0.0), dt, mesh_1d,
                       sigma_a, inv_eos, cv, np.zeros(n),
                       reflect=(False, True), geometry='slab')
bc1d  = info1d['boundary_emission']
exp1d = imc1d.__a * imc1d.__c * T_bc**4 / 4.0 * dt

print(f"1D boundary_emission:         {bc1d:.6e}")
print(f"1D expected (acT^4/4*dt):     {exp1d:.6e}")
print(f"1D ratio actual/expected:     {bc1d/exp1d:.6f}")

# ── 2D xy ──────────────────────────────────────────────────────────────────
x_e = np.linspace(0, L, n+1)
y_e = np.linspace(0, L, n+1)
s2d = imc2d.init_simulation(20000, np.full((n,n),1e-4), np.full((n,n),1e-4),
                             x_e, y_e, eos, inv_eos, geometry='xy')
_, info2d = imc2d.step(s2d, 20000, 12000, 0, 120000,
                       (T_bc, 0., 0., 0.), dt,
                       x_e, y_e, sigma_a, inv_eos, cv, np.zeros((n,n)),
                       reflect=(False, True, True, True), geometry='xy',
                       max_events_per_particle=100)
bc2d       = info2d['boundary_emission']
exp2d      = imc2d.__a * imc2d.__c * T_bc**4 / 4.0 * dt * L
per_area   = bc2d / L

print(f"\n2D boundary_emission:         {bc2d:.6e}")
print(f"2D expected (acT^4/4*dt*L):   {exp2d:.6e}")
print(f"2D ratio actual/expected:     {bc2d/exp2d:.6f}")
print(f"2D per unit transverse area:  {per_area:.6e}")
print(f"\nRatio (2D/L) / 1D:            {per_area/bc1d:.6f}   <- should be 1.0 if scaling is correct")
print(f"\nFor reference: 4/pi = {4.0/np.pi:.6f}  (would show up as ratio if /4 vs /pi mismatch)")
