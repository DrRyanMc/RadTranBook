#!/usr/bin/env python3
"""Time each phase of one rz step with exact MarshakWave2D.py parameters.

Runs with _timing=True so we see per-phase printouts even if a step hangs.
Uses max_events=10**6 (the argparse default from MarshakWave2D.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T):      return CV_VAL * T
def inv_eos(u):  return u / CV_VAL
def cv(T):       return 0.0*T + CV_VAL

NTARGET   = 24000
NBOUNDARY = 12000
NMAX      = 120000
DT        = 0.01
R0, DR, NR, NZ = 10.0, 0.2, 10, 60

r_edges = np.linspace(R0, R0+DR, NR+1)
z_edges = np.linspace(0.0, 0.2, NZ+1)
Tinit   = np.full((NR, NZ), 1e-4)
source  = np.zeros((NR, NZ))

# ── Warm up JIT with a single small-grid rz step (max_events=100) ────────────
print("Warming up JIT with rz step (max_events=100)...", flush=True)
t0 = time.perf_counter()
s_warm = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                                r_edges, z_edges, eos, inv_eos, geometry='rz')
s_warm, _ = imc2d.step(s_warm, NTARGET, NBOUNDARY, 0, NMAX,
    (0.,0.,1.,0.), DT, r_edges, z_edges,
    sigma_a_f, inv_eos, cv, source,
    reflect=(True,True,False,True), geometry='rz',
    max_events_per_particle=100)
print(f"JIT warmup done in {time.perf_counter()-t0:.2f}s", flush=True)

# ── Fresh state for baseline (max_events=100) ────────────────────────────────
print("\n=== Step with max_events=100 ===", flush=True)
state_100 = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                                   r_edges, z_edges, eos, inv_eos, geometry='rz')
t0 = time.perf_counter()
#step 3 times
for step in range(10):
     print(f"\n  -- step {step+1} (max_events=100) --", flush=True)
     state_100, _ = imc2d.step(state_100, NTARGET, NBOUNDARY, 0, NMAX,
         (0.,0.,1.,0.), DT, r_edges, z_edges,
         sigma_a_f, inv_eos, cv, source,
         reflect=(True,True,False,True), geometry='rz',
         max_events_per_particle=100, _timing=True)
print(f"\nBaseline (max_events=100) done in {time.perf_counter()-t0:.2f}s", flush=True)
print(f"  total: {time.perf_counter()-t0:.3f}s", flush=True)

# ── Same state for max_events=10^6 ──────────────────────────────────────────
print("\n=== Step with max_events=10^6 ===", flush=True)
state_1M = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                                  r_edges, z_edges, eos, inv_eos, geometry='rz')
t0 = time.perf_counter()

#step 3 times
for step in range(10):
    print(f"\n  -- step {step+1} (max_events=10^6) --", flush=True)
    state_1M, _ = imc2d.step(state_1M, NTARGET, NBOUNDARY, 0, NMAX,
    (0.,0.,1.,0.), DT, r_edges, z_edges,
    sigma_a_f, inv_eos, cv, source,
    reflect=(True,True,False,True), geometry='rz',
    max_events_per_particle=10**6, _timing=True)
print(f"  total: {time.perf_counter()-t0:.3f}s", flush=True)

print("\nDone.", flush=True)
