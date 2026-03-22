#!/usr/bin/env python3
"""Run 5 consecutive rz steps to see how per-step timing evolves with the wave front.

With max_events=10^6 (MarshakWave2D.py default), compare vs max_events=100.
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
NSTEPS    = 10

r_edges = np.linspace(R0, R0+DR, NR+1)
z_edges = np.linspace(0.0, 0.2, NZ+1)
Tinit   = np.full((NR, NZ), 1e-4)
source  = np.zeros((NR, NZ))

def run_steps(max_events, nsteps=NSTEPS, timing=True):
    state = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                                   r_edges, z_edges, eos, inv_eos, geometry='rz')
    for step in range(nsteps):
        print(f"\n  -- step {step+1} (max_events={max_events}) --", flush=True)
        t0 = time.perf_counter()
        state, info = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
            (0.,0.,1.,0.), DT, r_edges, z_edges,
            sigma_a_f, inv_eos, cv, source,
            reflect=(True,True,False,True), geometry='rz',
            max_events_per_particle=max_events,
            _timing=timing)
        print(f"  => total: {time.perf_counter()-t0:.3f}s  N={info['N_particles']}  T_max={info['temperature'].max():.4f}", flush=True)

# Warm up JIT first
print("Warming up JIT...", flush=True)
t0 = time.perf_counter()
s_w = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                              r_edges, z_edges, eos, inv_eos, geometry='rz')
s_w, _ = imc2d.step(s_w, NTARGET, NBOUNDARY, 0, NMAX,
    (0.,0.,1.,0.), DT, r_edges, z_edges,
    sigma_a_f, inv_eos, cv, source,
    reflect=(True,True,False,True), geometry='rz',
    max_events_per_particle=100)
print(f"Done in {time.perf_counter()-t0:.2f}s", flush=True)

print("\n\n=== max_events=100 ===")
run_steps(100)

print("\n\n=== max_events=10^6 ===")
run_steps(10**6)
