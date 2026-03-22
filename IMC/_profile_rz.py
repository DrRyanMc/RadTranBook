#!/usr/bin/env python3
"""Profile one rz step with the exact parameters used by MarshakWave2D.py."""
import sys, os, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import IMC2D as imc2d

CV_VAL = 0.3
def sigma_a_f(T): return 300.0 * np.maximum(T, 1e-4)**-3
def eos(T):      return CV_VAL * T
def inv_eos(u):  return u / CV_VAL
def cv(T):       return 0.0*T + CV_VAL

# MarshakWave2D.py defaults (from main() argparse and run_rz_planar_limit_check)
NTARGET  = 20000
NBOUNDARY = 12000
NMAX     = 120000
DT       = 0.01
MAX_EVENTS = 10**6
R0, DR, NR, NZ = 10.0, 0.2, 10, 60

r_edges = np.linspace(R0, R0+DR, NR+1)
z_edges = np.linspace(0.0, 0.2, NZ+1)
Tinit   = np.full((NR, NZ), 1e-4)
source  = np.zeros((NR, NZ))

# ── Warm up the JIT with the same call signature (max_events=10^6) ──────────
print("Warming up JIT with xy step (max_events=10^6)...", flush=True)
x_e = np.linspace(0.0, 0.2, 61); y_e = np.linspace(0.0, 0.2, 61)
s_xy = imc2d.init_simulation(NTARGET, np.full((60,60),1e-4), np.full((60,60),1e-4),
                              x_e, y_e, eos, inv_eos, geometry='xy')
s_xy, _ = imc2d.step(s_xy, NTARGET, NBOUNDARY, 0, NMAX, (1.,0.,0.,0.), DT,
                     x_e, y_e, sigma_a_f, inv_eos, cv, np.zeros((60,60)),
                     reflect=(False,True,True,True), geometry='xy',
                     max_events_per_particle=MAX_EVENTS)
print("Warmup done.", flush=True)

# ── Set up rz state ──────────────────────────────────────────────────────────
state = imc2d.init_simulation(NTARGET, Tinit.copy(), Tinit.copy(),
                               r_edges, z_edges, eos, inv_eos, geometry='rz')

# One unprofiledwarm-up rz step so the state is representative of step 2+
state, _ = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
    (0.,0.,1.,0.), DT, r_edges, z_edges,
    sigma_a_f, inv_eos, cv, source,
    reflect=(True,True,False,True), geometry='rz',
    max_events_per_particle=MAX_EVENTS)
print("rz step 1 (unprofiledwarm-up) done.", flush=True)

# ── Profile step 2 ───────────────────────────────────────────────────────────
print("Profiling step 2...", flush=True)
pr = cProfile.Profile()
pr.enable()
state, info = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
    (0.,0.,1.,0.), DT, r_edges, z_edges,
    sigma_a_f, inv_eos, cv, source,
    reflect=(True,True,False,True), geometry='rz',
    max_events_per_particle=MAX_EVENTS)
pr.disable()

print(f"Step 2 done. N={info['N_particles']}, T_max={state.temperature.max():.4f}", flush=True)

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
