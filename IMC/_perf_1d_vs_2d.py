#!/usr/bin/env python3
"""Performance comparison: 1D slab IMC vs 2D xy/rz IMC on equivalent problems.

Both runs use:
  - nx=100 spatial cells along the propagation axis
  - ny=1 in the transverse direction (2D with a single layer)
  - Same Ntarget, Nboundary, Nmax, dt, and material model
  - Constant opacity + warm start to avoid cold-material pathologies

The 2D grid has one cell in the degenerate direction so the physics is
identical to 1D.  Any remaining wall-clock difference is pure algorithmic
overhead (boundary distance computations, 2D array indexing, etc.).
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import IMC1D as imc1d
import IMC2D as imc2d

# ── problem parameters ────────────────────────────────────────────────────────
L          = 0.20        # cm, domain length
NX         = 100         # cells in propagation direction
NY         = 1           # always 1 for this test (degenerate 2D)
NTARGET    = 10_000
NBOUNDARY  = 10_000
NMAX       = 60_000
DT         = 0.01        # ns
N_STEPS    = 20          # steps to time (after JIT warm-up)

# Constant opacity + linear EOS → no temperature-dependent opacity divergence.
# T_init = T_bc = 1.0 means material starts near equilibrium.
SIGMA_A    = 10.0        # cm^{-1}  (moderately opaque)
CV_VAL     = 0.3
T_INIT     = 1.0         # keV  (warm start — avoids cold-material explosion)
T_BC       = 1.0         # keV  (boundary temperature)

def sigma_a_f(T):
    return np.full_like(T, SIGMA_A)

def eos(T):     return CV_VAL * T
def inv_eos(u): return u / CV_VAL
def cv(T):      return 0.0 * T + CV_VAL


# ── 1D setup ─────────────────────────────────────────────────────────────────

def run_1d(n_warmup=1):
    x_edges = np.linspace(0.0, L, NX + 1)
    mesh    = np.column_stack([x_edges[:-1], x_edges[1:]])
    Tinit   = np.full(NX, T_INIT)
    Trinit  = np.full(NX, T_INIT)
    source  = np.zeros(NX)
    T_bc    = (T_BC, 0.0)

    state = imc1d.init_simulation(NTARGET, Tinit, Trinit, mesh, eos, inv_eos,
                                   geometry='slab')

    # warm-up (includes numba compilation)
    for _ in range(n_warmup):
        state, _ = imc1d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, mesh, sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True), geometry='slab')

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        state, _ = imc1d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, mesh, sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True), geometry='slab')
    elapsed = time.perf_counter() - t0
    return elapsed, state


# ── 2D xy setup (x=propagation, y=degenerate) ────────────────────────────────

def run_2d_xy(n_warmup=1):
    x_edges = np.linspace(0.0, L, NX + 1)
    y_edges = np.linspace(0.0, L / NX, NY + 1)   # single thin strip
    Tinit   = np.full((NX, NY), T_INIT)
    Trinit  = np.full((NX, NY), T_INIT)
    source  = np.zeros((NX, NY))
    # T_boundary = (left, right, bottom, top)
    T_bc    = (T_BC, 0.0, 0.0, 0.0)

    state = imc2d.init_simulation(NTARGET, Tinit, Trinit, x_edges, y_edges,
                                   eos, inv_eos, geometry='xy')

    # warm-up (includes numba compilation)
    for _ in range(n_warmup):
        state, _ = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, x_edges, y_edges,
                               sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True, True, True),
                               geometry='xy')

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        state, _ = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, x_edges, y_edges,
                               sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True, True, True),
                               geometry='xy')
    elapsed = time.perf_counter() - t0
    return elapsed, state


# ── 2D rz setup (z=propagation, r=degenerate) ────────────────────────────────

def run_2d_rz(n_warmup=1):
    r0      = 10.0                                 # large radius → planar limit
    r_edges = np.linspace(r0, r0 + L / NX, NY + 1)
    z_edges = np.linspace(0.0, L, NX + 1)
    Tinit   = np.full((NY, NX), T_INIT)
    Trinit  = np.full((NY, NX), T_INIT)
    source  = np.zeros((NY, NX))
    # T_boundary = (rmin, rmax, zmin, zmax)
    T_bc    = (0.0, 0.0, T_BC, 0.0)

    state = imc2d.init_simulation(NTARGET, Tinit, Trinit, r_edges, z_edges,
                                   eos, inv_eos, geometry='rz')

    for _ in range(n_warmup):
        state, _ = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, r_edges, z_edges,
                               sigma_a_f, inv_eos, cv, source,
                               reflect=(True, True, False, True),
                               geometry='rz')

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        state, _ = imc2d.step(state, NTARGET, NBOUNDARY, 0, NMAX,
                               T_bc, DT, r_edges, z_edges,
                               sigma_a_f, inv_eos, cv, source,
                               reflect=(True, True, False, True),
                               geometry='rz')
    elapsed = time.perf_counter() - t0
    return elapsed, state


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        from numba import get_num_threads
        n_threads = get_num_threads()
    except Exception:
        n_threads = 1

    print('=' * 60)
    print(f'IMC performance: 1D slab  vs  2D xy  vs  2D rz')
    print(f'  NX={NX}, NY={NY}, Ntarget={NTARGET}, Nmax={NMAX}')
    print(f'  dt={DT} ns, N_STEPS={N_STEPS} timed steps (after warm-up)')
    print(f'  Numba threads: {n_threads}')
    print('=' * 60)

    print('\nRunning 1D slab (warm-up + timing)...')
    t1d, s1d = run_1d(n_warmup=2)

    print('\nRunning 2D xy  (warm-up + timing)...')
    t2d_xy, s2d_xy = run_2d_xy(n_warmup=2)

    print('\nRunning 2D rz  (warm-up + timing)...')
    t2d_rz, s2d_rz = run_2d_rz(n_warmup=2)

    print()
    print('=' * 60)
    print('Results:')
    print(f'  1D  slab : {t1d:.3f} s  ({t1d/N_STEPS*1000:.1f} ms/step)')
    print(f'  2D  xy   : {t2d_xy:.3f} s  ({t2d_xy/N_STEPS*1000:.1f} ms/step)   ratio vs 1D: {t2d_xy/t1d:.2f}x')
    print(f'  2D  rz   : {t2d_rz:.3f} s  ({t2d_rz/N_STEPS*1000:.1f} ms/step)   ratio vs 1D: {t2d_rz/t1d:.2f}x')
    print('=' * 60)

    # Quick sanity check: wave front should be roughly mid-domain after N_STEPS
    x_mid = 0.5 * (np.linspace(0.0, L, NX + 1)[:-1] + np.linspace(0.0, L, NX + 1)[1:])
    T1d = s1d.temperature
    T2d_xy = s2d_xy.temperature[:, 0]  # collapse degenerate y dimension
    T2d_rz = s2d_rz.temperature[0, :]  # collapse degenerate r dimension

    l2_xy  = np.linalg.norm(T2d_xy - T1d) / max(np.linalg.norm(T1d), 1e-30)
    l2_rz  = np.linalg.norm(T2d_rz - T1d) / max(np.linalg.norm(T1d), 1e-30)
    print(f'\nTemperature profile relative L2 error vs 1D:')
    print(f'  2D xy: {l2_xy:.4f}   (expect <0.15 for noisy MC)')
    print(f'  2D rz: {l2_rz:.4f}   (expect <0.15 for noisy MC)')
