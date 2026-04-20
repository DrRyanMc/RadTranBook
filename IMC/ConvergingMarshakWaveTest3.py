#!/usr/bin/env python3
"""
Converging Marshak Wave Test 3 — Implicit Monte Carlo, spherical geometry.

Companion to:
  nonEquilibriumDiffusion/problems/converging_marshak_wave_test3.py  (diffusion)
  IMC/ConvergingMarshakWave.py                                        (test 1, IMC)

Uses IMC1D.py with geometry='spherical'.  All physical parameters and the
analytic self-similar solution are reproduced from the diffusion companion.

Physical setup
--------------
  Domain:    r ∈ [0, R],  R = 0.001 cm (10 μm)
  Geometry:  spherical
  Density:   ρ(r) = (r/cm)^{−ω}  g/cm³,  ω = 0.45
  Opacity:   k_t = 10³ (T/HeV)^{−3.5} ρ(r)^{1.4}   cm⁻¹
  Energy:    u(T,r) = 10⁻³ (T/HeV)² ρ(r)^{0.75}    GJ/cm³

Because the density is spatially varying, the material-property functions
(opacity, eos, inv_eos, cv) are built as closures over the cell-midpoint
radii computed from the mesh.  IMC1D.py requires each function to accept
a single array argument (T or u, length = n_cells) and return an array of
the same length, which the closures provide.

Similarity parameters (from converging_marshak_wave_test3.py)
--------------------------------------------------------------
  δ     = 1.1157535873060416
  T_amp = 1.198199926365715  HeV
  T_exp = 0.0276392

Time convention
---------------
  Physical time from T_INIT_NS ≈ −7.875084 ns to T_FINAL_NS = −1 ns.
  IMC elapsed time τ = t_phys − T_INIT_NS  (starts at 0).

Units throughout: keV, ns, cm, GJ.
"""

import sys, os
import argparse
import pickle
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IMC1D as imc

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Add utils to path for plotting
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'utils'))
from plotfuncs import show

# =============================================================================
# PHYSICAL CONSTANTS  (match IMC1D)
# =============================================================================

C_LIGHT = imc.__c   # 29.98 cm/ns
A_RAD   = imc.__a   # 0.01372 GJ/(cm³ keV⁴)

# =============================================================================
# PROBLEM PARAMETERS
# =============================================================================

R     = 0.001   # cm  (outer radius = 10 μm)
OMEGA = 0.45    # density exponent:  ρ(r) = r^{-ω}

T_HEV_PER_KEV = 10.0   # 1 keV = 10 HeV

# Precomputed unit-conversion factors
_OPACITY_PREFACTOR = 1e3 * (T_HEV_PER_KEV ** (-3.5))   # = 10^{-0.5}
_ENERGY_PREFACTOR  = 1e-3 * (T_HEV_PER_KEV **  2)      # = 0.1

# Self-similar solution parameters (from converging_marshak_wave_test3.py)
DELTA          = 1.1157535873060416
T_ANALYTIC_AMP = 1.198199926365715     # amplitude in HeV
T_ANALYTIC_EXP = 0.0276392

# Physical time span
T_INIT_NS  = -(10.0 ** (1.0 / DELTA))  # ≈ −7.875084 ns
T_FINAL_NS = -1.0                       # ns

# Times at which to save snapshots for comparison
OUTPUT_TIMES_NS = [-6.591897629554719, -3.926450981261105, -1.0]

# =============================================================================
# DENSITY PROFILE
# =============================================================================

def rho_r(r):
    """ρ(r) = (r/cm)^{−ω}  g/cm³.  Safe at r > 0."""
    return np.maximum(r, 1e-30) ** (-OMEGA)

# =============================================================================
# MATERIAL PROPERTIES  (T in keV, r in cm, positional arrays)
# =============================================================================
# These functions take BOTH (T_keV, r) arrays.  They are wrapped into
# single-argument closures by make_material_funcs() below.

def _opacity_full(T_keV, r):
    """k_t = 10³ (T/HeV)^{−3.5} ρ(r)^{1.4}  cm⁻¹."""
    T_safe = np.maximum(T_keV, 1e-8)
    return _OPACITY_PREFACTOR * T_safe**(-3.5) * rho_r(r)**1.4


def _eos_full(T_keV, r):
    """u(T,r) = 10⁻³ (T/HeV)² ρ(r)^{0.75}  GJ/cm³."""
    return _ENERGY_PREFACTOR * T_keV**2 * rho_r(r)**0.75


def _inv_eos_full(u_GJ, r):
    """T(u,r) [keV] from u [GJ/cm³]."""
    T_keV = np.sqrt(np.maximum(u_GJ, 0.0) / (_ENERGY_PREFACTOR * rho_r(r)**0.75))
    return T_keV


def _cv_full(T_keV, r):
    """du/dT [GJ/(cm³ keV)] = 0.2 T_keV ρ(r)^{0.75}."""
    T_safe = np.maximum(T_keV, 1e-8)
    return 2.0 * _ENERGY_PREFACTOR * T_safe * rho_r(r)**0.75


def make_material_funcs(r_mid):
    """
    Return single-argument closure functions (opacity, eos, inv_eos, cv)
    for use with IMC1D, capturing per-cell midpoint radii r_mid.

    Each returned function takes a 1-D array of length n_cells.
    """
    r = r_mid.copy()   # snapshot to avoid accidental mutation

    def opacity(T_keV):
        return _opacity_full(T_keV, r)

    def eos(T_keV):
        return _eos_full(T_keV, r)

    def inv_eos(u_GJ):
        return _inv_eos_full(u_GJ, r)

    def cv(T_keV):
        return _cv_full(T_keV, r)

    return opacity, eos, inv_eos, cv

# =============================================================================
# ANALYTIC SELF-SIMILAR SOLUTION
# =============================================================================

def _W_scalar(xi):
    """Similarity profile W(ξ) — scalar version."""
    if xi >= 2.0:
        return (xi - 1.0)**0.210071 * (1.27048 - 0.0470724*xi + 0.00179721*xi**2)
    elif xi > 1.0:
        return (xi - 1.0)**0.357506 * (1.9792  - 0.619497 *xi + 0.110644  *xi**2)
    else:
        return 0.0

_W = np.vectorize(_W_scalar)


def _V_scalar(xi):
    """Similarity profile V(ξ) for Test 3 — scalar version.
    
    V(ξ) = 0.8879 ξ^{-2.233} + 0.2278 ξ^{-1.037}
    """
    return 0.8879 * xi**(-2.233) + 0.2278 * xi**(-1.037)


_V = np.vectorize(_V_scalar)


def xi_rt(r, t_ns):
    """Similarity coordinate  ξ(r, t) = (r / 10⁻⁴ cm) / (−t)^δ."""
    return (r / 1e-4) / (-t_ns)**DELTA


def T_analytic_keV(r, t_ns):
    """Analytic temperature profile [keV] at (r, t_ns).
    
    This is the interior temperature solution (for comparison).
    """
    xi    = xi_rt(r, t_ns)
    T_HeV = T_ANALYTIC_AMP * (-t_ns)**T_ANALYTIC_EXP * _W(xi)**0.5
    return T_HeV / T_HEV_PER_KEV


def _Lambda(xi):
    """Λ(ξ) = ξ^0.6625 V(ξ) W^{-1}(ξ) for Test 3 (equation 8.88)."""
    W_val = _W_scalar(xi)
    V_val = _V_scalar(xi)
    if W_val < 1e-30:
        return 0.0
    return (xi ** 0.6625) * V_val * (W_val ** (-1.0))


def T_bath_keV(t_ns):
    """Bath (outer boundary) temperature [keV] from equation (8.88).
    
    T_bath(t) = [1 + 0.075821 Λ(ξ_R(t)) (-t/ns)^{-0.316092}]^{1/4} T_s(t)
    
    where Λ(ξ) = ξ^0.6625 V(ξ) W^{-1}(ξ)
    """
    t_clamped = max(float(t_ns), -1e30)  # ensure negative
    xi_R = (R / 1e-4) / (-t_clamped)**DELTA
    
    # Surface temperature (interior solution at r=R)
    T_s_HeV = T_ANALYTIC_AMP * (-t_clamped)**T_ANALYTIC_EXP * _W_scalar(xi_R)**0.5
    T_s_keV = T_s_HeV / T_HEV_PER_KEV
    
    # Compute bath temperature correction
    Lambda_R = _Lambda(xi_R)
    correction = 1.0 + 0.075821 * Lambda_R * (-t_clamped) ** (-0.316092)
    correction = max(correction, 0.0) ** 0.25  # [...]^{1/4}
    
    return max(T_s_keV * correction, 1e-8)

# =============================================================================
# TIME-DEPENDENT OUTER BOUNDARY TEMPERATURE
# =============================================================================

def outer_T_keV(tau_elapsed):
    """
    Surface temperature [keV] as a function of elapsed simulation time τ [ns].

    τ = t_phys − T_INIT_NS, so t_phys = T_INIT_NS + τ.
    Uses the bath temperature formula combining W(ξ) and V(ξ).
    """
    t_phys = T_INIT_NS + tau_elapsed
    return T_bath_keV(t_phys)

# =============================================================================
# CHECKPOINT / RESTART HELPERS
# =============================================================================

CHECKPOINT_VERSION = 1


def _serialize_state(state):
    return {
        'weights':               state.weights,
        'mus':                   state.mus,
        'times':                 state.times,
        'positions':             state.positions,
        'cell_indices':          state.cell_indices,
        'internal_energy':       state.internal_energy,
        'temperature':           state.temperature,
        'radiation_temperature': state.radiation_temperature,
        'time':                  float(state.time),
        'previous_total_energy': float(state.previous_total_energy),
        'count':                 int(state.count),
    }


def _deserialize_state(data):
    return imc.SimulationState(
        weights=data['weights'],
        mus=data['mus'],
        times=data['times'],
        positions=data['positions'],
        cell_indices=data['cell_indices'],
        internal_energy=data['internal_energy'],
        temperature=data['temperature'],
        radiation_temperature=data['radiation_temperature'],
        time=float(data['time']),
        previous_total_energy=float(data['previous_total_energy']),
        count=int(data['count']),
    )


def save_checkpoint(path, state, step_count, snapshots, output_saved,
                    n_cells, dt, final_output_time, save_prefix):
    payload = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'n_cells':            int(n_cells),
        'dt':                 float(dt),
        'final_output_time':  float(final_output_time),
        'save_prefix':        str(save_prefix),
        'step_count':         int(step_count),
        'state':              _serialize_state(state),
        'output_saved':       sorted(float(t) for t in output_saved),
        'snapshots': [
            (float(t), T_mid.copy(), r_mid.copy())
            for (t, T_mid, r_mid) in snapshots
        ],
        'np_random_state': np.random.get_state(),
        'py_random_state': random.getstate(),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    version = payload.get('checkpoint_version', None)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f'Unsupported checkpoint version: {version} (expected {CHECKPOINT_VERSION})'
        )
    return payload

# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(snapshots, save_prefix='converging_marshak_wave_test3_imc'):
    """
    Save two figures:
      _T.png  — temperature profiles (HeV) vs analytic
      _u.png  — material energy density (10¹³ erg/cm³) vs analytic
    """
    r_anal = np.linspace(1e-7, R, 2000)
    colors = ['tab:blue', 'tab:orange', 'tab:red']

    # --- Temperature (HeV) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for (t_snap, T_keV, r_mid), col in zip(snapshots, colors):
        label   = f't = {t_snap:.2f} ns'
        T_HeV   = T_keV * T_HEV_PER_KEV
        ax.plot(r_mid / 1e-4, T_HeV, color=col, lw=2, label=label)
        T_ref = np.vectorize(lambda r: T_analytic_keV(r, t_snap))(r_anal) * T_HEV_PER_KEV
        ax.plot(r_anal / 1e-4, T_ref, color=col, lw=1.5, ls='--',
                label=f'analytic ({label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$T$ (HeV)')
    ax.set_xlim([0, R / 1e-4])
    #ax.set_title('Converging Marshak Wave Test 3 — IMC (temperature)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f'{save_prefix}_T.pdf'
    show(fname,close_after=True)
    print(f'Saved: {fname}')
    plt.close()

    # --- Energy density (10¹³ erg/cm³) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for (t_snap, T_keV, r_mid), col in zip(snapshots, colors):
        label = f't = {t_snap:.2f} ns'
        # u [GJ/cm³] × 1e3 → 10¹³ erg/cm³  (1 GJ = 10¹⁶ erg)
        u_sim = _eos_full(T_keV, r_mid) * 1e3
        ax.plot(r_mid / 1e-4, u_sim/1e3, color=col, lw=2, label=label)
        u_ref = _eos_full(
            np.vectorize(lambda r: T_analytic_keV(r, t_snap))(r_anal),
            r_anal) * 1e3
        ax.plot(r_anal / 1e-4, u_ref/1e3, color=col, lw=1.5, ls='--',
                label=f'analytic ({label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$e(T)$ (GJ/cm$^3$)')
    ax.set_xlim([0, R / 1e-4])
    #ax.set_title('Converging Marshak Wave Test 3 — IMC (energy density)')
    #ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f'{save_prefix}_u.pdf'
    show(fname,close_after=True)
    print(f'Saved: {fname}')
    plt.close()

# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run(n_cells=50, Ntarget=5000, Nboundary=2000, NMax=20000,
        dt=0.1, output_freq=10,
        save_prefix='converging_marshak_wave_test3_imc',
        final_output_time=None, checkpoint_path=None,
        checkpoint_every_steps=0, checkpoint_every_ns=None,
    restart_from=None, profile=False):
    """
    Run Converging Marshak Wave Test 3 with IMC in spherical geometry.

    Parameters
    ----------
    n_cells       : radial cells in [0, R]
    Ntarget       : target emission particles per step
    Nboundary     : particles injected from outer boundary per step
    NMax          : maximum census particles after combing
    dt            : time step [ns]
    output_freq   : diagnostic print cadence (steps)
    save_prefix   : prefix for PNG/NPZ output files
    final_output_time : stop at this physical time [ns]
    checkpoint_path   : filepath for checkpoint pickle
    checkpoint_every_steps : write checkpoint every N steps (0 disables)
    checkpoint_every_ns    : write checkpoint every elapsed ns (None disables)
    restart_from  : checkpoint filepath to restart from (None = fresh start)
    """
    if final_output_time is None:
        final_output_time = T_FINAL_NS

    duration = final_output_time - T_INIT_NS
    output_elapsed = sorted(t - T_INIT_NS
                            for t in OUTPUT_TIMES_NS
                            if t <= final_output_time + 1e-9)

    print('=' * 70)
    print('Converging Marshak Wave Test 3  (IMC / spherical / ρ∝r^{-ω})')
    print('=' * 70)
    print(f'  n_cells={n_cells},  Ntarget={Ntarget},  '
          f'Nboundary={Nboundary},  NMax={NMax}')
    print(f'  dt={dt} ns,  total elapsed={duration:.4f} ns')
    print(f'  ω={OMEGA},  R={R} cm')
    print(f'  T_INIT_NS={T_INIT_NS:.6f} ns,  T_FINAL_NS={final_output_time} ns')
    print('=' * 70)

    # --- Mesh: uniform radial shells ---
    r_edges = np.linspace(0.0, R, n_cells + 1)
    mesh    = np.column_stack([r_edges[:-1], r_edges[1:]])
    r_mid   = 0.5 * (mesh[:, 0] + mesh[:, 1])

    # Build single-argument material closures capturing per-cell geometry
    opacity, eos, inv_eos, cv = make_material_funcs(r_mid)

    # --- Initial condition: cold uniform material ---
    T_init_keV = 1e-4
    T_init     = np.full(n_cells, T_init_keV)
    Tr_init    = np.full(n_cells, T_init_keV)

    # --- Boundary conditions ---
    #   inner (r=0): reflecting
    #   outer (r=R): time-dependent Dirichlet
    T_boundary = (0.0, outer_T_keV)
    source     = np.zeros(n_cells)

    if checkpoint_path is None:
        checkpoint_path = f'{save_prefix}_checkpoint.pkl'

    output_t_ns = sorted(t for t in OUTPUT_TIMES_NS
                         if t <= final_output_time + 1e-9)

    # --- Initialise IMC state or restart from checkpoint ---
    if restart_from is not None:
        ckpt = load_checkpoint(restart_from)
        if int(ckpt['n_cells']) != n_cells:
            raise ValueError(
                f'Checkpoint n_cells={ckpt["n_cells"]} != run n_cells={n_cells}'
            )
        state        = _deserialize_state(ckpt['state'])
        step_count   = int(ckpt['step_count'])
        output_saved = set(float(t) for t in ckpt.get('output_saved', []))
        snapshots    = [
            (float(t), np.asarray(T_mid), np.asarray(rm))
            for (t, T_mid, rm) in ckpt.get('snapshots', [])
        ]
        if 'np_random_state' in ckpt:
            np.random.set_state(ckpt['np_random_state'])
        if 'py_random_state' in ckpt:
            random.setstate(ckpt['py_random_state'])
        print(f'Restarted from checkpoint: {restart_from}')
        print(f'  elapsed={state.time:.6f} ns  '
              f'physical={T_INIT_NS + state.time:.6f} ns  '
              f'step={step_count}')
    else:
        state        = imc.init_simulation(
            Ntarget, T_init, Tr_init, mesh, eos, inv_eos,
            geometry='spherical')
        snapshots    = []
        output_saved = set()
        step_count   = 0

    last_checkpoint_time = state.time
    if profile:
        phase_totals = np.zeros(4)
        event_totals = np.zeros(6, dtype=np.int64)
        transported_particles_total = 0

    # --- Time-stepping loop ---
    while state.time < duration - 1e-12:
        step_dt = min(dt, duration - state.time)
        for tau_out in output_elapsed:
            if tau_out > state.time and state.time + step_dt > tau_out + 1e-12:
                step_dt = tau_out - state.time
                break

        state, info = imc.step(
            state, Ntarget, Nboundary, 0, NMax,
            T_boundary, step_dt, mesh, opacity, inv_eos, cv, source,
            reflect=(True, False),
            geometry='spherical')

        step_count += 1
        t_phys = T_INIT_NS + state.time

        if profile and 'profiling' in info:
            phase = info['profiling']['phase_times_s']
            events = info['profiling']['transport_events']
            phase_totals[0] += phase['sampling']
            phase_totals[1] += phase['transport']
            phase_totals[2] += phase['postprocess']
            phase_totals[3] += phase['total']
            event_totals[0] += events['total']
            event_totals[1] += events['boundary_crossings']
            event_totals[2] += events['scatter_events']
            event_totals[3] += events['census_events']
            event_totals[4] += events['weight_floor_kills']
            event_totals[5] += events['reflections']
            transported_particles_total += events['n_particles_transported']

        if step_count % output_freq == 0 or step_count <= 2:
            T_bath_HeV = T_bath_keV(t_phys) * T_HEV_PER_KEV
            T_surf_HeV = T_analytic_keV(R, t_phys) * T_HEV_PER_KEV
            msg = (f'  step {step_count:5d}  t_phys={t_phys:9.4f} ns'
                   f'  T_surf={T_surf_HeV:.4f} HeV'
                   f'  T_bath={T_bath_HeV:.4f} HeV'
                   f'  T_center={state.temperature[0]*T_HEV_PER_KEV:.4f} HeV'
                   f'  N={info["N_particles"]:6d}'
                   f'  ΔE={info["energy_loss"]:.3e} GJ')
            if profile and 'profiling' in info:
                p = info['profiling']
                msg += (f'  t_step={p["phase_times_s"]["total"]:.3e}s'
                        f'  ev/part={p["transport_events"]["avg_events_per_particle"]:.2f}')
            print(msg)

        # Save snapshot when we hit an output time
        for t_out in output_t_ns:
            if t_out not in output_saved and abs(t_phys - t_out) < 1e-9:
                snapshots.append((t_out, state.temperature.copy(), r_mid.copy()))
                output_saved.add(t_out)
                T_max_HeV = state.temperature.max() * T_HEV_PER_KEV
                print(f'  >> Snapshot at t_phys={t_out:.4f} ns'
                      f'  T_max={T_max_HeV:.4f} HeV')

        should_checkpoint = False
        if checkpoint_every_steps and (step_count % checkpoint_every_steps == 0):
            should_checkpoint = True
        if (checkpoint_every_ns is not None
                and (state.time - last_checkpoint_time) >= checkpoint_every_ns - 1e-12):
            should_checkpoint = True

        if should_checkpoint:
            save_checkpoint(
                checkpoint_path, state, step_count, snapshots, output_saved,
                n_cells=n_cells, dt=dt, final_output_time=final_output_time,
                save_prefix=save_prefix)
            last_checkpoint_time = state.time
            print(f'  >> Checkpoint saved: {checkpoint_path}'
                  f'  (elapsed={state.time:.4f} ns)')

    print(f'\nDone. Total steps: {step_count}, '
          f'final t_phys={T_INIT_NS + state.time:.4f} ns')

    save_checkpoint(
        checkpoint_path, state, step_count, snapshots, output_saved,
        n_cells=n_cells, dt=dt, final_output_time=final_output_time,
        save_prefix=save_prefix)
    print(f'Final checkpoint saved: {checkpoint_path}')

    if profile and step_count > 0:
        avg_phase = phase_totals / step_count
        global_avg_events = event_totals[0] / max(transported_particles_total, 1)
        print('\nProfiling summary (FC):')
        print(f'  Avg step time      : {avg_phase[3]:.6e} s')
        print(f'  Avg sampling time  : {avg_phase[0]:.6e} s')
        print(f'  Avg transport time : {avg_phase[1]:.6e} s')
        print(f'  Avg post time      : {avg_phase[2]:.6e} s')
        print(f'  Total events       : {int(event_totals[0])}')
        print(f'  Boundary crossings : {int(event_totals[1])}')
        print(f'  Scatter events     : {int(event_totals[2])}')
        print(f'  Census events      : {int(event_totals[3])}')
        print(f'  Weight-floor kills : {int(event_totals[4])}')
        print(f'  Reflections        : {int(event_totals[5])}')
        print(f'  Avg events/particle: {global_avg_events:.3f}')

        profile_path = f'{save_prefix}_profile.npz'
        np.savez(profile_path,
                 phase_totals_s=phase_totals,
                 step_count=step_count,
                 event_totals=event_totals,
                 transported_particles_total=transported_particles_total,
                 avg_events_per_particle=global_avg_events)
        print(f'Saved: {profile_path}')

    if snapshots:
        plot_results(snapshots, save_prefix=save_prefix)

        snap_times = np.array([s[0] for s in snapshots])
        snap_T     = np.array([s[1] for s in snapshots])
        snap_r     = np.array([s[2] for s in snapshots])
        npz_path   = f'{save_prefix}.npz'
        np.savez(npz_path,
                 snap_times=snap_times,
                 snap_T_keV=snap_T,
                 snap_r_mid=snap_r,
                 n_cells=n_cells,
                 Ntarget=Ntarget,
                 Nboundary=Nboundary,
                 NMax=NMax,
                 dt=dt,
                 T_INIT_NS=T_INIT_NS,
                 R=R,
                 OMEGA=OMEGA)
        print(f'Saved: {npz_path}')
    else:
        print('No snapshots captured — check output time clipping.')

    return state, snapshots

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Converging Marshak Wave Test 3 IMC driver (spherical, ρ∝r^{-ω}).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n-cells', type=int, default=500,
                        help='Number of radial mesh cells.')
    parser.add_argument('--Ntarget', type=int, default=2*10**5,
                        help='Target particles for internal emission.')
    parser.add_argument('--Nboundary', type=int, default=10**5,
                        help='Boundary-source particles per step.')
    parser.add_argument('--NMax', type=int, default=10**6,
                        help='Maximum census particles after combing.')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step [ns].')
    parser.add_argument('--output-freq', type=int, default=10,
                        help='Diagnostic print cadence (steps).')
    parser.add_argument('--save-prefix', type=str,
                        default='converging_marshak_wave_test3_imc',
                        help='Prefix for PNG/NPZ output files.')
    parser.add_argument('--final-output-time', type=float,
                        default=OUTPUT_TIMES_NS[-1],
                        help='Final physical time [ns] to stop the run.')
    parser.add_argument('--run-full', action='store_true',
                        help='Ignore --final-output-time and run to T_FINAL_NS.')
    parser.add_argument('--checkpoint-path', type=str,
                        default='converging_marshak_wave_test3_imc_checkpoint.pkl',
                        help='Path to checkpoint file for save/restart.')
    parser.add_argument('--checkpoint-every-steps', type=int, default=100,
                        help='Write checkpoint every N steps (0 disables).')
    parser.add_argument('--checkpoint-every-ns', type=float, default=None,
                        help='Write checkpoint every elapsed ns (omit to disable).')
    parser.add_argument('--restart-from', type=str, default=None,
                        help='Restart from this checkpoint path.')
    parser.add_argument('--numba-threads', type=int, default=None,
                        help='Set Numba thread count for parallel transport.')
    parser.add_argument('--profile', action='store_true',
                        help='Collect and print per-step timing/event profiling.')
    return parser


def _configure_numba_threads(num_threads):
    if num_threads is None:
        return
    try:
        from numba import set_num_threads, get_num_threads
    except Exception as exc:
        print(f'Warning: unable to import numba thread controls ({exc}).')
        return
    if num_threads < 1:
        raise ValueError('--numba-threads must be >= 1')
    set_num_threads(num_threads)
    print(f'Numba threads set to {get_num_threads()}')


def main(argv=None):
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    _configure_numba_threads(args.numba_threads)

    final_output_time = T_FINAL_NS if args.run_full else args.final_output_time

    run(
        n_cells=args.n_cells,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        NMax=args.NMax,
        dt=args.dt,
        output_freq=args.output_freq,
        save_prefix=args.save_prefix,
        final_output_time=final_output_time,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every_steps=args.checkpoint_every_steps,
        checkpoint_every_ns=args.checkpoint_every_ns,
        restart_from=args.restart_from,
        profile=args.profile,
    )


if __name__ == '__main__':
    main()
