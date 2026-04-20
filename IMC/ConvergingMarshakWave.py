#!/usr/bin/env python3
"""
Converging Marshak Wave — Implicit Monte Carlo, spherical geometry.

Companion to:
  nonEquilibriumDiffusion/problems/converging_marshak_wave.py  (diffusion)
  IMC/MarshakWave.py                                           (slab IMC)

Uses IMC1D.py with geometry='spherical'.  All physical parameters and the
analytic self-similar solution are reproduced from the diffusion companion.

Physical setup
--------------
  Domain:    r ∈ [0, R],  R = 0.001 cm (10 μm)
  Geometry:  spherical
  Density:   ρ = 19.3 g/cm³
  Opacity:   σ_a = 7200 (T/HeV)^{-1.5} ρ^{1.2}   cm⁻¹
  Energy:    u   = 3.4×10⁻³ (T/HeV)^{1.6} ρ^{0.86}  GJ/cm³

Boundary conditions
-------------------
  r = 0  : reflecting (symmetry)
  r = R  : time-dependent Dirichlet  T_s(t) = self-similar surface temperature

Time convention
---------------
  Physical time runs from T_INIT_NS ≈ -29.626 ns to T_FINAL_NS = -1 ns.
  The IMC solver uses elapsed time  τ = t_phys − T_INIT_NS  starting at 0.

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

RHO = 19.3        # g/cm³
R   = 0.001       # cm  (outer radius = 10 μm)

T_HEV_PER_KEV = 10.0   # 1 keV = 10 HeV

# Self-similar solution parameters (from the paper)
DELTA          = 0.6795011543956738
T_ANALYTIC_AMP = 1.34503465             # amplitude in HeV
T_ANALYTIC_EXP = 0.0920519

# Physical time span
T_INIT_NS  = -(10.0 ** (1.0 / DELTA))  # ≈ -29.6256 ns  (W(ξ_R)→0 here)
T_FINAL_NS = -1.0                       # ns

# Times at which to save snapshots for comparison
OUTPUT_TIMES_NS = [-22.1223094, -9.44842435, -1.0]

# Precomputed density-dependence coefficients
_RHO_OP  = RHO ** 1.2    # opacity density factor
_RHO_EN  = RHO ** 0.86   # energy density factor
_U0_GJ   = 3.4e-3 * _RHO_EN  # coefficient:  u = _U0_GJ * T_HeV^1.6  GJ/cm³

# =============================================================================
# MATERIAL PROPERTIES  (T in keV throughout)
# =============================================================================

def opacity(T_keV):
    """
    Absorption/Rosseland opacity  σ_a(T) [cm⁻¹].
    σ_a = 7200 (T/HeV)^{-1.5} ρ^{1.2}
    """
    T_HeV = np.maximum(T_keV * T_HEV_PER_KEV, 1e-6)
    return 7200.0 * T_HeV**(-1.5) * _RHO_OP


def eos(T_keV):
    """Material internal energy density  u(T)  [GJ/cm³]."""
    T_HeV = T_keV * T_HEV_PER_KEV
    return _U0_GJ * T_HeV**1.6


def inv_eos(u_GJ):
    """Temperature [keV] from energy density [GJ/cm³]."""
    T_HeV = (np.maximum(u_GJ, 0.0) / _U0_GJ) ** 0.625
    return T_HeV / T_HEV_PER_KEV


def cv(T_keV):
    """
    Volumetric specific heat  du/dT  [GJ/(cm³ keV)].
    cv = _U0_GJ × 1.6 × T_HeV^{0.6} × (dT_HeV/dT_keV)
    """
    T_HeV = np.maximum(T_keV * T_HEV_PER_KEV, 1e-6)
    return _U0_GJ * 1.6 * T_HeV**0.6 * T_HEV_PER_KEV

# =============================================================================
# ANALYTIC SELF-SIMILAR SOLUTION
# =============================================================================

def _W_scalar(xi):
    """Similarity profile W(ξ) — scalar version."""
    if xi >= 2.0:
        return (xi - 1.0)**0.295477 * (1.08165 - 0.0271785*xi + 0.00105539*xi**2)
    elif xi > 1.0:
        return (xi - 1.0)**0.40574  * (1.52093 - 0.376185 *xi + 0.0655796 *xi**2)
    else:
        return 0.0

_W = np.vectorize(_W_scalar)


def _V_scalar(xi):
    """Similarity profile V(ξ) for Test 1 — scalar version.
    
    V(ξ) = 0.4345 ξ^{-2.752} + 0.2451 ξ^{-1.454}
    """
    return 0.4345 * xi**(-2.752) + 0.2451 * xi**(-1.454)


_V = np.vectorize(_V_scalar)


def xi_rt(r, t_ns):
    """Similarity coordinate  ξ(r, t) = (r / 0.1R) / (−t)^δ."""
    return (r / (R / 10.0)) / (-t_ns)**DELTA


def T_analytic_keV(r, t_ns):
    """Analytic temperature profile [keV] at (r, t_ns).
    
    This is the interior temperature solution (for comparison).
    """
    xi = xi_rt(r, t_ns)
    T_HeV = T_ANALYTIC_AMP * (-t_ns)**T_ANALYTIC_EXP * _W(xi)**0.625
    return T_HeV / T_HEV_PER_KEV

def _Lambda(xi):
    """Λ(ξ) = ξ^0.6625 V(ξ) W^{-1}(ξ) for Test 3 (equation 8.88)."""
    W_val = _W_scalar(xi)
    V_val = _V_scalar(xi)
    if W_val < 1e-30:
        return 0.0
    return (xi) * V_val * (W_val ** (-1.5))

def T_bath_keV(t_ns):
    """Bath (outer boundary) temperature [keV] from equation (8.88).
    
    T_bath(t) = [1 + 0.075821 Λ(ξ_R(t)) (-t/ns)^{-0.316092}]^{1/4} T_s(t)
    
    where Λ(ξ) = ξ^0.6625 V(ξ) W^{-1}(ξ)
    """
    t_clamped = max(float(t_ns), -1e30)  # ensure negative
    xi_R = (R / 1e-4) / (-t_clamped)**DELTA
    
    # Surface temperature (interior solution at r=R)
    T_s_HeV = T_ANALYTIC_AMP * (-t_clamped)**T_ANALYTIC_EXP * _W_scalar(xi_R)**(5/8)
    T_s_keV = T_s_HeV / T_HEV_PER_KEV
    
    # Compute bath temperature correction
    Lambda_R = _Lambda(xi_R)
    correction = 1.0 + 0.103502 * Lambda_R * (-t_clamped) ** (-0.541423)
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
    """Convert IMC SimulationState to a pickle-safe dict."""
    return {
        'weights': state.weights,
        'mus': state.mus,
        'times': state.times,
        'positions': state.positions,
        'cell_indices': state.cell_indices,
        'internal_energy': state.internal_energy,
        'temperature': state.temperature,
        'radiation_temperature': state.radiation_temperature,
        'time': float(state.time),
        'previous_total_energy': float(state.previous_total_energy),
        'count': int(state.count),
    }


def _deserialize_state(data):
    """Rebuild IMC SimulationState from checkpoint dict."""
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
    """Persist complete simulation state so runs can restart exactly."""
    payload = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'n_cells': int(n_cells),
        'dt': float(dt),
        'final_output_time': float(final_output_time),
        'save_prefix': str(save_prefix),
        'step_count': int(step_count),
        'state': _serialize_state(state),
        'output_saved': sorted(float(t) for t in output_saved),
        'snapshots': [
            (float(t), T_mid.copy(), r_mid.copy())
            for (t, T_mid, r_mid) in snapshots
        ],
        # Save RNG states so restarted Monte Carlo trajectories match bitwise.
        'np_random_state': np.random.get_state(),
        'py_random_state': random.getstate(),
    }

    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    """Load checkpoint payload and basic compatibility checks."""
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

def plot_results(snapshots, save_prefix='converging_marshak_wave_imc'):
    """
    Save two figures:
      _T.png  — temperature profiles vs analytic
      _u.png  — material energy density vs analytic
    snapshots : list of (t_ns, T_keV_array, r_mid_array)
    """
    r_anal = np.linspace(1e-7, R, 2000)
    colors = ['tab:blue', 'tab:orange', 'tab:red']

    # --- Temperature ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for (t_snap, T_mid, r_mid), col in zip(snapshots, colors):
        label = f't = {t_snap:.2f} ns'
        ax.plot(r_mid / 1e-4, T_mid, color=col, lw=2, label=label)
        T_ref = np.vectorize(lambda r: T_analytic_keV(r, t_snap))(r_anal)
        ax.plot(r_anal / 1e-4, T_ref, color=col, lw=1.5, ls='--',
                label=f'analytic ({label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    ax.set_ylabel(r'$T$ (keV)')
    ax.set_xlim([0, R / 1e-4])
    #ax.set_title('Converging Marshak Wave — IMC  (temperature)')
    #ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f'{save_prefix}_T.pdf'
    show(fname,close_after=True)
    print(f'Saved: {fname}')
    plt.close()

    # --- Material energy density ---
    # Unit conversion: 1 GJ/cm³ = 10^16 erg/cm³ → ×1e3 gives 10^13 erg/cm³
    fig, ax = plt.subplots(figsize=(7, 5))
    for (t_snap, T_mid, r_mid), col in zip(snapshots, colors):
        label = f't = {t_snap:.2f} ns'
        u_sim = eos(T_mid) * 1e3   # 10^13 erg/cm³
        ax.plot(r_mid / 1e-4, u_sim/1e3, color=col, lw=2, label=label)
        T_ref = np.vectorize(lambda r: T_analytic_keV(r, t_snap))(r_anal)
        u_ref = eos(T_ref) * 1e3
        ax.plot(r_anal / 1e-4, u_ref/1e3, color=col, lw=1.5, ls='--',
                label=f'analytic ({label})')
    ax.set_xlabel(r'$r$ ($\mu$m)')
    
    ax.set_ylabel(r'$e(T)$ (GJ/cm$^3$)')#ax.set_ylabel(r'$e(T)$  ($10^{13}$ erg/cm$^3$)')
    ax.set_xlim([0, R / 1e-4])
    #ax.set_title('Converging Marshak Wave — IMC  (energy density)')
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
        dt=0.1, output_freq=10, save_prefix='converging_marshak_wave_imc',
    final_output_time=None, checkpoint_path=None,
    checkpoint_every_steps=0, checkpoint_every_ns=None,
    restart_from=None):
    """
    Run the converging Marshak wave with IMC in spherical geometry.

    Parameters
    ----------
    n_cells      : radial cells in [0, R]
    Ntarget      : target number of particles for internal emission per step
    Nboundary    : particles injected from the outer boundary per step
    NMax         : maximum particles kept after combing per step
    dt           : time step size [ns]
    output_freq  : print diagnostics every this many steps
    save_prefix  : prefix for output PNG filenames
    final_output_time : stop at this physical time [ns] (default: T_FINAL_NS)
    checkpoint_path : filepath used for checkpoint I/O
    checkpoint_every_steps : write checkpoint every N steps (0 disables)
    checkpoint_every_ns : write checkpoint every elapsed ns (None disables)
    restart_from : checkpoint filepath to restart from (None = fresh start)
    """
    if final_output_time is None:
        final_output_time = T_FINAL_NS

    duration = final_output_time - T_INIT_NS   # elapsed time to run [ns]
    output_elapsed = sorted(t - T_INIT_NS
                            for t in OUTPUT_TIMES_NS
                            if t <= final_output_time + 1e-9)

    print('=' * 70)
    print('Converging Marshak Wave  (IMC / spherical geometry)')
    print('=' * 70)
    print(f'  n_cells={n_cells},  Ntarget={Ntarget},  '
          f'Nboundary={Nboundary},  NMax={NMax}')
    print(f'  dt={dt} ns,  total elapsed={duration:.4f} ns')
    print(f'  T_INIT_NS={T_INIT_NS:.6f} ns,  T_FINAL_NS={final_output_time} ns')
    print('=' * 70)

    # --- Mesh: uniform radial shells ---
    r_edges = np.linspace(0.0, R, n_cells + 1)
    mesh    = np.column_stack([r_edges[:-1], r_edges[1:]])
    r_mid   = 0.5 * (mesh[:, 0] + mesh[:, 1])

    # --- Initial condition: cold uniform material ---
    T_init_keV = 1e-4   # 0.01 HeV — effectively cold
    T_init  = np.full(n_cells, T_init_keV)
    Tr_init = np.full(n_cells, T_init_keV)

    # --- Boundary conditions ---
    #   inner (r=0): reflecting  → reflect[0]=True, T_boundary[0]=0
    #   outer (r=R): driven      → T_boundary[1] = callable(tau)
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
                f'Checkpoint n_cells={ckpt["n_cells"]} does not match run n_cells={n_cells}'
            )

        state = _deserialize_state(ckpt['state'])
        step_count = int(ckpt['step_count'])
        output_saved = set(float(t) for t in ckpt.get('output_saved', []))
        snapshots = [
            (float(t), np.asarray(T_mid), np.asarray(rm))
            for (t, T_mid, rm) in ckpt.get('snapshots', [])
        ]

        if 'np_random_state' in ckpt:
            np.random.set_state(ckpt['np_random_state'])
        if 'py_random_state' in ckpt:
            random.setstate(ckpt['py_random_state'])

        print(f'Restarted from checkpoint: {restart_from}')
        print(f'  elapsed time={state.time:.6f} ns  '
              f'physical time={T_INIT_NS + state.time:.6f} ns  '
              f'step_count={step_count}')
    else:
        state = imc.init_simulation(
            Ntarget, T_init, Tr_init, mesh, eos, inv_eos,
            geometry='spherical')
        snapshots = []
        output_saved = set()
        step_count = 0

    last_checkpoint_time = state.time

    # --- Time-stepping loop ---
    while state.time < duration - 1e-12:
        # Determine step size; clip to next snapshot or end
        step_dt = min(dt, duration - state.time)
        for tau_out in output_elapsed:
            if tau_out > state.time and state.time + step_dt > tau_out + 1e-12:
                step_dt = tau_out - state.time
                break

        state, info = imc.step(
            state, Ntarget, Nboundary, 0, NMax,
            T_boundary, step_dt, mesh, opacity, inv_eos, cv, source,
            reflect=(True, False),          # inner reflecting, outer absorbing
            geometry='spherical')

        step_count += 1
        t_phys = T_INIT_NS + state.time

        if step_count % output_freq == 0 or step_count <= 2:
            T_bath_HeV = T_bath_keV(t_phys) * T_HEV_PER_KEV
            T_surf_HeV = T_analytic_keV(R, t_phys) * T_HEV_PER_KEV
            print(f'  step {step_count:5d}  t_phys={t_phys:9.4f} ns'
                  f'  T_surf={T_surf_HeV:.4f} HeV'
                  f'  T_bath={T_bath_HeV:.4f} HeV'
                  f'  T_center={state.temperature[0]*T_HEV_PER_KEV:.4f} HeV'
                  f'  N={info["N_particles"]:6d}'
                  f'  ΔE={info["energy_loss"]:.3e} GJ')

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

    # Always write a final checkpoint at completion for reproducibility.
    save_checkpoint(
        checkpoint_path, state, step_count, snapshots, output_saved,
        n_cells=n_cells, dt=dt, final_output_time=final_output_time,
        save_prefix=save_prefix)
    print(f'Final checkpoint saved: {checkpoint_path}')

    if snapshots:
        plot_results(snapshots, save_prefix=save_prefix)

        # --- Save snapshots to NPZ ---
        snap_times = np.array([s[0] for s in snapshots])
        snap_T     = np.array([s[1] for s in snapshots])   # shape (n_snaps, n_cells)
        snap_r     = np.array([s[2] for s in snapshots])   # shape (n_snaps, n_cells)
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
                 R=R)
        print(f'Saved: {npz_path}')
    else:
        print('No snapshots captured — check output time clipping.')

    return state, snapshots


# =============================================================================
# ENTRY POINT
# =============================================================================


def _build_arg_parser():
    """Build command-line interface for run/restart/checkpoint workflows."""
    parser = argparse.ArgumentParser(
        description='Converging Marshak Wave IMC driver (spherical).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--n-cells', type=int, default=500,
                        help='Number of radial mesh cells.')
    parser.add_argument('--Ntarget', type=int, default=2 * 10**5,
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
                        default='converging_marshak_wave_imc',
                        help='Prefix for PNG/NPZ output files.')

    parser.add_argument('--final-output-time', type=float,
                        default=OUTPUT_TIMES_NS[-1],
                        help='Final physical time [ns] to stop the run.')
    parser.add_argument('--run-full', action='store_true',
                        help='Ignore --final-output-time and run to T_FINAL_NS.')

    parser.add_argument('--checkpoint-path', type=str,
                        default='converging_marshak_wave_imc_checkpoint.pkl',
                        help='Path to checkpoint file for save/restart.')
    parser.add_argument('--checkpoint-every-steps', type=int, default=100,
                        help='Write checkpoint every N steps (0 disables).')
    parser.add_argument('--checkpoint-every-ns', type=float, default=None,
                        help='Write checkpoint every elapsed ns (disabled if omitted).')
    parser.add_argument('--restart-from', type=str, default=None,
                        help='Restart from this checkpoint path.')

    parser.add_argument('--numba-threads', type=int, default=None,
                        help='Set Numba thread count for parallel transport kernels.')

    return parser


def _configure_numba_threads(num_threads):
    """Optionally set/get Numba thread count for prange-parallel kernels."""
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
    """CLI entrypoint."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

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
    )

if __name__ == '__main__':
    main()
