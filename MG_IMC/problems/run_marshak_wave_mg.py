#!/usr/bin/env python3
"""
Marshak Wave — Multigroup IMC run script.

Uses the 1-D spherical MG IMC solver (MG_IMC1D.py) in the large-radius limit
to simulate slab-geometry transport.  Setting R_inner = 1e4 cm ensures that
spherical-geometry corrections are O(L/R) ~ 7e-4 ≈ 0.07 %, negligible for
benchmark purposes.

Why the large-R trick works
---------------------------
  * Particle move:  r' = sqrt(r² + 2rμs + s²) → r + μs  (slab)  to O(s/R)
  * Direction update: μ' = (rμ + s)/r' → μ              (no streaming aberration)
  * Cell volumes: V_i ≈ 4πR²Δx, so T_mat = U/(c_v·ρ·V) is independent of R.
  * Boundary flux: the isotropic half-space flux through a sphere of radius R
    per unit surface area equals (a c / 4) T⁴ — identical to the slab case.

Physics
-------
  Domain :   0 → X_MAX = 7 cm  (slab coord x = r − R_INNER)
  Material : uniform  ρ = 0.01 g/cm³,  c_v = 0.05 GJ/(g·keV)
  Opacity  : σ_a(T, E) = C_OPA · ρ · T^A · E^B  (power-law)
               C_OPA = 10,  A = −0.5,  B = −3   →  σ_a = 10ρ T^{-1/2} E^{-3}
             Group opacity: geometric mean of boundary values
  Left BC  : blackbody at T_bc(t) [optionally ramped from T_START → T_END]
  Right BC : vacuum
  IC       : cold at T_INIT = 0.005 keV

Particle management (same strategy as run_dilute_spectrum_shell.py)
--------------------------------------------------------------------
  * Total particle budget split proportionally by emission power (fmin floor).
  * Census combing with growing Nmax ceiling.
  * Energy discrepancy from stochastic comb redistributed to surviving photons.

Usage
-----
  python run_marshak_wave_mg.py --mode quick
  python run_marshak_wave_mg.py --mode standard --G 32
  python run_marshak_wave_mg.py --mode publication --G 32 --no_time_bc
  python run_marshak_wave_mg.py --mode standard --resume

Output
------
  results/marshak_wave_mg/<tag>/snapshot_t_<time>ns.npz
  results/marshak_wave_mg/<tag>/history.npz
  results/marshak_wave_mg/<tag>/checkpoint.pkl
"""

import argparse
import os
import pickle
import random
import sys
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))   # problems → MG_IMC → RadTranBook
if _root not in sys.path:
    sys.path.insert(0, _root)

from MG_IMC.MG_IMC1D import SimulationState1DMG, init_simulation, step

try:
    from MG_IMC import A_RAD, C_LIGHT
except ImportError:
    A_RAD   = 0.01372   # GJ / (cm³ · keV⁴)
    C_LIGHT = 29.98     # cm / ns

CHECKPOINT_VERSION = 1

# ===========================================================================
# Problem constants
# ===========================================================================

R_INNER   = 1.0e4    # cm  —  inner radius; L/R ~ 7e-4, slab limit
X_MAX     = 7.0      # cm  —  slab domain length

RHO       = 0.01     # g / cm³
CV_SPEC   = 0.05     # GJ / (g · keV)  — specific heat

C_OPA     = 10.0     # opacity prefactor  [cm⁻¹ at T=E=1 keV]
A_OPA     = -0.5     # temperature exponent
B_OPA     = -3.0     # energy exponent
SIGMA_MAX = 1.0e14   # cm⁻¹  — opacity cap

T_INIT    = 0.005    # keV  — initial temperature
T_FLOOR   = 0.005    # keV  — opacity / EOS floor

T_BC_END   = 0.3     # keV  — constant BC temperature (or ramp end)
T_BC_START = 0.05    # keV  — ramp start temperature
T_RAMP     = 5.0     # ns   — ramp duration

NU_MIN    = 1.0e-4   # keV  — lower energy edge
NU_MAX    = 10.0     # keV  — upper energy edge
N_GROUPS_DEFAULT = 10

T_FINAL    = 10.0    # ns
DT_DEFAULT = 0.01    # ns
DUMP_TIMES = [1.0, 2.0, 5.0, 10.0]   # ns

T_EMIT_FLOOR = 0.005   # keV


# ===========================================================================
# Mesh helpers
# ===========================================================================

def make_mesh(n_cells, grid_beta=0.0):
    """Return (mesh, x_centers, rho_per_cell).

    Parameters
    ----------
    n_cells  : int
    grid_beta : float ≥ 0 — left-boundary clustering (0 = uniform).

    Returns
    -------
    mesh         : (n_cells, 2) — [[r_inner, r_outer], ...]
    x_centers    : (n_cells,)  — slab coordinates, x = r − R_INNER
    rho_per_cell : (n_cells,)  — uniform RHO
    """
    if grid_beta > 0.0:
        s = np.linspace(0.0, 1.0, n_cells + 1)
        x_edges = X_MAX * (np.exp(grid_beta * s) - 1.0) / (np.exp(grid_beta) - 1.0)
    else:
        x_edges = np.linspace(0.0, X_MAX, n_cells + 1)

    r_edges = R_INNER + x_edges
    mesh    = np.column_stack([r_edges[:-1], r_edges[1:]])
    x_centers    = 0.5 * (x_edges[:-1] + x_edges[1:])
    rho_per_cell = np.full(n_cells, RHO)
    return mesh, x_centers, rho_per_cell


def make_energy_edges(n_groups):
    return np.logspace(np.log10(NU_MIN), np.log10(NU_MAX), n_groups + 1)


# ===========================================================================
# Opacity (group-averaged)
# ===========================================================================

def make_sigma_a_funcs(energy_edges, rho_per_cell):
    """Return list of n_groups callables  σ_g(T_arr) → (n_cells,) array.

    Uses the geometric mean of the two boundary values,
    matching the diffusion code convention.
    """
    funcs = []
    for g in range(len(energy_edges) - 1):
        E_lo = float(energy_edges[g])
        E_hi = float(energy_edges[g + 1])

        def _sigma(T, elo=E_lo, ehi=E_hi):
            T_use = np.maximum(T, T_FLOOR)
            s_lo  = C_OPA * rho_per_cell * T_use**A_OPA * elo**B_OPA
            s_hi  = C_OPA * rho_per_cell * T_use**A_OPA * ehi**B_OPA
            return np.minimum(np.sqrt(s_lo * s_hi), SIGMA_MAX)

        funcs.append(_sigma)
    return funcs


# ===========================================================================
# EOS
# ===========================================================================

def make_eos_functions(rho_per_cell):
    cv_vol = rho_per_cell * CV_SPEC   # GJ / (cm³ · keV)

    def eos(T):
        return cv_vol * T

    def inv_eos(u):
        return u / cv_vol

    def cv_func(T):
        return cv_vol * np.ones_like(T)

    return eos, inv_eos, cv_func


# ===========================================================================
# Boundary temperature schedule
# ===========================================================================

def make_T_bc_func(time_dependent, T_start=T_BC_START, T_end=T_BC_END,
                   t_ramp=T_RAMP):
    """Return a callable T_bc(t) [keV]."""
    if not time_dependent:
        return lambda t: T_end

    def T_bc(t):
        if t < t_ramp:
            return T_start + (T_end - T_start) * (t / t_ramp)
        return T_end

    return T_bc


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _serialize_state(state):
    return {
        "weights":                   state.weights,
        "mus":                       state.mus,
        "times":                     state.times,
        "positions":                 state.positions,
        "cell_indices":              state.cell_indices,
        "groups":                    state.groups,
        "internal_energy":           state.internal_energy,
        "temperature":               state.temperature,
        "radiation_temperature":     state.radiation_temperature,
        "radiation_energy_by_group": state.radiation_energy_by_group,
        "time":                      float(state.time),
        "previous_total_energy":     float(state.previous_total_energy),
        "count":                     int(state.count),
    }


def _deserialize_state(data):
    return SimulationState1DMG(
        weights=data["weights"],
        mus=data["mus"],
        times=data["times"],
        positions=data["positions"],
        cell_indices=data["cell_indices"],
        groups=data["groups"],
        internal_energy=data["internal_energy"],
        temperature=data["temperature"],
        radiation_temperature=data["radiation_temperature"],
        radiation_energy_by_group=data["radiation_energy_by_group"],
        time=data["time"],
        previous_total_energy=data["previous_total_energy"],
        count=data["count"],
    )


def save_checkpoint(path, state, step_count, next_dump_idx,
                    cum_residual, metadata):
    payload = {
        "checkpoint_version":  CHECKPOINT_VERSION,
        "state":               _serialize_state(state),
        "step_count":          int(step_count),
        "next_dump_idx":       int(next_dump_idx),
        "cumulative_residual": float(cum_residual),
        "metadata":            metadata,
        "np_random_state":     np.random.get_state(),
        "py_random_state":     random.getstate(),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    v = payload.get("checkpoint_version")
    if v != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {v} (expected {CHECKPOINT_VERSION})")
    return payload


# ===========================================================================
# Snapshot saving — same .npz format as dilute_spectrum_shell snapshots
# ===========================================================================

def save_snapshot(out_dir, state, mesh, x_centers, rho_per_cell, energy_edges):
    t      = state.time
    fname  = os.path.join(out_dir, f"snapshot_t_{t:.5f}ns.npz")

    n_cells  = mesh.shape[0]
    volumes  = (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)

    E_rad_by_group = state.radiation_energy_by_group / volumes[np.newaxis, :]
    E_rad_total    = np.sum(E_rad_by_group, axis=0)
    F_rad_by_group = state.radiation_flux_by_group
    F_rad_total    = np.sum(F_rad_by_group, axis=0)

    r_edges = np.concatenate([mesh[:, 0], [mesh[-1, 1]]])

    np.savez_compressed(
        fname,
        # Slab coordinate (x) for easy comparison with diffusion results
        r_centers    = x_centers,
        r_edges      = r_edges - R_INNER,   # slab coord edges
        # Full spherical coordinates (for reference)
        r_centers_sph = mesh[:, 0] + 0.5 * (mesh[:, 1] - mesh[:, 0]),
        r_edges_sph    = r_edges,
        T_mat          = state.temperature,
        T_rad          = state.radiation_temperature,
        E_rad          = E_rad_total,
        E_rad_by_group = E_rad_by_group,
        F_rad          = F_rad_total,
        F_rad_by_group = F_rad_by_group,
        energy_edges   = energy_edges,
        rho            = rho_per_cell,
        time           = np.float64(t),
    )
    print(f"  *** Snapshot → {fname}")


# ===========================================================================
# Optical depth audit
# ===========================================================================

def print_optical_depth_audit(energy_edges):
    """Print mean-free-path and optical depth for key groups at T_BC_END."""
    print("\nOptical depth audit  (T = T_BC_END = {:.3f} keV)".format(T_BC_END))
    print(f"  {'Group':>5}  {'E_lo':>8}  {'E_hi':>8}  {'σ_a':>12}  "
          f"{'mfp (cm)':>10}  {'τ = L/mfp':>10}")
    for g in range(len(energy_edges) - 1):
        E_lo = energy_edges[g]
        E_hi = energy_edges[g + 1]
        sigma_g = (C_OPA * RHO * T_BC_END**A_OPA
                   * np.sqrt(E_lo**B_OPA * E_hi**B_OPA))
        sigma_g = min(sigma_g, SIGMA_MAX)
        mfp = 1.0 / max(sigma_g, 1e-300)
        tau = X_MAX / mfp
        print(f"  {g:>5d}  {E_lo:>8.4f}  {E_hi:>8.4f}  "
              f"{sigma_g:>12.4e}  {mfp:>10.4e}  {tau:>10.3f}")
    print()


# ===========================================================================
# Main run
# ===========================================================================

def run(args):
    # ------------------------------------------------------------------
    # Mode-dependent particle counts
    # ------------------------------------------------------------------
    mode_params = {
        "quick":       dict(Ntarget=5_000,    Nmax_init=10_000,
                            Nmax_growth=1_000,   Nmax_final=50_000),
        "standard":    dict(Ntarget=50_000,   Nmax_init=100_000,
                            Nmax_growth=20_000,  Nmax_final=2_000_000),
        "publication": dict(Ntarget=500_000,  Nmax_init=2_000_000,
                            Nmax_growth=100_000, Nmax_final=20_000_000),
    }
    p = mode_params[args.mode]
    Ntarget     = p["Ntarget"]
    Nboundary   = Ntarget   # unused when particle_budget_fmin > 0
    Nmax_init   = p["Nmax_init"]
    Nmax_growth = p["Nmax_growth"]
    Nmax_final  = p["Nmax_final"]

    if args.Nmax is not None:
        Nmax_init   = args.Nmax
        Nmax_growth = 0 if args.Nmax < 0 else Nmax_growth
        Nmax_final  = args.Nmax if args.Nmax < 0 else Nmax_final

    n_groups  = args.G
    dt        = args.dt
    n_cells   = args.nx
    time_dep  = not args.no_time_bc

    # ------------------------------------------------------------------
    # Build problem geometry and physics
    # ------------------------------------------------------------------
    energy_edges  = make_energy_edges(n_groups)
    mesh, x_centers, rho_per_cell = make_mesh(n_cells, grid_beta=args.grid_beta)
    sigma_a_funcs = make_sigma_a_funcs(energy_edges, rho_per_cell)
    eos, inv_eos, cv_func = make_eos_functions(rho_per_cell)

    T_bc_func = make_T_bc_func(time_dep, T_start=args.T_start,
                               T_end=args.T_end, t_ramp=args.t_ramp)

    source  = np.zeros(n_cells)
    reflect = (False, False)

    # ------------------------------------------------------------------
    # Output directory  e.g. marshak_wave_mg/imc_10g_standard/
    # ------------------------------------------------------------------
    bc_tag = "timeBC" if time_dep else "constBC"
    tag = f"imc_{n_groups}g_{args.mode}_{bc_tag}"
    out_dir = os.path.join("results", "marshak_wave_mg", tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pkl")

    # ------------------------------------------------------------------
    # Print info
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Marshak Wave — Multigroup IMC  (large-R spherical limit)")
    print(f"  Mode:       {args.mode}")
    print(f"  Groups:     {n_groups},  Cells: {n_cells}")
    print(f"  dt:         {dt} ns,  T_final = {T_FINAL} ns")
    print(f"  Ntarget:    {Ntarget}  (fmin=0.2)")
    print(f"  Nmax:       {Nmax_init} → {Nmax_final}  (growth {Nmax_growth}/step)")
    print(f"  Left BC:    blackbody T_bc(t)  "
          + ("(ramp: {:.3f}→{:.3f} keV over {:.1f} ns)".format(
              args.T_start, args.T_end, args.t_ramp)
             if time_dep else f"= {args.T_end:.3f} keV (constant)"))
    print(f"  Right BC:   vacuum")
    print(f"  Grid beta:  {args.grid_beta}  (0 = uniform)")
    print(f"  R_inner:    {R_INNER:.2e} cm  (L/R = {X_MAX/R_INNER:.2e})")
    print(f"  Output dir: {out_dir}")
    print("=" * 72)
    print_optical_depth_audit(energy_edges)

    # ------------------------------------------------------------------
    # Initialise or resume
    # ------------------------------------------------------------------
    dump_times    = sorted(DUMP_TIMES)
    next_dump_idx = 0
    step_count    = 0
    cum_residual  = 0.0
    saved_dumps   = set()
    Nmax_current  = Nmax_init

    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        payload      = load_checkpoint(ckpt_path)
        state        = _deserialize_state(payload["state"])
        step_count   = payload["step_count"]
        next_dump_idx = payload["next_dump_idx"]
        cum_residual = payload["cumulative_residual"]
        Nmax_current = payload["metadata"].get("Nmax_current", Nmax_init)
        np.random.set_state(payload["np_random_state"])
        random.setstate(payload["py_random_state"])
        saved_dumps  = set(range(next_dump_idx))
        print(f"  Resumed at t = {state.time:.4f} ns, step {step_count}, "
              f"Nmax = {Nmax_current}")
    else:
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)

        Tinit   = np.full(n_cells, T_INIT)
        Tr_init = np.full(n_cells, T_INIT)
        state = init_simulation(Ntarget, Tinit, Tr_init, mesh, energy_edges,
                                eos, inv_eos, T_emit_floor=T_EMIT_FLOOR)

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    time_tol = 1e-12 * max(T_FINAL, 1.0)

    print(f"\n{'Step':>6}  {'t (ns)':>9}  {'N_part':>7}  {'N_bc':>7}  "
          f"{'E_bc':>12}  {'E_tot':>12}  {'E_int':>12}  {'E_rad':>12}  "
          f"{'Resid':>10}")
    print("-" * 95)

    while state.time < T_FINAL - time_tol:
        # Adjust dt to land on dump times
        step_dt = dt
        if next_dump_idx < len(dump_times):
            gap = dump_times[next_dump_idx] - state.time
            if gap > 0.0:
                step_dt = min(step_dt, gap)
        step_dt = min(step_dt, T_FINAL - state.time)

        # Time-dependent BC: evaluate at current time
        T_inner = T_bc_func(state.time)
        T_boundary = (T_inner, 0.0)

        t0 = _time.perf_counter()
        state, info = step(
            state, Ntarget, Nboundary, 0,
            0 if args.no_comb else Nmax_current,
            T_boundary, step_dt, mesh, energy_edges,
            sigma_a_funcs, inv_eos, cv_func, source, reflect,
            theta=1.0,
            use_scalar_intensity_Tr=False,
            T_emit_floor=T_EMIT_FLOOR,
            particle_budget_fmin=0.2,
            conserve_comb_energy="radiation",
            Nmax_growth=Nmax_growth,
            Nmax_final=Nmax_final,
        )
        wall = _time.perf_counter() - t0

        step_count   += 1
        cum_residual += abs(info["energy_residual"])
        Nmax_current  = info["Nmax_next"]

        print(f"{step_count:>6}  {info['time']:>9.5f}  "
              f"{info['N_particles']:>7d}  "
              f"{info['N_boundary']:>7d}  "
              f"{info['boundary_emission']:>12.5e}  "
              f"{info['total_energy']:>12.5e}  "
              f"{info['total_internal_energy']:>12.5e}  "
              f"{info['total_radiation_energy']:>12.5e}  "
              f"{info['energy_residual']:>10.3e}"
              f"  [{wall:.1f}s]")

        # Dump snapshots
        while (next_dump_idx < len(dump_times) and
               state.time >= dump_times[next_dump_idx] - time_tol):
            if next_dump_idx not in saved_dumps:
                save_snapshot(out_dir, state, mesh, x_centers,
                              rho_per_cell, energy_edges)
                saved_dumps.add(next_dump_idx)
            next_dump_idx += 1

        # Checkpoint every 50 steps
        if step_count % 50 == 0:
            meta = dict(mode=args.mode, n_groups=n_groups, dt=dt,
                        n_cells=n_cells, Nmax_current=Nmax_current)
            save_checkpoint(ckpt_path, state, step_count, next_dump_idx,
                            cum_residual, meta)

    # Final snapshot if not already written
    if not any(abs(dump_times[i] - state.time) <= time_tol for i in saved_dumps):
        save_snapshot(out_dir, state, mesh, x_centers, rho_per_cell, energy_edges)

    # History summary
    hist_path = os.path.join(out_dir, "history.npz")
    np.savez_compressed(
        hist_path,
        step_count         = np.int64(step_count),
        final_time         = np.float64(state.time),
        cumulative_residual = np.float64(cum_residual),
        x_centers          = x_centers,
        energy_edges       = energy_edges,
        final_T_mat        = state.temperature,
        final_T_rad        = state.radiation_temperature,
    )
    print(f"\nHistory → {hist_path}")
    print(f"Cumulative energy residual: {cum_residual:.4e}")

    meta = dict(mode=args.mode, n_groups=n_groups, dt=dt,
                n_cells=n_cells, Nmax_current=Nmax_current)
    save_checkpoint(ckpt_path, state, step_count, next_dump_idx,
                    cum_residual, meta)
    print(f"Final checkpoint → {ckpt_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Marshak wave MG IMC benchmark (large-R slab limit)."
    )
    p.add_argument("--mode", choices=["quick", "standard", "publication"],
                   default="standard")
    p.add_argument("--G", type=int, default=N_GROUPS_DEFAULT, metavar="N_GROUPS",
                   help=f"Number of energy groups (default: {N_GROUPS_DEFAULT}).")
    p.add_argument("--nx", type=int, default=140, metavar="N_CELLS",
                   help="Number of spatial cells (default: 140).")
    p.add_argument("--dt", type=float, default=DT_DEFAULT, metavar="DT",
                   help=f"Nominal timestep in ns (default: {DT_DEFAULT}).")
    p.add_argument("--no_time_bc", action="store_true",
                   help="Use constant left BC at T_end instead of ramped BC.")
    p.add_argument("--T_start", type=float, default=T_BC_START, metavar="TSTART",
                   help=f"Ramp start temperature (keV, default: {T_BC_START}).")
    p.add_argument("--T_end", type=float, default=T_BC_END, metavar="TEND",
                   help=f"Ramp end / constant BC temperature (keV, default: {T_BC_END}).")
    p.add_argument("--t_ramp", type=float, default=T_RAMP, metavar="T_RAMP",
                   help=f"Ramp duration (ns, default: {T_RAMP}).")
    p.add_argument("--grid_beta", type=float, default=0.0, metavar="BETA",
                   help="Left-clustering strength (0 = uniform, positive clusters "
                        "cells near x=0; default: 0).")
    p.add_argument("--Nmax", type=int, default=None, metavar="NMAX",
                   help="Override the mode-preset particle cap.  Positive: always comb "
                        "to NMAX.  Negative: threshold mode (comb only when N ≥ |NMAX|).")
    p.add_argument("--no_comb", action="store_true",
                   help="Disable particle combing entirely.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing checkpoint in the output directory.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: unseeded).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
