#!/usr/bin/env python3
"""
Dilute Spectrum Shell — Multigroup IMC run script.

Runs the 1-D spherical multigroup IMC solver for the dilute-spectrum-shell
benchmark problem defined in dilute_spectrum_shell.py.  Results are written
to results/dilute_spectrum_shell/<tag>/ as compressed NumPy archives.

Usage examples
--------------
  python run_dilute_spectrum_shell.py --mode quick
  python run_dilute_spectrum_shell.py --mode standard --G 32
  python run_dilute_spectrum_shell.py --mode publication --G 64 --dt 0.005
  python run_dilute_spectrum_shell.py --mode standard --resume           # resume from checkpoint

Output files
------------
  results/dilute_spectrum_shell/<tag>/snapshot_t_<time>ns.npz
  results/dilute_spectrum_shell/<tag>/history.npz
  results/dilute_spectrum_shell/<tag>/checkpoint.pkl
"""

import argparse
import os
import pickle
import random
import sys
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
# Locate project root and add to path so package imports work from anywhere.
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # problems -> MG_IMC -> RadTranBook
if _root not in sys.path:
    sys.path.insert(0, _root)

from MG_IMC.MG_IMC1D import (
    SimulationState1DMG,
    init_simulation,
    step,
)

# Import shared problem parameters
from MG_IMC.problems.dilute_spectrum_shell import (
    C_LIGHT, A_RAD,
    R_S, R_1, R_2, R_OUT,
    T_S, T_INIT, T_FLOOR,
    RHO_CAVITY, RHO_SHELL, CV_SPEC,
    N_GROUPS_DEFAULT, NU_MIN, NU_MAX,
    T_FINAL, DT_DEFAULT, DUMP_TIMES,
    make_mesh, make_energy_edges,
    make_sigma_a_funcs, make_eos_functions,
    print_optical_depth_audit,
)

CHECKPOINT_VERSION = 1


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _serialize_state(state):
    """Convert SimulationState1DMG to a pickle-safe dict."""
    return {
        "weights":                    state.weights,
        "mus":                        state.mus,
        "times":                      state.times,
        "positions":                  state.positions,
        "cell_indices":               state.cell_indices,
        "groups":                     state.groups,
        "internal_energy":            state.internal_energy,
        "temperature":                state.temperature,
        "radiation_temperature":      state.radiation_temperature,
        "radiation_energy_by_group":  state.radiation_energy_by_group,
        "time":                       float(state.time),
        "previous_total_energy":      float(state.previous_total_energy),
        "count":                      int(state.count),
    }


def _deserialize_state(data):
    """Rebuild SimulationState1DMG from checkpoint dict."""
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


def save_checkpoint(path, state, step_count, next_dump_idx, cumulative_residual,
                    metadata):
    """Persist simulation state so the run can be resumed."""
    payload = {
        "checkpoint_version":  CHECKPOINT_VERSION,
        "state":               _serialize_state(state),
        "step_count":          int(step_count),
        "next_dump_idx":       int(next_dump_idx),
        "cumulative_residual": float(cumulative_residual),
        "metadata":            metadata,
        "np_random_state":     np.random.get_state(),
        "py_random_state":     random.getstate(),
    }
    # Atomic write via temp file
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Load and validate checkpoint; raise ValueError on version mismatch."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    v = payload.get("checkpoint_version")
    if v != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {v} (expected {CHECKPOINT_VERSION})")
    return payload


# ===========================================================================
# Snapshot saving
# ===========================================================================

def save_snapshot(out_dir, state, mesh, r_centers, rho_per_cell, energy_edges,
                  save_census=False):
    """Save the current radiation and material fields as a compressed .npz.

    If *save_census* is True, also store per-particle and per-cell census
    diagnostics (can produce large files for high particle counts):

    Raw arrays (one entry per surviving particle):
        census_weights, census_positions, census_mus,
        census_groups, census_cell_indices

    Per-cell summaries (shape (n_cells,) or (n_groups, n_cells)):
        census_N_per_cell      -- particle count in each cell
        census_W_per_cell      -- total weight in each cell  [GJ]
        census_W_by_group      -- total weight (n_groups, n_cells)  [GJ]
    """
    t = state.time
    fname = os.path.join(out_dir, f"snapshot_t_{t:.5f}ns.npz")

    n_cells  = mesh.shape[0]
    n_groups = len(energy_edges) - 1
    volumes = (4.0 / 3.0) * np.pi * (mesh[:, 1]**3 - mesh[:, 0]**3)
    E_rad_by_group = state.radiation_energy_by_group / volumes[np.newaxis, :]
    E_rad_total    = np.sum(E_rad_by_group, axis=0)
    F_rad_by_group = state.radiation_flux_by_group          # (n_groups, n_cells)
    F_rad_total    = np.sum(F_rad_by_group, axis=0)         # (n_cells,)

    # --- Build kwargs dict; optionally add census arrays ---
    save_kwargs = dict(
        # --- field arrays ---
        r_centers=r_centers,
        r_edges=np.concatenate([mesh[:, 0], [mesh[-1, 1]]]),
        T_mat=state.temperature,
        T_rad=state.radiation_temperature,
        E_rad=E_rad_total,
        E_rad_by_group=E_rad_by_group,
        F_rad=F_rad_total,
        F_rad_by_group=F_rad_by_group,
        energy_edges=energy_edges,
        rho=rho_per_cell,
        time=np.float64(t),
    )

    if save_census:
        # --- Vectorised per-cell census summary ---
        ci    = state.cell_indices.astype(np.intp)
        valid = (ci >= 0) & (ci < n_cells)
        ci_v  = ci[valid]
        w_v   = state.weights[valid]
        g_v   = state.groups[valid].astype(np.intp)

        census_N = np.bincount(ci_v, minlength=n_cells).astype(np.int64)
        census_W = np.bincount(ci_v, weights=w_v, minlength=n_cells)
        census_W_by_group = np.zeros((n_groups, n_cells))
        for g in range(n_groups):
            gm = g_v == g
            if gm.any():
                census_W_by_group[g] = np.bincount(ci_v[gm], weights=w_v[gm],
                                                   minlength=n_cells)

        save_kwargs.update(dict(
            # --- raw census (one entry per surviving particle) ---
            census_weights=state.weights,
            census_positions=state.positions,
            census_mus=state.mus,
            census_groups=state.groups,
            census_cell_indices=state.cell_indices,
            # --- per-cell census summary ---
            census_N_per_cell=census_N,
            census_W_per_cell=census_W,
            census_W_by_group=census_W_by_group,
        ))

    np.savez_compressed(fname, **save_kwargs)
    census_note = f", census: {len(state.weights):,} particles" if save_census else ""
    print(f"  *** Snapshot saved → {fname}{census_note}")


# ===========================================================================
# Main simulation function
# ===========================================================================

def run(args):
    # ------------------------------------------------------------------
    # Mode-dependent particle counts
    # ------------------------------------------------------------------
    mode_params = {
        # Ntarget     = total particle budget (shared by boundary + emission + source)
        # Nmax_init   = starting particle cap
        # Nmax_growth = cap increase per step
        # Nmax_final  = maximum particle cap
        "quick":       dict(Ntarget=5_000,     Nmax_init=10_000,    Nmax_growth=1_000,   Nmax_final=20_000),
        "standard":    dict(Ntarget=50_000,   Nmax_init=100_000,   Nmax_growth=30_000, Nmax_final=1_300_000),
        "publication": dict(Ntarget=1_000_000, Nmax_init=3_000_000, Nmax_growth=100_000, Nmax_final=30_000_000),
    }
    p = mode_params[args.mode]
    Ntarget     = p["Ntarget"]
    Nboundary   = Ntarget   # unused when particle_budget_fmin > 0
    Nmax_init   = p["Nmax_init"]
    Nmax_growth = p["Nmax_growth"]
    Nmax_final  = p["Nmax_final"]

    # --Nmax overrides the mode preset.
    #   Positive  → always comb to Nmax (standard behaviour).
    #   Negative  → threshold mode: only comb once N ≥ |Nmax|, target = |Nmax|.
    #               Nmax_growth is forced to 0 so the threshold stays fixed.
    if args.Nmax is not None:
        Nmax_init   = args.Nmax
        Nmax_growth = 0 if args.Nmax < 0 else Nmax_growth
        Nmax_final  = args.Nmax if args.Nmax < 0 else Nmax_final

    n_groups = args.G
    dt       = args.dt
    mesh_mode = "quick" if args.mode == "quick" else "standard"

    # ------------------------------------------------------------------
    # Build problem geometry and physics
    # ------------------------------------------------------------------
    mesh, r_centers, rho_per_cell = make_mesh(mode=mesh_mode)
    energy_edges = make_energy_edges(n_groups)
    sigma_a_funcs = make_sigma_a_funcs(energy_edges, rho_per_cell)
    eos, inv_eos, cv_func = make_eos_functions(rho_per_cell)

    n_cells = mesh.shape[0]
    source  = np.zeros(n_cells)    # no volumetric source (only boundary)
    T_boundary = (T_S, 0.0)        # inner blackbody; outer vacuum
    reflect    = (False, False)    # vacuum boundaries on both sides

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    tag     = f"imc_{n_groups}g_{args.mode}"
    out_dir = os.path.join("results", "dilute_spectrum_shell", tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pkl")

    # ------------------------------------------------------------------
    # Print problem info
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Dilute Spectrum Shell — Multigroup IMC")
    print(f"  Mode:          {args.mode}")
    print(f"  Groups:        {n_groups}")
    print(f"  Cells:         {n_cells}")
    print(f"  dt:            {dt} ns,  T_final = {T_FINAL} ns")
    print(f"  Ntarget:       {Ntarget} (total budget, fmin=0.05)")
    print(f"  Nmax:          {Nmax_init} → {Nmax_final} (growth {Nmax_growth}/step)")
    print(f"  Inner BC:      blackbody T = {T_S} keV at r = {R_S} cm")
    print(f"  Output dir:    {out_dir}")
    print("=" * 72)
    print_optical_depth_audit(mesh, energy_edges, rho_per_cell)

    # ------------------------------------------------------------------
    # Initialise or resume
    # ------------------------------------------------------------------
    dump_times     = sorted(DUMP_TIMES)
    next_dump_idx  = 0
    step_count     = 0
    cum_residual   = 0.0

    # Track which dump times have already been saved
    saved_dumps = set()

    if args.resume and os.path.exists(ckpt_path):
        print(f"\nResuming from checkpoint: {ckpt_path}")
        payload = load_checkpoint(ckpt_path)
        state         = _deserialize_state(payload["state"])
        step_count    = payload["step_count"]
        next_dump_idx = payload["next_dump_idx"]
        cum_residual  = payload["cumulative_residual"]
        Nmax_current  = payload["metadata"].get("Nmax_current", Nmax_init)
        np.random.set_state(payload["np_random_state"])
        random.setstate(payload["py_random_state"])
        # Any dump time already past is considered saved
        saved_dumps = {i for i in range(next_dump_idx)}
        print(f"  Resumed at t = {state.time:.4f} ns,  step {step_count},  Nmax = {Nmax_current}")
    else:
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)

        Tinit   = np.full(n_cells, T_INIT)
        Tr_init = np.full(n_cells, T_INIT)  # start cold

        state = init_simulation(Ntarget, Tinit, Tr_init, mesh, energy_edges,
                                eos, inv_eos, T_emit_floor=0.025)
        Nmax_current = Nmax_init

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    time_tol = 1e-12 * max(T_FINAL, 1.0)

    print(f"\n{'Step':>6}  {'t (ns)':>9}  {'N_part':>7}  {'N_bc':>7}  "
          f"{'E_bc':>12}  {'E_tot':>12}  {'E_int':>12}  {'E_rad':>12}  "
          f"{'Resid':>10}")
    print("-" * 95)

    while state.time < T_FINAL - time_tol:
        # ------ Adjust dt to land exactly on dump times ------
        step_dt = dt
        if next_dump_idx < len(dump_times):
            t_next_dump = dump_times[next_dump_idx]
            if state.time < t_next_dump:
                step_dt = min(step_dt, t_next_dump - state.time)
        step_dt = min(step_dt, T_FINAL - state.time)

        t0 = _time.perf_counter()
        state, info = step(
            state, Ntarget, Nboundary, 0,
            0 if args.no_comb else Nmax_current,
            T_boundary, step_dt, mesh, energy_edges,
            sigma_a_funcs, inv_eos, cv_func, source, reflect,
            theta=1.0, use_scalar_intensity_Tr=False,
            T_emit_floor=0.025,          # suppress emission below 0.025 keV
            particle_budget_fmin=0.2,   # proportional budget; 10% floor per source
            conserve_comb_energy="radiation",   # redistribute discrepancy to photons
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

        # ------ Check if a dump time was just reached ------
        while (next_dump_idx < len(dump_times) and
               state.time >= dump_times[next_dump_idx] - time_tol):
            if next_dump_idx not in saved_dumps:
                save_snapshot(out_dir, state, mesh, r_centers, rho_per_cell,
                              energy_edges, save_census=args.save_census)
                saved_dumps.add(next_dump_idx)
            next_dump_idx += 1

        # ------ Checkpoint every 50 steps ------
        if step_count % 50 == 0:
            meta = dict(mode=args.mode, n_groups=n_groups, dt=dt,
                        T_final=T_FINAL, n_cells=n_cells, Nmax_current=Nmax_current)
            save_checkpoint(ckpt_path, state, step_count, next_dump_idx,
                            cum_residual, meta)

    # ------------------------------------------------------------------
    # Ensure final state is saved if it wasn't already written as a dump
    # ------------------------------------------------------------------
    final_already_saved = any(
        abs(dump_times[i] - state.time) <= time_tol for i in saved_dumps
    )
    if state.time >= T_FINAL - time_tol and not final_already_saved:
        save_snapshot(out_dir, state, mesh, r_centers, rho_per_cell, energy_edges,
                      save_census=args.save_census)

    # ------------------------------------------------------------------
    # Save history summary
    # ------------------------------------------------------------------
    hist_path = os.path.join(out_dir, "history.npz")
    np.savez_compressed(hist_path,
                        step_count=np.int64(step_count),
                        final_time=np.float64(state.time),
                        cumulative_residual=np.float64(cum_residual),
                        r_centers=r_centers,
                        energy_edges=energy_edges,
                        final_T_mat=state.temperature,
                        final_T_rad=state.radiation_temperature)
    print(f"\nHistory saved → {hist_path}")
    print(f"Cumulative energy residual:  {cum_residual:.4e}")

    # Final checkpoint
    meta = dict(mode=args.mode, n_groups=n_groups, dt=dt,
                T_final=T_FINAL, n_cells=n_cells, Nmax_current=Nmax_current)
    save_checkpoint(ckpt_path, state, step_count, next_dump_idx,
                    cum_residual, meta)
    print(f"Final checkpoint saved → {ckpt_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the dilute-spectrum-shell MG IMC benchmark."
    )
    p.add_argument(
        "--mode", choices=["quick", "standard", "publication"],
        default="standard",
        help="Particle count preset (default: standard).",
    )
    p.add_argument(
        "--G", type=int, default=N_GROUPS_DEFAULT, metavar="N_GROUPS",
        help=f"Number of energy groups (default: {N_GROUPS_DEFAULT}).",
    )
    p.add_argument(
        "--dt", type=float, default=DT_DEFAULT, metavar="DT",
        help=f"Nominal timestep in ns (default: {DT_DEFAULT}).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: unseeded).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint in the output directory.",
    )
    p.add_argument(
        "--no_comb", action="store_true",
        help="Disable particle combing (Nmax=0); particles accumulate every step.",
    )
    p.add_argument(
        "--Nmax", type=int, default=None, metavar="NMAX",
        help="Override the mode-preset particle cap.  "
             "Positive: always comb to NMAX.  "
             "Negative: threshold mode — only comb when N ≥ |NMAX|, "
             "target = |NMAX| (Nmax_growth forced to 0).",
    )
    p.add_argument(
        "--save_census", action="store_true",
        help="Include per-particle and per-cell census arrays in snapshots "
             "(default: off; can significantly increase file size).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
