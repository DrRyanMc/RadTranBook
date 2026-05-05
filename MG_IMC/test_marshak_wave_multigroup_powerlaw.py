#!/usr/bin/env python3
"""
Marshak Wave Problem - Multigroup IMC Version with Power-Law Opacity

IMC analogue of nonEquilibriumDiffusion/problems/marshak_wave_multigroup_powerlaw.py.

Problem setup:
- Multigroup opacity: sigma_a(T,E) = 10 * rho * T^(-1/2) * E^(-3)
- Group opacity uses geometric mean at group boundaries
- Left boundary: blackbody source with optional time-dependent T_bc(t)
- Right boundary: cold incoming at T_init (approximately vacuum)
- 1D slab represented as 2D xy with ny=1
"""

import argparse
import os
import pickle
import random
import sys
import time as _time

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for package imports and plotting utilities.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MG_IMC import A_RAD, C_LIGHT, SimulationState2DMG, init_simulation, step

from utils.plotfuncs import font, show

RHO = 0.01  # g/cm^3
CHECKPOINT_VERSION = 1


def _serialize_state(state):
    """Convert SimulationState2DMG to a pickle-safe payload."""
    payload = {
        "weights": state.weights,
        "dir1": state.dir1,
        "dir2": state.dir2,
        "times": state.times,
        "pos1": state.pos1,
        "pos2": state.pos2,
        "cell_i": state.cell_i,
        "cell_j": state.cell_j,
        "groups": state.groups,
        "internal_energy": state.internal_energy,
        "temperature": state.temperature,
        "radiation_temperature": state.radiation_temperature,
        "radiation_energy_by_group": state.radiation_energy_by_group,
        "time": float(state.time),
        "previous_total_energy": float(state.previous_total_energy),
        "count": int(state.count),
    }
    payload["radiation_energy_by_group_postcomb"] = state.radiation_energy_by_group_postcomb
    return payload


def _deserialize_state(data):
    """Rebuild SimulationState2DMG from checkpoint payload."""
    return SimulationState2DMG(
        weights=data["weights"],
        dir1=data["dir1"],
        dir2=data["dir2"],
        times=data["times"],
        pos1=data["pos1"],
        pos2=data["pos2"],
        cell_i=data["cell_i"],
        cell_j=data["cell_j"],
        groups=data["groups"],
        internal_energy=data["internal_energy"],
        temperature=data["temperature"],
        radiation_temperature=data["radiation_temperature"],
        radiation_energy_by_group=data["radiation_energy_by_group"],
        time=float(data["time"]),
        previous_total_energy=float(data["previous_total_energy"]),
        count=int(data["count"]),
        radiation_energy_by_group_postcomb=data.get("radiation_energy_by_group_postcomb", None),
    )


def save_checkpoint(path, state, step_count, cumulative_residual, history, metadata):
    """Persist simulation state and run progress so the run can resume."""
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state": _serialize_state(state),
        "step_count": int(step_count),
        "cumulative_residual": float(cumulative_residual),
        "history": history,
        "metadata": metadata,
        "np_random_state": np.random.get_state(),
        "py_random_state": random.getstate(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    """Load and validate checkpoint payload."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    version = payload.get("checkpoint_version", None)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {version} (expected {CHECKPOINT_VERSION})"
        )
    return payload


def _planck_group_integral(E_low, E_high, T):
    """Planck group integral B_g(T) in gray units used by MG_IMC."""
    if T <= 0.0:
        return 0.0
    nquad = 80
    E = np.linspace(E_low, E_high, nquad)
    B_E = (2.0 * E**3 / C_LIGHT**2) / (np.exp(E / T) - 1.0 + 1e-300)
    # NumPy compatibility: newer builds may only expose trapezoid.
    if hasattr(np, "trapezoid"):
        return np.trapezoid(B_E, E)
    return np.trapz(B_E, E)


def Bg_multigroup(energy_edges, T):
    """Return per-group Planck integrals for temperature T."""
    n_groups = len(energy_edges) - 1
    out = np.zeros(n_groups)
    for g in range(n_groups):
        out[g] = _planck_group_integral(energy_edges[g], energy_edges[g + 1], T)
    return out


def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """Power-law opacity sigma_a(T,E) = 10 * rho * T^(-1/2) * E^(-3)."""
    T_use = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * (T_use ** -0.5) * (E ** -3.0), 1e14)


def make_powerlaw_opacity_func(E_low, E_high, rho=1.0):
    """Group opacity from geometric mean of boundary energies."""

    def opacity_func(T):
        sigma_low = powerlaw_opacity_at_energy(T, E_low, rho)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho)
        return np.sqrt(sigma_low * sigma_high)

    return opacity_func


def boundary_temperature_fn_factory(time_dependent_bc, t_ramp=5.0, T_start=0.05, T_end=0.25):
    """Build left boundary temperature function for IMC boundary source."""

    if not time_dependent_bc:
        return lambda t: T_end

    def T_bc_fn(t):
        if t < t_ramp:
            return T_start + (T_end - T_start) * (t / t_ramp)
        return T_end

    return T_bc_fn


def build_left_clustered_edges(x_min, x_max, nx, grid_beta=0.0):
    """Build x-edges with optional left-boundary clustering.

    Parameters
    ----------
    x_min, x_max : float
        Domain limits.
    nx : int
        Number of cells.
    grid_beta : float
        Clustering strength. ``0`` gives uniform spacing. Positive values
        cluster cells near ``x_min`` using an exponential map.
    """
    if grid_beta <= 0.0:
        return np.linspace(x_min, x_max, nx + 1)

    s = np.linspace(0.0, 1.0, nx + 1)
    mapped = (np.exp(grid_beta * s) - 1.0) / (np.exp(grid_beta) - 1.0)
    return x_min + (x_max - x_min) * mapped


def run_marshak_wave_multigroup_powerlaw_imc(
    n_groups=10,
    time_dependent_bc=True,
    ntarget=200000,
    nboundary=100000,
    nmax=400000,
    use_scalar_intensity_Tr=True,
    nx=140,
    dt=0.01,
    final_time=10.0,
    grid_beta=0.0,
    checkpoint_every=10,
    checkpoint_file=None,
    restart_from=None,
):
    print("=" * 80)
    print(f"Marshak Wave Problem - Multigroup IMC ({n_groups} Groups) with Power-Law Opacity")
    print("=" * 80)

    # Problem setup to mirror diffusion case.
    x_min = 0.0
    x_max = 7.0
    ny = 1
    target_times = [t for t in [1.0, 2.0, 5.0, 10.0] if t <= final_time + 1e-12]
    if len(target_times) == 0:
        target_times = [final_time]

    rho = RHO
    cv_mass = 0.05  # GJ/(g keV)
    cv = cv_mass * rho  # volumetric c_v in GJ/(cm^3 keV), consistent with DO

    energy_edges = np.logspace(np.log10(1e-4), np.log10(10.0), n_groups + 1)

    print("Material properties:")
    print("  Opacity: sigma_a(T,E) = 10.0 * rho * T^(-1/2) * E^(-3)")
    print("  Group opacity: geometric mean at group boundaries")
    print(f"  Density: rho = {rho} g/cm^3")
    print(f"  Heat capacity: c_v = {cv_mass} GJ/(g keV)  (volumetric: {cv:.6e} GJ/(cm^3 keV))")
    print("  Left BC: blackbody source")
    print("  Right BC: cold incoming at T_init (approximately vacuum)")
    print(f"  Scalar-intensity Tr estimator: {use_scalar_intensity_Tr}")

    x_edges = build_left_clustered_edges(x_min, x_max, nx, grid_beta=grid_beta)
    y_edges = np.array([0.0, 1.0])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Group opacities.
    sigma_a_funcs = []
    for g in range(n_groups):
        sigma_a_funcs.append(make_powerlaw_opacity_func(energy_edges[g], energy_edges[g + 1], rho))

    T_bc_func = boundary_temperature_fn_factory(time_dependent_bc)
    T_bc_ref = T_bc_func(0.0)

    # Diagnostics at reference boundary temperature.
    B_g_bc = Bg_multigroup(energy_edges, max(T_bc_ref, 1e-8))
    chi = B_g_bc / (np.sum(B_g_bc) + 1e-300)

    print(f"\nEnergy group edges (keV): {energy_edges}")
    print(f"Emission fractions at T = {T_bc_ref:.3f} keV:")
    for g in range(n_groups):
        sigma_g = sigma_a_funcs[g](T_bc_ref)
        sigma_val = float(np.mean(sigma_g)) if np.ndim(sigma_g) > 0 else float(sigma_g)
        print(
            f"  Group {g:2d} [{energy_edges[g]:8.4f}, {energy_edges[g+1]:8.4f}] keV: "
            f"chi = {chi[g]:.6f}, sigma_a = {sigma_val:.3e} cm^-1"
        )

    # Opacity vs energy plot (matching diffusion script style).
    E_fine = np.logspace(np.log10(energy_edges[0]), np.log10(energy_edges[-1]), 600)
    sigma_fine = powerlaw_opacity_at_energy(max(T_bc_ref, 1e-8), E_fine, rho)

    fig_op, ax_op = plt.subplots(figsize=(7, 5))
    ax_op.loglog(E_fine, sigma_fine, "k-", linewidth=1.5, label=r"$\sigma_a$ (continuous)")

    group_emission = Bg_multigroup(energy_edges, max(T_bc_ref, 1e-8))
    group_emission = group_emission / (np.sum(group_emission) + 1e-300)
    group_emission = group_emission * sigma_fine.max() * 0.5 / (group_emission.max() + 1e-300)

    for g in range(n_groups):
        sigma_g_val = float(np.mean(sigma_a_funcs[g](T_bc_ref)))
        ax_op.hlines(sigma_g_val, energy_edges[g], energy_edges[g + 1], colors="tab:blue", linewidths=2.5)
        ax_op.hlines(group_emission[g], energy_edges[g], energy_edges[g + 1], colors="tab:orange", linewidths=2.5)

    ax_op.hlines([], [], [], colors="tab:blue", linewidths=2.5, label=r"Group-averaged $\sigma_{a,g}$")
    ax_op.hlines([], [], [], colors="tab:orange", linewidths=2.5, label=r"Group-averaged $B_g$")
    ax_op.set_xlabel("Photon energy (keV)", fontsize=12)
    ax_op.set_ylabel(r"Opacity $\sigma_a$ (cm$^{-1}$)", fontsize=12)
    ax_op.set_title(f"Opacity vs. Energy at $T_b = {T_bc_ref:.3f}$ keV", fontsize=13, fontweight="bold")
    ax_op.legend(fontsize=11)
    ax_op.grid(True, which="both", ls="--", alpha=0.4)
    fig_op.tight_layout()
    plt.savefig("imc_opacity_vs_energy_Tb.png", dpi=150)
    plt.close(fig_op)
    print("Saved: imc_opacity_vs_energy_Tb.png")

    # Initial conditions.
    T_init = 0.005
    Tinit = np.full((nx, ny), T_init)
    Tr_init = np.full((nx, ny), T_init)

    def eos(T):
        return cv * T

    def inv_eos(e):
        return e / cv

    def cv_func(T):
        return cv * np.ones_like(T)

    # No fixed volumetric source.
    source = np.zeros((n_groups, nx, ny))

    # Left boundary emits; right boundary uses cold incoming T_init to mirror DO.
    T_boundary = (T_bc_func, T_init, 0.0, 0.0)
    reflect = (False, False, True, True)

    print("\nRunning IMC simulation...")
    print(f"  Domain: [{x_min}, {x_max}] cm with {nx} cells")
    print(f"  dt: {dt} ns, final_time: {final_time} ns")
    if grid_beta > 0.0:
        dx = np.diff(x_edges)
        print(f"  Grid: left-clustered exponential map (beta={grid_beta:.3f})")
        print(f"  dx_min={dx.min():.4e} cm, dx_max={dx.max():.4e} cm, dx_max/dx_min={dx.max()/dx.min():.2f}")
    else:
        print("  Grid: uniform")
    print(f"  Particles: Ntarget={ntarget}, Nboundary={nboundary}, Nmax={nmax}")
    if checkpoint_every > 0:
        print(f"  Checkpoint cadence: every {checkpoint_every} steps")
    else:
        print("  Checkpoint cadence: disabled")

    # Boundary emission scale diagnostic (per step, left boundary only).
    # This helps interpret the solver table, which prints fixed-point values.
    left_area = y_edges[-1] - y_edges[0]
    print("  Expected left-boundary emission per step (scientific notation):")
    for t_probe in (0.0, 0.05, 0.1, 1.0, 5.0):
        Tb_probe = T_bc_func(t_probe)
        E_probe = A_RAD * C_LIGHT * Tb_probe**4 / 4.0 * left_area * dt
        print(f"    t={t_probe:4.2f} ns: T_bc={Tb_probe:.4f} keV -> E_step={E_probe:.6e} GJ")

    output_freq = max(1, int(np.ceil(max(target_times) / dt)) // 200)
    metadata = {
        "n_groups": int(n_groups),
        "nx": int(nx),
        "dt": float(dt),
        "final_time": float(final_time),
        "time_dependent_bc": bool(time_dependent_bc),
        "grid_beta": float(grid_beta),
        "use_scalar_intensity_Tr": bool(use_scalar_intensity_Tr),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "energy_edges": energy_edges,
    }

    if checkpoint_file is None:
        checkpoint_file = f"marshak_wave_multigroup_powerlaw_imc_{n_groups}g_checkpoint.pkl"

    if restart_from is not None:
        print(f"Restarting from checkpoint: {restart_from}")
        payload = load_checkpoint(restart_from)
        saved_meta = payload.get("metadata", {})
        if int(saved_meta.get("n_groups", n_groups)) != int(n_groups):
            raise ValueError("Checkpoint group count does not match current --groups.")
        if int(saved_meta.get("nx", nx)) != int(nx):
            raise ValueError("Checkpoint nx does not match current --nx.")
        if abs(float(saved_meta.get("dt", dt)) - float(dt)) > 1e-14:
            raise ValueError("Checkpoint dt does not match current --dt.")
        if abs(float(saved_meta.get("grid_beta", grid_beta)) - float(grid_beta)) > 1e-14:
            raise ValueError("Checkpoint grid_beta does not match current --grid-beta.")
        if bool(saved_meta.get("time_dependent_bc", time_dependent_bc)) != bool(time_dependent_bc):
            raise ValueError("Checkpoint boundary mode does not match current --no-time-bc setting.")
        state = _deserialize_state(payload["state"])
        step_count = int(payload.get("step_count", 0))
        cumulative_residual = float(payload.get("cumulative_residual", 0.0))
        history = payload.get("history", [])
        np_state = payload.get("np_random_state", None)
        if np_state is not None:
            np.random.set_state(np_state)
        py_state = payload.get("py_random_state", None)
        if py_state is not None:
            random.setstate(py_state)
        print(
            f"Restart state: t={state.time:.6f} ns, step={step_count}, "
            f"history_entries={len(history)}"
        )
    else:
        state = init_simulation(
            Ntarget=ntarget,
            Tinit=Tinit,
            Tr_init=Tr_init,
            edges1=x_edges,
            edges2=y_edges,
            energy_edges=energy_edges,
            eos=eos,
            inv_eos=inv_eos,
            Ntarget_ic=ntarget,
            geometry="xy",
        )
        step_count = 0
        cumulative_residual = 0.0
        history = []

    t = float(state.time)
    time_tol = max(1e-15, 1e-12 * max(final_time, 1.0))
    n_steps_total = int(np.ceil((final_time - t) / dt))
    _wall_start = _time.perf_counter()
    _step_wall_start = _wall_start
    print(
        f"{'Step':>6}  {'t (ns)':>10}  {'dt (ns)':>9}  {'N_par':>8}  "
        f"{'E_mat (GJ)':>12}  {'E_rad (GJ)':>12}  {'wall/step (s)':>14}  {'ETA (s)':>9}"
    )
    print("-" * 95)
    while t < final_time - time_tol:
        dt_step = min(dt, final_time - t)
        if dt_step <= time_tol:
            break

        state, info = step(
            state,
            Ntarget=ntarget,
            Nboundary=nboundary,
            Nsource=0,
            Nmax=nmax,
            T_boundary=T_boundary,
            dt=dt_step,
            edges1=x_edges,
            edges2=y_edges,
            energy_edges=energy_edges,
            sigma_a_funcs=sigma_a_funcs,
            inv_eos=inv_eos,
            cv=cv_func,
            source=source,
            reflect=reflect,
            theta=1.0,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=False,
            geometry="xy",
            max_events_per_particle=100000,
        )

        t = float(state.time)
        step_count += 1
        cumulative_residual += float(info.get("energy_residual", 0.0))

        _now = _time.perf_counter()
        wall_step = _now - _step_wall_start
        _step_wall_start = _now
        elapsed = _now - _wall_start
        steps_done = step_count
        steps_left = max(0, int(np.ceil((final_time - t) / dt)))
        eta = (elapsed / steps_done) * steps_left if steps_done > 0 else 0.0
        E_mat = float(info.get("total_internal_energy", 0.0))
        E_rad = float(info.get("total_radiation_energy", 0.0))
        N_par = int(info.get("N_particles", 0))
        print(
            f"{step_count:>6}  {t:>10.5f}  {dt_step:>9.5f}  {N_par:>8d}  "
            f"{E_mat:>12.4e}  {E_rad:>12.4e}  {wall_step:>14.3f}  {eta:>9.1f}",
            flush=True,
        )

        if step_count % output_freq == 0 or (final_time - t) < time_tol:
            info["cumulative_energy_residual"] = cumulative_residual
            info["net_boundary_energy"] = info["boundary_emission"] - info["boundary_outgoing"]
            history.append(info)
            print(
                "{:.6e}".format(t),
                info["N_particles"],
                "{:.6e}".format(info["total_energy"]),
                "{:.6e}".format(info["total_internal_energy"]),
                "{:.6e}".format(info["total_radiation_energy"]),
                "{:.6e}".format(info["boundary_emission"]),
                "{:.6e}".format(info["boundary_outgoing"]),
                "{:.6e}".format(info["source_emission"]),
                "{:.6e}".format(info["energy_residual"]),
                sep="\t",
            )

        if checkpoint_every > 0 and (step_count % checkpoint_every == 0):
            save_checkpoint(
                checkpoint_file,
                state,
                step_count,
                cumulative_residual,
                history,
                metadata,
            )
            print(f"Saved checkpoint: {checkpoint_file} (step={step_count}, t={t:.6f} ns)")

    if checkpoint_every > 0:
        save_checkpoint(
            checkpoint_file,
            state,
            step_count,
            cumulative_residual,
            history,
            metadata,
        )
        print(f"Saved final checkpoint: {checkpoint_file}")

    final_state = state

    # Print early-time diagnostics with scientific notation so tiny-but-nonzero
    # values are visible (fixed-point table rounds many of these to 0.000000).
    print("\nEarly-time diagnostic (scientific notation):")
    for info in history[: min(8, len(history))]:
        print(
            f"  t={info['time']:.4f} ns, "
            f"E_rad_total={info['total_radiation_energy']:.6e} GJ, "
            f"E_boundary_step={info['boundary_emission']:.6e} GJ"
        )

    # Boundary energy ledger: track outgoing per step and by side.
    # Side order for xy geometry is [left, right, bottom, top].
    if len(history) > 0:
        print("\nBoundary energy ledger (per step):")
        print(
            "  step   t(ns)      E_in_step     E_out_step    "
            "E_out_L      E_out_R      E_out_B      E_out_T      "
            "E_net_step    cum_residual"
        )
        cumulative_in = 0.0
        cumulative_out = 0.0
        for i, info in enumerate(history, start=1):
            e_in = float(info.get("boundary_emission", 0.0))
            e_out = float(info.get("boundary_outgoing", info.get("boundary_loss", 0.0)))
            out_side = np.asarray(info.get("boundary_outgoing_by_side", np.zeros(4)), dtype=float)
            if out_side.size < 4:
                tmp = np.zeros(4)
                tmp[:out_side.size] = out_side
                out_side = tmp
            cumulative_in += e_in
            cumulative_out += e_out
            e_net = e_in - e_out
            print(
                f"  {i:4d}  {info['time']:8.4f}  {e_in:11.4e}  {e_out:11.4e}  "
                f"{out_side[0]:11.4e}  {out_side[1]:11.4e}  "
                f"{out_side[2]:11.4e}  {out_side[3]:11.4e}  "
                f"{e_net:11.4e}  {info.get('cumulative_energy_residual', np.nan):+11.4e}"
            )

        cumulative_net = cumulative_in - cumulative_out
        left_out = float(np.sum([
            np.asarray(h.get("boundary_outgoing_by_side", np.zeros(4)), dtype=float)[0]
            if np.asarray(h.get("boundary_outgoing_by_side", np.zeros(4)), dtype=float).size > 0 else 0.0
            for h in history
        ]))
        right_out = float(np.sum([
            np.asarray(h.get("boundary_outgoing_by_side", np.zeros(4)), dtype=float)[1]
            if np.asarray(h.get("boundary_outgoing_by_side", np.zeros(4)), dtype=float).size > 1 else 0.0
            for h in history
        ]))

        print("\nBoundary energy summary:")
        print(f"  Cumulative incoming:         {cumulative_in:.6e} GJ")
        print(f"  Cumulative outgoing:         {cumulative_out:.6e} GJ")
        print(f"  Cumulative net boundary:     {cumulative_net:.6e} GJ")
        print(f"  Outgoing through left side:  {left_out:.6e} GJ")
        print(f"  Outgoing through right side: {right_out:.6e} GJ")
        if cumulative_out > 0.0:
            print(f"  Left fraction of outgoing:   {left_out / cumulative_out:.3f}")
            print(f"  Right fraction of outgoing:  {right_out / cumulative_out:.3f}")

    # Save snapshots at target times.
    solutions = []
    solutions_postcomb = []
    for t_target in target_times:
        if len(history) == 0:
            continue
        idx = int(np.argmin([abs(h["time"] - t_target) for h in history]))
        info = history[idx]

        E_r_groups = info["radiation_energy_by_group"][:, :, 0].copy()
        E_r_groups_postcomb = None
        if "radiation_energy_by_group_postcomb" in info:
            E_r_groups_postcomb = info["radiation_energy_by_group_postcomb"][:, :, 0].copy()
        E_r = np.sum(E_r_groups, axis=0)
        T_mat = info["temperature"][:, 0].copy()
        T_rad = info["radiation_temperature"][:, 0].copy()
        phi_groups = E_r_groups * C_LIGHT

        solutions.append(
            {
                "time": info["time"],
                "r": x_centers.copy(),
                "T": T_mat,
                "E_r": E_r,
                "T_rad": T_rad,
                "phi_groups": phi_groups,
                "E_r_groups": E_r_groups,
            }
        )
        if E_r_groups_postcomb is not None:
            solutions_postcomb.append(E_r_groups_postcomb)
        print(
            f"Saved snapshot t={info['time']:.3f} ns, "
            f"T_max={T_mat.max():.5f} keV, E_r_max={E_r.max():.5e}"
        )

    if not solutions:
        raise RuntimeError("No snapshots were recorded; history is empty.")

    # Build structured arrays for saving (same style as diffusion script).
    times_arr = np.array([s["time"] for s in solutions])
    r_arr = solutions[0]["r"]
    T_mat_arr = np.array([s["T"] for s in solutions])
    T_rad_arr = np.array([s["T_rad"] for s in solutions])
    E_r_arr = np.array([s["E_r"] for s in solutions])
    phi_groups_arr = np.array([s["phi_groups"] for s in solutions])
    E_r_groups_arr = np.array([s["E_r_groups"] for s in solutions])
    E_r_groups_postcomb_arr = np.array(solutions_postcomb) if len(solutions_postcomb) == len(solutions) else None

    base = f"marshak_wave_multigroup_powerlaw_imc_{n_groups}g{'_timeBC' if time_dependent_bc else ''}"

    save_kwargs = dict(
        times=times_arr,
        r=r_arr,
        energy_edges=energy_edges,
        T_mat=T_mat_arr,
        T_rad=T_rad_arr,
        E_r=E_r_arr,
        phi_groups=phi_groups_arr,
        E_r_groups=E_r_groups_arr,
    )
    if E_r_groups_postcomb_arr is not None:
        save_kwargs["E_r_groups_postcomb"] = E_r_groups_postcomb_arr

    np.savez(f"{base}.npz", **save_kwargs)
    print(f"Saved: {base}.npz")

    # Plot styling to mirror diffusion output.
    colors = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "brown", "olive", "teal"]

    # Figure 1: Material and radiation temperatures (linear).
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    for idx, sol in enumerate(solutions):
        c = colors[idx % len(colors)]
        ax1.plot(sol["r"], sol["T"], color=c, linewidth=2, label="material" if idx == 0 else None)
        ax1.plot(sol["r"], sol["T_rad"], color=c, linewidth=2, linestyle="--", label="radiation" if idx == 0 else None)
    ax1.set_xlabel("position (cm)", fontsize=12)
    ax1.set_ylabel("temperature (keV)", fontsize=12)
    ax1.legend(prop=font, facecolor="white", edgecolor="none", fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    show(f"{base}_T_mat.pdf", close_after=True)
    print(f"Saved: {base}_T_mat.pdf")

    # Figure 2: Material and radiation temperatures (log-log).
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for idx, sol in enumerate(solutions):
        c = colors[idx % len(colors)]
        ax2.plot(sol["r"], sol["T"], color=c, linewidth=2, label="material" if idx == 0 else None)
        ax2.plot(sol["r"], sol["T_rad"], color=c, linewidth=2, linestyle="--", label="radiation" if idx == 0 else None)
    ax2.set_xlabel("position (cm)", fontsize=12)
    ax2.set_ylabel("temperature (keV)", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(prop=font, facecolor="white", edgecolor="none", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    show(f"{base}_T_log.pdf", close_after=True)
    print(f"Saved: {base}_T_log.pdf")

    # Figure 3: Total radiation energy density.
    fig3, ax3 = plt.subplots(figsize=(7.5, 5.25))
    for idx, sol in enumerate(solutions):
        c = colors[idx % len(colors)]
        ax3.semilogy(sol["r"], sol["E_r"], color=c, linewidth=2, label=f"t = {sol['time']:.1f} ns")
    ax3.set_xlabel("Position (cm)", fontsize=12)
    ax3.set_ylabel(r"Radiation Energy (GJ/cm$^3$)", fontsize=12)
    ax3.legend(prop=font)
    ax3.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    show(f"{base}_E_rad.pdf", close_after=True)
    print(f"Saved: {base}_E_rad.pdf")

    # Figure 4: Spectral energy density by group at each saved time.
    E_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    fig4, ax4 = plt.subplots(figsize=(7.5, 5.25))
    for idx, sol in enumerate(solutions):
        spec = np.mean(sol["E_r_groups"], axis=1)
        c = colors[idx % len(colors)]
        ax4.loglog(E_mid, spec, marker="o", color=c, linewidth=1.8, label=f"t = {sol['time']:.1f} ns")
    ax4.set_xlabel("Photon energy (keV)", fontsize=12)
    ax4.set_ylabel(r"Mean $E_{r,g}$ (GJ/cm$^3$)", fontsize=12)
    ax4.legend(prop=font)
    ax4.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    show(f"{base}_spectrum.pdf", close_after=True)
    print(f"Saved: {base}_spectrum.pdf")

    print("\nSimulation complete")
    print(f"Final time: {final_state.time:.3f} ns")
    print(f"Final particles: {len(final_state.weights)}")

    return final_state, solutions


def main():
    parser = argparse.ArgumentParser(description="Marshak wave multigroup IMC with power-law opacity")
    parser.add_argument("--groups", type=int, default=10, help="Number of energy groups")
    parser.add_argument("--no-time-bc", action="store_true", help="Disable time-dependent left boundary temperature")
    parser.add_argument("--Ntarget", type=int, default=500_000, help="Material emission particles per step")
    parser.add_argument("--Nboundary", type=int, default=500_000, help="Boundary source particles per side per step")
    parser.add_argument("--Nmax", type=int, default=1_000_000, help="Census comb target")
    parser.add_argument("--nx", type=int, default=140, help="Number of x-cells (default: 140)")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep in ns (default: 0.01)")
    parser.add_argument("--final-time", type=float, default=10.0, help="Final time in ns (default: 10.0)")
    parser.add_argument(
        "--grid-beta",
        type=float,
        default=0.0,
        help=(
            "Left-boundary clustering strength for x-grid. "
            "0 = uniform; positive values cluster near x=0 (default: 0)."
        ),
    )
    parser.add_argument(
        "--use-particle-binning-Tr",
        action="store_true",
        help="Use particle binning instead of scalar-intensity estimator for Tr",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N timesteps (default: 10, <=0 disables)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to checkpoint file (default: auto-generated from group count)",
    )
    parser.add_argument(
        "--restart-from",
        type=str,
        default=None,
        help="Checkpoint file to restart from",
    )
    args = parser.parse_args()

    run_marshak_wave_multigroup_powerlaw_imc(
        n_groups=args.groups,
        time_dependent_bc=not args.no_time_bc,
        ntarget=args.Ntarget,
        nboundary=args.Nboundary,
        nmax=args.Nmax,
        use_scalar_intensity_Tr=not args.use_particle_binning_Tr,
        nx=args.nx,
        dt=args.dt,
        final_time=args.final_time,
        grid_beta=args.grid_beta,
        checkpoint_every=args.checkpoint_every,
        checkpoint_file=args.checkpoint_file,
        restart_from=args.restart_from,
    )


if __name__ == "__main__":
    main()
