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
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for package imports and plotting utilities.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MG_IMC import A_RAD, C_LIGHT, run_simulation

from utils.plotfuncs import font, show

RHO = 0.01  # g/cm^3


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

    # Boundary emission scale diagnostic (per step, left boundary only).
    # This helps interpret the solver table, which prints fixed-point values.
    left_area = y_edges[-1] - y_edges[0]
    print("  Expected left-boundary emission per step (scientific notation):")
    for t_probe in (0.0, 0.05, 0.1, 1.0, 5.0):
        Tb_probe = T_bc_func(t_probe)
        E_probe = A_RAD * C_LIGHT * Tb_probe**4 / 4.0 * left_area * dt
        print(f"    t={t_probe:4.2f} ns: T_bc={Tb_probe:.4f} keV -> E_step={E_probe:.6e} GJ")

    history, final_state = run_simulation(
        Ntarget=ntarget,
        Nboundary=nboundary,
        Nsource=0,
        Nmax=nmax,
        Tinit=Tinit,
        Tr_init=Tr_init,
        T_boundary=T_boundary,
        dt=dt,
        edges1=x_edges,
        edges2=y_edges,
        energy_edges=energy_edges,
        sigma_a_funcs=sigma_a_funcs,
        eos=eos,
        inv_eos=inv_eos,
        cv=cv_func,
        source=source,
        final_time=final_time,
        reflect=reflect,
        output_freq=max(1, int(np.ceil(max(target_times) / dt)) // 200),
        theta=1.0,
        use_scalar_intensity_Tr=use_scalar_intensity_Tr,
        Ntarget_ic=ntarget,
        conserve_comb_energy=False,
        geometry="xy",
        max_events_per_particle=1_000_000,
    )

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
    )


if __name__ == "__main__":
    main()
