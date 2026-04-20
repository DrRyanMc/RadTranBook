#!/usr/bin/env python3
"""
Infinite-medium multigroup IMC test with reflecting boundaries.

Requested setup:
- Energy range: [1e-4, 40.0] keV
- Group opacity: normalized Planck-band form
    sigma_g(T) = sigma0*(exp(-v1/T)-exp(-v2/T)) / (sqrt(T)*N(v1,v2,T))
    where N uses the Li4/Li3/Li2/log normalization expression.
- sigma0 = 10
- Initial group radiation energy density:
    E_g = (15 a Trad^4 / (pi^4 Tc^3)) * (
                    6 Tc^3 [Li4(exp(-v1/Tc)) - Li4(exp(-v2/Tc))]
                + 6 Tc^2 [v1 Li3(exp(-v1/Tc)) - v2 Li3(exp(-v2/Tc))]
                + 3 Tc   [v1^2 Li2(exp(-v1/Tc)) - v2^2 Li2(exp(-v2/Tc))]
                - v1^3 log(1 - exp(-v1/Tc))
                + v2^3 log(1 - exp(-v2/Tc))
            )
  with Tc = 1 keV, Trad = 0.5 keV.
- Initial material temperature: 0.4 keV
- rho*Cv = 0.01 GJ/(cm^3 keV)

Implementation notes:
- Uses a single spatial cell with all boundaries reflecting as an infinite-medium proxy.
- Uses MG_IMC2D.step directly so the initial per-group radiation energy can be set exactly.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Keep numba cache local-safe for quick test scripts.
os.environ.setdefault("NUMBA_CACHE_DIR", "")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from MG_IMC2D import (  # pylint: disable=wrong-import-position
    __a,
    init_simulation,
    step,
    _sample_isotropic_xy,
)

try:
    from scipy.special import polylog as _scipy_polylog

    def _polylog(n, x):
        return np.asarray(_scipy_polylog(n, x), dtype=float)

except Exception:  # pragma: no cover
    try:
        from scipy.special import spence as _spence

        def _li2_spence(x):
            # scipy.special.spence(y) = Li2(1-y)
            return _spence(1.0 - x)

    except Exception:
        _li2_spence = None

    import mpmath as _mp

    def _polylog(n, x):
        if n == 2 and _li2_spence is not None:
            return np.asarray(_li2_spence(x), dtype=float)
        if np.isscalar(x):
            return float(_mp.polylog(n, x))
        out = np.empty_like(x, dtype=float)
        it = np.nditer(x, flags=["multi_index"])
        for val in it:
            out[it.multi_index] = float(_mp.polylog(n, float(val)))
        return out


def _li2(x):
    return _polylog(2, x)


def _li3(x):
    return _polylog(3, x)


def _li4(x):
    return _polylog(4, x)


def _log_one_minus_exp_neg(x):
    """Stable log(1 - exp(-x)) for x > 0."""
    return np.log(-np.expm1(-x))


def group_energy_density(v1, v2, Tc, Trad, a_rad):
    """Requested analytic initial group radiation energy density."""
    z1 = np.exp(-v1 / Tc)
    z2 = np.exp(-v2 / Tc)
    bracket = (
        6.0 * Tc**3 * (_li4(z1) - _li4(z2))
        + 6.0 * Tc**2 * (v1 * _li3(z1) - v2 * _li3(z2))
        + 3.0 * Tc * (v1**2 * _li2(z1) - v2**2 * _li2(z2))
        - v1**3 * _log_one_minus_exp_neg(v1 / Tc)
        + v2**3 * _log_one_minus_exp_neg(v2 / Tc)
    )
    return (15.0 * a_rad * Trad**4 / (np.pi**4 * Tc**3)) * bracket


def make_group_opacity(v1, v2, sigma0=10.0, t_floor=1e-8):
    """Create normalized sigma_g(T) for group [v1, v2]."""

    def sigma_func(T):
        T_use = np.maximum(T, t_floor)

        z1 = np.exp(-v1 / T_use)
        z2 = np.exp(-v2 / T_use)

        norm_inner = (
            3.0
            * T_use
            * (
                v1**2 * _li2(z1)
                + 2.0 * T_use * (v1 * _li3(z1) + T_use * _li4(z1) - v2 * _li3(z2) - T_use * _li4(z2))
                - v2**2 * _li2(z2)
            )
            - v1**3 * _log_one_minus_exp_neg(v1 / T_use)
            + v2**3 * _log_one_minus_exp_neg(v2 / T_use)
        )

        denom = np.sqrt(T_use) * norm_inner
        denom_safe = np.where(denom > 1e-300, denom, np.inf)
        val = sigma0 * (z1 - z2) / denom_safe
        return np.maximum(val, 0.0)

    return sigma_func


def analytic_monochromatic_opacity(nu, T, sigma0=10.0, t_floor=1e-8):
    """Analytic opacity from the reference: sigma_a'(nu)=sigma0*(1-exp(-nu/T))/(nu^3*sqrt(T))."""
    T_use = max(float(T), t_floor)
    nu_use = np.maximum(np.asarray(nu, dtype=float), 1e-300)
    return sigma0 * (1.0 - np.exp(-nu_use / T_use)) / (nu_use**3 * np.sqrt(T_use))


def analytic_initial_spectral_energy_density(nu, Tc, Trad, a_rad):
    """Analytic initial spectral energy density E_nu = phi_nu / c."""
    nu_use = np.maximum(np.asarray(nu, dtype=float), 1e-300)
    scale = a_rad * (Trad**4 / Tc**4) * (15.0 / np.pi**4)
    return scale * nu_use**3 / np.expm1(nu_use / Tc)


def configure_initial_state_from_group_energies(state, group_energy_density_vals, edges1, edges2, n_particles):
    """Overwrite state particle field so each group matches target initial energy exactly."""
    nx = len(edges1) - 1
    ny = len(edges2) - 1
    assert nx == 1 and ny == 1, "This helper is intended for the 1-cell infinite-medium setup"

    volume = (edges1[1] - edges1[0]) * (edges2[1] - edges2[0])
    group_energies = group_energy_density_vals * volume

    total_energy = np.sum(group_energies)
    if total_energy <= 0.0:
        raise ValueError("Total initial radiation energy is non-positive")

    probs = group_energies / total_energy
    groups = np.random.choice(len(group_energies), size=n_particles, p=probs).astype(np.int32)

    # Ensure every group with non-zero target energy has at least one particle.
    nonzero_groups = np.where(group_energies > 0.0)[0]
    for g in nonzero_groups:
        if not np.any(groups == g):
            groups[np.random.randint(0, n_particles)] = g

    counts = np.bincount(groups, minlength=len(group_energies))
    weights = np.zeros(n_particles, dtype=float)
    for g in range(len(group_energies)):
        mask = groups == g
        if counts[g] > 0:
            weights[mask] = group_energies[g] / counts[g]

    ux, uy = _sample_isotropic_xy(n_particles)

    x = np.random.uniform(edges1[0], edges1[1], n_particles)
    y = np.random.uniform(edges2[0], edges2[1], n_particles)

    state.weights = weights
    state.dir1 = ux
    state.dir2 = uy
    state.times = np.zeros(n_particles)
    state.pos1 = x
    state.pos2 = y
    state.cell_i = np.zeros(n_particles, dtype=np.int32)
    state.cell_j = np.zeros(n_particles, dtype=np.int32)
    state.groups = groups

    n_groups = len(group_energies)
    radiation_energy_by_group = np.zeros((n_groups, nx, ny), dtype=float)
    for g in range(n_groups):
        radiation_energy_by_group[g, 0, 0] = np.sum(weights[groups == g]) / volume

    state.radiation_energy_by_group = radiation_energy_by_group
    total_rad_density = np.sum(radiation_energy_by_group, axis=0)
    state.radiation_temperature = (total_rad_density / __a) ** 0.25

    total_internal = float(np.sum(state.internal_energy) * volume)
    total_radiation = float(np.sum(weights))
    state.previous_total_energy = total_internal + total_radiation


def _nearest_time_indices(times, requested_times):
    """Return indices in `times` nearest to each requested time."""
    indices = []
    for t_req in requested_times:
        idx = int(np.argmin(np.abs(times - t_req)))
        if idx not in indices:
            indices.append(idx)
    return indices


def make_plots(times, t_mat, t_rad, energy_edges, group_energy_history, requested_times=(0.0, 0.01, 0.1, 1.0), n_groups=50):
    """Create the two requested plots: temperature evolution and group spectrum."""
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    dE = np.diff(energy_edges)
    plot_indices = _nearest_time_indices(times, requested_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 10.2))

    # (a) Material and radiation temperature vs time
    ax1.plot(times, t_mat, color="tab:blue", linewidth=1.6, label="T")
    ax1.plot(times, t_rad, color="tab:orange", linestyle="--", linewidth=1.5, label=r"$T_r$")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("Temperature (keV)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # (b) Spectrum as spectral energy density at selected times
    styles = ["-", "--", "-.", ":"]
    for line_idx, hist_idx in enumerate(plot_indices):
        t_label = times[hist_idx]
        ax2.plot(
            e_mid,
            group_energy_history[hist_idx] / dE,
            linestyle=styles[line_idx % len(styles)],
            linewidth=1.6,
            label=f"t={t_label:.2f} ns",
        )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Photon Energy, $E_\nu$ (keV)")
    ax2.set_ylabel(r"Spectral energy density $E_\nu$ (GJ/cm$^3$/keV)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    out_png = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_plots.png"
    out_pdf = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_plots.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def make_opacity_comparison_plot(energy_edges, sigma_a_funcs, T_compare, sigma0=10.0, n_groups=50):
    """Plot multigroup opacity vs analytic monochromatic opacity on log-log axes."""
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    sigma_group = np.array([float(np.asarray(func(T_compare))) for func in sigma_a_funcs])

    nu_dense = np.logspace(np.log10(energy_edges[0]), np.log10(energy_edges[-1]), 800)
    sigma_analytic = analytic_monochromatic_opacity(nu_dense, T=T_compare, sigma0=sigma0)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.loglog(nu_dense, sigma_analytic, color="tab:red", linewidth=1.8, label="Analytic $\\sigma'_a(\\nu)$")
    ax.step(e_mid, sigma_group, where="mid", color="tab:blue", linewidth=1.6, label="Multigroup $\\sigma_g$")
    ax.set_xlabel(r"Photon Energy, $E_\nu = \nu$ (keV)")
    ax.set_ylabel(r"Opacity")
    ax.set_title(f"Opacity Comparison at T={T_compare:.3g} keV")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    out_png = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_opacity_compare.png"
    out_pdf = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_opacity_compare.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def make_initial_spectrum_comparison_plot(energy_edges, sampled_group_energy_density, Tc, Trad, a_rad, n_groups=50):
    """Compare sampled initial group spectrum against analytic E_nu = phi_nu / c."""
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    dE = np.diff(energy_edges)
    sampled_spectral_density = sampled_group_energy_density / dE

    nu_dense = np.logspace(np.log10(energy_edges[0]), np.log10(energy_edges[-1]), 1200)
    analytic_dense = analytic_initial_spectral_energy_density(nu_dense, Tc=Tc, Trad=Trad, a_rad=a_rad)
    analytic_group_avg = group_energy_density(energy_edges[:-1], energy_edges[1:], Tc=Tc, Trad=Trad, a_rad=a_rad) / dE

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.plot(nu_dense, analytic_dense, color="tab:red", linewidth=1.8, label=r"Analytic $\phi_\nu/c$")
    ax.step(e_mid, sampled_spectral_density, where="mid", color="tab:blue", linewidth=1.6, label="Sampled initial spectrum")
    ax.step(e_mid, analytic_group_avg, where="mid", color="tab:green", linestyle="--", linewidth=1.3, label="Analytic group average")
    ax.set_xlabel(r"Photon Energy, $E_\nu = \nu$ (keV)")
    ax.set_ylabel(r"Spectral energy density $E_\nu$ (GJ/cm$^3$/keV)")
    ax.set_title("Initial Spectrum: Sampled vs Analytic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    out_png = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_initial_spectrum_compare.png"
    out_pdf = f"infinite_medium_multigroup_sigmaexp_{n_groups}g_initial_spectrum_compare.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def run_problem(
    n_groups=50,
    sigma0=10.0,
    Tc=1.0,
    Trad=0.5,
    Tmat0=0.4,
    rho_cv=0.01,
    dt_initial=1.0e-4,
    dt_max=1.0e-2,
    dt_growth=1.1,
    final_time=1.0,
    n_particles_ic=200000,
    n_target=50000,
):
    print("=" * 88)
    print("Infinite Medium Multigroup IMC Test")
    print("=" * 88)

    energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), n_groups + 1)

    # 1 cell, all reflecting -> infinite-medium proxy (no leakage, no gradients).
    edges1 = np.array([0.0, 1.0])
    edges2 = np.array([0.0, 1.0])

    sigma_a_funcs = [
        make_group_opacity(energy_edges[g], energy_edges[g + 1], sigma0=sigma0)
        for g in range(n_groups)
    ]

    # Requested material model: rho*Cv = 0.01 GJ/(cm^3 keV).
    eos = lambda T: rho_cv * T
    inv_eos = lambda e: e / rho_cv
    cv_func = lambda T: rho_cv + 0.0 * T

    # Initial material and radiation placeholders (state overwritten for radiation below).
    Tinit = np.full((1, 1), Tmat0)
    Tr_init = np.full((1, 1), Trad)

    state = init_simulation(
        Ntarget=n_particles_ic,
        Tinit=Tinit,
        Tr_init=Tr_init,
        edges1=edges1,
        edges2=edges2,
        energy_edges=energy_edges,
        eos=eos,
        inv_eos=inv_eos,
        Ntarget_ic=n_particles_ic,
        geometry="xy",
    )

    Eg0 = group_energy_density(energy_edges[:-1], energy_edges[1:], Tc=Tc, Trad=Trad, a_rad=__a)
    configure_initial_state_from_group_energies(
        state,
        group_energy_density_vals=Eg0,
        edges1=edges1,
        edges2=edges2,
        n_particles=n_particles_ic,
    )

    make_initial_spectrum_comparison_plot(
        energy_edges=energy_edges,
        sampled_group_energy_density=state.radiation_energy_by_group[:, 0, 0].copy(),
        Tc=Tc,
        Trad=Trad,
        a_rad=__a,
        n_groups=n_groups,
    )

    print(f"Groups: {n_groups}")
    print(f"Energy range: [{energy_edges[0]:.3e}, {energy_edges[-1]:.3e}] keV")
    print(f"sigma0 = {sigma0}")
    print(f"Tc = {Tc} keV, Trad = {Trad} keV, Tmat0 = {Tmat0} keV")
    print(f"rho*Cv = {rho_cv} GJ/(cm^3 keV)")
    print(f"dt schedule: dt0={dt_initial:.3e} ns, dt_max={dt_max:.3e} ns, growth={dt_growth:.3f}")
    print(f"Initial total radiation energy density = {Eg0.sum():.6e} GJ/cm^3")

    source = np.zeros((1, 1), dtype=float)
    t_boundary = [0.0, 0.0, 0.0, 0.0]

    times = [state.time]
    t_mat = [float(state.temperature[0, 0])]
    t_rad = [float(state.radiation_temperature[0, 0])]
    group_energy_history = [state.radiation_energy_by_group[:, 0, 0].copy()]
    initial_total_energy = float(state.previous_total_energy)

    step_idx = 0
    dt_current = max(float(dt_initial), 1e-14)
    dt_cap = max(float(dt_max), dt_current)
    growth = max(float(dt_growth), 1.0)

    while state.time < final_time - 1e-14:
        step_idx += 1
        dt_step = min(dt_current, dt_cap, final_time - state.time)
        if dt_step <= 0.0:
            break

        state, info = step(
            state=state,
            Ntarget=n_target,
            Nboundary=0,
            Nsource=0,
            Nmax=(n_target+n_particles_ic)*2,
            T_boundary=t_boundary,
            dt=dt_step,
            edges1=edges1,
            edges2=edges2,
            energy_edges=energy_edges,
            sigma_a_funcs=sigma_a_funcs,
            inv_eos=inv_eos,
            cv=cv_func,
            source=source,
            reflect=(True, True, True, True),
            theta=1.0,
            use_scalar_intensity_Tr=False,
            conserve_comb_energy=True,
            geometry="xy",
            max_events_per_particle=10_000_000,
        )

        times.append(info["time"])
        t_mat.append(float(info["temperature"][0, 0]))
        t_rad.append(float(info["radiation_temperature"][0, 0]))
        group_energy_history.append(state.radiation_energy_by_group[:, 0, 0].copy())
        total_energy = float(info["total_energy"])
        rel_energy_drift = (total_energy - initial_total_energy) / max(abs(initial_total_energy), 1e-300)

        if step_idx <= 5 or step_idx % 10 == 0 or abs(info["time"] - final_time) < 1e-14:
            print(
                f"step {step_idx:5d}  t={info['time']:.4e} ns  dt={dt_step:.3e} ns  "
                f"T_mat={t_mat[-1]:.6f} keV  T_rad={t_rad[-1]:.6f} keV  "
                f"N={info['N_particles']}  E_tot={total_energy:.8e} GJ  dE/E0={rel_energy_drift:.3e}"
            )

        dt_current = min(dt_cap, dt_current * growth)

    np.savez(
        f"infinite_medium_multigroup_sigmaexp_{n_groups}g.npz",
        times=np.array(times),
        T_mat=np.array(t_mat),
        T_rad=np.array(t_rad),
        energy_edges=energy_edges,
        group_energy_history=np.array(group_energy_history),
        initial_group_energy_density=Eg0,
        final_group_energy_density=state.radiation_energy_by_group[:, 0, 0],
        final_material_temperature=state.temperature[0, 0],
    )
    print(f"Saved: infinite_medium_multigroup_sigmaexp_{n_groups}g.npz")

    make_plots(
        times=np.array(times),
        t_mat=np.array(t_mat),
        t_rad=np.array(t_rad),
        energy_edges=energy_edges,
        group_energy_history=np.array(group_energy_history),
        requested_times=(0.0, 0.01, 0.1, 1.0),
        n_groups=n_groups,
    )

    make_opacity_comparison_plot(
        energy_edges=energy_edges,
        sigma_a_funcs=sigma_a_funcs,
        T_compare=Tc,
        sigma0=sigma0,
        n_groups=n_groups,
    )

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infinite medium multigroup IMC test")
    parser.add_argument("--groups", type=int, default=50, help="Number of energy groups")
    parser.add_argument("--dt-initial", type=float, default=1.0e-4, help="Initial time step in ns")
    parser.add_argument("--dt-max", type=float, default=1.0e-2, help="Maximum time step in ns")
    parser.add_argument("--dt-growth", type=float, default=1.1, help="Multiplicative dt growth factor per step")
    parser.add_argument("--final-time", type=float, default=1.0, help="Final time in ns")
    parser.add_argument("--n-ic", type=int, default=200000, help="Initial particle count")
    parser.add_argument("--n-target", type=int, default=50000, help="Material emission particle target per step")
    args = parser.parse_args()

    run_problem(
        n_groups=args.groups,
        dt_initial=args.dt_initial,
        dt_max=args.dt_max,
        dt_growth=args.dt_growth,
        final_time=args.final_time,
        n_particles_ic=args.n_ic,
        n_target=args.n_target,
    )
