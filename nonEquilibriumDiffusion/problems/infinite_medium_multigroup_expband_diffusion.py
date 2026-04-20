#!/usr/bin/env python3
"""
Infinite-medium multigroup non-equilibrium diffusion test.

Mirrors the MG_IMC infinite-medium exp-band setup with:
- Energy range: [1e-4, 40.0] keV
- Group opacity: normalized Planck-band form
- Initial group radiation energy density from Li4/Li3/Li2/log expression
- Initial material temperature: 0.4 keV
- rho*Cv = 0.01 GJ/(cm^3 keV)
- Variable dt schedule: start small, grow geometrically to capped max
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NON_EQ_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, NON_EQ_ROOT)

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

try:
    from scipy.special import polylog as _scipy_polylog

    def _polylog(n, x):
        return np.asarray(_scipy_polylog(n, x), dtype=float)

except Exception:  # pragma: no cover
    try:
        from scipy.special import spence as _spence

        def _li2_spence(x):
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
    return np.log(-np.expm1(-x))


def group_energy_density(v1, v2, Tc, Trad, a_rad):
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
    def sigma_func(T, r):
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


def make_group_diffusion(v1, v2, sigma0=10.0, t_floor=1e-8):
    sigma_func = make_group_opacity(v1, v2, sigma0=sigma0, t_floor=t_floor)

    def diffusion_func(T, r):
        sigma = sigma_func(T, r)
        return C_LIGHT / (3.0 * np.maximum(sigma, 1e-30))

    return diffusion_func


def analytic_monochromatic_opacity(nu, T, sigma0=10.0, t_floor=1e-8):
    T_use = max(float(T), t_floor)
    nu_use = np.maximum(np.asarray(nu, dtype=float), 1e-300)
    return sigma0 * (1.0 - np.exp(-nu_use / T_use)) / (nu_use**3 * np.sqrt(T_use))


def _nearest_time_indices(times, requested_times):
    indices = []
    for t_req in requested_times:
        idx = int(np.argmin(np.abs(times - t_req)))
        if idx not in indices:
            indices.append(idx)
    return indices


def make_plots(times, t_mat, t_rad, energy_edges, group_energy_history, n_groups, requested_times=(0.0, 0.01, 0.1, 1.0)):
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    dE = np.diff(energy_edges)
    plot_indices = _nearest_time_indices(times, requested_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 10.2))

    ax1.plot(times, t_mat, color="tab:blue", linewidth=1.6, label="T")
    ax1.plot(times, t_rad, color="tab:orange", linestyle="--", linewidth=1.5, label=r"$T_r$")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("Temperature (keV)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

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
    out_png = f"infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g_plots.png"
    out_pdf = f"infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g_plots.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def make_opacity_comparison_plot(energy_edges, sigma_funcs, T_compare, sigma0, n_groups):
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    sigma_group = np.array([float(np.asarray(func(T_compare, 0.0))) for func in sigma_funcs])

    nu_dense = np.logspace(np.log10(energy_edges[0]), np.log10(energy_edges[-1]), 800)
    sigma_analytic = analytic_monochromatic_opacity(nu_dense, T=T_compare, sigma0=sigma0)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.loglog(nu_dense, sigma_analytic, color="tab:red", linewidth=1.8, label="Analytic $\\sigma'_a(\\nu)$")
    ax.step(e_mid, sigma_group, where="mid", color="tab:blue", linewidth=1.6, label="Multigroup $\\sigma_g$")
    ax.set_xlabel(r"Photon Energy, $E_\nu = \nu$ (keV)")
    ax.set_ylabel("Opacity")
    ax.set_title(f"Opacity Comparison at T={T_compare:.3g} keV")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    out_png = f"infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g_opacity_compare.png"
    out_pdf = f"infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g_opacity_compare.pdf"
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
    max_newton_iter=30,
    newton_tol=1e-8,
    gmres_tol=1e-8,
    gmres_maxiter=400,
    max_relative_change=0.25,
    use_preconditioner=True,
):
    print("=" * 88)
    print("Infinite Medium Multigroup Diffusion Test")
    print("=" * 88)

    energy_edges = np.logspace(np.log10(1e-4), np.log10(20.0), n_groups + 1)

    sigma_funcs = [
        make_group_opacity(energy_edges[g], energy_edges[g + 1], sigma0=sigma0)
        for g in range(n_groups)
    ]
    diff_funcs = [
        make_group_diffusion(energy_edges[g], energy_edges[g + 1], sigma0=sigma0)
        for g in range(n_groups)
    ]

    zero_flux_bc = lambda phi, r: (0.0, 1.0, 0.0)
    left_bc_funcs = [zero_flux_bc] * n_groups
    right_bc_funcs = [zero_flux_bc] * n_groups

    rho = 1.0
    cv = rho_cv / rho

    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=1.0,
        n_cells=1,
        energy_edges=energy_edges,
        geometry="planar",
        dt=dt_initial,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv,
    )

    Eg0 = group_energy_density(energy_edges[:-1], energy_edges[1:], Tc=Tc, Trad=Trad, a_rad=A_RAD)
    Eg0_total = float(np.sum(Eg0))

    solver.T = np.array([Tmat0], dtype=float)
    solver.T_old = solver.T.copy()
    solver.E_r = np.array([Eg0_total], dtype=float)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(1, dtype=float)
    solver.kappa_old = np.zeros(1, dtype=float)

    frac0 = Eg0 / max(Eg0_total, 1e-300)
    solver.phi_g_fraction[:, 0] = frac0
    solver.phi_g_stored[:, 0] = Eg0 * C_LIGHT

    print(f"Groups: {n_groups}")
    print(f"Energy range: [{energy_edges[0]:.3e}, {energy_edges[-1]:.3e}] keV")
    print(f"sigma0 = {sigma0}")
    print(f"Tc = {Tc} keV, Trad = {Trad} keV, Tmat0 = {Tmat0} keV")
    print(f"rho*Cv = {rho_cv} GJ/(cm^3 keV)")
    print(f"dt schedule: dt0={dt_initial:.3e} ns, dt_max={dt_max:.3e} ns, growth={dt_growth:.3f}")
    print(f"solver controls: newton_max={max_newton_iter}, gmres_max={gmres_maxiter}, precond={use_preconditioner}")
    print(f"Initial total radiation energy density = {Eg0_total:.6e} GJ/cm^3")

    times = [0.0]
    t_mat = [float(solver.T[0])]
    t_rad = [float((solver.E_r[0] / A_RAD) ** 0.25)]
    group_energy_history = [Eg0.copy()]
    material_energy_history = [rho_cv * t_mat[0]]
    radiation_energy_history = [float(solver.E_r[0])]
    total_energy_history = [material_energy_history[0] + radiation_energy_history[0]]
    total_energy_initial = total_energy_history[0]

    step_idx = 0
    dt_current = max(float(dt_initial), 1e-14)
    dt_cap = max(float(dt_max), dt_current)
    growth = max(float(dt_growth), 1.0)

    while solver.t < final_time - 1e-14:
        step_idx += 1
        dt_step = min(dt_current, dt_cap, final_time - solver.t)
        if dt_step <= 0.0:
            break

        solver.dt = dt_step
        info = solver.step(
            max_newton_iter=max_newton_iter,
            newton_tol=newton_tol,
            gmres_tol=gmres_tol,
            gmres_maxiter=gmres_maxiter,
            use_preconditioner=use_preconditioner,
            max_relative_change=max_relative_change,
            verbose=False,
        )
        solver.advance_time()

        times.append(float(solver.t))
        t_mat.append(float(solver.T[0]))
        t_rad.append(float((solver.E_r[0] / A_RAD) ** 0.25))
        group_energy_history.append((solver.phi_g_fraction[:, 0] * solver.E_r[0]).copy())
        e_mat = rho_cv * t_mat[-1]
        e_rad = float(solver.E_r[0])
        e_tot = e_mat + e_rad
        material_energy_history.append(e_mat)
        radiation_energy_history.append(e_rad)
        total_energy_history.append(e_tot)
        rel_drift = (e_tot - total_energy_initial) / max(total_energy_initial, 1e-300)

        if step_idx <= 5 or step_idx % 10 == 0 or abs(solver.t - final_time) < 1e-14:
            print(
                f"step {step_idx:5d}  t={solver.t:.4e} ns  dt={dt_step:.3e} ns  "
                f"T_mat={t_mat[-1]:.6f} keV  T_rad={t_rad[-1]:.6f} keV  "
                f"newton={info['newton_iter']} conv={info['converged']}  "
                f"dE/E0={rel_drift:.3e}"
            )

        dt_current = min(dt_cap, dt_current * growth)

    np.savez(
        f"infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g.npz",
        times=np.array(times),
        T_mat=np.array(t_mat),
        T_rad=np.array(t_rad),
        energy_edges=energy_edges,
        group_energy_history=np.array(group_energy_history),
        material_energy_history=np.array(material_energy_history),
        radiation_energy_history=np.array(radiation_energy_history),
        total_energy_history=np.array(total_energy_history),
        initial_group_energy_density=Eg0,
        final_group_energy_density=group_energy_history[-1],
        final_material_temperature=solver.T[0],
    )
    print(f"Saved: infinite_medium_multigroup_sigmaexp_diffusion_{n_groups}g.npz")
    final_rel_drift = (total_energy_history[-1] - total_energy_initial) / max(total_energy_initial, 1e-300)
    print(f"Energy drift: dE/E0 = {final_rel_drift:.6e}")

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
        sigma_funcs=sigma_funcs,
        T_compare=Tc,
        sigma0=sigma0,
        n_groups=n_groups,
    )


def main():
    parser = argparse.ArgumentParser(description="Infinite medium multigroup diffusion test")
    parser.add_argument("--groups", type=int, default=50, help="Number of energy groups")
    parser.add_argument("--dt-initial", type=float, default=1.0e-4, help="Initial time step in ns")
    parser.add_argument("--dt-max", type=float, default=1.0e-2, help="Maximum time step in ns")
    parser.add_argument("--dt-growth", type=float, default=1.1, help="Multiplicative dt growth factor per step")
    parser.add_argument("--final-time", type=float, default=1.0, help="Final time in ns")
    parser.add_argument("--newton-max", type=int, default=30, help="Maximum Newton iterations per step")
    parser.add_argument("--newton-tol", type=float, default=1e-8, help="Newton convergence tolerance")
    parser.add_argument("--gmres-tol", type=float, default=1e-8, help="GMRES tolerance")
    parser.add_argument("--gmres-max", type=int, default=100, help="Maximum GMRES iterations")
    parser.add_argument("--max-rel-change", type=float, default=0.25, help="Max relative temperature change per Newton update")
    parser.add_argument("--no-precond", action="store_false", help="Disable LMFG preconditioner")
    args = parser.parse_args()

    run_problem(
        n_groups=args.groups,
        dt_initial=args.dt_initial,
        dt_max=args.dt_max,
        dt_growth=args.dt_growth,
        final_time=args.final_time,
        max_newton_iter=args.newton_max,
        newton_tol=args.newton_tol,
        gmres_tol=args.gmres_tol,
        gmres_maxiter=args.gmres_max,
        max_relative_change=args.max_rel_change,
        use_preconditioner=(not args.no_precond),
    )


if __name__ == "__main__":
    main()
