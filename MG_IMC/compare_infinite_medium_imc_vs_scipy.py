#!/usr/bin/env python3
"""Compare MG-IMC infinite-medium output against a high-resolution scipy.integrate reference.

This script expects one or more IMC NPZ files produced by:
  MG_IMC/test_infinite_medium_multigroup_expband.py

For each IMC NPZ input, it:
1) Solves the deterministic 0D multigroup-continuum reference with solve_ivp.
2) Writes a deterministic NPZ file.
3) Produces a temperature-history comparison plot.
4) Produces a spectrum-at-times comparison plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

mpl.rcParams.update(
    {
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "font.variant": "small-caps",
        "axes.titlesize": 18,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",
        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        # Axes/spines
        "axes.edgecolor": "black",
        "axes.linewidth": 1.15,
        "axes.grid": False,
        # Ticks
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Lines
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        # Legend
        "legend.frameon": False,
    }
)

C_LIGHT = 2.99792458e1  # cm/ns
A_RAD = 0.0137202  # GJ/(cm^3 keV^4)
PI_CONST = 15.0 / np.pi**4

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _require_keys(npz: np.lib.npyio.NpzFile, keys: tuple[str, ...], file_path: Path) -> None:
    missing = [k for k in keys if k not in npz]
    if missing:
        raise KeyError(f"{file_path}: missing required keys {missing}")


def _load_imc_npz(file_path: Path) -> dict[str, np.ndarray]:
    with np.load(file_path, allow_pickle=False) as data:
        _require_keys(
            data,
            (
                "times",
                "T_mat",
                "T_rad",
                "energy_edges",
                "group_energy_history",
            ),
            file_path,
        )
        return {
            "times": np.asarray(data["times"], dtype=float),
            "T_mat": np.asarray(data["T_mat"], dtype=float),
            "T_rad": np.asarray(data["T_rad"], dtype=float),
            "energy_edges": np.asarray(data["energy_edges"], dtype=float),
            "group_energy_history": np.asarray(data["group_energy_history"], dtype=float),
        }


def _sigma_abs(energy: np.ndarray, T: float, sigma0: float, t_floor: float = 1e-12) -> np.ndarray:
    T_use = max(float(T), t_floor)
    e = np.maximum(np.asarray(energy, dtype=float), 1e-300)
    return sigma0 * (1.0 - np.exp(-e / T_use)) / (e**3 * np.sqrt(T_use))


def _emis_term(energy: np.ndarray, T: float, sigma0: float, t_floor: float = 1e-12) -> np.ndarray:
    T_use = max(float(T), t_floor)
    e = np.asarray(energy, dtype=float)
    return sigma0 * PI_CONST * A_RAD * C_LIGHT * np.exp(-e / T_use) / np.sqrt(T_use)


def _bcolor(energy: np.ndarray, Trad: float, Tc: float) -> np.ndarray:
    # Colored Planck intensity phi_nu at t=0, matching the IMC setup.
    e = np.maximum(np.asarray(energy, dtype=float), 1e-300)
    return A_RAD * C_LIGHT * (Trad**4 / Tc**4) * (e**3 / np.expm1(e / Tc)) * PI_CONST


def solve_deterministic_reference(
    final_time: float,
    energy_min: float,
    energy_max: float,
    n_energy: int,
    sigma0: float,
    cv: float,
    Tc0: float,
    Trad0: float,
    Tmat0: float,
    t_eval: np.ndarray,
    rtol: float,
    atol: float,
    max_step: float,
) -> dict[str, np.ndarray]:
    energies = np.logspace(np.log10(energy_min), np.log10(energy_max), n_energy)

    def inv_eos(e_internal: float) -> float:
        return e_internal / cv

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        phi = y[:-1]
        Tm = inv_eos(y[-1])

        sig = _sigma_abs(energies, Tm, sigma0=sigma0)
        emis = _emis_term(energies, Tm, sigma0=sigma0)

        dphi_dt = C_LIGHT * (emis - sig * phi)
        dEint_dt = _trapz(sig * phi - emis, energies)

        out = np.empty_like(y)
        out[:-1] = dphi_dt
        out[-1] = dEint_dt
        return out

    y0 = np.zeros(n_energy + 1, dtype=float)
    y0[:-1] = _bcolor(energies, Trad=Trad0, Tc=Tc0)
    y0[-1] = cv * Tmat0

    sol = solve_ivp(
        rhs,
        (0.0, final_time),
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Deterministic solve failed: {sol.message}")

    phi_hist = np.asarray(sol.y[:-1, :], dtype=float)
    T_mat = np.asarray(sol.y[-1, :] / cv, dtype=float)

    E_rad = _trapz(phi_hist / C_LIGHT, energies, axis=0)
    T_rad = np.power(np.maximum(E_rad / A_RAD, 0.0), 0.25)

    return {
        "times": np.asarray(sol.t, dtype=float),
        "energies": energies,
        "phi_history": phi_hist,
        "T_mat": T_mat,
        "T_rad": T_rad,
    }


def _group_average_from_continuous(energy_edges: np.ndarray, energies: np.ndarray, spectral_density: np.ndarray) -> np.ndarray:
    # Compute group-averaged E_nu over each [E_g, E_{g+1}] bin.
    out = np.zeros(len(energy_edges) - 1, dtype=float)
    for g in range(len(out)):
        e1, e2 = energy_edges[g], energy_edges[g + 1]
        mask = (energies >= e1) & (energies <= e2)

        if np.count_nonzero(mask) < 2:
            local_e = np.array([e1, e2], dtype=float)
            local_y = np.interp(local_e, energies, spectral_density)
        else:
            local_e = energies[mask]
            local_y = spectral_density[mask]
            if local_e[0] > e1:
                local_e = np.insert(local_e, 0, e1)
                local_y = np.insert(local_y, 0, np.interp(e1, energies, spectral_density))
            if local_e[-1] < e2:
                local_e = np.append(local_e, e2)
                local_y = np.append(local_y, np.interp(e2, energies, spectral_density))

        out[g] = _trapz(local_y, local_e) / (e2 - e1)
    return out


def _nearest_indices(times: np.ndarray, requested: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(times - t))) for t in requested], dtype=int)


def make_temperature_plot(
    imc_data: dict[str, np.ndarray],
    det_data: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.plot(imc_data["times"], imc_data["T_mat"], color="tab:blue", linewidth=1.7, label="$T$")
    ax.plot(imc_data["times"], imc_data["T_rad"], color="tab:orange", linewidth=1.7, label="$T_r$")
    ax.plot(det_data["times"], det_data["T_mat"], color="tab:blue", linestyle="--", linewidth=1.5)
    ax.plot(det_data["times"], det_data["T_rad"], color="tab:orange", linestyle="--", linewidth=1.5)

    ax.set_xlabel("t (ns)")
    ax.set_ylabel("temperature (keV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    fig.tight_layout()
    png = output_base.with_name(output_base.name + "_temperature_compare.png")
    pdf = output_base.with_name(output_base.name + "_temperature_compare.pdf")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def make_spectrum_plot(
    imc_data: dict[str, np.ndarray],
    det_data: dict[str, np.ndarray],
    compare_times: np.ndarray,
    output_base: Path,
) -> None:
    energy_edges = imc_data["energy_edges"]
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    dE = np.diff(energy_edges)

    imc_idx = _nearest_indices(imc_data["times"], compare_times)
    det_idx = _nearest_indices(det_data["times"], compare_times)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    cmap = plt.cm.plasma

    for k, (ii, idet) in enumerate(zip(imc_idx, det_idx)):
        #make colors follow the standard matplotlib color cycle.
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        # cmap(k / max(len(compare_times) - 1, 1))

        t_imc = imc_data["times"][ii]
        t_det = det_data["times"][idet]
        t_ask = compare_times[np.abs(compare_times - t_imc).argmin()]

        imc_spec = imc_data["group_energy_history"][ii] / dE

        det_spec_cont = det_data["phi_history"][:, idet] / C_LIGHT
        det_spec_group = _group_average_from_continuous(
            energy_edges=energy_edges,
            energies=det_data["energies"],
            spectral_density=det_spec_cont,
        )

        ax.step(
            e_mid,
            imc_spec,
            where="mid",
            color=color,
            linewidth=1.7,
            label=f"t={t_ask:.3g} ns", alpha=0.8
        )
        # ax.plot(
        #     e_mid,
        #     det_spec_group,
        #     color=color,
        #     linestyle="--",
        #     linewidth=1.5,
        #     label=f"Scipy t={t_det:.4g} ns",
        # )
        ax.plot(
            det_data["energies"],
            det_spec_cont,
            color=color,
            linestyle="--",
            linewidth=1.5,
            #label=f"Scipy t={t_det:.4g} ns",
        )
    #ax.set_xscale("log")
    #ax.set_ylim(bottom=1e-11)  # Avoid log(0) issues; adjust as needed for visibility
    #ax.set_yscale("log")
    ax.set_xlabel(r"photon energy, $E_\nu$ (keV)")
    ax.set_ylabel(r"spectral energy density (GJ/cm$^3$/keV)")
    ax.set_xlim(xmax=13)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    fig.tight_layout()
    png = output_base.with_name(output_base.name + "_spectrum_compare.png")
    pdf = output_base.with_name(output_base.name + "_spectrum_compare.pdf")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _default_compare_times(final_time: float) -> np.ndarray:
    base = np.array([0.0, 0.01, 0.1, 1.0], dtype=float)
    clipped = np.clip(base, 0.0, final_time)
    return np.unique(clipped)


def run_comparison_for_file(
    imc_file: Path,
    sigma0: float,
    cv: float,
    Tc0: float,
    Trad0: float,
    Tmat0: float,
    n_energy: int,
    history_points: int,
    compare_times: np.ndarray | None,
    rtol: float,
    atol: float,
    max_step: float,
    output_prefix: str | None,
) -> None:
    imc = _load_imc_npz(imc_file)

    final_time = float(imc["times"][-1])
    energy_min = float(imc["energy_edges"][0])
    energy_max = float(imc["energy_edges"][-1])

    t_dense = np.linspace(0.0, final_time, max(history_points, 2))
    t_eval = np.unique(np.concatenate([t_dense, imc["times"]]))

    det = solve_deterministic_reference(
        final_time=final_time,
        energy_min=energy_min,
        energy_max=energy_max,
        n_energy=2000,
        sigma0=sigma0,
        cv=cv,
        Tc0=Tc0,
        Trad0=Trad0,
        Tmat0=Tmat0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    stem = output_prefix if output_prefix else imc_file.stem
    out_base = imc_file.with_name(stem)

    det_npz = out_base.with_name(out_base.name + "_scipy_reference.npz")
    np.savez(
        det_npz,
        times=det["times"],
        energies=det["energies"],
        phi_history=det["phi_history"],
        T_mat=det["T_mat"],
        T_rad=det["T_rad"],
    )
    print(f"Saved: {det_npz}")

    make_temperature_plot(imc, det, out_base)

    if compare_times is None:
        use_times = _default_compare_times(final_time)
    else:
        use_times = np.unique(np.clip(compare_times, 0.0, final_time))

    make_spectrum_plot(imc, det, use_times, out_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare MG-IMC infinite-medium NPZ output against a high-resolution "
            "deterministic scipy.integrate reference."
        )
    )
    parser.add_argument(
        "--imc-files",
        nargs="+",
        required=True,
        help="One or more IMC NPZ files from test_infinite_medium_multigroup_expband.py",
    )
    parser.add_argument("--sigma0", type=float, default=10.0, help="Opacity coefficient sigma0")
    parser.add_argument("--cv", type=float, default=0.01, help="rho*Cv [GJ/(cm^3 keV)]")
    parser.add_argument("--Tc0", type=float, default=1.0, help="Initial color temperature Tc [keV]")
    parser.add_argument("--Trad0", type=float, default=0.5, help="Initial radiation temperature Trad [keV]")
    parser.add_argument("--Tmat0", type=float, default=0.4, help="Initial material temperature [keV]")
    parser.add_argument("--det-energy-points", type=int, default=3000, help="Deterministic energy resolution")
    parser.add_argument("--history-points", type=int, default=600, help="Dense time samples for deterministic history")
    parser.add_argument(
        "--compare-times",
        nargs="*",
        type=float,
        default=None,
        help="Times (ns) for spectrum comparison, e.g. --compare-times 0 0.01 0.1 1.0",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="solve_ivp relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="solve_ivp absolute tolerance")
    parser.add_argument("--max-step", type=float, default=1e-3, help="Maximum deterministic time step [ns]")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Optional output basename prefix. If omitted, each file uses its IMC stem. "
            "When comparing multiple IMC files, this should usually be omitted."
        ),
    )

    args = parser.parse_args()

    for file_name in args.imc_files:
        run_comparison_for_file(
            imc_file=Path(file_name),
            sigma0=args.sigma0,
            cv=args.cv,
            Tc0=args.Tc0,
            Trad0=args.Trad0,
            Tmat0=args.Tmat0,
            n_energy=args.det_energy_points,
            history_points=args.history_points,
            compare_times=None if args.compare_times is None else np.array(args.compare_times, dtype=float),
            rtol=args.rtol,
            atol=args.atol,
            max_step=args.max_step,
            output_prefix=args.output_prefix,
        )
