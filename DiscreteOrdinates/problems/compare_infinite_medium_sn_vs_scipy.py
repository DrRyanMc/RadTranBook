#!/usr/bin/env python3
"""Compare multigroup S_N infinite-medium output against a high-resolution scipy reference.

This script expects one or more S_N NPZ files produced by:
  DiscreteOrdinates/problems/test_infinite_medium_multigroup_expband_sn.py

For each S_N NPZ input, it:
1) Solves the deterministic 0-D multigroup-continuum reference with solve_ivp.
2) Writes a deterministic NPZ file (``*_scipy_reference.npz``).
3) Produces a temperature-history comparison plot.
4) Produces a spectrum-at-times comparison plot.

Run from the DiscreteOrdinates directory::

    python problems/compare_infinite_medium_sn_vs_scipy.py \\
        --sn-files infinite_medium_multigroup_expband_sn_50g.npz

Physical setup (defaults match the S_N test script)::

    sigma0  = 10        (see --sigma0)
    cv      = 0.01      rho*Cv  GJ/(cm³·keV)
    Tc0     = 1.0       colour temperature of initial spectrum  keV
    Trad0   = 0.5       initial radiation temperature  keV
    Tmat0   = 0.4       initial material temperature   keV

The continuous opacity kernel used by the reference ODE is::

    sigma(E, T) = sigma0 * (1 - exp(-E/T)) / (E³ sqrt(T))    [absorption]
    emis(E, T)  = sigma0 * (15/π⁴) * a·c * exp(-E/T) / sqrt(T)  [emission]

which satisfies detailed balance: emis = sigma * B_Planck(E, T).
The normalised band-group opacity in the S_N test is derived from the same
kernel, so this continuous ODE is the correct reference for both.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "font.variant": "small-caps",
        "axes.titlesize": 18,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.15,
        "axes.grid": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
    }
)

# ── Physical constants ─────────────────────────────────────────────────────────
C_LIGHT = 2.99792458e1   # cm/ns
A_RAD   = 0.0137202      # GJ/(cm³·keV⁴)
PI_CONST = 15.0 / np.pi**4

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _require_keys(npz: np.lib.npyio.NpzFile, keys: tuple[str, ...], file_path: Path) -> None:
    missing = [k for k in keys if k not in npz]
    if missing:
        raise KeyError(f"{file_path}: missing required keys {missing}")


def _load_sn_npz(file_path: Path) -> dict[str, np.ndarray]:
    with np.load(file_path, allow_pickle=False) as data:
        _require_keys(
            data,
            ("times", "T_mat", "T_rad", "energy_edges", "group_energy_history"),
            file_path,
        )
        return {
            "times":                np.asarray(data["times"],                dtype=float),
            "T_mat":                np.asarray(data["T_mat"],                dtype=float),
            "T_rad":                np.asarray(data["T_rad"],                dtype=float),
            "energy_edges":         np.asarray(data["energy_edges"],         dtype=float),
            "group_energy_history": np.asarray(data["group_energy_history"], dtype=float),
        }


# ── Continuous opacity / emission (same kernel as in compare_infinite_medium_imc_vs_scipy) ──

def _sigma_abs(energy: np.ndarray, T: float, sigma0: float, t_floor: float = 1e-12) -> np.ndarray:
    """sigma0 * (1 - exp(-E/T)) / (E³ sqrt(T))  – absorption with stimulated emission."""
    T_use = max(float(T), t_floor)
    e = np.maximum(np.asarray(energy, dtype=float), 1e-300)
    return sigma0 * (1.0 - np.exp(-e / T_use)) / (e**3 * np.sqrt(T_use))


def _emis_term(energy: np.ndarray, T: float, sigma0: float, t_floor: float = 1e-12) -> np.ndarray:
    """Spontaneous-emission spectral source term (satisfies detailed balance)."""
    T_use = max(float(T), t_floor)
    e = np.asarray(energy, dtype=float)
    return sigma0 * PI_CONST * A_RAD * C_LIGHT * np.exp(-e / T_use) / np.sqrt(T_use)


def _bcolor(energy: np.ndarray, Trad: float, Tc: float) -> np.ndarray:
    """Coloured Planck spectral flux phi_E at t=0; total energy density = a·c·Trad⁴."""
    e = np.maximum(np.asarray(energy, dtype=float), 1e-300)
    return A_RAD * C_LIGHT * (Trad**4 / Tc**4) * (e**3 / np.expm1(e / Tc)) * PI_CONST


# ── Deterministic 0-D reference ───────────────────────────────────────────────

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
    """Solve the 0-D multigroup-continuum ODE with BDF from scipy.integrate."""
    energies = np.logspace(np.log10(energy_min), np.log10(energy_max), n_energy)

    def inv_eos(e_internal: float) -> float:
        return e_internal / cv

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        phi = y[:-1]
        Tm = inv_eos(y[-1])

        sig  = _sigma_abs(energies, Tm, sigma0=sigma0)
        emis = _emis_term(energies, Tm, sigma0=sigma0)

        dphi_dt  = C_LIGHT * (emis - sig * phi)
        dEint_dt = _trapz(sig * phi - emis, energies)

        out = np.empty_like(y)
        out[:-1] = dphi_dt
        out[-1]  = dEint_dt
        return out

    y0 = np.zeros(n_energy + 1, dtype=float)
    y0[:-1] = _bcolor(energies, Trad=Trad0, Tc=Tc0)
    y0[-1]  = cv * Tmat0

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
    T_mat    = np.asarray(sol.y[-1, :] / cv, dtype=float)

    E_rad = _trapz(phi_hist / C_LIGHT, energies, axis=0)
    T_rad = np.power(np.maximum(E_rad / A_RAD, 0.0), 0.25)

    return {
        "times":       np.asarray(sol.t,   dtype=float),
        "energies":    energies,
        "phi_history": phi_hist,
        "T_mat":       T_mat,
        "T_rad":       T_rad,
    }


# ── Group-averaging helper ────────────────────────────────────────────────────

def _group_average_from_continuous(
    energy_edges: np.ndarray,
    energies: np.ndarray,
    spectral_density: np.ndarray,
) -> np.ndarray:
    """Integrate spectral_density over each group bin, divided by ΔE."""
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


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_temperature_plot(
    sn_data: dict[str, np.ndarray],
    det_data: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.plot(sn_data["times"], sn_data["T_mat"],
            color="tab:blue",   lw=1.7, label=r"$T_\mathrm{mat}$ (S$_N$)")
    ax.plot(sn_data["times"], sn_data["T_rad"],
            color="tab:orange", lw=1.7, label=r"$T_\mathrm{rad}$ (S$_N$)")
    ax.plot(det_data["times"], det_data["T_mat"],
            color="tab:blue",   ls="--", lw=1.5, label=r"$T_\mathrm{mat}$ (ref.)")
    ax.plot(det_data["times"], det_data["T_rad"],
            color="tab:orange", ls="--", lw=1.5, label=r"$T_\mathrm{rad}$ (ref.)")

    ax.set_xlabel("t (ns)")
    ax.set_ylabel("temperature (keV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        out = output_base.with_name(output_base.name + "_temperature_compare" + ext)
        fig.savefig(out, dpi=160 if ext == ".png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def make_spectrum_plot(
    sn_data: dict[str, np.ndarray],
    det_data: dict[str, np.ndarray],
    compare_times: np.ndarray,
    output_base: Path,
) -> None:
    energy_edges = sn_data["energy_edges"]
    e_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    dE    = np.diff(energy_edges)

    sn_idx  = _nearest_indices(sn_data["times"], compare_times)
    det_idx = _nearest_indices(det_data["times"], compare_times)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, (isn, idet) in enumerate(zip(sn_idx, det_idx)):
        color  = colors[k % len(colors)]
        t_ask  = compare_times[np.abs(compare_times - sn_data["times"][isn]).argmin()]

        sn_spec = sn_data["group_energy_history"][isn] / dE

        det_spec_cont  = det_data["phi_history"][:, idet] / C_LIGHT
        det_spec_group = _group_average_from_continuous(
            energy_edges=energy_edges,
            energies=det_data["energies"],
            spectral_density=det_spec_cont,
        )

        ax.step(e_mid, sn_spec,
                where="mid", color=color, lw=1.7, alpha=0.8,
                label=f"t={t_ask:.3g} ns")
        ax.plot(det_data["energies"], det_spec_cont,
                color=color, ls="--", lw=1.5)

    ax.set_xlabel(r"photon energy, $E_\nu$ (keV)")
    ax.set_ylabel(r"spectral energy density (GJ cm$^{-3}$ keV$^{-1}$)")
    ax.set_xlim(xmax=13)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        out = output_base.with_name(output_base.name + "_spectrum_compare" + ext)
        fig.savefig(out, dpi=160 if ext == ".png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ── Per-file driver ───────────────────────────────────────────────────────────

def _default_compare_times(final_time: float) -> np.ndarray:
    base = np.array([0.0, 0.01, 0.1, 1.0], dtype=float)
    return np.unique(np.clip(base, 0.0, final_time))


def run_comparison_for_file(
    sn_file: Path,
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
    print(f"\n{'='*65}")
    print(f"Processing: {sn_file.name}")
    print(f"{'='*65}")

    sn = _load_sn_npz(sn_file)

    final_time  = float(sn["times"][-1])
    energy_min  = float(sn["energy_edges"][0])
    energy_max  = float(sn["energy_edges"][-1])

    t_dense = np.linspace(0.0, final_time, max(history_points, 2))
    t_eval  = np.unique(np.concatenate([t_dense, sn["times"]]))

    print(f"Running scipy reference (n_energy={n_energy}, ntimes={len(t_eval)}) …")
    det = solve_deterministic_reference(
        final_time  = final_time,
        energy_min  = energy_min,
        energy_max  = energy_max,
        n_energy    = n_energy,
        sigma0      = sigma0,
        cv          = cv,
        Tc0         = Tc0,
        Trad0       = Trad0,
        Tmat0       = Tmat0,
        t_eval      = t_eval,
        rtol        = rtol,
        atol        = atol,
        max_step    = max_step,
    )
    print("  scipy solve complete.")

    stem     = output_prefix if output_prefix else sn_file.stem
    out_base = sn_file.with_name(stem)

    det_npz = out_base.with_name(out_base.name + "_scipy_reference.npz")
    np.savez(
        det_npz,
        times       = det["times"],
        energies    = det["energies"],
        phi_history = det["phi_history"],
        T_mat       = det["T_mat"],
        T_rad       = det["T_rad"],
    )
    print(f"Saved: {det_npz}")

    make_temperature_plot(sn, det, out_base)

    use_times = (_default_compare_times(final_time)
                 if compare_times is None
                 else np.unique(np.clip(compare_times, 0.0, final_time)))
    make_spectrum_plot(sn, det, use_times, out_base)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare multigroup S_N infinite-medium NPZ output against a "
            "high-resolution scipy.integrate reference."
        )
    )
    parser.add_argument(
        "--sn-files",
        nargs="+",
        required=True,
        metavar="NPZ",
        help="One or more S_N NPZ files from test_infinite_medium_multigroup_expband_sn.py",
    )
    parser.add_argument("--sigma0",   type=float, default=10.0,
                        help="Opacity coefficient sigma0 (default: 10.0)")
    parser.add_argument("--cv",       type=float, default=0.01,
                        help="rho*Cv [GJ/(cm³·keV)] (default: 0.01)")
    parser.add_argument("--Tc0",      type=float, default=1.0,
                        help="Colour temperature of initial spectrum [keV] (default: 1.0)")
    parser.add_argument("--Trad0",    type=float, default=0.5,
                        help="Initial radiation temperature [keV] (default: 0.5)")
    parser.add_argument("--Tmat0",    type=float, default=0.4,
                        help="Initial material temperature [keV] (default: 0.4)")
    parser.add_argument("--det-energy-points", type=int, default=2000,
                        help="Deterministic energy-grid resolution (default: 2000)")
    parser.add_argument("--history-points",    type=int, default=600,
                        help="Dense time samples for deterministic history (default: 600)")
    parser.add_argument(
        "--compare-times",
        nargs="*",
        type=float,
        default=None,
        metavar="T",
        help="Times (ns) for spectrum comparison, e.g. --compare-times 0 0.01 0.1 1.0",
    )
    parser.add_argument("--rtol",     type=float, default=1e-6,
                        help="solve_ivp relative tolerance (default: 1e-6)")
    parser.add_argument("--atol",     type=float, default=1e-8,
                        help="solve_ivp absolute tolerance (default: 1e-8)")
    parser.add_argument("--max-step", type=float, default=1e-3,
                        help="Maximum deterministic time step [ns] (default: 1e-3)")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Optional output basename prefix (default: same stem as input NPZ). "
            "When comparing multiple files, omit this flag."
        ),
    )

    args = parser.parse_args()

    for file_name in args.sn_files:
        run_comparison_for_file(
            sn_file        = Path(file_name),
            sigma0         = args.sigma0,
            cv             = args.cv,
            Tc0            = args.Tc0,
            Trad0          = args.Trad0,
            Tmat0          = args.Tmat0,
            n_energy       = args.det_energy_points,
            history_points = args.history_points,
            compare_times  = (None if args.compare_times is None
                              else np.array(args.compare_times, dtype=float)),
            rtol           = args.rtol,
            atol           = args.atol,
            max_step       = args.max_step,
            output_prefix  = args.output_prefix,
        )
