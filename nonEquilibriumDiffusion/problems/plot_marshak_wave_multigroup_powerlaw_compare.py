#!/usr/bin/env python3
"""Compare multigroup power-law Marshak-wave diffusion vs IMC outputs.

Expected NPZ schema (both files):
- times: (n_times,)
- r: (n_cells,)
- energy_edges: (n_groups + 1,)
- T_mat: (n_times, n_cells)
- T_rad: (n_times, n_cells)
- E_r: (n_times, n_cells)
- E_r_groups: (n_times, n_groups, n_cells)

Usage examples:
python nonEquilibriumDiffusion/problems/plot_marshak_wave_multigroup_powerlaw_compare.py
python nonEquilibriumDiffusion/problems/plot_marshak_wave_multigroup_powerlaw_compare.py \
  --diff-file marshak_wave_multigroup_powerlaw_10g_no_precond_timeBC_solutions.npz \
  --imc-file  marshak_wave_multigroup_powerlaw_imc_10g_timeBC.npz
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Path setup: script lives in nonEquilibriumDiffusion/problems/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_noneq_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_noneq_dir)
for _p in (_noneq_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.plotfuncs import font, show


_TIME_COLORS = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
]


def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    required = ["times", "r", "energy_edges", "T_mat", "T_rad", "E_r", "E_r_groups"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return {
        "path": path,
        "times": np.asarray(d["times"], dtype=float),
        "r": np.asarray(d["r"], dtype=float),
        "energy_edges": np.asarray(d["energy_edges"], dtype=float),
        "T_mat": np.asarray(d["T_mat"], dtype=float),
        "T_rad": np.asarray(d["T_rad"], dtype=float),
        "E_r": np.asarray(d["E_r"], dtype=float),
        "E_r_groups": np.asarray(d["E_r_groups"], dtype=float),
    }


def _pick_latest(patterns, search_dir):
    matches = []
    for p in patterns:
        matches.extend(glob.glob(os.path.join(search_dir, p)))
    if not matches:
        return None
    matches = sorted(set(matches), key=os.path.getmtime)
    return matches[-1]


def _match_times(times_ref, times_cmp, tol):
    pairs = []
    for i, t in enumerate(times_ref):
        j = int(np.argmin(np.abs(times_cmp - t)))
        dt = abs(times_cmp[j] - t)
        if dt <= tol:
            pairs.append((i, j, t, times_cmp[j], dt))
    return pairs


def _interp_to_grid(y_src, x_src, x_dst):
    return np.interp(x_dst, x_src, y_src)


def _build_aligned(diff, imc, time_tol):
    pairs = _match_times(diff["times"], imc["times"], tol=time_tol)
    if not pairs:
        raise RuntimeError(
            f"No matching times found within tolerance={time_tol}. "
            f"Diff times={diff['times']}, IMC times={imc['times']}"
        )

    aligned = []
    for i_d, i_i, t_d, t_i, dt in pairs:
        r_d = diff["r"]
        r_i = imc["r"]
        # Put everything on diffusion grid for direct overlays/errors.
        Tm_d = diff["T_mat"][i_d]
        Tr_d = diff["T_rad"][i_d]
        Er_d = diff["E_r"][i_d]

        Tm_i = _interp_to_grid(imc["T_mat"][i_i], r_i, r_d)
        Tr_i = _interp_to_grid(imc["T_rad"][i_i], r_i, r_d)
        Er_i = _interp_to_grid(imc["E_r"][i_i], r_i, r_d)

        # Spectra: spatially averaged group energy density.
        spec_d = np.mean(diff["E_r_groups"][i_d], axis=1)
        spec_i = np.mean(imc["E_r_groups"][i_i], axis=1)

        aligned.append(
            {
                "t_diff": t_d,
                "t_imc": t_i,
                "dt": dt,
                "r": r_d,
                "T_mat_diff": Tm_d,
                "T_mat_imc": Tm_i,
                "T_rad_diff": Tr_d,
                "T_rad_imc": Tr_i,
                "E_r_diff": Er_d,
                "E_r_imc": Er_i,
                "spec_diff": spec_d,
                "spec_imc": spec_i,
            }
        )
    return aligned


def _plot_temperature_linear(aligned, outbase):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)

    for k, s in enumerate(aligned):
        c = _TIME_COLORS[k % len(_TIME_COLORS)]
        t_lbl = f"t={s['t_diff']:.1f} ns"
        ax1.plot(s["r"], s["T_mat_diff"], color=c, lw=2.0, label=f"Diff {t_lbl}")
        ax1.plot(s["r"], s["T_mat_imc"], color=c, lw=2.0, ls="--", label=f"IMC {t_lbl}")

        ax2.plot(s["r"], s["T_rad_diff"], color=c, lw=2.0, label=f"Diff {t_lbl}")
        ax2.plot(s["r"], s["T_rad_imc"], color=c, lw=2.0, ls="--", label=f"IMC {t_lbl}")

    ax1.set_xlabel("position (cm)")
    ax1.set_ylabel("material temperature (keV)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8, ncol=2)

    ax2.set_xlabel("position (cm)")
    ax2.set_ylabel("radiation temperature (keV)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8, ncol=2)

    plt.tight_layout()
    show(f"{outbase}_T_compare.pdf", close_after=True)
    print(f"Saved: {outbase}_T_compare.pdf")


def _plot_temperature_log(aligned, outbase):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)

    for k, s in enumerate(aligned):
        c = _TIME_COLORS[k % len(_TIME_COLORS)]
        t_lbl = f"t={s['t_diff']:.1f} ns"
        ax1.plot(s["r"], s["T_mat_diff"], color=c, lw=2.0, label=f"Diff {t_lbl}")
        ax1.plot(s["r"], s["T_mat_imc"], color=c, lw=2.0, ls="--", label=f"IMC {t_lbl}")

        ax2.plot(s["r"], s["T_rad_diff"], color=c, lw=2.0, label=f"Diff {t_lbl}")
        ax2.plot(s["r"], s["T_rad_imc"], color=c, lw=2.0, ls="--", label=f"IMC {t_lbl}")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8, ncol=2)

    ax1.set_xlabel("position (cm)")
    ax1.set_ylabel("material temperature (keV)")
    ax2.set_xlabel("position (cm)")
    ax2.set_ylabel("radiation temperature (keV)")

    plt.tight_layout()
    show(f"{outbase}_T_log_compare.pdf", close_after=True)
    print(f"Saved: {outbase}_T_log_compare.pdf")


def _plot_energy_compare(aligned, outbase):
    fig, ax = plt.subplots(figsize=(7.5, 5.25))

    for k, s in enumerate(aligned):
        c = _TIME_COLORS[k % len(_TIME_COLORS)]
        t_lbl = f"t={s['t_diff']:.1f} ns"
        ax.semilogy(s["r"], s["E_r_diff"], color=c, lw=2.0, label=f"Diff {t_lbl}")
        ax.semilogy(s["r"], s["E_r_imc"], color=c, lw=2.0, ls="--", label=f"IMC {t_lbl}")

    ax.set_xlabel("position (cm)")
    ax.set_ylabel(r"radiation energy (GJ/cm$^3$/keV)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8, ncol=2)

    plt.tight_layout()
    show(f"{outbase}_E_rad_compare.pdf", close_after=True)
    print(f"Saved: {outbase}_E_rad_compare.pdf")


def _plot_spectrum_compare(aligned, energy_edges, outbase):
    E_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    fig, ax = plt.subplots(figsize=(7.5, 5.25))

    for k, s in enumerate(aligned):
        c = _TIME_COLORS[k % len(_TIME_COLORS)]
        t_lbl = f"t={s['t_diff']:.1f} ns"
        ax.loglog(E_mid, s["spec_diff"], color=c, lw=2.0, marker="o", ms=4, label=f"Diff {t_lbl}")
        ax.loglog(E_mid, s["spec_imc"], color=c, lw=2.0, ls="--", marker="s", ms=4, label=f"IMC {t_lbl}")

    ax.set_xlabel("photon energy (keV)")
    ax.set_ylabel(r"mean $E_{r,g}$ (GJ/cm$^3$/keV)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=8, ncol=2)

    plt.tight_layout()
    show(f"{outbase}_spectrum_compare.pdf", close_after=True)
    print(f"Saved: {outbase}_spectrum_compare.pdf")


def _print_error_summary(aligned):
    print("\nRelative L2 errors on aligned times (IMC vs diffusion):")
    for s in aligned:
        eps = 1e-300
        e_tmat = np.linalg.norm(s["T_mat_imc"] - s["T_mat_diff"]) / (np.linalg.norm(s["T_mat_diff"]) + eps)
        e_trad = np.linalg.norm(s["T_rad_imc"] - s["T_rad_diff"]) / (np.linalg.norm(s["T_rad_diff"]) + eps)
        e_erad = np.linalg.norm(s["E_r_imc"] - s["E_r_diff"]) / (np.linalg.norm(s["E_r_diff"]) + eps)
        print(
            f"  t_diff={s['t_diff']:.3f} ns, t_imc={s['t_imc']:.3f} ns (|dt|={s['dt']:.3e}): "
            f"T_mat={e_tmat:.3e}, T_rad={e_trad:.3e}, E_r={e_erad:.3e}"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare multigroup power-law Marshak wave diffusion and IMC outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--diff-file", type=str, default=None, help="Diffusion NPZ file path")
    p.add_argument("--imc-file", type=str, default=None, help="IMC NPZ file path")
    p.add_argument("--search-dir", type=str, default=".", help="Directory for auto-detecting files")
    p.add_argument("--time-tol", type=float, default=0.06, help="Max allowed |t_diff - t_imc| for matching")
    p.add_argument(
        "--output-base",
        type=str,
        default="marshak_wave_multigroup_powerlaw_imc_vs_diff",
        help="Output file prefix",
    )
    return p.parse_args()


def main():
    args = parse_args()

    diff_file = args.diff_file
    imc_file = args.imc_file

    if diff_file is None:
        diff_file = _pick_latest(
            patterns=[
                "marshak_wave_multigroup_powerlaw_*g*_solutions.npz",
                "marshak_wave_multigroup_powerlaw_*g_*.npz",
            ],
            search_dir=args.search_dir,
        )

    if imc_file is None:
        imc_file = _pick_latest(
            patterns=["marshak_wave_multigroup_powerlaw_imc_*g*.npz"],
            search_dir=args.search_dir,
        )

    if diff_file is None or imc_file is None:
        raise FileNotFoundError(
            "Could not auto-detect both files. Please pass --diff-file and --imc-file explicitly."
        )

    print(f"Diffusion file: {diff_file}")
    print(f"IMC file:       {imc_file}")

    diff = _load_npz(diff_file)
    imc = _load_npz(imc_file)

    # Basic compatibility checks.
    if diff["E_r_groups"].shape[1] != imc["E_r_groups"].shape[1]:
        raise ValueError(
            f"Group-count mismatch: diff has {diff['E_r_groups'].shape[1]}, imc has {imc['E_r_groups'].shape[1]}"
        )

    if len(diff["energy_edges"]) != len(imc["energy_edges"]):
        raise ValueError("Energy grid mismatch: different number of group edges")

    aligned = _build_aligned(diff, imc, time_tol=args.time_tol)

    _plot_temperature_linear(aligned, args.output_base)
    _plot_temperature_log(aligned, args.output_base)
    _plot_energy_compare(aligned, args.output_base)
    _plot_spectrum_compare(aligned, diff["energy_edges"], args.output_base)

    _print_error_summary(aligned)


if __name__ == "__main__":
    main()
