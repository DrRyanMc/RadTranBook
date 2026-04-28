#!/usr/bin/env python3
"""Compare 10-group Marshak-wave power-law IMC and S_N (DO) results.

This script can optionally run both solvers and then produce side-by-side
comparison plots for:
- material temperature T_mat(x)
- radiation temperature T_rad(x)
- group spectra E_{r,g}(x*)/dE at selected times

The plotting style is intentionally similar to
MG_IMC/plot_marshak_wave_multigroup_powerlaw_imc_compare.py.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BPoly

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.plotfuncs import font, show

from MG_IMC import C_LIGHT
from MG_IMC import test_marshak_wave_multigroup_powerlaw as imc_case
from DiscreteOrdinates.problems import test_marshak_wave_multigroup_powerlaw as do_case

_RUN_COLORS = {
    "IMC": "tab:blue",
    "S8": "tab:red",
    "Diffusion": "tab:green",
}


def _parse_time_list(s):
    if s is None or s.strip() == "":
        return None
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals if vals else None


def _nearest_indices(ref_times, other_times, tol):
    pairs = []
    for i, t in enumerate(ref_times):
        j = int(np.argmin(np.abs(other_times - t)))
        if abs(other_times[j] - t) <= tol:
            pairs.append((i, j))
    return pairs


def _save_do_rich_npz(results, path):
    """Save DO outputs with group-resolved radiation for spectrum plots."""
    x = np.asarray(results["x"], dtype=float)
    nx = len(x)
    Lx = float(results["Lx"])
    edges = np.linspace(0.0, Lx, nx + 1)
    G = int(results["n_groups"])

    solutions = results["solutions"]
    times = np.array([s["time"] for s in solutions], dtype=float)

    T_mat = np.zeros((len(solutions), nx), dtype=float)
    T_rad = np.zeros((len(solutions), nx), dtype=float)
    E_r_groups = np.zeros((len(solutions), G, nx), dtype=float)

    for it, sol in enumerate(solutions):
        T_mat[it] = BPoly(sol["T"].T, edges)(x)
        T_rad[it] = BPoly(sol["T_rad"].T, edges)(x)
        for g in range(G):
            phi_g = BPoly(sol["phi_g"][g].T, edges)(x)
            E_r_groups[it, g] = phi_g / C_LIGHT

    E_r = np.sum(E_r_groups, axis=1)

    np.savez_compressed(
        path,
        times=times,
        r=x,
        energy_edges=np.asarray(results["energy_edges"], dtype=float),
        T_mat=T_mat,
        T_rad=T_rad,
        E_r=E_r,
        E_r_groups=E_r_groups,
    )


def _run_cases(args):
    target_times = [t for t in [1.0, 2.0, 5.0, 10.0] if t <= args.tfinal + 1e-12]
    if not target_times:
        target_times = [args.tfinal]

    print("Running IMC case...")
    imc_case.run_marshak_wave_multigroup_powerlaw_imc(
        n_groups=args.groups,
        time_dependent_bc=not args.no_time_bc,
        ntarget=args.imc_ntarget,
        nboundary=args.imc_nboundary,
        nmax=args.imc_nmax,
        use_scalar_intensity_Tr=True,
        nx=args.zones,
        dt=args.dt_max,
        final_time=args.tfinal,
        grid_beta=0.0,
    )

    print("Running S_N (S8) case...")
    do_results = do_case.setup_and_run(
        I=args.zones,
        order=args.order,
        N=args.sn_order,
        n_groups=args.groups,
        Lx=args.Lx,
        tfinal=args.tfinal,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        K=args.K,
        maxits=args.maxits,
        LOUD=args.loud,
        time_dependent_bc=not args.no_time_bc,
        output_times=np.array(target_times),
        fix=args.fix,
        fleck_mode=args.fleck_mode,
    )
    _save_do_rich_npz(do_results, args.do_npz)
    print(f"Saved: {args.do_npz}")


def _load_dataset(path, label):
    d = np.load(path, allow_pickle=True)
    keys = ["times", "r", "energy_edges", "T_mat", "T_rad", "E_r", "E_r_groups"]
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    return {
        "label": label,
        "path": path,
        "times": np.asarray(d["times"], dtype=float),
        "r": np.asarray(d["r"], dtype=float),
        "energy_edges": np.asarray(d["energy_edges"], dtype=float),
        "T_mat": np.asarray(d["T_mat"], dtype=float),
        "T_rad": np.asarray(d["T_rad"], dtype=float),
        "E_r": np.asarray(d["E_r"], dtype=float),
        "E_r_groups": np.asarray(d["E_r_groups"], dtype=float),
    }


def _align_rows(ds_imc, ds_do, tol, ds_diff=None):
    pairs = _nearest_indices(ds_imc["times"], ds_do["times"], tol)
    if not pairs:
        raise RuntimeError(
            f"No common times found within tolerance={tol}. "
            "Try increasing --time-tol."
        )

    diff_pairs = None
    if ds_diff is not None:
        diff_pairs = _nearest_indices(ds_imc["times"], ds_diff["times"], tol)
        if not diff_pairs:
            raise RuntimeError(
                f"No IMC/diffusion common times found within tolerance={tol}. "
                "Try increasing --time-tol."
            )
        diff_map = {i_imc: i_diff for i_imc, i_diff in diff_pairs}

    rows = []
    for i_imc, i_do in pairs:
        row = {"time": ds_imc["times"][i_imc], "imc": i_imc, "do": i_do}
        if ds_diff is not None:
            if i_imc not in diff_map:
                continue
            row["diff"] = diff_map[i_imc]
        rows.append(row)

    if not rows:
        raise RuntimeError("No aligned rows remained after IMC/DO/diffusion time matching.")

    return rows


def _plot_temperature_profiles(ds_imc, ds_do, rows, quantity_key, ylabel, outname, ds_diff=None):
    n_panels = len(rows)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x_ref = ds_imc["r"]
    for ax, row in zip(axes, rows):
        t = row["time"]

        y_imc = ds_imc[quantity_key][row["imc"]]
        ax.plot(
            ds_imc["r"],
            y_imc,
            color=_RUN_COLORS["IMC"],
            linewidth=2.2,
            label="IMC",
        )

        y_do = ds_do[quantity_key][row["do"]]
        x_do = ds_do["r"]
        if len(x_do) != len(x_ref) or np.max(np.abs(x_do - x_ref)) > 1e-12:
            y_do = np.interp(x_ref, x_do, y_do)
            x_do = x_ref

        ax.plot(
            x_do,
            y_do,
            color=_RUN_COLORS["S8"],
            linewidth=2.2,
            linestyle="--",
            label="S8",
        )

        if ds_diff is not None and "diff" in row:
            y_diff = ds_diff[quantity_key][row["diff"]]
            x_diff = ds_diff["r"]
            if len(x_diff) != len(x_ref) or np.max(np.abs(x_diff - x_ref)) > 1e-12:
                y_diff = np.interp(x_ref, x_diff, y_diff)
                x_diff = x_ref

            ax.plot(
                x_diff,
                y_diff,
                color=_RUN_COLORS["Diffusion"],
                linewidth=2.2,
                linestyle="-.",
                label="Diffusion",
            )

        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlabel("position (cm)")
        ax.set_title(f"t = {t:.2f} ns", fontsize=11)

    axes[0].set_ylabel(ylabel)
    leg = axes[-1].legend(prop=font, facecolor="white", edgecolor="none", fontsize=10)
    leg.get_frame().set_alpha(None)

    plt.tight_layout()
    show(outname, close_after=True)
    print(f"Saved: {outname}")


def _plot_spectra(ds_imc, ds_do, rows, outbase, x_target, selected_times, time_tol, ds_diff=None):
    if selected_times is None:
        use_rows = rows
    else:
        use_rows = []
        for t_req in selected_times:
            j = int(np.argmin([abs(r["time"] - t_req) for r in rows]))
            if abs(rows[j]["time"] - t_req) <= time_tol:
                use_rows.append(rows[j])

    if not use_rows:
        print("No spectrum times selected within tolerance; skipping spectra.")
        return

    E_edges_imc = ds_imc["energy_edges"]
    E_mid_imc = np.sqrt(E_edges_imc[:-1] * E_edges_imc[1:])
    dE_imc = E_edges_imc[1:] - E_edges_imc[:-1]

    E_edges_do = ds_do["energy_edges"]
    E_mid_do = np.sqrt(E_edges_do[:-1] * E_edges_do[1:])
    dE_do = E_edges_do[1:] - E_edges_do[:-1]

    for row in use_rows:
        t = row["time"]

        i_x_imc = int(np.argmin(np.abs(ds_imc["r"] - x_target)))
        x_imc = ds_imc["r"][i_x_imc]
        spec_imc = ds_imc["E_r_groups"][row["imc"], :, i_x_imc] / dE_imc

        i_x_do = int(np.argmin(np.abs(ds_do["r"] - x_target)))
        x_do = ds_do["r"][i_x_do]
        spec_do = ds_do["E_r_groups"][row["do"], :, i_x_do] / dE_do

        spec_diff = None
        x_diff = None
        E_mid_diff = None
        if ds_diff is not None and "diff" in row and "E_r_groups" in ds_diff:
            E_edges_diff = ds_diff["energy_edges"]
            E_mid_diff = np.sqrt(E_edges_diff[:-1] * E_edges_diff[1:])
            dE_diff = E_edges_diff[1:] - E_edges_diff[:-1]
            i_x_diff = int(np.argmin(np.abs(ds_diff["r"] - x_target)))
            x_diff = ds_diff["r"][i_x_diff]
            spec_diff = ds_diff["E_r_groups"][row["diff"], :, i_x_diff] / dE_diff

        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        ax.loglog(
            E_mid_imc,
            spec_imc,
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=_RUN_COLORS["IMC"],
            label=f"IMC (x={x_imc:.3f} cm)",
        )
        ax.loglog(
            E_mid_do,
            spec_do,
            marker="s",
            markersize=4,
            linewidth=2.2,
            linestyle="--",
            color=_RUN_COLORS["S8"],
            label=f"S8 (x={x_do:.3f} cm)",
        )
        if spec_diff is not None:
            ax.loglog(
                E_mid_diff,
                spec_diff,
                marker="^",
                markersize=4,
                linewidth=2.2,
                linestyle="-.",
                color=_RUN_COLORS["Diffusion"],
                label=f"Diffusion (x={x_diff:.3f} cm)",
            )

        ax.set_xlabel("photon energy (keV)")
        ax.set_ylabel(r"$E_{r,g}$ (GJ cm$^{-3}$ keV$^{-1}$)")
        ax.set_title(f"Spectrum at t = {t:.2f} ns, x = {x_target:.3f} cm")
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        leg = ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=10)
        leg.get_frame().set_alpha(None)

        plt.tight_layout()
        outname = f"{outbase}_spectrum_t{t:.2f}ns_x{x_target:.3f}cm.pdf"
        show(outname, close_after=True)
        print(f"Saved: {outname}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare IMC and S8 Marshak-wave power-law runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-cases", action="store_true", help="Run IMC and S8 before plotting")

    p.add_argument("--groups", type=int, default=10, help="Number of energy groups")
    p.add_argument("--sn-order", type=int, default=8, help="S_N angular order N")
    p.add_argument("--zones", type=int, default=140, help="Number of spatial cells")
    p.add_argument("--order", type=int, default=3, help="DO Bernstein order")
    p.add_argument("--Lx", type=float, default=7.0, help="Domain length (cm)")
    p.add_argument("--tfinal", type=float, default=10.0, help="Final time (ns)")
    p.add_argument("--dt-min", type=float, default=1e-4, help="DO minimum dt (ns)")
    p.add_argument("--dt-max", type=float, default=0.01, help="DO max dt / IMC dt (ns)")
    p.add_argument("--no-time-bc", action="store_true", help="Use constant boundary temperature")

    p.add_argument("--K", type=int, default=800, help="DO DMD iterations")
    p.add_argument("--maxits", type=int, default=2000, help="DO max iterations per step")
    p.add_argument("--loud", type=int, default=0, help="DO verbosity")
    p.add_argument("--fix", type=int, default=0, help="DO positivity fix")
    p.add_argument("--fleck-mode", choices=["legacy", "imc"], default="legacy", help="DO Fleck mode")

    p.add_argument("--imc-ntarget", type=int, default=50000, help="IMC material particles per step")
    p.add_argument("--imc-nboundary", type=int, default=50000, help="IMC boundary particles per step")
    p.add_argument("--imc-nmax", type=int, default=200000, help="IMC census comb target")

    p.add_argument(
        "--imc-npz",
        type=str,
        default="marshak_wave_multigroup_powerlaw_imc_10g_timeBC.npz",
        help="Path to IMC NPZ file",
    )
    p.add_argument(
        "--do-npz",
        type=str,
        default="marshak_wave_powerlaw_sn_10g_timeBC_rich.npz",
        help="Path to S8 NPZ file with E_r_groups",
    )
    p.add_argument(
        "--diff-npz",
        type=str,
        default="",
        help="Optional diffusion NPZ file to include in comparisons",
    )

    p.add_argument("--time-tol", type=float, default=0.06, help="Snapshot time matching tolerance (ns)")
    p.add_argument("--output-base", type=str, default="marshak_wave_powerlaw_imc_vs_s8_10g", help="Output plot prefix")
    p.add_argument("--spectrum-x", type=float, default=0.5, help="Spectrum location (cm)")
    p.add_argument("--spectrum-times", type=str, default="1.0,5.0,10.0", help="Comma-separated spectrum times")

    return p.parse_args()


def main():
    args = parse_args()

    if args.run_cases:
        _run_cases(args)

    if not os.path.exists(args.imc_npz):
        raise FileNotFoundError(f"IMC file not found: {args.imc_npz}")
    if not os.path.exists(args.do_npz):
        raise FileNotFoundError(
            f"S8 file not found: {args.do_npz}. "
            "Run with --run-cases or provide --do-npz."
        )

    ds_imc = _load_dataset(args.imc_npz, "IMC")
    ds_do = _load_dataset(args.do_npz, "S8")
    ds_diff = None
    if args.diff_npz:
        if not os.path.exists(args.diff_npz):
            raise FileNotFoundError(f"Diffusion file not found: {args.diff_npz}")
        ds_diff = _load_dataset(args.diff_npz, "Diffusion")

    rows = _align_rows(ds_imc, ds_do, tol=args.time_tol, ds_diff=ds_diff)
    print("Common times:", [f"{r['time']:.3f}" for r in rows])

    _plot_temperature_profiles(
        ds_imc,
        ds_do,
        rows,
        quantity_key="T_mat",
        ylabel="material temperature (keV)",
        outname=f"{args.output_base}_T_mat.pdf",
        ds_diff=ds_diff,
    )

    _plot_temperature_profiles(
        ds_imc,
        ds_do,
        rows,
        quantity_key="T_rad",
        ylabel="radiation temperature (keV)",
        outname=f"{args.output_base}_T_rad.pdf",
        ds_diff=ds_diff,
    )

    selected_times = _parse_time_list(args.spectrum_times)
    _plot_spectra(
        ds_imc,
        ds_do,
        rows,
        outbase=args.output_base,
        x_target=args.spectrum_x,
        selected_times=selected_times,
        time_tol=args.time_tol,
        ds_diff=ds_diff,
    )


if __name__ == "__main__":
    main()
