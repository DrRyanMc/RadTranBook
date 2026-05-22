#!/usr/bin/env python3
"""
Dilute Spectrum Shell — multi-method comparison figures.

Loads snapshots from two or more results directories (each produced by
run_dilute_spectrum_shell.py or run_dilute_spectrum_shell_comparison.py) and
overlays them on the same axes.

Supported figures
-----------------
  fig2  — Temperature profiles  T_r(r), T_mat(r), [T_c(r)]
  fig4  — Radiation energy density  E_r(r)  (log–log)

Usage
-----
  # Compare MG IMC reference against gray IMC variants
  python plot_comparison.py \\
      --dirs  results/dilute_spectrum_shell/imc_32g_publication \\
              results/dilute_spectrum_shell/imc_gray_planck_publication \\
              results/dilute_spectrum_shell/imc_gray_rosseland_src_publication \\
      --labels "32g MG IMC" "Gray IMC (Planck)" "Gray IMC (Ross, T_src)" \\
      --snap_time 1.0 \\
      --figs fig2 fig4

  # Compare diffusion + flux-limiter variants
  python plot_comparison.py \\
      --dirs  results/dilute_spectrum_shell/mg_fl_32g_publication \\
              results/dilute_spectrum_shell/gray_fl_1g_publication \\
              results/dilute_spectrum_shell/gray_fl_src_1g_publication \\
      --snap_time 1.0

  # Glob shortcut: compare every subdirectory that matches a pattern
  python plot_comparison.py \\
      --glob "results/dilute_spectrum_shell/imc_gray_*_publication" \\
      --snap_time 1.0

Output
------
  Figures are saved in --out_dir (default: results/dilute_spectrum_shell/comparison/).
  The filename encodes the snapshot time, e.g.
      fig2_temperature_profiles_t1.000ns.pdf / .png
      fig4_Er_loglog_t1.000ns.pdf / .png
"""

import argparse
import glob as _glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.plotfuncs import font, show

from MG_IMC.problems.dilute_spectrum_shell import (
    R_S, R_1, R_2, R_OUT, T_S, C_LIGHT, A_RAD,
    free_streaming_Tr, free_streaming_Er,
)

# Reuse the fitting / loading helpers from the single-run plot script.
from MG_IMC.visualization.plot_dilute_spectrum_shell import (
    load_snapshots, pick_snapshot,
    fit_color_temperature,
    _infer_method_label,
)


# ===========================================================================
# Auto-label helpers
# ===========================================================================

# Maps common directory-name fragments to concise LaTeX-friendly labels.
_LABEL_MAP = [
    ("imc_gray_rosseland_src",  "Gray IMC (Ross, $T_S$)"),
    ("imc_gray_rosseland",      "Gray IMC (Rosseland)"),
    ("imc_gray_planck_src",     "Gray IMC (Planck, $T_S$)"),
    ("imc_gray_planck",         "Gray IMC (Planck)"),
    ("imc_gray",                "Gray IMC"),
    ("gray_fl_src",             "Gray Diff+FL ($T_S$)"),
    ("gray_fl",                 "Gray Diff+FL"),
    ("gray_src",                "Gray Diffusion ($T_S$)"),
    ("gray",                    "Gray Diffusion"),
    ("mg_fl",                   "MG Diff+FL"),
    ("mg",                      "MG Diffusion"),
    ("imc",                     "MG IMC"),
]

def _auto_label(results_dir):
    tag = os.path.basename(os.path.normpath(results_dir)).lower()
    for fragment, label in _LABEL_MAP:
        if fragment in tag:
            # Try to append the number-of-groups if present
            parts = tag.split("_")
            ng = next((p.rstrip("g") for p in parts
                       if p.endswith("g") and p[:-1].isdigit()), None)
            if ng and "MG" in label:
                return label + f" ({ng}g)"
            return label
    return _infer_method_label(results_dir)


def _deduplicate_labels(labels):
    """Append a numeric suffix to any repeated labels."""
    seen: dict = {}
    out = list(labels)
    for i, lbl in enumerate(labels):
        if labels.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            out[i] = f"{lbl} [{seen[lbl]}]"
    return out


# ===========================================================================
# Colour / style cycle
# ===========================================================================

# Linestyle cycle for time encoding in fig5
_LS_CYCLE = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]

_COLORS = [
    "#1f77b4",  # tab:blue
    "#d62728",  # tab:red
    "#2ca02c",  # tab:green
    "#ff7f0e",  # tab:orange
    "#9467bd",  # tab:purple
    "#8c564b",  # tab:brown
    "#e377c2",  # tab:pink
    "#17becf",  # tab:cyan
    "#bcbd22",  # tab:olive
    "#7f7f7f",  # tab:gray
]

def _color(i):
    return _COLORS[i % len(_COLORS)]


# ===========================================================================
# Figure 2 — Temperature profiles, multiple methods
# ===========================================================================

def fig_temperature_profiles(datasets, out_dir, snap_time=1.0,
                              show_Tc=False, show_Tm=True):
    """
    Overlay T_r (and optionally T_mat / T_c) for multiple methods.

    Parameters
    ----------
    datasets : list of (label, snapshots) pairs
    out_dir  : str — output directory
    snap_time : float — target snapshot time (ns)
    show_Tc   : bool — also plot colour temperature T_c (MG runs only)
    show_Tm   : bool — also plot material temperature T_mat
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    actual_t = None
    for i, (label, snapshots) in enumerate(datasets):
        col = _color(i)
        try:
            snap = pick_snapshot(snapshots, snap_time)
        except RuntimeError as e:
            print(f"  [{label}] Skipped — {e}")
            continue

        actual_t = snap["time"]
        r  = snap["r_centers"]
        Tr = snap["T_rad"]
        Tm = snap["T_mat"]

        ax.semilogy(r, Tr, color=col, lw=1.8, label=f"$T_r$  {label}")

        if show_Tm:
            ax.semilogy(r, Tm, color=col, lw=1.4, ls="--",
                        label=f"$T_{{\\rm mat}}$  {label}")

        if show_Tc and "E_rad_by_group" in snap and "energy_edges" in snap:
            Eg    = snap["E_rad_by_group"]
            edges = snap["energy_edges"]
            if len(edges) > 2:          # colour temperature only meaningful MG
                Tc   = np.array([fit_color_temperature(Eg[:, j], edges)
                                 for j in range(len(r))])
                mask = np.isfinite(Tc)
                ax.semilogy(r[mask], Tc[mask], color=col, lw=1.1, ls="-.",
                            label=f"$T_c$  {label}")

    # Analytic free-streaming reference
    r_cav  = np.linspace(R_S, R_1, 200)
    Tr_fs  = free_streaming_Tr(r_cav)
    ax.semilogy(r_cav, Tr_fs, "k:", lw=1.4,
                label=r"$T_S\sqrt{R_S/2r}$ (free-stream)")

    ax.axvline(R_1, color="0.55", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.55", lw=1.0, ls="--")
    ax.text(0.5*(R_1+R_2), ax.get_ylim()[0]*1.5, "shell",
            ha="center", va="bottom", fontproperties=font,
            fontsize=8, color="0.45")

    t_label = f"{actual_t:.3f}" if actual_t is not None else f"{snap_time:.3f}"
    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel("Temperature  (keV)", fontproperties=font)
    ax.set_title(f"Temperature profiles,  t = {t_label} ns", fontproperties=font)
    ax.legend(prop=font, fontsize=8.5)

    stem = f"fig2_temperature_profiles_t{snap_time:.3f}ns"
    show(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/.png")


# ===========================================================================
# Figure 5 — Material temperature near the shell, multiple methods × times
# ===========================================================================

def fig_shell_heating(datasets, out_dir, target_times=None):
    """
    Overlay T_mat near the shell for multiple methods and times.

    Color encodes method; linestyle encodes time.  A two-part legend
    (methods / times) keeps the plot readable with up to ~4 methods × 4 times.

    Parameters
    ----------
    datasets     : list of (label, snapshots) pairs
    out_dir      : str
    target_times : list of float (ns) — which snapshots to draw
    """
    from matplotlib.lines import Line2D

    if target_times is None:
        target_times = [0.5, 1.0, 2.0, 4.0]

    ls_list = [_LS_CYCLE[j % len(_LS_CYCLE)] for j in range(len(target_times))]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i, (label, snapshots) in enumerate(datasets):
        col = _color(i)
        for j, t_tgt in enumerate(target_times):
            try:
                snap = pick_snapshot(snapshots, t_tgt, tol=0.15)
            except RuntimeError:
                continue
            r  = snap["r_centers"]
            Tm = snap["T_mat"]
            mask = (r > R_1 - 2.0) & (r < R_2 + 1.5)
            ax.plot(r[mask], Tm[mask], lw=1.6, color=col, ls=ls_list[j])

    ax.axvline(R_1, color="0.5", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.5", lw=1.0, ls="--")
    ylo = ax.get_ylim()[0]
    ax.text(R_1 + 0.05, ylo, "inner\nshell", fontsize=8,
            va="bottom", ha="left", fontproperties=font, color="0.4")
    ax.text(R_2 + 0.05, ylo, "outer\nshell", fontsize=8,
            va="bottom", ha="left", fontproperties=font, color="0.4")

    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel(r"$T_{\rm mat}$  (keV)", fontproperties=font)
    ax.set_title("Material temperature near the shell", fontproperties=font)

    # Two-part legend: method colors + time linestyles
    method_handles = [
        Line2D([0], [0], color=_color(i), lw=1.8, ls="-", label=lbl)
        for i, (lbl, _) in enumerate(datasets)
    ]
    time_handles = [
        Line2D([0], [0], color="k", lw=1.4, ls=ls_list[j], label=f"t = {t:.2g} ns")
        for j, t in enumerate(target_times)
    ]

    leg1 = ax.legend(handles=method_handles, loc="upper left",
                     prop=font, fontsize=8.5, title="Method",
                     title_fontproperties=font)
    ax.add_artist(leg1)
    ax.legend(handles=time_handles, loc="upper right",
              prop=font, fontsize=8.5, title="Time",
              title_fontproperties=font)

    stem = "fig5_shell_heating"
    show(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/.png")


# ===========================================================================
# Figure 4 — E_r log–log, multiple methods
# ===========================================================================

def fig_Er_loglog(datasets, out_dir, snap_time=1.0):
    """Overlay E_r(r) for multiple methods with free-streaming reference."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    actual_t = None
    first_cav_Er = None
    first_cav_r  = None

    for i, (label, snapshots) in enumerate(datasets):
        col = _color(i)
        try:
            snap = pick_snapshot(snapshots, snap_time)
        except RuntimeError as e:
            print(f"  [{label}] Skipped — {e}")
            continue

        actual_t = snap["time"]
        r  = snap["r_centers"]
        Er = snap["E_rad"]

        ax.loglog(r, Er, color=col, lw=1.8, label=label)

        if first_cav_Er is None:
            mask = r < R_1
            if np.any(mask):
                first_cav_r  = r[mask]
                first_cav_Er = Er[mask]

    # Analytic free-streaming reference (normalised to first dataset)
    r_cav = np.linspace(R_S, R_1, 200)
    Er_fs = free_streaming_Er(r_cav)
    ax.loglog(r_cav, Er_fs, "k--", lw=1.3,
              label=r"$(R_S/r)^2$ free-stream")

    ax.axvline(R_1, color="0.55", lw=1.0, ls="--")
    ax.axvline(R_2, color="0.55", lw=1.0, ls="--")
    ax.text(0.5*(R_1+R_2), ax.get_ylim()[0]*2, "shell",
            ha="center", va="bottom", fontproperties=font,
            fontsize=8, color="0.45")

    t_label = f"{actual_t:.3f}" if actual_t is not None else f"{snap_time:.3f}"
    ax.set_xlabel("r  (cm)", fontproperties=font)
    ax.set_ylabel(r"$E_r$  (GJ / cm³)", fontproperties=font)
    ax.set_title(f"Radiation energy density,  t = {t_label} ns", fontproperties=font)
    ax.legend(prop=font, fontsize=8.5)

    stem = f"fig4_Er_loglog_t{snap_time:.3f}ns"
    show(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/.png")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay multiple dilute-spectrum-shell results on the same plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dirs", nargs="+", metavar="DIR",
        help="List of results directories to compare.",
    )
    src.add_argument(
        "--glob", metavar="PATTERN",
        help="Shell-style glob pattern matching results directories, e.g. "
             "\"results/dilute_spectrum_shell/imc_gray_*_publication\".",
    )

    p.add_argument(
        "--labels", nargs="+", metavar="LABEL", default=None,
        help="Human-readable label for each directory (in the same order as --dirs). "
             "If omitted, labels are inferred automatically from directory names.",
    )
    p.add_argument(
        "--snap_time", type=float, default=1.0, metavar="T_NS",
        help="Snapshot time (ns) to compare (default: 1.0).",
    )
    p.add_argument(
        "--figs", nargs="+",
        choices=["fig2", "fig4", "fig5"], default=["fig2"],
        help="Which comparison figures to produce (default: fig2).",
    )
    p.add_argument(
        "--shell_times", nargs="+", type=float,
        default=[0.5, 1.0, 2.0, 4.0], metavar="T_NS",
        help="Snapshot times (ns) to overlay in fig5 (default: 0.5 1.0 2.0 4.0).",
    )
    p.add_argument(
        "--show_Tc", action="store_true",
        help="Also overlay the colour temperature T_c in fig2 (MG runs only).",
    )
    p.add_argument(
        "--hide_Tm", action="store_true",
        help="Omit the material temperature T_mat from fig2.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Directory for output figures (default: results/dilute_spectrum_shell/comparison/).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve directories
    if args.dirs:
        dirs = args.dirs
    else:
        dirs = sorted(_glob.glob(args.glob))
        if not dirs:
            sys.exit(f"No directories matched: {args.glob}")

    # Validate
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        sys.exit("Directories not found:\n" + "\n".join(f"  {d}" for d in missing))

    # Labels
    if args.labels:
        if len(args.labels) != len(dirs):
            sys.exit(f"--labels count ({len(args.labels)}) must match "
                     f"--dirs count ({len(dirs)}).")
        labels = args.labels
    else:
        labels = [_auto_label(d) for d in dirs]
    labels = _deduplicate_labels(labels)

    # Output directory
    out_dir = args.out_dir or os.path.join(
        "results", "dilute_spectrum_shell", "comparison"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load snapshots
    print(f"Loading {len(dirs)} result set(s) …")
    datasets = []
    for label, d in zip(labels, dirs):
        snaps = load_snapshots(d)
        if not snaps:
            print(f"  WARNING: no snapshots in {d}, skipping.")
            continue
        print(f"  {label:40s}  ({len(snaps)} snapshots, "
              + ", ".join(f"{s['time']:.3f}" for s in snaps) + " ns)")
        datasets.append((label, snaps))

    if not datasets:
        sys.exit("No valid datasets found.")

    print(f"\nOutput → {out_dir}\n")

    if "fig2" in args.figs:
        print("Figure 2: temperature profiles …")
        fig_temperature_profiles(
            datasets, out_dir,
            snap_time=args.snap_time,
            show_Tc=args.show_Tc,
            show_Tm=not args.hide_Tm,
        )

    if "fig4" in args.figs:
        print("Figure 4: E_r log–log …")
        fig_Er_loglog(datasets, out_dir, snap_time=args.snap_time)

    if "fig5" in args.figs:
        print("Figure 5: shell heating …")
        fig_shell_heating(datasets, out_dir, target_times=args.shell_times)

    print("\nDone.")


if __name__ == "__main__":
    main()
