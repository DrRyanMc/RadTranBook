"""
Compare multigroup Marshak wave solutions (2g, 10g, 50g) from MG-IMC.
Produces two PDFs:
  - multigroup_T_mat_comparison_imc.pdf   (material temperature)
  - multigroup_T_rad_comparison_imc.pdf   (radiation temperature)
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_here))   # visualization -> MG_IMC -> RadTranBook
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import font, show  # noqa: E402

# ── configuration ──────────────────────────────────────────────────────────────
NPZ_DIR      = project_root
GROUP_COUNTS = [10, 50]
FILE_PATTERN = "marshak_wave_multigroup_powerlaw_imc_{G}g_timeBC.npz"

COLORS     = {1.0: "blue", 2.0: "red", 5.0: "green", 10.0: "orange"}
LINESTYLES = {2: ":", 10: "-.", 50: "--"}
TIMES_NS   = list(COLORS.keys())   # [1.0, 2.0, 5.0, 10.0]


# ── helpers ────────────────────────────────────────────────────────────────────
def _parse_float_list(s):
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _guess_group_label(path, fallback_idx):
    base = os.path.basename(path)
    import re
    m = re.search(r"(\d+)g", base)
    if m:
        return f"{int(m.group(1))} groups"
    return f"run {fallback_idx + 1}"


def load_solutions(path):
    data = np.load(path, allow_pickle=True)

    times = np.asarray(data["times"], dtype=float)
    r     = np.asarray(data["r"],     dtype=float)
    T_mat = np.asarray(data["T_mat"], dtype=float)
    T_rad = np.asarray(data["T_rad"], dtype=float)

    n_saved = min(len(times), T_mat.shape[0], T_rad.shape[0])
    return [
        {"time": float(times[i]), "r": r, "T": T_mat[i, :], "T_rad": T_rad[i, :]}
        for i in range(n_saved)
    ]


def find_solution(solutions, target_time):
    for sol in solutions:
        if abs(sol["time"] - target_time) < 0.05:
            return sol
    return None


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare Marshak-wave MG-IMC NPZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "npz_files",
        nargs="*",
        default=None,
        help="NPZ files to compare. If omitted, uses the historical default group files.",
    )
    p.add_argument(
        "--times",
        type=str,
        default="1.0,2.0,5.0,10.0",
        help="Comma-separated times (ns) to plot.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="multigroup",
        help="Output filename prefix.",
    )
    return p.parse_args()


# ── plotting ───────────────────────────────────────────────────────────────────
def make_plot(key: str, ylabel: str, outfile: str, runs, times_ns) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for count_group, run in enumerate(runs):
        solutions = run["solutions"]
        ls = run["linestyle"]
        for t_ns in times_ns:
            sol = find_solution(solutions, t_ns)
            if sol is None:
                continue
            color = COLORS[t_ns]
            ax.plot(sol["r"], sol[key], ls=ls, color=color, lw=1.4)
            if count_group == 0 and t_ns < times_ns[-1]:
                ax.text(sol["r"][0] * 0.95, np.max(sol[key]) * 0.95,
                        f"{t_ns:.0f} ns", color=color, fontsize=8,
                        ha='right', va='top')
            elif count_group == 0 and t_ns >= times_ns[-1]:
                ax.text(sol["r"][0] * 0.95, np.max(sol[key]) * 1.15,
                        f"{t_ns:.0f} ns", color=color, fontsize=8,
                        ha='right', va='top')

    from matplotlib.lines import Line2D
    group_handles = [
        Line2D([0], [0], ls=run["linestyle"], color="black", lw=1.4,
               label=run["label"])
        for run in runs
    ]
    ax.legend(handles=group_handles, prop=font, facecolor="white",
              edgecolor="none", fontsize=8, ncol=1, framealpha=1.0)
    ax.set_xlabel("position (cm)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.1, 1.0])
    ax.set_xlim(0.01, 7)
    ax.grid(True, alpha=0.2, which='both')
    show(outfile, close_after=True)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    args = _parse_args()

    times_ns = _parse_float_list(args.times)
    if not times_ns:
        raise ValueError("No valid --times were provided.")
    unknown = [t for t in times_ns if t not in COLORS]
    if unknown:
        raise ValueError(
            f"Times {unknown} are not in the color map keys {sorted(COLORS.keys())}. "
            "Update COLORS in the script or choose supported times."
        )

    if args.npz_files:
        npz_files = args.npz_files
    else:
        npz_files = [
            os.path.join(NPZ_DIR, FILE_PATTERN.format(G=G))
            for G in GROUP_COUNTS
        ]

    for path in npz_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing NPZ file: {path}")

    fallback_styles = [":", "-.", "--", "-", (0, (5, 2)), (0, (3, 1, 1, 1))]
    runs = []
    for i, path in enumerate(npz_files):
        runs.append({
            "path": path,
            "label": _guess_group_label(path, i),
            "linestyle": fallback_styles[i % len(fallback_styles)],
            "solutions": load_solutions(path),
        })

    make_plot(
        "T",
        "material temperature (keV)",
        f"{args.out_prefix}_T_mat_comparison_imc.pdf",
        runs,
        times_ns,
    )
    make_plot(
        "T_rad",
        "radiation temperature (keV)",
        f"{args.out_prefix}_T_rad_comparison_imc.pdf",
        runs,
        times_ns,
    )
