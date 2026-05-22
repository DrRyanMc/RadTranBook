"""
Compare multigroup Marshak wave solutions (2g, 10g, 50g) from MG-IMC.
Produces two PDFs:
  - multigroup_T_mat_comparison_imc.pdf   (material temperature)
  - multigroup_T_rad_comparison_imc.pdf   (radiation temperature)
"""
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
def load_solutions(G):
    path = os.path.join(NPZ_DIR, FILE_PATTERN.format(G=G))
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


# ── plotting ───────────────────────────────────────────────────────────────────
def make_plot(key: str, ylabel: str, outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for count_group, G in enumerate(GROUP_COUNTS):
        solutions = load_solutions(G)
        ls = LINESTYLES[G]
        for t_ns in TIMES_NS:
            sol = find_solution(solutions, t_ns)
            if sol is None:
                continue
            color = COLORS[t_ns]
            ax.plot(sol["r"], sol[key], ls=ls, color=color, lw=1.4)
            if count_group == 0 and t_ns < TIMES_NS[-1]:
                ax.text(sol["r"][0] * 0.95, np.max(sol[key]) * 0.95,
                        f"{t_ns:.0f} ns", color=color, fontsize=8,
                        ha='right', va='top')
            elif count_group == 0 and t_ns >= TIMES_NS[-1]:
                ax.text(sol["r"][0] * 0.95, np.max(sol[key]) * 1.15,
                        f"{t_ns:.0f} ns", color=color, fontsize=8,
                        ha='right', va='top')

    from matplotlib.lines import Line2D
    group_handles = [
        Line2D([0], [0], ls=LINESTYLES[G], color="black", lw=1.4,
               label=f"{G} groups")
        for G in GROUP_COUNTS
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
    make_plot("T",     "material temperature (keV)", "multigroup_T_mat_comparison_imc.pdf")
    make_plot("T_rad", "radiation temperature (keV)", "multigroup_T_rad_comparison_imc.pdf")
