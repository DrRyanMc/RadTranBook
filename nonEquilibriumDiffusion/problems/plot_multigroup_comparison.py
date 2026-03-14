"""
Compare multigroup Marshak wave solutions (2g, 10g, 50g, 100g) using the
Larsen flux limiter.  Produces two PDFs:
  - multigroup_T_mat_comparison_larsen.pdf  (material temperature)
  - multigroup_T_rad_comparison_larsen.pdf  (radiation temperature)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_here))   # problems → nonEquilibriumDiffusion → RadTranBook
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import font, hide_spines, show  # noqa: E402

# ── configuration ──────────────────────────────────────────────────────────────
# NPZ files live in the RadTranBook root (same dir as project_root)
NPZ_DIR = project_root

GROUP_COUNTS = [2, 10, 50, 100]
FILE_PATTERN = (
    "marshak_wave_multigroup_powerlaw_{G}g_no_precond_timeBC_larsen_solutions.npz"
)

COLORS     = {1.0: "blue", 2.0: "red", 5.0: "green", 10.0: "orange"}
LINESTYLES = {2: ":", 10: "-.", 50: "--", 100: "-"}
TIMES_NS   = list(COLORS.keys())   # [1.0, 2.0, 5.0, 10.0]


# ── helpers ────────────────────────────────────────────────────────────────────
def load_solutions(G):
    path = os.path.join(NPZ_DIR, FILE_PATTERN.format(G=G))
    data = np.load(path, allow_pickle=True)
    return list(data["solutions"])   # list of dicts


def find_solution(solutions, target_time):
    for sol in solutions:
        if abs(sol["time"] - target_time) < 0.05:
            return sol
    return None


# ── plotting ───────────────────────────────────────────────────────────────────
def make_plot(key: str, ylabel: str, outfile: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    count_group = 0
    for G in GROUP_COUNTS:
        solutions = load_solutions(G)
        ls = LINESTYLES[G]
        for t_ns in TIMES_NS:
            sol = find_solution(solutions, t_ns)
            if sol is None:
                continue
            color  = COLORS[t_ns]
            label  = f"{G}g, t={t_ns:.0f} ns" if t_ns == TIMES_NS[0] else f"{G}g, t={int(t_ns)} ns"
            ax.plot(sol["r"], sol[key], ls=ls, color=color, lw=1.4, label=label)
            #text to label times
            if count_group == 0:
                ax.text(sol["r"][0]*0.95, np.max(sol[key])*.95, f"{t_ns:.0f} ns", color=color,fontsize=8, ha='right', va='top')
        count_group += 1

    # Legend: group-count legend entries (linestyle) separate from time entries (color)
    # Build proxy artists to keep the legend compact
    from matplotlib.lines import Line2D
    group_handles = [
        Line2D([0], [0], ls=LINESTYLES[G], color="black", lw=1.4, label=f"{G} groups")
        for G in GROUP_COUNTS
    ]
    # time_handles = [
    #     Line2D([0], [0], ls="-", color=COLORS[t], lw=1.4, label=f"t = {t:.0f} ns")
    #     for t in TIMES_NS
    # ]
    all_handles = group_handles #+ time_handles
    ax.legend(handles=all_handles, prop=font, facecolor="white", edgecolor="none",
              fontsize=8, ncol=1,framealpha=1.0)
    ax.set_xlabel("position (cm)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.1,1.0])
    ax.set_xlim(0.02,7)
    ax.grid(True, alpha=0.2,which='both')
    #hide_spines(ax, intx=True)
    show(outfile, close_after=True)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    make_plot("T",     "material temperature (keV)", "multigroup_T_mat_comparison_larsen.pdf")
    make_plot("T_rad", "radiation temperature (keV)", "multigroup_T_rad_comparison_larsen.pdf")
