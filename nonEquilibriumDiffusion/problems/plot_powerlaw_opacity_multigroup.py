"""
Plot power-law opacity with multigroup approximations overlaid.

One PDF per temperature (T = 0.05, 0.10, 0.25 keV), each showing:
  - continuous opacity σ_a(T, E) = 10 * ρ * T^{-1/2} * E^{-3}
  - horizontal bars for geometric-mean group opacities with G = 2, 10, 50, 100 groups
    (energy edges: np.logspace(-4, 1, G+1))

ρ = 0.1 g/cm³
"""
import os, sys
__c = 29.98 #cm/ns
__a = 0.01372
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_here))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.plotfuncs import font, hide_spines, show

# ── opacity ────────────────────────────────────────────────────────────────────
def sigma_a(T, E, rho=1.0):
    T_use = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * T_use**(-0.5) * E**(-3.0), 1e14)

def group_sigma(T, E_lo, E_hi, rho=1.0):
    """Geometric mean of opacity at group boundaries."""
    return np.sqrt(sigma_a(T, E_lo, rho) * sigma_a(T, E_hi, rho))

# ── parameters ─────────────────────────────────────────────────────────────────
rho          = 0.1
temps        = [0.05, 0.10, 0.25]   # keV
group_counts = [2, 10, 50, 100]
E_fine       = np.logspace(-4, 1, 800)

# Colors and linestyles for group counts (matches multigroup comparison style)
group_colors     = {2: "tab:blue", 10: "tab:orange", 50: "tab:green", 100: "tab:red"}
group_linestyles = {2: ":", 10: "-.", 50: "--", 100: "-"}
group_markers = {2: "o", 10: "s", 50: None, 100: None}
# ── one figure per temperature ─────────────────────────────────────────────────
for T in temps:
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Continuous opacity
    ax.plot(E_fine, sigma_a(T, E_fine, rho), color="black", lw=1.2,
           label="continuous $\\sigma_a$", zorder=5)
    ax.plot(E_fine, 1e7*E_fine**3/(np.exp(E_fine/T)-1),color="black",alpha=0.5)
    # Multigroup approximations
    for G in group_counts:
        edges = np.logspace(-4, 1, G + 1)
        col = group_colors[G]
        ls  = group_linestyles[G]
        for g in range(G):
            sg = group_sigma(T, edges[g], edges[g + 1], rho)
            ax.hlines(sg, edges[g], edges[g + 1],
                      colors=col, linewidths=1.8, linestyles=ls)
            ax.plot([edges[g], edges[g + 1]], [sg, sg],
                    marker=group_markers[G], color=col, markersize=6, linewidth=0)

    # Legend: continuous + one proxy per group count
    proxy_handles = [
        Line2D([0], [0], color="black", lw=1.6, label="continuous $\\sigma_a$"),
    ] + [
        Line2D([0], [0], color=group_colors[G], lw=1.8,
               linestyle=group_linestyles[G],
               marker=group_markers[G], markersize=6,
               label=f"{G} groups")
        for G in group_counts
    ]
    ax.legend(handles=proxy_handles, prop=font, facecolor="white",
              edgecolor="none", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2*1e12)
    ax.set_xlabel(r"photon energy $\nu$ (keV)", fontsize=10)
    ax.set_ylabel(r"$\sigma_a$ (cm$^{-1}$)", fontsize=10)
    ax.grid(True, alpha=0.2, which="both")

    T_str   = f"{T:.2f}".replace(".", "p")
    outfile = f"powerlaw_opacity_multigroup_T{T_str}.pdf"
    show(outfile, close_after=True)
    print(f"Saved {outfile}")
