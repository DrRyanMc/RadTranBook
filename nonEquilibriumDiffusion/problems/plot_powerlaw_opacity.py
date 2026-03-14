"""
Plot power-law opacity  σ_a(T, E) = 10 * ρ * T^{-1/2} * E^{-3}  (cm^-1)
at rho = 0.1 g/cm³ for T = 0.05, 0.1, 0.25 keV as a function of photon energy.
"""
import os, sys

import numpy as np
import matplotlib.pyplot as plt
__c = 29.98 #cm/ns
__a = 0.01372
_here        = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(_here))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.plotfuncs import font, hide_spines, show

# ── opacity ────────────────────────────────────────────────────────────────────
def sigma_a(T, E, rho=1.0):
    T_use = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * T_use**(-0.5) * E**(-3.0), 1e14)

# ── parameters ─────────────────────────────────────────────────────────────────
rho         = 0.1                    # g/cm³
temps       = [0.05, 0.1, 0.25]      # keV
linestyles  = ["-", "--", ":"]
E_vals      = np.logspace(-4, 1, 500)  # keV

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

for T, ls in zip(temps, linestyles):
    ax.plot(E_vals, sigma_a(T, E_vals, rho), ls=ls, lw=1.6,
            label=f"$T = {T}$ keV")

#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("photon energy $E$ (keV)", fontsize=10)
ax.set_ylabel(r"$\sigma_a$ (cm$^{-1}$)", fontsize=10)
ax.legend(prop=font, facecolor="white", edgecolor="none", fontsize=9)
ax.grid(True, alpha=0.2, which="both")
show("powerlaw_opacity_vs_energy.pdf", close_after=True)
print("Saved powerlaw_opacity_vs_energy.pdf")
