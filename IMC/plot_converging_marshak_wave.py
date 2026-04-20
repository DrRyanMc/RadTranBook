"""Plot converging Marshak wave IMC results from a saved .npz file.

Usage:
    python plot_converging_marshak_wave.py                          # uses default filename
    python plot_converging_marshak_wave.py converging_marshak_wave_imc.npz
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import analytic solution from the IMC driver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ConvergingMarshakWave import T_analytic_keV, T_HEV_PER_KEV

mpl.rcParams.update({
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
})

# --- Load file ---
fname = sys.argv[1] if len(sys.argv) > 1 else "converging_marshak_wave_imc.npz"
data = np.load(fname, allow_pickle=True)

snap_times = data['snap_times']
snap_T_keV = data['snap_T_keV']
snap_r_mid = data['snap_r_mid']
R = float(data['R'])

# --- Plot ---
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(6, 4.5))

# Create fine radial grid for analytic solution
r_anal = np.linspace(1e-7, R, 500)

for idx, (t_snap, T_imc, r_mid) in enumerate(zip(snap_times, snap_T_keV, snap_r_mid)):
    color_imc = my_colors[idx % len(my_colors)]
    
    # Convert to HeV for plotting (if desired, or keep in keV)
    T_imc_HeV = T_imc * T_HEV_PER_KEV
    
    # Compute analytic solution at this time
    T_anal_HeV = np.array([T_analytic_keV(r, t_snap) * T_HEV_PER_KEV for r in r_anal])
    
    # Plot IMC result
    ax.plot(r_mid / 1e-4, T_imc_HeV, color=color_imc, linestyle='-',
            label=f"IMC ($t$ = {t_snap:.1f} ns)" if idx < 3 else None)
    
    # Plot analytic solution
    ax.plot(r_anal / 1e-4, T_anal_HeV, color=color_imc, linestyle=':', linewidth=1.5,
            label=f"analytic ($t$ = {t_snap:.1f} ns)" if idx < 3 else None)

ax.set_xlim([0, R / 1e-4])
ax.set_xlabel(r"radius ($\mu$m)")
ax.set_ylabel("temperature (HeV)")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
out_fname = fname.replace('.npz', '_plot.pdf')
plt.savefig(out_fname, dpi=600)
print(f"Saved: {out_fname}")
plt.show()

#now make the same plot for the material energy density
rho = 19.3  # g/cm^3

my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(6, 4.5))

for idx, (t_snap, T_imc, r_mid) in enumerate(zip(snap_times, snap_T_keV, snap_r_mid)):
    color_imc = my_colors[idx % len(my_colors)]
    
    # Convert to HeV for plotting (if desired, or keep in keV)
    T_imc_HeV = T_imc * T_HEV_PER_KEV
    energy_density = 3.4e-3*(T_imc_HeV)**1.6*rho**.86 #GJ/cm^3
    # Compute analytic solution at this time
    T_anal_HeV = np.array([T_analytic_keV(r, t_snap) * T_HEV_PER_KEV for r in r_anal])
    
    # Plot IMC result
    ax.plot(r_mid / 1e-4, energy_density, color=color_imc, linestyle='-',
            label=f"IMC ($t$ = {t_snap:.1f} ns)" if idx < 3 else None)
    
    # Plot analytic solution
    ax.plot(r_anal / 1e-4, 3.4e-3*T_anal_HeV**1.6*rho**.86, color=color_imc, linestyle=':', linewidth=1.5,
            label=f"analytic ($t$ = {t_snap:.1f} ns)" if idx < 3 else None)

ax.set_xlim([0, R / 1e-4])
ax.set_xlabel(r"radius ($\mu$m)")
ax.set_ylabel("energy density (GJ/cm³)")
#ax.legend(fontsize=8, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
out_fname = out_fname.replace('_plot', '_e_plot')
plt.savefig(out_fname, dpi=600)
print(f"Saved: {out_fname}")
plt.show()