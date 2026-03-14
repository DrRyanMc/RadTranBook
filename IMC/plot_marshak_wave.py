"""Plot Marshak wave IMC results from a saved .npz file.

Usage:
    python plot_marshak_wave.py                          # uses default filename
    python plot_marshak_wave.py marshak_wave_output_10000ps.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
fname = sys.argv[1] if len(sys.argv) > 1 else "marshak_wave_output_10000ps.npz"
data = np.load(fname, allow_pickle=True)

prob        = data['problem_parameters'].item()
ss          = data['self_similar_parameters'].item()
snap_times  = data['snap_times']
snap_T_mat  = data['snap_T_mat']
snap_T_rad  = data['snap_T_rad']

# --- Reconstruct mesh midpoints ---
L  = prob['L']
I  = prob['I']
dx = L / I
mesh_midpoints = np.linspace(dx / 2, L - dx / 2, I)

# --- Self-similar setup ---
xi_max  = ss['xi_max']
omega   = ss['omega']
K_const = ss['K_const']
T_bc    = ss['T_bc']
xi_vals = np.linspace(0, xi_max, 300)
self_similar = lambda xi: (xi < xi_max) * np.power(
    np.where(xi < xi_max, (1 - xi/xi_max)*(1 + omega*xi/xi_max), 1e-30), 1/6)

# --- Plot ---
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(6, 4.5))

for idx, (t_snap, T_mat, T_rad) in enumerate(zip(snap_times, snap_T_mat, snap_T_rad)):
    color = my_colors[idx % len(my_colors)]
    color = "black"
    r_ss = xi_vals * (K_const * t_snap)**0.5
    T_ss = T_bc * self_similar(xi_vals)
    ax.plot(mesh_midpoints, T_mat, color=my_colors[0], linestyle='-',
            label=f"material" if idx == 0 else None)
    ax.plot(mesh_midpoints, T_rad, color=my_colors[1], linestyle='--',
            label=f"radiation" if idx == 0 else None)
    ax.plot(r_ss, T_ss,            color="black", linestyle=':', linewidth=1.5,
            label=f"self-similar" if idx == 0 else None)

ax.set_xlim([0, L])
ax.set_xlabel("position (cm)")
ax.set_ylabel("temperature (keV)")
ax.legend(fontsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
out_fname = fname.replace('.npz', '.pdf')
plt.savefig(out_fname, dpi=600)
plt.show()
