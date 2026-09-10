"""Su-Olson test problem comparing M1 closures.

Runs the standard Su-Olson benchmark with five M1 closures and compares
normalised radiation energy against the transport reference solution.

Problem setup (mirrors the FLD su_olson_flux_limiter_comparison.py):
  - 1-D slab, x = 0..20 mfp
  - σ_a = 1 cm⁻¹, no scattering
  - Material EOS: e(T) = a T⁴
  - Isotropic radiation source  Q = a c σ  for  0 < x < 0.5 mfp
  - Source duration: 10 mean-free times  τ₀ = 1/(cσ)
  - Left BC: reflecting; right BC: Marshak vacuum
  - Output at τ = 0.1, 1.0, 3.16, 10, 31.6, 100

Note on source magnitude
------------------------
The transport-equation source per steradian is Q/(4π).  Integrating over all
angles to form the energy equation removes the solid-angle factor, so the
source in the E_r equation is the full Q = a c σ (not Q/2).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Allow running from any directory.
_here = os.path.dirname(os.path.abspath(__file__))
_m1_root = os.path.dirname(_here)              # M1/
_project_root = os.path.dirname(_m1_root)      # RadTranBook/
sys.path.insert(0, _m1_root)
sys.path.insert(0, _project_root)

from M1.src.m1_1d import (
    M1Solver1D,
    closure_p1,
    closure_kershaw,
    closure_levermore,
    closure_minerbo_poly,
    closure_minerbo_rational,
    C_LIGHT,
    A_RAD,
)
from utils.plotfuncs import show

# ---------------------------------------------------------------------------
# Transport reference data (same tables used in the FLD comparison script)
# ---------------------------------------------------------------------------

su_olson_x = np.array([
    0.01000, 0.10000, 0.17783, 0.31623, 0.45000, 0.50000,
    0.56234, 0.75000, 1.00000, 1.33352, 1.77828, 3.16228,
    5.62341, 10.00000, 17.78279,
])
su_olson_tau = np.array([0.10000, 0.31623, 1.00000, 3.16228, 10.00000, 31.6228, 100.000])

transport_rad_energy = np.array([
    [0.09531, 0.27526, 0.64308, 1.20052, 2.23575, 0.69020, 0.35720],
    [0.09531, 0.27526, 0.63585, 1.18869, 2.21944, 0.68974, 0.35714],
    [0.09532, 0.27527, 0.61958, 1.16190, 2.18344, 0.68878, 0.35702],
    [0.09529, 0.26262, 0.56187, 1.07175, 2.06448, 0.68569, 0.35664],
    [0.08823, 0.20312, 0.44711, 0.90951, 1.86072, 0.68111, 0.35599],
    [0.04765, 0.13762, 0.35801, 0.79902, 1.73178, 0.67908, 0.35574],
    [0.00375, 0.06277, 0.25374, 0.66678, 1.57496, 0.67619, 0.35538],
    [np.nan,  0.00280, 0.11430, 0.44675, 1.27398, 0.66548, 0.35393],
    [np.nan,  np.nan,  0.03648, 0.27540, 0.98782, 0.64691, 0.35141],
    [np.nan,  np.nan,  0.00291, 0.14531, 0.70822, 0.61538, 0.34697],
    [np.nan,  np.nan,  np.nan,  0.05968, 0.45016, 0.56353, 0.33924],
    [np.nan,  np.nan,  np.nan,  0.00123, 0.09673, 0.36965, 0.30346],
    [np.nan,  np.nan,  np.nan,  np.nan,  0.00375, 0.10830, 0.21382],
    [np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  0.00390, 0.07200],
    [np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  0.00272],
])

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------

sigma_a      = 1.0             # cm⁻¹
x_min        = 0.0             # cm
x_max        = 20.0            # cm
n_cells      = 400
source_width = 0.5             # cm  (source for  0 < x < source_width)
T_cold       = 1e-3            # keV  (cold initial temperature)

# Source magnitude in the E_r equation: full Q = acσ (not Q/2).
source_magnitude = A_RAD * C_LIGHT   # GJ/(cm³ ns)

mean_free_time  = 1.0 / (C_LIGHT * sigma_a)   # ns
source_duration = 10.0 * mean_free_time        # ns

early_tau = [0.1, 1.0, 3.16228, 10.0]
late_tau  = [31.6228, 100.0]
all_tau   = early_tau #+ late_tau

dt       = 0.02 * mean_free_time
n_steps  = int(np.ceil(max(all_tau) * mean_free_time / dt))

print(f"Mean-free time τ₀ = {mean_free_time:.4e} ns")
print(f"dt = {dt:.4e} ns  ({dt/mean_free_time:.3f} τ₀),  n_steps = {n_steps}")

# ---------------------------------------------------------------------------
# Material property functions (same EOS as the FLD comparison)
# ---------------------------------------------------------------------------

def sigma_func(T):
    return np.full_like(T, sigma_a)

def EOS(T):
    return A_RAD * T ** 4

def invEOS(e):
    return (e / A_RAD) ** 0.25

# ---------------------------------------------------------------------------
# Closures to compare
# ---------------------------------------------------------------------------

closures = {
    "P1":              closure_p1,
    "Kershaw":         closure_kershaw,
    "Levermore":       closure_levermore,
    "Minerbo poly":    closure_minerbo_poly,
    "Minerbo rational": closure_minerbo_rational,
}

# ---------------------------------------------------------------------------
# Run each closure
# ---------------------------------------------------------------------------

results = {}

output_times_ns = [tau * mean_free_time for tau in all_tau]

for name, closure_func in closures.items():
    print(f"\n── {name} ──")

    solver = M1Solver1D(
        x_min=x_min,
        x_max=x_max,
        n_cells=n_cells,
        d=0,
        sigma_func=sigma_func,
        EOS=EOS,
        invEOS=invEOS,
        dt=dt,
        closure_func=closure_func,
        left_bc_mode="reflect",
        right_bc_mode="marshak",
        max_nonlinear_iter=40,
        nonlinear_tol=1e-8,
    )
    solver.initialize(T_init=T_cold)

    source_mask    = solver.x_centers < source_width
    source_profile = np.where(source_mask, source_magnitude, 0.0)

    def source_func(t, _src=source_profile):
        return _src if t <= source_duration else np.zeros(n_cells)

    snapshots_ns = solver.run(
        n_steps=n_steps,
        source_func=source_func,
        output_times=output_times_ns,
    )

    # Re-key by tau and compute normalised radiation energy.
    snaps_tau = {}
    for tau, t_ns in zip(all_tau, output_times_ns):
        if t_ns in snapshots_ns:
            s = snapshots_ns[t_ns]
            snaps_tau[tau] = {
                "x":        s["x"],
                "Er":       s["Er"],
                "F":        s["F"],
                "T":        s["T"],
                "Er_norm":  s["Er"] / A_RAD,   # E_r / a  [keV⁴]
            }
            print(f"  τ = {tau:8.4f}  max(Er/a) = {snaps_tau[tau]['Er_norm'].max():.4f}")

    results[name] = snaps_tau

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

closure_colors = {
    "P1":               "#1f77b4",
    "Kershaw":          "#2ca02c",
    "Levermore":        "#d62728",
    "Minerbo poly":     "#9467bd",
    "Minerbo rational": "#8c564b",
}
closure_styles = {
    "P1":               ":",
    "Kershaw":          "-.",
    "Levermore":        "--",
    "Minerbo poly":     "-",
    "Minerbo rational": (0, (3, 1, 1, 1)),
}
tau_colors = ["blue", "green", "red", "purple", "orange", "brown"]

legend_lines = [
    Line2D([0], [0], color=closure_colors[n], linestyle=closure_styles[n],
           linewidth=2, label=rf"P$_{n[1]}$" if n == "P1" else n )
    for n in closures
] + [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor="gray", markeredgecolor="black",
           markersize=6, linestyle="", label="Reference"),
]


def _plot_panel(tau_list, x_max_plot, outname):
    fig, ax = plt.subplots(figsize=(8.5, 5.25))

    # Numerical results
    for name, snaps in results.items():
        for tau in tau_list:
            if tau not in snaps:
                continue
            s    = snaps[tau]
            mask = s["x"] <= x_max_plot
            ax.plot(s["x"][mask], s["Er_norm"][mask],
                    color=closure_colors[name], linestyle=closure_styles[name],
                    linewidth=2.0, alpha=0.8)

    # Transport reference
    for idx, tau in enumerate(tau_list):
        j = np.argmin(np.abs(su_olson_tau - tau))
        if np.abs(su_olson_tau[j] - tau) > 0.11:
            continue
        mask = su_olson_x <= x_max_plot
        ax.plot(su_olson_x[mask], transport_rad_energy[mask, j],
                marker="s", linestyle="", markersize=6,
                markerfacecolor=tau_colors[idx % len(tau_colors)],
                markeredgecolor="black", alpha=0.9, zorder=10)

    ax.axvline(x=source_width, color="gray", linestyle="--", alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Position (mean-free path)", fontsize=14)
    ax.set_ylabel(r"Norm. $E_\mathrm{r}$", fontsize=14)
    ax.legend(handles=legend_lines, fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    if x_max_plot <= 10.0:
        ax.set_xlim([0.2, 8.0])
        ax.set_ylim([1e-3, 3e0])
    else:
        ax.set_xlim([0.2, 2e1])
        ax.set_ylim([1e-3, 1e0])

    plt.tight_layout()
    show(outname, close_after=True)
    print(f"Saved {outname}")


# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

_plot_panel(early_tau, x_max_plot=10.0,
            outname="su_olson_m1_closures_early_times.pdf")
_plot_panel(late_tau,  x_max_plot=20.0,
            outname="su_olson_m1_closures_late_times.pdf")

print("\nDone.")
