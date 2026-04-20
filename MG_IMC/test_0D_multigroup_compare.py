#!/usr/bin/env python3
"""
0-D Infinite Medium Test - Gray vs Multigroup Comparison

This test compares the gray IMC (IMCSlab) with the multigroup IMC (MG_IMC2D)
for a simple 0-D infinite medium problem with gray opacities. Both simulations
should produce identical results since all energy groups have the same opacity.

The problem setup:
- Single cell with reflecting boundaries (infinite medium approximation)
- Constant opacity: σ = 1.0 cm⁻¹ (same for all groups)
- Initial conditions: T = 0.4 keV, Tr = 1.0 keV
- Material: e(T) = cv*T with cv = 0.01 GJ/(g·keV)
- No boundary sources, just internal relaxation toward equilibrium
"""

# Disable Numba cache to avoid compatibility issues
import os
os.environ['NUMBA_CACHE_DIR'] = ''

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'IMC'))

# Import gray IMC
import IMCSlab as imc

# Import multigroup IMC
from MG_IMC2D import run_simulation as mg_run_simulation
from MG_IMC2D import __c, __a

# Plotting style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std","TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
})

INK = "black"
ACCENT = "#005cb9"
GRAY_COLOR = "#666666"
MG_COLOR = "#d62728"  # Red for multigroup

print("="*80)
print("0-D INFINITE MEDIUM TEST: Gray vs Multigroup Comparison")
print("="*80)

# --- Problem parameters ---
Ntarget = 10000  # Increased for better statistics
Nboundary = 0
NMax = 10**6
Nsource = 0
dt = 0.025
L = 0.1  # Length of slab
I = 1    # Number of cells (single cell for 0-D)

# Gray IMC mesh
mesh = np.zeros((I, 2))
dx = L / I
for i in range(I):
    mesh[i] = [i*dx, (i+1)*dx]

# Initial conditions
Tinit = np.zeros(I) + 0.4
Trinit = np.zeros(I) + 0.5
T_boundary = (0.0, 0)
source = np.zeros(I)

# Material properties
sigma_a_value = 1.0  # Constant gray opacity
sigma_a_f = lambda T: sigma_a_value + 0*T
cv_val = 0.01
eos = lambda T: cv_val * T
inv_eos = lambda u: u / cv_val
cv = lambda T: cv_val

final_time = dt * 15

print(f"\nProblem setup:")
print(f"  Domain: {L} cm single cell (0-D)")
print(f"  Opacity: σ = {sigma_a_value} cm⁻¹ (gray)")
print(f"  Material: e = {cv_val}*T GJ/g")
print(f"  Initial T = {Tinit[0]} keV, Tr = {Trinit[0]} keV")
print(f"  Time step: dt = {dt} ns")
print(f"  Final time: {final_time} ns")
print(f"  Particles: Ntarget = {Ntarget}")

# --- Run gray IMC simulation ---
print(f"\n{'='*80}")
print("Running Gray IMC (IMCSlab)")
print(f"{'='*80}")

times_gray, radiation_temperatures_gray, temperatures_gray = imc.run_simulation(
    Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit,
    T_boundary, dt, mesh, sigma_a_f,
    eos, inv_eos, cv, source, final_time, 
    reflect=(True, True)
)

print(f"Gray IMC complete: {len(times_gray)} timesteps")

# --- Set up multigroup problem ---
print(f"\n{'='*80}")
print("Setting up Multigroup IMC")
print(f"{'='*80}")

# Energy groups - 5 groups covering reasonable range
energy_edges = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # keV
n_groups = len(energy_edges) - 1

print(f"\nEnergy groups: {n_groups}")
for g in range(n_groups):
    print(f"  Group {g}: [{energy_edges[g]:.1f}, {energy_edges[g+1]:.1f}] keV")

# All groups have same opacity (gray)
sigma_a_funcs = [lambda T, sigma_val=sigma_a_value: sigma_val + 0*T for _ in range(n_groups)]

# 2D mesh for MG_IMC2D (effectively 0-D with 1x1 cells)
# Make y dimension = 1.0 cm so cell volume = L * 1.0 = 0.1 cm³ (matches gray)
x_edges = np.array([0.0, L])
y_edges = np.array([0.0, 1.0])  # Unit length in y to match gray volume
nx, ny = 1, 1

# Initial conditions as 2D arrays
Tinit_2d = np.full((nx, ny), Tinit[0])
Trinit_2d = np.full((nx, ny), Trinit[0])
T_boundary_2d = [0.0, 0.0, 0.0, 0.0]  # No boundary sources

# Source function for MG_IMC2D (no sources)
def source_mg(t, dt):
    return ("boundary", None)

print(f"\nMultigroup setup:")
print(f"  Mesh: {nx}x{ny} cells")
print(f"  All groups: σ = {sigma_a_value} cm⁻¹")
print(f"  Boundaries: all reflecting (infinite medium)")

# --- Run multigroup IMC simulation ---
print(f"\n{'='*80}")
print("Running Multigroup IMC (MG_IMC2D)")
print(f"{'='*80}")

history_mg, final_state_mg = mg_run_simulation(
    Ntarget=Ntarget,
    Nboundary=Nboundary,
    Nsource=Nsource,
    Nmax=NMax,
    Tinit=Tinit_2d,
    Tr_init=Trinit_2d,
    T_boundary=T_boundary_2d,
    dt=dt,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=eos,
    inv_eos=inv_eos,
    cv=cv,
    source=source_mg,
    final_time=final_time,
    reflect=(True, True, True, True),  # All boundaries reflecting
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=False,  # Try without scalar intensity
    Ntarget_ic=Ntarget,
    conserve_comb_energy=False,
    geometry="xy",
    max_events_per_particle=1_000_000,
)

print(f"Multigroup IMC complete: {len(history_mg)} timesteps")

# --- Extract multigroup results ---
times_mg = np.array([h['time'] for h in history_mg])
temperatures_mg = np.array([h['temperature'][0, 0] for h in history_mg])
radiation_temperatures_mg = np.array([h['radiation_temperature'][0, 0] for h in history_mg])

# --- Compare results ---
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

print(f"\n{'Time':>8} {'Gray T':>10} {'MG T':>10} {'Gray Tr':>10} {'MG Tr':>10} {'ΔT':>10} {'ΔTr':>10}")
print("-"*80)
for i in range(len(times_gray)):
    # Find closest MG time
    idx_mg = np.argmin(np.abs(times_mg - times_gray[i]))
    delta_T = temperatures_mg[idx_mg] - temperatures_gray[i, 0]
    delta_Tr = radiation_temperatures_mg[idx_mg] - radiation_temperatures_gray[i, 0]
    print(f"{times_gray[i]:8.4f} {temperatures_gray[i,0]:10.6f} {temperatures_mg[idx_mg]:10.6f} "
          f"{radiation_temperatures_gray[i,0]:10.6f} {radiation_temperatures_mg[idx_mg]:10.6f} "
          f"{delta_T:10.6f} {delta_Tr:10.6f}")

# Compute RMS differences (handle different lengths)
min_len = min(len(temperatures_mg), len(temperatures_gray[:, 0]))
rms_T = np.sqrt(np.mean((temperatures_mg[:min_len] - temperatures_gray[:min_len, 0])**2))
rms_Tr = np.sqrt(np.mean((radiation_temperatures_mg[:min_len] - radiation_temperatures_gray[:min_len, 0])**2))

print(f"\nRMS differences:")
print(f"  Temperature:          {rms_T:.6f} keV")
print(f"  Radiation temperature: {rms_Tr:.6f} keV")

# --- Solve ODE for comparison ---
from scipy.integrate import solve_ivp

def RHS(t, y):
    T = inv_eos(y[1])
    Tr = (y[0] / __a) ** 0.25
    emission = __a * __c * sigma_a_f(T) * (Tr**4 - T**4)
    return [-emission, emission]

time_eval = np.linspace(0, final_time, 1000)
sol = solve_ivp(RHS, [0, np.max(times_gray)], 
                [Trinit[0]**4 * __a, eos(Tinit[0])], 
                t_eval=time_eval)

# --- Plot results ---
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(7, 5))

# Deterministic solution
plt.plot(time_eval, inv_eos(sol.y[1]), "-", color="black", alpha=0.3, 
         linewidth=2.5, label="Deterministic", zorder=1)
plt.plot(time_eval, (sol.y[0] / __a) ** 0.25, "--", color="black", alpha=0.3,
         linewidth=2.5, zorder=1)

# Gray IMC
plt.plot(times_gray, temperatures_gray[:, 0], "o", color=GRAY_COLOR, 
         alpha=0.8, label="Gray IMC", markersize=9, markerfacecolor='white', 
         markeredgewidth=1.5, zorder=3)
plt.plot(times_gray, radiation_temperatures_gray[:, 0], "o", color=GRAY_COLOR, 
         alpha=0.8, markersize=9, markerfacecolor='white', 
         markeredgewidth=1.5, zorder=3)

# Multigroup IMC
plt.plot(times_mg, temperatures_mg, "^", color=MG_COLOR, 
         alpha=0.8, label="Multigroup IMC", markersize=9, markerfacecolor='white', 
         markeredgewidth=1.5, zorder=4)
plt.plot(times_mg, radiation_temperatures_mg, "^", color=MG_COLOR, 
         alpha=0.8, markersize=9, markerfacecolor='white', 
         markeredgewidth=1.5, zorder=4)

plt.xlabel("t (ns)")
plt.ylabel("T (keV)")
#plt.ylim([0.35, 1.075])
plt.xlim([0., final_time * 1.05])
plt.legend(loc="lower right", frameon=True, fancybox=False, edgecolor='black')

# Add text box with RMS differences
textstr = f'RMS differences:\nΔT = {rms_T:.4f} keV\nΔTr = {rms_Tr:.4f} keV'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

# Remove outer spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
fname = "0D_gray_vs_multigroup.pdf"
plt.savefig(fname, dpi=600)
print(f"\nPlot saved to {fname}")
plt.show()

# --- Summary statistics ---
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nGray IMC (averaged over timesteps 1-):")
print(f"  Mean T  = {np.mean(temperatures_gray[1:, 0]):.6f} keV")
print(f"  Mean Tr = {np.mean(radiation_temperatures_gray[1:, 0]):.6f} keV")

print(f"\nMultigroup IMC (averaged over timesteps 1-):")
print(f"  Mean T  = {np.mean(temperatures_mg[1:]):.6f} keV")
print(f"  Mean Tr = {np.mean(radiation_temperatures_mg[1:]):.6f} keV")

print(f"\nDeterministic solution (averaged over t > 0):")
print(f"  Mean T  = {np.mean(inv_eos(sol.y[1][1:])):.6f} keV")
print(f"  Mean Tr = {np.mean((sol.y[0][1:] / __a) ** 0.25):.6f} keV")

if rms_T < 0.01 and rms_Tr < 0.01:
    print(f"\n✓ VALIDATION PASSED: Gray and multigroup results agree within tolerance")
else:
    print(f"\n⚠ WARNING: Gray and multigroup results differ by more than expected")
    print(f"   This may be due to Monte Carlo noise with Ntarget={Ntarget}")
    print(f"   Consider increasing Ntarget for better agreement")

print()
