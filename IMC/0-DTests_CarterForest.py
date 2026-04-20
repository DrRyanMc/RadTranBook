import IMC1D_CarterForest as imc_cf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std","TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
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

    # Legend, if used
    "legend.frameon": False,
})
INK = "black"
MUTED = "#6f6f6f"       # close to black!55
PANEL_FILL = "#f2f2f2"  # close to black!5
PROJ = "#b0b0b0"        # light dashed guides
ACCENT = "#005cb9"      # your TikZ accent RGB(0,92,185)

# Set parameters
Ntarget = 10000  # Increased significantly to reduce variance
Nboundary = 0
NMax = 50000  # Increased proportionally
Nsource = 0
dt = 0.01  # Match paper's stable time step
L = 0.1  # length of slab
I = 1  # number of cells
mesh = np.zeros((I, 2))
dx = L / I
for i in range(I):
    mesh[i] = [i*dx, (i+1)*dx]

Tinit = np.zeros(I) + 0.4
Trinit = np.zeros(I) + 1.0
T_boundary = (0.0, 0)
source = np.zeros(I)
sigma_a_f = lambda T: 1.0 + 0*T  # Match paper: σ = 100 cm⁻¹
cv_val = 0.01
eos = lambda T: cv_val*T
inv_eos = lambda u: (u/cv_val)
cv = lambda T: cv_val + 0*T
final_time = 0.15  # Match paper's time range

# Run Carter-Forest simulation
print("="*70)
print("Running Carter-Forest IMC (time-delayed re-emission)")
print("="*70)
times, radiation_temperatures, temperatures = imc_cf.run_simulation(
    Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit,
    T_boundary, dt, mesh, sigma_a_f,
    eos, inv_eos, cv, source, final_time, reflect=(True, True), output_freq=1,
    use_scalar_intensity_Tr=False)  # Use census method, not track-length estimator

# Solve problem with scipy integrate (deterministic solution)
from scipy.integrate import solve_ivp

def RHS(t, y):
    T = inv_eos(y[1])
    Tr = (y[0]/(imc_cf.__a))**0.25
    emission = imc_cf.__a * imc_cf.__c * sigma_a_f(T) * (Tr**4 - T**4)
    return [-emission, emission]

time_eval = np.linspace(0, final_time, 1000)
sol = solve_ivp(RHS, [0, final_time], 
                [Trinit[0]**4*(imc_cf.__a), eos(Tinit[0])], 
                t_eval=time_eval)

# Get first matplotlib colors
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
MAT = my_colors[0]
RAD = my_colors[0]
MAT_B = my_colors[1]
RAD_B = my_colors[1]
MAT_C = my_colors[2]
RAD_C = my_colors[2]

plt.figure(figsize=(6, 4.5))

# Plot deterministic solution
plt.plot(time_eval, inv_eos(sol.y[1]), "-", color="black", alpha=0.5, label='Deterministic')
plt.plot(time_eval, ((sol.y[0]/(imc_cf.__a))**0.25), "--", color="black", alpha=0.5)

if I == 1:  # Single cell test - compare different time steps
    # Run with dt = 0.025
    for i in range(I):
        plt.plot(times, temperatures[:, i], color=MAT, alpha=0.8, 
                label=r"Carter-Forest $\Delta $" + f"t={dt}" if i==0 else None, 
                marker='^', markersize=8, markerfacecolor='white', 
                markeredgewidth=0.5, zorder=5)
        plt.plot(times, radiation_temperatures[:, i], "--", color=RAD, alpha=0.8, 
                marker='^', markersize=8, markerfacecolor='white', 
                markeredgewidth=0.5, zorder=5)
    
    # Run with dt = 0.05 (larger time step)
    dt2 = 0.025  # Restored to match paper comparison
    print("\n" + "="*70)
    print(f"Running Carter-Forest IMC with dt = {dt2}")
    print("="*70)
    times2, radiation_temperatures2, temperatures2 = imc_cf.run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit,
        T_boundary, dt2, mesh, sigma_a_f,
        eos, inv_eos, cv, source, final_time, reflect=(True, True), output_freq=1,
        use_scalar_intensity_Tr=False)
    
    for i in range(I):  
        plt.plot(times2, temperatures2[:, i], color=MAT_C, alpha=0.8, 
                label=r"Carter-Forest $\Delta $" + f"t={dt2}" if i==0 else None, 
                marker='o', markersize=8, markerfacecolor='white', markeredgewidth=0.5)
        plt.plot(times2, radiation_temperatures2[:, i], "--", color=RAD_C, alpha=0.8, 
                marker='o', markersize=8, markerfacecolor='white', markeredgewidth=0.5)
    
    # Run with dt = 0.01 (smaller time step)
    dt3 = 0.005  # Smallest time step for comparison
    print("\n" + "="*70)
    print(f"Running Carter-Forest IMC with dt = {dt3}")
    print("="*70)
    times3, radiation_temperatures3, temperatures3 = imc_cf.run_simulation(
        Ntarget, Nboundary, Nsource, NMax, Tinit, Trinit,
        T_boundary, dt3, mesh, sigma_a_f,
        eos, inv_eos, cv, source, final_time, reflect=(True, True), output_freq=1,
        use_scalar_intensity_Tr=False)
    
    for i in range(I):  
        plt.plot(times3, temperatures3[:, i], color=MAT_B, alpha=0.8, 
                label=r"Carter-Forest $\Delta $" + f"t={dt3}" if i==0 else None)
        plt.plot(times3, radiation_temperatures3[:, i], "--", color=RAD_B, alpha=0.8)
else:
    for i in range(I):
        plt.plot(times, temperatures[:, i], color=MAT, alpha=0.8, 
                label=r"Carter-Forest $\Delta $" + f"t={dt}" if i==-1 else None)
        plt.plot(times, radiation_temperatures[:, i], "--", color=RAD, alpha=0.8)

plt.xlabel("t (ns)")
plt.ylabel("T (keV)")
#plt.ylim([0.4, 1.075])
plt.xlim([0., 0.15])
plt.legend(loc="lower right", ncol=2, fontsize=9)

# Statistics
print("\n" + "="*70)
print("STATISTICS (excluding initial condition)")
print("="*70)
print(f"Carter-Forest (dt={dt}):")
print(f"  Mean radiation temperature: {np.mean(radiation_temperatures[1:, 0]):.6f} keV")
print(f"  Mean material temperature:  {np.mean(temperatures[1:, 0]):.6f} keV")
print(f"  Mean sum (T + Tr)/2:        {0.5*np.mean(radiation_temperatures[3:, 0] + temperatures[3:, 0]):.6f} keV")
print(f"\nDeterministic solution:")
print(f"  Mean radiation temperature: {np.mean(((sol.y[0][1:]/(imc_cf.__a))**0.25)):.6f} keV")
print(f"  Mean material temperature:  {np.mean(inv_eos(sol.y[1][1:])):.6f} keV")
print("="*70)

# Remove outer spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

fname = f"InfMedia_CarterForest_results_I={I}_dt={dt}.pdf"
plt.savefig(fname, dpi=600)
print(f"\nSaved: {fname}")
plt.show()
