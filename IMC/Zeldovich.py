"""Zel'dovich–Kompaneets–Barenblatt nonlinear radiation diffusion solved with IMC.

Plane geometry (N=1): a pulse of radiation energy E0 is deposited at x=0.
A reflecting boundary at x=0 simulates the symmetric half-space.
The self-similar analytic solution is overlaid at each output time.

Physical model
--------------
  sigma(T) = sigma0 * T^{-n}        (power-law opacity)
  cv        <<  a*T^3               (radiation-dominated: material energy negligible)

Self-similar solution
---------------------
  T(r,t) = t^{-alpha} * A^{1/(m-s)} * (1 - r^2/R(t)^2)^{1/(m-s)}   for r < R(t)
  R(t)   = eta_f * t^beta
  m = s + n = 7,  p = m/s = 7/4,  beta = 4/11,  alpha = 1/11   (N=1)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import gamma, pi, sqrt
from scipy.special import beta as Beta
import IMCSlab as imc

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
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

# ------------------------------------------------------------------ #
#  Problem parameters                                                  #
# ------------------------------------------------------------------ #
s      = 4          # radiation: energy density ~ T^s
n      = 1.0        # opacity exponent: sigma ~ T^{-n}
sigma0 = 300.0      # opacity coefficient (cm^{-1} keV^n)
a      = imc.__a    # radiation constant  (GJ cm^{-3} keV^{-4})
c      = imc.__c    # speed of light      (cm ns^{-1})
ac     = a * c
E0     = 1.0        # total injected energy per unit area (GJ cm^{-2})
cv_val = 1e-5       # negligible heat capacity → radiation-dominated (Fleck f << 1)

m = s + n           # 4
p = m / s           # 4

# ------------------------------------------------------------------ #
#  Self-similar analytic solution (N = 1, plane geometry)             #
# ------------------------------------------------------------------ #
N = 1

kappa = (4.0 / (4.0 + n)) * (ac / (3.0 * sigma0))
D     = kappa / a   # diffusion coefficient in the u-equation

beta_ss  = 1.0 / (N * (p - 1.0) + 2.0)        # 4/11
alpha_ss = (N * beta_ss) / s                   # 1/11
lam      = (p - 1.0) * beta_ss / (2.0 * p * D)

# Compute constant A from conservation of total energy M = E0/a
M    = E0 / (np.max([a,cv_val]))
SN   = 2.0 * pi**(N / 2.0) / gamma(N / 2.0)   # = 2  (surface area of unit circle in R^1)
a_   = N / 2.0                                  # = 1/2
bpar = 1.0 / (p - 1.0) + 1.0                   # = 7/3
B    = Beta(a_, bpar)
power = 1.0 / (p - 1.0) + N / 2.0             # = 11/6
A    = ((2.0 * M * lam**(N / 2.0)) / (SN * B)) ** (1.0 / power)
eta_f = sqrt(A / lam)

print(f"Self-similar parameters: alpha={alpha_ss:.6f}, beta={beta_ss:.6f}")
print(f"  A={A:.6f}, lambda={lam:.6f}, eta_f={eta_f:.6f}")


def T_analytic(r, t):
    """Analytic temperature profile at time t."""
    R      = eta_f * t**beta_ss
    inside = 1.0 - (r / R)**2
    T      = np.zeros_like(r, dtype=float)
    mask   = inside > 0.0
    T[mask] = t**(-alpha_ss) * A**(1.0 / (m - s)) * inside[mask]**(1.0 / (m - s))
    return T, R

# ------------------------------------------------------------------ #
#  Mesh: sized to contain the wave front at the final output time      #
# ------------------------------------------------------------------ #
output_times = [0.5] #, 0.3, 1.0, 3.0]
final_time   = max(output_times)
t_start      = 0.01    # initialise from self-similar solution at this time

R_final = eta_f * final_time**beta_ss
L  = 1.5 * R_final
I  = 100
mesh = np.zeros((I, 2))
dx   = L / I
for i in range(I):
    mesh[i] = [i * dx, (i + 1) * dx]
mesh_midpoints = 0.5 * (mesh[:, 0] + mesh[:, 1])

print(f"Domain: L={L:.4f} cm,  R(t={final_time})={R_final:.4f} cm,  dx={dx:.4f} cm")

# Initial condition: self-similar solution at t_start
Trinit, R_start = T_analytic(mesh_midpoints, t_start)
# Einit = 2*a * Trinit**4
# Trinit = (Einit/a)**.25
Tmin = 0.01
Trinit = np.maximum(Trinit, Tmin)   # avoid zero temperature
Tinit  = Trinit.copy()              # material in equilibrium with radiation

print(f"Starting from self-similar solution at t_start={t_start} ns,  R(t_start)={R_start:.4f} cm")
print(f"Peak radiation temperature: {Trinit.max():.4f} keV")

T_boundary   = (0.0, 0.0)           # vacuum right; left is reflecting (see reflect below)
source       = np.zeros(I)
sigma_a_f    = lambda T: sigma0 * T**(-n)
eos          = lambda T: cv_val * T
inv_eos      = lambda u: u / cv_val
cv           = lambda T: cv_val

# ------------------------------------------------------------------ #
#  IMC parameters                                                      #
# ------------------------------------------------------------------ #
Ntarget    = 10**5
Ntarget_init = 10**5
Nboundary  = 0
NMax       = 2 * Ntarget_init
Nsource    = 0
dt         = 0.01
output_freq = 50

# ------------------------------------------------------------------ #
#  Run simulation, capturing snapshots at each output time             #
# ------------------------------------------------------------------ #
state      = imc.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos, Ntarget_ic=Ntarget_init)
state.time = t_start   # shift clock to match the self-similar initial condition
snapshots  = []
step_count = 0

for target_t in sorted(output_times):
    while state.time < target_t - 1e-12:
        step_dt = min(dt, target_t - state.time)
        state, info = imc.step(state, Ntarget, Nboundary, Nsource, NMax,
                               T_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
                               reflect=(True, False),use_scalar_intensity_Tr=True)
        if (state.time >= target_t - 1e-12) or (step_count % output_freq == 0):
            print("{:.6f}".format(info['time']), info['N_particles'],
                  "{:.6f}".format(info['total_energy']),
                  "{:.6f}".format(info['total_internal_energy']),
                  "{:.6f}".format(info['total_radiation_energy']),
                  "{:.6f}".format(info['boundary_emission']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')
        step_count += 1
    snapshots.append((state.time, state.temperature.copy(), state.radiation_temperature.copy()))

# ------------------------------------------------------------------ #
#  Plot                                                                #
# ------------------------------------------------------------------ #
my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
r_plot = np.linspace(0, L, 2000)
line_styles = ['-', '--', '-.', (0, (8, 2, 2, 2))]

fig, ax = plt.subplots(figsize=(6, 4.5))
for idx, (t_snap, T_mat, T_rad) in enumerate(snapshots):
    color = my_colors[idx % len(my_colors)]
    T_an, R_an = T_analytic(r_plot, t_snap)
    lstyle = line_styles[idx % len(line_styles)]
    ax.plot(mesh_midpoints, T_mat, color=color, linestyle=lstyle,
            label=f"IMC  t={t_snap:g} ns")
    ax.plot(mesh_midpoints, T_rad, color=color, linestyle='--',
            label=f"IMC  t={t_snap:g} ns")
    ax.plot(r_plot, T_an,          color=color, linestyle=lstyle, linewidth=1.0,
            alpha=0.5, label=f"Analytic  t={t_snap:g} ns")

ax.set_xlim([0, L])
ax.set_xlabel("position (cm)")
ax.set_ylabel("temperature (keV)")
ax.legend(fontsize=7, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig("zeldovich_imc.pdf", dpi=600)
plt.show()

# ------------------------------------------------------------------ #
#  Save output                                                         #
# ------------------------------------------------------------------ #
snap_times = np.array([s[0] for s in snapshots])
snap_T_mat = np.array([s[1] for s in snapshots])
snap_T_rad = np.array([s[2] for s in snapshots])

fname = f"zeldovich_imc_output_{int(final_time*1e3)}ps.npz"
np.savez(fname,
    problem_parameters=np.array({
        "s": s, "n": n, "sigma0": sigma0, "E0": E0, "cv_val": cv_val,
        "L": L, "I": I, "dx": dx,
    }, dtype=object),
    analytic_parameters=np.array({
        "alpha": alpha_ss, "beta": beta_ss,
        "A": A, "lam": lam, "eta_f": eta_f,
        "m": m, "p": p,
    }, dtype=object),
    snap_times=snap_times,
    snap_T_mat=snap_T_mat,
    snap_T_rad=snap_T_rad,
    mesh_midpoints=mesh_midpoints,
)
print(f"Saved to {fname}")
