"""Quick Marshak wave check using IMC1D (slab geometry, t=1 ns only)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import IMC1D as imc
import numpy as np

a = imc.__a
c = imc.__c

Ntarget    = 10000
Nboundary  = 10000
NMax       = 4*10**4
Nsource    = 0
dt         = 0.01
L          = 0.20
I          = 50
mesh       = np.zeros((I, 2))
dx         = L / I
for i in range(I):
    mesh[i] = [i*dx, (i+1)*dx]
mesh_midpoints = 0.5*(mesh[:,0] + mesh[:,1])
Tinit      = np.zeros(I) + 1e-4
Trinit     = np.zeros(I) + 1e-4
T_boundary = (1.0, 0)
source     = np.zeros(I)
sigma_a_f  = lambda T: 300*T**-3
cv_val     = 0.3
eos        = lambda T: cv_val*T
inv_eos    = lambda u: u/cv_val
cv         = lambda T: cv_val

output_times = [1.0]   # keep fast: single snapshot
final_time   = max(output_times)

# Self-similar solution
sigma_0  = sigma_a_f(T_boundary[0])
rho      = 1.0
xi_max   = 1.11305
omega    = 0.05989
K_const  = 8*a*c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)
T_bc     = T_boundary[0]
xi_vals  = np.linspace(0, xi_max, 300)
self_similar = lambda xi: (xi < xi_max) * np.power(
    np.where(xi < xi_max, (1 - xi/xi_max)*(1 + omega*xi/xi_max), 1e-30), 1/6)

state     = imc.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos,
                                geometry='slab')
snapshots = []
output_freq = 10
step_count  = 0
for target_t in sorted(output_times):
    while state.time < target_t - 1e-12:
        step_dt = min(dt, target_t - state.time)
        state, info = imc.step(state, Ntarget, Nboundary, Nsource, NMax,
                               T_boundary, step_dt, mesh, sigma_a_f, inv_eos,
                               cv, source, reflect=(False, True),
                               geometry='slab')
        if (state.time >= target_t - 1e-12) or (step_count % output_freq == 0):
            print("{:.3f}".format(info['time']), info['N_particles'],
                  "{:.6e}".format(info['total_energy']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')
        step_count += 1
    snapshots.append((state.time,
                      state.temperature.copy(),
                      state.radiation_temperature.copy()))

t_snap, T_mat, T_rad = snapshots[0]
print("\nt =", t_snap)
print("Max material T  :", T_mat.max())
print("Max radiation T :", T_rad.max())
front_idx = np.where(T_mat > 0.01)[0]
if len(front_idx):
    print("Wave front (T_mat > 0.01) at x =", mesh[front_idx[-1], 1])
else:
    print("Front not yet reaching 0.01 keV")

# Self-similar front for comparison
ss_front = xi_vals[-1] * (K_const * t_snap)**0.5
print("Self-similar front position   :", ss_front)

# Save a plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(mesh_midpoints, T_mat, 'C0-',  label="Material T  (IMC1D)")
ax.plot(mesh_midpoints, T_rad, 'C0--', label="Radiation T (IMC1D)")
r_ss = xi_vals * (K_const * t_snap)**0.5
T_ss = T_bc * self_similar(xi_vals)
ax.plot(r_ss, T_ss, 'k:', linewidth=1.5, label="Self-similar")
ax.set_xlim([0, L]); ax.set_xlabel("x (cm)"); ax.set_ylabel("T (keV)")
ax.set_title(f"Marshak wave via IMC1D  (t = {t_snap:.1f} ns)")
ax.legend()
plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "_marshak_imc1d_check.png")
plt.savefig(outfile, dpi=150)
print("\nPlot saved to", outfile)
