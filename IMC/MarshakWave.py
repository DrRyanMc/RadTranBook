import IMCSlab as imc
import numpy as np
import matplotlib.pyplot as plt

a = imc.__a
c = imc.__c

# --- Problem parameters ---
Ntarget    = 100000
Nboundary  = 100000
NMax       = 4*10**5
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

# --- Output times (ns) ---
output_times = [1.0, 5.0, 10.0]
final_time   = max(output_times)

# --- Self-similar solution parameters ---
sigma_0      = sigma_a_f(T_boundary[0])
rho          = 1.0
xi_max       = 1.11305
omega        = 0.05989
K_const      = 8*a*c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)
T_bc         = T_boundary[0]
xi_vals      = np.linspace(0, xi_max, 300)
self_similar = lambda xi: (xi < xi_max) * np.power(
    np.where(xi < xi_max, (1 - xi/xi_max)*(1 + omega*xi/xi_max), 1e-30), 1/6)

# --- Run simulation, capturing snapshots at each output time ---
state     = imc.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos)
snapshots = []   # list of (time, T_material, T_radiation)
output_freq = 10
step = 0
for target_t in sorted(output_times):
    while state.time < target_t - 1e-12:
        step_dt = min(dt, target_t - state.time)
        state, info = imc.step(state, Ntarget, Nboundary, Nsource, NMax,
                               T_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True))
        if (state.time >= target_t - 1e-12) or (step % output_freq == 0):
            print("{:.6f}".format(info['time']), info['N_particles'],
                "{:.6f}".format(info['total_energy']),
                "{:.6f}".format(info['total_internal_energy']),
                "{:.6f}".format(info['total_radiation_energy']),
                "{:.6f}".format(info['boundary_emission']),
                "{:.6e}".format(info['energy_loss']), sep='\t')
        step += 1
    snapshots.append((state.time, state.temperature.copy(), state.radiation_temperature.copy()))

# --- Plot ---
fig, ax = plt.subplots()
colors = [f'C{i}' for i in range(len(snapshots))]

for (t_snap, T_mat, T_rad), color in zip(snapshots, colors):
    r_ss = xi_vals * (K_const * t_snap)**0.5
    T_ss = T_bc * self_similar(xi_vals)
    ax.plot(mesh_midpoints, T_mat, color=color, linestyle='-',
            label=f"Material T (t={t_snap:.1f})")
    ax.plot(mesh_midpoints, T_rad, color=color, linestyle='--',
            label=f"Radiation T (t={t_snap:.1f})")
    ax.plot(r_ss, T_ss,            color=color, linestyle=':',  linewidth=1.5,
            label=f"Self-similar (t={t_snap:.1f})")

ax.set_xlim([0, L])
ax.set_xlabel("Position (cm)")
ax.set_ylabel("Temperature (keV)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

#save the output data to an npz
# dicts are saved with allow_pickle; snapshots are split into flat arrays
snap_times = np.array([s[0] for s in snapshots])
snap_T_mat = np.array([s[1] for s in snapshots])
snap_T_rad = np.array([s[2] for s in snapshots])

fname = f"marshak_wave_output_{int(final_time*1e3)}ps_{Ntarget}.npz"
np.savez(fname,
    problem_parameters=np.array({
        "Ntarget": Ntarget, "Nboundary": Nboundary, "NMax": NMax, "Nsource": Nsource,
        "dt": dt, "L": L, "I": I, "T_boundary": T_boundary,
        "sigma_a_f": sigma_a_f.__name__, "cv_val": cv_val,
    }, dtype=object),
    self_similar_parameters=np.array({
        "sigma_0": sigma_0, "rho": rho, "xi_max": xi_max,
        "omega": omega, "K_const": K_const, "T_bc": T_bc,
    }, dtype=object),
    snap_times=snap_times,
    snap_T_mat=snap_T_mat,
    snap_T_rad=snap_T_rad,
)