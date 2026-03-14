"""compare_Tr_estimators.py

Runs the Marshak-wave problem to t=1 ns twice with identical random seeds and
compares the two radiation-temperature estimators available in IMCSlab.step():

  1. Scalar-intensity estimator (default):
         T_r = (scalar_intensity / (a*c))^0.25
     The scalar intensity is the time-averaged flux accumulated during particle
     transport, so it includes contributions from particles that escape during
     the step.

  2. Weight-bincount estimator:
         T_r = (sum_of_weights_in_cell / (dx * a))^0.25
     Uses the end-of-step particle weights — a snapshot of surviving particles.
"""

import numpy as np
import matplotlib.pyplot as plt
import IMCSlab as imc

a = imc.__a
c = imc.__c

# ── Problem parameters (same as MarshakWave.py) ────────────────────────────
Ntarget    = 10000
Nboundary  = 10000
NMax       = 4 * 10**4
Nsource    = 0
dt         = 0.01
L          = 0.20
I          = 50
mesh       = np.zeros((I, 2))
dx         = L / I
for i in range(I):
    mesh[i] = [i*dx, (i+1)*dx]
mesh_midpoints = 0.5 * (mesh[:, 0] + mesh[:, 1])

Tinit      = np.zeros(I) + 1e-4
Trinit     = np.zeros(I) + 1e-4
T_boundary = (1.0, 0)
source     = np.zeros(I)
sigma_a_f  = lambda T: 300 * T**-3
cv_val     = 0.3
eos        = lambda T: cv_val * T
inv_eos    = lambda u: u / cv_val
cv         = lambda T: cv_val

final_time  = 1.0
output_freq = 10

# ── Self-similar reference ──────────────────────────────────────────────────
sigma_0   = sigma_a_f(T_boundary[0])
rho       = 1.0
xi_max    = 1.11305
omega     = 0.05989
K_const   = 8 * a * c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)
T_bc      = T_boundary[0]
xi_vals   = np.linspace(0, xi_max, 300)
self_similar = lambda xi: (xi < xi_max) * np.power(
    np.where(xi < xi_max, (1 - xi/xi_max) * (1 + omega*xi/xi_max), 1e-30), 1/6)


def run(use_scalar_intensity_Tr: bool, seed: int = 42):
    """Run to final_time and return final (T_material, T_radiation)."""
    np.random.seed(seed)
    state = imc.init_simulation(Ntarget, Tinit, Trinit, mesh, eos, inv_eos)
    step_count = 0
    while state.time < final_time - 1e-12:
        step_dt = min(dt, final_time - state.time)
        state, info = imc.step(
            state, Ntarget, Nboundary, Nsource, NMax,
            T_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
            reflect=(False, True),
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
        )
        if step_count % output_freq == 0 or state.time >= final_time - 1e-12:
            print("{:.6f}".format(info['time']), info['N_particles'],
                  "{:.6f}".format(info['total_energy']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')
        step_count += 1
    return state.temperature.copy(), state.radiation_temperature.copy()


# ── Run both estimators ─────────────────────────────────────────────────────
print("=== Scalar-intensity estimator ===")
T_mat_si, T_rad_si = run(use_scalar_intensity_Tr=True,  seed=42)

print("\n=== Weight-bincount estimator ===")
T_mat_wt, T_rad_wt = run(use_scalar_intensity_Tr=False, seed=42)

# ── Self-similar reference at t=1 ns ───────────────────────────────────────
r_ss = xi_vals * (K_const * final_time) ** 0.5
T_ss = T_bc * self_similar(xi_vals)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, (T_mat, T_rad, label) in zip(axes, [
    (T_mat_si, T_rad_si, "Scalar intensity"),
    (T_mat_wt, T_rad_wt, "Weight bincount"),
]):
    ax.plot(mesh_midpoints, T_mat, 'C0-',  label="Material T")
    ax.plot(mesh_midpoints, T_rad, 'C1--', label=r"Radiation $T_r$")
    ax.plot(r_ss,           T_ss,  'k:',  linewidth=1.5, label="Self-similar")
    ax.set_title(f"$T_r$ estimator: {label}", fontsize=10)
    ax.set_xlabel("Position (cm)")
    ax.set_xlim([0, L])
    ax.legend(fontsize=8)
axes[0].set_ylabel("Temperature (keV)")

fig.suptitle(f"Marshak wave at $t = {final_time}$ ns — comparison of $T_r$ estimators",
             fontsize=11)
plt.tight_layout()
plt.savefig("compare_Tr_estimators.pdf")
plt.show()

# ── Difference panel ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(mesh_midpoints, T_rad_si - T_rad_wt, 'C2-')
ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax2.set_xlabel("Position (cm)")
ax2.set_ylabel(r"$T_r^{\rm SI} - T_r^{\rm WB}$ (keV)")
ax2.set_title(r"Difference: scalar-intensity minus weight-bincount $T_r$", fontsize=10)
ax2.set_xlim([0, L])
plt.tight_layout()
plt.savefig("compare_Tr_estimators_diff.pdf")
plt.show()
