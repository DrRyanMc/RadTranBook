"""Check what g values the solver is computing at origin."""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/utils')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/DiscreteOrdinates')
from zeldovich import T_of_r_t
from sn_solver import ac

# Replicate the Zeldovich IC and source evaluation
T_MAX = 5.0  # keV
R_MAX = 3.0  # cm
I = 40
dr_uniform = R_MAX / I
r_left = np.arange(I) * dr_uniform
r_right = (np.arange(I) + 1) * dr_uniform
r_centers = 0.5 * (r_left + r_right)

# IC at t_init = 0.01 ns
t_init = 0.01
T_ic = np.zeros((I, 2))
for j in range(I):
    T_ic[j, 0], _ = T_of_r_t(np.array([r_left[j]]), t_init, N=3)
    T_ic[j, 1], _ = T_of_r_t(np.array([r_right[j]]), t_init, N=3)
T_ic = np.maximum(T_ic, 0.01)

phi_ic = ac * T_ic**4

print(f"IC at t={t_init} ns, origin cell (j=0):")
print(f"  r_left[0] = {r_left[0]:.6f} cm")
print(f"  r_right[0] = {r_right[0]:.6f} cm")
print(f"  T_ic[0,0] = {T_ic[0,0]:.6f} keV")
print(f"  T_ic[0,1] = {T_ic[0,1]:.6f} keV")
print(f"  φ_ic[0,0] = {phi_ic[0,0]:.6e} GJ/cm³")
print(f"  φ_ic[0,1] = {phi_ic[0,1]:.6e} GJ/cm³")
print(f"  Expected g[0] = φ_avg/2 = {0.5*np.mean(phi_ic[0,:]):.6e}")
print()

# For equilibrium, the source (Q_n) for the angular edge calculation comes from:
# Q_g = (sigma * phi - divergence of flux)
# At equilibrium (no flux divergence): Q_g = sigma * phi
# But in the solver, Q_g is constructed differently...

# Let's check what the solver would compute for g using the upwind formula
sigma = 300.0 * T_ic**(-3)  # Gray scattering
sigma_hat = sigma

# Origin cell g computation
j = 0
dz = r_right[j] - r_left[j]
sh_l = sigma_hat[j, 0]  # σ(T_l)
sh_r = sigma_hat[j, 1]  # σ(T_r)
Ql = phi_ic[j, 0]  # Q_g ≈ phi
Qr = phi_ic[j, 1]

print(f"Upwind g formula at j=0:")
print(f"  dz = {dz:.6f} cm")
print(f"  σ_hat[0,0] = {sh_l:.6e} cm⁻¹")
print(f"  σ_hat[0,1] = {sh_r:.6e} cm⁻¹")
print(f"  Q_l (φ_left) = {Ql:.6e} GJ/cm³")
print(f"  Q_r (φ_right) = {Qr:.6e} GJ/cm³")
print()

# With my current code, I compute:
# phi_est_l = Q_l / (2*σ_l), phi_est_r = Q_r / (2*σ_r), then g = (phi_est_l + phi_est_r)/2 * 0.5
phi_est_l = Ql / (2.0 * sh_l) if sh_l > 0.0 else 0.0
phi_est_r = Qr / (2.0 * sh_r) if sh_r > 0.0 else 0.0
phi_est = 0.5 * (phi_est_l + phi_est_r)
g_cell = 0.5 * phi_est
print(f"Current origin g formula:")
print(f"  phi_est_l = Q_l / (2*σ_l) = {phi_est_l:.6e}")
print(f"  phi_est_r = Q_r / (2*σ_r) = {phi_est_r:.6e}")
print(f"  phi_est = (phi_est_l + phi_est_r)/2 = {phi_est:.6e}")
print(f"  g_cell = 0.5 * phi_est = {g_cell:.6e}")
print()

# What about the standard upwind formula (non-origin)?
Q_avg = 0.5 * (Ql + Qr)
sh_avg = 0.5 * (sh_l + sh_r)
# g_in at outer wall (should be reflecting, so bc_g_outer = 0)
bc_g_outer = 0.0
g_in = bc_g_outer
denom = sh_avg + 1.0 / dz
g_upwind = (Q_avg + g_in / dz) / denom if denom > 0.0 else g_in
print(f"Standard upwind formula:")
print(f"  Q_avg = {Q_avg:.6e}")
print(f"  σ_avg = {sh_avg:.6e}")
print(f"  g_in = {g_in:.6e} (reflecting BC)")
print(f"  g_upwind = (Q_avg + g_in/dz) / (σ_avg + 1/dz) = {g_upwind:.6e}")
print()

print("ANALYSIS:")
print(f"  Expected g[0] from equilibrium: {0.5*np.mean(phi_ic[0,:]):.6e}")
print(f"  Current origin formula gives: {g_cell:.6e}")
print(f"  Upwind formula gives: {g_upwind:.6e}")
print(f"  Ratio g_current/φ_avg = {g_cell / np.mean(phi_ic[0,:]):.6f}")
print(f"  Ratio g_upwind/φ_avg = {g_upwind / np.mean(phi_ic[0,:]):.6f}")
