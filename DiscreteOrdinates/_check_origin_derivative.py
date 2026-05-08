"""Check that d(phi)/dr = 0 at the origin for the Zeldovich solution."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

project_root = '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook'
for p in [project_root+'/Problems',
          project_root+'/nonEquilibriumDiffusion',
          project_root+'/nonEquilibriumDiffusion/problems']:
    if p not in sys.path:
        sys.path.insert(0, p)
from zeldovich import T_of_r_t

I, N = 40, 8
R_MAX = 3.0; T_INIT = 0.01; T_COLD = 0.01; CV_VOL = 3e-6; AC = ac
dr_val = R_MAX / I
r_left  = np.arange(I, dtype=np.float64) * dr_val
dr      = np.full(I, dr_val, dtype=np.float64)
r_right = r_left + dr
r_edges_flat = np.concatenate([r_left, r_right])

@njit
def sigma_func(T): return 300.0 * np.maximum(T, 1e-3)**(-3)
@njit
def scat_func(T): return np.zeros_like(T)
@njit
def eos(T): return CV_VOL * np.maximum(T, 0.0)
@njit
def invEOS(e): return np.maximum(e, 0.0) / CV_VOL

# Initial condition
T_flat, R_front = T_of_r_t(r_edges_flat, T_INIT, N=3)
T_ic = np.zeros((I, 2))
T_ic[:, 0] = np.maximum(T_flat[:I],  T_COLD)
T_ic[:, 1] = np.maximum(T_flat[I:],  T_COLD)
phi_ic = AC * T_ic**4

print("Initial Condition (t = 0.01 ns):")
print(f"  Origin cell (j=0): phi_left = {phi_ic[0,0]:.6e}, phi_right = {phi_ic[0,1]:.6e}")
print(f"  Relative difference: {abs(phi_ic[0,1]-phi_ic[0,0])/phi_ic[0,0]:.4e}")
print(f"  Finite-difference d(phi)/dr at origin ≈ {(phi_ic[0,1]-phi_ic[0,0])/dr_val:.4e}")
print()

# Analytical phi at several radii near origin
r_test = np.array([0.0, 0.001, 0.005, 0.01, 0.02])
T_test, _ = T_of_r_t(r_test, T_INIT, N=3)
phi_test = AC * np.maximum(T_test, T_COLD)**4
print("Analytical solution near origin:")
for i, (r_val, phi_val) in enumerate(zip(r_test, phi_test)):
    print(f"  r = {r_val:.3f} cm: phi = {phi_val:.6e}")
if phi_test[1] > 0:
    dphi_dr_origin = (phi_test[1] - phi_test[0]) / (r_test[1] - r_test[0])
    print(f"  Analytical d(phi)/dr at origin ≈ {dphi_dr_origin:.4e}")
print()

# Run solver
psi = np.broadcast_to(phi_ic[:,None,:], (I,N,2)).copy()
g_init = phi_ic.copy()
q_n = np.zeros((I, N, 2))
q_g = np.zeros((I, 2))
bc0 = np.zeros((N, 2))
def BCs(t): return bc0, 0.0

output_time = 0.1  # ns (physical)
time_rel = output_time - T_INIT
phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi_ic, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-2, tfinal=time_rel,
    maxits=500, K=50, R=3, LOUD=False,
    reflect_outer=True, time_outputs=np.array([time_rel]),
)

# Extract solution at t=0.1 ns
idx = 0  # only one output time
phi_final = phis[idx + 1]
T_final = Ts[idx + 1]

print(f"Numerical solution at t = {output_time} ns:")
print(f"  Origin cell (j=0): phi_left = {phi_final[0,0]:.6e}, phi_right = {phi_final[0,1]:.6e}")
print(f"  Relative difference: {abs(phi_final[0,1]-phi_final[0,0])/phi_final[0,0]:.4e}")
print(f"  Finite-difference d(phi)/dr at origin ≈ {(phi_final[0,1]-phi_final[0,0])/dr_val:.4e}")
print()

# Compare with analytical at t=0.1 ns
T_test_final, _ = T_of_r_t(r_test, output_time, N=3)
phi_test_final = AC * np.maximum(T_test_final, T_COLD)**4
print("Analytical solution near origin at t = 0.1 ns:")
for i, (r_val, phi_val) in enumerate(zip(r_test, phi_test_final)):
    print(f"  r = {r_val:.3f} cm: phi = {phi_val:.6e}")
if phi_test_final[1] > 0:
    dphi_dr_final = (phi_test_final[1] - phi_test_final[0]) / (r_test[1] - r_test[0])
    print(f"  Analytical d(phi)/dr at origin ≈ {dphi_dr_final:.4e}")
print()

# Check first few cells
print("Numerical phi profile for first 5 cells:")
for j in range(min(5, I)):
    r_center = r_left[j] + 0.5*dr_val
    phi_avg = 0.5*(phi_final[j,0] + phi_final[j,1])
    print(f"  j={j}: r_center={r_center:.4f} cm, phi_avg={phi_avg:.6e}")
