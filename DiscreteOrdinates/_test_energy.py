"""Diagnostic: verify energy conservation with uniform hot sphere."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

I, N = 20, 4
R_MAX = 3.0
dr_val = R_MAX / I
r_left = np.arange(I, dtype=np.float64) * dr_val
dr     = np.full(I, dr_val, dtype=np.float64)
CV_VOL = 3e-6
AC = ac

@njit
def sigma_func(T):
    T_safe = np.maximum(T, 1e-3)
    return 300.0 * T_safe**(-3)

@njit
def scat_func(T):
    return np.zeros_like(T)

@njit
def eos(T):
    return CV_VOL * np.maximum(T, 0.0)

@njit
def invEOS(e):
    return np.maximum(e, 0.0) / CV_VOL

# --- Test 1: Uniform T=1 keV (should be stable) ---
T_ic   = np.full((I, 2), 1.0)
phi    = AC * T_ic**4
psi    = np.broadcast_to(phi[:,None,:], (I,N,2)).copy()
g_init = phi.copy()
q_n    = np.zeros((I, N, 2))
q_g    = np.zeros((I, 2))
bc0    = np.zeros((N, 2))

def BCs(t):
    return bc0, 0.0

r_right = r_left + dr
V = (4*np.pi/3) * (r_right**3 - r_left**3)
E0_rad = np.sum(0.5*(phi[:,0]+phi[:,1]) * V) / c
E0_mat = np.sum(0.5*(T_ic[:,0]+T_ic[:,1]) * CV_VOL * V)
print(f"Test 1 — Uniform T=1 keV, reflect_outer=True")
print(f"  Initial:  E_rad={E0_rad:.4e}  E_mat={E0_mat:.4e}  T_max={T_ic.max():.4f}")

phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-3, tfinal=1e-3,
    maxits=500, K=50, R=3, LOUD=False,
    reflect_outer=True, time_outputs=np.array([1e-3]),
)
T_final = Ts[0]
phi_final = phis[0]
E1_rad = np.sum(0.5*(phi_final[:,0]+phi_final[:,1]) * V) / c
E1_mat = np.sum(0.5*(T_final[:,0]+T_final[:,1]) * CV_VOL * V)
print(f"  Final:    E_rad={E1_rad:.4e}  E_mat={E1_mat:.4e}  T_max={T_final.max():.4f}")
print(f"  E ratio:  {(E1_rad+E1_mat)/(E0_rad+E0_mat):.6f}")
print()

# --- Test 2: Warm background T=0.01 keV (cold background) ---
T_ic2  = np.full((I, 2), 0.01)
phi2   = AC * T_ic2**4
psi2   = np.broadcast_to(phi2[:,None,:], (I,N,2)).copy()
g_init2 = phi2.copy()

E0_rad2 = np.sum(0.5*(phi2[:,0]+phi2[:,1]) * V) / c
E0_mat2 = np.sum(0.5*(T_ic2[:,0]+T_ic2[:,1]) * CV_VOL * V)
print(f"Test 2 — Uniform T=0.01 keV (cold)")
print(f"  Initial:  E_rad={E0_rad2:.4e}  E_mat={E0_mat2:.4e}  T_max={T_ic2.max():.4f}")

phis2, Ts2, gs2, its2, ts2, ips2 = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi2, psi2, T_ic2, g_init2,
    dt_min=1e-4, dt_max=1e-3, tfinal=1e-3,
    maxits=500, K=50, R=3, LOUD=False,
    reflect_outer=True, time_outputs=np.array([1e-3]),
)
T_f2 = Ts2[0]
phi_f2 = phis2[0]
E1_rad2 = np.sum(0.5*(phi_f2[:,0]+phi_f2[:,1]) * V) / c
E1_mat2 = np.sum(0.5*(T_f2[:,0]+T_f2[:,1]) * CV_VOL * V)
print(f"  Final:    E_rad={E1_rad2:.4e}  E_mat={E1_mat2:.4e}  T_max={T_f2.max():.4f}")
print(f"  E ratio:  {(E1_rad2+E1_mat2)/(E0_rad2+E0_mat2):.6f}")
