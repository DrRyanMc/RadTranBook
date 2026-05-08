"""Correct energy conservation test: uniform T=1 keV and T=0.01 keV."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

I, N = 20, 4
R_MAX = 3.0; CV_VOL = 3e-6; AC = ac
dr_val = R_MAX / I
r_left  = np.arange(I, dtype=np.float64) * dr_val
dr      = np.full(I, dr_val, dtype=np.float64)
r_right = r_left + dr
V = (4*np.pi/3) * (r_right**3 - r_left**3)

@njit
def sigma_func(T):
    return 300.0 * np.maximum(T, 1e-3)**(-3)
@njit
def scat_func(T): return np.zeros_like(T)
@njit
def eos(T): return CV_VOL * np.maximum(T, 0.0)
@njit
def invEOS(e): return np.maximum(e, 0.0) / CV_VOL

def total_E(phi_arr, T_arr):
    return float(np.sum((0.5*(phi_arr[:,0]+phi_arr[:,1])/c +
                         0.5*(T_arr[:,0]+T_arr[:,1])*CV_VOL) * V))

bc0 = np.zeros((N, 2))
def BCs(t): return bc0, 0.0

for T_test in [1.0, 0.01]:
    T_ic   = np.full((I, 2), T_test)
    phi    = AC * T_ic**4                         # phi = int I dmu = ac*T^4 at equil.
    psi    = np.broadcast_to((phi/2)[:,None,:], (I,N,2)).copy()  # I_n = phi/2 at equil.
    g_init = phi.copy() / 2                       # g(mu=-1) = phi/2 at equil.
    q_n    = np.zeros((I, N, 2))
    q_g    = np.zeros((I, 2))

    E0 = total_E(phi, T_ic)
    print(f"Test T={T_test} keV: E0={E0:.4e}")

    phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
        I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
        BCs, eos, invEOS, phi, psi, T_ic, g_init,
        dt_min=1e-4, dt_max=1e-4, tfinal=5e-4,
        maxits=500, K=50, R=3, LOUD=False,
        reflect_outer=True, print_stride=1,
    )
    # phis[-1] is the FINAL state (after all steps)
    E_f = total_E(phis[-1], Ts[-1])
    print(f"  After {len(ts)-1} steps: T_max={Ts[-1].max():.4f}  E={E_f:.4e}  ratio={E_f/E0:.6f}\n")
