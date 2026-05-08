"""Quick hollow-shell smoke test after source-term fix."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

I, N = 20, 4
R_inner, R_outer = 1.0, 3.0
dr_val = (R_outer - R_inner) / I
r_left = R_inner + np.arange(I, dtype=np.float64) * dr_val
dr     = np.full(I, dr_val, dtype=np.float64)
r_right = r_left + dr
V = (4*np.pi/3) * (r_right**3 - r_left**3)
CV_VOL = 3e-6; AC = ac

@njit
def sigma_func(T): return 300.0 * np.maximum(T, 1e-3)**(-3)
@njit
def scat_func(T): return np.zeros_like(T)
@njit
def eos(T): return CV_VOL * np.maximum(T, 0.0)
@njit
def invEOS(e): return np.maximum(e, 0.0) / CV_VOL

T_ic   = np.full((I, 2), 1.0)  # uniform 1 keV
phi    = AC * T_ic**4
psi    = np.broadcast_to((phi/2)[:,None,:], (I,N,2)).copy()  # I_n = phi/2 at equil.
g_init = phi.copy() / 2                                       # g = phi/2 at equil.
q_n    = np.zeros((I, N, 2))
q_g    = np.zeros((I, 2))
bc0    = np.zeros((N, 2))

def BCs(t): return bc0, 0.0

E0 = np.sum((0.5*(phi[:,0]+phi[:,1])/c + T_ic[:,0]*CV_VOL) * V)
print(f"Hollow shell T=1 keV: E0={E0:.4e}")

phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-4, tfinal=5e-4,
    maxits=200, K=20, R=3, LOUD=False,
    reflect_outer=True, reflect_inner=True,
    print_stride=1,
)
Ef = np.sum((0.5*(phis[-1][:,0]+phis[-1][:,1])/c + Ts[-1][:,0]*CV_VOL) * V)
print(f"After 5 steps: T_max={Ts[-1].max():.4f}  E={Ef:.4e}  ratio={Ef/E0:.6f}")
