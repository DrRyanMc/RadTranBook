"""Diagnostic: trace temperature evolution with Zeldovich IC."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

# Mimic zeldovich_sph_sn.py exactly
project_root = '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook'
for p in [project_root+'/Problems',
          project_root+'/nonEquilibriumDiffusion',
          project_root+'/nonEquilibriumDiffusion/problems']:
    if p not in sys.path:
        sys.path.insert(0, p)
from zeldovich import T_of_r_t

I, N = 20, 4
R_MAX = 3.0; T_INIT = 0.01; T_COLD = 0.01; CV_VOL = 3e-6; AC = ac

dr_val = R_MAX / I
r_left  = np.arange(I, dtype=np.float64) * dr_val
dr      = np.full(I, dr_val, dtype=np.float64)
r_right = r_left + dr
r_edges_flat = np.concatenate([r_left, r_right])
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

T_flat, R_front = T_of_r_t(r_edges_flat, T_INIT, N=3)
T_ic = np.zeros((I, 2))
T_ic[:, 0] = np.maximum(T_flat[:I],  T_COLD)
T_ic[:, 1] = np.maximum(T_flat[I:],  T_COLD)

phi    = AC * T_ic**4
psi    = np.broadcast_to(phi[:,None,:], (I,N,2)).copy()
g_init = phi.copy()   # NOT phi/2 — test with phi
q_n    = np.zeros((I, N, 2))
q_g    = np.zeros((I, 2))
bc0    = np.zeros((N, 2))

def BCs(t): return bc0, 0.0

E_init = np.sum((0.5*(phi[:,0]+phi[:,1])/c + 0.5*(T_ic[:,0]+T_ic[:,1])*CV_VOL) * V)
print(f"IC: T_max={T_ic.max():.3f}  T_min={T_ic.min():.4f}  R_front={R_front:.4f}")
print(f"Initial total E = {E_init:.6e} GJ")

phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-3, tfinal=1e-3,
    maxits=500, K=50, R=3, LOUD=False,
    reflect_outer=True, time_outputs=np.array([1e-3]),
    print_stride=1,
)
T_f = Ts[0]; phi_f = phis[0]
E_final = np.sum((0.5*(phi_f[:,0]+phi_f[:,1])/c + 0.5*(T_f[:,0]+T_f[:,1])*CV_VOL) * V)
print(f"\nAfter 1 step: T_max={T_f.max():.4f} keV  phi_max={phi_f.max():.3e}")
print(f"Final   total E = {E_final:.6e} GJ")
print(f"E ratio = {E_final/E_init:.6f}")
