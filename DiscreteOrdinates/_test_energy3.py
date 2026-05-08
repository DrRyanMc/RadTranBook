"""Deep energy diagnostic: track E_rad and E_mat separately step by step."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld_sphere import temp_solve_sph_ld, single_sweep_phi_sph_ld
from sn_solver import ac, c
from numba import njit

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
phi  = AC * T_ic**4
psi  = np.broadcast_to(phi[:,None,:], (I,N,2)).copy()
g_init = phi.copy()
q_n    = np.zeros((I, N, 2))
q_g    = np.zeros((I, 2))
bc0    = np.zeros((N, 2))
def BCs(t): return bc0, 0.0

def total_E(phi_arr, T_arr):
    return np.sum((0.5*(phi_arr[:,0]+phi_arr[:,1])/c + 0.5*(T_arr[:,0]+T_arr[:,1])*CV_VOL) * V)
def E_rad(phi_arr):
    return np.sum(0.5*(phi_arr[:,0]+phi_arr[:,1])/c * V)
def E_mat(T_arr):
    return np.sum(0.5*(T_arr[:,0]+T_arr[:,1])*CV_VOL * V)

E0 = total_E(phi, T_ic)
print(f"IC: T_max={T_ic.max():.3f}  T_min={T_ic.min():.4f}  R_front={R_front:.4f}")
print(f"Initial: E_rad={E_rad(phi):.4e}  E_mat={E_mat(T_ic):.4e}  E_total={E0:.4e}")
print()

# Run step by step using small dt_max
phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-4, tfinal=5e-4,
    maxits=500, K=50, R=3, LOUD=False,
    reflect_outer=True, time_outputs=None,
    print_stride=1,
)

print("\nStep-by-step energy:")
for i, (t_val, phi_s, T_s) in enumerate(zip(ts, phis, Ts)):
    Er = E_rad(phi_s); Em = E_mat(T_s); Et = Er + Em
    print(f"  t={t_val:.4e}: T_max={T_s.max():.4f}  E_rad={Er:.4e}  E_mat={Em:.4e}  E_total={Et:.4e}  ratio={Et/E0:.6f}")
