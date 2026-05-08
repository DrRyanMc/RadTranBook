"""Instrumented run to see what phi values are at origin."""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/rraymcclarren/Dropbox/Papers/RadTranBook/utils')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/DiscreteOrdinates')
from zeldovich import T_of_r_t
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, mu_GL, w_GL

# Minimal Zeldovich setup
T_MAX = 5.0
R_MAX = 3.0
I = 40
N = 8
K = 50
maxits = 500

dr = R_MAX / I
r_left = np.arange(I) * dr
r_right = (np.arange(I) + 1) * dr
r_centers = 0.5 * (r_left + r_right)

# IC at t_init = 0.01 ns
t_init = 0.01
T_ic = np.zeros((I, 2))
for j in range(I):
    T_temp, _ = T_of_r_t(np.array([r_left[j]]), t_init, N=3)
    T_ic[j, 0] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
    T_temp, _ = T_of_r_t(np.array([r_right[j]]), t_init, N=3)
    T_ic[j, 1] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
T_ic = np.maximum(T_ic, 0.01)

# Setup
def EOS(T):
    return np.maximum(T, 0.01) * 3e-6

def invEOS(e):
    return np.maximum(e / 3e-6, 0.01)

def sigma_func(T):
    return 300.0 / np.maximum(T**3, 0.01**3)

# No external source
q_n = np.zeros((I, N, 2))
q_g = np.zeros((I, 2))

BCs = np.zeros((N, 2))  # Reflecting outer BC
BCs_inner = None

# Run short solve
print("Running solver for 0.001 ns...")
try:
    phis, Ts, gs, iterations, ts, its_per_step = temp_solve_sph_ld(
        I, r_left, R_MAX/I * np.ones(I), q_n, q_g, 
        sigma_func, lambda T: 0.0,  # scattering = 0
        N, BCs, EOS, invEOS,
        phi_init=None, psi_init=None, T_init=T_ic, g_init=None,
        dt_min=1e-4, dt_max=1e-2, tfinal=0.001, maxits=maxits,
        time_outputs=np.array([0.0, 0.001]),
        MU=mu_GL[N], W=w_GL[N], mu_edges=np.array([-1.0, 0.0, 1.0]),
        reflect_outer=True, reflect_inner=False, BCs_inner=BCs_inner, full_sphere=True,
        verbose=True
    )
    print("\nSolver completed")
    print(f"Number of outputs: {len(phis)}")
    for i, (t, phi) in enumerate(phis.items()):
        print(f"\n  Output {i}: t={t:.6f} ns")
        print(f"    φ[0,0] = {phi[0,0]:.6e}")
        print(f"    φ[0,1] = {phi[0,1]:.6e}")
        print(f"    φ[0] average = {0.5*(phi[0,0] + phi[0,1]):.6e}")
        print(f"    Expected g[0] = φ[0]/2 = {0.25*(phi[0,0] + phi[0,1]):.6e}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
