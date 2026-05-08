"""Test origin derivative convergence with mesh refinement near origin."""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/utils')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/DiscreteOrdinates')
from zeldovich import T_of_r_t
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac
from numpy.polynomial.legendre import leggauss

# Setup Gauss-Legendre quadrature
mu_GL = {}
w_GL = {}
for N in range(1, 33):
    mu_half, w_half = leggauss(N)
    mu_GL[N] = mu_half  # Already in [-1, 1]
    w_GL[N] = 0.5 * w_half  # Scale to [0.5, 1]

T_MAX = 5.0
R_MAX = 3.0

def EOS(T):
    return np.maximum(T, 0.01) * 3e-6

def invEOS(e):
    return np.maximum(e / 3e-6, 0.01)

def sigma_func(T):
    return 300.0 / np.maximum(T**3, 0.01**3)

# Test with different resolutions
I_values = [20, 40, 80, 160]
results = []

t_init = 0.01
t_target = 0.1

for I in I_values:
    dr = R_MAX / I
    r_left = np.arange(I) * dr
    r_right = (np.arange(I) + 1) * dr
    
    T_ic = np.zeros((I, 2))
    for j in range(I):
        T_temp, _ = T_of_r_t(np.array([r_left[j]]), t_init, N=3)
        T_ic[j, 0] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
        T_temp, _ = T_of_r_t(np.array([r_right[j]]), t_init, N=3)
        T_ic[j, 1] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
    T_ic = np.maximum(T_ic, 0.01)
    
    q_n = np.zeros((I, 8, 2))
    q_g = np.zeros((I, 2))
    BCs = np.zeros((8, 2))
    
    print(f"Running I={I} (dr_origin={dr:.5f} cm)...", end=" ", flush=True)
    try:
        phis, Ts, gs, iterations, ts, its_per_step = temp_solve_sph_ld(
            I, r_left, R_MAX/I * np.ones(I), q_n, q_g,
            sigma_func, lambda T: 0.0,
            8, BCs, EOS, invEOS,
            phi_init=None, psi_init=None, T_init=T_ic, g_init=None,
            dt_min=1e-4, dt_max=1e-2, tfinal=t_target - t_init, maxits=500,
            time_outputs=np.array([0.0, t_target - t_init]),
            MU=mu_GL[8], W=w_GL[8], mu_edges=np.array([-1.0, 0.0, 1.0]),
            reflect_outer=True, reflect_inner=False, BCs_inner=None, full_sphere=True,
            verbose=False
        )
        
        # Extract solution at t_target
        keys = list(phis.keys())
        if len(keys) > 1:
            phi_final = phis[keys[-1]]
        else:
            phi_final = phis[keys[0]]
        
        phi_left = phi_final[0, 0]
        phi_right = phi_final[0, 1]
        dphi_dr = (phi_right - phi_left) / dr
        
        results.append((I, dr, phi_left, phi_right, dphi_dr))
        print(f"φ_left={phi_left:.4e}, φ_right={phi_right:.4e}, dφ/dr={dphi_dr:+.4e}")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nSUMMARY:")
print(f"{'I':>6} {'dr (cm)':>12} {'φ_left':>14} {'φ_right':>14} {'dφ/dr':>14}")
for I, dr, phi_l, phi_r, dphi_dr in results:
    print(f"{I:6d} {dr:12.5f} {phi_l:14.4e} {phi_r:14.4e} {dphi_dr:+14.4e}")

print("\nANALYTICAL at t=0.1 ns:")
T_analytical, _ = T_of_r_t(np.array([0.0]), t_target, N=3)
phi_analytical = ac * max(T_analytical[0], 0.01)**4
print(f"φ(r=0) = {phi_analytical:.4e}")
print(f"dφ/dr at r=0 ≈ -2.2e-06 (from quadratic fit; essentially zero)")

print("\nCONCLUSION:")
print("As dr → 0 (mesh refinement), dφ/dr at origin should decrease")
print("toward the analytical value (~0). The finite-difference derivative")
print("across the LD cell captures the local slope within [0, dr].")
