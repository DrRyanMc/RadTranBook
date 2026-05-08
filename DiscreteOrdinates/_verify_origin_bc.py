"""Verify origin boundary condition is correct after post-processing."""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/utils')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/DiscreteOrdinates')
from zeldovich import T_of_r_t
from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac
from numpy.polynomial.legendre import leggauss

# Setup Gauss-Legendre quadrature
N = 8
mu_half, w_half = leggauss(N)
MU = mu_half
W = 0.5 * w_half

T_MAX = 5.0
R_MAX = 3.0
I = 40

# IC and setup
dr = R_MAX / I
r_left = np.arange(I) * dr
r_right = (np.arange(I) + 1) * dr

t_init = 0.01
T_ic = np.zeros((I, 2))
for j in range(I):
    T_temp, _ = T_of_r_t(np.array([r_left[j]]), t_init, N=3)
    T_ic[j, 0] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
    T_temp, _ = T_of_r_t(np.array([r_right[j]]), t_init, N=3)
    T_ic[j, 1] = T_temp[0] if isinstance(T_temp, np.ndarray) else T_temp
T_ic = np.maximum(T_ic, 0.01)

def EOS(T):
    return np.maximum(T, 0.01) * 3e-6

def invEOS(e):
    return np.maximum(e / 3e-6, 0.01)

def sigma_func(T):
    return 300.0 / np.maximum(T**3, 0.01**3)

q_n = np.zeros((I, N, 2))
q_g = np.zeros((I, 2))
BCs = np.zeros((N, 2))

t_target = 0.1

print("Running Zeldovich LD-S_N solve...")
phis, Ts, gs, iterations, ts, its_per_step = temp_solve_sph_ld(
    I, r_left, R_MAX/I * np.ones(I), q_n, q_g,
    sigma_func, lambda T: 0.0,
    N, BCs, EOS, invEOS,
    None, None, T_ic, None,  # phi, psi, T, g_init
    dt_min=1e-4, dt_max=1e-2, tfinal=t_target - t_init, maxits=500,
    time_outputs=np.array([0.0, t_target - t_init]),
    W=W, mu_edges=np.array([-1.0, 0.0, 1.0]),
    reflect_outer=True, reflect_inner=False, BCs_inner=None, full_sphere=True,
    LOUD=False
)

# Extract final solution
keys = list(phis.keys())
phi_final = phis[keys[-1]]

print(f"\nAt t = {t_target} ns:")
print(f"  Numerical solution at origin cell (j=0):")
print(f"    φ[0, 0] (left)  = {phi_final[0, 0]:.6e} GJ/cm³")
print(f"    φ[0, 1] (right) = {phi_final[0, 1]:.6e} GJ/cm³")
phi_avg = 0.5 * (phi_final[0, 0] + phi_final[0, 1])
g_val = gs[keys[-1]][0, 0]  # Both g[0,0] and g[0,1] should be equal by post-processing
print(f"    Average φ at origin = {phi_avg:.6e}")
print(f"    g at origin = {g_val:.6e}")
print(f"    Ratio g/φ_avg = {g_val / phi_avg:.6f} (should be 0.5 for origin regularity)")
print()

# Compare with analytical
T_analytical, _ = T_of_r_t(np.array([0.0]), t_target, N=3)
phi_analytical = ac * max(T_analytical[0], 0.01)**4
print(f"  Analytical solution at origin:")
print(f"    φ(0) = {phi_analytical:.6e} GJ/cm³")
print(f"    Error in φ = {abs(phi_analytical - phi_avg):.2e} ({100*abs(phi_analytical - phi_avg)/phi_analytical:.1f}%)")
print()

# Check origin regularity: g should equal φ/2
print("  Origin regularity verification (g = φ/2):")
print(f"    φ_avg/2 = {phi_avg/2:.6e}")
print(f"    g        = {g_val:.6e}")
print(f"    Match: {abs(g_val - phi_avg/2) / (phi_avg/2) < 1e-10}")
