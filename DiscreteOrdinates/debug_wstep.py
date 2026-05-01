"""Debug single time step with W>0 to find where T goes non-physical."""
import sys, numpy as np
sys.path.insert(0, '.')
import sn_solver_ld as sol

I = 50; N = 8; L = 0.20; hx = L / I
sigma_0 = 300.0; cv_val = 0.3
a = 0.01372; c = 29.98; ac = a * c
T_bc = 1.0; T_init = 1e-4

sigma_func = lambda T: sigma_0 * np.maximum(T, 1e-10)**(-3) * np.ones((I, 2))
scat_func  = lambda T: np.zeros((I, 2))
def eos(T):    return cv_val * T
def invEOS(e): return e / cv_val

MU_fixed, _ = np.polynomial.legendre.leggauss(N)
def BCFunc(t):
    bc = np.zeros((N, 2))
    for n, mu in enumerate(MU_fixed):
        if mu > 0:
            bc[n, 1] = ac * T_bc**4
    return bc

q   = np.zeros((I, N, 2))
phi = np.zeros((I, 2))
psi = np.zeros((I, N, 2))
T   = T_init * np.ones((I, 2))

phis, Ts, its, ts, ips = sol.temp_solve_ld(
    I, hx, q, sigma_func, scat_func, N, BCFunc, eos, invEOS,
    phi, psi, T,
    dt_min=0.025, dt_max=0.025, tfinal=0.15,
    tolerance=1e-8, maxits=50000,
    LOUD=False, fix=1, K=10, R=3,
    reflect_left=False, reflect_right=True,
    print_stride=1,
    use_dmd=False, W=5,
    tau_phi_max=0.1, tau_T=1e-6, omega_T=0.5,
)
for t, Ti in zip(ts, Ts):
    print(f't={t:.4f}  T_min={np.min(Ti):.4e}  T_max={np.max(Ti):.4e}  finite={np.all(np.isfinite(Ti))}')
