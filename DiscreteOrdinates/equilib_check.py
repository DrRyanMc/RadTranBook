"""Determine whether phi_eq = ac*T^4 or 2*ac*T^4 in this code."""
import sys
sys.path.insert(0, '.')
import numpy as np
import sn_solver

a = 0.01372; c = 29.98; ac = a * c
T0 = 0.5
sigma = 10.0; rho_cv = 0.01
N = 4; order = 2; I = 1; hx = 0.1
nop1 = order + 1

def run_one_step(phi0_scalar, T0_scalar, dt=1e-2):
    phi = np.full((I, nop1), phi0_scalar)
    T   = np.full((I, nop1), T0_scalar)
    psi = np.broadcast_to(phi[:, None, :] / 2, (I, N, nop1)).copy()
    sigma_func = lambda T: np.full_like(T, sigma)
    scat_func  = lambda T: np.zeros_like(T)
    EOS        = lambda T: rho_cv * T
    invEOS     = lambda e: e / rho_cv
    BCs        = lambda t: np.zeros((N, nop1))
    phis, Ts, _, ts = sn_solver.temp_solve_dmd_inc(
        I=I, hx=hx, q=np.zeros((I, N, nop1)),
        sigma_func=sigma_func, scat_func=scat_func,
        N=N, BCs=BCs, EOS=EOS, invEOS=invEOS,
        phi=phi.copy(), psi=psi.copy(), T=T.copy(),
        dt_min=dt, dt_max=dt, tfinal=dt,
        LOUD=False, order=order, fix=1, K=100, R=3,
        reflect_left=True, reflect_right=True)
    return float(np.mean(phis[-1])), float(np.mean(Ts[-1]))

# Convention 1: phi_eq = ac*T^4 -> perfect equilibrium at (phi=ac*T^4, T=T0)
phi1, T1 = run_one_step(ac * T0**4, T0)
print("C1 phi: %.6e -> %.6e  dT = %.4e" % (ac*T0**4, phi1, T1 - T0))

# Convention 2: phi_eq = 2*ac*T^4 -> perfect equilibrium at (phi=2*ac*T^4, T=T0)
phi2, T2 = run_one_step(2 * ac * T0**4, T0)
print("C2 phi: %.6e -> %.6e  dT = %.4e" % (2*ac*T0**4, phi2, T2 - T0))
