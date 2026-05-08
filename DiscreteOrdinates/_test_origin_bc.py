# -*- coding: utf-8 -*-
"""Sanity checks for the spherical LDG origin/source convention.

Check A: constant solution preservation.
  Use a thick-cell isotropic source so the exact solution is I = I0 for all
  r and mu.  Verify phi = 2*I0 everywhere (physical scalar intensity).

Check B: source normalization at equilibrium.
  Uniform T=1 keV sphere.  After 3 steps (reflecting wall, exact equilibrium
  IC), phi should remain ac*T^4 and T should remain 1 keV.

Check C: origin regularity.
  After equilibrium solve, g(0) should be phi(0)/2 (isotropic: I = phi/2).
"""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/DiscreteOrdinates')
from sn_solver_ld_sphere import single_sweep_psi_sph_ld, temp_solve_sph_ld, _get_quadrature
from sn_solver import ac

I = 20
R = 3.0
dr = np.full(I, R / I)
r_left = np.arange(I, dtype=np.float64) * (R / I)

# -----------------------------------------------------------------------
# Check A: constant solution  I = I0  =>  phi = 2*I0
# Thick cells (sigma*dz >> 1) so streaming is negligible; source = sigma*I0
# -----------------------------------------------------------------------
print("=" * 60)
print("Check A: constant solution (phi = 2*I0)")
print("=" * 60)
sigma_val = 100.0         # thick
I0 = 0.01372
sigma_hat = np.full((I, 2), sigma_val)
for N in [4, 8]:
    src_n = np.full((I, N, 2), sigma_val * I0)   # Q_n = sigma * I0
    src_g = np.full((I, 2), sigma_val * I0)
    bc = np.zeros((N, 2))
    for n_idx in range(N):
        MU, _ = _get_quadrature(N)
        if MU[n_idx] < 0:
            bc[n_idx, 0] = I0            # outer inflow for mu < 0
    psi, phi_out, g = single_sweep_psi_sph_ld(
        I, r_left, dr, src_n, src_g, sigma_hat, N, bc, I0, fix=1
    )
    phi_mean = 0.5 * (phi_out[:, 0] + phi_out[:, 1])
    rel_err = np.abs(phi_mean - 2.0 * I0) / (2.0 * I0)
    ok = np.max(rel_err) < 0.02       # 2% relative error (thick-cell approx)
    print(f"  N={N}: phi_mean in [{phi_mean.min():.4e}, {phi_mean.max():.4e}]  "
          f"2*I0={2*I0:.4e}  max_rel_err={np.max(rel_err):.2e}  "
          f"{'PASS' if ok else '*** FAIL ***'}")

# -----------------------------------------------------------------------
# Check B: equilibrium maintained (phi = ac*T^4, T stays at 1 keV)
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Check B: equilibrium stability (T stays at 1 keV)")
print("=" * 60)
N = 8
MU, W_quad = _get_quadrature(N)
sigma_equil = 10.0
T_ic = np.full((I, 2), 1.0)
cv = 3e-6
phi_ic = ac * T_ic**4                                         # phi = ac*T^4 at equil.
psi_ic = np.broadcast_to((phi_ic / 2)[:, None, :], (I, N, 2)).copy()  # I_n = phi/2
g_ic   = phi_ic.copy() / 2                                   # g = phi/2 at equil.

def EOS(T):
    return np.maximum(T, 1e-10) * cv
def invEOS(e):
    return np.maximum(e / cv, 1e-10)
def sigma_func(T):
    return np.full_like(T, sigma_equil)
def BCs_func(t):
    return np.zeros((N, 2)), 0.0

phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr,
    np.zeros((I, N, 2)), np.zeros((I, 2)),
    sigma_func, lambda T: 0.0, N,
    BCs_func, EOS, invEOS,
    phi_ic, psi_ic, T_ic, g_ic,
    dt_min=1e-3, dt_max=1e-3, tfinal=3e-3, maxits=200,
    K=50, LOUD=False, reflect_outer=True,
    time_outputs=np.array([0.0, 1e-3, 2e-3, 3e-3]),
)

Tf = Ts[-1]
gf = gs[-1]
pf = phis[-1]

T_max_err = abs(Tf.max() - 1.0)
phi_eq = ac * Tf**4
phi_err = np.max(np.abs(pf - phi_eq)) / np.max(phi_eq)
ok_B1 = T_max_err < 0.01
ok_B2 = phi_err < 0.01
print(f"  T_max = {Tf.max():.6f} keV  (should be ~1.0)")
print(f"  T stable --> {'PASS' if ok_B1 else f'*** FAIL *** err={T_max_err:.2e}'}")
print(f"  phi vs ac*T^4: max_rel_err={phi_err:.2e}  "
      f"{'PASS' if ok_B2 else '*** FAIL ***'}")

# -----------------------------------------------------------------------
# Check C: origin regularity — g(0) = phi(0)/2 (isotropy)
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Check C: origin regularity  g(0) ~ phi(0)/2")
print("=" * 60)
g0 = gf[0, 0]
phi0_val = pf[0, 0]
expected = phi0_val / 2.0
rel_C = abs(g0 - expected) / max(abs(expected), 1e-30)
ok_C = rel_C < 0.01
print(f"  g(0)      = {g0:.6e}")
print(f"  phi(0)/2  = {expected:.6e}")
print(f"  rel err   = {rel_C:.2e}")
print(f"  --> {'PASS' if ok_C else '*** FAIL ***'}")

print()
overall = ok_B1 and ok_B2 and ok_C
print("=" * 60)
print(f"Overall: {'ALL PASS' if overall else '*** SOME CHECKS FAILED ***'}")
print("=" * 60)
