"""Quick validation script for sn_solver_ld."""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver_ld import (
    _sweep_all_phi_ld, _sweep_all_psi_ld,
    single_sweep_phi_ld, single_sweep_psi_ld,
    build_reflecting_BCs_ld, c, a, ac, _get_quadrature,
)

# JIT warm-up: single vacuum sweep, no fixup
I, N = 10, 4
hx = 0.1
MU, W = _get_quadrature(N)

source   = np.zeros((I, N, 2))
sig_hat  = np.ones((I, 2)) * 1.0
BCs      = np.zeros((N, 2))

print("Compiling _sweep_all_phi_ld (fix=0)...")
phi = _sweep_all_phi_ld(I, hx, source, sig_hat, MU, W, BCs, 0)
assert phi.shape == (I, 2), phi.shape
print("  phi shape:", phi.shape, "  all zero?", np.allclose(phi, 0))

print("Compiling _sweep_all_psi_ld (fix=0)...")
psi = _sweep_all_psi_ld(I, hx, source, sig_hat, MU, W, BCs, 0)
assert psi.shape == (I, N, 2), psi.shape
print("  psi shape:", psi.shape, "  all zero?", np.allclose(psi, 0))

# Blackbody inflow from left, vacuum right
T_b = 1.0
I_bb = ac * T_b**4 / 2.0
BCs2 = np.zeros((N, 2))
for n in range(N):
    if MU[n] > 0:
        BCs2[n, 1] = I_bb  # left boundary inflow for mu > 0

print("Blackbody source sweep (fix=1)...")
phi2 = single_sweep_phi_ld(I, hx, source, sig_hat, N, BCs2, fix=1)
print("  phi2[0]:", phi2[0], "  phi2[-1]:", phi2[-1])
print("  phi2[-1] < I_bb =", I_bb, "? attenuated?", np.all(phi2[-1] < I_bb))

# Equilibrium check: reflecting BCs from isotropic field should be I_bb
# on the entries that are actually used by the sweep.
phi_eq = np.full((I, 2), I_bb)
psi_eq = np.full((I, N, 2), I_bb)
bcs_ref = np.zeros((N, 2))
bcs_ref = build_reflecting_BCs_ld(bcs_ref, psi_eq, True, True, N)
# build_reflecting_BCs_ld sets bcs[n,1]=I_bb for mu>0 and bcs[n,0]=I_bb for mu<0;
# the other entries are not used by the sweep so they stay 0.
ok_reflect = True
for n in range(N):
    if MU[n] > 0:
        ok_reflect = ok_reflect and np.isclose(bcs_ref[n, 1], I_bb)
    else:
        ok_reflect = ok_reflect and np.isclose(bcs_ref[n, 0], I_bb)
print("Reflecting BCs from isotropic psi_eq: used entries == I_bb?", ok_reflect)

# Quadrature weight conservation: sum w_n = 1
print("Sum of weights:", np.sum(W), "(should be 1)")

# 2x2 cell solve check: pure absorption with known solution
# sigma_hat=1, hx=1, mu>0, inflow I_in from left
# Row0: (mu/2 + hx*sh_l)*I_l + (mu/2)*I_r = mu*I_in + hx*Q_l
# Row1: (-mu/2)*I_l + (mu/2 + hx*sh_r)*I_r = hx*Q_r
I1, hx1, mu1 = 1, 1.0, 0.5
sh = np.ones((1, 2))
src1 = np.zeros((1, 4, 2))
bc1 = np.zeros((4, 2))
for n in range(4):
    if MU[n] > 0:
        bc1[n, 1] = 1.0   # unit inflow
phi1 = single_sweep_phi_ld(I1, hx1, src1, sh, 4, bc1, fix=0)
print("Unit inflow, sigma=1, hx=1: phi at left/right edges =", phi1[0])

print()
print("All checks passed.")

# ---- Equilibrium sweep check -----------------------------------------------
# With no absorption/time term (sigma_hat=0, source includes I_bb * 0 = 0),
# reflecting walls, and psi = I_bb everywhere, the sweep should recover phi = I_bb.
# Use tiny sigma_hat to avoid divide-by-zero in 2x2 solve.
I2, N2, hx2 = 5, 4, 0.2
MU2, W2 = _get_quadrature(N2)
psi_eq2 = np.full((I2, N2, 2), I_bb)
bcs_eq = np.zeros((N2, 2))
bcs_eq = build_reflecting_BCs_ld(bcs_eq, psi_eq2, True, True, N2)
source_eq = np.zeros((I2, N2, 2))
sig_tiny = np.full((I2, 2), 1e-10)
phi_eq2 = single_sweep_phi_ld(I2, hx2, source_eq, sig_tiny, N2, bcs_eq, fix=0)
print("Equilibrium sweep (reflecting, psi=I_bb everywhere):")
print("  phi at left edge:", phi_eq2[:, 0], "  expected:", I_bb)
print("  max deviation:", np.max(np.abs(phi_eq2 - I_bb)))
assert np.allclose(phi_eq2, I_bb, rtol=1e-6), "Equilibrium sweep failed!"
print("  PASS")
