"""
Quick smoke test for the 2-D S_N solver package.
Runs minimal versions of the Marshak wave (x and y) and verifies basic correctness.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from DiscreteOrdinates2D.src.sn_solver_2d import temp_solve_2d, c, a, ac
from quadratures import (get_2d_quadrature, level_symmetric_quadrature,
                         equal_weight_quadrature, product_quadrature_square,
                         product_quadrature_triangular)

print("="*60)
print("  2-D S_N Solver Smoke Test")
print("="*60)

# 1. Test quadratures
print("\n[1] Quadrature tests...")
for N in [2, 4, 6, 8]:
    om, w = level_symmetric_quadrature(N)
    assert abs(np.sum(w) - 1.0) < 1e-12, f"LS S{N} weights don't sum to 1"
    assert abs(np.sum(w * om[:, 0])) < 1e-12, f"LS S{N} first moment not zero"
print("  Level-symmetric S2-S8: OK")

for N in [4, 6, 8, 10, 12, 14]:
    om, w = equal_weight_quadrature(N)
    assert abs(np.sum(w) - 1.0) < 1e-12
print("  Equal-weight EQ4-EQ14: OK")

for N in [4, 6, 8]:
    om, w = product_quadrature_square(N)
    assert abs(np.sum(w) - 1.0) < 1e-12
    om, w = product_quadrature_triangular(N)
    assert abs(np.sum(w) - 1.0) < 1e-12
print("  Product square/triangular: OK")

# 2. Test sweep in x-direction
print("\n[2] X-direction Marshak wave...")
Ix, Iy = 10, 2
Ox, Oy, W = get_2d_quadrature('level_symmetric', 4)
M = len(Ox)
sigma_val, Cv_val, Tinit = 100.0, 1.0, 1e-3
dx_arr = np.full(Ix, 0.05)
dy_arr = np.full(Iy, 0.05)

def sf(T): return np.full_like(T, sigma_val)
def sc(T): return np.zeros_like(T)
def eos(T): return Cv_val * T
def inv_eos(e): return e / Cv_val
def bc_x(t):
    bc = np.zeros((Iy, M, 2))
    for n in range(M):
        if Ox[n] > 0: bc[:, n, :] = ac
    return {'xlo': bc, 'xhi': None, 'ylo': None, 'yhi': None}

Ti = np.full((Ix, Iy, 4), Tinit)
pi = ac * Ti**4
q = np.zeros((Ix, Iy, 4))
_, Ts_x, _, ts_x, _ = temp_solve_2d(
    Ix, Iy, dx_arr, dy_arr, q, sf, sc, 'level_symmetric', 4,
    bc_x, eos, inv_eos, pi, Ti,
    dt_min=1e-4, dt_max=1e-3, tfinal=0.005,
    tolerance=1e-5, Linf_tol=1e-3, maxits=30, K=20, R=3,
    reflect_xhi=True, reflect_ylo=True, reflect_yhi=True, use_dmd=False)
T_x = np.mean(Ts_x[-1], axis=(1, 2))
assert T_x[0] > T_x[-1], "X-wave not propagating!"
print(f"  X-prop: T_max={T_x[0]:.5f}, T_min={T_x[-1]:.5f}, steps={len(ts_x)-1}")

# 3. Test sweep in y-direction
print("\n[3] Y-direction Marshak wave...")
Ix2, Iy2 = 2, 10
dx2, dy2 = np.full(Ix2, 0.05), np.full(Iy2, 0.05)

def bc_y(t):
    bc = np.zeros((Ix2, M, 2))
    for n in range(M):
        if Oy[n] > 0: bc[:, n, :] = ac
    return {'xlo': None, 'xhi': None, 'ylo': bc, 'yhi': None}

Ti2 = np.full((Ix2, Iy2, 4), Tinit)
pi2 = ac * Ti2**4
q2 = np.zeros((Ix2, Iy2, 4))
_, Ts_y, _, ts_y, _ = temp_solve_2d(
    Ix2, Iy2, dx2, dy2, q2, sf, sc, 'level_symmetric', 4,
    bc_y, eos, inv_eos, pi2, Ti2,
    dt_min=1e-4, dt_max=1e-3, tfinal=0.005,
    tolerance=1e-5, Linf_tol=1e-3, maxits=30, K=20, R=3,
    reflect_xlo=True, reflect_xhi=True, reflect_yhi=True, use_dmd=False)
T_y = np.mean(Ts_y[-1], axis=(0, 2))
assert T_y[0] > T_y[-1], "Y-wave not propagating!"
print(f"  Y-prop: T_max={T_y[0]:.5f}, T_min={T_y[-1]:.5f}, steps={len(ts_y)-1}")

# 4. Verify x-y agreement
diff = abs(T_x[0] - T_y[0])
print(f"\n[4] X-Y agreement: |T_x_max - T_y_max| = {diff:.2e}")
assert diff < 1e-4, "X and Y directions give different results!"

# 5. Test Zeldovich self-similar
print("\n[5] Zeldovich self-similar solution...")
from problems.zeldovich_wave_2d import zeldovich_self_similar
r = np.linspace(0.01, 1.5, 50)
T_z, R = zeldovich_self_similar(r, 1.0, N=2)
assert R > 0.5 and R < 2.0, f"Unexpected front radius: {R}"
assert T_z.max() > 0.01, "Temperature too low"
print(f"  t=1.0: R_front={R:.4f} cm, T_max={T_z.max():.4f} keV")

print("\n" + "="*60)
print("  ALL TESTS PASSED!")
print("="*60)
