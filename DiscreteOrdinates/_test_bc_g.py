"""Quick test: print bc_g_outer_use to verify the fix is applied."""
import sys, numpy as np
sys.path.insert(0, '.')

# Monkey-patch single_sweep_phi_sph_ld to print bc_g_outer
import sn_solver_ld_sphere as _mod
_orig_sweep = _mod.single_sweep_phi_sph_ld

_call_count = [0]
def _patched_sweep(*args, **kwargs):
    _call_count[0] += 1
    if _call_count[0] <= 3:
        bc_g = args[8]  # bc_g_outer is the 9th positional arg
        print(f"  [call {_call_count[0]}] bc_g_outer_use = {bc_g:.4e}")
    return _orig_sweep(*args, **kwargs)

_mod.single_sweep_phi_sph_ld = _patched_sweep

from sn_solver_ld_sphere import temp_solve_sph_ld
from sn_solver import ac, c
from numba import njit

I, N = 5, 4
R_MAX = 1.0; CV_VOL = 3e-6; AC = ac
dr_val = R_MAX / I
r_left  = np.arange(I, dtype=np.float64) * dr_val
dr      = np.full(I, dr_val, dtype=np.float64)

@njit
def sigma_func(T): return 300.0 * np.maximum(T, 1e-3)**(-3)
@njit
def scat_func(T): return np.zeros_like(T)
@njit
def eos(T): return CV_VOL * np.maximum(T, 0.0)
@njit
def invEOS(e): return np.maximum(e, 0.0) / CV_VOL

T_ic   = np.full((I, 2), 1.0)
phi    = AC * T_ic**4
psi    = np.broadcast_to(phi[:,None,:], (I,N,2)).copy()
g_init = phi.copy()
q_n    = np.zeros((I, N, 2))
q_g    = np.zeros((I, 2))
bc0    = np.zeros((N, 2))
def BCs(t): return bc0, 0.0

print(f"phi = {phi[0,0]:.4e}  (equilibrium for T=1 keV)")
print(f"Expected bc_g_outer_use for reflecting: {phi[0,0]:.4e}")
print()

phis, Ts, gs, its, ts, ips = temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N,
    BCs, eos, invEOS, phi, psi, T_ic, g_init,
    dt_min=1e-4, dt_max=1e-4, tfinal=1e-4,
    maxits=50, K=10, R=3, LOUD=False,
    reflect_outer=True, print_stride=0,
)
print(f"\nAfter 1 step: T_max = {Ts[-1].max():.4f} keV")
