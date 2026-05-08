"""Theoretical origin regularity: g(0) = I_n(0) for all ordinates.
This means at the origin node (r=0), the angular flux should be isotropic.
For a full sphere LD cell [0, dr], the left node is AT r=0.
So psi_l[0] should equal g_l[0] = phi[0]/2 (for equilibrium).
"""
import sys, numpy as np
sys.path.insert(0, '.')
from sn_solver import ac, c
from numba import njit

project_root = '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook'
for p in [project_root+'/Problems',
          project_root+'/nonEquilibriumDiffusion',
          project_root+'/nonEquilibriumDiffusion/problems']:
    if p not in sys.path:
        sys.path.insert(0, p)
from zeldovich import T_of_r_t

CV_VOL = 3e-6; AC = ac

# IC at origin
r_origin = 0.0
T_origin, _ = T_of_r_t(np.array([r_origin]), 0.01, N=3)
phi_origin = AC * max(T_origin[0], 0.01)**4

print(f"At origin (r=0), t=0.01 ns:")
print(f"  T = {T_origin[0]:.6f} keV")
print(f"  phi = {phi_origin:.6e} GJ/cm^3")
print(f"  For isotropic equilibrium at origin: g = phi/2 = {phi_origin/2:.6e}")
print()

# But the LD mesh places nodes at edges: r_left[0]=0, r_right[0]=dr
# So we get phi_left[0] at r=0 and phi_right[0] at r=dr
r_test = np.array([0.0, 0.075])  # dr_val = 3/40 = 0.075 for I=40
T_test, _ = T_of_r_t(r_test, 0.01, N=3)
phi_test = AC * np.maximum(T_test, 0.01)**4

print(f"With LD mesh (I=40, dr=0.075 cm):")
print(f"  phi_left[0] (at r=0.0): {phi_test[0]:.6e}")
print(f"  phi_right[0] (at r=0.075): {phi_test[1]:.6e}")
print()

# In an optically thick cell in equilibrium, LD gives:
# phi_left ≈ phi_right (linear extrapolation)
# If isotropic at origin: g[0] = phi_left[0]/2 should be conserved.
# But our upwind g formula gives:  g = (Q + g_in/dz) / (sigma_hat + 1/dz)
# At equilibrium Q = sigma*phi, so:
# g = (sigma*phi + g_in/dz) / (sigma + 1/dz)
# For the origin cell (g_in = g[0] itself from origin regularity):
# g[0] = (sigma*phi[0] + g[0]/dz) / (sigma + 1/dz)
# => g[0]*(sigma + 1/dz) = sigma*phi[0] + g[0]/dz
# => g[0]*sigma = sigma*phi[0]
# => g[0] = phi[0]  !!!
print("Upwind g formula at equilibrium for origin cell:")
print("  If g_in = g[0] (regularity), then g[0] = phi[0]")
print("  NOT g[0] = phi[0]/2 (LD result)")
print()
print("This is a CONFLICT:")
print("  - Origin regularity (isotropic at r=0) requires: g[0] = phi[0]/2")
print("  - Upwind formula at equilibrium gives: g[0] = phi[0]")
print("  - Solution: cannot enforce both simultaneously!")
