"""Check analytical T and phi derivatives near origin."""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/Problems')
from zeldovich import T_of_r_t
from sn_solver import ac

AC = ac
t = 0.01  # ns

# Evaluate T at fine resolution near origin
r_fine = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
T_fine, _ = T_of_r_t(r_fine, t, N=3)
phi_fine = AC * np.maximum(T_fine, 0.01)**4

print(f"At t = {t} ns, near origin:")
print(f"{'r (cm)':>10} {'T (keV)':>12} {'phi (GJ/cm³)':>15} {'dT/dr':>12} {'dphi/dr':>12}")
for i in range(len(r_fine)):
    if i == 0:
        dT_dr = "  —  "
        dphi_dr = "  —  "
    else:
        dT_dr = f"{(T_fine[i]-T_fine[i-1])/(r_fine[i]-r_fine[i-1]):12.4e}"
        dphi_dr = f"{(phi_fine[i]-phi_fine[i-1])/(r_fine[i]-r_fine[i-1]):12.4e}"
    print(f"{r_fine[i]:10.5f} {T_fine[i]:12.6f} {phi_fine[i]:15.6e} {dT_dr:>12} {dphi_dr:>12}")
print()

# Check second derivative of T at origin (regularity condition)
r_tiny = np.array([0.0, 0.0001, 0.0002])
T_tiny, _ = T_of_r_t(r_tiny, t, N=3)
print(f"Second-order finite diff of T at origin:")
print(f"  T(0) = {T_tiny[0]:.6f}")
print(f"  T(0.0001) = {T_tiny[1]:.6f}")
print(f"  T(0.0002) = {T_tiny[2]:.6f}")
if abs(r_tiny[1] - r_tiny[0]) > 1e-10:
    d2T_dr2 = (T_tiny[2] - 2*T_tiny[1] + T_tiny[0]) / ((r_tiny[1]-r_tiny[0])**2)
    print(f"  d²T/dr² ≈ {d2T_dr2:.4e}")
print()
print("KEY QUESTION: Is T truly flat (dT/dr=0) at origin?")
print(f"  From analytical formula: dT/dr at origin = {(T_fine[1]-T_fine[0])/(r_fine[1]-r_fine[0]):.4e}")
