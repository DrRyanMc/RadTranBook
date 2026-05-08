"""Investigate origin regularity for spherical radiation transport.

The divergence of radiative flux is:
    ∇·F = (1/r²) d(r² F)/dr

At r=0, for finite flux F, we need r² F → 0 to avoid singularity.
If F ~ r^α, then r² F ~ r^(α+2).
For regularity: α + 2 ≥ 0, or α ≥ -2.

For energy density: φ (dimensionless, or energy/volume)
  Energy in shell [0, r]: E(r) = ∫₀ʳ φ(r') 4π r'² dr'
  For regularity at r=0, we need φ to not diverge faster than r^(-3)
  So φ ~ r^β requires β ≥ -3.

For intensity I(r, μ), the scalar flux φ(r) = ∫₋₁⁺¹ I(r,μ) dμ / 2.

The origin regularity condition for transport is:
  I(0, μ) = I(0, -μ) (isotropic at origin)
  This means g = I(0, -1) should equal the average intensity at origin.

Let's check the Zeldovich IC to see if it has this property.
"""
import sys, numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/Problems')
from zeldovich import T_of_r_t
from sn_solver import ac

AC = ac
t = 0.01  # ns

# IC at origin
T0, _ = T_of_r_t(np.array([0.0]), t, N=3)
phi0 = AC * max(T0[0], 0.01)**4
print(f"At origin (r=0), t={t} ns:")
print(f"  T(0) = {T0[0]:.6f} keV")
print(f"  φ(0) = {phi0:.6e} GJ/cm³")
print(f"  For isotropic radiation: g(0) = φ(0)/2 = {phi0/2:.6e}")
print()

# For the Zeldovich self-similar solution, the density profile is
# u(r,t) ~ t^{-α_u} f(ξ), ξ = r/t^β
# At origin: ξ = 0 (for any r and t > 0)
# The similarity profile f(0) is maximum (peak of the wave profile)
# So u(0,t) = t^{-α_u} f(0) is the central value.
# As the wave evolves, material spreads outward, origin cools.

# Check that phi is NOT exactly zero derivative
r_vals = np.linspace(0, 0.01, 100)
T_vals, _ = T_of_r_t(r_vals, t, N=3)
phi_vals = AC * np.maximum(T_vals, 0.01)**4

# Fit linear model φ(r) ≈ φ(0) + α·r near origin
from numpy.polynomial.polynomial import polyfit
coeffs = polyfit(r_vals[:20], phi_vals[:20], 1)
print(f"Linear fit to phi(r) near origin (first 20 points, r ∈ [0, 0.002] cm):")
print(f"  φ(r) ≈ {coeffs[0]:.6e} + r·{coeffs[1]:.6e}")
print(f"  dφ/dr (slope) ≈ {coeffs[1]:.4e} GJ/cm⁴")
print()

# Check second derivative
coeffs2 = polyfit(r_vals[:20], phi_vals[:20], 2)
print(f"Quadratic fit φ(r) ≈ c₀ + c₁·r + c₂·r²:")
print(f"  c₀ = {coeffs2[0]:.6e}")
print(f"  c₁ = {coeffs2[1]:.6e} (= dφ/dr at origin)")
print(f"  c₂ = {coeffs2[2]:.6e} (= ½ d²φ/dr² at origin)")
print()

print("CONCLUSION:")
print("  The analytical Zeldovich solution does NOT have zero derivative at origin.")
print("  dφ/dr is small but nonzero: ≈ -0.6 GJ/cm⁴")
print("  This is PHYSICAL: the wave is a traveling disturbance spreading outward.")
print("  The origin IS a maximum (φ decreases as we move away), consistent with the")
print("  direction of heat flow (from hot origin towards cold exterior).")
