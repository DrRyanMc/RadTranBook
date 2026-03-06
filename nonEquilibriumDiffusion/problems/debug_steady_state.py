"""
Debug why T_mat ≠ T_rad in steady state for single cell with incoming BC.
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

# Physical constants
A_RAD = 0.01372  # GJ/(cm³·keV⁴)
C_LIGHT = 29.9792458  # cm/ns

# Parameters
rho = 1.0  # g/cm³
cv = 0.05  # GJ/(g·keV)
sigma = 100.0  # cm⁻¹
T_bc = 1.0  # keV

# Observed steady state values
T_mat = 1.096455  # keV
E_r = 1.372000e-02  # GJ/cm³
T_rad = (E_r / A_RAD) ** 0.25

print("="*80)
print("STEADY STATE ANALYSIS")
print("="*80)
print(f"\nObserved values:")
print(f"  T_mat = {T_mat:.6f} keV")
print(f"  E_r   = {E_r:.6e} GJ/cm³")
print(f"  T_rad = {T_rad:.6f} keV")
print(f"  T_bc  = {T_bc:.6f} keV")

# What should E_r be if in equilibrium with T_mat?
E_r_equilibrium = A_RAD * T_mat**4
print(f"\nFor equilibrium with T_mat:")
print(f"  Required E_r = a·T_mat⁴ = {E_r_equilibrium:.6e} GJ/cm³")
print(f"  Actual E_r   = {E_r:.6e} GJ/cm³")
print(f"  Discrepancy  = {(E_r - E_r_equilibrium)/E_r_equilibrium * 100:.2f}%")

# Material-radiation coupling term
coupling = sigma * C_LIGHT * (E_r - A_RAD * T_mat**4)
print(f"\nMaterial-radiation coupling:")
print(f"  σ·c·(E_r - a·T_mat⁴) = {coupling:.6e} GJ/(cm³·ns)")
print(f"  dT_mat/dt = coupling/(cv·ρ) = {coupling/(cv * rho):.6e} keV/ns")

if abs(coupling) > 1e-10:
    equilibration_time = abs((T_mat - T_rad) / (coupling / (cv * rho)))
    print(f"  Time to equilibrate ≈ {equilibration_time:.3e} ns")
    print(f"  ⚠ NOT in equilibrium!")
else:
    print(f"  ✓ In equilibrium (coupling ≈ 0)")

# Check the boundary flux
F_total_bc = (A_RAD * C_LIGHT * T_bc**4) / 2.0
print(f"\nIncoming boundary flux:")
print(f"  F_inc = (a·c·T_bc⁴)/2 = {F_total_bc:.6e} GJ/(cm²·ns)")

# For a single cell of width dx, the net power in is F_inc * Area
# In 1D planar, Area = 1, so power in = F_inc per unit area
dx = 0.1  # cm (single cell)
print(f"\nSingle cell (dx = {dx} cm):")
print(f"  Volume = {dx:.3f} cm³ (per unit area)")

# Energy balance: incoming flux - absorbed power
# Power absorbed per volume = σ·c·(E_r - a·T_mat⁴)
# For steady state, net boundary flux should equal net absorption:
# F_in / dx = -σ·c·(E_r - a·T_mat⁴)  [negative because coupling is cooling]
net_absorption_rate = -coupling  # per volume
expected_boundary_flux = net_absorption_rate * dx
print(f"  Net absorption rate = {net_absorption_rate:.6e} GJ/(cm³·ns)")
print(f"  Expected boundary flux = {expected_boundary_flux:.6e} GJ/(cm²·ns)")
print(f"  Actual boundary flux = {F_total_bc:.6e} GJ/(cm²·ns)")
print(f"  Ratio = {expected_boundary_flux / F_total_bc:.3f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
if abs(coupling / (cv * rho)) > 1e-6:
    print("❌ System is NOT in equilibrium!")
    print(f"   Material is {'cooling' if coupling < 0 else 'heating'} at {abs(coupling/(cv*rho)):.3e} keV/ns")
    print(f"   T_mat should be {'decreasing' if coupling < 0 else 'increasing'} toward T_rad")
    print("\nPossible causes:")
    print("  1. Solver hasn't converged to true steady state")
    print("  2. Time step is too large")  
    print("  3. Newton iteration not converging properly")
    print("  4. Boundary condition implementation issue")
else:
    print("✓ System is in equilibrium")
