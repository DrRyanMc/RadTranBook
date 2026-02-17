"""
Test script to diagnose Marshak boundary condition issue
"""
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import RadiationDiffusionSolver, A_RAD, C_LIGHT, RHO

# Material functions from marshak_wave.py
def marshak_opacity(T):
    sigma_coeff = 300.0  # cm^-1
    return sigma_coeff * T**(-3)

def marshak_specific_heat(T):
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric / RHO

def marshak_material_energy(T):
    cv_volumetric = 0.3  # GJ/(cm^3·keV)
    return cv_volumetric * T

def marshak_left_bc(Er, x):
    T_bc = 1.0  # keV
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def marshak_right_bc(Er, x):
    return 0.0, 1.0, 0.0

# Create solver
solver = RadiationDiffusionSolver(
    r_min=0.0, 
    r_max=0.1,  # Very small domain
    n_cells=20, 
    d=0,
    dt=0.001,
    max_newton_iter=20,
    newton_tol=1e-8,
    rosseland_opacity_func=marshak_opacity,
    specific_heat_func=marshak_specific_heat,
    material_energy_func=marshak_material_energy,
    left_bc_func=marshak_left_bc,
    right_bc_func=marshak_right_bc
)

# Enable nonlinear corrections
solver.use_nonlinear_correction = True
solver.max_newton_iter_per_step = 50

# Initial condition
T_init = 0.01  # keV
solver.set_initial_condition(lambda r: np.full_like(r, A_RAD * T_init**4))

# Expected boundary value
T_bc = 1.0
Er_bc_expected = A_RAD * T_bc**4

print("="*70)
print("MARSHAK BOUNDARY CONDITION DIAGNOSTIC")
print("="*70)
print(f"\nBoundary condition:")
print(f"  T_bc = {T_bc} keV")
print(f"  Er_bc (expected) = a*T_bc^4 = {Er_bc_expected:.6e} GJ/cm^3")
print(f"  a = {A_RAD} GJ/(cm^3·keV^4)")
print(f"\nInitial condition:")
print(f"  T_init = {T_init} keV")
print(f"  Er_init = {solver.Er[0]:.6e} GJ/cm^3")

# Take a few time steps
print(f"\nTaking time steps with dt = {solver.dt} ns...")
for step in range(10):
    solver.time_step(n_steps=1, verbose=False)
    T = (solver.Er / A_RAD)**0.25
    print(f"  Step {step+1}: Er[0] = {solver.Er[0]:.6e}, T[0] = {T[0]:.6e} keV")

print(f"\nFinal state:")
print(f"  Er[0] = {solver.Er[0]:.6e} GJ/cm^3")
print(f"  T[0] = {T[0]:.6e} keV")
print(f"  Er_bc_expected = {Er_bc_expected:.6e} GJ/cm^3")
print(f"  Ratio Er[0]/Er_bc = {solver.Er[0]/Er_bc_expected:.6e}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

T_final = (solver.Er / A_RAD)**0.25
ax1.plot(solver.r, solver.Er, 'b-', linewidth=2, label='E_r')
ax1.axhline(Er_bc_expected, color='r', linestyle='--', label='E_r at BC (expected)')
ax1.set_xlabel('Position (cm)')
ax1.set_ylabel('Radiation Energy Density (GJ/cm³)')
ax1.set_title('Radiation Energy Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(solver.r, T_final, 'b-', linewidth=2, label='T')
ax2.axhline(T_bc, color='r', linestyle='--', label='T at BC (expected)')
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('Temperature (keV)')
ax2.set_title('Temperature Profile')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_marshak_bc.png', dpi=150)
print("\nPlot saved as 'test_marshak_bc.png'")
plt.close()

print("="*70)
