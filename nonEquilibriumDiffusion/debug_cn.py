"""Debug Crank-Nicolson implementation by comparing step-by-step with reference"""
import numpy as np
import sys
sys.path.insert(0, '../Problems')
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Physical constants from equilibrationTest.py
C_v = 0.01  # GJ/keV/cm^3
c = 2.99792458e1  # cm/ns
sigma = 5.0  # cm^-1 (note: reference uses 5.0, but test uses 10.0)
a_rad = 0.01372  # GJ/cm^3/keV^4
T_m_initial = 0.4  # keV
T_r_initial = 1.0  # keV
Delta_t = 0.01  # ns

# Reference implementation from equilibrationTest.py
def CN_update_ref(Tstar, Tn, Ern, max_iters=20, dt=Delta_t):
    iteration_count = 0
    converged = False
    while (iteration_count < max_iters) and not(converged):
        beta = 4*a_rad*Tstar**3/C_v
        f = 1/(1 + 0.5*beta*dt*c*sigma)
        Er_new =(Ern + f*sigma*dt*c*(a_rad*Tstar**4 - 0.5*Ern) - (1-f)*(C_v*Tstar-C_v*Tn))/(1+f*dt*c*sigma*0.5)
        T_new = (C_v*Tn+f*c*sigma*dt*(0.5*(Er_new + Ern) - a_rad*Tstar**4) + (1-f)*(C_v*Tstar-C_v*Tn))/(C_v)
        # Check for convergence
        if np.abs(T_new - Tstar) < 1e-6:
            converged = True
        Tstar = T_new
        iteration_count += 1
    return Er_new, T_new

# Test reference implementation
print("="*70)
print("Reference CN Implementation")
print("="*70)
print(f"Initial: T_mat = {T_m_initial} keV, T_rad = {T_r_initial} keV")
print(f"         E_r = {a_rad*T_r_initial**4:.6e} GJ/cm³")
print(f"         E_total = {C_v*T_m_initial + a_rad*T_r_initial**4:.6e} GJ/cm³")

Er_new, T_new = CN_update_ref(T_m_initial, T_m_initial, a_rad*T_r_initial**4)
T_rad_new = (Er_new/a_rad)**0.25

print(f"\nAfter 1 CN step (dt = {Delta_t} ns):")
print(f"  T_mat = {T_new:.6f} keV")
print(f"  T_rad = {T_rad_new:.6f} keV")
print(f"  E_r = {Er_new:.6e} GJ/cm³")
print(f"  E_total = {C_v*T_new + Er_new:.6e} GJ/cm³")
print(f"  ΔE/E_init = {(C_v*T_new + Er_new - (C_v*T_m_initial + a_rad*T_r_initial**4))/(C_v*T_m_initial + a_rad*T_r_initial**4):.6e}")

# Now test my implementation BUT with sigma=5.0 to match reference
print("\n" + "="*70)
print("My CN Implementation (σ = 5.0 to match reference)")
print("="*70)

def specific_heat(T):
    return C_v / 1.0  # ρ = 1.0

def material_energy(T):
    return C_v * T

def planck_opacity(T):
    return sigma

def rosseland_opacity(T):
    return sigma

def reflecting_bc_left(phi, x):
    return 0.0, 1.0, 0.0

def reflecting_bc_right(phi, x):
    return 0.0, 1.0, 0.0

solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=0.0,
    r_max=1.0,
    n_cells=1,  # Just 1 cell for 0-D problem
    d=0,
    dt=Delta_t,
    max_newton_iter=50,
    newton_tol=1e-8,
    rosseland_opacity_func=rosseland_opacity,
    planck_opacity_func=planck_opacity,
    specific_heat_func=specific_heat,
    material_energy_func=material_energy,
    left_bc_func=reflecting_bc_left,
    right_bc_func=reflecting_bc_right,
    theta=0.5  # Crank-Nicolson
)

phi_init = a_rad * c * T_r_initial**4  # φ = c*E_r
solver.set_initial_condition(phi_init=phi_init, T_init=T_m_initial)

print(f"Initial: T_mat = {solver.T[0]} keV, T_rad = {T_r_initial} keV")
print(f"         φ = {solver.phi[0]:.6e} GJ/cm²")
print(f"         E_r = {solver.phi[0]/c:.6e} GJ/cm³")
print(f"         E_total = {C_v*solver.T[0] + solver.phi[0]/c:.6e} GJ/cm³")

solver.time_step(n_steps=1, verbose=True)

T_mat_new = solver.T[0]
phi_new = solver.phi[0]
Er_new_mine = phi_new / c
T_rad_new_mine = (Er_new_mine/a_rad)**0.25

print(f"\nAfter 1 CN step (dt = {Delta_t} ns):")
print(f"  T_mat = {T_mat_new:.6f} keV")
print(f"  T_rad = {T_rad_new_mine:.6f} keV")
print(f"  φ = {phi_new:.6e} GJ/cm²")
print(f"  E_r = {Er_new_mine:.6e} GJ/cm³")
print(f"  E_total = {C_v*T_mat_new + Er_new_mine:.6e} GJ/cm³")
print(f"  ΔE/E_init = {(C_v*T_mat_new + Er_new_mine - (C_v*T_m_initial + a_rad*T_r_initial**4))/(C_v*T_m_initial + a_rad*T_r_initial**4):.6e}")

print("\n" + "="*70)
print("Comparison")
print("="*70)
print(f"ΔT_mat = {T_mat_new - T_new:.6e} keV")
print(f"ΔT_rad = {T_rad_new_mine - T_rad_new:.6e} keV")
print(f"ΔE_r = {Er_new_mine - Er_new:.6e} GJ/cm³")
