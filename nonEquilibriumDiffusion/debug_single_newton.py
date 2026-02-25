"""
Detailed debug of single Newton iteration to find source of discrepancy
"""
import numpy as np
import sys
sys.path.insert(0, '../Problems')
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372  # GJ/cm³/keV⁴
RHO = 1.0  # g/cm³

# Test parameters
C_v = 0.01  # GJ/cm³/keV
sigma_P = 10.0  # cm⁻¹
T_n = 0.4  # keV
T_rad_n = 1.0  # keV
phi_n = A_RAD * C_LIGHT * T_rad_n**4  # GJ/cm²
theta = 1.0  # Backward Euler
dt = 0.001  # ns

print("="*70)
print("DETAILED DEBUG OF SINGLE NEWTON ITERATION")
print("="*70)
print(f"\nθ = {theta}, Δt = {dt} ns")
print(f"T^n = {T_n} keV, φ^n = {phi_n:.6e} GJ/cm²")

# Step 1: Compute analytical solution
T_star = T_n
e_n = C_v * T_n
e_star = C_v * T_star
Delta_e = e_star - e_n  # Should be 0

beta = (4.0 * A_RAD * T_star**3) / C_v
f = 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)
acT4_star = A_RAD * C_LIGHT * T_star**4

print(f"\nLinearization parameters:")
print(f"  β = {beta:.10f}")
print(f"  f = {f:.10f}")
print(f"  acT★⁴ = {acT4_star:.10e}")
print(f"  Δe = {Delta_e:.10e}")

# φ equation coefficients
diag_coeff = 1.0 / (C_LIGHT * dt) + sigma_P * f * theta
rhs_phi = phi_n / (C_LIGHT * dt) + sigma_P * f * acT4_star - sigma_P * f * (1.0 - theta) * phi_n

print(f"\nφ equation:")
print(f"  Diagonal coefficient = {diag_coeff:.10e}")
print(f"  RHS = {rhs_phi:.10e}")

phi_np1_analytical = rhs_phi / diag_coeff
print(f"  φ^{{n+1}} (analytical) = {phi_np1_analytical:.10e}")

# T equation
phi_tilde = theta * phi_np1_analytical + (1.0 - theta) * phi_n
e_np1 = e_n + dt * f * sigma_P * (phi_tilde - acT4_star)
T_np1_analytical = e_np1 / C_v

print(f"\nT equation:")
print(f"  φ̃ = {phi_tilde:.10e}")
print(f"  e^{{n+1}} = {e_np1:.10e}")
print(f"  T^{{n+1}} (analytical) = {T_np1_analytical:.10f}")

# Step 2: Run solver with detailed output
print("\n" + "="*70)
print("SOLVER WITH DETAILED OUTPUT")
print("="*70)

def specific_heat(T):
    return C_v / RHO

def material_energy(T):
    return C_v * T

def planck_opacity(T):
    return sigma_P

def rosseland_opacity(T):
    return sigma_P

def reflecting_bc_left(phi, x):
    return 0.0, 1.0, 0.0

def reflecting_bc_right(phi, x):
    return 0.0, 1.0, 0.0

solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=0.0,
    r_max=1.0,
    n_cells=1,
    d=0,
    dt=dt,
    max_newton_iter=1,
    newton_tol=1e-20,
    rosseland_opacity_func=rosseland_opacity,
    planck_opacity_func=planck_opacity,
    specific_heat_func=specific_heat,
    material_energy_func=material_energy,
    left_bc_func=reflecting_bc_left,
    right_bc_func=reflecting_bc_right,
    theta=theta
)

solver.set_initial_condition(phi_init=phi_n, T_init=T_n)

# Manually call the Newton step to inspect internals
print(f"\nCalling newton_step with:")
print(f"  phi_prev = {solver.phi_old[0]:.10e}")
print(f"  T_prev = {solver.T_old[0]:.10f}")

# Call assemble_phi_equation directly to see what it produces
A_phi, rhs_phi_solver = solver.assemble_phi_equation(
    solver.phi_old, solver.T_old, solver.phi_old, solver.T_old, theta=theta)

print(f"\nBefore boundary conditions:")
print(f"  A_phi['diag'][0] = {A_phi['diag'][0]:.10e}")
print(f"  rhs_phi[0] = {rhs_phi_solver[0]:.10e}")

# Apply boundary conditions
solver.apply_boundary_conditions_phi(A_phi, rhs_phi_solver, solver.phi_old)

print(f"\nAfter boundary conditions:")
print(f"  A_phi['diag'][0] = {A_phi['diag'][0]:.10e}")
print(f"  rhs_phi[0] = {rhs_phi_solver[0]:.10e}")

# Solve (manually for single cell)
from oneDFV import solve_tridiagonal
phi_np1_solver = solve_tridiagonal(A_phi, rhs_phi_solver)[0]

print(f"\nφ^{{n+1}} (solver) = {phi_np1_solver:.10e}")

# Now solve T equation
T_np1_array = solver.solve_T_equation(
    np.array([phi_np1_solver]), solver.T_old, solver.phi_old, solver.T_old, theta=theta)
T_np1_solver = T_np1_array[0]

print(f"T^{{n+1}} (solver) = {T_np1_solver:.10f}")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Δφ = {abs(phi_np1_solver - phi_np1_analytical):.10e}")
print(f"ΔT = {abs(T_np1_solver - T_np1_analytical):.10e}")
print(f"\nφ relative error = {abs(phi_np1_solver - phi_np1_analytical)/abs(phi_np1_analytical):.10e}")
print(f"T relative error = {abs(T_np1_solver - T_np1_analytical)/abs(T_np1_analytical):.10e}")

# Check if boundary conditions changed things
if abs(A_phi['diag'][0] - diag_coeff) > 1e-15:
    print(f"\n⚠ Boundary conditions modified diagonal coefficient!")
    print(f"  Expected: {diag_coeff:.10e}")
    print(f"  Got:      {A_phi['diag'][0]:.10e}")
    print(f"  Difference: {abs(A_phi['diag'][0] - diag_coeff):.10e}")

if abs(rhs_phi_solver[0] - rhs_phi) > 1e-15:
    print(f"\n⚠ Boundary conditions modified RHS!")
    print(f"  Expected: {rhs_phi:.10e}")
    print(f"  Got:      {rhs_phi_solver[0]:.10e}")
    print(f"  Difference: {abs(rhs_phi_solver[0] - rhs_phi):.10e}")
