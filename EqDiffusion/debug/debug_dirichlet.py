"""
Debug Dirichlet BC implementation
"""
import numpy as np
from oneDFV import RadiationDiffusionSolver, A_RAD, C_LIGHT, RHO

# Material functions
def marshak_opacity(T):
    return 300.0 * T**(-3)

def marshak_specific_heat(T):
    return 0.3 / RHO

def marshak_material_energy(T):
    return 0.3 * T

def marshak_left_bc(Er, x):
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def marshak_right_bc(Er, x):
    return 0.0, 1.0, 0.0

# Create very simple case
solver = RadiationDiffusionSolver(
    r_min=0.0, 
    r_max=0.01,  # 0.01 cm = 100 microns
    n_cells=5,    # Just 5 cells
    d=0,
    dt=0.001,
    max_newton_iter=50,  # Allow convergence
    newton_tol=1e-8,
    rosseland_opacity_func=marshak_opacity,
    specific_heat_func=marshak_specific_heat,
    material_energy_func=marshak_material_energy,
    left_bc_func=marshak_left_bc,
    right_bc_func=marshak_right_bc
)

solver.use_nonlinear_correction = False
solver.max_newton_iter_per_step = 50  # Use convergence

# Initial condition
T_init = 0.01
solver.set_initial_condition(lambda r: np.full_like(r, A_RAD * T_init**4))

print("="*70)
print("DIRICHLET BC DEBUG")
print("="*70)
print(f"Domain: [{solver.r_faces[0]}, {solver.r_faces[-1]}] cm")
print(f"Cells: {solver.n_cells}")
print(f"Cell centers: {solver.r_centers}")
print(f"dx_half[0] = {solver.r_centers[0] - solver.r_faces[0]}")
print(f"V[0] = {solver.V_cells[0]}")
print(f"A[0] = {solver.A_faces[0]}")

T_bc = 1.0
Er_bc = A_RAD * T_bc**4
Er_init = solver.Er[0]
T_init_actual = (Er_init / A_RAD)**0.25

print(f"\nBoundary:")
print(f"  T_bc = {T_bc} keV")
print(f"  Er_bc = {Er_bc:.6e} GJ/cm^3")

print(f"\nInitial cell 0:")
print(f"  T[0] = {T_init_actual} keV")
print(f"  Er[0] = {Er_init:.6e} GJ/cm^3")

# Compute diffusion coefficients
D_cold = C_LIGHT / (3 * marshak_opacity(T_init_actual))
D_hot = C_LIGHT / (3 * marshak_opacity(T_bc))
D_avg = C_LIGHT / (3 * marshak_opacity(0.5*(T_init_actual + T_bc)))

print(f"\nDiffusion coefficients:")
print(f"  D(T_cold) = {D_cold:.6e} cm")
print(f"  D(T_hot) = {D_hot:.6e} cm")
print(f"  D(T_avg) = {D_avg:.6e} cm")

# Manually compute what the flux should be
dx_half = solver.r_centers[0] - solver.r_faces[0]
flux_coeff = (solver.A_faces[0] * D_avg) / (solver.V_cells[0] * dx_half)

print(f"\nFlux coefficient:")
print(f"  flux_coeff = {flux_coeff:.6e}")
print(f"  Expected contribution to Er[0]: {flux_coeff * Er_bc * solver.dt:.6e}")
print(f"  As fraction of Er_bc: {flux_coeff * solver.dt:.6e}")

# Now take multiple steps and see whatres happens
print(f"\nTaking 100 time steps...")
for step in range(100):
    solver.time_step(n_steps=1, verbose=False)
    if (step+1) % 20 == 0:
        T_now = (solver.Er[0]/A_RAD)**0.25
        print(f"  Step {step+1}: Er[0] = {solver.Er[0]:.6e}, T[0] = {T_now:.6e} keV")

print(f"\nAfter 100 steps:")
print(f"  Er[0] = {solver.Er[0]:.6e} GJ/cm^3")
print(f"  T[0] = {(solver.Er[0]/A_RAD)**0.25:.6e} keV")
print(f"  Should approach Er_bc = {Er_bc:.6e}")

print("="*70)
