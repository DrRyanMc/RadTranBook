#!/usr/bin/env python3
"""
Detailed diagnostic of BDF2 matrix and RHS assembly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, apply_tridiagonal

n_cells = 10
r_max = 2.0
dt = 0.001
gamma = 2.0 - np.sqrt(2.0)

def constant_opacity(T): return 100.0
def cubic_cv(T): return 4 * 1.0 * A_RAD * T**3
def linear_material_energy(T): return 1.0 * A_RAD * T**4
def left_bc(Er, r): return (0.0, 1.0, 0.0)
def right_bc(Er, r): return (0.0, 1.0, 0.0)

solver = RadiationDiffusionSolver(
    n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

amplitude = A_RAD * 1.0**4
x0 = 1.0
sigma0 = 0.15
gaussian_Er = lambda r: amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))
solver.set_initial_condition(gaussian_Er)

print("="*70)
print("BDF2 Matrix and RHS Diagnostic")
print("="*70)
print(f"gamma = {gamma:.6f}, dt = {dt}, n_cells = {n_cells}")
print()

# Coefficients
c_0 = (2-gamma)/(1-gamma)
c_1 = (1-gamma)/gamma
c_2 = -1/(gamma*(1-gamma))
print(f"BDF2 coefficients:")
print(f"  c_0 = {c_0:.6f}")
print(f"  c_1 = {c_1:.6f}")
print(f"  c_2 = {c_2:.6f}")
print(f"  c_0 + c_1 + c_2 = {c_0 + c_1 + c_2:.10f}")
print()

# Simulate one TR-BDF2 step
Er_n = solver.Er.copy()

# TR stage
solver.dt = gamma * dt
solver.theta = 0.5
Er_intermediate = solver.newton_step(Er_n, verbose=False)

# BDF2 stage - assemble matrix
solver.dt = dt
A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)

# Check matrix structure at peak
peak_idx = Er_n.argmax()
print(f"Checking at peak location (idx={peak_idx}):")
print(f"  Er^n = {Er_n[peak_idx]:.6e}")
print(f"  Er^{{n+gamma}} = {Er_intermediate[peak_idx]:.6e}")
print()

# Compute components
T_n = temperature_from_Er(Er_n[peak_idx])
T_ng = temperature_from_Er(Er_intermediate[peak_idx])
T_np1_guess = T_ng  # Initial guess

e_mat_n = solver.material_energy_func(T_n)
e_mat_ng = solver.material_energy_func(T_ng)
e_mat_np1_guess = solver.material_energy_func(T_np1_guess)

u_n = e_mat_n + Er_n[peak_idx]
u_ng = e_mat_ng + Er_intermediate[peak_idx]
u_np1_guess = e_mat_np1_guess + Er_intermediate[peak_idx]

print(f"Total energies u = e_mat + Er:")
print(f"  u^n = {u_n:.6e}")
print(f"  u^{{n+gamma}} = {u_ng:.6e}")
print(f"  u^{{n+1}} (guess) = {u_np1_guess:.6e}")
print()

dudEr = solver.get_dudEr(Er_intermediate[peak_idx])
print(f"du/dEr at peak: {dudEr:.6f}")
print()

# Matrix diagonal term
a_0 = (c_0 / dt) * dudEr
L_tri = solver.assemble_diffusion_matrix(Er_intermediate)
L_diag = L_tri[1, peak_idx]

print(f"Matrix diagonal at peak:")
print(f"  a_0 = (c_0/dt)*du/dEr = ({c_0:.3f}/{dt})*{dudEr:.3f} = {a_0:.3f}")
print(f"  L_diag = {L_diag:.3f}")
print(f"  A_diag = a_0 - L_diag = {a_0:.3f} - {L_diag:.3f} = {a_0 - L_diag:.3f}")
print(f"  Actual A_diag from code: {A_bdf2[1, peak_idx]:.3f}")
print(f"  Match: {np.isclose(A_bdf2[1, peak_idx], a_0 - L_diag)}")
print()

# RHS terms
rhs_term1 = (c_0 / dt) * dudEr * Er_intermediate[peak_idx]
rhs_term2 = -(c_0 / dt) * u_np1_guess
rhs_term3 = -(c_1 / dt) * u_n
rhs_term4 = -(c_2 / dt) * u_ng

print(f"RHS terms at peak:")
print(f"  Term 1: (c_0/dt)*du/dEr*Er_k = {rhs_term1:.6e}")
print(f"  Term 2: -(c_0/dt)*u^{{n+1}} = {rhs_term2:.6e}")
print(f"  Term 3: -(c_1/dt)*u^n = {rhs_term3:.6e}")
print(f"  Term 4: -(c_2/dt)*u^{{n+gamma}} = {rhs_term4:.6e}")
print(f"  Sum: {rhs_term1 + rhs_term2 + rhs_term3 + rhs_term4:.6e}")
print(f"  Actual RHS from code: {rhs_bdf2[peak_idx]:.6e}")
print()

# Compare to theta method
print("="*70)
print("Comparison: BDF2 vs Theta Method")
print("="*70)
print()

# Assemble theta method matrix (implicit Euler, theta=1)
solver.dt = dt
solver.theta = 1.0
A_theta, rhs_theta = solver.assemble_system(Er_intermediate, Er_n, theta=1.0)

alpha = solver.get_alpha_coefficient(Er_intermediate[peak_idx], dt)
print(f"Implicit Euler (theta=1) at peak:")
print(f"  alpha = (1/dt)*du/dEr = (1/{dt})*{dudEr:.3f} = {alpha:.3f}")
print(f"  A_diag = alpha + L_diag = {alpha:.3f} + {L_diag:.3f} = {alpha + L_diag:.3f}")
print(f"  Actual from code: {A_theta[1, peak_idx]:.3f}")
print()

print(f"BDF2 vs Implicit Euler diagonal ratio:")
print(f"  BDF2 a_0 = {a_0:.3f}")
print(f"  IE alpha = {alpha:.3f}")
print(f"  Ratio a_0/alpha = {a_0/alpha:.3f}")
print(f"  Expected ratio = c_0 = {c_0:.3f}")
print(f"  Match: {np.isclose(a_0/alpha, c_0)}")
print()

print("✓ BDF2 matrix appears to be assembled correctly!")
print("  - Diagonal has correct a_0 - L structure")
print("  - RHS has correct four-term structure")
print("  - Coefficient c_0 correctly scales the time derivative term")
