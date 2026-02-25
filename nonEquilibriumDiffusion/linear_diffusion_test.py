"""
Test linear diffusion with decoupled material temperature.

Setup:
- Material energy: e(T) = C_v * T with C_v very large (10^10)
- This effectively decouples material from radiation
- Initial Gaussian radiation profile
- σ_a = σ_P = σ_R = 1.0 cm^-1
- D = c/(3σ_R) = 10 cm²/ns
- Initial T very small

The radiation should diffuse as a Gaussian:
φ(x,t) = φ_0 * (σ₀/σ(t)) * exp(-x²/(2σ(t)²))
where σ(t)² = σ₀² + 2Dt
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Physical constants (from oneDFV.py)
C_LIGHT = 2.99792458e1  # cm/ns
A_RAD = 0.01372  # GJ/(cm^3·keV^4)
RHO = 1.0  # g/cm^3

# Problem parameters
x_min = 0.0    # cm
x_max = 50.0   # cm
n_cells = 200

# Opacities
sigma_a = 0.0  # cm^-1 (no absorption/coupling)
sigma_p = 0.0  # cm^-1 (no Planck absorption - turns off coupling)
sigma_r = 1.0  # cm^-1 (Rosseland for diffusion)

# Diffusion coefficient
D = C_LIGHT / (3.0 * sigma_r)
print(f"Diffusion coefficient D = {D:.6f} cm²/ns")

# Gaussian parameters
x_center = 0.0  # Center at left boundary (reflecting)
sigma_0 = 2.0   # Initial width (cm)
phi_0 = 1.0     # Peak amplitude

# Time parameters (based on diffusion timescale, not absorption)
diffusion_time = sigma_0**2 / D  # Characteristic diffusion time
print(f"Diffusion time t_diff = σ₀²/D = {diffusion_time:.6e} ns")

dt = 0.01 * diffusion_time
n_steps = 100  # Run to 1 diffusion time
final_time = n_steps * dt

print(f"Timestep: {dt:.6e} ns = {dt/diffusion_time:.3f} t_diff")
print(f"Final time: {final_time:.6e} ns = {final_time/diffusion_time:.3f} t_diff")

# Material properties (decoupled)
CV_LARGE = 1e10  # Very large specific heat to decouple material from radiation

def decoupled_opacity(T):
    """Planck opacity (zero to turn off coupling)"""
    return sigma_p

def decoupled_rosseland_opacity(T):
    """Constant Rosseland opacity"""
    return sigma_r

def decoupled_specific_heat(T):
    """Very large constant specific heat"""
    return CV_LARGE / RHO

def decoupled_energy(T):
    """Linear material energy: e = C_v * T"""
    return CV_LARGE * T

def decoupled_inverse_energy(e):
    """Inverse: T from e"""
    return e / CV_LARGE

# Boundary conditions
def reflecting_bc(phi, x):
    """Reflecting boundary: zero flux"""
    return 0.0, 1.0, 0.0  # A=0, B=1, C=0

def vacuum_bc(phi, x):
    """Vacuum boundary: φ = 0"""
    return 1.0, 0.0, 0.0  # A=1, B=0, C=0

# Create solver with Backward Euler
print(f"\nInitializing solver with {n_cells} cells...")
solver = NonEquilibriumRadiationDiffusionSolver(
    r_min=x_min, r_max=x_max, n_cells=n_cells, d=0,  # Planar geometry
    dt=dt, max_newton_iter=10, newton_tol=1e-8,
    rosseland_opacity_func=decoupled_rosseland_opacity,
    planck_opacity_func=decoupled_opacity,
    specific_heat_func=decoupled_specific_heat,
    material_energy_func=decoupled_energy,
    inverse_material_energy_func=decoupled_inverse_energy,
    left_bc_func=reflecting_bc,
    right_bc_func=vacuum_bc,
    theta=1.0  # Backward Euler for stability
)

# Initial φ profile
x_cells = solver.r_centers
phi_init = phi_0 * np.exp(-x_cells**2 / (2 * sigma_0**2))

# Initial temperature (very small, essentially decoupled)
T_init = 1e-6  # keV

solver.phi = phi_init.copy()
solver.T = np.ones(n_cells) * T_init

print(f"Initial conditions:")
print(f"  T = {T_init} keV (constant, very small)")
print(f"  φ: Gaussian with σ₀ = {sigma_0} cm, peak = {phi_0} GJ/cm³")
print(f"  φ: max = {np.max(phi_init):.6f}, integral = {np.sum(phi_init * solver.V_cells):.6f} GJ")

# Analytic solution at final time
sigma_t = np.sqrt(sigma_0**2 + 2 * D * final_time)
phi_analytic_final = phi_0 * (sigma_0 / sigma_t) * np.exp(-x_cells**2 / (2 * sigma_t**2))

print(f"\nAnalytic solution at t = {final_time:.6e} ns:")
print(f"  σ(t) = {sigma_t:.6f} cm (initial: {sigma_0} cm)")
print(f"  Peak should decay by factor: {sigma_0/sigma_t:.6f}")

# Newton iteration function (no source term)
def newton_step_no_source(solver, phi_prev, T_prev, verbose=True):
    """Perform Newton iterations without source term"""
    phi_star = phi_prev.copy()
    T_star = T_prev.copy()
    
    for k in range(solver.max_newton_iter):
        # Assemble φ equation
        A_phi, rhs_phi = solver.assemble_phi_equation(
            phi_star, T_star, phi_prev, T_prev, theta=solver.theta)
        
        # Apply boundary conditions
        solver.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
        
        # Solve for φ
        from oneDFV import solve_tridiagonal
        phi_np1 = solve_tridiagonal(A_phi, rhs_phi)
        
        # Solve for T
        T_np1 = solver.solve_T_equation(
            phi_np1, T_star, phi_prev, T_prev, theta=solver.theta)
        
        # Check convergence
        r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
        r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
        
        if verbose:
            print(f"  Newton iteration {k+1}: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
        
        if r_phi < solver.newton_tol and r_T < solver.newton_tol:
            if verbose:
                print(f"  Converged in {k+1} iterations!")
            return phi_np1, T_np1
        
        phi_star = phi_np1.copy()
        T_star = T_np1.copy()
    
    print(f"  Warning: Max iterations reached")
    return phi_star, T_star

print(f"\n{'='*80}")
print("Starting time evolution...")
print(f"{'='*80}")

# Time evolution loop
current_time = 0.0
for step in range(n_steps):
    current_time = (step + 1) * dt
    
    # Print progress every 20 steps
    verbose = (step < 3) or (step % 20 == 0) or (step == n_steps - 1)
    
    if verbose:
        print(f"\nStep {step+1}/{n_steps}: t = {current_time:.6e} ns = {current_time/diffusion_time:.3f} t_diff")
    
    # Take timestep
    phi_new, T_new = newton_step_no_source(
        solver, solver.phi, solver.T, verbose=verbose
    )
    
    solver.phi = phi_new
    solver.T = T_new

print(f"\nCompleted {n_steps} steps to t = {current_time:.6e} ns")

# Compare with analytic solution
phi_final = solver.phi
error = phi_final - phi_analytic_final
rms_error = np.sqrt(np.mean(error**2))
max_error = np.max(np.abs(error))
relative_rms_error = rms_error / np.max(phi_analytic_final)

print(f"\n{'='*80}")
print("Comparison with analytic solution:")
print(f"{'='*80}")
print(f"Numerical solution:")
print(f"  φ: max = {np.max(phi_final):.6f}, integral = {np.sum(phi_final * solver.V_cells):.6f} GJ")
print(f"Analytic solution:")
print(f"  φ: max = {np.max(phi_analytic_final):.6f}, integral = {np.sum(phi_analytic_final * solver.V_cells):.6f} GJ")
print(f"Error:")
print(f"  RMS error = {rms_error:.6e}")
print(f"  Max error = {max_error:.6e}")
print(f"  Relative RMS error = {relative_rms_error:.6%}")

# Plotting
print(f"\n{'='*80}")
print("Creating plots...")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial and final φ
ax = axes[0, 0]
ax.plot(x_cells, phi_init, 'b--', linewidth=2, label='Initial (t=0)')
ax.plot(x_cells, phi_final, 'r-', linewidth=2, label=f'Numerical (t={final_time:.4e} ns)')
ax.plot(x_cells, phi_analytic_final, 'k:', linewidth=3, label='Analytic')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Radiation Energy Density φ (GJ/cm³)', fontsize=12)
ax.set_title('Linear Diffusion Test', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

# Plot 2: Error
ax = axes[0, 1]
ax.plot(x_cells, error, 'r-', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Error (Numerical - Analytic)', fontsize=12)
ax.set_title(f'Error (RMS = {rms_error:.4e})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

# Plot 3: Log scale comparison
ax = axes[1, 0]
ax.semilogy(x_cells, phi_init, 'b--', linewidth=2, label='Initial')
ax.semilogy(x_cells, phi_final, 'r-', linewidth=2, label='Numerical')
ax.semilogy(x_cells, phi_analytic_final, 'k:', linewidth=3, label='Analytic')
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Radiation Energy Density φ (GJ/cm³)', fontsize=12)
ax.set_title('Log Scale', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])
ax.set_ylim([1e-6, 2])

# Plot 4: Material temperature
ax = axes[1, 1]
ax.plot(x_cells, solver.T * 1e6, 'b-', linewidth=2)
ax.set_xlabel('Position x (cm)', fontsize=12)
ax.set_ylabel('Material Temperature T (10⁻⁶ keV)', fontsize=12)
ax.set_title('Material Temperature (should be nearly constant)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 20])

plt.tight_layout()
plt.savefig('linear_diffusion_test.png', dpi=150, bbox_inches='tight')
print("Saved plot to linear_diffusion_test.png")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Problem: Pure radiation diffusion with no material coupling")
print(f"Domain: x ∈ [0, {x_max}] cm with {n_cells} cells")
print(f"Diffusion coefficient: D = c/(3σ_R) = {D:.6f} cm²/ns")
print(f"Material: e(T) = C_v·T with C_v = {CV_LARGE:.0e} (decoupled)")
print(f"Opacity: σ_P = {sigma_p} cm^-1 (no coupling), σ_R = {sigma_r} cm^-1")
print(f"Time discretization: Backward Euler (θ = 1.0)")
print(f"Timestep: dt = {dt:.6e} ns = {dt/diffusion_time:.3f} t_diff")
print(f"Final time: t = {final_time:.6e} ns = {final_time/diffusion_time:.3f} t_diff")
print(f"\nInitial Gaussian: σ₀ = {sigma_0} cm")
print(f"Final Gaussian (analytic): σ(t) = {sigma_t:.6f} cm")
print(f"Width increase factor: {sigma_t/sigma_0:.6f}")
print(f"\nRelative RMS error: {relative_rms_error:.6%}")
