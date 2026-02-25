"""
0-D Manufactured solution test for Crank-Nicolson (θ = 0.5)

This test verifies that the Crank-Nicolson implementation correctly solves
the 0-D equations by comparing against analytical solutions.
"""
import sys
from oneDFV import NonEquilibriumRadiationDiffusionSolver
import numpy as np

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm^3 keV^4)
RHO = 1.0          # g/cm^3
CV_CONST = 0.1     # GJ/(g keV)

def analytical_0d_solution(phi_n, T_n, dt, theta, sigma_P):
    """Compute analytical 0-D solution for one timestep
    
    Equations:
    φ^{n+1} = [φ^n/(c·dt) + σ_P·f·a·c·T^4 - σ_P·f·(1-θ)·φ^n] / [1/(c·dt) + σ_P·f·θ]
    T^{n+1} = T^n + (dt/C_v)·f·σ_P·(φ̃ - a·c·T^4)
    
    where:
    - f = 1 / (1 + β·σ_P·c·θ·dt)
    - β = 4·a·T^3/(ρ·C_v)
    - φ̃ = θ·φ^{n+1} + (1-θ)·φ^n
    """
    c = C_LIGHT
    a = A_RAD
    C_v = RHO * CV_CONST
    
    # Calculate β at T^n (linearization point)
    beta = (4.0 * A_RAD * T_n**3) / (RHO * CV_CONST)
    
    # Calculate f factor
    f = 1.0 / (1.0 + beta * sigma_P * c * theta * dt)
    
    # Solve for φ^{n+1}
    acT4 = a * c * T_n**4
    diag_coeff = 1.0/(c*dt) + sigma_P * f * theta
    rhs = phi_n/(c*dt) + sigma_P*f*acT4 - sigma_P*f*(1.0-theta)*phi_n
    phi_np1 = rhs / diag_coeff
    
    # Solve for T^{n+1}
    phi_tilde = theta * phi_np1 + (1.0 - theta) * phi_n
    T_np1 = T_n + (dt/C_v) * f * sigma_P * (phi_tilde - acT4)
    
    return phi_np1, T_np1, f, beta

def test_crank_nicolson_single_step(dt, phi_init, T_init, sigma_P, test_name):
    """Test a single Crank-Nicolson timestep"""
    
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Initial: φ = {phi_init:.10e}, T = {T_init:.10e}")
    print(f"Δt = {dt} ns, σ_P = {sigma_P} cm^-1, θ = 0.5 (Crank-Nicolson)")
    
    # Material properties
    def custom_opacity(T):
        return sigma_P

    def custom_specific_heat(T):
        return CV_CONST

    def custom_material_energy(T):
        return RHO * CV_CONST * T

    # Reflecting boundary conditions
    def reflecting_bc(phi, x):
        return 0.0, 1.0, 0.0  # A=0, B=1, C=0 => zero flux

    # Create 1-cell solver with Crank-Nicolson
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=1, d=0,
        dt=dt, max_newton_iter=1, newton_tol=1e-10,
        rosseland_opacity_func=custom_opacity,
        planck_opacity_func=custom_opacity,
        specific_heat_func=custom_specific_heat,
        material_energy_func=custom_material_energy,
        left_bc_func=reflecting_bc,
        right_bc_func=reflecting_bc,
        theta=0.5  # Crank-Nicolson!
    )

    # Set initial conditions
    solver.phi = np.array([phi_init])
    solver.T = np.array([T_init])
    
    # Get analytical solution
    phi_analytical, T_analytical, f, beta = analytical_0d_solution(
        phi_init, T_init, dt, theta=0.5, sigma_P=sigma_P
    )
    
    print(f"\nAnalytical 0-D solution:")
    print(f"  β = {beta:.10e}")
    print(f"  f = {f:.10e}")
    print(f"  φ^{{n+1}} = {phi_analytical:.10e}")
    print(f"  T^{{n+1}} = {T_analytical:.10e}")
    
    # Check energy conservation
    C_v = RHO * CV_CONST
    E_init = phi_init + C_v * T_init
    E_final = phi_analytical + C_v * T_analytical
    print(f"  E_init = {E_init:.10e}, E_final = {E_final:.10e}")
    print(f"  |ΔE|/E = {abs(E_final - E_init)/E_init:.6e}")
    
    # Solve with 1-cell solver
    A_tri, rhs = solver.assemble_phi_equation(
        solver.phi, solver.T, solver.phi, solver.T, theta=0.5
    )
    solver.apply_boundary_conditions_phi(A_tri, rhs, solver.phi)
    
    phi_solver = np.linalg.solve(np.diag(A_tri['diag']), rhs)[0]
    
    # Solve for T
    T_solver_array = solver.solve_T_equation(
        np.array([phi_solver]), solver.T, solver.phi, solver.T, theta=0.5
    )
    T_solver = T_solver_array[0]
    
    print(f"\n1-cell solver (θ=0.5):")
    print(f"  φ^{{n+1}} = {phi_solver:.10e}")
    print(f"  T^{{n+1}} = {T_solver:.10e}")
    
    # Check solver energy conservation
    E_solver = phi_solver + C_v * T_solver
    print(f"  E_solver = {E_solver:.10e}")
    print(f"  |ΔE|/E = {abs(E_solver - E_init)/E_init:.6e}")
    
    # Compare
    phi_error_rel = abs(phi_solver - phi_analytical) / abs(phi_analytical) if phi_analytical != 0 else abs(phi_solver - phi_analytical)
    T_error_rel = abs(T_solver - T_analytical) / abs(T_analytical) if T_analytical != 0 else abs(T_solver - T_analytical)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON:")
    print(f"{'='*80}")
    print(f"Δφ (absolute): {abs(phi_solver - phi_analytical):.6e}")
    print(f"Δφ (relative):  {phi_error_rel:.6e}")
    print(f"ΔT (absolute): {abs(T_solver - T_analytical):.6e}")
    print(f"ΔT (relative):  {T_error_rel:.6e}")
    
    tolerance = 1e-4  # 0.01% tolerance
    passed = phi_error_rel < tolerance and T_error_rel < tolerance
    
    if passed:
        print(f"\n✓ PASS: Crank-Nicolson matches analytical solution (within {tolerance*100:.3f}%)")
    else:
        print(f"\n✗ FAIL: Errors exceed tolerance of {tolerance*100:.3f}%")
        
    return passed

def test_crank_nicolson_multi_step(n_steps, dt, phi_init, T_init, sigma_P):
    """Test multiple Crank-Nicolson timesteps"""
    
    print(f"\n{'='*80}")
    print(f"Multi-step test: {n_steps} timesteps of Δt = {dt} ns")
    print(f"{'='*80}")
    
    # Material properties
    def custom_opacity(T):
        return sigma_P

    def custom_specific_heat(T):
        return CV_CONST

    def custom_material_energy(T):
        return RHO * CV_CONST * T

    # Reflecting boundary conditions
    def reflecting_bc(phi, x):
        return 0.0, 1.0, 0.0

    # Create 1-cell solver
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=1, d=0,
        dt=dt, max_newton_iter=5, newton_tol=1e-10,
        rosseland_opacity_func=custom_opacity,
        planck_opacity_func=custom_opacity,
        specific_heat_func=custom_specific_heat,
        material_energy_func=custom_material_energy,
        left_bc_func=reflecting_bc,
        right_bc_func=reflecting_bc,
        theta=0.5
    )

    # Set initial conditions for both
    solver.phi = np.array([phi_init])
    solver.T = np.array([T_init])
    
    phi_analytical = phi_init
    T_analytical = T_init
    
    C_v = RHO * CV_CONST
    E_init = phi_init + C_v * T_init
    
    print(f"Initial: φ = {phi_init:.10e}, T = {T_init:.10e}, E = {E_init:.10e}\n")
    
    # Evolve both
    for step in range(n_steps):
        # Analytical step
        phi_analytical, T_analytical, f, beta = analytical_0d_solution(
            phi_analytical, T_analytical, dt, theta=0.5, sigma_P=sigma_P
        )
        
        # Solver step - capture returned values and update
        phi_new, T_new = solver.newton_step(solver.phi.copy(), solver.T.copy(), verbose=False)
        solver.phi = phi_new
        solver.T = T_new
        
        if step == n_steps - 1:
            print(f"Step {step+1}/{n_steps}:")
            print(f"  Analytical: φ = {phi_analytical:.10e}, T = {T_analytical:.10e}")
            print(f"  Solver:     φ = {solver.phi[0]:.10e}, T = {solver.T[0]:.10e}")
            print(f"  Δφ (rel): {abs(solver.phi[0] - phi_analytical)/abs(phi_analytical):.6e}")
            print(f"  ΔT (rel): {abs(solver.T[0] - T_analytical)/abs(T_analytical):.6e}")
    
    # Final comparison
    E_analytical = phi_analytical + C_v * T_analytical
    E_solver = solver.phi[0] + C_v * solver.T[0]
    
    print(f"\nFinal energies:")
    print(f"  Analytical: {E_analytical:.10e}, ΔE/E = {abs(E_analytical-E_init)/E_init:.6e}")
    print(f"  Solver:     {E_solver:.10e}, ΔE/E = {abs(E_solver-E_init)/E_init:.6e}")
    
    phi_error_rel = abs(solver.phi[0] - phi_analytical) / abs(phi_analytical)
    T_error_rel = abs(solver.T[0] - T_analytical) / abs(T_analytical)
    
    tolerance = 1e-4
    passed = phi_error_rel < tolerance and T_error_rel < tolerance
    
    if passed:
        print(f"\n✓ PASS: Multi-step Crank-Nicolson matches analytical (within {tolerance*100:.3f}%)")
    else:
        print(f"\n✗ FAIL: Multi-step errors exceed tolerance")
        
    return passed

if __name__ == "__main__":
    print("="*80)
    print("0-D MANUFACTURED SOLUTION TESTS FOR CRANK-NICOLSON")
    print("="*80)
    
    # Test parameters
    phi_init = 0.01372  # GJ/cm^3 (T_rad = 1 keV)
    T_init = 0.4        # keV
    sigma_P = 100.0     # cm^-1
    
    all_passed = True
    
    # Test 1: Small timestep
    passed = test_crank_nicolson_single_step(
        dt=0.001, phi_init=phi_init, T_init=T_init, sigma_P=sigma_P,
        test_name="Small timestep (dt=0.001 ns)"
    )
    all_passed = all_passed and passed
    
    # Test 2: Larger timestep
    passed = test_crank_nicolson_single_step(
        dt=0.01, phi_init=phi_init, T_init=T_init, sigma_P=sigma_P,
        test_name="Larger timestep (dt=0.01 ns)"
    )
    all_passed = all_passed and passed
    
    # Test 3: Different initial conditions (cooler material)
    passed = test_crank_nicolson_single_step(
        dt=0.001, phi_init=0.01372, T_init=0.1, sigma_P=100.0,
        test_name="Cooler initial temperature (T=0.1 keV)"
    )
    all_passed = all_passed and passed
    
    # Test 4: Different initial conditions (hotter material)
    passed = test_crank_nicolson_single_step(
        dt=0.001, phi_init=0.01372, T_init=1.0, sigma_P=100.0,
        test_name="Hotter initial temperature (T=1.0 keV)"
    )
    all_passed = all_passed and passed
    
    # Test 5: Multiple timesteps
    passed = test_crank_nicolson_multi_step(
        n_steps=10, dt=0.001, phi_init=phi_init, T_init=T_init, sigma_P=sigma_P
    )
    all_passed = all_passed and passed
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nConclusion: The Crank-Nicolson implementation correctly solves")
        print("the 0-D non-equilibrium radiation diffusion equations.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the implementation.")
