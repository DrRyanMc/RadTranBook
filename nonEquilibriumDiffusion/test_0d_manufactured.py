"""
Test 0-D equations with manufactured solution
Verify that one Newton iteration gives the analytically expected result
"""
import numpy as np
import sys
sys.path.insert(0, '../Problems')
from oneDFV import NonEquilibriumRadiationDiffusionSolver

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372  # GJ/cm³/keV⁴
RHO = 1.0  # g/cm³

def test_single_newton_iteration(theta, dt, T_n, phi_n, sigma_P, C_v):
    """
    Test a single Newton iteration analytically.
    
    With 1 Newton iteration, T★ = T^n, so Δe = e(T★) - e(T^n) = 0
    
    Equation 8.59a (no spatial derivatives):
    (φ^{n+1} - φ^n)/(c·Δt) = σ_P·f·(acT★⁴ - φ̃) - (1-f)·Δe/Δt
    
    where φ̃ = θφ^{n+1} + (1-θ)φ^n
    
    Rearranging:
    φ^{n+1}·[1/(c·Δt) + σ_P·f·θ] = φ^n/(c·Δt) + σ_P·f·acT★⁴ - σ_P·f·(1-θ)φ^n
    
    Equation 8.59b:
    e(T_{n+1}) = e(T_n) + Δt·[f·σ_P(φ̃ - acT★⁴)]
    
    where φ̃ uses the φ^{n+1} computed above
    """
    print("="*70)
    print(f"Testing θ = {theta}, Δt = {dt} ns")
    print("="*70)
    
    # Initial conditions
    T_star = T_n  # Linearization point for 1 Newton iteration
    e_n = C_v * T_n
    e_star = C_v * T_star
    Delta_e = e_star - e_n  # Should be zero for 1 iteration
    
    # Compute f and β
    beta = (4.0 * A_RAD * T_star**3) / C_v
    f = 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)
    
    print(f"\nInput:")
    print(f"  T^n = {T_n:.6f} keV")
    print(f"  φ^n = {phi_n:.6e} GJ/cm²")
    print(f"  E_r^n = {phi_n/C_LIGHT:.6e} GJ/cm³")
    print(f"  σ_P = {sigma_P} cm⁻¹")
    print(f"  C_v = {C_v} GJ/(cm³·keV)")
    print(f"  β = {beta:.6f}")
    print(f"  f = {f:.6f}")
    print(f"  Δe = {Delta_e:.6e} (should be 0 for 1 iteration)")
    
    # Analytical solution for φ^{n+1}
    acT4_star = A_RAD * C_LIGHT * T_star**4
    
    # φ^{n+1} equation
    lhs_coeff = 1.0 / (C_LIGHT * dt) + sigma_P * f * theta
    rhs = phi_n / (C_LIGHT * dt) + sigma_P * f * acT4_star - sigma_P * f * (1.0 - theta) * phi_n
    if Delta_e != 0:
        rhs -= (1.0 - f) * Delta_e / dt
    
    phi_np1_analytical = rhs / lhs_coeff
    
    # Compute φ̃ using the φ^{n+1} we just found
    phi_tilde = theta * phi_np1_analytical + (1.0 - theta) * phi_n
    
    # T_{n+1} equation
    e_np1_analytical = e_n + dt * f * sigma_P * (phi_tilde - acT4_star)
    if Delta_e != 0:
        e_np1_analytical += (1.0 - f) * Delta_e
    T_np1_analytical = e_np1_analytical / C_v
    
    print(f"\nAnalytical result (1 Newton iteration):")
    print(f"  φ^{{n+1}} = {phi_np1_analytical:.6e} GJ/cm²")
    print(f"  T^{{n+1}} = {T_np1_analytical:.6f} keV")
    print(f"  E_r^{{n+1}} = {phi_np1_analytical/C_LIGHT:.6e} GJ/cm³")
    print(f"  φ̃ = {phi_tilde:.6e} GJ/cm²")
    
    # Check energy conservation
    E_total_n = C_v * T_n + phi_n / C_LIGHT
    E_total_np1 = C_v * T_np1_analytical + phi_np1_analytical / C_LIGHT
    print(f"\nEnergy check:")
    print(f"  E_total^n = {E_total_n:.6e} GJ/cm³")
    print(f"  E_total^{{n+1}} = {E_total_np1:.6e} GJ/cm³")
    print(f"  ΔE/E = {abs(E_total_np1 - E_total_n)/E_total_n:.6e}")
    
    # Now test with the solver (force 1 Newton iteration)
    print(f"\n" + "-"*70)
    print("Testing with solver (max_newton_iter=1)...")
    print("-"*70)
    
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
        n_cells=1,  # 0-D: single cell
        d=0,
        dt=dt,
        max_newton_iter=1,  # Force exactly 1 iteration
        newton_tol=1e-20,  # Set very tight so it doesn't accidentally converge early
        rosseland_opacity_func=rosseland_opacity,
        planck_opacity_func=planck_opacity,
        specific_heat_func=specific_heat,
        material_energy_func=material_energy,
        left_bc_func=reflecting_bc_left,
        right_bc_func=reflecting_bc_right,
        theta=theta
    )
    
    # Set initial conditions
    solver.set_initial_condition(phi_init=phi_n, T_init=T_n)
    
    # Take one time step (which will do exactly 1 Newton iteration)
    solver.time_step(n_steps=1, verbose=False)
    
    phi_np1_solver = solver.phi[0]
    T_np1_solver = solver.T[0]
    
    print(f"\nSolver result (1 Newton iteration):")
    print(f"  φ^{{n+1}} = {phi_np1_solver:.6e} GJ/cm²")
    print(f"  T^{{n+1}} = {T_np1_solver:.6f} keV")
    print(f"  E_r^{{n+1}} = {phi_np1_solver/C_LIGHT:.6e} GJ/cm³")
    
    # Compare
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    phi_error = abs(phi_np1_solver - phi_np1_analytical)
    T_error = abs(T_np1_solver - T_np1_analytical)
    phi_rel_error = phi_error / abs(phi_np1_analytical)
    T_rel_error = T_error / abs(T_np1_analytical)
    
    print(f"Δφ = {phi_error:.6e} (relative: {phi_rel_error:.6e})")
    print(f"ΔT = {T_error:.6e} (relative: {T_rel_error:.6e})")
    
    # Check if they match
    tolerance = 1e-10
    phi_matches = phi_rel_error < tolerance
    T_matches = T_rel_error < tolerance
    
    if phi_matches and T_matches:
        print(f"\n✓ PASS: Solver matches analytical solution to {tolerance}")
        return True
    else:
        print(f"\n✗ FAIL: Solver does NOT match analytical solution")
        print(f"  Expected tolerance: {tolerance}")
        if not phi_matches:
            print(f"  φ relative error: {phi_rel_error:.6e} > {tolerance}")
        if not T_matches:
            print(f"  T relative error: {T_rel_error:.6e} > {tolerance}")
        return False


def main():
    print("\n" + "="*70)
    print("0-D MANUFACTURED SOLUTION TEST")
    print("Verify single Newton iteration matches analytical solution")
    print("="*70)
    
    # Test parameters (similar to equilibration test)
    C_v = 0.01  # GJ/cm³/keV
    sigma_P = 10.0  # cm⁻¹
    T_n = 0.4  # keV
    T_rad_n = 1.0  # keV
    phi_n = A_RAD * C_LIGHT * T_rad_n**4  # GJ/cm²
    
    # Test different time integrators
    print("\n")
    test_cases = [
        ("Backward Euler", 1.0, 0.001),
        ("Backward Euler", 1.0, 0.01),
        ("Crank-Nicolson", 0.5, 0.001),
        ("Crank-Nicolson", 0.5, 0.01),
    ]
    
    results = []
    for name, theta, dt in test_cases:
        print("\n")
        passed = test_single_newton_iteration(theta, dt, T_n, phi_n, sigma_P, C_v)
        results.append((name, theta, dt, passed))
        print("\n")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, theta, dt, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name} (θ={theta}, Δt={dt} ns)")
    
    all_passed = all(r[3] for r in results)
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70)


if __name__ == "__main__":
    main()
