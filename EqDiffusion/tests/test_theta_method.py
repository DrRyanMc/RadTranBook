#!/usr/bin/env python3
"""
Test the theta method implementation for time integration.

This test uses a simplified linear diffusion problem where we can verify:
1. The explicit RHS is constant during Newton iterations
2. Theta=0.5 (Crank-Nicolson) gives better accuracy than theta=1.0
3. Energy/solution evolution is physically reasonable
"""

import numpy as np
import matplotlib.pyplot as plt
from oneDFV import RadiationDiffusionSolver, A_RAD

def constant_opacity(T):
    """Constant opacity for linear diffusion test"""
    return 1.0

def constant_cv(T):
    """Constant specific heat"""
    return 1.0

def material_energy(T):
    """Linear material energy"""
    return constant_cv(T) * T

def left_bc(Er, r):
    """Dirichlet BC: T=1 at left boundary"""
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def right_bc(Er, r):
    """Vacuum BC at right boundary"""
    return 1.0, -1.0, 0.0

def test_explicit_rhs_constant():
    """Test 1: Verify explicit RHS doesn't change during Newton iterations"""
    print("="*70)
    print("Test 1: Explicit RHS should be constant during Newton iterations")
    print("="*70)
    
    # Create solver with theta = 0.5
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=20, d=0,
        dt=0.01, theta=0.5,
        max_newton_iter=5, newton_tol=1e-10,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=constant_cv,
        material_energy_func=material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    # Set initial condition: T=0.1 everywhere
    T_init = 0.1
    Er_init = A_RAD * T_init**4
    solver.set_initial_condition(Er_init)
    
    # Instrument the solver to track explicit RHS during Newton iterations
    original_assemble = solver.assemble_system
    explicit_rhs_history = []
    
    def instrumented_assemble(Er_k, Er_prev):
        A_tri, rhs = original_assemble(Er_k, Er_prev)
        
        # Compute the explicit diffusion contribution
        if solver.theta < 1.0:
            n_cells = len(Er_prev)
            explicit_contrib = np.zeros(n_cells)
            
            # Recompute explicit RHS (this duplicates code but allows verification)
            D_cells_prev = np.array([solver.get_diffusion_coefficient(Er_prev[i]) for i in range(n_cells)])
            D_faces_prev = np.zeros(len(solver.r_faces))
            
            for i in range(1, len(solver.r_faces) - 1):
                from oneDFV import temperature_from_Er
                T_left = temperature_from_Er(Er_prev[i-1])
                T_right = temperature_from_Er(Er_prev[i])
                T_face = 0.5 * (T_left + T_right)
                Er_face = A_RAD * T_face**4
                D_faces_prev[i] = solver.get_diffusion_coefficient(Er_face)
            
            D_faces_prev[0] = D_cells_prev[0]
            D_faces_prev[-1] = D_cells_prev[-1]
            
            for i in range(n_cells):
                V_i = solver.V_cells[i]
                A_left = solver.A_faces[i]
                A_right = solver.A_faces[i + 1]
                
                if i == 0:
                    dr_left = solver.r_centers[i] - solver.r_faces[i]
                else:
                    dr_left = solver.r_centers[i] - solver.r_centers[i - 1]
                    
                if i == n_cells - 1:
                    dr_right = solver.r_faces[i + 1] - solver.r_centers[i]
                else:
                    dr_right = solver.r_centers[i + 1] - solver.r_centers[i]
                
                flux_left = 0.0
                flux_right = 0.0
                
                if i > 0:
                    flux_left = -D_faces_prev[i] * (Er_prev[i] - Er_prev[i-1]) / dr_left
                if i < n_cells - 1:
                    flux_right = -D_faces_prev[i+1] * (Er_prev[i+1] - Er_prev[i]) / dr_right
                
                explicit_contrib[i] = (A_right * flux_right - A_left * flux_left) / V_i
            
            explicit_rhs_history.append(explicit_contrib.copy())
        
        return A_tri, rhs
    
    solver.assemble_system = instrumented_assemble
    
    # Take one time step
    solver.time_step(n_steps=1, verbose=False)
    
    # Check that explicit RHS was the same for all Newton iterations
    if len(explicit_rhs_history) > 1:
        max_variation = 0.0
        for i in range(1, len(explicit_rhs_history)):
            variation = np.max(np.abs(explicit_rhs_history[i] - explicit_rhs_history[0]))
            max_variation = max(max_variation, variation)
        
        print(f"Number of Newton iterations: {len(explicit_rhs_history)}")
        print(f"Max variation in explicit RHS across iterations: {max_variation:.2e}")
        
        if max_variation < 1e-14:
            print("✓ PASSED: Explicit RHS is constant during Newton iterations")
        else:
            print("✗ FAILED: Explicit RHS changes during Newton iterations!")
            print("This indicates Er_prev is being modified or not properly passed")
    else:
        print("Only one Newton iteration performed, cannot verify constancy")
    
    print()
    return max_variation < 1e-14 if len(explicit_rhs_history) > 1 else True

def test_theta_convergence():
    """Test 2: Verify convergence rate with different theta values"""
    print("="*70)
    print("Test 2: Temporal convergence with different theta values")
    print("="*70)
    
    # Final time
    t_final = 0.1
    
    # Test different dt values
    dt_values = [0.01, 0.005, 0.0025]
    theta_values = [0.5, 1.0]
    
    results = {}
    
    for theta in theta_values:
        solutions = []
        
        for dt in dt_values:
            n_steps = int(t_final / dt)
            
            solver = RadiationDiffusionSolver(
                r_min=0.0, r_max=1.0, n_cells=50, d=0,
                dt=dt, theta=theta,
                max_newton_iter=10, newton_tol=1e-10,
                rosseland_opacity_func=constant_opacity,
                specific_heat_func=constant_cv,
                material_energy_func=material_energy,
                left_bc_func=left_bc,
                right_bc_func=right_bc
            )
            
            T_init = 0.1
            Er_init = A_RAD * T_init**4
            solver.set_initial_condition(Er_init)
            
            solver.time_step(n_steps=n_steps, verbose=False)
            r, Er = solver.get_solution()
            
            solutions.append(Er.copy())
        
        results[theta] = solutions
    
    # Compute convergence rates
    print("\nConvergence analysis:")
    for theta in theta_values:
        print(f"\nTheta = {theta} ({'Crank-Nicolson' if theta==0.5 else 'Implicit Euler'}):")
        
        # Compare finest grid solution to coarser grids
        for i in range(len(dt_values) - 1):
            error = np.linalg.norm(results[theta][i] - results[theta][-1])
            print(f"  dt={dt_values[i]:.4f}: ||E_r - E_r_finest|| = {error:.4e}")
        
        # Estimate convergence rate
        if len(dt_values) >= 3:
            error_coarse = np.linalg.norm(results[theta][0] - results[theta][-1])
            error_medium = np.linalg.norm(results[theta][1] - results[theta][-1])
            
            if error_medium > 1e-14 and error_coarse > 1e-14:
                rate = np.log(error_coarse / error_medium) / np.log(dt_values[0] / dt_values[1])
                print(f"  Estimated convergence rate: {rate:.2f}")
                print(f"  Expected: ~{2 if theta==0.5 else 1}")
                
                if theta == 0.5 and rate > 1.5:
                    print("  ✓ Good: Crank-Nicolson showing second-order convergence")
                elif theta == 1.0 and 0.8 < rate < 1.5:
                    print("  ✓ Good: Implicit Euler showing first-order convergence")
                else:
                    print(f"  ⚠ Warning: Unexpected convergence rate")
    
    print()

def test_energy_conservation():
    """Test 3: Check energy conservation/balance"""
    print("="*70)
    print("Test 3: Energy evolution and conservation")
    print("="*70)
    
    theta_values = [0.5, 1.0]
    dt = 0.01
    n_steps = 20
    
    for theta in theta_values:
        print(f"\nTheta = {theta}:")
        
        solver = RadiationDiffusionSolver(
            r_min=0.0, r_max=1.0, n_cells=30, d=0,
            dt=dt, theta=theta,
            max_newton_iter=10, newton_tol=1e-10,
            rosseland_opacity_func=constant_opacity,
            specific_heat_func=constant_cv,
            material_energy_func=material_energy,
            left_bc_func=left_bc,
            right_bc_func=right_bc
        )
        
        T_init = 0.1
        Er_init = A_RAD * T_init**4
        solver.set_initial_condition(Er_init)
        
        # Track total energy over time
        energies = []
        times = []
        
        for step in range(n_steps):
            r, Er = solver.get_solution()
            # Integrate radiation energy over domain
            total_Er = np.sum(Er * solver.V_cells)
            energies.append(total_Er)
            times.append(step * dt)
            
            solver.time_step(n_steps=1, verbose=False)
        
        # Final measurement
        r, Er = solver.get_solution()
        total_Er = np.sum(Er * solver.V_cells)
        energies.append(total_Er)
        times.append(n_steps * dt)
        
        energies = np.array(energies)
        
        # Energy should be increasing (heat flowing in from boundary)
        if np.all(np.diff(energies) > 0):
            print("  ✓ Energy is monotonically increasing (expected with hot BC)")
        else:
            print("  ✗ Energy is not monotonic!")
            neg_changes = np.sum(np.diff(energies) < 0)
            print(f"    {neg_changes} time steps had decreasing energy")
        
        # Check for reasonable values
        if np.all(Er > 0) and np.all(np.isfinite(Er)):
            print("  ✓ Solution remains positive and finite")
        else:
            print("  ✗ Solution has negative or infinite values!")
    
    print()

def plot_comparison():
    """Visual comparison of different theta values"""
    print("="*70)
    print("Creating visual comparison plot")
    print("="*70)
    
    dt = 0.02
    n_steps = 25
    theta_values = [0.0, 0.5, 1.0]
    labels = ['Explicit (θ=0)', 'Crank-Nicolson (θ=0.5)', 'Implicit (θ=1)']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for theta, label in zip(theta_values, labels):
        solver = RadiationDiffusionSolver(
            r_min=0.0, r_max=1.0, n_cells=40, d=0,
            dt=dt, theta=theta,
            max_newton_iter=10, newton_tol=1e-10,
            rosseland_opacity_func=constant_opacity,
            specific_heat_func=constant_cv,
            material_energy_func=material_energy,
            left_bc_func=left_bc,
            right_bc_func=right_bc
        )
        
        T_init = 0.1
        Er_init = A_RAD * T_init**4
        solver.set_initial_condition(Er_init)
        
        try:
            solver.time_step(n_steps=n_steps, verbose=False)
            r, Er = solver.get_solution()
            
            # Convert to temperature for visualization
            from oneDFV import temperature_from_Er
            T = np.array([temperature_from_Er(er) for er in Er])
            
            axes[0].plot(r, T, label=label, marker='o', markersize=3)
            
            # Track energy evolution
            energies = []
            times = []
            
            solver.set_initial_condition(Er_init)
            for step in range(n_steps):
                r_temp, Er_temp = solver.get_solution()
                energies.append(np.sum(Er_temp * solver.V_cells))
                times.append(step * dt)
                solver.time_step(n_steps=1, verbose=False)
            
            r_temp, Er_temp = solver.get_solution()
            energies.append(np.sum(Er_temp * solver.V_cells))
            times.append(n_steps * dt)
            
            axes[1].plot(times, energies, label=label)
            
        except Exception as e:
            print(f"  ⚠ {label} failed: {str(e)}")
    
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Temperature')
    axes[0].set_title(f'Temperature Profile at t={n_steps*dt:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Total Radiation Energy')
    axes[1].set_title('Energy Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theta_method_comparison.png', dpi=150)
    print("Plot saved as 'theta_method_comparison.png'")
    print()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("THETA METHOD VERIFICATION TESTS")
    print("="*70 + "\n")
    
    # Run tests
    test1_passed = test_explicit_rhs_constant()
    test_theta_convergence()
    test_energy_conservation()
    plot_comparison()
    
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("Run the tests above and verify:")
    print("1. Explicit RHS is constant during Newton iterations")
    print("2. Crank-Nicolson (θ=0.5) shows ~2nd order convergence")
    print("3. Implicit Euler (θ=1.0) shows ~1st order convergence")
    print("4. Energy evolution is physically reasonable")
    print("5. Visual comparison shows consistent behavior")
    print("="*70)
