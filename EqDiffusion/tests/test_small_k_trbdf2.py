#!/usr/bin/env python3
"""
Test TR-BDF2 with SMALL k_coefficient using test_double_trapezoidal setup
"""

import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT

def test_with_small_k():
    n_cells = 200
    r_max = 2.0
    dt = 0.001
    n_steps = 50
    sigma_R = 100.0
    D = C_LIGHT / (3.0 * sigma_R)
    k_coupling = 0.01  # SMALL k
    x0 = 1.0
    sigma0 = 0.15
    amplitude = A_RAD * 1.0**4
    t_final = n_steps * dt
    gamma = 2.0 - np.sqrt(2.0)
    
    def constant_opacity(T):
        return sigma_R
    
    def cubic_cv(T):
        return 4 * k_coupling * A_RAD * T**3
    
    def linear_material_energy(T):
        return k_coupling * A_RAD * T**4
    
    def left_bc(r, t):
        return (0.0, 1.0, 0.0)
    
    def right_bc(r, t):
        return (0.0, 1.0, 0.0)
    
    def gaussian_Er(r):
        return amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))
    
    print("="*70)
    print(f"Testing TR-BDF2 with k_coupling = {k_coupling}")
    print("="*70)
    print(f"n_cells = {n_cells}, dt = {dt}, n_steps = {n_steps}")
    print()
    
    solver = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver.set_initial_condition(gaussian_Er)
    
    print(f"Initial peak T = {temperature_from_Er(solver.Er).max():.4f} keV")
    
    for step in range(n_steps):
        try:
            solver.time_step_trbdf2(n_steps=1, gamma=gamma, verbose=False)
            if step % 10 == 0:
                T = temperature_from_Er(solver.Er)
                print(f"  Step {step}: Peak T = {T.max():.4f} keV")
                
                # Check for problems
                if np.any(~np.isfinite(solver.Er)):
                    print(f"    ✗ ERROR: NaN/Inf values appeared!")
                    return False
                    
        except Exception as e:
            print(f"  ✗ FAILED at step {step}: {e}")
            return False
    
    T_final = temperature_from_Er(solver.Er)
    print(f"\nFinal peak T = {T_final.max():.4f} keV")
    print("✓ Test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_with_small_k()
    if not success:
        print("\n" + "="*70)
        print("TR-BDF2 FAILED with small k_coefficient!")
        print("This suggests a problem in the BDF2 implementation that")
        print("doesn't appear with larger k values or with other methods.")
        print("="*70)
