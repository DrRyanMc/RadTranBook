#!/usr/bin/env python3
"""
Test two-stage trapezoidal method instead of TR-BDF2
Stage 1: theta=0.5 from t^n to t^{n+gamma}
Stage 2: theta=0.5 from t^{n+gamma} to t^{n+1}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT

def test_double_trap_vs_trbdf2():
    """Compare double trapezoidal to TR-BDF2"""
    
    n_cells = 200
    r_max = 2.0
    dt = 0.001
    n_steps = 50
    sigma_R = 100.0
    D = C_LIGHT / (3.0 * sigma_R)
    k_coupling = 1.0
    x0 = 1.0
    sigma0 = 0.15
    amplitude = A_RAD * 1.0**4
    t_final = n_steps * dt
    
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
    
    def analytical_solution(r, t):
        D_eff = D / (1.0 + k_coupling)
        sigma_t_sq = sigma0**2 + 2 * D_eff * t
        sigma_t = np.sqrt(sigma_t_sq)
        amplitude_t = amplitude * sigma0 / sigma_t
        return amplitude_t * np.exp(-(r - x0)**2 / (2 * sigma_t_sq))
    
    print("="*70)
    print("Comparing Double Trapezoidal vs TR-BDF2")
    print("="*70)
    
    gamma = 2.0 - np.sqrt(2.0)
    
    # Test 1: Double Trapezoidal
    print(f"\nTest 1: Double Trapezoidal (gamma={gamma:.6f})")
    print("-"*70)
    
    solver_dt = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver_dt.set_initial_condition(gaussian_Er)
    
    original_dt = dt
    for step in range(n_steps):
        # Stage 1: theta=0.5 from t^n to t^{n+gamma}
        Er_n = solver_dt.Er.copy()
        solver_dt.dt = gamma * original_dt
        solver_dt.theta = 0.5
        Er_intermediate = solver_dt.newton_step(Er_n, verbose=False)
        
        # Stage 2: theta=0.5 from t^{n+gamma} to t^{n+1}
        solver_dt.dt = (1.0 - gamma) * original_dt
        solver_dt.theta = 0.5
        solver_dt.Er = solver_dt.newton_step(Er_intermediate, verbose=False)
        
        if step % 10 == 0:
            T = temperature_from_Er(solver_dt.Er)
            print(f"  Step {step}: Peak T = {T.max():.4f} keV")
    
    r, Er_dt = solver_dt.get_solution()
    T_dt = temperature_from_Er(Er_dt)
    
    # Test 2: TR-BDF2
    print(f"\nTest 2: TR-BDF2 (gamma={gamma:.6f})")
    print("-"*70)
    
    solver_trbdf2 = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver_trbdf2.set_initial_condition(gaussian_Er)
    
    for step in range(n_steps):
        solver_trbdf2.time_step_trbdf2(n_steps=1, gamma=gamma, verbose=False)
        if step % 10 == 0:
            T = temperature_from_Er(solver_trbdf2.Er)
            print(f"  Step {step}: Peak T = {T.max():.4f} keV")
    
    r, Er_trbdf2 = solver_trbdf2.get_solution()
    T_trbdf2 = temperature_from_Er(Er_trbdf2)
    
    # Test 3: Implicit Euler
    print(f"\nTest 3: Implicit Euler (baseline)")
    print("-"*70)
    
    solver_ie = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt, theta=1.0, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver_ie.set_initial_condition(gaussian_Er)
    solver_ie.time_step(n_steps=n_steps, verbose=False)
    
    r, Er_ie = solver_ie.get_solution()
    T_ie = temperature_from_Er(Er_ie)
    
    # Analytical solution
    Er_analytical = analytical_solution(r, t_final)
    T_analytical = temperature_from_Er(Er_analytical)
    
    # Compare results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    error_dt = np.sqrt(np.mean((Er_dt - Er_analytical)**2))
    rel_error_dt = error_dt / np.sqrt(np.mean(Er_analytical**2))
    
    error_trbdf2 = np.sqrt(np.mean((Er_trbdf2 - Er_analytical)**2))
    rel_error_trbdf2 = error_trbdf2 / np.sqrt(np.mean(Er_analytical**2))
    
    error_ie = np.sqrt(np.mean((Er_ie - Er_analytical)**2))
    rel_error_ie = error_ie / np.sqrt(np.mean(Er_analytical**2))
    
    print(f"\nAnalytical:          Peak T = {T_analytical.max():.4f} keV")
    print(f"Double Trapezoidal:  Peak T = {T_dt.max():.4f} keV, Error = {rel_error_dt*100:.2f}%")
    print(f"TR-BDF2:             Peak T = {T_trbdf2.max():.4f} keV, Error = {rel_error_trbdf2*100:.2f}%")
    print(f"Implicit Euler:      Peak T = {T_ie.max():.4f} keV, Error = {rel_error_ie*100:.2f}%")
    
    # Peak decay comparison
    peak_decay_analytical = (1.0 - T_analytical.max()) * 100
    peak_decay_dt = (1.0 - T_dt.max()) * 100
    peak_decay_trbdf2 = (1.0 - T_trbdf2.max()) * 100
    peak_decay_ie = (1.0 - T_ie.max()) * 100
    
    print(f"\nPeak decay from initial 1.0 keV:")
    print(f"  Analytical:          {peak_decay_analytical:.2f}%")
    print(f"  Double Trapezoidal:  {peak_decay_dt:.2f}% (should match analytical)")
    print(f"  TR-BDF2:             {peak_decay_trbdf2:.2f}% (should match analytical)")
    print(f"  Implicit Euler:      {peak_decay_ie:.2f}%")
    
    print(f"\n{'='*70}")
    if rel_error_dt < rel_error_trbdf2:
        print("✓ Double Trapezoidal is MORE accurate than TR-BDF2!")
        print("  This suggests the BDF2 stage has a problem.")
    else:
        print("✗ TR-BDF2 is more accurate - the BDF2 stage is working correctly.")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_double_trap_vs_trbdf2()
