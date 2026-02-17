#!/usr/bin/env python3
"""
Debug why TR-BDF2 fails with small k_coefficient in LINEAR case
"""

import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD

def test_with_k(k_value):
    """Test TR-BDF2 with specific k_coefficient"""
    
    n_cells = 50
    r_max = 2.0
    dt = 0.001
    gamma = 2.0 - np.sqrt(2.0)
    
    def constant_opacity(T): return 100.0
    def cubic_cv(T): return 4 * k_value * A_RAD * T**3
    def linear_material_energy(T): return k_value * A_RAD * T**4
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
    
    print(f"\n{'='*70}")
    print(f"Testing with k_coefficient = {k_value}")
    print(f"{'='*70}")
    
    Er_n = solver.Er.copy()
    peak_idx = Er_n.argmax()
    
    # Check dudEr
    dudEr_peak = solver.get_dudEr(Er_n[peak_idx])
    print(f"At peak: Er = {Er_n[peak_idx]:.6e}")
    print(f"  du/dEr = 1 + k = {dudEr_peak:.6f} (expected: {1 + k_value:.6f})")
    
    # TR stage
    print(f"\nTR Stage:")
    solver.dt = gamma * dt
    solver.theta = 0.5
    try:
        Er_intermediate = solver.newton_step(Er_n, verbose=False)
        print(f"  ✓ Succeeded")
        print(f"  Peak Er changed: {Er_n[peak_idx]:.6e} -> {Er_intermediate[peak_idx]:.6e}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # BDF2 stage
    print(f"\nBDF2 Stage:")
    solver.dt = dt
    
    # Check what's in the matrix and RHS
    A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)
    
    c_0 = (2-gamma)/(1-gamma)
    a_0_peak = (c_0 / dt) * dudEr_peak
    L_tri = solver.assemble_diffusion_matrix(Er_intermediate)
    L_peak = L_tri[1, peak_idx]
    
    print(f"  Matrix at peak:")
    print(f"    a_0 = (c_0/dt)*du/dEr = {a_0_peak:.3e}")
    print(f"    L_diag = {L_peak:.3e}")
    print(f"    A_diag = a_0 - L = {A_bdf2[1, peak_idx]:.3e}")
    print(f"    Ratio a_0/L = {a_0_peak/L_peak:.3f}")
    
    # Check RHS terms
    T_intermediate = temperature_from_Er(Er_intermediate[peak_idx])
    e_mat_intermediate = solver.material_energy_func(T_intermediate)
    u_intermediate = e_mat_intermediate + Er_intermediate[peak_idx]
    
    print(f"  At peak:")
    print(f"    Er = {Er_intermediate[peak_idx]:.6e}")
    print(f"    e_mat = k*Er = {e_mat_intermediate:.6e}")
    print(f"    u = e_mat + Er = {u_intermediate:.6e}")
    print(f"    u/Er ratio = {u_intermediate/Er_intermediate[peak_idx]:.6f} (expected: {1+k_value:.6f})")
    
    try:
        Er_final = solver.newton_step_bdf2(Er_n, Er_intermediate, gamma, verbose=False)
        print(f"  ✓ Succeeded")
        print(f"  Peak Er: {Er_intermediate[peak_idx]:.6e} -> {Er_final[peak_idx]:.6e}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        
        # Check for problematic values
        if np.any(~np.isfinite(A_bdf2)):
            print(f"    Matrix has NaN/Inf!")
        if np.any(~np.isfinite(rhs_bdf2)):
            print(f"    RHS has NaN/Inf!")
        if np.any(A_bdf2[1, :] < 0):
            neg_diag = np.where(A_bdf2[1, :] < 0)[0]
            print(f"    Matrix has negative diagonal at {len(neg_diag)} cells!")
            print(f"      First few: {neg_diag[:5]}")
        
        return False

# Test with different k values
print("\n" + "="*70)
print("Testing TR-BDF2 with different k_coefficient values")
print("="*70)

k_values = [1.0, 0.5, 0.1, 0.05, 0.01]
results = {}

for k in k_values:
    success = test_with_k(k)
    results[k] = success

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for k, success in results.items():
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"k = {k:5.2f}: {status}")

print("\nIf some k values fail, the BDF2 implementation has a problem")
print("specific to weak material coupling that doesn't appear in")
print("implicit Euler or trapezoidal methods.")
