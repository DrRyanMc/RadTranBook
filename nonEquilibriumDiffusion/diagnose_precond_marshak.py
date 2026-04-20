#!/usr/bin/env python3
"""
Diagnostic script to compare why the preconditioner works on the test case
but not on the full Marshak problem.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, Bg_multigroup
from diffusion_operator_solver import C_LIGHT, A_RAD

print("="*80)
print("DIAGNOSTIC: Comparing Test Case vs Full Marshak Problem")
print("="*80)

# Common parameters
n_groups = 10
energy_edges = np.array([0.00001, 0.000316, 0.001, 0.00316, 0.01,
                          0.0316,  0.1,     0.316, 1.0,     3.16, 10.0])

def analyze_case(name, n_cells, r_max, dt, T_init, T_bc, rho=1.0, cv=0.1):
    """Analyze operator characteristics for a given case"""
    
    print(f"\n{'='*80}")
    print(f"Case: {name}")
    print(f"{'='*80}")
    print(f"  n_cells = {n_cells}")
    print(f"  r_max   = {r_max} cm")
    print(f"  dt      = {dt} ns")
    print(f"  T_init  = {T_init} keV")
    print(f"  T_bc    = {T_bc} keV")
    
    # Compute characteristic terms
    inv_c_dt = 1.0 / (C_LIGHT * dt)
    print(f"\n  Key operator terms:")
    print(f"    1/(c·dt) = {inv_c_dt:.6e} cm⁻¹")
    
    # Estimate sigma_a at T_init for middle energy
    E_mid = 0.1  # keV, roughly middle of range
    T_safe = max(T_init, 0.01)
    sigma_a_est = 100000.0 * rho * (T_safe)**(-0.5) * E_mid**(-3.0)
    sigma_a_est = min(sigma_a_est, 1e14)
    D_est = C_LIGHT / (3.0 * sigma_a_est)
    
    print(f"    σ_a(T_init, E_mid) ≈ {sigma_a_est:.6e} cm⁻¹")
    print(f"    D(T_init, E_mid)   ≈ {D_est:.6e} cm²/ns")
    
    # Estimate Fleck factor denominator
    # f = 1 / (1 + 4π (dt/c_v) Σ σ_a ∂B/∂T)
    # For rough estimate: ∂B/∂T ~ B/T ~ a·T³
    dB_dT_est = 4.0 * A_RAD * T_safe**3
    coupling = 4.0 * np.pi * (dt / cv) * sigma_a_est * dB_dT_est * n_groups
    fleck_denom = 1.0 + coupling
    fleck_est = 1.0 / fleck_denom
    
    print(f"    4π(dt/c_v)Σσ_a·∂B/∂T ≈ {coupling:.6e}")
    print(f"    Fleck factor f ≈ {fleck_est:.6f}")
    print(f"    (1-f) ≈ {1.0 - fleck_est:.6f}")
    
    # H operator absorption term: σ_a*(1-f) + 1/(c*dt)
    H_absorption = sigma_a_est * (1.0 - fleck_est) + inv_c_dt
    print(f"\n  Gray operator H terms (estimated):")
    print(f"    σ_a·(1-f) = {sigma_a_est * (1.0 - fleck_est):.6e} cm⁻¹")
    print(f"    1/(c·dt)  = {inv_c_dt:.6e} cm⁻¹")
    print(f"    Total absorption: {H_absorption:.6e} cm⁻¹")
    
    # Diffusion length scale
    diffusion_length = np.sqrt(D_est / H_absorption)
    spatial_scale = r_max / n_cells
    
    print(f"\n  Characteristic length scales:")
    print(f"    Diffusion length λ_D = √(D/σ_eff) ≈ {diffusion_length:.6e} cm")
    print(f"    Spatial resolution Δx ≈ {spatial_scale:.6e} cm")
    print(f"    Ratio λ_D/Δx ≈ {diffusion_length/spatial_scale:.3f}")
    
    # Key diagnostic: ratio of time vs space terms
    if sigma_a_est * (1.0 - fleck_est) > 0:
        ratio = inv_c_dt / (sigma_a_est * (1.0 - fleck_est))
        print(f"\n  *** CRITICAL: [1/(c·dt)] / [σ_a·(1-f)] = {ratio:.3f} ***")
        if ratio > 10:
            print(f"      → Time term DOMINATES (ratio > 10)")
            print(f"      → Spatial coupling is weak → preconditioner less effective")
        elif ratio > 1:
            print(f"      → Time term is significant (ratio > 1)")
            print(f"      → Preconditioner may help but effect is reduced")
        else:
            print(f"      → Spatial term dominates (ratio < 1)")
            print(f"      → Preconditioner should be effective")
    
    return {
        'inv_c_dt': inv_c_dt,
        'sigma_a': sigma_a_est,
        'D': D_est,
        'fleck': fleck_est,
        'H_absorption': H_absorption,
        'diffusion_length': diffusion_length,
        'spatial_scale': spatial_scale
    }

# ============================================================================
# Analyze the test case that WORKS
# ============================================================================
test_info = analyze_case(
    name="Test Case (test_preconditioner_marshak_bc.py)",
    n_cells=30,
    r_max=5.0,
    dt=0.5,      # ns
    T_init=0.05,
    T_bc=0.5,
    rho=1.0,
    cv=0.1
)

# ============================================================================
# Analyze the full problem that DOESN'T WORK
# ============================================================================
full_info = analyze_case(
    name="Full Marshak Problem (marshak_wave_multigroup_powerlaw.py)",
    n_cells=50,
    r_max=1.0,
    dt=0.01,     # ns - MUCH SMALLER!
    T_init=0.05,
    T_bc=0.5,
    rho=1.0,
    cv=0.05      # Also different!
)

# ============================================================================
# Summary comparison
# ============================================================================
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

print(f"\nTimestep ratio (full/test): {full_info['inv_c_dt']/test_info['inv_c_dt']:.1f}x")
print(f"  → Full problem has {full_info['inv_c_dt']/test_info['inv_c_dt']:.1f}x larger 1/(c·dt)")

print(f"\nSpatial resolution:")
print(f"  Test: Δx ≈ {test_info['spatial_scale']:.4f} cm")
print(f"  Full: Δx ≈ {full_info['spatial_scale']:.4f} cm")

print(f"\nDiffusion length:")
print(f"  Test: λ_D ≈ {test_info['diffusion_length']:.4f} cm")
print(f"  Full: λ_D ≈ {full_info['diffusion_length']:.4f} cm")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("""
The full Marshak problem uses dt = 0.01 ns (50x smaller than the test case).
This makes the 1/(c·dt) term dominate the operator, which reduces spatial
coupling between groups. The preconditioner approximates the spatial structure
of B via the gray operator H, but when time-coupling dominates, the operator
becomes nearly block-diagonal in space, and the gray approximation provides
little benefit.

RECOMMENDATION:
1. Try larger timesteps if stability permits (dt ~ 0.1 ns)
2. Accept that preconditioner is less effective for small-timestep problems
3. Monitor the [1/(c·dt)] / [σ_a·(1-f)] ratio - when > 10, expect weak benefit
""")
