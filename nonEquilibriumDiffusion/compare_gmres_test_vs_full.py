#!/usr/bin/env python3
"""
Side-by-side GMRES comparison for test vs full Marshak setup.
This actually runs one timestep with/without preconditioner for both cases.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')

from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D, Bg_multigroup
)
from diffusion_operator_solver import C_LIGHT, A_RAD

print("="*80)
print("Side-by-side GMRES Iteration Comparison")
print("="*80)

# Common setup
n_groups = 10
energy_edges = np.array([0.00001, 0.000316, 0.001, 0.00316, 0.01,
                          0.0316,  0.1,     0.316, 1.0,     3.16, 10.0])
rho = 1.0

def sigma_func(T, r, E_low, E_high):
    E_mid = 2.0 * E_high * E_low / (E_high + E_low)
    T_safe = max(T, 0.01)
    return min(100000.0 * rho * (T_safe)**(-0.5) * E_mid**(-3.0), 1e14)

sigma_funcs = []
diff_funcs = []
for g in range(n_groups):
    sigma_funcs.append(lambda T, r, g=g: sigma_func(T, r, energy_edges[g], energy_edges[g+1]))
    diff_funcs.append(lambda T, r, g=g: C_LIGHT / (3.0 * sigma_func(T, r, energy_edges[g], energy_edges[g+1])))

def run_test_case(use_precond, verbose=False):
    """Test case: dt=0.5, n_cells=30, r_max=5.0, heterogeneous"""
    n_cells = 30
    r_max = 5.0
    dt = 0.5
    cv = 0.1
    T_bc = 0.5
    
    # Heterogeneous sigma (factor of 10 at r=2.5)
    def hetero_sigma(T, r, g):
        scale = 1.0 if r < 2.5 else 10.0
        return scale * sigma_func(T, r, energy_edges[g], energy_edges[g+1])
    
    def hetero_diff(T, r, g):
        return C_LIGHT / (3.0 * hetero_sigma(T, r, g))
    
    sigma_funcs_h = [lambda T, r, g=g: hetero_sigma(T, r, g) for g in range(n_groups)]
    diff_funcs_h = [lambda T, r, g=g: hetero_diff(T, r, g) for g in range(n_groups)]
    
    # Marshak BC
    B_g_bc = Bg_multigroup(energy_edges, T_bc)
    chi_bc = B_g_bc / B_g_bc.sum()
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    
    left_bc_funcs = []
    for g in range(n_groups):
        D_g_bc = C_LIGHT / (3.0 * hetero_sigma(T_bc, 0.0, g))
        F_g = chi_bc[g] * F_total
        def make_bc(D=D_g_bc, F=F_g):
            return lambda phi, r: (0.5, 2.0 * D, F)
        left_bc_funcs.append(make_bc())
    
    right_bc_funcs = [lambda phi, r: (0.0, 1.0, 0.0)] * n_groups
    
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs_h,
        absorption_coeff_funcs=sigma_funcs_h,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    T_init = 0.05 * np.ones(n_cells)
    solver.T = T_init.copy()
    solver.T_old = T_init.copy()
    solver.E_r = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    info = solver.step(
        max_newton_iter=1,
        newton_tol=1e-8,
        gmres_tol=1e-12,
        gmres_maxiter=300,
        use_preconditioner=use_precond,
        verbose=verbose
    )
    
    return info['gmres_info']['iterations']

def run_full_problem(use_precond, verbose=False):
    """Full Marshak: dt=0.01, n_cells=50, r_max=1.0, homogeneous"""
    n_cells = 50
    r_max = 1.0
    dt = 0.01
    cv = 0.05
    T_bc = 0.5
    
    # Homogeneous (no spatial variation)
    
    # Marshak BC
    B_g_bc = Bg_multigroup(energy_edges, T_bc)
    chi_bc = B_g_bc / B_g_bc.sum()
    F_total = (A_RAD * C_LIGHT * T_bc**4) / 2.0
    
    left_bc_funcs = []
    for g in range(n_groups):
        D_g_bc = diff_funcs[g](T_bc, 0.0)
        F_g = chi_bc[g] * F_total
        def make_bc(D=D_g_bc, F=F_g):
            return lambda phi, r: (0.5, 2.0 * D, F)
        left_bc_funcs.append(make_bc())
    
    right_bc_funcs = [lambda phi, r: (0.0, 1.0, 0.0)] * n_groups
    
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    T_init = 0.05 * np.ones(n_cells)
    solver.T = T_init.copy()
    solver.T_old = T_init.copy()
    solver.E_r = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    info = solver.step(
        max_newton_iter=1,
        newton_tol=1e-8,
        gmres_tol=1e-8,  # Different tolerance!
        gmres_maxiter=300,
        use_preconditioner=use_precond,
        verbose=verbose
    )
    
    return info['gmres_info']['iterations']

# ============================================================================
# Run Test Case
# ============================================================================
print("\n" + "="*80)
print("TEST CASE (heterogeneous, dt=0.5 ns, tol=1e-12)")
print("="*80)

test_no_prec = run_test_case(use_precond=False, verbose=False)
test_with_prec = run_test_case(use_precond=True, verbose=False)

print(f"  Without preconditioner: {test_no_prec} iterations")
print(f"  With preconditioner:    {test_with_prec} iterations")
print(f"  → {100*(test_no_prec - test_with_prec)/test_no_prec:.1f}% reduction")

# ============================================================================
# Run Full Marshak Problem
# ============================================================================
print("\n" + "="*80)
print("FULL MARSHAK (homogeneous, dt=0.01 ns, tol=1e-8)")
print("="*80)

full_no_prec = run_full_problem(use_precond=False, verbose=False)
full_with_prec = run_full_problem(use_precond=True, verbose=False)

print(f"  Without preconditioner: {full_no_prec} iterations")
print(f"  With preconditioner:    {full_with_prec} iterations")
if full_no_prec > full_with_prec:
    print(f"  → {100*(full_no_prec - full_with_prec)/full_no_prec:.1f}% reduction")
else:
    print(f"  → NO IMPROVEMENT (actually {full_with_prec - full_no_prec} more iterations!)")

# ============================================================================
# Key Differences
# ============================================================================
print("\n" + "="*80)
print("KEY DIFFERENCES")
print("="*80)
print("""
Test Case (WORKS):
  - Spatial heterogeneity: factor of 10 opacity jump at r=2.5
  - Larger domain: r_max=5.0 cm
  - Larger timestep: dt=0.5 ns  
  - Tighter GMRES tol: 1e-12
  - More cells resolve the opacity jump
  
Full Marshak (DOESN'T WORK):
  - Spatially homogeneous: uniform opacity everywhere
  - Smaller domain: r_max=1.0 cm
  - Smaller timestep: dt=0.01 ns
  - Looser GMRES tol: 1e-8
  - No heterogeneity to create group coupling
  
DIAGNOSIS:
The preconditioner improves convergence by exploiting the gray approximation
to capture spatial coupling between groups. In the TEST CASE, the opacity
heterogeneity creates spatial variation that couples groups differently at
different locations - the gray operator approximates this.

In the FULL MARSHAK, the domain is spatially homogeneous. All groups see the
same spatial structure, so there's less for the gray averaging to capture.
The problem is more "group-diagonal" already.

The preconditioner is most effective when:
  1. Strong spatial heterogeneity (material properties vary in space)
  2. Groups have different responses to that heterogeneity
  3. The gray average captures the dominant spatial coupling
""")
