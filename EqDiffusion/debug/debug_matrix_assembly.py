#!/usr/bin/env python3
"""
Debug matrix assembly issues with small k_coupling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT

def test_matrix_assembly(k_coupling, verbose=True):
    """Test matrix assembly with given k_coupling"""
    
    n_cells = 10  # Small for easier debugging
    r_max = 2.0
    dt = 0.001
    sigma_R = 100.0
    x0 = 1.0
    sigma0 = 0.15
    amplitude = A_RAD * 1.0**4
    
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
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing with k_coupling = {k_coupling}")
        print(f"{'='*70}")
    
    solver = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver.set_initial_condition(gaussian_Er)
    
    # Get initial values
    Er_init = solver.Er.copy()
    r = solver.r_centers
    
    if verbose:
        print(f"\nInitial Er: min={Er_init.min():.6e}, max={Er_init.max():.6e}")
        T_init = temperature_from_Er(Er_init)
        print(f"Initial T: min={T_init.min():.6e}, max={T_init.max():.6e}")
    
    # Test Stage 1: Trapezoidal
    gamma = 2.0 - np.sqrt(2.0)
    Er_n = Er_init.copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STAGE 1: Trapezoidal rule (dt = {gamma*dt:.6e})")
        print(f"{'='*70}")
    
    # Set up for trapezoidal
    solver.dt = gamma * dt
    solver.theta = 0.5
    
    # Manually assemble system to inspect it
    Er_k = Er_n.copy()
    
    try:
        A_tri, rhs = solver.assemble_system(Er_k, Er_n, theta=0.5)
        
        if verbose:
            print(f"\nMatrix after assembly (before BC):")
            print(f"  Diagonal: min={A_tri[1,:].min():.6e}, max={A_tri[1,:].max():.6e}")
            print(f"  Sub-diag: min={A_tri[0,:].min():.6e}, max={A_tri[0,:].max():.6e}")
            print(f"  Super-diag: min={A_tri[2,:].min():.6e}, max={A_tri[2,:].max():.6e}")
            print(f"  RHS: min={rhs.min():.6e}, max={rhs.max():.6e}")
            
            # Check for invalid values
            if np.any(~np.isfinite(A_tri)):
                print(f"  ⚠ WARNING: Matrix contains NaN or Inf!")
                invalid_count = np.sum(~np.isfinite(A_tri))
                print(f"    Number of invalid entries: {invalid_count}")
                
            if np.any(~np.isfinite(rhs)):
                print(f"  ⚠ WARNING: RHS contains NaN or Inf!")
            
            if np.any(np.abs(A_tri[1,:]) < 1e-14):
                print(f"  ⚠ WARNING: Diagonal has near-zero elements!")
                zero_diag = np.where(np.abs(A_tri[1,:]) < 1e-14)[0]
                print(f"    Indices with near-zero diagonal: {zero_diag}")
        
        # Apply boundary conditions
        solver.apply_boundary_conditions(A_tri, rhs, Er_k)
        
        if verbose:
            print(f"\nMatrix after BC:")
            print(f"  Diagonal: min={A_tri[1,:].min():.6e}, max={A_tri[1,:].max():.6e}")
            print(f"  Diagonal values: {A_tri[1,:]}")
            
            # Check conditioning
            if np.any(np.abs(A_tri[1,:]) < 1e-14):
                print(f"  ✗ ERROR: Diagonal has near-zero elements after BC!")
                zero_diag = np.where(np.abs(A_tri[1,:]) < 1e-14)[0]
                print(f"    Indices: {zero_diag}")
                print(f"    Values: {A_tri[1,zero_diag]}")
                return False
            
        print(f"  ✓ Stage 1 matrix assembly OK")
        
        # Now try to actually solve Stage 1
        try:
            Er_intermediate = solver.newton_step(Er_n, verbose=False)
            if verbose:
                print(f"\nStage 1 solution:")
                print(f"  Er_intermediate: min={Er_intermediate.min():.6e}, max={Er_intermediate.max():.6e}")
                T_inter = temperature_from_Er(Er_intermediate)
                print(f"  T_intermediate: min={T_inter.min():.6e}, max={T_inter.max():.6e}")
                
                if np.any(~np.isfinite(Er_intermediate)):
                    print(f"  ✗ ERROR: Stage 1 produced invalid values!")
                    return False
                    
            # Now try Stage 2: BDF2
            if verbose:
                print(f"\n{'='*70}")
                print(f"STAGE 2: BDF2")
                print(f"{'='*70}")
            
            solver.dt = dt  # Full time step
            A_tri_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)
            
            if verbose:
                print(f"\nBDF2 Matrix after assembly (before BC):")
                print(f"  Diagonal: min={A_tri_bdf2[1,:].min():.6e}, max={A_tri_bdf2[1,:].max():.6e}")
                print(f"  Sub-diag: min={A_tri_bdf2[0,:].min():.6e}, max={A_tri_bdf2[0,:].max():.6e}")
                print(f"  Super-diag: min={A_tri_bdf2[2,:].min():.6e}, max={A_tri_bdf2[2,:].max():.6e}")
                print(f"  RHS: min={rhs_bdf2.min():.6e}, max={rhs_bdf2.max():.6e}")
                
                # Check for invalid values
                if np.any(~np.isfinite(A_tri_bdf2)):
                    print(f"  ⚠ WARNING: BDF2 Matrix contains NaN or Inf!")
                    
                if np.any(~np.isfinite(rhs_bdf2)):
                    print(f"  ⚠ WARNING: BDF2 RHS contains NaN or Inf!")
                    
                if np.any(np.abs(A_tri_bdf2[1,:]) < 1e-14):
                    print(f"  ⚠ WARNING: BDF2 Diagonal has near-zero elements!")
                    
            print(f"  ✓ Stage 2 matrix assembly OK")
            return True
            
        except Exception as e2:
            if verbose:
                print(f"  ✗ ERROR during Stage 1 solve or Stage 2: {type(e2).__name__}: {e2}")
                import traceback
                traceback.print_exc()
            return False
        
    except Exception as e:
        if verbose:
            print(f"  ✗ ERROR during assembly: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return False

# Run tests
print("="*70)
print("Matrix Assembly Diagnostics")
print("="*70)

k_values = [0.01, 0.1, 0.5, 1.0]

for k in k_values:
    success = test_matrix_assembly(k, verbose=True)
    if not success:
        print(f"\n⚠ FAILED at k_coupling = {k}")
        break

print("\n" + "="*70)
print("Diagnosis complete")
print("="*70)
