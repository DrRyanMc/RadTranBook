"""
Test Marshak wave boundary behavior with nonlinear corrections
Focus on the first few cells to understand the boundary interface
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from oneDFV import RadiationDiffusionSolver

# Physical constants
A_RAD = 0.01372  # GJ/(cm^3·keV^4)
SPEED_OF_LIGHT = 29.9792458  # cm/ns

def marshak_opacity(T):
    """σ_R = 300 * T^-3"""
    return 300.0 * T**(-3.0)

def marshak_specific_heat(T):
    """c_v = 0.3 GJ/(cm^3·keV)"""
    return 0.3  # constant

def marshak_material_energy(T):
    """U_m = c_v * T"""
    return marshak_specific_heat(T) * T

def temperature_from_Er(Er):
    """Compute temperature from radiation energy density"""
    return (Er / A_RAD)**0.25

def marshak_left_bc(Er, x):
    """Left boundary: Dirich let at T_bc = 1 keV"""
    T_bc = 1.0  # keV
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc  # Dirichlet: E_r = Er_bc

def marshak_right_bc(Er, x):
    """Right boundary: zero flux"""
    return 0.0, 1.0, 0.0  # dE_r/dx = 0


def test_boundary_region():
    """Test with just a few cells to diagnose boundary interface"""
    
    print("="*60)
    print("Marshak Boundary Test - Focus on First Few Cells")
    print("="*60)
    
    # Small domain with few cells
    r_min = 0.0
    r_max = 0.05  # cm (very small)
    n_cells = 10   # Few cells to see detail
    
    dt = 0.001  # ns (small time step)
    
    # Create solver
    solver = RadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=0,  # Planar
        dt=dt,
        max_newton_iter=50,
        newton_tol=1e-6,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_specific_heat,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    # Enable nonlinear corrections with Dirichlet BC (now using numerical flux)
    solver.use_nonlinear_correction = True
    solver.use_secant_derivative = False
    solver.nonlinear_skip_boundary_cells = 0  # No need to skip
    
    # Initial condition: cold material
    def initial_Er(r):
        T_init = 0.1  # keV
        return np.full_like(r, A_RAD * T_init**4)
    
    solver.set_initial_condition(initial_Er)
    
    # Take just one time step
    print(f"\nInitial state:")
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    print(f"Cell centers (cm): {r[:5]}")
    print(f"T (keV):           {T[:5]}")
    print(f"E_r (GJ/cm^3):     {Er[:5]}")
    
    print(f"\nTaking one time step (dt = {dt:.4f} ns)...")
    
    # Check what the boundary condition returns
    r, Er = solver.get_solution()
    A_bc, B_bc, C_bc = marshak_left_bc(Er[0], solver.r_faces[0])
    print(f"  Boundary condition: A={A_bc:.2f}, B={B_bc:.2e}, C={C_bc:.4e}")
    if abs(B_bc) < 1e-14:
        Er_bc = C_bc / A_bc
        T_bc_expected = (Er_bc / A_RAD)**0.25
        print(f"  Dirichlet BC: Er_boundary = {Er_bc:.4e}, T_boundary = {T_bc_expected:.4f} keV")
    
    solver.time_step(n_steps=1, verbose=True)
    
    r, Er = solver.get_solution()
    T = temperature_from_Er(Er)
    print(f"\nAfter one time step:")
    print(f"Cell centers (cm): {r[:5]}")
    print(f"T (keV):           {T[:5]}")
    print(f"E_r (GJ/cm^3):     {Er[:5]}")
    
    # Check if there's a spike at cell 1 or 2
    if T[1] > 1.5 * T[2] or T[2] > 1.5 * T[3]:
        print(f"\n⚠ WARNING: Possible spike detected!")
        print(f"  T[0] = {T[0]:.4f}")
        print(f"  T[1] = {T[1]:.4f}")
        print(f"  T[2] = {T[2]:.4f}")
        print(f"  T[3] = {T[3]:.4f}")
    
    # Take a few more steps
    print(f"\nTaking 10 more time steps...")
    for i in range(10):
        solver.time_step(n_steps=1, verbose=False)
        r, Er = solver.get_solution()
        T = temperature_from_Er(Er)
        print(f"Step {i+2}: T[0:5] = {T[:5]}")
    
    # Final plot
    plt.figure(figsize=(10, 6))
    plt.plot(r, T, 'o-', label='Final T profile')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Boundary T')
    plt.xlabel('Position r (cm)')
    plt.ylabel('Temperature T (keV)')
    plt.title('Marshak Wave - Boundary Region Test')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_marshak_boundary.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'test_marshak_boundary.png'")
    plt.show()


if __name__ == "__main__":
    test_boundary_region()
