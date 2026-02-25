#!/usr/bin/env python3
"""
Zeldovich Wave Problem - Non-Equilibrium Radiation Diffusion
Radiative heat wave with initial energy pulse in first cell

Problem setup:
- Left boundary: reflecting (zero flux)
- Right boundary: reflecting (zero flux)
- Initial condition: cold everywhere except first cell with energy pulse
- Material opacity: σ_R = σ_P = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 3e-6 GJ/(cm^3·keV) [much smaller than Marshak wave!]
- Supports planar, cylindrical, and spherical geometries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import (
    NonEquilibriumRadiationDiffusionSolver,
    A_RAD, 
    C_LIGHT,
    RHO
)

# Add path to import analytical solution
try:
    # Try parent directory first (nonEquilibriumDiffusion/)
    parent_dir = str(Path(__file__).resolve().parent.parent)
    eq_utils_dir = str(Path(__file__).resolve().parent.parent.parent / "EqDiffusion" / "utils")
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, eq_utils_dir)
    from zeldovich import T_of_r_t
    ANALYTICAL_AVAILABLE = True
except ImportError as e:
    try:
        # Fall back to Overleaf directory
        sys.path.insert(0, str(Path.home() / "Dropbox/Apps/Overleaf/RadTranBook/img/equilibriumDiffusion"))
        from zeldovich import T_of_r_t
        ANALYTICAL_AVAILABLE = True
    except ImportError as e2:
        print("Warning: Analytical solution module not found. Will plot numerical only.")
        ANALYTICAL_AVAILABLE = False
except Exception as e:
    print(f"Error importing zeldovich: {e}")
    ANALYTICAL_AVAILABLE = False

# =============================================================================
# ZELDOVICH WAVE MATERIAL PROPERTIES
# =============================================================================

def zeldovich_rosseland_opacity(T):
    """Temperature-dependent Rosseland opacity: σ_R = 300 * T^-3
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    sigma_R : float
        Rosseland opacity (cm^-1)
    """
    n = 3
    T_min = 0.001  # Minimum temperature to prevent overflow (keV) - must be below initial T!
    T_safe = max(T, T_min)
    return 300.0 * T_safe**(-n)


def zeldovich_planck_opacity(T):
    """Temperature-dependent Planck opacity: σ_P = 300 * T^-3
    
    Same as Rosseland for this problem.
    """
    return zeldovich_rosseland_opacity(T)


def zeldovich_specific_heat(T):
    """Specific heat: c_v = 3e-6 GJ/(cm^3·keV)
    
    Note: Much smaller than Marshak wave (0.3 vs 3e-6)
    This is volumetric heat capacity divided by density.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    cv : float
        Specific heat capacity per unit mass (GJ/(g·keV))
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return cv_volumetric / RHO  # GJ/(g·keV)


def zeldovich_material_energy(T):
    """Material energy density: e = ρ·c_v·T (volumetric)
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    
    Returns:
    --------
    e : float
        Material energy density (GJ/cm^3)
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return RHO * cv_volumetric * T


def zeldovich_inverse_material_energy(e):
    """Inverse: T from e = ρ·c_v·T → T = e/(ρ·c_v)
    
    Parameters:
    -----------
    e : float
        Material energy density (GJ/cm^3)
    
    Returns:
    --------
    T : float
        Temperature (keV)
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    return e / (RHO * cv_volumetric)


def equilibrium_temperature(E_total):
    """Solve for equilibrium temperature given total energy density
    
    In equilibrium: E_total = a·T^4 + ρ·c_v·T
    where the radiation energy density is φ/c = a·T^4
    
    This is solved iteratively using Newton's method:
    f(T) = a·T^4 + ρ·c_v·T - E_total = 0
    f'(T) = 4·a·T^3 + ρ·c_v
    
    Parameters:
    -----------
    E_total : float
        Total energy density (GJ/cm^3)
    
    Returns:
    --------
    T_eq : float
        Equilibrium temperature (keV)
    """
    cv_volumetric = 3e-6  # GJ/(cm^3·keV)
    
    # Initial guess: radiation-dominated approximation T ≈ (E/a)^(1/4)
    T = (E_total / A_RAD)**0.25
    
    # Newton iterations
    for _ in range(20):
        f = A_RAD * T**4 + RHO * cv_volumetric * T - E_total
        df_dT = 4.0 * A_RAD * T**3 + RHO * cv_volumetric
        
        T_new = T - f / df_dT
        
        # Check convergence
        if abs(T_new - T) / (T + 1e-10) < 1e-10:
            return T_new
        
        T = max(T_new, 1e-6)  # Prevent negative temperature
    
    return T


# =============================================================================
# ZELDOVICH WAVE BOUNDARY CONDITIONS
# =============================================================================

def zeldovich_left_bc(phi, x):
    """Left boundary: reflecting (zero flux)
    
    Use zero flux condition: dφ/dx = 0
    This is Robin BC: A·φ + B·(dφ/dx) = C
    For zero flux: A=0, B=1, C=0
    """
    return 0.0, 1.0, 0.0  # A, B, C


def zeldovich_right_bc(phi, x):
    """Right boundary: reflecting (zero flux)
    
    Use zero flux condition: dφ/dx = 0
    """
    return 1.0, 0.0, 0.0  # A, B, C


# =============================================================================
# ZELDOVICH WAVE SIMULATION
# =============================================================================

def run_zeldovich_wave(d=0, use_trbdf2=False, n_cells=100):
    """Run Zeldovich wave simulation and plot results at specified times
    
    Parameters:
    -----------
    d : int
        Geometry dimension (0=planar, 1=cylindrical, 2=spherical)
    use_trbdf2 : bool
        If True, use TR-BDF2 time integration; otherwise Backward Euler
    n_cells : int
        Number of cells in mesh
    
    Returns:
    --------
    solutions : list of tuples
        List of (time, r, phi, T) solutions
    solver : NonEquilibriumRadiationDiffusionSolver
        The solver object
    """
    
    N = d + 1  # Spatial dimension for analytical solution
    
    print("="*80)
    print("Zeldovich Wave Problem - Non-Equilibrium Radiation Diffusion")
    print("="*80)
    print("Material properties:")
    print("  Opacity: σ_R = σ_P = 300 * T^-3 (cm^-1)")
    print("  Heat capacity: c_v = 3e-6 GJ/(cm^3·keV)")
    print("  Left BC: Reflecting (zero flux)")
    print("  Right BC: Reflecting (zero flux)")
    print("  Initial condition: Energy pulse in first few cells (equilibrium φ-T split)")
    geometry_names = ['Planar', 'Cylindrical', 'Spherical']
    print(f"  Geometry: {geometry_names[d]} (d={d})")
    if use_trbdf2:
        print("  Time integration: TR-BDF2")
    else:
        print("  Time integration: Backward Euler")
    print("="*80)
    
    # Problem setup
    r_min = 0.0    # cm
    r_max = 3.0    # cm
    
    # Time stepping parameters
    # Start with very small dt to handle steep initial gradients
    dt_initial = 1e-8  # ns (very small to handle steep gradients)
    dt_max = 1e-3      # ns (maximum timestep once solution is smooth)
    dt = dt_initial
    # Target times for solution output (these will be physical times = t_numerical + t_init)
    target_times = [0.1, 0.3, 1.0, 3.0]  # ns (physical times)
    
    # Create solver
    print(f"\nInitializing solver with {n_cells} cells...")
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=r_min, 
        r_max=r_max, 
        n_cells=n_cells, 
        d=d,
        dt=dt,
        max_newton_iter=100,
        newton_tol=1e-8,
        rosseland_opacity_func=zeldovich_rosseland_opacity,
        planck_opacity_func=zeldovich_planck_opacity,
        specific_heat_func=zeldovich_specific_heat,
        material_energy_func=zeldovich_material_energy,
        inverse_material_energy_func=zeldovich_inverse_material_energy,
        left_bc_func=zeldovich_left_bc,
        right_bc_func=zeldovich_right_bc,
        theta=1.0  # Backward Euler for stability
    )
    
    # Initial condition: Use self-similar solution at early time
    # This provides a smooth initial condition that matches the analytical solution
    t_init = 0.01  # ns - initial time for self-similar solution
    
    print(f"\nInitializing from self-similar solution at t = {t_init} ns")
    
    T_init = np.zeros(n_cells)
    phi_init = np.zeros(n_cells)
    use_analytical = ANALYTICAL_AVAILABLE  # Local flag
    
    # Get analytical solution at t_init
    if use_analytical:
        try:
            T_analytical, R_front = T_of_r_t(solver.r_centers, t_init, N=N)
            # Apply minimum temperature floor to avoid zero φ
            T_min_floor = 0.01  # keV
            T_init = np.maximum(T_analytical, T_min_floor)
            # Set φ in equilibrium with T: φ = acT^4
            phi_init = A_RAD * C_LIGHT * T_init**4
            print(f"  Successfully initialized from analytical solution")
            print(f"  Wave front position at t_init: R_front = {R_front:.4f} cm")
        except Exception as e:
            print(f"  Warning: Could not use analytical solution: {e}")
            print(f"  Falling back to delta function IC")
            use_analytical = False
    
    if not use_analytical:
        # Fallback: use delta function if analytical not available
        T_cold = 0.01
        T_init = np.full(n_cells, T_cold)
        phi_init = A_RAD * C_LIGHT * T_init**4
        T_init[0] = 5.9
        phi_init[0] = A_RAD * C_LIGHT * T_init[0]**4
        t_init = 0.0  # No time offset
    
    solver.set_initial_condition(phi_init=phi_init, T_init=T_init)
    
    # Print initial condition info
    r, phi, T = solver.get_solution()
    print(f"\nInitial condition at t = {t_init} ns:")
    dx = (r_max - r_min) / n_cells
    print(f"  Cell width: dx = {dx:.4f} cm")
    print(f"  First cell: T = {T[0]:.4f} keV, φ = {phi[0]:.4e} GJ/cm³")
    print(f"  Max T: {T.max():.4f} keV at r = {r[np.argmax(T)]:.4f} cm")
    print(f"  Background: T = {T[-1]:.4f} keV, φ = {phi[-1]:.4e} GJ/cm³")
    
    # Verify equilibrium in first cell
    E_rad_0 = phi[0] / C_LIGHT
    E_mat_0 = zeldovich_material_energy(T[0])
    E_total_0 = E_rad_0 + E_mat_0
    print(f"  First cell energy split: E_rad = {E_rad_0:.4e}, E_mat = {E_mat_0:.4e}, ratio = {E_rad_0/E_mat_0:.2e}")
    
    # Compute initial total energy
    E_rad = np.sum(phi * solver.V_cells)
    E_mat = np.sum(zeldovich_material_energy(T) * solver.V_cells)
    E_total_init = E_rad + E_mat
    print(f"\nInitial energy:")
    print(f"  Radiation: {E_rad:.6e} GJ")
    print(f"  Material:  {E_mat:.6e} GJ")
    print(f"  Total:     {E_total_init:.6e} GJ")
    
    # Time evolution
    print(f"\nTime evolution (numerical time starts at t_init = {t_init} ns):")
    current_time = 0.0  # Numerical solver time (starts at 0)
    solutions = []
    step_count = 0
    
    #plot initial conditions for debugging
    plt.figure(figsize=(8, 6))
    plt.plot(r, T, label='Temperature (keV)')
    plt.plot(r, phi, label='Radiation Energy Density (GJ/cm³)')
    plt.xlabel('Position (cm)')
    plt.ylabel('Values')
    plt.title('Initial Conditions')
    plt.legend()
    #save as png
    plt.savefig('initial_conditions.png')

    #plot initial sigma_R
    sigma_R_init = [zeldovich_rosseland_opacity(Ti) for Ti in T]
    plt.figure(figsize=(8, 6))
    plt.plot(r, sigma_R_init, label='Rosseland Opacity (cm⁻¹)')
    plt.xlabel('Position (cm)')
    plt.ylabel('σ_R (cm⁻¹)')
    plt.title('Initial Rosseland Opacity')
    plt.legend()
    plt.savefig('initial_sigma_R.png')

    # Convert physical target times to numerical solver times
    for target_time_physical in target_times:
        target_time_numerical = target_time_physical - t_init
        
        # Skip if target is at or before initial time
        if target_time_numerical <= 0:
            print(f"  Skipping t = {target_time_physical:.1f} ns (initial condition)")
            continue
        
        while current_time < target_time_numerical:
            # Adjust time step if needed to hit target exactly
            if current_time + dt > target_time_numerical:
                dt = target_time_numerical - current_time
            
            # Set solver's dt
            solver.dt = dt
            
            # Store old values in case Newton fails badly
            phi_old = solver.phi.copy()
            T_old = solver.T.copy()
            
            # Time stepping
            verbose_step = (step_count < 5)  # Verbose for first few steps
            
            if use_trbdf2 and step_count > 0:
                # TR-BDF2: need at least one BE step first
                solver.time_step_trbdf2(n_steps=1, verbose=verbose_step)
            else:
                # Backward Euler (always for first step)
                solver.time_step(n_steps=1, verbose=verbose_step)
            
            # Check solution quality
            phi_change = np.max(np.abs(solver.phi - phi_old) / (np.abs(phi_old) + 1e-10))
            T_change = np.max(np.abs(solver.T - T_old) / (np.abs(T_old) + 1e-10))
            
            # Adaptive timestep control
            # If changes are small and reasonable, can increase dt
            if phi_change < 0.1 and T_change < 0.1 and dt < dt_max:
                dt_new = min(dt * 1.5, dt_max)
                if step_count % 100 == 0 and dt_new > dt:
                    print(f"  Step {step_count}: t={current_time:.6e} ns, increasing dt {dt:.2e} → {dt_new:.2e} ns")
                dt = dt_new
            elif phi_change > 0.5 or T_change > 0.5:
                # Changes too large - reduce dt next time
                dt = max(dt * 0.5, dt_initial)
                if verbose_step or step_count % 10 == 0:
                    print(f"  Step {step_count}: Large changes (φ:{phi_change:.2f}, T:{T_change:.2f}), reducing dt to {dt:.2e} ns")
            
            current_time += solver.dt
            step_count += 1
        
        # Store solution at target time (with time offset for physical time)
        r, phi, T = solver.get_solution()
        physical_time = current_time + t_init
        solutions.append((physical_time, r.copy(), phi.copy(), T.copy()))
        
        # Debug: print first few cells after timestep
        if step_count <= 2:
            print(f"\n  After timestep {step_count}, first 20 cells:")
            for i in range(min(20, len(T))):
                print(f"    Cell {i}: T = {T[i]:.4f} keV, φ = {phi[i]:.4e} GJ/cm³")
        
        # Check energy conservation
        E_rad = np.sum(phi * solver.V_cells)
        E_mat = np.sum(zeldovich_material_energy(T) * solver.V_cells)
        E_total = E_rad + E_mat
        E_error = abs(E_total - E_total_init) / E_total_init * 100
        
        print(f"  t = {physical_time:.1f} ns (t_numerical = {current_time:.1f} ns):")
        print(f"    max T = {T.max():.4f} keV, max φ = {phi.max():.4e} GJ/cm³")
        print(f"    Total energy = {E_total:.6e} GJ (error: {E_error:.2e}%)")
    
    return solutions, solver


def plot_zeldovich_wave(solutions, d=0, include_analytical=True):
    """Plot Zeldovich wave solutions comparing numerical and analytical
    
    Parameters:
    -----------
    solutions : list of tuples
        List of (time, r, phi, T) numerical solutions
    d : int
        Geometry dimension (0=planar, 1=cylindrical, 2=spherical)
    include_analytical : bool
        If True, overlay analytical self-similar solution
    """
    
    N = d + 1  # Spatial dimension for analytical solution
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8/1.518))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot temperature profiles
    for i, (t, r, phi, T) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # Numerical solution
        ax.plot(r, T, color=color, linewidth=2, linestyle='-', 
                label=f't = {t:.1f} ns (Numerical)', marker='o', markersize=3, markevery=10)
        ax.plot(r,(phi/(A_RAD*C_LIGHT))**0.25, color=color, linewidth=2, linestyle=':', alpha=0.5,label=f't = {t:.1f} ns (φ)')
        
        # Analytical solution
        if include_analytical and ANALYTICAL_AVAILABLE:
            try:
                T_analytical, R_front = T_of_r_t(r, t, N=N)
                ax.plot(r, T_analytical, color=color, linewidth=1.5, linestyle='--', 
                        alpha=0.7, label=f't = {t:.1f} ns (Analytical)')
                # Mark wave front
                ax.axvline(R_front, color=color, linestyle=':', alpha=0.3, linewidth=1)
            except Exception as e:
                print(f"Warning: Could not compute analytical solution at t={t}: {e}")
    
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature T (keV)', fontsize=12)
    geometry_names = ['Planar', 'Cylindrical', 'Spherical']
    ax.set_title(f'Zeldovich Wave: {geometry_names[d]} Geometry (d={d})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(0, solutions[-1][1][-1])
    
    plt.tight_layout()
    filename = f'zeldovich_wave_noneq_d{d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{filename}'")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Run Zeldovich wave simulation
    print("\nRunning Zeldovich wave simulation...")
    
    # Choose geometry: 0=planar, 1=cylindrical, 2=spherical
    d = 2
    use_trbdf2 = False  # Use Backward Euler for stability
    
    solutions, solver = run_zeldovich_wave(d=d, use_trbdf2=use_trbdf2, n_cells=100)
    
    # Plot results
    print("\nPlotting Zeldovich wave results...")
    plot_zeldovich_wave(solutions, d=d, include_analytical=True)
    
    print("\n" + "="*80)
    print("Zeldovich wave simulation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
