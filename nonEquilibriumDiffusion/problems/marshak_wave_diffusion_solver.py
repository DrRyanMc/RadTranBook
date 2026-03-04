#!/usr/bin/env python3
"""
Marshak Wave Problem using Generalized Diffusion Operator Solver

This implementation validates the DiffusionOperatorSolver1D by solving the
non-equilibrium radiation diffusion problem.

Problem setup:
- Left boundary: incoming flux from blackbody at 1 keV
- Material opacity: σ_R = σ_P = 300 * T^-3 (cm^-1, T in keV)
- Heat capacity: c_v = 0.3 GJ/(cm^3·keV)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from diffusion_operator_solver import DiffusionOperatorSolver1D

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³
CV_VOLUMETRIC = 0.3  # GJ/(cm^3·keV)

# =============================================================================
# MATERIAL PROPERTY FUNCTIONS FOR MARSHAK WAVE
# =============================================================================

def marshak_opacity(T):
    """Rosseland opacity: σ_R = 300 * T^-3"""
    n = 3
    T_min = 0.01  # Minimum temperature to prevent overflow (keV)
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)


def marshak_diffusion_coeff(T, r, *args):
    """
    Radiation diffusion coefficient: D = c / (3·σ_R)
    
    Follows the interface for DiffusionOperatorSolver1D:
    D(T, r) or D(T, r, phi_left, phi_right, dx)
    """
    sigma_R = marshak_opacity(T)
    return C_LIGHT / (3.0 * sigma_R)


def marshak_absorption_coeff(T, r):
    """Absorption coefficient: σ_a = σ_R (for simplified case)"""
    return marshak_opacity(T)


# =============================================================================
# MARSHAK WAVE SOLVER CLASS
# =============================================================================

class MarshakWaveSolver:
    """
    Non-equilibrium radiation diffusion solver using DiffusionOperatorSolver1D.
    
    Couples radiation energy density (φ) with material temperature (T).
    """
    
    def __init__(self, r_min=0.0, r_max=0.2, n_cells=50, dt=0.01):
        """
        Initialize the Marshak wave solver.
        
        Parameters:
        -----------
        r_min, r_max : float
            Domain boundaries (cm)
        n_cells : int
            Number of cells
        dt : float
            Initial time step (ns)
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_cells = n_cells
        self.dt = dt
        
        # Initialize radiation diffusion solver
        self.diffusion_solver = DiffusionOperatorSolver1D(
            r_min=r_min,
            r_max=r_max,
            n_cells=n_cells,
            geometry='planar',
            diffusion_coeff_func=marshak_diffusion_coeff,
            absorption_coeff_func=self._absorption_coeff_wrapper,
            dt=dt,
            left_bc_func=self._left_bc,
            right_bc_func=self._right_bc
        )
        
        # Mesh information
        self.r_centers = self.diffusion_solver.r_centers.copy()
        self.r_faces = self.diffusion_solver.r_faces.copy()
        
        # Initialize fields
        T_init = 0.01  # keV (cold material)
        phi_init = C_LIGHT * A_RAD * T_init**4
        
        self.T = np.full(n_cells, T_init)  # Material temperature
        self.phi = np.full(n_cells, phi_init)  # Radiation energy density
        
        # Temporary storage for effective absorption coefficient (used during solve)
        self._sigma_a_effective = None
        
        # Storage for history
        self.time_history = [0.0]
        self.T_history = [self.T.copy()]
        self.phi_history = [self.phi.copy()]
    
    def _left_bc(self, phi_boundary, r_boundary):
        """
        Left boundary: blackbody at T = 1.0 keV
        
        Returns (A, B, C) for Robin BC: A·φ + B·∇φ = C
        For Dirichlet: A=1, B=0, C=φ_bc
        """
        T_bc = 1.0  # keV
        phi_bc = C_LIGHT * A_RAD * T_bc**4
        return 1.0, 0.0, phi_bc
    
    def _right_bc(self, phi_boundary, r_boundary):
        """
        Right boundary: zero flux (reflecting)
        
        Returns (A, B, C) for Robin BC: A·φ + B·∇φ = C
        For Neumann zero flux: A=0, B=1, C=0
        """
        return 0.0, 1.0, 0.0
    
    def _absorption_coeff_wrapper(self, T, r):
        """
        Wrapper for absorption coefficient that uses effective value if available.
        
        During the solve step, we temporarily store σ_a_effective = θ·σ_P·f
        which includes the linearization factor f.
        """
        # Find the cell index corresponding to position r
        idx = np.argmin(np.abs(self.r_centers - r))
        
        if self._sigma_a_effective is not None:
            return self._sigma_a_effective[idx]
        else:
            # Fallback to standard absorption coefficient
            return marshak_absorption_coeff(T, r)
    
    def _compute_equilibrium_temperature(self, phi):
        """
        Compute radiation equilibrium temperature from φ.
        
        For equilibrium: φ = c·E_r = c·a·T_rad^4
        => T_rad = (φ / (c·a))^(1/4)
        """
        with np.errstate(invalid='ignore'):
            T_eq = np.power(phi / (C_LIGHT * A_RAD), 0.25)
        T_eq[phi <= 0] = 0.0
        return T_eq
    
    def time_step(self, verbose=True):
        """
        Execute one time step of the non-equilibrium radiation diffusion solver.
        
        Algorithm (matching oneDFV.py's newton_step approach):
        1. Solve radiation diffusion equation using DiffusionOperatorSolver
           Solves: (1/c·Δt + σ_a)·φ - ∇·D∇φ = φ^n/(c·Δt) + σ_a·φ_eq
        2. Update material temperature using linearized implicit method
           e_{n+1} = e_n + Δt·[f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt]
           where f = 1/(1 + β·σ_P·c·θ·Δt) and β = 4aT³/cv
           
        This provides implicit coupling between radiation and material energy.
        """
        n = self.n_cells
        
        # Use theta-method for time discretization (theta=1.0 for backward Euler)
        theta = 1.0
        
        # Use current state as linearization point T_star (simplified - no Newton iteration)
        T_star = self.T.copy()
        phi_star = self.phi.copy()
        
        # Material energy at current timestep
        e_star = RHO * CV_VOLUMETRIC * T_star
        
        # Evaluate coupling parameters at linearization point
        sigma_P = np.array([marshak_absorption_coeff(T_star[i], self.r_centers[i]) 
                           for i in range(n)])
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Compute f factor: f = 1 / (1 + β·σ_P·c·θ·Δt) where β = 4aT³/cv
        beta = 4.0 * A_RAD * T_star**3 / CV_VOLUMETRIC
        f = 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * self.dt)
        
        # The diffusion_solver expects σ_a to represent the coupling term
        # In oneDFV.py, the diagonal has: 1/(c·Δt) + θ·σ_P·f
        # So we need: σ_a = θ·σ_P·f
        sigma_a_effective = theta * sigma_P * f
        
        # Update diffusion_solver's dt (in case it changed)
        self.diffusion_solver.dt = self.dt
        
        # Construct RHS matching oneDFV.py's assemble_phi_equation
        # RHS: φ^n/(c·Δt) + σ_P·f·c·a·T★⁴ - (1-θ)·σ_P·f·φ^n
        # Note: The (1-f)·Δe/Δt term is zero when not using Newton iteration (Δe=0)
        rhs = (self.phi / (C_LIGHT * self.dt)) + sigma_P * f * acT4_star - (1.0 - theta) * sigma_P * f * self.phi
        
        # Store effective absorption coefficient for use in diffusion_solver
        # This allows the diffusion solver to use σ_a = θ·σ_P·f on the diagonal
        self._sigma_a_effective = sigma_a_effective
        
        # Solve: A·φ = b where A has diagonal: 1/(c·Δt) + σ_a_effective
        try:
            phi_new = self.diffusion_solver.solve(
                rhs=rhs,
                temperature=T_star,
                phi_guess=phi_star,
                use_iterative=False
            )
        finally:
            # Clear temporary storage
            self._sigma_a_effective = None
        
        # Update material temperature using linearized implicit approach from oneDFV.py
        # This follows solve_T_equation: e(T_{n+1}) = e(T_n) + Δt·[f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt]
        
        # Compute φ̃ = θ·φ^{n+1} + (1-θ)·φ^n
        phi_tilde = theta * phi_new + (1.0 - theta) * self.phi
        
        # Material energy at previous timestep
        e_prev = RHO * CV_VOLUMETRIC * self.T  # e = ρ·c_v·T
        
        # When not using Newton iteration, Δe = e_star - e_prev = 0 (T_star = T_prev)
        # So the (1-f)·Δe/Δt term vanishes
        
        # Update material energy: e_{n+1} = e_prev + Δt·f·σ_P·(φ̃ - acT★⁴)
        # Note: We already computed f, sigma_P, and acT4_star above
        e_new = e_prev + self.dt * f * sigma_P * (phi_tilde - acT4_star)
        
        # Convert back to temperature: T = e / (ρ·c_v)
        T_new = e_new / (RHO * CV_VOLUMETRIC)
        #T_new = np.maximum(T_new, 1e-10)  # Ensure positive temperature
        
        # Update state
        self.phi = phi_new
        self.T = T_new
        
        if verbose:
            T_rad = self._compute_equilibrium_temperature(phi_new)
            print(f"  Material T: min={self.T.min():.4e}, max={self.T.max():.4e}")
            print(f"  Radiation T_rad: min={T_rad.min():.4e}, max={T_rad.max():.4e}")
    
    def evolve_to_time(self, target_time, dt_initial=None, verbose=True):
        """
        Evolve the solution to a target time using adaptive time stepping.
        
        Parameters:
        -----------
        target_time : float
            Target simulation time (ns)
        dt_initial : float or None
            Initial time step. If None, uses self.dt
        verbose : bool
            Print progress information
            
        Returns:
        --------
        current_time : float
            Actual time reached (should equal target_time)
        """
        if dt_initial is not None:
            self.dt = dt_initial
        
        current_time = self.time_history[-1]
        
        if verbose:
            print(f"\nEvolving from t={current_time:.4f} ns to t={target_time:.4f} ns")
        
        step_count = 0
        while current_time < target_time:
            # Adjust dt for final step
            if current_time + self.dt > target_time:
                self.dt = target_time - current_time
            
            self.time_step(verbose=(step_count % 10 == 0 and verbose))
            current_time += self.dt
            step_count += 1
            
            # Store history
            self.time_history.append(current_time)
            self.T_history.append(self.T.copy())
            self.phi_history.append(self.phi.copy())
        
        if verbose:
            print(f"Completed {step_count} time steps")
            T_rad = self._compute_equilibrium_temperature(self.phi)
            print(f"Final material T: {self.T.max():.4f} keV")
            print(f"Final radiation T_rad: {T_rad.max():.4f} keV")
        
        return current_time
    
    def get_solution(self, time_index=-1):
        """
        Get solution snapshot at specified time index.
        
        Parameters:
        -----------
        time_index : int
            Index into time history (-1 for latest)
            
        Returns:
        --------
        dict with keys: time, r, T, phi, T_rad, Er
        """
        t = self.time_history[time_index]
        T = self.T_history[time_index]
        phi = self.phi_history[time_index]
        T_rad = self._compute_equilibrium_temperature(phi)
        Er = phi / C_LIGHT
        
        return {
            'time': t,
            'r': self.r_centers.copy(),
            'T': T,
            'phi': phi,
            'T_rad': T_rad,
            'Er': Er
        }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_material_properties():
    """Plot material property functions"""
    T_range = np.logspace(-2, 0.5, 200)
    sigma_R = marshak_opacity(T_range)
    D = C_LIGHT / (3.0 * sigma_R)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.loglog(T_range, sigma_R, 'b-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Opacity σ_R (cm⁻¹)', fontsize=12)
    ax.set_title('Rosseland Opacity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[1]
    ax.loglog(T_range, D, 'r-', linewidth=2)
    ax.set_xlabel('Temperature T (keV)', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient D (cm²/ns)', fontsize=12)
    ax.set_title('Radiation Diffusion Coefficient', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('marshak_material_properties_diffusion_solver.png', dpi=150, bbox_inches='tight')
    print("Saved: marshak_material_properties_diffusion_solver.png")


def plot_results(solver):
    """Plot temperature and energy density profiles"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(solver.time_history)))
    
    # Plot 1: Material temperature evolution
    ax = axes[0, 0]
    for i in [0, len(solver.T_history)//4, len(solver.T_history)//2, -1]:
        if i < len(solver.T_history):
            sol = solver.get_solution(i)
            ax.plot(sol['r'], sol['T'], 'o-', color=colors[i], 
                   label=f"t = {sol['time']:.2f} ns", markersize=3, linewidth=2)
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Material Temperature T (keV)', fontsize=12)
    ax.set_title('Material Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([solver.r_centers[0], solver.r_centers[-1]])
    
    # Plot 2: Radiation temperature evolution
    ax = axes[0, 1]
    for i in [0, len(solver.T_history)//4, len(solver.T_history)//2, -1]:
        if i < len(solver.T_history):
            sol = solver.get_solution(i)
            ax.plot(sol['r'], sol['T_rad'], 's-', color=colors[i], 
                   label=f"t = {sol['time']:.2f} ns", markersize=3, linewidth=2)
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Temperature T_rad (keV)', fontsize=12)
    ax.set_title('Radiation Temperature Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([solver.r_centers[0], solver.r_centers[-1]])
    
    # Plot 3: Final state - both temperatures
    ax = axes[1, 0]
    sol_final = solver.get_solution(-1)
    ax.plot(sol_final['r'], sol_final['T'], 'b-', linewidth=2.5, label='Material T')
    ax.plot(sol_final['r'], sol_final['T_rad'], 'r--', linewidth=2.5, label='Radiation T_rad')
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Temperature (keV)', fontsize=12)
    ax.set_title(f'Temperature Comparison at t = {sol_final["time"]:.2f} ns', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([solver.r_centers[0], solver.r_centers[-1]])
    
    # Plot 4: Radiation energy density
    ax = axes[1, 1]
    for i in [0, len(solver.phi_history)//4, len(solver.phi_history)//2, -1]:
        if i < len(solver.phi_history):
            sol = solver.get_solution(i)
            ax.plot(sol['r'], sol['Er'], 'o-', color=colors[i], 
                   label=f"t = {sol['time']:.2f} ns", markersize=3, linewidth=2)
    ax.set_xlabel('Position r (cm)', fontsize=12)
    ax.set_ylabel('Radiation Energy Density E_r (GJ/cm³)', fontsize=12)
    ax.set_title('Radiation Energy Density Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([solver.r_centers[0], solver.r_centers[-1]])
    
    plt.tight_layout()
    plt.savefig('marshak_wave_diffusion_solver.png', dpi=150, bbox_inches='tight')
    print("Saved: marshak_wave_diffusion_solver.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Marshak wave simulation with diffusion operator solver"""
    
    print("="*80)
    print("Marshak Wave Problem - Non-Equilibrium Radiation Diffusion")
    print("Using Generalized DiffusionOperatorSolver1D")
    print("="*80)
    print("\nMaterial Properties:")
    print("  Opacity: σ_R = 300 * T^-3 (cm^-1, T in keV)")
    print("  Heat capacity: c_v = 0.3 GJ/(cm^3·keV)")
    print("  Diffusion coefficient: D = c / (3·σ_R)")
    print("\nBoundary Conditions:")
    print("  Left boundary: Dirichlet, T = 1.0 keV blackbody")
    print("  Right boundary: Neumann, zero flux")
    print("="*80)
    
    # Create solver
    dt = 0.001
    solver = MarshakWaveSolver(r_min=0.0, r_max=0.2, n_cells=50, dt=dt)
    
    print(f"\nInitialized solver with {solver.n_cells} cells")
    print(f"Domain: r ∈ [{solver.r_min}, {solver.r_max}] cm")
    print(f"Initial step size: Δt = {solver.dt} ns")
    
    # Plot material properties
    print("\nPlotting material properties...")
    plot_material_properties()
    
    # Evolve to specified times
    target_times = [1.0] #[1.0, 10.0, 20.0]
    
    try:
        for target_time in target_times:
            current_time = solver.time_history[-1]
            if current_time < target_time:
                solver.evolve_to_time(target_time, verbose=True)
    except Exception as e:
        print(f"\nWarning: Simulation interrupted with error: {e}")
        print(f"Continuing with available data ({len(solver.time_history)} time steps)")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(solver)
    
    print("\n" + "="*80)
    print("Simulation completed!")
    print(f"Total time steps: {len(solver.time_history)}")
    print(f"Final simulation time: {solver.time_history[-1]:.4f} ns")
    print("="*80)
    
    return solver


if __name__ == "__main__":
    solver = main()
