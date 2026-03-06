#!/usr/bin/env python3
"""
2D Multigroup Non-Equilibrium Diffusion Solver

Extends the 1D multigroup solver to 2D geometry. Solves the multigroup 
non-equilibrium diffusion system (equations 8.83-8.91) in 2D:

For each group g:
    A_g φ_g^{n+1} = χ_g(1-f)κ + ξ_g + Q_g

where:
    A_g = -∇·D_g(T_★)∇ + (σ*_{a,g} + 1/(cΔt))
    κ = Σ_g σ*_{a,g} φ_g^{n+1}  (absorption rate density)
    Q_g = external radiation source (optional)

Algorithm:
1. Solve Bκ = RHS for κ using GMRES
   where B = I - Σ_g σ*_{a,g} A_g^{-1} χ_g(1-f)
   and RHS = Σ_g σ*_{a,g} A_g^{-1} ξ_g

2. Compute φ_g^{n+1} = A_g^{-1}(χ_g(1-f)κ + ξ_g) for each group

3. Update material temperature using:
   e(T_{n+1}) - e(T_n) = Δt·f·Σ_g σ*_{a,g}(φ_g^{n+1} - 4πB_g(T_★)) + (1-f)Δe

4. Iterate until convergence

LMFG Preconditioning:
---------------------
The Linear Multifrequency Gray (LMFG) preconditioner is extended to 2D:
    C = I + ⟨σ_a⟩ · H^{-1} · (1-f)

where H is the 2D gray operator:
    H = -∇·⟨D⟩∇ + ⟨σ_a⟩(1-f) + 1/(c·Δt)

The gray weights λ̃_g are computed using temperature-dependent Rosseland integrals.
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from typing import Callable, Optional, List, Tuple, Dict
import warnings

# Import the 2D diffusion operator solver
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from diffusion_operator_solver import (
        DiffusionOperatorSolver2D,
        C_LIGHT,
        A_RAD
    )
except ImportError as e:
    print(f"Warning: Could not import from diffusion_operator_solver: {e}")
    C_LIGHT = 2.99792458e1  # cm/ns
    A_RAD = 0.01372       # GJ/(cm³·keV⁴)
    raise ImportError("diffusion_operator_solver module is required but not found.")

# Import flux limiter functions
def flux_limiter_standard(R):
    """Standard diffusion: λ = 1/3"""
    if isinstance(R, np.ndarray):
        return np.full_like(R, 1.0/3.0)
    return 1.0/3.0

def flux_limiter_levermore_pomraning(R):
    """Levermore-Pomraning flux limiter: λ = (2+R)/(6+3R+R²)"""
    R = np.atleast_1d(R)
    result = (2.0 + R) / (6.0 + 3.0*R + R**2)
    return result if len(result) > 1 else result[0]

def flux_limiter_larsen(R, n=2.0):
    """Larsen flux limiter: λ = (3^n + R^n)^(-1/n)"""
    return (3.0**n + R**n)**(-1.0/n)

def flux_limiter_sum(R):
    """Sum flux limiter: λ = 1/(3+R)"""
    return 1.0 / (3.0 + R)

def flux_limiter_max(R):
    """Max flux limiter: λ = 1/max(3, R)"""
    if isinstance(R, np.ndarray):
        return 1.0 / np.maximum(3.0, R)
    return 1.0 / max(3.0, R)

# Import Planck integral library
try:
    from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup
except ImportError:
    print("Warning: planck_integrals module not found. Using gray approximations.")
    def Bg(E_low, E_high, T):
        """Gray approximation: B = σT⁴/(4π)"""
        return (C_LIGHT * A_RAD / 4) * T**4 / (4 * np.pi)
    
    def dBgdT(E_low, E_high, T):
        """Gray approximation: dB/dT = 4σT³/(4π)"""
        return (C_LIGHT * A_RAD) * T**3 / (4 * np.pi)
    
    def Bg_multigroup(energy_edges, T):
        """Gray approximation for all groups"""
        n_groups = len(energy_edges) - 1
        total_B = (C_LIGHT * A_RAD / 4) * T**4 / (4 * np.pi)
        return np.full(n_groups, total_B / n_groups)
    
    def dBgdT_multigroup(energy_edges, T):
        """Gray approximation derivatives for all groups"""
        n_groups = len(energy_edges) - 1
        total_dB = (C_LIGHT * A_RAD) * T**3 / (4 * np.pi)
        return np.full(n_groups, total_dB / n_groups)

SIGMA_SB = C_LIGHT * A_RAD / 4  # Stefan-Boltzmann constant


# =============================================================================
# HELPER FUNCTION FOR 2D FLATTENING
# =============================================================================

def flatten_2d(array_2d, nx, ny):
    """Flatten 2D array (nx, ny) to 1D array (n_total) using row-major (C) order"""
    return array_2d.flatten(order='C')


def unflatten_2d(array_1d, nx, ny):
    """Unflatten 1D array (n_total) to 2D array (nx, ny) using row-major (C) order"""
    return array_1d.reshape((nx, ny), order='C')


# =============================================================================
# MULTIGROUP 2D DIFFUSION SOLVER
# =============================================================================

class MultigroupDiffusionSolver2D:
    """
    2D Multigroup Non-Equilibrium Diffusion Solver.
    
    Solves the coupled system using the reduction to absorption rate density κ
    as described in equations 8.83-8.91, extended to 2D geometry.
    """
    
    def __init__(self,
                 n_groups: int,
                 x_min: float,
                 x_max: float,
                 nx_cells: int,
                 y_min: float,
                 y_max: float,
                 ny_cells: int,
                 energy_edges: np.ndarray,
                 geometry: str = 'cartesian',
                 dt: float = 1e-3,
                 diffusion_coeff_funcs: Optional[List[Callable]] = None,
                 absorption_coeff_funcs: Optional[List[Callable]] = None,
                 rosseland_opacity_funcs: Optional[List[Callable]] = None,
                 planck_opacity_funcs: Optional[List[Callable]] = None,
                 emission_fractions: Optional[np.ndarray] = None,
                 planck_funcs: Optional[List[Callable]] = None,
                 dplanck_dT_funcs: Optional[List[Callable]] = None,
                 material_energy_func: Optional[Callable] = None,
                 inverse_material_energy_func: Optional[Callable] = None,
                 specific_heat_func: Optional[Callable] = None,
                 boundary_funcs: Optional[Dict] = None,
                 source_funcs: Optional[List[Callable]] = None,
                 flux_limiter_funcs: Optional[List[Callable]] = None,
                 rho: float = 1.0,
                 cv: float = 1.0,
                 x_stretch: float = 1.0,
                 y_stretch: float = 1.0,
                 max_newton_iter: int = 10,
                 newton_tol: float = 1e-8,
                 theta: float = 1.0):
        """
        Initialize 2D multigroup solver.
        
        Parameters:
        -----------
        n_groups : int
            Number of energy groups G
        x_min, x_max : float
            Domain boundaries in first coordinate
        nx_cells : int
            Number of cells in first coordinate
        y_min, y_max : float
            Domain boundaries in second coordinate
        ny_cells : int
            Number of cells in second coordinate
        energy_edges : ndarray
            Energy group edges (length n_groups + 1), in keV
        geometry : str
            'cartesian' for (x,y) or 'cylindrical' for (r,z)
        dt : float
            Time step
        diffusion_coeff_funcs : list of callable or None
            List of G functions D_g(T, x, y) for each group
        absorption_coeff_funcs : list of callable or None
            List of G functions σ_{a,g}(T, x, y) for each group
        rosseland_opacity_funcs : list of callable or None
            List of G Rosseland opacity functions σ_R,g(T, x, y)
        planck_opacity_funcs : list of callable or None
            List of G Planck opacity functions σ_P,g(T, x, y)
        emission_fractions : ndarray or None
            Manual override of emission fractions χ_g (length n_groups)
        planck_funcs : list of callable or None
            User-defined Planck functions B_g(T) for each group
        dplanck_dT_funcs : list of callable or None
            User-defined temperature derivatives dB_g/dT(T)
        material_energy_func : callable or None
            Function e(T, x, y) returning material energy density
        inverse_material_energy_func : callable or None
            Function T(e, x, y) inverse of material energy
        specific_heat_func : callable or None
            Function c_v(T, x, y) for specific heat
        boundary_funcs : dict or None
            Boundary condition functions for each side
        source_funcs : list of callable or None
            External source functions Q_g(x, y, t)
        flux_limiter_funcs : list of callable or None
            Flux limiter functions λ(R) for each group
        rho : float
            Material density (g/cm³)
        cv : float
            Specific heat (GJ/(g·keV)) - only used if specific_heat_func is None
        x_stretch, y_stretch : float
            Grid stretching factors
        max_newton_iter : int
            Maximum Newton iterations per time step
        newton_tol : float
            Newton convergence tolerance
        theta : float
            Time discretization parameter (1.0 = implicit Euler)
        """
        self.n_groups = n_groups
        self.x_min = x_min
        self.x_max = x_max
        self.nx_cells = nx_cells
        self.y_min = y_min
        self.y_max = y_max
        self.ny_cells = ny_cells
        self.n_total = nx_cells * ny_cells
        self.geometry = geometry
        self.dt = dt
        self.rho = rho
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.theta = theta
        
        # Store cv as either scalar or function
        if callable(cv):
            self.cv_func = cv
            self.cv_is_function = True
        else:
            self.cv = cv
            self.cv_is_function = False
        
        self.t = 0.0  # Current time
        
        # Fleck factor array (computed locally for each cell)
        self.fleck_factor = np.ones(self.n_total, dtype=np.float64)
        
        # Store energy group edges
        self.energy_edges = np.array(energy_edges, dtype=np.float64)
        if len(self.energy_edges) != n_groups + 1:
            raise ValueError(f"energy_edges must have length n_groups+1 = {n_groups+1}, "
                           f"got {len(self.energy_edges)}")
        
        # Default coefficient functions if not provided
        if diffusion_coeff_funcs is None:
            diffusion_coeff_funcs = [lambda T, x, y: 1.0] * n_groups
        
        if absorption_coeff_funcs is None:
            absorption_coeff_funcs = [lambda T, x, y: 1.0] * n_groups
        
        if rosseland_opacity_funcs is None:
            rosseland_opacity_funcs = absorption_coeff_funcs
        
        if planck_opacity_funcs is None:
            planck_opacity_funcs = absorption_coeff_funcs
        
        # External source functions Q_g(x, y, t)
        if source_funcs is None:
            source_funcs = [lambda x, y, t: 0.0] * n_groups
        self.source_funcs = source_funcs
        
        # Store functions
        self.diffusion_coeff_funcs = diffusion_coeff_funcs
        self.absorption_coeff_funcs = absorption_coeff_funcs
        self.rosseland_opacity_funcs = rosseland_opacity_funcs
        self.planck_opacity_funcs = planck_opacity_funcs
        
        # Flux limiters
        if flux_limiter_funcs is None:
            self.flux_limiter_funcs = None
            self.use_flux_limiters = False
        else:
            self.use_flux_limiters = True
            if callable(flux_limiter_funcs):
                self.flux_limiter_funcs = [flux_limiter_funcs] * n_groups
            else:
                self.flux_limiter_funcs = flux_limiter_funcs
        
        # Material property functions
        if specific_heat_func is None:
            self.specific_heat_func = lambda T, x, y: self.cv
        else:
            self.specific_heat_func = specific_heat_func
        
        if material_energy_func is None:
            self.material_energy_func = lambda T, x, y: self.rho * self.specific_heat_func(T, x, y) * T
        else:
            self.material_energy_func = material_energy_func
        
        if inverse_material_energy_func is None:
            self.inverse_material_energy_func = lambda e, x, y: e / (self.rho * self.cv)
        else:
            self.inverse_material_energy_func = inverse_material_energy_func
        
        # Set up boundary conditions for DiffusionOperatorSolver2D
        # Convert from dict format to simple string format (simplified for now)
        if boundary_funcs is None:
            self.left_bc = 'neumann'
            self.right_bc = 'neumann'
            self.bottom_bc = 'neumann'
            self.top_bc = 'neumann'
            self.left_bc_value = 0.0
            self.right_bc_value = 0.0
            self.bottom_bc_value = 0.0
            self.top_bc_value = 0.0
        else:
            # For now, use default Neumann (reflecting) BCs
            # TODO: Convert boundary_funcs dict to BC parameters
            self.left_bc = 'neumann'
            self.right_bc = 'neumann'
            self.bottom_bc = 'neumann'
            self.top_bc = 'neumann'
            self.left_bc_value = 0.0
            self.right_bc_value = 0.0
            self.bottom_bc_value = 0.0
            self.top_bc_value = 0.0
        
        # Create group solvers (one DiffusionOperatorSolver2D per group)
        self.solvers = []
        for g in range(n_groups):
            solver = DiffusionOperatorSolver2D(
                x_min=x_min, x_max=x_max, nx_cells=nx_cells,
                y_min=y_min, y_max=y_max, ny_cells=ny_cells,
                geometry=geometry,
                diffusion_coeff_func=diffusion_coeff_funcs[g],
                absorption_coeff_func=absorption_coeff_funcs[g],
                dt=dt,
                left_bc=self.left_bc, right_bc=self.right_bc,
                bottom_bc=self.bottom_bc, top_bc=self.top_bc,
                left_bc_value=self.left_bc_value, right_bc_value=self.right_bc_value,
                bottom_bc_value=self.bottom_bc_value, top_bc_value=self.top_bc_value
            )
            self.solvers.append(solver)
        
        # Extract grid information from first solver
        self.x_centers = self.solvers[0].x_centers
        self.y_centers = self.solvers[0].y_centers
        self.x_faces = self.solvers[0].x_faces
        self.y_faces = self.solvers[0].y_faces
        self.X_centers = self.solvers[0].X_centers
        self.Y_centers = self.solvers[0].Y_centers
        
        # Create flattened coordinate arrays for material properties
        self.X_flat = self.X_centers.flatten(order='C')
        self.Y_flat = self.Y_centers.flatten(order='C')
        
        # Storage for group quantities (all 1D flattened arrays)
        self.phi_g_stored = np.zeros((n_groups, self.n_total), dtype=np.float64)
        self.phi_g_fraction = np.ones((n_groups, self.n_total), dtype=np.float64) / n_groups
        self.sigma_a = np.zeros((n_groups, self.n_total), dtype=np.float64)
        
        # Storage for radiation energy density E_r
        self.E_r = np.ones(self.n_total, dtype=np.float64)
        self.E_r_old = np.ones(self.n_total, dtype=np.float64)
        
        # Storage for temperature
        self.T = np.ones(self.n_total, dtype=np.float64)
        self.T_old = np.ones(self.n_total, dtype=np.float64)
        
        # Planck function setup
        if planck_funcs is not None:
            if callable(planck_funcs):
                self.planck_funcs = [planck_funcs] * n_groups
            else:
                self.planck_funcs = planck_funcs
        else:
            # Use library Planck integrals
            self.planck_funcs = []
            for g in range(n_groups):
                E_low = energy_edges[g]
                E_high = energy_edges[g + 1]
                self.planck_funcs.append(lambda T, El=E_low, Eh=E_high: Bg(El, Eh, T))
        
        if dplanck_dT_funcs is not None:
            if callable(dplanck_dT_funcs):
                self.dplanck_dT_funcs = [dplanck_dT_funcs] * n_groups
            else:
                self.dplanck_dT_funcs = dplanck_dT_funcs
        else:
            # Use library Planck derivatives
            self.dplanck_dT_funcs = []
            for g in range(n_groups):
                E_low = energy_edges[g]
                E_high = energy_edges[g + 1]
                self.dplanck_dT_funcs.append(lambda T, El=E_low, Eh=E_high: dBgdT(El, Eh, T))
        
        # Emission fractions χ_g
        if emission_fractions is not None:
            self.chi = np.array(emission_fractions, dtype=np.float64)
            if len(self.chi) != n_groups:
                raise ValueError(f"emission_fractions must have length n_groups = {n_groups}")
            if not np.isclose(np.sum(self.chi), 1.0):
                warnings.warn(f"emission_fractions sum to {np.sum(self.chi):.6f}, not 1.0")
        else:
            # Compute from Rosseland integrals
            self.chi = self.compute_emission_fractions(T_ref=1.0)
        
        print(f"\n2D Multigroup Solver initialized:")
        print(f"  Geometry: {geometry}")
        print(f"  Grid: {nx_cells} × {ny_cells} = {self.n_total} cells")
        print(f"  Domain: [{x_min:.3f}, {x_max:.3f}] × [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Groups: {n_groups}")
        print(f"  Energy edges: {energy_edges}")
        print(f"  Emission fractions χ_g: {self.chi}")
        print(f"  Δt={dt:.2e}, θ={theta}, max_newton_iter={max_newton_iter}")
    
    def compute_emission_fractions(self, T_ref: float = 1.0) -> np.ndarray:
        """
        Compute emission fractions χ_g from Rosseland integrals.
        
        χ_g = ∫_{E_g}^{E_{g+1}} (∂B/∂T) dE / Σ_{g'} ∫_{E_{g'}}^{E_{g'+1}} (∂B/∂T) dE
        
        Parameters:
        -----------
        T_ref : float
            Reference temperature for computing χ_g
        
        Returns:
        --------
        chi : ndarray
            Emission fractions (length n_groups)
        """
        chi = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            chi[g] = self.dplanck_dT_funcs[g](T_ref)
        
        total = np.sum(chi)
        if total > 0:
            chi /= total
        else:
            chi = np.ones(self.n_groups) / self.n_groups
        
        return chi
    
    def update_absorption_coefficients(self, T: np.ndarray):
        """
        Update group absorption coefficients σ*_{a,g} at all cells.
        
        Parameters:
        -----------
        T : ndarray
            Temperature at all cells (1D flattened array)
        """
        for g in range(self.n_groups):
            for i in range(self.n_total):
                x = self.X_flat[i]
                y = self.Y_flat[i]
                self.sigma_a[g, i] = self.absorption_coeff_funcs[g](T[i], x, y)
    
    def compute_fleck_factor(self, T: np.ndarray) -> np.ndarray:
        """
        Compute local Fleck factor f = 1 / (1 + β·Δt) at each cell.
        
        Uses temperature-dependent Rosseland integrals:
        β = (4π/C_v) Σ_g σ*_{a,g} ∂B_g/∂T
        
        Parameters:
        -----------
        T : ndarray
            Temperature at all cells (1D flattened array)
        
        Returns:
        --------
        f : ndarray
            Fleck factor at all cells
        """
        f = np.ones(self.n_total, dtype=np.float64)
        
        for i in range(self.n_total):
            # Compute Σ_g σ*_{a,g} ∂B_g/∂T
            sum_sigma_dB = 0.0
            for g in range(self.n_groups):
                dB_g = self.dplanck_dT_funcs[g](T[i])
                sum_sigma_dB += self.sigma_a[g, i] * dB_g
            
            # Get cv at this temperature
            x = self.X_flat[i]
            y = self.Y_flat[i]
            if self.cv_is_function:
                cv_i = self.cv_func(T[i], x, y)
            else:
                cv_i = self.cv
            
            # f = 1 / (1 + 4π (Δt/C_v) Σ_g σ*_{a,g} ∂B_g/∂T)
            denominator = 1.0 + 4.0 * np.pi * (self.dt / cv_i) * sum_sigma_dB
            f[i] = 1.0 / denominator
        
        return f
    
    def get_2d_array(self, array_1d: np.ndarray) -> np.ndarray:
        """Convert 1D flattened array to 2D (nx, ny) array"""
        return unflatten_2d(array_1d, self.nx_cells, self.ny_cells)
    
    def set_from_2d_array(self, array_2d: np.ndarray) -> np.ndarray:
        """Convert 2D (nx, ny) array to 1D flattened array"""
        return flatten_2d(array_2d, self.nx_cells, self.ny_cells)
    
    def step(self, verbose: bool = False, gmres_tol: float = 1e-6,
             gmres_maxiter: int = 200, use_preconditioner: bool = False) -> dict:
        """
        Advance solution by one time step using Newton iteration.
        
        Parameters:
        -----------
        verbose : bool
            Print iteration info
        gmres_tol : float
            GMRES convergence tolerance
        gmres_maxiter : int
            Maximum GMRES iterations
        use_preconditioner : bool
            Use LMFG preconditioner
        
        Returns:
        --------
        info : dict
            Dictionary with convergence info
        """
        if verbose:
            print(f"\n=== Time step at t = {self.t:.4e} ===")
        
        # Store old values
        self.T_old[:] = self.T
        self.E_r_old[:] = self.E_r
        
        # Newton iteration
        for newton_iter in range(self.max_newton_iter):
            if verbose:
                print(f"\n  Newton iteration {newton_iter + 1}")
            
            # Linearization temperature
            T_star = self.T.copy()
            
            # Update absorption coefficients
            self.update_absorption_coefficients(T_star)
            
            # Compute Fleck factor
            self.fleck_factor = self.compute_fleck_factor(T_star)
            
            # Compute source terms ξ_g for all groups
            xi_g_list = []
            for g in range(self.n_groups):
                xi_g = self.compute_source_xi(g, T_star, self.t + self.dt)
                xi_g_list.append(xi_g)
            
            # Solve for κ
            kappa, gmres_info = self.solve_for_kappa(
                T_star, xi_g_list,
                gmres_tol=gmres_tol,
                gmres_maxiter=gmres_maxiter,
                use_preconditioner=use_preconditioner,
                verbose=verbose
            )
            
            # Compute radiation energy from κ
            E_r_new, phi_g_fraction = self.compute_radiation_energy_from_kappa(
                kappa, T_star, xi_g_list
            )
            
            # Update temperature
            T_new = self.update_temperature(kappa, T_star)
            
            # Check convergence
            r_E = np.linalg.norm(E_r_new - self.E_r) / (np.linalg.norm(self.E_r) + 1e-14)
            r_T = np.linalg.norm(T_new - self.T) / (np.linalg.norm(self.T) + 1e-14)
            
            if verbose:
                print(f"    Residuals: r_E={r_E:.3e}, r_T={r_T:.3e}")
            
            # Update solution
            self.E_r[:] = E_r_new
            self.T[:] = T_new
            self.phi_g_fraction[:] = phi_g_fraction
            
            # Check convergence
            if r_E < self.newton_tol and r_T < self.newton_tol:
                if verbose:
                    print(f"  Newton converged in {newton_iter + 1} iterations")
                break
        
        # Update time
        self.t += self.dt
        
        return {
            'newton_iterations': newton_iter + 1,
            'gmres_info': gmres_info,
            'final_residuals': {'r_E': r_E, 'r_T': r_T}
        }
    
    def compute_source_xi(self, g: int, T_star: np.ndarray, t: float) -> np.ndarray:
        """
        Compute source term ξ_g for group g.
        
        ξ_g = (1/cΔt)φ_g^n + 4π·σ*_{a,g}·B_g(T_★) 
              - χ_g(1-f)·[Σ_{g'} 4π·σ*_{a,g'}·B_{g'}(T_★) + Δe/Δt] + Q_g
        
        Parameters:
        -----------
        g : int
            Group index
        T_star : ndarray
            Linearization temperature
        t : float
            Current time
        
        Returns:
        --------
        xi_g : ndarray
            Source term for group g (1D flattened)
        """
        f = self.fleck_factor
        
        # Material energy change
        e_star = np.array([self.material_energy_func(T_star[i], self.X_flat[i], self.Y_flat[i]) 
                          for i in range(self.n_total)])
        e_n = np.array([self.material_energy_func(self.T_old[i], self.X_flat[i], self.Y_flat[i]) 
                       for i in range(self.n_total)])
        Delta_e = e_star - e_n
        
        # Planck functions for this group
        B_g_star = np.array([self.planck_funcs[g](T_star[i]) for i in range(self.n_total)])
        
        # Sum over all groups for emission term
        sum_emission = np.zeros(self.n_total)
        for gp in range(self.n_groups):
            B_gp_star = np.array([self.planck_funcs[gp](T_star[i]) for i in range(self.n_total)])
            sum_emission += self.sigma_a[gp, :] * 4.0 * np.pi * B_gp_star
        
        # Coupling term
        coupling_term = sum_emission + Delta_e / self.dt
        
        # φ_g^n from stored fractional distribution
        phi_total_old = self.E_r_old * C_LIGHT
        phi_g_old = phi_total_old * self.phi_g_fraction[g, :]
        
        # Assemble ξ_g
        xi_g = (1.0 / (C_LIGHT * self.dt)) * phi_g_old + \
               4.0 * np.pi * self.sigma_a[g, :] * B_g_star - \
               self.chi[g] * (1.0 - f) * coupling_term
        
        # Add external source Q_g(x, y, t)
        Q_g = np.array([self.source_funcs[g](self.X_flat[i], self.Y_flat[i], t) 
                       for i in range(self.n_total)])
        xi_g += Q_g
        
        return xi_g
    
    def solve_for_kappa(self, T_star: np.ndarray, xi_g_list: List[np.ndarray],
                       gmres_tol: float = 1e-6, gmres_maxiter: int = 200,
                       use_preconditioner: bool = False,
                       verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Solve B·κ = RHS for κ using GMRES.
        
        B·κ = κ - Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        RHS = Σ_g σ*_{a,g}·A_g^{-1}·ξ_g
        
        If use_preconditioner=True, uses LMFG preconditioner C and solves (C·B)κ = C·rhs.
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature for operator evaluation
        xi_g_list : list of ndarray
            Source terms for each group
        gmres_tol : float
            GMRES tolerance
        gmres_maxiter : int
            Maximum GMRES iterations
        use_preconditioner : bool
            Use LMFG preconditioner
        verbose : bool
            Print diagnostic info
        
        Returns:
        --------
        kappa : ndarray
            Absorption rate density
        info_dict : dict
            GMRES convergence info
        """
        # Compute RHS
        rhs = self.compute_rhs_for_kappa(T_star, xi_g_list)
        rhs_norm = np.linalg.norm(rhs) + 1e-30
        
        # Define B operator
        def matvec(kappa_vec):
            return self.apply_operator_B(kappa_vec, T_star, xi_g_list)
        
        B_operator = LinearOperator((self.n_total, self.n_total), matvec=matvec)
        kappa_initial = np.zeros(self.n_total)
        
        # Storage for residuals
        self.gmres_precond_resids = []
        self.gmres_true_resids = []
        self.gmres_callback_type = 'pr_norm'
        
        def _callback_pr_norm(pr_norm):
            self.gmres_precond_resids.append(float(pr_norm))
            if verbose:
                print(f"    GMRES iter {len(self.gmres_precond_resids)}: pr_norm={float(pr_norm):.3e}")
        
        # Create preconditioner if requested
        C_operator = None
        if use_preconditioner:
            if verbose:
                print("  Creating LMFG preconditioner...")
            C_operator = self.create_lmfg_preconditioner(T_star, verbose=verbose)
        
        # Set up operator and RHS
        restart_val = min(30, self.n_total)
        if use_preconditioner:
            def CB_matvec(x):
                return C_operator.matvec(B_operator.matvec(x))
            Aop = LinearOperator((self.n_total, self.n_total), matvec=CB_matvec)
            bop = C_operator.matvec(rhs)
        else:
            Aop = B_operator
            bop = rhs
        
        # Solve with GMRES
        if verbose:
            print(f"  Solving with GMRES (tol={gmres_tol}, maxiter={gmres_maxiter})...")
        
        kappa, info = gmres(
            Aop, bop,
            x0=kappa_initial,
            rtol=gmres_tol,
            atol=gmres_tol * 1e-2,
            restart=restart_val,
            maxiter=gmres_maxiter,
            callback=_callback_pr_norm,
            callback_type='pr_norm'
        )
        
        # Compute true residual
        true_res = rhs - B_operator.matvec(kappa)
        rel_true_res = np.linalg.norm(true_res) / rhs_norm
        self.gmres_true_resids.append(rel_true_res)
        
        iters = len(self.gmres_precond_resids)
        info_dict = {
            'info': info,
            'iterations': iters,
            'callback_type': self.gmres_callback_type,
            'final_true_resid': rel_true_res,
        }
        
        if verbose:
            if info == 0:
                print(f"  GMRES converged in {iters} iterations")
            else:
                print(f"  GMRES info: {info} (0=success, >0=did not converge)")
            print(f"  True residual: {rel_true_res:.3e}")
        
        return kappa, info_dict
    
    def apply_operator_B(self, kappa: np.ndarray, T_star: np.ndarray, 
                        xi_g_list: List[np.ndarray]) -> np.ndarray:
        """
        Apply operator B to vector κ.
        
        B·κ = κ - Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        
        Parameters:
        -----------
        kappa : ndarray
            Input vector (1D flattened)
        T_star : ndarray
            Temperature (1D flattened)
        xi_g_list : list of ndarray
            Source terms (not used in B operator)
        
        Returns:
        --------
        result : ndarray
            B·κ (1D flattened)
        """
        f = self.fleck_factor
        result = kappa.copy().astype(np.float64)
        
        # Reshape temperature to 2D for solvers
        T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
        
        # Subtract Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        for g in range(self.n_groups):
            # RHS for group g: χ_g(1-f)κ
            rhs_g = self.chi[g] * (1.0 - f) * kappa
            
            # Reshape to 2D for solver
            rhs_g_2d = unflatten_2d(rhs_g, self.nx_cells, self.ny_cells)
            
            # Solve A_g φ_g = rhs_g
            phi_g_2d = self.solvers[g].solve(rhs_g_2d, T_2d)
            
            # Flatten back to 1D
            phi_g = flatten_2d(phi_g_2d, self.nx_cells, self.ny_cells)
            
            # Subtract σ*_{a,g} φ_g
            result -= self.sigma_a[g, :] * phi_g
        
        return result
    
    def compute_rhs_for_kappa(self, T_star: np.ndarray, 
                             xi_g_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute RHS for κ equation.
        
        RHS = Σ_g σ*_{a,g}·A_g^{-1}·ξ_g
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature (1D flattened)
        xi_g_list : list of ndarray
            Source terms for each group (1D flattened)
        
        Returns:
        --------
        rhs : ndarray
            Right-hand side (1D flattened)
        """
        rhs = np.zeros(self.n_total)
        
        # Reshape temperature to 2D for solvers
        T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
        
        for g in range(self.n_groups):
            # Reshape ξ_g to 2D
            xi_g_2d = unflatten_2d(xi_g_list[g], self.nx_cells, self.ny_cells)
            
            # Solve A_g φ_g = ξ_g
            phi_g_2d = self.solvers[g].solve(xi_g_2d, T_2d)
            
            # Flatten back to 1D
            phi_g = flatten_2d(phi_g_2d, self.nx_cells, self.ny_cells)
            
            # Add σ*_{a,g} φ_g
            rhs += self.sigma_a[g, :] * phi_g
        
        return rhs
    
    def compute_radiation_energy_from_kappa(self, kappa: np.ndarray, T_star: np.ndarray,
                                           xi_g_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radiation energy density E_r from κ.
        
        φ_g^{n+1} = A_g^{-1}(χ_g(1-f)κ + ξ_g)
        E_r = (1/c) Σ_g φ_g^{n+1}
        
        Parameters:
        -----------
        kappa : ndarray
            Absorption rate density (1D flattened)
        T_star : ndarray
            Temperature (1D flattened)
        xi_g_list : list of ndarray
            Source terms (1D flattened)
        
        Returns:
        --------
        E_r : ndarray
            Radiation energy density (1D flattened)
        phi_g_fraction : ndarray
            Fractional distribution (n_groups, n_total)
        """
        f = self.fleck_factor
        phi_total = np.zeros(self.n_total)
        phi_g_array = np.zeros((self.n_groups, self.n_total))
        
        # Reshape temperature to 2D for solvers
        T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
        
        for g in range(self.n_groups):
            # RHS for group g: χ_g(1-f)κ + ξ_g
            rhs_g = self.chi[g] * (1.0 - f) * kappa + xi_g_list[g]
            
            # Reshape to 2D
            rhs_g_2d = unflatten_2d(rhs_g, self.nx_cells, self.ny_cells)
            
            # Solve A_g φ_g = rhs_g
            phi_g_2d = self.solvers[g].solve(rhs_g_2d, T_2d)
            
            # Flatten back to 1D
            phi_g = flatten_2d(phi_g_2d, self.nx_cells, self.ny_cells)
            
            # Store and accumulate
            phi_g_array[g, :] = phi_g
            phi_total += phi_g
        
        # Compute E_r = (1/c) Σ_g φ_g
        E_r = phi_total / C_LIGHT
        
        # Compute fractional distribution
        phi_g_fraction = np.zeros((self.n_groups, self.n_total))
        for g in range(self.n_groups):
            phi_g_fraction[g, :] = phi_g_array[g, :] / (phi_total + 1e-30)
        
        # Store for next iteration
        self.phi_g_stored[:] = phi_g_array
        
        return E_r, phi_g_fraction
   
    def update_temperature(self, kappa: np.ndarray, T_star: np.ndarray) -> np.ndarray:
        """
        Update material temperature from κ.
        
        e(T_{n+1}) = e(T_n) + Δt·f·[κ - Σ_g σ*_{a,g}·4πB_g(T_★)] + (1-f)Δe
        
        Parameters:
        -----------
        kappa : ndarray
            Absorption rate density (1D flattened)
        T_star : ndarray
            Linearization temperature (1D flattened)
        
        Returns:
        --------
        T_new : ndarray
            Updated temperature (1D flattened)
        """
        f = self.fleck_factor
        
        # Old material energy
        e_n = np.array([self.material_energy_func(self.T_old[i], self.X_flat[i], self.Y_flat[i])
                       for i in range(self.n_total)])
        
        # Linearization material energy
        e_star = np.array([self.material_energy_func(T_star[i], self.X_flat[i], self.Y_flat[i])
                          for i in range(self.n_total)])
        Delta_e = e_star - e_n
        
        # Planck emission sum
        sum_planck = np.zeros(self.n_total)
        for g in range(self.n_groups):
            B_g_star = np.array([self.planck_funcs[g](T_star[i]) for i in range(self.n_total)])
            sum_planck += self.sigma_a[g, :] * 4.0 * np.pi * B_g_star
        
        # New material energy
        e_new = e_n + self.dt * f * (kappa - sum_planck) + (1.0 - f) * Delta_e
        
        # Invert to get temperature
        T_new = np.array([self.inverse_material_energy_func(e_new[i], self.X_flat[i], self.Y_flat[i])
                         for i in range(self.n_total)])
        
        return T_new
    
    def create_lmfg_preconditioner(self, T_star: np.ndarray,
                                   verbose: bool = False) -> LinearOperator:
        """
        Create Linear Multifrequency Gray (LMFG) preconditioner for 2D.
        
        C = I + ⟨σ_a⟩ · H^{-1} · (1-f)
        
        where H is the 2D gray operator:
            H = -∇·⟨D⟩∇ + ⟨σ_a⟩(1-f) + 1/(c·Δt)
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature (1D flattened)
        verbose : bool
            Print diagnostic info
        
        Returns:
        --------
        precond_operator : LinearOperator
            Preconditioner operator C
        """
        f = self.fleck_factor
        nu = 1.0 - f
        nu_floor = 1e-10
        
        if np.max(np.abs(nu)) < nu_floor:
            if verbose:
                print("  LMFG: nu=(1-f) ~ 0 everywhere -> returning identity preconditioner")
            return LinearOperator(
                (self.n_total, self.n_total),
                matvec=lambda x: x,
                dtype=np.float64
            )
        
        # Compute gray weights
        lambda_tilde = self.compute_gray_weights(T_star, verbose=verbose)
        
        # Compute gray operator coefficients
        sigma_a_gray = np.zeros(self.n_total)
        D_gray = np.zeros(self.n_total)
        
        for i in range(self.n_total):
            for g in range(self.n_groups):
                sigma_a_gray[i] += self.sigma_a[g, i] * lambda_tilde[g, i]
                # For D_gray, we need to evaluate diffusion coefficient
                # Use a representative value at this cell
                x = self.X_flat[i]
                y = self.Y_flat[i]
                D_g = self.diffusion_coeff_funcs[g](T_star[i], x, y)
                D_gray[i] += D_g * lambda_tilde[g, i]
        
        if verbose:
            print(f"  LMFG Preconditioner setup:")
            print(f"    ⟨σ_a⟩: min={sigma_a_gray.min():.3e}, max={sigma_a_gray.max():.3e}")
            print(f"    ⟨D⟩: min={D_gray.min():.3e}, max={D_gray.max():.3e}")
        
        # Create gray diffusion coefficient and absorption functions
        # Use 2D interpolation from computed values
        D_gray_2d = unflatten_2d(D_gray, self.nx_cells, self.ny_cells)
        sigma_a_gray_2d = unflatten_2d(sigma_a_gray, self.nx_cells, self.ny_cells)
        nu_2d = unflatten_2d(nu, self.nx_cells, self.ny_cells)
        
        def gray_diffusion_func(T, x, y, *args):
            """Gray diffusion coefficient from interpolation"""
            # Find nearest cell
            i = np.searchsorted(self.x_centers, x)
            j = np.searchsorted(self.y_centers, y)
            i = np.clip(i, 0, self.nx_cells - 1)
            j = np.clip(j, 0, self.ny_cells - 1)
            return D_gray_2d[i, j]
        
        def gray_absorption_func(T, x, y):
            """Gray absorption coefficient: ⟨σ_a⟩(1-f)"""
            # Find nearest cell
            i = np.searchsorted(self.x_centers, x)
            j = np.searchsorted(self.y_centers, y)
            i = np.clip(i, 0, self.nx_cells - 1)
            j = np.clip(j, 0, self.ny_cells - 1)
            return sigma_a_gray_2d[i, j] * nu_2d[i, j]
        
        # Create gray diffusion solver
        gray_solver = DiffusionOperatorSolver2D(
            x_min=self.x_min, x_max=self.x_max, nx_cells=self.nx_cells,
            y_min=self.y_min, y_max=self.y_max, ny_cells=self.ny_cells,
            geometry=self.geometry,
            dt=self.dt,
            diffusion_coeff_func=gray_diffusion_func,
            absorption_coeff_func=gray_absorption_func,
            left_bc=self.left_bc, right_bc=self.right_bc,
            bottom_bc=self.bottom_bc, top_bc=self.top_bc,
            left_bc_value=0.0, right_bc_value=0.0,  # Homogeneous BCs
            bottom_bc_value=0.0, top_bc_value=0.0
        )
        
        # Define preconditioner application
        def precond_matvec(y: np.ndarray) -> np.ndarray:
            """
            Apply LMFG left-multiplier C to a vector y.
            
            C·y = y + ⟨σ_a⟩·U,   where   H·U = (1-f)·y
            """
            # RHS for gray solve: (1-f)·y
            rhs_gray = nu * y
            
            # Reshape to 2D
            rhs_gray_2d = unflatten_2d(rhs_gray, self.nx_cells, self.ny_cells)
            T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)
            
            # Solve H·U = rhs_gray
            U_2d = gray_solver.solve(rhs_gray_2d, T_2d)
            
            # Flatten back to 1D
            U = flatten_2d(U_2d, self.nx_cells, self.ny_cells)
            
            # Apply left-multiplier: C·y = y + ⟨σ_a⟩·U
            return y + sigma_a_gray * U
        
        # Create LinearOperator for preconditioner
        precond_operator = LinearOperator(
            (self.n_total, self.n_total),
            matvec=precond_matvec,
            dtype=np.float64
        )
        
        if verbose:
            print(f"    Gray solver created with {self.n_total} cells")
        
        return precond_operator
    
    def compute_gray_weights(self, T_star: np.ndarray, 
                            verbose: bool = False) -> np.ndarray:
        """
        Compute gray weights λ̃_g for LMFG preconditioner.
        
        λ̃_g = ∂B_g/∂T / Σ_{g'} ∂B_{g'}/∂T
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature (1D flattened)
        verbose : bool
            Print diagnostic info
        
        Returns:
        --------
        lambda_tilde : ndarray
            Gray weights (n_groups, n_total)
        """
        lambda_tilde = np.zeros((self.n_groups, self.n_total))
        
        for i in range(self.n_total):
            dB_sum = 0.0
            dB_g = np.zeros(self.n_groups)
            
            for g in range(self.n_groups):
                dB_g[g] = self.dplanck_dT_funcs[g](T_star[i])
                dB_sum += dB_g[g]
            
            if dB_sum > 0:
                lambda_tilde[:, i] = dB_g / dB_sum
            else:
                lambda_tilde[:, i] = 1.0 / self.n_groups
        
        return lambda_tilde


if __name__ == "__main__":
    print("2D Multigroup Diffusion Solver")
    print("="*70)
    print("\nSTATUS: Core implementation complete using DiffusionOperatorSolver2D!")
    print("\nImplemented methods:")
    print("  ✓ apply_operator_B() - B operator with G diffusion solves")
    print("  ✓ compute_rhs_for_kappa() - RHS computation with G diffusion solves")
    print("  ✓ compute_radiation_energy_from_kappa() - Extract E_r and φ_g fractions")
    print("  ✓ create_lmfg_preconditioner() - 2D LMFG preconditioner with gray solve")
    print("  ✓ step() - Newton iteration for time advancement")
    print()
    print("Ready to use! Example:")
    print("  solver = MultigroupDiffusionSolver2D(")
    print("      n_groups=3, x_min=0, x_max=1, nx_cells=10,")
    print("      y_min=0, y_max=1, ny_cells=10, ...)")
    print("  info = solver.step(verbose=True, use_preconditioner=True)")
    print()
    print("Next steps:")
    print("  1. Create test_2d_multigroup.py with simple test problem")
    print("  2. Verify convergence and preconditioner effectiveness")
    print("  3. Create 2D Marshak wave or other physics test")
    print("="*70)
    print("     - H = -∇·⟨D⟩∇ + ⟨σ_a⟩(1-f) + 1/(c·Δt)")
    print("     - Compute gray weights λ̃_g from Rosseland integrals")
    print()
    print("APPROACH:")
    print("  The 1D solver uses DiffusionOperatorSolver1D which provides:")
    print("    - solve(rhs, T, phi_guess) -> phi")
    print()
    print("  For 2D, need equivalent DiffusionOperatorSolver2D or extract from twoDFV:")
    print("    - assemble_implicit_matrix_coo() builds sparse matrix")
    print("    - apply_boundary_conditions_phi() applies BCs")
    print("    - spsolve(A, rhs) gives solution")
    print()
    print("SUGGESTED PATH:")
    print("  1. Create test_2d_multigroup.py with simple 2x2 or 4x4 grid")
    print("  2. Implement SimpleDiffusionOperator2D as standalone class")
    print("  3. Test single group solve first")
    print("  4. Then implement full multigroup with LMFG")
    print("="*70)
