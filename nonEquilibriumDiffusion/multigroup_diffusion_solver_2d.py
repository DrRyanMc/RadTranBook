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
# FAST BATCH PLANCK EVALUATION (Numba JIT, parallel over cells)
# =============================================================================
# Requires planck_integrals (Numba-compiled Bg / dBgdT) and numba.
# Falls back gracefully when either is unavailable.
_PLANCK_MULTICELL_AVAILABLE = False
try:
    from numba import njit, prange
    from numba.core.registry import CPUDispatcher as _CPUDispatcher
    if isinstance(Bg, _CPUDispatcher):
        @njit(fastmath=True, cache=True, parallel=True)
        def _planck_multicell(nu_bounds, T_arr):
            """Evaluate B_g(T_i) and dB_g/dT(T_i) for all groups g and cells i.

            Parameters
            ----------
            nu_bounds : 1-D float64 array, length n_groups+1
            T_arr     : 1-D float64 array, length n_cells

            Returns
            -------
            B  : (n_groups, n_cells) float64
            dB : (n_groups, n_cells) float64
            """
            n_groups = len(nu_bounds) - 1
            n_cells  = len(T_arr)
            B  = np.zeros((n_groups, n_cells))
            dB = np.zeros((n_groups, n_cells))
            for i in prange(n_cells):          # parallel over cells
                Ti = T_arr[i]
                for g in range(n_groups):      # serial over groups (small)
                    B[g,  i] = Bg(   nu_bounds[g], nu_bounds[g + 1], Ti)
                    dB[g, i] = dBgdT(nu_bounds[g], nu_bounds[g + 1], Ti)
            return B, dB

        _PLANCK_MULTICELL_AVAILABLE = True
except Exception:
    pass

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
        # Extract boundary functions for each group
        if boundary_funcs is None:
            # Default: Neumann (reflecting) BCs
            self.left_bc_funcs = None
            self.right_bc_funcs = None
            self.bottom_bc_funcs = None
            self.top_bc_funcs = None
            self.left_bc = 'neumann'
            self.right_bc = 'neumann'
            self.bottom_bc = 'neumann'
            self.top_bc = 'neumann'
            self.left_bc_value = 0.0
            self.right_bc_value = 0.0
            self.bottom_bc_value = 0.0
            self.top_bc_value = 0.0
        else:
            # Extract boundary function lists for each side
            self.left_bc_funcs = boundary_funcs.get('left', None)
            self.right_bc_funcs = boundary_funcs.get('right', None)
            self.bottom_bc_funcs = boundary_funcs.get('bottom', None)
            self.top_bc_funcs = boundary_funcs.get('top', None)
            # Fallback to Neumann if not provided
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
            # Get boundary functions for this group
            left_bc_func_g = self.left_bc_funcs[g] if self.left_bc_funcs is not None else None
            right_bc_func_g = self.right_bc_funcs[g] if self.right_bc_funcs is not None else None
            bottom_bc_func_g = self.bottom_bc_funcs[g] if self.bottom_bc_funcs is not None else None
            top_bc_func_g = self.top_bc_funcs[g] if self.top_bc_funcs is not None else None
            
            solver = DiffusionOperatorSolver2D(
                x_min=x_min, x_max=x_max, nx_cells=nx_cells,
                y_min=y_min, y_max=y_max, ny_cells=ny_cells,
                geometry=geometry,
                diffusion_coeff_func=diffusion_coeff_funcs[g],
                absorption_coeff_func=absorption_coeff_funcs[g],
                dt=dt,
                left_bc_func=left_bc_func_g,
                right_bc_func=right_bc_func_g,
                bottom_bc_func=bottom_bc_func_g,
                top_bc_func=top_bc_func_g,
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

        # Use fast Numba batch path when no custom Planck functions are provided.
        self._use_default_planck = (
            _PLANCK_MULTICELL_AVAILABLE
            and (planck_funcs is None)
            and (dplanck_dT_funcs is None)
        )

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
        """Update group absorption coefficients σ*_{a,g} at all cells."""
        for g in range(self.n_groups):
            self.sigma_a[g, :] = np.vectorize(self.absorption_coeff_funcs[g])(
                T, self.X_flat, self.Y_flat
            )

    def _compute_planck_arrays(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute B_g(T) and dB_g/dT(T) for all groups and cells."""
        if self._use_default_planck:
            # Fast path: single Numba call, parallel over cells
            return _planck_multicell(self.energy_edges, T)
        # Fallback for custom planck_funcs
        B_star  = np.zeros((self.n_groups, self.n_total), dtype=np.float64)
        dB_star = np.zeros((self.n_groups, self.n_total), dtype=np.float64)
        for g in range(self.n_groups):
            planck_g  = self.planck_funcs[g]
            dplanck_g = self.dplanck_dT_funcs[g]
            for i in range(self.n_total):
                Ti = T[i]
                B_star[g, i]  = planck_g(Ti)
                dB_star[g, i] = dplanck_g(Ti)
        return B_star, dB_star

    def _compute_material_energy(self, T: np.ndarray) -> np.ndarray:
        """Compute material energy e(T, x, y) for all cells."""
        return np.vectorize(self.material_energy_func)(T, self.X_flat, self.Y_flat)

    def _compute_external_source(self, g: int, t: float) -> np.ndarray:
        """Compute external source Q_g(x, y, t) for one group."""
        return np.vectorize(self.source_funcs[g])(self.X_flat, self.Y_flat, t)
    
    def compute_fleck_factor(self, T: np.ndarray,
                             dB_star: Optional[np.ndarray] = None) -> np.ndarray:
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
        if dB_star is None:
            _, dB_star = self._compute_planck_arrays(T)

        sum_sigma_dB = np.sum(self.sigma_a * dB_star, axis=0)

        if self.cv_is_function:
            cv_arr = np.vectorize(self.cv_func)(T, self.X_flat, self.Y_flat)
        else:
            cv_arr = self.cv

        denominator = 1.0 + 4.0 * np.pi * (self.dt / cv_arr) * sum_sigma_dB
        return 1.0 / denominator
    
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
        
        # Track total GMRES iterations
        total_gmres_iters = 0
        
        # Newton iteration
        for newton_iter in range(self.max_newton_iter):
            if verbose:
                print(f"\n  Newton iteration {newton_iter + 1}")
            
            # Linearization temperature
            T_star = self.T.copy()
            t_eval = self.t + self.dt
            
            # Update absorption coefficients
            self.update_absorption_coefficients(T_star)
            
            # Compute shared thermodynamic fields once per Newton iteration.
            B_star, dB_star = self._compute_planck_arrays(T_star)
            e_star = self._compute_material_energy(T_star)
            e_n = self._compute_material_energy(self.T_old)

            # Compute Fleck factor
            self.fleck_factor = self.compute_fleck_factor(T_star, dB_star=dB_star)

            # Compute source terms ξ_g for all groups using shared arrays
            xi_g_list = self.compute_all_source_xi(
                T_star,
                t_eval,
                B_star=B_star,
                e_star=e_star,
                e_n=e_n
            )
            
            # Solve for κ
            kappa, gmres_info = self.solve_for_kappa(
                T_star, xi_g_list,
                gmres_tol=gmres_tol,
                gmres_maxiter=gmres_maxiter,
                use_preconditioner=use_preconditioner,
                bc_time=t_eval,
                verbose=verbose
            )
            
            # Track GMRES iterations
            total_gmres_iters += gmres_info.get('iterations', 0)
            
            # Compute radiation energy from κ
            E_r_new, phi_g_fraction = self.compute_radiation_energy_from_kappa(
                kappa, T_star, xi_g_list, bc_time=t_eval
            )
            
            # Update temperature
            T_new = self.update_temperature(kappa, T_star, B_star=B_star, e_star=e_star, e_n=e_n)
            
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
                    print(f"  Total GMRES iterations: {total_gmres_iters}")
                break
        
        # Update time
        self.t += self.dt
        
        return {
            'newton_iterations': newton_iter + 1,
            'gmres_info': gmres_info,
            'total_gmres_iterations': total_gmres_iters,
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
        return self.compute_all_source_xi(T_star, t)[g]

    def compute_all_source_xi(self, T_star: np.ndarray, t: float,
                              B_star: Optional[np.ndarray] = None,
                              e_star: Optional[np.ndarray] = None,
                              e_n: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Compute source terms ξ_g for all groups with shared precomputed terms.
        """
        f = self.fleck_factor

        if B_star is None:
            B_star, _ = self._compute_planck_arrays(T_star)

        if e_star is None:
            e_star = self._compute_material_energy(T_star)
        if e_n is None:
            e_n = self._compute_material_energy(self.T_old)

        Delta_e = e_star - e_n
        coupling_term = 4.0 * np.pi * np.sum(self.sigma_a * B_star, axis=0) + Delta_e / self.dt

        phi_total_old = self.E_r_old * C_LIGHT
        phi_old_scale = (1.0 / (C_LIGHT * self.dt)) * phi_total_old
        nu = 1.0 - f

        xi_g_list = []
        for g in range(self.n_groups):
            xi_g = (
                phi_old_scale * self.phi_g_fraction[g, :]
                + 4.0 * np.pi * self.sigma_a[g, :] * B_star[g, :]
                - self.chi[g] * nu * coupling_term
            )
            xi_g += self._compute_external_source(g, t)
            xi_g_list.append(xi_g)

        return xi_g_list
    
    def solve_for_kappa(self, T_star: np.ndarray, xi_g_list: List[np.ndarray],
                       gmres_tol: float = 1e-6, gmres_maxiter: int = 200,
                       use_preconditioner: bool = False,
                       bc_time: float = 0.0,
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
        bc_time : float
            Time used to evaluate time-dependent boundary conditions
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
        rhs = self.compute_rhs_for_kappa(T_star, xi_g_list, bc_time=bc_time)
        rhs_norm = np.linalg.norm(rhs) + 1e-30
        
        if verbose:
            print(f"  Problem size: {self.n_total} unknowns ({self.nx_cells}×{self.ny_cells} cells)")
            print(f"  RHS norm: {rhs_norm:.3e}, min: {rhs.min():.3e}, max: {rhs.max():.3e}")
        
        # Define B operator
        def matvec(kappa_vec):
            return self.apply_operator_B(kappa_vec, T_star, xi_g_list, bc_time=bc_time)
        
        B_operator = LinearOperator((self.n_total, self.n_total), matvec=matvec)
        kappa_initial = np.zeros(self.n_total)
        
        # Storage for residuals
        self.gmres_precond_resids = []
        self.gmres_true_resids = []
        self.gmres_callback_type = 'pr_norm'
        
        def _callback_pr_norm(pr_norm):
            self.gmres_precond_resids.append(float(pr_norm))
            if verbose:
                print(f"    gmres iter: pr_norm={float(pr_norm):.3e}")
        
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
        
        # Test operator linearity
        if verbose:
            test_x = np.random.randn(self.n_total)
            test_y = np.random.randn(self.n_total)
            a, b = 2.3, -1.7
            
            # Test B operator linearity
            Bx = B_operator.matvec(test_x)
            By = B_operator.matvec(test_y)
            Baxy = B_operator.matvec(a*test_x + b*test_y)
            lin_err_B = np.linalg.norm(Baxy - (a*Bx + b*By)) / (np.linalg.norm(Baxy) + 1e-30)
            print(f"  B operator linearity error: {lin_err_B:.3e} (should be ~0)")
            
            # Detailed breakdown
            if lin_err_B > 1e-10:
                print(f"    WARNING: Large linearity error!")
                print(f"    ||B(ax+by)||  = {np.linalg.norm(Baxy):.3e}")
                print(f"    ||aB(x)+bB(y)|| = {np.linalg.norm(a*Bx + b*By):.3e}")
                print(f"    ||difference|| = {np.linalg.norm(Baxy - (a*Bx + b*By)):.3e}")
                # Note: This error is likely due to numerical round-off in sparse solves
                # For 10 groups × 150 cells, we perform 30 sparse solves total
                # With double precision (~1e-16), accumulated error of ~1e-4 is acceptable
                if lin_err_B < 1e-3:
                    print(f"    Note: error < 1e-3 is acceptable for {self.n_groups} groups × {self.n_total} unknowns")
                    print(f"          (30 sparse solves with accumulated round-off)")
            
            # Test homogeneity: B(αx) = αB(x)
            alpha = 3.7
            Bax = B_operator.matvec(alpha * test_x)
            aBx = alpha * Bx
            hom_err_B = np.linalg.norm(Bax - aBx) / (np.linalg.norm(Bax) + 1e-30)
            print(f"  B operator homogeneity error: {hom_err_B:.3e} (should be ~0)")
            if hom_err_B > 1e-10:
                print(f"    WARNING: Large homogeneity error!")
                print(f"    ||B(αx)||  = {np.linalg.norm(Bax):.3e}")
                print(f"    ||αB(x)||  = {np.linalg.norm(aBx):.3e}")
                print(f"    ||difference|| = {np.linalg.norm(Bax - aBx):.3e}")
            
            if use_preconditioner:
                # Test C operator linearity
                Cx = C_operator.matvec(test_x)
                Cy = C_operator.matvec(test_y)
                Caxy = C_operator.matvec(a*test_x + b*test_y)
                lin_err_C = np.linalg.norm(Caxy - (a*Cx + b*Cy)) / (np.linalg.norm(Caxy) + 1e-30)
                print(f"  C operator linearity error: {lin_err_C:.3e} (should be ~0)")
        
        if verbose:
            print(f"  Solving with GMRES (tol={gmres_tol}, maxiter={gmres_maxiter})...")

        
        kappa, info = gmres(
            Aop, bop,
            x0=kappa_initial,
            rtol=gmres_tol,
            atol=gmres_tol*1e-2,
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
                print(f"  GMRES converged successfully in {iters} iterations")
            elif info > 0:
                print(f"  Warning: GMRES did not fully converge (reached maxiter={gmres_maxiter})")
            else:
                print(f"  Warning: GMRES illegal input or breakdown (info={info})")
            print(f"  DEBUG true residual: ||b - B·kappa||/||b|| = {rel_true_res:.3e}")
        
        return kappa, info_dict
    
    @staticmethod
    def _make_homogeneous_bc(bc_func):
        """
        Create a homogeneous version of a boundary condition function.
        
        For Robin BC: A·φ + B·∇φ = C, this sets C = 0 to get
        the homogeneous operator part: A·φ + B·∇φ = 0.
        
        This is needed for apply_operator_B which should only apply
        the operator part, not the forcing/source part of the BC.
        
        Parameters:
        -----------
        bc_func : callable or None
            Original BC function (phi, pos, t) -> (A, B, C)
        
        Returns:
        --------
        homogeneous_bc : callable or None
            Homogeneous BC function that returns (A, B, 0.0)
        """
        if bc_func is None:
            return None
        
        def homogeneous_bc(phi, pos, t):
            A, B, C = bc_func(phi, pos, t)
            return A, B, 0.0  # Force C = 0 for homogeneous operator
        
        return homogeneous_bc
    
    def apply_operator_B(self, kappa: np.ndarray, T_star: np.ndarray,
                        xi_g_list: List[np.ndarray], bc_time: float = 0.0) -> np.ndarray:
        """
        Apply operator B to vector κ.
        
        B·κ = κ - Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        
        CRITICAL: This applies the HOMOGENEOUS operator A_g^{-1} (with C_bc = 0).
        The inhomogeneous BC forcing is handled separately in compute_rhs_for_kappa.
        
        Parameters:
        -----------
        kappa : ndarray
            Input vector (1D flattened)
        T_star : ndarray
            Temperature (1D flattened)
        xi_g_list : list of ndarray
            Source terms (not used in B operator)
        bc_time : float
            Time used to evaluate time-dependent boundary conditions
        
        Returns:
        --------
        result : ndarray
            B·κ (1D flattened)
        """
        f = self.fleck_factor
        result = kappa.copy().astype(np.float64)
        
        # Reshape temperature to 2D for solvers
        T_2d = unflatten_2d(T_star, self.nx_cells, self.ny_cells)

        # Build homogeneous BC wrappers once per matvec call.
        override_left = [self._make_homogeneous_bc(self.left_bc_funcs[g]) for g in range(self.n_groups)] if self.left_bc_funcs is not None else None
        override_right = [self._make_homogeneous_bc(self.right_bc_funcs[g]) for g in range(self.n_groups)] if self.right_bc_funcs is not None else None
        override_bottom = [self._make_homogeneous_bc(self.bottom_bc_funcs[g]) for g in range(self.n_groups)] if self.bottom_bc_funcs is not None else None
        override_top = [self._make_homogeneous_bc(self.top_bc_funcs[g]) for g in range(self.n_groups)] if self.top_bc_funcs is not None else None
        
        # Subtract Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        for g in range(self.n_groups):
            # RHS for group g: χ_g(1-f)κ
            rhs_g = self.chi[g] * (1.0 - f) * kappa
            
            # Reshape to 2D for solver
            rhs_g_2d = unflatten_2d(rhs_g, self.nx_cells, self.ny_cells)
            
            # Solve A_g φ_g = rhs_g with HOMOGENEOUS BCs
            phi_g_2d = self.solvers[g].solve(
                rhs_g_2d, T_2d,
                bc_time=bc_time,
                override_left_bc=override_left[g] if override_left is not None else None,
                override_right_bc=override_right[g] if override_right is not None else None,
                override_bottom_bc=override_bottom[g] if override_bottom is not None else None,
                override_top_bc=override_top[g] if override_top is not None else None
            )
            
            # Flatten back to 1D
            phi_g = flatten_2d(phi_g_2d, self.nx_cells, self.ny_cells)
            
            # Subtract σ*_{a,g} φ_g
            result -= self.sigma_a[g, :] * phi_g
        
        return result
    
    def compute_rhs_for_kappa(self, T_star: np.ndarray,
                             xi_g_list: List[np.ndarray], bc_time: float = 0.0) -> np.ndarray:
        """
        Compute RHS for κ equation.
        
        RHS = Σ_g σ*_{a,g}·A_g^{-1}·ξ_g
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature (1D flattened)
        xi_g_list : list of ndarray
            Source terms for each group (1D flattened)
        bc_time : float
            Time used to evaluate time-dependent boundary conditions
        
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
            phi_g_2d = self.solvers[g].solve(xi_g_2d, T_2d, bc_time=bc_time)
            
            # Flatten back to 1D
            phi_g = flatten_2d(phi_g_2d, self.nx_cells, self.ny_cells)
            
            # Add σ*_{a,g} φ_g
            rhs += self.sigma_a[g, :] * phi_g
        
        return rhs
    
    def compute_radiation_energy_from_kappa(self, kappa: np.ndarray, T_star: np.ndarray,
                                           xi_g_list: List[np.ndarray], bc_time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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
        bc_time : float
            Time used to evaluate time-dependent boundary conditions
        
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
            phi_g_2d = self.solvers[g].solve(rhs_g_2d, T_2d, bc_time=bc_time)
            
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
   
    def update_temperature(self, kappa: np.ndarray, T_star: np.ndarray,
                           B_star: Optional[np.ndarray] = None,
                           e_star: Optional[np.ndarray] = None,
                           e_n: Optional[np.ndarray] = None) -> np.ndarray:
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
        
        if e_n is None:
            e_n = self._compute_material_energy(self.T_old)

        if e_star is None:
            e_star = self._compute_material_energy(T_star)
        Delta_e = e_star - e_n

        if B_star is None:
            B_star, _ = self._compute_planck_arrays(T_star)
        sum_planck = 4.0 * np.pi * np.sum(self.sigma_a * B_star, axis=0)
        
        # New material energy
        e_new = e_n + self.dt * f * (kappa - sum_planck) + (1.0 - f) * Delta_e
        
        # Invert to get temperature
        T_new = np.vectorize(self.inverse_material_energy_func)(e_new, self.X_flat, self.Y_flat)
        
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

        # Compute gray operator coefficients (vectorised where possible)
        # σ_a_gray = Σ_g σ_a[g,:] * λ̃[g,:]  — pure NumPy, no loop
        sigma_a_gray = np.sum(self.sigma_a * lambda_tilde, axis=0)  # (n_total,)

        # D_gray requires evaluating the user diffusion functions for each group/cell.
        # Outer loop over groups (small), inner list-comp avoids attribute lookups.
        D_gray = np.zeros(self.n_total)
        _Xf = self.X_flat
        _Yf = self.Y_flat
        for g in range(self.n_groups):
            _D_g = self.diffusion_coeff_funcs[g]
            D_g_vals = np.array([_D_g(T_star[i], _Xf[i], _Yf[i])
                                  for i in range(self.n_total)])
            D_gray += D_g_vals * lambda_tilde[g, :]
        
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
        """
        if self._use_default_planck:
            _, dB_star = _planck_multicell(self.energy_edges, T_star)
        else:
            _, dB_star = self._compute_planck_arrays(T_star)

        dB_sum = np.sum(dB_star, axis=0)          # (n_total,)
        safe_sum = np.where(dB_sum > 0, dB_sum, 1.0)
        lambda_tilde = dB_star / safe_sum[np.newaxis, :]
        lambda_tilde[:, dB_sum <= 0] = 1.0 / self.n_groups
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
