#!/usr/bin/env python3
"""
Multigroup Non-Equilibrium Diffusion Solver

Solves the multigroup non-equilibrium diffusion system (equations 8.83-8.91):

For each group g:
    A_g φ_g^{n+1} = χ_g(1-f)κ + ξ_g + Q_g

where:
    A_g = -∇·D_g(T_★)∇ + (σ*_{a,g} + 1/(cΔt))
    κ = Σ_g σ*_{a,g} φ_g^{n+1}  (absorption rate density)
    Q_g = external radiation source (optional)
    D_g = diffusion coefficient (can include flux limiters)

Flux Limiters:
--------------
The solver supports flux-limited diffusion where D_g = λ(R_g) / σ_{R,g}
with R_g = |∇φ_g| / (σ_{R,g} · φ_g). Available limiters include:
  - flux_limiter_standard: λ = 1/3 (no limiting, standard diffusion)
  - flux_limiter_levermore_pomraning: λ = (2+R)/(6+3R+R²)
  - flux_limiter_larsen: λ = (3^n + R^n)^(-1/n)
  - flux_limiter_sum: λ = 1/(3+R)
  - flux_limiter_max: λ = 1/max(3,R)

Algorithm:
1. Solve Bκ = RHS for κ using GMRES
   where B = I - Σ_g σ*_{a,g} A_g^{-1} χ_g(1-f)
   and RHS = Σ_g σ*_{a,g} A_g^{-1} ξ_g

2. Compute φ_g^{n+1} = A_g^{-1}(χ_g(1-f)κ + ξ_g) for each group

3. Update material temperature using:
   e(T_{n+1}) - e(T_n) = Δt·f·Σ_g σ*_{a,g}(φ_g^{n+1} - 4πB_g(T_★)) + (1-f)Δe

4. Iterate until convergence

Planck Function Integration:
-----------------------------
This solver uses the planck_integrals library for accurate computation of
group-integrated Planck functions Bg(T) = ∫_{E_g}^{E_{g+1}} B_E(T) dE using
rational polynomial approximations. The library provides:
  - Bg(E_low, E_high, T): Incomplete Planck integral
  - Bg_multigroup(energy_edges, T): All groups in parallel
  - dBgdT(E_low, E_high, T): Temperature derivative (Rosseland integral)

If the library is not available, the solver falls back to gray approximation.
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from typing import Callable, Optional, List, Tuple
import time

# Try to import the diffusion operator solvers
try:
    from diffusion_operator_solver import (
        DiffusionOperatorSolver1D, 
        DiffusionOperatorSolver2D,
        C_LIGHT, A_RAD
    )
except ImportError:
    print("Warning: Could not import from diffusion_operator_solver")
    C_LIGHT = 2.99792458e1  # cm/ns
    A_RAD = 0.01372       # GJ/(cm³·keV⁴)
    raise ImportError("diffusion_operator_solver module is required but not found.")

# Import Planck integral library (required)
from planck_integrals import Bg, dBgdT, Bg_multigroup, dBgdT_multigroup

SIGMA_SB = C_LIGHT * A_RAD / 4  # Stefan-Boltzmann constant


# =============================================================================
# FLUX LIMITER FUNCTIONS
# =============================================================================

def flux_limiter_standard(R):
    """Standard diffusion: λ = 1/3 (no limiting)
    
    Parameters:
    -----------
    R : float or ndarray
        Dimensionless ratio R = |∇φ|/(σ_R * φ)
    
    Returns:
    --------
    lambda : float or ndarray
        Flux limiter value
    """
    if isinstance(R, np.ndarray):
        return np.full_like(R, 1.0/3.0)
    return 1.0/3.0


def flux_limiter_levermore_pomraning(R):
    """Levermore-Pomraning flux limiter
    
    λ^LP(R) = (1/R)(coth R - 1/R) ≈ (2+R)/(6+3R+R²)
    
    Parameters:
    -----------
    R : float or ndarray
        Dimensionless ratio R = |∇φ|/(σ_R * φ)
    
    Returns:
    --------
    lambda : float or ndarray
        Flux limiter value
    """
    R = np.atleast_1d(R)
    result = (2.0 + R) / (6.0 + 3.0*R + R**2)
    return result if len(result) > 1 else result[0]


def flux_limiter_larsen(R, n=2.0):
    """Larsen flux limiter
    
    λ^L(R) = (3^n + R^n)^(-1/n)
    
    Parameters:
    -----------
    R : float or ndarray
        Dimensionless ratio R = |∇φ|/(σ_R * φ)
    n : float
        Exponent parameter (default: 2.0)
    
    Returns:
    --------
    lambda : float or ndarray
        Flux limiter value
    """
    return (3.0**n + R**n)**(-1.0/n)


def flux_limiter_sum(R):
    """Sum flux limiter
    
    λ^sum(R) = 1/(3+R)
    
    Parameters:
    -----------
    R : float or ndarray
        Dimensionless ratio R = |∇φ|/(σ_R * φ)
    
    Returns:
    --------
    lambda : float or ndarray
        Flux limiter value
    """
    return 1.0 / (3.0 + R)


def flux_limiter_max(R):
    """Max flux limiter
    
    λ^max(R) = 1/max(3, R)
    
    Parameters:
    -----------
    R : float or ndarray
        Dimensionless ratio R = |∇φ|/(σ_R * φ)
    
    Returns:
    --------
    lambda : float or ndarray
        Flux limiter value
    """
    if isinstance(R, np.ndarray):
        return 1.0 / np.maximum(3.0, R)
    return 1.0 / max(3.0, R)


# =============================================================================
# MULTIGROUP PLANCK FUNCTION AND EMISSION FRACTIONS
# =============================================================================

def planck_function_integral(T, E_min, E_max):
    """
    Integrate Planck function over energy range [E_min, E_max].
    
    B_g(T) = ∫_{E_min}^{E_max} B_E(T) dE
    
    Uses the planck_integrals library for accurate computation via
    rational approximations to the incomplete Planck integral.
    
    Parameters:
    -----------
    T : float or ndarray
        Temperature (keV)
    E_min, E_max : float
        Energy bounds for group (keV)
    
    Returns:
    --------
    B_g : float or ndarray
        Integrated Planck function, units: GJ/(cm²·ns)
        Note: The library returns (σ_SB/π)·T⁴·[Π(x_high) - Π(x_low)]
    """
    return Bg(E_min, E_max, T)


def compute_emission_fractions_gray(n_groups):
    """
    Compute emission fractions χ_g for gray approximation.
    
    For gray split into n_groups, each group gets equal fraction:
    χ_g = 1/n_groups
    
    Parameters:
    -----------
    n_groups : int
        Number of groups
    
    Returns:
    --------
    chi : ndarray
        Emission fractions [χ_1, χ_2, ..., χ_G]
    """
    return np.ones(n_groups) / n_groups


def compute_emission_fractions_from_edges(energy_edges, T_ref=1.0, sigma_a_groups=None):
    """
    Compute emission fractions χ_g from energy group edges.
    
    χ_g(T) = (σ*_{a,g} · ∂B_g/∂T) / (Σ_{g'} σ*_{a,g'} · ∂B_g'/∂T)
    
    Uses the Rosseland integral (temperature derivative of Planck function)
    via the planck_integrals library dBgdT_multigroup function.
    
    If absorption coefficients are not provided, assumes σ*_{a,g} = 1 for all groups.
    
    Parameters:
    -----------
    energy_edges : ndarray
        Energy group edges (length n_groups + 1), in keV
    T_ref : float
        Reference temperature for computing fractions (keV)
    sigma_a_groups : ndarray or None
        Absorption coefficients for each group (length n_groups)
        If None, assumes equal absorption (σ_a = 1 for all groups)
    
    Returns:
    --------
    chi : ndarray
        Emission fractions [χ_0, χ_1, ..., χ_{G-1}]
    """
    n_groups = len(energy_edges) - 1
    
    # Default to equal absorption if not provided
    if sigma_a_groups is None:
        sigma_a_groups = np.ones(n_groups)
    
    # Use accurate Rosseland integration (dB/dT)
    dB_groups = dBgdT_multigroup(energy_edges, T_ref)
    
    # Weight by absorption coefficients
    weighted_dB = sigma_a_groups * dB_groups
    
    # Normalize to get emission fractions
    total_weighted = np.sum(weighted_dB)
    chi = weighted_dB / total_weighted
    
    return chi


# =============================================================================
# MULTIGROUP DIFFUSION SOLVER CLASS
# =============================================================================

class MultigroupDiffusionSolver1D:
    """
    1D Multigroup Non-Equilibrium Diffusion Solver.
    
    Solves the coupled system using the reduction to absorption rate density κ
    as described in equations 8.83-8.91.
    """
    
    def __init__(self,
                 n_groups: int,
                 r_min: float,
                 r_max: float,
                 n_cells: int,
                 energy_edges: np.ndarray,
                 geometry: str = 'planar',
                 dt: float = 1e-3,
                 diffusion_coeff_funcs: Optional[List[Callable]] = None,
                 absorption_coeff_funcs: Optional[List[Callable]] = None,
                 left_bc_funcs: Optional[List[Callable]] = None,
                 right_bc_funcs: Optional[List[Callable]] = None,
                 source_funcs: Optional[List[Callable]] = None,
                 flux_limiter_funcs: Optional[List[Callable]] = None,
                 rosseland_opacity_funcs: Optional[List[Callable]] = None,
                 emission_fractions: Optional[np.ndarray] = None,
                 planck_funcs: Optional[List[Callable]] = None,
                 dplanck_dT_funcs: Optional[List[Callable]] = None,
                 material_energy_func: Optional[Callable] = None,
                 inverse_material_energy_func: Optional[Callable] = None,
                 rho: float = 1.0,
                 cv: float = 1.0):
        """
        Initialize multigroup solver.
        
        Parameters:
        -----------
        n_groups : int
            Number of groups G
        r_min, r_max : float
            Domain boundaries
        n_cells : int
            Number of spatial cells
        geometry : str
            'planar', 'cylindrical', or 'spherical'
        dt : float
            Time step
        energy_edges : ndarray
            Energy group edges (length n_groups + 1), in keV
            Example: [0.0, 0.5, 1.0, 5.0] for 3 groups
        diffusion_coeff_funcs : list of callable or None
            List of G functions D_g(T, r) for each group
            If None, uses constant D = 1.0 for all groups
        absorption_coeff_funcs : list of callable or None
            List of G functions σ_{a,g}(T, r) for each group
            If None, uses constant σ_a = 1.0 for all groups
        left_bc_funcs, right_bc_funcs : list of callable or None
            List of G boundary condition functions for each group
            Each function returns (A, B, C) for Robin BC: A·φ + B·∇φ = C
            Function signature: (phi_boundary, r_boundary) -> (A, B, C)
            If None, uses default reflecting BC (0, 1, 0) for all groups
        source_funcs : list of callable or None
            List of G external source functions Q_g(r, t) for each group
            Each function has signature: (r, t) -> Q_g
            Units: same as ξ_g (GJ/(cm³·ns) or similar)
            If None, no external source (Q_g = 0 for all groups)
        flux_limiter_funcs : list of callable or None
            List of G flux limiter functions λ(R) for each group
            Each function has signature: (R) -> λ where R = |∇φ|/(σ_R·φ)
            If None, uses standard diffusion (λ = 1/3) for all groups
            Can also be a single function to use for all groups
        rosseland_opacity_funcs : list of callable or None
            List of G Rosseland opacity functions σ_R,g(T, r) for each group
            Required if using flux limiters. If None, assumes σ_R = σ_a
            Can also be a single function to use for all groups
        emission_fractions : ndarray or None
            Optional manual override of emission fractions χ_g (length n_groups)
            If provided, uses these values instead of computing from Rosseland integrals
            Must sum to 1.0. Useful for gray approximations where equal fractions are desired.
            If None, computes from energy_edges using Rosseland integrals.
        planck_funcs : list of callable or None
            Optional user-defined Planck functions B_g(T) for each group
            Each function has signature: (T) -> B_g where T is temperature (keV)
            Returns integrated Planck function in GJ/(cm²·ns)
            If None, uses planck_integrals library: Bg(E_low, E_high, T)
            Can also be a single function to use for all groups (will be replicated)
        dplanck_dT_funcs : list of callable or None
            Optional user-defined temperature derivatives dB_g/dT(T) for each group
            Each function has signature: (T) -> dB_g/dT in GJ/(cm²·ns·keV)
            If None, uses planck_integrals library: dBgdT(E_low, E_high, T)
            Can also be a single function to use for all groups (will be replicated)
            Required for computing Fleck factors and emission fractions
        material_energy_func : callable or None
            Function e(T) returning material energy density
            If None, uses e = ρ·c_v·T
        inverse_material_energy_func : callable or None
            Function T(e) inverse of material energy
            If None, uses T = e/(ρ·c_v)
        rho : float
            Material density (g/cm³)
        cv : float or callable
            Specific heat (GJ/(g·keV))
            Can be a constant or a function c_v(T) for temperature-dependent specific heat
            For radiation-dominated materials with e=a·T^4, use c_v(T) = 4a·T^3/ρ
        """
        self.n_groups = n_groups
        self.r_min = r_min
        self.r_max = r_max
        self.n_cells = n_cells
        self.geometry = geometry
        self.dt = dt
        self.rho = rho
        
        # Store cv as either scalar or function
        if callable(cv):
            self.cv_func = cv
            self.cv_is_function = True
        else:
            self.cv = cv
            self.cv_is_function = False
        self.t = 0.0  # Current time
        
        # Fleck factor array (computed locally for each cell)
        self.fleck_factor = np.ones(n_cells, dtype=np.float64)
        
        # Store energy group edges (required)
        self.energy_edges = np.array(energy_edges, dtype=np.float64)
        if len(self.energy_edges) != n_groups + 1:
            raise ValueError(f"energy_edges must have length n_groups+1 = {n_groups+1}, got {len(self.energy_edges)}")
        
        # Initialize group solvers
        self.solvers = []
        
        if diffusion_coeff_funcs is None:
            diffusion_coeff_funcs = [lambda T, r: 1.0] * n_groups
        
        if absorption_coeff_funcs is None:
            absorption_coeff_funcs = [lambda T, r: 1.0] * n_groups
        
        if left_bc_funcs is None:
            # Default: reflecting BC (zero flux)
            left_bc_funcs = [lambda phi, r: (0.0, 1.0, 0.0)] * n_groups
        
        if right_bc_funcs is None:
            # Default: reflecting BC (zero flux)
            right_bc_funcs = [lambda phi, r: (0.0, 1.0, 0.0)] * n_groups
        
        # External source functions Q_g(r, t)
        if source_funcs is None:
            # Default: no external source
            source_funcs = [lambda r, t: 0.0] * n_groups
        self.source_funcs = source_funcs
        
        # Flux limiter functions
        if flux_limiter_funcs is None:
            # Default: standard diffusion (no limiting)
            self.flux_limiter_funcs = None
            self.use_flux_limiters = False
        else:
            self.use_flux_limiters = True
            # Allow single function for all groups or list of functions
            if callable(flux_limiter_funcs):
                self.flux_limiter_funcs = [flux_limiter_funcs] * n_groups
            else:
                self.flux_limiter_funcs = flux_limiter_funcs
            
            # Rosseland opacity functions (required for flux limiters)
            if rosseland_opacity_funcs is None:
                # Default: use absorption opacity as Rosseland opacity
                self.rosseland_opacity_funcs = absorption_coeff_funcs
            elif callable(rosseland_opacity_funcs):
                self.rosseland_opacity_funcs = [rosseland_opacity_funcs] * n_groups
            else:
                self.rosseland_opacity_funcs = rosseland_opacity_funcs
        
        # Wrap diffusion coefficient functions with flux limiting if requested
        if self.use_flux_limiters:
            diffusion_coeff_funcs_wrapped = []
            for g in range(n_groups):
                base_diffusion_func = diffusion_coeff_funcs[g] if diffusion_coeff_funcs else (lambda T, r: 1.0)
                limiter_func = self.flux_limiter_funcs[g]
                rosseland_func = self.rosseland_opacity_funcs[g]
                
                # Create flux-limited diffusion coefficient function
                def make_flux_limited_diff(base_func, limiter, ross_func, group_idx=g):
                    R_values_seen = []  # Track R values for diagnostics
                    
                    def flux_limited_diff(T, r, phi_left, phi_right, dx):
                        """Flux-limited diffusion: D = λ(R) / σ_R"""
                        # Get Rosseland opacity
                        sigma_R = ross_func(T, r)
                        
                        # Compute flux ratio R = |∇φ| / (σ_R · φ)
                        phi_avg = 0.5 * (phi_left + phi_right) + 1e-14
                        grad_phi = abs(phi_right - phi_left) / (dx + 1e-14)
                        R = grad_phi / (sigma_R * phi_avg + 1e-14)
                        
                        # Track R values (only store a few for diagnostics)
                        if len(R_values_seen) < 5 and R > 0.5:
                            R_values_seen.append(R)
                            if len(R_values_seen) == 1:
                                print(f"    [Group {group_idx}] First significant R = {R:.4f} at r={r:.4f}")
                        
                        # Apply flux limiter
                        lambda_val = limiter(R)
                        
                        # Return D = λ / σ_R
                        return lambda_val / (sigma_R + 1e-14)
                    return flux_limited_diff
                
                diffusion_coeff_funcs_wrapped.append(
                    make_flux_limited_diff(base_diffusion_func, limiter_func, rosseland_func, g)
                )
            diffusion_coeff_funcs = diffusion_coeff_funcs_wrapped
        
        for g in range(n_groups):
            solver = DiffusionOperatorSolver1D(
                r_min=r_min,
                r_max=r_max,
                n_cells=n_cells,
                geometry=geometry,
                diffusion_coeff_func=diffusion_coeff_funcs[g],
                absorption_coeff_func=absorption_coeff_funcs[g],
                dt=dt,
                left_bc_func=left_bc_funcs[g],
                right_bc_func=right_bc_funcs[g]
            )
            self.solvers.append(solver)
        
        # Store user-defined Planck functions or use library defaults
        if planck_funcs is not None:
            if callable(planck_funcs):
                # Single function provided, replicate for all groups
                self.planck_funcs = [planck_funcs] * n_groups
                self.user_planck = True
            else:
                # List of functions provided
                if len(planck_funcs) != n_groups:
                    raise ValueError(f"planck_funcs must have length n_groups = {n_groups}")
                self.planck_funcs = planck_funcs
                self.user_planck = True
            print(f"  Using user-defined Planck functions")
        else:
            # Use library functions
            self.planck_funcs = [
                lambda T, g=g: Bg(self.energy_edges[g], self.energy_edges[g+1], T)
                for g in range(n_groups)
            ]
            self.user_planck = False
        
        # Store user-defined dB/dT functions or use library defaults
        if dplanck_dT_funcs is not None:
            if callable(dplanck_dT_funcs):
                # Single function provided, replicate for all groups
                self.dplanck_dT_funcs = [dplanck_dT_funcs] * n_groups
                self.user_dplanck = True
            else:
                # List of functions provided
                if len(dplanck_dT_funcs) != n_groups:
                    raise ValueError(f"dplanck_dT_funcs must have length n_groups = {n_groups}")
                self.dplanck_dT_funcs = dplanck_dT_funcs
                self.user_dplanck = True
            print(f"  Using user-defined dB/dT functions")
        else:
            # Use library functions
            self.dplanck_dT_funcs = [
                lambda T, g=g: dBgdT(self.energy_edges[g], self.energy_edges[g+1], T)
                for g in range(n_groups)
            ]
            self.user_dplanck = False
        
        # Compute emission fractions χ_g from energy edges (or use manual override)
        if emission_fractions is not None:
            self.chi = np.array(emission_fractions, dtype=np.float64)
            if len(self.chi) != n_groups:
                raise ValueError(f"emission_fractions must have length n_groups = {n_groups}")
            if abs(np.sum(self.chi) - 1.0) > 1e-10:
                raise ValueError(f"emission_fractions must sum to 1.0, got {np.sum(self.chi)}")
            print(f"  Using manual emission fractions: {self.chi}")
        else:
            # Compute from dB/dT using either user functions or library functions
            self.chi = self._compute_emission_fractions(T_ref=1.0)
            print(f"  Computed emission fractions: {self.chi}")
        
        # Material energy functions
        if material_energy_func is None:
            self.material_energy_func = lambda T: self.rho * self.cv * T
        else:
            self.material_energy_func = material_energy_func
        
        if inverse_material_energy_func is None:
            self.inverse_material_energy_func = lambda e: e / (self.rho * self.cv)
        else:
            self.inverse_material_energy_func = inverse_material_energy_func
        
        # Solution arrays - store only κ and total E_r
        self.r_centers = self.solvers[0].r_centers
        self.kappa = np.zeros(n_cells, dtype=np.float64)  # Absorption rate density κ
        self.E_r = np.ones(n_cells, dtype=np.float64) * A_RAD * 0.01**4  # Total radiation energy density
        self.T = np.ones(n_cells, dtype=np.float64)  # Material temperature
        self.kappa_old = np.zeros(n_cells, dtype=np.float64)
        self.E_r_old = np.ones(n_cells, dtype=np.float64) * A_RAD * 0.01**4
        self.T_old = np.ones(n_cells, dtype=np.float64)
        
        # Store fractional distribution of phi_g for accurate source term computation
        # phi_g_fraction[g,:] = phi_g / phi_total (stored from previous timestep)
        self.phi_g_fraction = np.ones((n_groups, n_cells)) / n_groups  # Initialize uniform
        
        # Store phi_g values for each group (for flux limiter phi_guess)
        # Initialize with small non-zero values
        self.phi_g_stored = np.ones((n_groups, n_cells), dtype=np.float64) * A_RAD * C_LIGHT * 0.01**4 / n_groups
        
        # Store absorption coefficients (evaluated at cell centers)
        self.sigma_a = np.zeros((n_groups, n_cells))
        
        print(f"Initialized multigroup solver:")
        print(f"  Groups: {n_groups}")
        print(f"  Cells: {n_cells}")
        print(f"  Geometry: {geometry}")
        print(f"  Energy edges (keV): {self.energy_edges}")
        print(f"  Fleck factor: computed locally for each cell")
        print(f"  Emission fractions χ: {self.chi}")
        print(f"  Using flux limiters: {self.use_flux_limiters}")
    
    def _compute_emission_fractions(self, T_ref: float = 1.0) -> np.ndarray:
        """
        Compute emission fractions χ_g from dB/dT evaluated at reference temperature.
        
        χ_g = (σ*_{a,g} · dB_g/dT) / (Σ_{g'} σ*_{a,g'} · dB_g'/dT)
        
        Uses self.dplanck_dT_funcs which may be user-defined or library functions.
        
        Parameters:
        -----------
        T_ref : float
            Reference temperature for computing fractions (keV)
        
        Returns:
        --------
        chi : ndarray
            Emission fractions [χ_0, χ_1, ..., χ_{G-1}]
        """
        # Evaluate dB/dT for all groups at reference temperature
        dB_groups = np.array([self.dplanck_dT_funcs[g](T_ref) for g in range(self.n_groups)])
        
        # For emission fractions, assume equal absorption if not yet initialized
        # (This is only called during initialization before sigma_a is set up)
        sigma_a_groups = np.ones(self.n_groups)
        
        # Weight by absorption coefficients
        weighted_dB = sigma_a_groups * dB_groups
        
        # Normalize to get emission fractions
        total_weighted = np.sum(weighted_dB)
        if total_weighted > 0:
            chi = weighted_dB / total_weighted
        else:
            # Fallback to uniform if something goes wrong
            chi = np.ones(self.n_groups) / self.n_groups
        
        return chi
    
    def update_absorption_coefficients(self, T: np.ndarray):
        """
        Update absorption coefficients σ*_{a,g} at current temperature.
        
        Parameters:
        -----------
        T : ndarray
            Temperature at cell centers
        """
        for g in range(self.n_groups):
            for i in range(self.n_cells):
                self.sigma_a[g, i] = self.solvers[g].absorption_coeff_func(T[i], self.r_centers[i])
    
    def compute_fleck_factor(self, T: np.ndarray) -> np.ndarray:
        """
        Compute local Fleck factor for each cell.
        
        f = 1 / (1 + 4π (Δt/C_v) Σ_{g=1}^G σ*_{a,g} ∂B_g/∂T)
        
        Uses temperature-dependent Rosseland integrals to compute the
        coupling factor between radiation and material.
        
        Parameters:
        -----------
        T : ndarray
            Temperature at cell centers
        
        Returns:
        --------
        f : ndarray
            Fleck factor for each cell
        """
        # Compute local Fleck factor
        f = np.ones(self.n_cells, dtype=np.float64)
        
        for i in range(self.n_cells):
            # Compute Σ_g σ*_{a,g} ∂B_g/∂T at this cell
            sum_sigma_dB = 0.0
            for g in range(self.n_groups):
                # Use user-defined or library dB/dT function
                dB_g = self.dplanck_dT_funcs[g](T[i])
                sum_sigma_dB += self.sigma_a[g, i] * dB_g
            
            # Get cv at this temperature (if temperature-dependent)
            if self.cv_is_function:
                cv_i = self.cv_func(T[i])
            else:
                cv_i = self.cv
            
            # f = 1 / (1 + 4π (Δt/C_v) Σ_g σ*_{a,g} ∂B_g/∂T)
            denominator = 1.0 + 4.0 * np.pi * (self.dt / cv_i) * sum_sigma_dB
            f[i] = 1.0 / denominator
        
        return f
    
    def compute_source_xi(self, g: int, T_star: np.ndarray, t: float) -> np.ndarray:
        """
        Compute source term ξ_g for group g (equation 8.84 plus external source).
        
        ξ_g = (1/cΔt)φ_g^n + 4π·σ*_{a,g}·B_g(T_★) - χ_g(1-f)·[Σ_{g'} 4π·σ*_{a,g'}·B_{g'}(T_★) + Δe/Δt] + Q_g(r,t)
        
        Parameters:
        -----------
        g : int
            Group index
        T_star : ndarray
            Linearization temperature T_★
        t : float
            Current time (for external source evaluation)
        
        Returns:
        --------
        xi_g : ndarray
            Source term for group g
        """
        # Get local Fleck factor for each cell
        f = self.fleck_factor
        
        # Material energy change
        e_star = self.material_energy_func(T_star)
        e_n = self.material_energy_func(self.T_old)
        Delta_e = e_star - e_n
        
        # Planck functions
        B_g_star = np.array([self.planck_funcs[g](T_star[i]) for i in range(self.n_cells)])
        
        # Sum over all groups for emission term only
        sum_emission = np.zeros(self.n_cells)
        for gp in range(self.n_groups):
            B_gp_star = np.array([self.planck_funcs[gp](T_star[i]) for i in range(self.n_cells)])
            sum_emission += self.sigma_a[gp, :] * 4.0 * np.pi * B_gp_star
        
        # Coupling term: [Σ_{g'} 4π·σ*_{a,g'}·B_{g'}(T_★) + Δe/Δt]
        coupling_term = sum_emission + Delta_e / self.dt
        
        # Debug: Check for huge values
        if np.any(np.abs(coupling_term) > 1e10):
            print(f"    WARNING in compute_source_xi for group {g}:")
            print(f"      sum_emission max: {np.max(np.abs(sum_emission)):.3e}")
            print(f"      Delta_e/dt max: {np.max(np.abs(Delta_e / self.dt)):.3e}")
            print(f"      coupling_term max: {np.max(np.abs(coupling_term)):.3e}")
            print(f"      This will cause huge source terms and GMRES failure!")
        
        # Assemble ξ_g
        # Use phi_g^n from stored fractional distribution
        phi_total_old = self.E_r_old * C_LIGHT
        phi_g_old = phi_total_old * self.phi_g_fraction[g, :]
        
        xi_g = (1.0 / (C_LIGHT * self.dt)) * phi_g_old + \
               4.0 * np.pi * self.sigma_a[g, :] * B_g_star - \
               self.chi[g] * (1.0 - f) * coupling_term
        
        # Add external source Q_g(r, t)
        Q_g = np.array([self.source_funcs[g](self.r_centers[i], t) for i in range(self.n_cells)])
        xi_g += Q_g
        
        # Debug: Check final xi_g
        if np.any(np.abs(xi_g) > 1e10):
            print(f"    WARNING: xi_{g} has huge values! max: {np.max(np.abs(xi_g)):.3e}")
        
        return xi_g
    
    def apply_operator_B(self, kappa: np.ndarray, T_star: np.ndarray, xi_g_list: List[np.ndarray]) -> np.ndarray:
        """
        Apply operator B to vector κ (equation 8.91).
        
        B·κ = κ - Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        
        Parameters:
        -----------
        kappa : ndarray
            Input vector (absorption rate density)
        T_star : ndarray
            Temperature for evaluating operators
        xi_g_list : list of ndarray
            Precomputed ξ_g for each group (not used in B operator, but kept for consistency)
        
        Returns:
        --------
        result : ndarray
            B·κ
        """
        # Get local Fleck factor for each cell
        f = self.fleck_factor
        result = kappa.copy().astype(np.float64)
        
        # Debug: Check input
        if np.any(np.abs(kappa) > 1e10) or np.any(~np.isfinite(kappa)):
            print(f"    WARNING in B operator: bad input kappa! max: {np.max(np.abs(kappa)):.3e}")
        
        # Define homogeneous boundary conditions for B operator
        # The actual BC forcing goes into the RHS, not the operator
        # Use same A and B as actual BCs, but set C = 0
        def make_homogeneous_bc(bc_func):
            """Convert a Robin BC function to its homogeneous version (C=0)"""
            def homogeneous_bc(phi, r):
                A, B, C = bc_func(phi, r)
                return A, B, 0.0  # Keep A and B, set C = 0
            return homogeneous_bc
        
        # Subtract Σ_g σ*_{a,g}·A_g^{-1}[χ_g(1-f)κ]
        for g in range(self.n_groups):
            # RHS for group g: χ_g(1-f)κ
            rhs_g = self.chi[g] * (1.0 - f) * kappa
            
            # Solve A_g φ_g = rhs_g with HOMOGENEOUS BCs
            # NOTE: This is critical! The B operator must use homogeneous BCs.
            # The actual boundary forcing goes into the RHS computation.
            homogeneous_left_bc = make_homogeneous_bc(self.solvers[g].left_bc_func)
            homogeneous_right_bc = make_homogeneous_bc(self.solvers[g].right_bc_func)
            
            phi_g = self.solvers[g].solve(rhs_g, T_star, phi_guess=self.phi_g_stored[g, :],
                                          override_left_bc=homogeneous_left_bc,
                                          override_right_bc=homogeneous_right_bc)
            
            # Debug: Check if A_g solve produced bad values
            if np.any(~np.isfinite(phi_g)) or np.any(np.abs(phi_g) > 1e10):
                print(f"    WARNING: Group {g} A_g solve produced bad phi! max: {np.max(np.abs(phi_g)):.3e}")
                print(f"             rhs_g max: {np.max(np.abs(rhs_g)):.3e}")
            
            # Subtract σ*_{a,g} φ_g
            result -= self.sigma_a[g, :] * phi_g
        
        # Debug: Check result
        if np.any(np.abs(result) > 1e10) or np.any(~np.isfinite(result)):
            print(f"    WARNING: B operator result has bad values! max: {np.max(np.abs(result)):.3e}")
        
        return result
    
    def compute_rhs_for_kappa(self, T_star: np.ndarray, xi_g_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute RHS for κ equation (equation 8.90).
        
        RHS = Σ_g σ*_{a,g}·A_g^{-1}·ξ_g
        
        Parameters:
        -----------
        T_star : ndarray
            Temperature for evaluating operators
        xi_g_list : list of ndarray
            Source terms ξ_g for each group
        
        Returns:
        --------
        rhs : ndarray
            Right-hand side for κ equation
        """
        rhs = np.zeros(self.n_cells)
        
        for g in range(self.n_groups):
            # Solve A_g φ_g = ξ_g (use previous timestep's phi as guess for flux limiter)
            # DO NOT update phi_g_stored here - would make B operator non-linear!
            phi_g = self.solvers[g].solve(xi_g_list[g], T_star, phi_guess=self.phi_g_stored[g, :])
            
            # Add σ*_{a,g} φ_g
            rhs += self.sigma_a[g, :] * phi_g
        
        # Debug: Check RHS magnitude
        if np.any(np.abs(rhs) > 1e10):
            print(f"    WARNING: RHS for kappa has huge values! max: {np.max(np.abs(rhs)):.3e}")
        
        return rhs
    
    def solve_for_kappa(self, T_star: np.ndarray, xi_g_list: List[np.ndarray],
                       gmres_tol: float = 1e-6, gmres_maxiter: int = 200,
                       verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Solve B·κ = RHS for κ using GMRES (equation 8.90).
        
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
        verbose : bool
            Print GMRES convergence info
        
        Returns:
        --------
        kappa : ndarray
            Absorption rate density κ
        info_dict : dict
            Information about GMRES convergence
        """
        # Compute RHS
        rhs = self.compute_rhs_for_kappa(T_star, xi_g_list)
        
        if verbose:
            print(f"  RHS for kappa: max = {np.max(np.abs(rhs)):.3e}, contains NaN/Inf: {np.any(~np.isfinite(rhs))}")
        
        # Create linear operator for B
        def matvec(kappa_vec):
            return self.apply_operator_B(kappa_vec, T_star, xi_g_list)
        
        B_operator = LinearOperator((self.n_cells, self.n_cells), matvec=matvec)
        
        # Initial guess for kappa
        # Use E_r_old to estimate: kappa ≈ Σ_g σ_{a,g} * χ_g * E_r * c
        # Or use zero for robustness
        # kappa_initial = np.sum(self.sigma_a * self.chi[:, np.newaxis] * self.E_r_old * C_LIGHT, axis=0)
        kappa_initial = np.zeros(self.n_cells)  # Try zero initial guess for robustness
        
        if verbose:
            print(f"  Initial kappa guess: max = {np.max(np.abs(kappa_initial)):.3e}")
            # Test B operator with zero input (should give zero output with homogeneous BCs!)
            test_zero = matvec(np.zeros(self.n_cells))
            print(f"  B·0: max = {np.max(np.abs(test_zero)):.3e} (should be ~0)")
            # Test B operator with initial guess
            test_result = matvec(kappa_initial)
            print(f"  B·kappa_initial: max = {np.max(np.abs(test_result)):.3e}")
            print(f"  ||RHS - B·kappa_initial|| / ||RHS|| = {np.linalg.norm(rhs - test_result) / (np.linalg.norm(rhs) + 1e-30):.3e}")
            
            # Test if B is reasonable by checking diagonal scaling
            # Compute B·e_i for first few basis vectors
            diag_approx = []
            for i in range(min(5, self.n_cells)):
                ei = np.zeros(self.n_cells)
                ei[i] = 1.0
                Bei = matvec(ei)
                diag_approx.append(Bei[i])
            print(f"  B diagonal samples: {diag_approx[:5]}")
        
        # Solve using GMRES with restart
        if verbose:
            print(f"  Solving for κ with GMRES (tol={gmres_tol}, maxiter={gmres_maxiter})...")
        
        restart_val = min(30, self.n_cells)
        kappa, info = gmres(B_operator, rhs, x0=kappa_initial, 
                           rtol=gmres_tol, maxiter=gmres_maxiter, 
                           restart=restart_val, atol=gmres_tol*1e-2)
        
        info_dict = {'info': info, 'iterations': info if info > 0 else gmres_maxiter}
        
        if info == 0:
            if verbose:
                print(f"  GMRES converged successfully")
        elif info > 0:
            if verbose:
                print(f"  Warning: GMRES did not fully converge ({info} iterations)")
        else:
            print(f"  Warning: GMRES illegal input or breakdown (info={info})")
        
        return kappa, info_dict
    
    def compute_radiation_energy_from_kappa(self, kappa: np.ndarray, T_star: np.ndarray, 
                                            xi_g_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total radiation energy density E_r from κ using equation 8.88.
        
        For each group: φ_g^{n+1} = A_g^{-1}(χ_g(1-f)κ + ξ_g)
        Then: E_r = Σ_g E_{r,g} = (1/c) Σ_g φ_g^{n+1}
        
        This computes E_r on-the-fly while solving for φ_g, avoiding redundant operator inversions.
        Also computes and returns the fractional distribution for accurate future source terms.
        
        Parameters:
        -----------
        kappa : ndarray
            Absorption rate density
        T_star : ndarray
            Temperature for operator evaluation
        xi_g_list : list of ndarray
            Source terms for each group
            
        Returns:
        --------
        E_r : ndarray
            Total radiation energy density
        phi_g_fraction : ndarray (n_groups, n_cells)
            Fractional distribution: phi_g / phi_total for each group
        """
        # Get local Fleck factor for each cell
        f = self.fleck_factor
        E_r = np.zeros(self.n_cells)
        phi_g_all = np.zeros((self.n_groups, self.n_cells))
        
        for g in range(self.n_groups):
            # RHS = χ_g(1-f)κ + ξ_g
            rhs_g = self.chi[g] * (1.0 - f) * kappa + xi_g_list[g]
            
            # Solve A_g φ_g = rhs_g (use previous timestep's phi as guess for flux limiter)
            phi_g = self.solvers[g].solve(rhs_g, T_star, phi_guess=self.phi_g_stored[g, :])
            
            # Check for negative phi (unphysical) - ignore machine precision noise
            neg_threshold = -1e-10  # Only warn if significantly negative
            significantly_negative = phi_g < neg_threshold
            if np.any(significantly_negative):
                n_neg = np.sum(significantly_negative)
                min_phi = np.min(phi_g)
                print(f"    WARNING: Group {g} has {n_neg} significantly negative phi values (min={min_phi:.3e})")
                print(f"             This indicates GMRES did not converge sufficiently!")
            
            # Update stored phi values
            self.phi_g_stored[g, :] = phi_g
            
            phi_g_all[g, :] = phi_g
            
            # Accumulate E_r = (1/c) Σ_g φ_g
            E_r += phi_g / C_LIGHT
        
        # Compute fractional distribution
        phi_total = E_r * C_LIGHT
        phi_g_fraction = np.zeros((self.n_groups, self.n_cells))
        for g in range(self.n_groups):
            # Avoid division by zero
            mask = phi_total > 1e-50
            phi_g_fraction[g, mask] = phi_g_all[g, mask] / phi_total[mask]
            # Use uniform distribution where phi_total is very small
            phi_g_fraction[g, ~mask] = 1.0 / self.n_groups
        
        return E_r, phi_g_fraction
    
    def update_temperature(self, kappa: np.ndarray, T_star: np.ndarray) -> np.ndarray:
        """
        Update material temperature using equation 8.83b with κ directly.
        
        Using κ = Σ_g σ*_{a,g} φ_g^{n+1}, we can write:
        e(T_{n+1}) = e(T_n) + Δt·f·[κ - Σ_g σ*_{a,g}·4πB_g(T_★)] + (1-f)Δe
        
        This avoids needing to store individual φ_g values.
        
        Parameters:
        -----------
        kappa : ndarray
            Absorption rate density κ = Σ_g σ*_{a,g} φ_g^{n+1}
        T_star : ndarray
            Linearization temperature T_★
        
        Returns:
        --------
        T_new : ndarray
            Updated temperature
        """
        # Get local Fleck factor for each cell
        f = self.fleck_factor
        
        # Old energy
        e_n = self.material_energy_func(self.T_old)
        e_star = self.material_energy_func(T_star)
        Delta_e = e_star - e_n
        
        # Compute sum of σ*_{a,g}·4πB_g(T_★)
        sigma_B_sum = np.zeros(self.n_cells)
        for g in range(self.n_groups):
            B_g_star = np.array([self.planck_funcs[g](T_star[i]) for i in range(self.n_cells)])
            sigma_B_sum += self.sigma_a[g, :] * 4.0 * np.pi * B_g_star
        
        # Energy change using κ directly
        energy_change = kappa - sigma_B_sum
        
        # New energy
        e_new = e_n + self.dt * f * energy_change + (1.0 - f) * Delta_e
        
        # Convert to temperature
        T_new = np.array([self.inverse_material_energy_func(e_new[i]) for i in range(self.n_cells)])
        
        return T_new
    
    def step(self, max_newton_iter: int = 10, newton_tol: float = 1e-6,
            gmres_tol: float = 1e-6, gmres_maxiter: int = 200,
            verbose: bool = False) -> dict:
        """
        Take one time step using Newton iteration.
        
        Parameters:
        -----------
        max_newton_iter : int
            Maximum Newton iterations
        newton_tol : float
            Newton convergence tolerance
        gmres_tol : float
            GMRES tolerance for κ solve
        gmres_maxiter : int
            Maximum GMRES iterations
        verbose : bool
            Print convergence information
        
        Returns:
        --------
        info : dict
            Convergence information
        """
        # Initial guess: T_★ = T_old
        T_star = self.T_old.copy()
        
        for newton_iter in range(max_newton_iter):
            if verbose:
                print(f"\nNewton iteration {newton_iter + 1}/{max_newton_iter}")
            
            # Update absorption coefficients at T_★
            self.update_absorption_coefficients(T_star)
            
            # Compute local Fleck factor at T_★
            self.fleck_factor = self.compute_fleck_factor(T_star)
            
            if verbose:
                print(f"  Fleck factor: min = {self.fleck_factor.min():.3e}, max = {self.fleck_factor.max():.3e}")
            
            # Compute source terms ξ_g for all groups
            xi_g_list = [self.compute_source_xi(g, T_star, self.t) for g in range(self.n_groups)]
            
            # Solve for κ using GMRES
            kappa, gmres_info = self.solve_for_kappa(T_star, xi_g_list, 
                                                     gmres_tol=gmres_tol,
                                                     gmres_maxiter=gmres_maxiter,
                                                     verbose=verbose)
            
            # Compute E_r from κ (accumulates while solving for φ_g)
            E_r_new, phi_g_fraction_new = self.compute_radiation_energy_from_kappa(kappa, T_star, xi_g_list)
            
            # Update temperature using κ directly
            T_new = self.update_temperature(kappa, T_star)
            
            # Check convergence
            T_change = np.linalg.norm(T_new - T_star) / (np.linalg.norm(T_new) + 1e-14)
            
            if verbose:
                print(f"  Temperature change: {T_change:.3e}")
                print(f"  T max: {T_new.max():.6f}, T min: {T_new.min():.6f}")
                print(f"  E_r max: {E_r_new.max():.6e}, min: {E_r_new.min():.6e}")
                print(f"  κ max: {kappa.max():.6e}, min: {kappa.min():.6e}")
            
            if T_change < newton_tol:
                if verbose:
                    print(f"  Newton converged after {newton_iter + 1} iterations")
                
                # Store final converged values
                self.T = T_new
                self.kappa = kappa
                self.E_r = E_r_new
                self.phi_g_fraction = phi_g_fraction_new
                
                return {
                    'converged': True,
                    'newton_iter': newton_iter + 1,
                    'T_change': T_change,
                    'gmres_info': gmres_info
                }
            
            # Update T_★ for next iteration
            T_star = T_new
        
        # Did not converge - store last iteration values anyway
        print(f"Warning: Newton did not converge after {max_newton_iter} iterations")
        self.T = T_new
        self.kappa = kappa
        self.E_r = E_r_new
        self.phi_g_fraction = phi_g_fraction_new
        self.T = T_new
        return {
            'converged': False,
            'newton_iter': max_newton_iter,
            'T_change': T_change,
            'gmres_info': gmres_info
        }
    
    def advance_time(self):
        """Store current solution as old solution and advance time."""
        self.kappa_old = self.kappa.copy()
        self.E_r_old = self.E_r.copy()
        self.T_old = self.T.copy()
        self.t += self.dt
    
    def compute_phi_g(self, g: int, kappa: Optional[np.ndarray] = None, 
                     T_star: Optional[np.ndarray] = None,
                     xi_g_list: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Compute φ_g for a specific group (optional, for diagnostics).
        
        Parameters:
        -----------
        g : int
            Group index
        kappa : ndarray or None
            Absorption rate density (uses self.kappa if None)
        T_star : ndarray or None
            Temperature (uses self.T if None)
        xi_g_list : list or None
            Source terms (recomputes if None)
            
        Returns:
        --------
        phi_g : ndarray
            Radiation variable for group g
        """
        if kappa is None:
            kappa = self.kappa
        if T_star is None:
            T_star = self.T
        if xi_g_list is None:
            self.update_absorption_coefficients(T_star)
            self.fleck_factor = self.compute_fleck_factor(T_star)
            xi_g_list = [self.compute_source_xi(gp, T_star, self.t) for gp in range(self.n_groups)]
        
        f = self.fleck_factor
        rhs_g = self.chi[g] * (1.0 - f) * kappa + xi_g_list[g]
        phi_g = self.solvers[g].solve(rhs_g, T_star)
        
        return phi_g


# =============================================================================
# EXAMPLE: GRAY MARSHAK WAVE SPLIT INTO 3 GROUPS
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MULTIGROUP DIFFUSION SOLVER - Marshak Wave Test")
    print("="*70)
    print("\nTest: Gray problem split into 3 groups")
    print("Should give same answer as single gray group\n")
    
    # Problem parameters (Marshak wave with Su-Olson material)
    n_groups = 3
    n_cells = 50
    r_max = 10.0  # cm
    dt = 0.01  # ns
    
    # Energy group edges (keV) - for gray, we use arbitrary equal spacing
    # Groups: [0.0-1.0], [1.0-2.0], [2.0-3.0] keV
    energy_edges = np.linspace(0.0, 3.0, n_groups + 1)
    
    # Material properties
    rho = 1.0  # g/cm³
    cv = 0.1   # GJ/(g·keV)
    
    # Gray opacity (split equally among groups)
    sigma_a_gray = 1.0  # cm⁻¹
    sigma_a_per_group = sigma_a_gray / n_groups
    
    # Rosseland opacity for diffusion (also gray)
    sigma_r_gray = 1.0  # cm⁻¹
    
    # Diffusion coefficient: D = 1/(3σ_R)
    def diffusion_coeff(T, r):
        return 1.0 / (3.0 * sigma_r_gray)
    
    # Absorption coefficient (same for all groups)
    def absorption_coeff(T, r):
        return sigma_a_per_group
    
    # Boundary conditions: left boundary at equilibrium T_bc, right boundary at T=0
    T_bc = 1.0  # keV (left boundary temperature)
    phi_bc_per_group = (A_RAD * C_LIGHT * T_bc**4) / n_groups  # φ = E_r·c = a·c·T^4, split equally
    
    # Define BC functions
    def make_dirichlet_left_bc(phi_val):
        def left_bc(phi, r):
            return 1.0, 0.0, phi_val  # Dirichlet: A*phi = C
        return left_bc
    
    def dirichlet_right_bc(phi, r):
        return 1.0, 0.0, 0.0  # Dirichlet: phi = 0
    
    left_bc_funcs = [make_dirichlet_left_bc(phi_bc_per_group) for _ in range(n_groups)]
    right_bc_funcs = [dirichlet_right_bc] * n_groups
    
    # Initialize solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=r_max,
        n_cells=n_cells,
        geometry='planar',
        dt=dt,
        energy_edges=energy_edges,
        diffusion_coeff_funcs=[diffusion_coeff] * n_groups,
        absorption_coeff_funcs=[absorption_coeff] * n_groups,
        left_bc_funcs=left_bc_funcs,
        right_bc_funcs=right_bc_funcs,
        rho=rho,
        cv=cv
    )
    
    # Initial conditions: cold material
    solver.T = np.ones(n_cells) * 0.01  # keV
    solver.T_old = solver.T.copy()
    solver.E_r = np.ones(n_cells) * A_RAD * 0.01**4  # Cold radiation
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    # Time evolution
    n_steps = 5
    t = 0.0
    
    print(f"\nTime evolution:")
    print(f"{'Step':<6} {'Time':<10} {'T_max':<12} {'T_min':<12} {'E_r_max':<15} {'Newton':<8} {'Conv':<4}")
    print("-" * 75)
    
    for step in range(n_steps):
        # Take time step
        info = solver.step(max_newton_iter=10, newton_tol=1e-8,
                          gmres_tol=1e-6, gmres_maxiter=200,
                          verbose=(step == 0))  # Verbose for first step only
        
        t += dt
        
        # Print progress
        converged_str = "✓" if info['converged'] else "✗"
        print(f"{step+1:<6} {t:<10.4f} {solver.T.max():<12.6f} {solver.T.min():<12.6f} "
              f"{solver.E_r.max():<15.6e} {info['newton_iter']:<8} {converged_str}")
        
        # Store solution for next step
        solver.advance_time()
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)
    
    # Summary
    print(f"\nFinal state at t = {t:.4f} ns:")
    print(f"  Temperature: max = {solver.T.max():.6f} keV, min = {solver.T.min():.6f} keV")
    print(f"  E_r (total): max = {solver.E_r.max():.6e} GJ/cm³, min = {solver.E_r.min():.6e} GJ/cm³")
    print(f"  κ: max = {solver.kappa.max():.6e}, min = {solver.kappa.min():.6e}")
    
    # Check consistency: compute individual groups and verify they sum to E_r
    print(f"\nGroup consistency check:")
    E_r_check = np.zeros(n_cells)
    for g in range(n_groups):
        phi_g = solver.compute_phi_g(g)
        E_r_g = phi_g / C_LIGHT
        E_r_check += E_r_g
        print(f"  Group {g}: max E_r = {E_r_g.max():.6e}, min E_r = {E_r_g.min():.6e}")
    
    # Verify E_r consistency
    relative_error = np.linalg.norm(E_r_check - solver.E_r) / np.linalg.norm(solver.E_r)
    print(f"\nE_r consistency: ||E_r_check - E_r|| / ||E_r|| = {relative_error:.3e}")
    if relative_error < 1e-10:
        print("✓ E_r computed correctly from individual groups!")
    
    # Check that groups are equal (for gray problem)
    print(f"\nGray consistency check (groups should be equal):")
    phi_0 = solver.compute_phi_g(0)
    for g in range(1, n_groups):
        phi_g = solver.compute_phi_g(g)
        max_diff = np.abs(phi_g - phi_0).max()
        print(f"  |φ_{g} - φ_0|_max = {max_diff:.3e}")
