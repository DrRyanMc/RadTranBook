#!/usr/bin/env python3
"""
1D Finite Volume Solver for Non-Equilibrium Radiation Diffusion Equation
using Newton iterations with coupled φ(r,t) and T(r,t) variables.

Variables:
  φ(r,t) = E_r(r,t) * c   (radiation energy density × speed of light)
  T(r,t)                  (material temperature)

Coupled PDEs:
  (1/c) ∂φ/∂t - ∇·(D∇φ) = σ_P·f(acT_★^4 - φ̃) - (1-f)·Δe/Δt
  ∂e/∂t = f·σ_P(φ̃ - acT_★^4) + (1-f)·Δe/Δt
  
Time discretization: θ-method or TR-BDF2
Newton method for nonlinear coupling
"""

import inspect
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from typing import Tuple, Callable
import time


# =============================================================================
# PHYSICAL CONSTANTS AND PARAMETERS
# =============================================================================

# Physical constants
C_LIGHT = 2.99792458e1  # speed of light (cm/ns)
A_RAD = 0.01372       # radiation constant (GJ/(cm³·keV⁴))
SIGMA_SB = C_LIGHT*A_RAD/4   # Stefan-Boltzmann constant (GJ/(cm²·ns·keV⁴))


# Default material properties (can be overridden)
RHO = 1.0                # density (g/cm³)
CV_CONST = 1.0         # specific heat (GJ/(g·keV))
# Note: β is computed dynamically as β = 4aT_★³/C_v_★, not a constant


# =============================================================================
# MATERIAL PROPERTY FUNCTIONS
# =============================================================================

@njit
def phi_from_Er(Er):
    """Convert radiation energy density to φ: φ = E_r * c"""
    return Er * C_LIGHT


@njit
def Er_from_phi(phi):
    """Convert φ to radiation energy density: E_r = φ / c"""
    return phi / C_LIGHT


@njit
def temperature_from_phi_equilibrium(phi):
    """Convert φ to equilibrium temperature: T = (E_r/a)^(1/4) = (φ/(ac))^(1/4)"""
    Er = Er_from_phi(phi)
    return (Er / A_RAD) ** 0.25


@njit
def specific_heat_cv(T):
    """Specific heat capacity c_v(T). Default: constant."""
    return CV_CONST


@njit
def material_energy_density(T):
    """Material energy density e(T) = ρ c_v T"""
    return RHO * specific_heat_cv(T) * T


def inverse_material_energy_density(e):
    """Inverse: T from e. Default assumes e = ρ·c_v·T => T = e/(ρ·c_v)"""
    return e / (RHO * CV_CONST)


def _wrap_spatial_func(func):
    """Wrap a single-argument material property function to also accept an
    optional second positional-or-keyword argument *r* (spatial coordinate).

    Functions that already accept two or more parameters are returned unchanged.
    This lets new problem files define  f(T, r)  while all existing T-only
    functions continue to work without modification.
    """
    try:
        sig = inspect.signature(func)
        if len(sig.parameters) >= 2:
            return func          # already accepts r
    except (ValueError, TypeError):
        pass
    def _wrapped(T, r=None):
        return func(T)
    return _wrapped


@njit
def planck_opacity(T):
    """Planck mean opacity σ_P(T). Default: constant for testing."""
    return 1.0  # cm⁻¹


@njit
def rosseland_opacity(T):
    """Rosseland opacity σ_R(T). Default: constant for testing."""
    return 0.1  # cm⁻¹


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
    result = np.zeros_like(R)
    
    # Use approximation to avoid numerical issues
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
    
    λ^max(R) = max(3, R)^(-1) = 1/max(3, R)
    
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
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

@njit
def robin_bc_coefficients(phi, x, is_left=True):
    """
    Robin boundary condition: A_bc * \u03c6 + B_bc(\u03c6) * (n·\u2207\u03c6) = C_bc
    Returns (A_bc, B_bc, C_bc)
    
    For testing: use simple Dirichlet-like conditions
    """
    if is_left:
        # Left boundary (r = r_min)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet: \u03c6 = C_bc
        T_bc = 1.0  # Temperature at boundary (keV)
        phi_bc = A_RAD * T_bc**4 * C_LIGHT  # \u03c6 = acT^4
        C_bc = phi_bc
    else:
        # Right boundary (r = r_max)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet: \u03c6 = C_bc
        T_bc = 0.316  # Temperature at boundary (keV)
        phi_bc = A_RAD * T_bc**4 * C_LIGHT  # \u03c6 = acT^4
        C_bc = phi_bc
    
    return A_bc, B_bc, C_bc


# =============================================================================
# GRID GENERATION AND GEOMETRY
# =============================================================================

def generate_grid(r_min: float, r_max: float, n_cells: int, 
                  stretch_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nonuniform radial grid
    
    Parameters:
    -----------
    r_min, r_max : float
        Domain boundaries
    n_cells : int
        Number of cells
    stretch_factor : float
        Grid stretching (1.0 = uniform)
    
    Returns:
    --------
    r_faces : ndarray
        Face positions (length n_cells + 1)
    r_centers : ndarray
        Cell center positions (length n_cells)
    """
    if stretch_factor == 1.0:
        # Uniform grid
        r_faces = np.linspace(r_min, r_max, n_cells + 1)
    else:
        # Stretched grid
        xi = np.linspace(0, 1, n_cells + 1)
        r_faces = r_min + (r_max - r_min) * ((stretch_factor**xi - 1) / (stretch_factor - 1))
    
    # Cell centers
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    
    return r_faces, r_centers


@njit
def compute_geometry_factors(r_faces, d):
    """
    Compute geometry factors for coordinate system d = 0,1,2 (plane/cylindrical/spherical)
    
    Parameters:
    -----------
    r_faces : ndarray
        Face positions
    d : int
        Coordinate system (0=plane, 1=cylindrical, 2=spherical)
    
    Returns:
    --------
    A_faces : ndarray
        Face areas
    V_cells : ndarray
        Cell volumes
    """
    n_faces = len(r_faces)
    n_cells = n_faces - 1
    
    # Solid angle factors
    if d == 0:
        omega_d = 1.0
    elif d == 1:
        omega_d = 2.0 * np.pi
    else:  # d == 2
        omega_d = 4.0 * np.pi
    
    # Face areas: A_{i+1/2} = ω_d * r_{i+1/2}^d
    A_faces = omega_d * r_faces**d
    
    # Cell volumes: |V_i| = ω_d/(d+1) * (r_{i+1/2}^{d+1} - r_{i-1/2}^{d+1})
    V_cells = np.zeros(n_cells)
    for i in range(n_cells):
        r_left = r_faces[i]
        r_right = r_faces[i + 1]
        V_cells[i] = omega_d / (d + 1) * (r_right**(d + 1) - r_left**(d + 1))
    
    return A_faces, V_cells


# =============================================================================
# HARMONIC AVERAGING FOR FACE VALUES
# =============================================================================

@njit
def harmonic_average_weighted(D_left, D_right, dx_left, dx_right):
    """Distance-weighted harmonic average for non-uniform grids
    
    D_{i+1/2} = D_i * D_{i+1} * (dx_left + dx_right) / 
                (D_{i+1} * dx_left + D_i * dx_right)
    
    where dx_left = r_{i+1/2} - r_i, dx_right = r_{i+1} - r_{i+1/2}
    """
    if D_left <= 0 or D_right <= 0:
        return 0.0
    
    total_dx = dx_left + dx_right
    if total_dx <= 0:
        return 0.0
        
    return D_left * D_right * total_dx / (D_right * dx_left + D_left * dx_right)


# =============================================================================
# TRIDIAGONAL SOLVER (THOMAS ALGORITHM)
# =============================================================================

# =============================================================================
# TRIDIAGONAL SOLVER (THOMAS ALGORITHM)
# =============================================================================

@njit
def _solve_tridiagonal_arrays(a, b, c, rhs):
    """
    Solve tridiagonal system using Thomas algorithm (Numba-compiled)
    
    Parameters:
    -----------
    a, b, c : ndarray
        Sub-diagonal, main diagonal, and super-diagonal
    rhs : ndarray
        Right-hand side vector
    
    Returns:
    --------
    x : ndarray
        Solution vector
    """
    n = len(rhs)
    x = np.zeros(n)
    
    # Work on copies to avoid modifying input arrays
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = rhs[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        if i < n-1:
            c_prime[i] = c[i] / denominator
        d_prime[i] = (rhs[i] - a[i] * d_prime[i-1]) / denominator
    
    # Back substitution
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


def solve_tridiagonal(A_tri, rhs):
    """
    Solve tridiagonal system using Thomas algorithm
    
    Parameters:
    -----------
    A_tri : dict or ndarray
        If dict: {'sub': sub-diagonal, 'diag': main diagonal, 'super': super-diagonal}
        If ndarray (3, n): A_tri[0, :] = sub, A_tri[1, :] = diag, A_tri[2, :] = super
    rhs : ndarray (n,)
        Right-hand side vector
    
    Returns:
    --------
    x : ndarray (n,)
        Solution vector
    """
    n = len(rhs)
    
    # Extract diagonals - handle both dict and array formats
    if isinstance(A_tri, dict):
        a = np.zeros(n)  # sub-diagonal
        a[1:] = A_tri['sub']
        b = A_tri['diag']  # main diagonal  
        c = np.zeros(n)  # super-diagonal
        c[:-1] = A_tri['super']
    else:
        a = np.zeros(n)  # sub-diagonal
        a[1:] = A_tri[0, 1:]  
        b = A_tri[1, :]  # main diagonal  
        c = np.zeros(n)  # super-diagonal
        c[:-1] = A_tri[2, :-1]
    
    return _solve_tridiagonal_arrays(a, b, c, rhs)


def apply_tridiagonal(A_tri, x):
    """Apply tridiagonal matrix to vector: y = A * x
    
    Parameters:
    -----------
    A_tri : ndarray (3, n)
        Tridiagonal matrix with format:
        A_tri[0, :] = sub-diagonal (a_i)
        A_tri[1, :] = main diagonal (b_i) 
        A_tri[2, :] = super-diagonal (c_i)
    x : ndarray (n,)
        Input vector
    
    Returns:
    --------
    y : ndarray (n,)
        Result vector y = A * x
    """
    n = len(x)
    y = np.zeros(n)
    
    # Extract diagonals
    a = A_tri[0, :]  # sub-diagonal
    b = A_tri[1, :]  # main diagonal  
    c = A_tri[2, :]  # super-diagonal
    
    # First row: b[0]*x[0] + c[0]*x[1]
    y[0] = b[0] * x[0]
    if n > 1:
        y[0] += c[0] * x[1]
    
    # Interior rows: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
    for i in range(1, n-1):
        y[i] = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1]
    
    # Last row: a[n-1]*x[n-2] + b[n-1]*x[n-1]
    if n > 1:
        y[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1]
    else:
        y[0] = b[0] * x[0]
    
    return y


# =============================================================================
# BLOCK TRIDIAGONAL SOLVER FOR COUPLED SYSTEM
# =============================================================================

def solve_block_tridiagonal_2x2(A_blocks, b1, b2):
    """
    Solve a block tridiagonal system where each block is 2x2
    
    System: A * [φ; T] = [b1; b2]
    
    Block structure for each cell i:
    [[A11_i, A12_i],  coupling to [[φ_i],
     [A21_i, A22_i]]               [T_i]]
    
    Parameters:
    -----------
    A_blocks : dict with keys 'sub', 'diag', 'super'
        Each value is array of shape (n_cells, 2, 2) containing 2x2 blocks
        'diag': main diagonal blocks (n_cells, 2, 2)
        'super': super-diagonal blocks for φ equation only (n_cells-1,) - scalar coupling
        'sub': sub-diagonal blocks for φ equation only (n_cells-1,) - scalar coupling
    b1, b2 : ndarray (n_cells,)
        Right-hand side vectors for φ and T equations
    
    Returns:
    --------
    phi, T : ndarray (n_cells,)
        Solution vectors
    """
    n = len(b1)
    
    # Extract blocks
    diag = A_blocks['diag']  # (n, 2, 2)
    
    # For off-diagonals in diffusion, only φ equation couples spatially
    # super_phi and sub_phi are scalars (coefficients) for each face
    if 'super' in A_blocks and A_blocks['super'] is not None:
        super_phi = A_blocks['super']  # (n-1,) - coupling φ_i to φ_{i+1}
    else:
        super_phi = np.zeros(n-1)
    
    if 'sub' in A_blocks and A_blocks['sub'] is not None:
        sub_phi = A_blocks['sub']  # (n-1,) - coupling φ_i to φ_{i-1}
    else:
        sub_phi = np.zeros(n-1)
    
    # Forward elimination with block operations
    # Modified blocks during elimination
    diag_mod = diag.copy()
    b1_mod = b1.copy()
    b2_mod = b2.copy()
    
    for i in range(n-1):
        # Current block is 2x2
        # [[d11, d12],
        #  [d21, d22]]
        
        # Invert current diagonal block
        d11, d12 = diag_mod[i, 0, 0], diag_mod[i, 0, 1]
        d21, d22 = diag_mod[i, 1, 0], diag_mod[i, 1, 1]
        
        det = d11 * d22 - d12 * d21
        if abs(det) < 1e-14:
            print(f"Warning: near-singular block at i={i}, det={det}")
            det = 1e-14 if det >= 0 else -1e-14
        
        # Inverse of 2x2 block
        inv11 = d22 / det
        inv12 = -d12 / det
        inv21 = -d21 / det
        inv22 = d11 / det
        
        # The super-diagonal only affects φ equation (first row)
        # Elimination factor for φ_i -> φ_{i+1} coupling
        s = super_phi[i]  # scalar
        
        # Update next diagonal block
        # New diagonal block -= sub * inv_diag * super
        # Since only φ couples spatially: sub affects φ, super affects φ
        if i < n-1:
            sub_next = sub_phi[i] if i < len(sub_phi) else 0
            
            # Multiply: [sub_next, 0] * [[inv11, inv12], * [[s],
            #                             [inv21, inv22]]   [0]]
            # = [sub_next, 0] * [[inv11*s], = [sub_next*inv11*s,
            #                    [inv21*s]]     sub_next*inv21*s]
            
            diag_mod[i+1, 0, 0] -= sub_next * inv11 * s
            diag_mod[i+1, 0, 1] -= sub_next * inv12 * s
            diag_mod[i+1, 1, 0] -= sub_next * inv21 * s
            diag_mod[i+1, 1, 1] -= sub_next * inv22 * s
            
            # Update RHS
            # b_mod[i+1] -= sub * inv_diag * b_mod[i]
            b1_mod[i+1] -= sub_next * (inv11 * b1_mod[i] + inv12 * b2_mod[i])
            b2_mod[i+1] -= sub_next * (inv21 * b1_mod[i] + inv22 * b2_mod[i])
    
    # Back substitution
    phi = np.zeros(n)
    T = np.zeros(n)
    
    # Last block
    d11, d12 = diag_mod[n-1, 0, 0], diag_mod[n-1, 0, 1]
    d21, d22 = diag_mod[n-1, 1, 0], diag_mod[n-1, 1, 1]
    det = d11 * d22 - d12 * d21
    if abs(det) < 1e-14:
        print(f"Warning: near-singular last block, det={det}")
        det = 1e-14 if det >= 0 else -1e-14
    
    phi[n-1] = (d22 * b1_mod[n-1] - d12 * b2_mod[n-1]) / det
    T[n-1] = (-d21 * b1_mod[n-1] + d11 * b2_mod[n-1]) / det
    
    # Backward sweep
    for i in range(n-2, -1, -1):
        # Solve block system with modification from super-diagonal
        d11, d12 = diag_mod[i, 0, 0], diag_mod[i, 0, 1]
        d21, d22 = diag_mod[i, 1, 0], diag_mod[i, 1, 1]
        
        # Contribution from φ_{i+1}
        s = super_phi[i]
        rhs1 = b1_mod[i] - s * phi[i+1]
        rhs2 = b2_mod[i]  # T doesn't couple spatially via super-diagonal
        
        det = d11 * d22 - d12 * d21
        if abs(det) < 1e-14:
            det = 1e-14 if det >= 0 else -1e-14
        
        phi[i] = (d22 * rhs1 - d12 * rhs2) / det
        T[i] = (-d21 * rhs1 + d11 * rhs2) / det
    
    return phi, T





# =============================================================================
# MAIN SOLVER CLASS
# =============================================================================

class NonEquilibriumRadiationDiffusionSolver:
    """1D Finite Volume Non-Equilibrium Radiation Diffusion Solver
    
    Solves coupled system for φ(r,t) = E_r * c and T(r,t)
    """
    
    def __init__(self, r_min=0.1, r_max=1.0, n_cells=50, d=0, dt=1e-3,
                 max_newton_iter=10, newton_tol=1e-8, stretch_factor=1.0,
                 rosseland_opacity_func=None, planck_opacity_func=None,
                 specific_heat_func=None, material_energy_func=None,
                 inverse_material_energy_func=None,
                 left_bc_func=None, right_bc_func=None,
                 theta=1.0, flux_limiter_func=None):
        
        self.r_min = r_min
        self.r_max = r_max  
        self.n_cells = n_cells
        self.d = d  # Coordinate system
        self.dt = dt
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.theta = theta  # Time discretization parameter (1.0 = implicit Euler)
        # Note: β computed dynamically as β = 4aT_★³/C_v_★
        
        # Generate grid and geometry
        self.r_faces, self.r_centers = generate_grid(r_min, r_max, n_cells, stretch_factor)
        self.A_faces, self.V_cells = compute_geometry_factors(self.r_faces, d)
        
        # Solution arrays for non-equilibrium: φ and T
        self.phi = np.ones(n_cells)  # Current φ = E_r * c
        self.T = np.ones(n_cells)    # Current temperature
        self.phi_old = np.ones(n_cells)  # Previous time step
        self.T_old = np.ones(n_cells)    # Previous time step
        
        # Material property functions (use defaults if not provided).
        # Each is wrapped so it can be called as  f(T, r)  regardless of whether
        # the user-supplied function accepts a spatial argument or not.
        self.rosseland_opacity_func = _wrap_spatial_func(rosseland_opacity_func or rosseland_opacity)
        self.planck_opacity_func = _wrap_spatial_func(planck_opacity_func or planck_opacity)
        self.specific_heat_func = _wrap_spatial_func(specific_heat_func or specific_heat_cv)
        self.material_energy_func = _wrap_spatial_func(material_energy_func or material_energy_density)
        self.inverse_material_energy_func = _wrap_spatial_func(inverse_material_energy_func or inverse_material_energy_density)
        
        # Flux limiter function (use standard λ=1/3 if not provided)
        self.flux_limiter_func = flux_limiter_func or flux_limiter_standard
        
        # Boundary condition functions (use defaults if not provided)
        self.left_bc_func = left_bc_func or (lambda phi, x: robin_bc_coefficients(phi, x, True))
        self.right_bc_func = right_bc_func or (lambda phi, x: robin_bc_coefficients(phi, x, False))
        
        print(f"Initialized non-equilibrium solver: {n_cells} cells, r ∈ [{r_min:.3f}, {r_max:.3f}]")
        print(f"Coordinate system d={d}, Δt={dt:.2e}")
    
    def get_diffusion_coefficient(self, T, phi_left=None, phi_right=None, dx=None, r=None):
        """Get flux-limited diffusion coefficient D = λ(R)/σ_R
        
        Parameters:
        -----------
        T : float
            Temperature at the face (used to compute σ_R)
        phi_left : float, optional
            φ value on left side of face
        phi_right : float, optional
            φ value on right side of face
        dx : float, optional
            Distance between cell centers across the face
        
        Returns:
        --------
        D : float
            Diffusion coefficient
        
        Notes:
        ------
        - If phi_left, phi_right, dx are not provided, uses standard λ=1/3
        - R = |∇φ|/(σ_R * φ_face) where φ_face = (phi_left + phi_right)/2
        - λ(R) is determined by self.flux_limiter_func
        - Standard diffusion: D = 1/(3σ_R), flux-limited: D = λ(R)/σ_R
        """
        sigma_R = self.rosseland_opacity_func(T, r)
        
        # If no phi information provided, use standard diffusion
        if phi_left is None or phi_right is None or dx is None:
            return 1.0 / (3.0 * sigma_R)
        
        # Compute |∇φ| at the face
        grad_phi_mag = abs(phi_right - phi_left) / dx
        
        # Compute φ at the face (arithmetic average)
        phi_face = 0.5 * (phi_left + phi_right)
        
        # Avoid division by zero
        if phi_face < 1e-30:
            # If φ ≈ 0, use standard diffusion
            return 1.0 / (3.0 * sigma_R)
        
        # Compute dimensionless ratio R = |∇φ|/(σ_R * φ)
        R = grad_phi_mag / (sigma_R * phi_face)
        
        # Compute flux limiter λ(R)
        lambda_R = self.flux_limiter_func(R)
        
        # Return flux-limited diffusion coefficient D = λ(R)/σ_R
        return lambda_R / sigma_R
    
    def get_beta(self, T_star, r=None):
        """Compute coupling parameter β = 4aT_★³/C_v_★
        
        From definition: β = (∂aT⁴/∂T) / (∂e(T)/∂T)|_{T=T_★}
        """
        cv_star = self.specific_heat_func(T_star, r)
        return (4.0 * A_RAD * T_star**3) / (RHO * cv_star)
    
    def get_f_factor(self, T_star, dt, theta, r=None):
        """Compute linearization factor f from equation (8.60)
        
        f = 1 / (1 + β·σ_P·c·θ·Δt)
        where β = 4aT_★³/C_v_★
        """
        beta = self.get_beta(T_star, r)
        sigma_P = self.planck_opacity_func(T_star, r)
        return 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)
    
    def get_f_factor_trbdf2(self, T_star, dt, Lambda, r=None):
        """Compute linearization factor f_TB for TR-BDF2 from equation (8.62)
        
        f_TB = 1 / (1 + [(1-Λ)/(2-Λ)] · β·σ_P·c·Δt)
        where β = 4aT_★³/C_v_★
        """
        beta = self.get_beta(T_star, r)
        sigma_P = self.planck_opacity_func(T_star, r)
        coeff = (1.0 - Lambda) / (2.0 - Lambda)
        return 1.0 / (1.0 + coeff * beta * sigma_P * C_LIGHT * dt)
    
    def set_initial_condition(self, phi_init=None, T_init=None):
        """Set initial conditions for φ and T
        
        Parameters:
        -----------
        phi_init : callable or float or array
            Initial φ = E_r * c
        T_init : callable or float or array
            Initial temperature
        """
        # Handle φ initialization
        if phi_init is None:
            # Default: equilibrium with T
            if T_init is None:
                T_init = 1.0
            if callable(T_init):
                self.T = T_init(self.r_centers)
            elif hasattr(T_init, '__len__'):
                self.T = np.array(T_init)
            else:
                self.T = np.full(self.n_cells, T_init)
            self.phi = A_RAD * C_LIGHT * self.T**4  # Equilibrium: φ = acT^4
        else:
            if callable(phi_init):
                self.phi = phi_init(self.r_centers)
            elif hasattr(phi_init, '__len__'):
                self.phi = np.array(phi_init)
            else:
                self.phi = np.full(self.n_cells, phi_init)
        
        # Handle T initialization
        if T_init is not None:
            if callable(T_init):
                self.T = T_init(self.r_centers)
            elif hasattr(T_init, '__len__'):
                self.T = np.array(T_init)
            else:
                self.T = np.full(self.n_cells, T_init)
        elif phi_init is not None:
            # If only φ given, set T from equilibrium
            self.T = temperature_from_phi_equilibrium(self.phi)
        
        # Store as old values
        self.phi_old = self.phi.copy()
        self.T_old = self.T.copy()
        
        # Validate
        if np.any(self.phi <= 0):
            raise ValueError("Initial φ must be positive everywhere")
        if np.any(self.T <= 0):
            raise ValueError("Initial temperature must be positive everywhere")
    
    def get_solution(self):
        """Return current solution"""
        return self.r_centers.copy(), self.phi.copy(), self.T.copy()
    
    def newton_step(self, phi_prev_timestep, T_prev_timestep, source=None, verbose=False):
        """Perform Newton iterations for coupled φ-T system
        
        Implements Algorithm 1 from the equations:
        1. Set φ★ ← φⁿ and T★ ← Tⁿ
        2. Repeat:
           - Solve equation 8.59a for φⁿ⁺¹ with D and σ_P evaluated using φ★ and T★
           - Solve equation 8.59b for T_{n+1} using φⁿ⁺¹
           - Compute residuals r_T and r_φ
           - Update φ★ ← φⁿ⁺¹, T★ ← T_{n+1}
        3. Until r_T and r_φ are below tolerance
        
        The equations are SEQUENTIALLY solved, not as a coupled 2×2 block system.
        
        Parameters:
        -----------
        source : ndarray or None, optional
            Source term for phi equation. Default: None (no source)
        """
        # Initial guess: use previous time step values
        phi_star = phi_prev_timestep.copy()
        T_star = T_prev_timestep.copy()
        
        for k in range(self.max_newton_iter):
            # Step 1: Solve equation 8.59a for φⁿ⁺¹
            # This is a tridiagonal system (spatial coupling only)
            A_phi, rhs_phi = self.assemble_phi_equation(
                phi_star, T_star, phi_prev_timestep, T_prev_timestep, theta=self.theta, source=source)
            
            # Apply boundary conditions for φ
            self.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
            
            # Solve tridiagonal system for φⁿ⁺¹
            phi_np1 = solve_tridiagonal(A_phi, rhs_phi)
            
            # Step 2: Solve equation 8.59b for T_{n+1}
            # This is decoupled (no spatial coupling) - just solve cell by cell
            T_np1 = self.solve_T_equation(
                phi_np1, T_star, phi_prev_timestep, T_prev_timestep, theta=self.theta)
            
            # Check for validity
            if np.any(~np.isfinite(phi_np1)) or np.any(~np.isfinite(T_np1)):
                if verbose:
                    print(f"    Newton iteration {k+1} produced invalid values!")
                # Conservative fallback
                phi_np1 = phi_star + 0.01 * (phi_np1 - phi_star)
                T_np1 = T_star + 0.01 * (T_np1 - T_star)
            
            # Check for negative values and apply damping if needed
            if np.any(phi_np1 <= 0) or np.any(T_np1 <= 0):
                # Line search
                alpha = 1.0
                for i in range(len(phi_star)):
                    if phi_np1[i] <= 0:
                        alpha = min(alpha, 0.9 * phi_star[i] / (phi_star[i] - phi_np1[i]))
                    if T_np1[i] <= 0:
                        alpha = min(alpha, 0.9 * T_star[i] / (T_star[i] - T_np1[i]))
                
                alpha = max(0.001, min(alpha, 0.9))
                phi_np1 = phi_star + alpha * (phi_np1 - phi_star)
                T_np1 = T_star + alpha * (T_np1 - T_star)
                
                if verbose:
                    print(f"    Applied damping: alpha = {alpha:.3f}")
            
            # Compute residuals
            r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
            r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
            
            if verbose and k == 0:
                print(f"    Newton iteration {k+1}: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
            
            if r_phi < self.newton_tol and r_T < self.newton_tol:
                if verbose:
                    print(f"    Newton converged in {k+1} iterations")
                    print(f"      r_φ={r_phi:.2e}, r_T={r_T:.2e}")
                return phi_np1, T_np1
            
            # Update φ★ and T★ for next iteration
            phi_star = phi_np1.copy()
            T_star = T_np1.copy()
        
        if verbose:
            print(f"    Newton max iterations reached: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
        return phi_star, T_star
    
    def assemble_phi_equation(self, phi_star, T_star, phi_prev, T_prev, theta=1.0, source=None):
        """Assemble equation 8.59a for φ^{n+1}
        
        Equation 8.59a:
        (φ^{n+1} - φ^n)/(c·Δt) + θ∇·D∇φ^{n+1} + (1-θ)∇·D∇φ^n = 
            σ_P·f(acT★⁴ - φ̃) - (1-f)·Δe/Δt + S
        
        where φ̃ = θ·φ^{n+1} + (1-θ)·φ^n and S is an optional source term
        
        This forms a tridiagonal system for φ^{n+1}.
        
        Parameters:
        -----------
        source : ndarray or None, optional
            Source term S in units [energy/(volume·time)]. Default: None (no source)
        
        Returns:
        --------
        A_tri : dict with 'sub', 'diag', 'super'
            Tridiagonal matrix coefficients
        rhs : ndarray
            Right-hand side vector
        """
        n_cells = len(phi_star)
        dt = self.dt
        
        # Initialize tridiagonal matrix and RHS
        sub = np.zeros(n_cells - 1)
        diag = np.zeros(n_cells)
        super_diag = np.zeros(n_cells - 1)
        rhs = np.zeros(n_cells)
        
        # Evaluate material energy at previous and current (linearization) timesteps
        e_n = np.array([self.material_energy_func(T_prev[i], self.r_centers[i]) for i in range(n_cells)])
        e_star = np.array([self.material_energy_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        Delta_e = e_star - e_n
        
        # Evaluate coupling parameters at linearization point T★
        sigma_P = np.array([self.planck_opacity_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        f = np.array([self.get_f_factor(T_star[i], dt, theta, self.r_centers[i]) for i in range(n_cells)])
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Diffusion coefficients at linearization point (flux-limited)
        D_faces = np.zeros(len(self.r_faces))
        for i in range(1, len(self.r_faces) - 1):
            # Temperature at face (arithmetic average)
            T_face = 0.5 * (T_star[i-1] + T_star[i])
            
            # φ values on either side of face and distance between cell centers
            phi_left = phi_star[i-1]
            phi_right = phi_star[i]
            dx = self.r_centers[i] - self.r_centers[i-1]
            
            # Compute flux-limited diffusion coefficient
            D_faces[i] = self.get_diffusion_coefficient(T_face, phi_left, phi_right, dx, self.r_faces[i])
        
        # Boundary faces (use cell-centered values, no flux limiting at boundaries)
        D_faces[0] = self.get_diffusion_coefficient(T_star[0], r=self.r_faces[0])
        D_faces[-1] = self.get_diffusion_coefficient(T_star[-1], r=self.r_faces[-1])
        
        # Assemble equation for each cell
        for i in range(n_cells):
            # Time derivative term: (1/(c·Δt)) on diagonal
            diag[i] = 1.0 / (C_LIGHT * dt)
            
            # Diffusion operator: +θ∇·D∇φ^{n+1}
            # Standard ∇·D∇φ discretization: +coeff on diag, -coeff on off-diag
            # So +θ∇·D∇φ gives: +coeff on diag, -coeff on off-diag
            V_i = self.V_cells[i]
            
            if i > 0:  # Left face
                A_left = self.A_faces[i]
                dr_left = self.r_centers[i] - self.r_centers[i-1]
                coeff = theta * A_left * D_faces[i] / (dr_left * V_i)
                diag[i] += coeff  # Diagonal (positive for +θ∇·D∇φ)
                sub[i-1] = -coeff  # Sub-diagonal (negative for +θ∇·D∇φ)
            
            if i < n_cells - 1:  # Right face
                A_right = self.A_faces[i+1]
                dr_right = self.r_centers[i+1] - self.r_centers[i]
                coeff = theta * A_right * D_faces[i+1] / (dr_right * V_i)
                diag[i] += coeff  # Diagonal (positive for +θ∇·D∇φ)
                super_diag[i] = -coeff  # Super-diagonal (negative for +θ∇·D∇φ)
            
            # Coupling term σ_P·f·θ (from -σ_P·f·θ·φ^{n+1} term after rearranging φ̃)
            diag[i] += theta * sigma_P[i] * f[i]
            
            # RHS: φ^n/(c·Δt) + σ_P·f·c·a·T★⁴ - (1-f)·Δe/Δt - (1-θ)·σ_P·f·φ^n
            rhs[i] = phi_prev[i] / (C_LIGHT * dt)
            rhs[i] += sigma_P[i] * f[i] * acT4_star[i]
            rhs[i] -= (1.0 - f[i]) * Delta_e[i] / dt
            rhs[i] -= (1.0 - theta) * sigma_P[i] * f[i] * phi_prev[i]
        
        # Explicit diffusion term (if θ < 1): +(1-θ)∇·D∇φ^n
        if theta < 1.0:
            for i in range(n_cells):
                V_i = self.V_cells[i]
                explicit_diffusion = 0.0
                
                if i > 0:  # Left face contribution
                    A_left = self.A_faces[i]
                    dr_left = self.r_centers[i] - self.r_centers[i-1]
                    flux_left = -D_faces[i] * (phi_prev[i] - phi_prev[i-1]) / dr_left
                    explicit_diffusion += A_left * flux_left / V_i
                
                if i < n_cells - 1:  # Right face contribution
                    A_right = self.A_faces[i+1]
                    dr_right = self.r_centers[i+1] - self.r_centers[i]
                    flux_right = -D_faces[i+1] * (phi_prev[i+1] - phi_prev[i]) / dr_right
                    explicit_diffusion -= A_right * flux_right / V_i
                
                # Add (1-θ) * divergence to RHS
                rhs[i] += (1.0 - theta) * explicit_diffusion
        
        # Add external source term if provided
        if source is not None:
            rhs += source
        
        A_tri = {'sub': sub, 'diag': diag, 'super': super_diag}
        return A_tri, rhs
    
    def solve_T_equation(self, phi_np1, T_star, phi_prev, T_prev, theta=1.0):
        """Solve equation 8.59b for T_{n+1} given φ^{n+1}
        
        Equation 8.59b:
        (e(T_{n+1}) - e(T_n))/Δt = f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt
        
        where φ̃ = θ·φ^{n+1} + (1-θ)·φ^n
        
        This is a scalar equation at each cell (no spatial coupling).
        
        Parameters:
        -----------
        phi_np1 : ndarray
            Solution φ^{n+1} from equation 8.59a
        T_star : ndarray
            Linearization point T★
        phi_prev : ndarray
            φ^n from previous timestep
        T_prev : ndarray
            T^n from previous timestep
        theta : float
            Time discretization parameter
        
        Returns:
        --------
        T_np1 : ndarray
            Solution T_{n+1}
        """
        n_cells = len(phi_np1)
        dt = self.dt
        T_np1 = np.zeros(n_cells)
        
        # Compute φ̃ = θ·φ^{n+1} + (1-θ)·φ^n
        phi_tilde = theta * phi_np1 + (1.0 - theta) * phi_prev
        
        # Evaluate material energy at previous timestep
        e_n = np.array([self.material_energy_func(T_prev[i], self.r_centers[i]) for i in range(n_cells)])
        e_star = np.array([self.material_energy_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        Delta_e = e_star - e_n
        
        # Evaluate coupling parameters at linearization point T★
        sigma_P = np.array([self.planck_opacity_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        f = np.array([self.get_f_factor(T_star[i], dt, theta, self.r_centers[i]) for i in range(n_cells)])
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Solve for T_{n+1} at each cell
        # e(T_{n+1}) = e(T_n) + Δt·[f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt]
        for i in range(n_cells):
            e_np1 = e_n[i] + dt * f[i] * sigma_P[i] * (phi_tilde[i] - acT4_star[i]) + (1.0 - f[i]) * Delta_e[i]
            
            # Use user-supplied inverse function to get T from e
            T_np1[i] = self.inverse_material_energy_func(e_np1, self.r_centers[i])
        
        return T_np1
    
    def apply_boundary_conditions_phi(self, A_tri, rhs, phi):
        """Apply boundary conditions for φ equation (equation 8.59a)
        
        Modifies the boundary rows of the tridiagonal matrix.
        
        Parameters:
        -----------
        A_tri : dict with 'sub', 'diag', 'super'
            Tridiagonal matrix
        rhs : ndarray
            Right-hand side vector
        phi : ndarray
            Current φ values (for evaluating BC)
        """
        n_cells = len(phi)
        
        # Left boundary (i = 0)
        A_bc_left, B_bc_left, C_bc_left = self.left_bc_func(phi[0], self.r_faces[0])
        
        if abs(B_bc_left) < 1e-14:
            # Dirichlet BC: A·φ = C, so φ = C/A
            phi_boundary = C_bc_left / A_bc_left
            # Average phi to get temperature for diffusion coefficient
            # This prevents using the cold interior T, which gives tiny D
            phi_avg = 0.5 * (phi[0] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25  # T = (φ/(ac))^0.25
            D_boundary = self.get_diffusion_coefficient(T_avg, r=self.r_faces[0])
            
            dx_half = self.r_centers[0] - self.r_faces[0]
            flux_coeff = (self.A_faces[0] * D_boundary) / (self.V_cells[0] * dx_half)
            
            A_tri['diag'][0] += flux_coeff
            rhs[0] += flux_coeff * phi_boundary
        else:
            # Robin BC: A·φ + B·(∂φ/∂n) = C
            T_avg = self.T[0] if hasattr(self, 'T') else 1.0
            D_boundary = self.get_diffusion_coefficient(T_avg, r=self.r_faces[0])
            flux_coeff = (self.A_faces[0] * D_boundary * A_bc_left) / (B_bc_left * self.V_cells[0])
            A_tri['diag'][0] += flux_coeff
            rhs[0] += self.A_faces[0] * D_boundary * C_bc_left / (B_bc_left * self.V_cells[0])
        
        # Right boundary (i = n_cells - 1)
        A_bc_right, B_bc_right, C_bc_right = self.right_bc_func(phi[-1], self.r_faces[-1])
        
        if abs(B_bc_right) < 1e-14:
            # Dirichlet BC
            phi_boundary = C_bc_right / A_bc_right
            # Average phi to get temperature for diffusion coefficient
            phi_avg = 0.5 * (phi[-1] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
            D_boundary = self.get_diffusion_coefficient(T_avg, r=self.r_faces[-1])
            
            dx_half = self.r_faces[-1] - self.r_centers[-1]
            flux_coeff = (self.A_faces[-1] * D_boundary) / (self.V_cells[-1] * dx_half)
            
            A_tri['diag'][-1] += flux_coeff
            rhs[-1] += flux_coeff * phi_boundary
        else:
            # Robin BC
            T_avg = self.T[-1] if hasattr(self, 'T') else 1.0
            D_boundary = self.get_diffusion_coefficient(T_avg, r=self.r_faces[-1])
            flux_coeff = (self.A_faces[-1] * D_boundary * A_bc_right) / (B_bc_right * self.V_cells[-1])
            A_tri['diag'][-1] += flux_coeff
            rhs[-1] += self.A_faces[-1] * D_boundary * C_bc_right / (B_bc_right * self.V_cells[-1])
    
    def check_matrix_conditioning(self, A_tri):
        """Check if the tridiagonal matrix is well-conditioned"""
        
        n = A_tri.shape[1]
        if n < 2:
            return True
            
        # Check diagonal dominance and positivity
        for i in range(n):
            diag = abs(A_tri[1, i])  # Main diagonal
            off_diag_sum = 0.0
            
            # Sum off-diagonal elements
            if i > 0 and A_tri.shape[0] > 0:  # Lower diagonal
                off_diag_sum += abs(A_tri[0, i])
            if i < n-1 and A_tri.shape[0] > 2:  # Upper diagonal  
                off_diag_sum += abs(A_tri[2, i])
            
            # Check diagonal dominance
            if diag <= off_diag_sum * 0.9:  # Allow some tolerance
                return False
                
            # Check for very small diagonal elements
            if diag < 1e-15:
                return False
        
        return True

    def time_step(self, n_steps=1, source=None, verbose=True):
        """Advance solution by n_steps time steps for coupled φ-T system
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        source : ndarray, callable, or None, optional
            Source term for phi equation. Can be:
            - ndarray: constant source array
            - callable: function(t) returning source array for time t
            - None: no source (default)
        verbose : bool
            Print progress information
        """
        
        for step in range(n_steps):
            if verbose:
                print(f"Time step {step+1}/{n_steps}")
            
            # Store previous solution
            phi_prev = self.phi.copy()
            T_prev = self.T.copy()
            
            # Evaluate source at current time if it's a function
            current_time = step * self.dt
            if callable(source):
                source_at_t = source(current_time)
            else:
                source_at_t = source
            
            # Newton iterations for coupled system
            self.phi, self.T = self.newton_step(phi_prev, T_prev, source=source_at_t, verbose=verbose)
            
            # Update old values for next time step
            self.phi_old = phi_prev.copy()
            self.T_old = T_prev.copy()
    
    def time_step_trbdf2(self, n_steps=1, Lambda=None, source=None, verbose=True):
        """Advance solution using TR-BDF2 method for coupled φ-T system
        
        TR-BDF2 is a two-stage composite method:
        Stage 1: Trapezoidal rule from t^n to t^{n+Λ}
        Stage 2: BDF2 from t^n, t^{n+Λ} to t^{n+1}
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        Lambda : float, optional
            Intermediate time fraction (default: 2 - sqrt(2) ≈ 0.586)
            Note: Using Λ to match notation in equations (8.61, 8.62)
        source : ndarray, callable, or None, optional
            Source term for phi equation. Can be:
            - ndarray: constant source array
            - callable: function(t) returning source array for time t
            - None: no source (default)
        verbose : bool
            Print progress information
        """
        if Lambda is None:
            Lambda = 2.0 - np.sqrt(2.0)  # Optimal value for L-stability
        
        # Store original parameters
        original_dt = self.dt
        original_theta = self.theta
        
        for step in range(n_steps):
            if verbose:
                print(f"TR-BDF2 step {step+1}/{n_steps}")
            
            # Store solution at t^n
            phi_n = self.phi.copy()
            T_n = self.T.copy()
            
            # Evaluate source at current time if it's a function
            current_time = step * original_dt
            if callable(source):
                # For time-varying sources, evaluate at appropriate times
                source_stage1 = source(current_time)  # Use source at t^n for stage 1
                source_stage2 = source(current_time)  # Use source at t^n for stage 2
            else:
                # For constant sources, use as-is
                source_stage1 = source
                source_stage2 = source
            
            # Stage 1: Trapezoidal rule (θ=0.5) to intermediate point t^{n+Λ}
            if verbose:
                print(f"  Stage 1: TR to t^{{n+{Lambda:.3f}}}")
            
            self.dt = Lambda * original_dt
            self.theta = 0.5  # Trapezoidal rule
            phi_Lambda, T_Lambda = self.newton_step(phi_n, T_n, source=source_stage1, verbose=verbose)
            
            # Stage 2: BDF2 from t^n and t^{n+Λ} to t^{n+1}
            if verbose:
                print(f"  Stage 2: BDF2 to t^{{n+1}}")
            
            self.dt = original_dt
            # Set initial guess to be intermediate solution
            self.phi = phi_Lambda.copy()
            self.T = T_Lambda.copy()
            self.phi, self.T = self.newton_step_bdf2(phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=source_stage2, verbose=verbose)
            
            # Update old values for next time step
            self.phi_old = phi_n.copy()
            self.T_old = T_n.copy()
        
        # Restore original parameters
        self.dt = original_dt
        self.theta = original_theta
    
    def newton_step_bdf2(self, phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=None, verbose=True):
        """Perform Newton iterations for BDF2 stage of TR-BDF2
        
        Implements equations (8.61a) and (8.61b) for the second stage using sequential solution:
        1. Solve equation 8.61a for φ^{n+1}
        2. Solve equation 8.61b for T_{n+1} using φ^{n+1}
        
        Parameters:
        -----------
        source : ndarray or None, optional
            Source term for phi equation. Default: None (no source)
        """
        # Initial guess: use intermediate solution
        phi_star = self.phi.copy()
        T_star = self.T.copy()
        
        for k in range(self.max_newton_iter):
            # Step 1: Solve equation 8.61a for φ^{n+1}
            A_phi, rhs_phi = self.assemble_phi_equation_trbdf2(
                phi_star, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=source)
            
            # Apply boundary conditions for φ
            self.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
            
            # Solve tridiagonal system for φ^{n+1}
            phi_np1 = solve_tridiagonal(A_phi, rhs_phi)
            
            # Step 2: Solve equation 8.61b for T_{n+1}
            T_np1 = self.solve_T_equation_trbdf2(
                phi_np1, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda)
            
            # Check for validity
            if np.any(~np.isfinite(phi_np1)) or np.any(~np.isfinite(T_np1)):
                if verbose:
                    print(f"    BDF2 Newton iteration {k+1} produced invalid values!")
                phi_np1 = phi_star + 0.01 * (phi_np1 - phi_star)
                T_np1 = T_star + 0.01 * (T_np1 - T_star)
            
            # Check for negative values
            if np.any(phi_np1 <= 0) or np.any(T_np1 <= 0):
                alpha = 1.0
                for i in range(len(phi_star)):
                    if phi_np1[i] <= 0:
                        alpha = min(alpha, 0.9 * phi_star[i] / (phi_star[i] - phi_np1[i]))
                    if T_np1[i] <= 0:
                        alpha = min(alpha, 0.9 * T_star[i] / (T_star[i] - T_np1[i]))
                
                alpha = max(0.1, min(alpha, 0.9))
                phi_np1 = phi_star + alpha * (phi_np1 - phi_star)
                T_np1 = T_star + alpha * (T_np1 - T_star)
                
                if verbose:
                    print(f"    BDF2 damping: alpha = {alpha:.3f}")
            
            # Compute residuals
            r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
            r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
            
            if verbose and k == 0:
                print(f"    BDF2 Newton iteration {k+1}: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
            
            if r_phi < self.newton_tol and r_T < self.newton_tol:
                if verbose:
                    print(f"    BDF2 Newton converged in {k+1} iterations")
                    print(f"      r_φ={r_phi:.2e}, r_T={r_T:.2e}")
                return phi_np1, T_np1
            
            # Update for next iteration
            phi_star = phi_np1.copy()
            T_star = T_np1.copy()
        
        if verbose:
            print(f"    BDF2 Newton max iterations: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
        return phi_star, T_star
    
    def assemble_phi_equation_trbdf2(self, phi_star, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=None):
        """Assemble equation 8.61a for φ^{n+1} in TR-BDF2
        
        Equation 8.61a:
        (1/c·Δt)·[(2-Λ)/(1-Λ)·φⁿ⁺¹ - 1/(Λ(1-Λ))·φⁿ⁺ᴧ + (1-Λ)/Λ·φⁿ] + ∇·D∇φⁿ⁺¹ =
            f_TB·σ_P(acT_★⁴ - φⁿ⁺¹) - (1-f_TB)·Δe/Δt + S
        
        where S is an optional source term
        
        NOTE: Uses +∇·D∇φ (positive sign), same as theta-method
        
        Parameters:
        -----------
        source : ndarray or None, optional
            Source term S in units [energy/(volume·time)]. Default: None (no source)
        
        Returns:
        --------
        A_tri : dict with 'sub', 'diag', 'super'
            Tridiagonal matrix coefficients
        rhs : ndarray
            Right-hand side vector
        """
        n_cells = len(phi_star)
        dt = self.dt  # Full time step
        
        # Initialize tridiagonal matrix and RHS
        sub = np.zeros(n_cells - 1)
        diag = np.zeros(n_cells)
        super_diag = np.zeros(n_cells - 1)
        rhs = np.zeros(n_cells)
        
        # BDF2 coefficients from equation (8.62)
        c_np1 = (2.0 - Lambda) / (1.0 - Lambda)  # Coefficient of solution at n+1
        c_nL = -1.0 / (Lambda * (1.0 - Lambda))   # Coefficient of solution at n+Λ
        c_n = (1.0 - Lambda) / Lambda             # Coefficient of solution at n
        
        # Evaluate material energies
        e_star = np.array([self.material_energy_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        e_nL = np.array([self.material_energy_func(T_Lambda[i], self.r_centers[i]) for i in range(n_cells)])
        e_n = np.array([self.material_energy_func(T_n[i], self.r_centers[i]) for i in range(n_cells)])
        
        # Compute Δe from equation (8.62)
        Delta_e = c_np1 * e_star + c_nL * e_nL + c_n * e_n
        
        # Evaluate coupling parameters at linearization point T_★
        sigma_P = np.array([self.planck_opacity_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        f_TB = np.array([self.get_f_factor_trbdf2(T_star[i], dt, Lambda, self.r_centers[i]) for i in range(n_cells)])
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Diffusion coefficients at linearization point (flux-limited)
        D_faces = np.zeros(len(self.r_faces))
        for i in range(1, len(self.r_faces) - 1):
            # Temperature at face (arithmetic average)
            T_face = 0.5 * (T_star[i-1] + T_star[i])
            
            # φ values on either side of face and distance between cell centers
            phi_left = phi_star[i-1]
            phi_right = phi_star[i]
            dx = self.r_centers[i] - self.r_centers[i-1]
            
            # Compute flux-limited diffusion coefficient
            D_faces[i] = self.get_diffusion_coefficient(T_face, phi_left, phi_right, dx, self.r_faces[i])
        
        # Boundary faces (use cell-centered values, no flux limiting at boundaries)
        D_faces[0] = self.get_diffusion_coefficient(T_star[0], r=self.r_faces[0])
        D_faces[-1] = self.get_diffusion_coefficient(T_star[-1], r=self.r_faces[-1])
        
        # Assemble equation for each cell
        for i in range(n_cells):
            # Time derivative term: (c_np1/(c·Δt)) on diagonal
            diag[i] = c_np1 / (C_LIGHT * dt)
            
            # Diffusion operator: +∇·D∇φ^{n+1} (fully implicit, SAME SIGN as theta-method)
            # Standard ∇·D∇φ discretization: +coeff on diag, -coeff on off-diag
            # So +∇·D∇φ gives: +coeff on diag, -coeff on off-diag
            V_i = self.V_cells[i]
            
            if i > 0:  # Left face
                A_left = self.A_faces[i]
                dr_left = self.r_centers[i] - self.r_centers[i-1]
                coeff = A_left * D_faces[i] / (dr_left * V_i)
                diag[i] += coeff  # Positive for +∇·D∇φ
                sub[i-1] = -coeff  # Negative for +∇·D∇φ
            
            if i < n_cells - 1:  # Right face
                A_right = self.A_faces[i+1]
                dr_right = self.r_centers[i+1] - self.r_centers[i]
                coeff = A_right * D_faces[i+1] / (dr_right * V_i)
                diag[i] += coeff  # Positive for +∇·D∇φ
                super_diag[i] = -coeff  # Negative for +∇·D∇φ
            
            # Coupling term: f_TB·σ_P·φ^{n+1}
            diag[i] += f_TB[i] * sigma_P[i]
            
            # RHS from equation 8.61a
            # Time derivative history terms: (-c_nL·φⁿ⁺ᴧ - c_n·φⁿ)/(c·Δt)
            rhs[i] = (-c_nL * phi_Lambda[i] - c_n * phi_n[i]) / (C_LIGHT * dt)
            # Source terms: f_TB·σ_P·acT_★⁴ - (1-f_TB)·Δe/Δt
            rhs[i] += f_TB[i] * sigma_P[i] * acT4_star[i]
            rhs[i] -= (1.0 - f_TB[i]) * Delta_e[i] / dt
        
        # Add external source term if provided
        if source is not None:
            rhs += source
        
        A_tri = {'sub': sub, 'diag': diag, 'super': super_diag}
        return A_tri, rhs
    
    def solve_T_equation_trbdf2(self, phi_np1, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda):
        """Solve equation 8.61b for T_{n+1} given φ^{n+1}
        
        Equation 8.61b:
        (1/Δt)·[(2-Λ)/(1-Λ)·e(T_{n+1}) - 1/(Λ(1-Λ))·e(T_{n+Λ}) + (1-Λ)/Λ·e(T_n)] =
            f_TB·σ_P(φ̄ - acT_★⁴) + (1-f_TB)·Δe/Δt
        
        where φ̄ = φ^{n+1} in TR-BDF2 (fully implicit for T equation)
        
        Parameters:
        -----------
        phi_np1 : ndarray
            Solution φ^{n+1} from equation 8.61a
        T_star : ndarray
            Linearization point T_★
        phi_n, T_n : ndarray
            Solutions at time level n
        phi_Lambda, T_Lambda : ndarray
            Solutions at intermediate time level n+Λ
        Lambda : float
            TR-BDF2 parameter (2 - √2)
        
        Returns:
        --------
        T_np1 : ndarray
            Solution T_{n+1}
        """
        n_cells = len(phi_np1)
        dt = self.dt
        T_np1 = np.zeros(n_cells)
        
        # BDF2 coefficients
        c_np1 = (2.0 - Lambda) / (1.0 - Lambda)
        c_nL = -1.0 / (Lambda * (1.0 - Lambda))
        c_n = (1.0 - Lambda) / Lambda
        
        # Evaluate material energies at previous time levels
        e_nL = np.array([self.material_energy_func(T_Lambda[i], self.r_centers[i]) for i in range(n_cells)])
        e_n = np.array([self.material_energy_func(T_n[i], self.r_centers[i]) for i in range(n_cells)])
        e_star = np.array([self.material_energy_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        Delta_e = c_np1 * e_star + c_nL * e_nL + c_n * e_n
        
        # Evaluate coupling parameters at linearization point
        sigma_P = np.array([self.planck_opacity_func(T_star[i], self.r_centers[i]) for i in range(n_cells)])
        f_TB = np.array([self.get_f_factor_trbdf2(T_star[i], dt, Lambda, self.r_centers[i]) for i in range(n_cells)])
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Solve for T_{n+1} at each cell
        # c_np1·e(T_{n+1})/Δt = -c_nL·e(T_{n+Λ})/Δt - c_n·e(T_n)/Δt + f_TB·σ_P(φ^{n+1} - acT_★⁴) + (1-f_TB)·Δe/Δt
        for i in range(n_cells):
            rhs_T = (-c_nL * e_nL[i] - c_n * e_n[i]) / dt
            rhs_T += f_TB[i] * sigma_P[i] * (phi_np1[i] - acT4_star[i])
            rhs_T += (1.0 - f_TB[i]) * Delta_e[i] / dt
            
            # e_{n+1} = (Δt/c_np1) * rhs_T
            e_np1 = (dt / c_np1) * rhs_T
            
            # Invert material energy to get T using user-supplied inverse function
            T_np1[i] = self.inverse_material_energy_func(e_np1, self.r_centers[i])
        
        return T_np1

# =============================================================================
# OLD EQUILIBRIUM DIFFUSION METHODS (NOT USED IN NON-EQUILIBRIUM SOLVER)
# =============================================================================
# The following methods are from the original equilibrium solver and are kept
# for reference but not used by the non-equilibrium NonEquilibriumRadiationDiffusionSolver class.
# They can be safely removed if not needed.

# def assemble_system_bdf2(...): OLD EQUILIBRIUM BDF2 METHOD - NOT USED
# def get_solution_old(...): OLD METHOD - replaced by new get_solution() returning (r, phi, T)
# def assemble_diffusion_matrix(...): OLD METHOD - replaced by inline assembly in assemble_coupled_system
# def assemble_system(...): OLD METHOD - replaced by assemble_coupled_system
# def apply_boundary_conditions(...): OLD METHOD - replaced by apply_boundary_conditions_coupled

    def assemble_coupled_system(self, phi_k, T_k, phi_prev, T_prev, theta=1.0):
        """Assemble coupled block tridiagonal system for non-equilibrium diffusion
        
        Implements equations (8.59a) and (8.59b) with θ-method:
        
        φ equation:
        (1/c·Δt)(φⁿ⁺¹ - φⁿ) - θ∇·D∇φⁿ⁺¹ - (1-θ)∇·D∇φⁿ = 
            σ_P·f(acT_★⁴ - φ̃) - (1-f)·Δe/Δt
        
        T equation:
        (e(Tₙ₊₁) - e(Tₙ))/Δt = f·σ_P(φ̃ - acT_★⁴) + (1-f)·Δe/Δt
        
        where φ̃ = θφⁿ⁺¹ + (1-θ)φⁿ and f = 1/(1 + β·σ_P·c·θ·Δt)
        
        Returns:
        --------
        A_blocks : dict with 'diag', 'super', 'sub'
            Block tridiagonal matrix
        rhs_phi, rhs_T : ndarray
            Right-hand side vectors
        """
        n_cells = len(phi_k)
        dt = self.dt
        
        # Initialize block matrix and RHS
        diag_blocks = np.zeros((n_cells, 2, 2))  # 2x2 block for each cell
        rhs_phi =np.zeros(n_cells)
        rhs_T = np.zeros(n_cells)
        
        # Compute φ̃ for nonlinear coupling: φ̃ = θφⁿ⁺¹ + (1-θ)φⁿ
        # In Newton: use current iterate φ_k for φⁿ⁺¹
        phi_tilde = theta * phi_k + (1.0 - theta) * phi_prev
        
        # Evaluate material energy at current and previous time steps
        e_n = np.array([self.material_energy_func(T_prev[i]) for i in range(n_cells)])
        e_np1 = np.array([self.material_energy_func(T_k[i]) for i in range(n_cells)])
        
        # Compute Δe = e(T★) - e(Tₙ) where T★ = T_k (linearization point)
        # From definition provided by user
        Delta_e = e_np1 - e_n
        
        # Evaluate coupling parameters at linearization point (T_k for Newton)
        sigma_P_cells = np.array([self.planck_opacity_func(T_k[i]) for i in range(n_cells)])
        f_cells = np.array([self.get_f_factor(T_k[i], dt, theta) for i in range(n_cells)])
        
        # Compute acT★⁴ at linearization point
        acT4_star = A_RAD * C_LIGHT * T_k**4
        
        # Assemble diffusion matrix for φ (spatial coupling)
        D_cells = np.array([self.get_diffusion_coefficient(T_k[i]) for i in range(n_cells)])
        
        # Face diffusion coefficients (flux-limited)
        D_faces = np.zeros(len(self.r_faces))
        for i in range(1, len(self.r_faces) - 1):
            # Average temperature at face
            T_face = 0.5 * (T_k[i-1] + T_k[i])
            
            # φ values on either side of face and distance between cell centers
            phi_left = phi_k[i-1]
            phi_right = phi_k[i]
            dx = self.r_centers[i] - self.r_centers[i-1]
            
            # Compute flux-limited diffusion coefficient
            D_faces[i] = self.get_diffusion_coefficient(T_face, phi_left, phi_right, dx)
        
        # Boundary faces (use cell-centered values, no flux limiting at boundaries)
        D_faces[0] = D_cells[0]
        D_faces[-1] = D_cells[-1]
        
        # Compute diffusion coefficients for super/sub diagonals
        super_phi = np.zeros(n_cells - 1)
        sub_phi = np.zeros(n_cells - 1)
        
        for i in range(n_cells):
            # === Assemble φ equation (row 0 of 2x2 block) ===
            
            # Time derivative term: (1/c·Δt) on diagonal
            diag_blocks[i, 0, 0] = 1.0 / (C_LIGHT * dt)
            
            # Diffusion operator: -θ∇·D∇φ
            # Contributions to diagonal and off-diagonals
            V_i = self.V_cells[i]
            
            if i > 0:  # Left face contribution
                A_left = self.A_faces[i]
                dr_left = self.r_centers[i] - self.r_centers[i-1]
                coeff_left = theta * A_left * D_faces[i] / (dr_left * V_i)
                diag_blocks[i, 0, 0] += coeff_left  # Diagonal
                sub_phi[i-1] = -coeff_left  # Sub-diagonal (negative for diffusion)
            
            if i < n_cells - 1:  # Right face contribution
                A_right = self.A_faces[i+1]
                dr_right = self.r_centers[i+1] - self.r_centers[i]
                coeff_right = theta * A_right * D_faces[i+1] / (dr_right * V_i)
                diag_blocks[i, 0, 0] += coeff_right  # Diagonal
                super_phi[i] = -coeff_right  # Super-diagonal (negative for diffusion)
            
            # Coupling to T through σ_P·f term (off-diagonal block element)
            # ∂/∂T of [σ_P·f·acT⁴] term: σ_P·f·(4acT³)
            # This appears on RHS, so Jacobian is negative
            # Also from +(1-f)·Δe/Δt term: (1-f)·∂e/∂T/Δt
            deriv_acT4_dT = 4.0 * A_RAD * C_LIGHT * T_k[i]**3
            cv = self.specific_heat_func(T_k[i])
            de_dT = RHO * cv
            diag_blocks[i, 0, 1] = -sigma_P_cells[i] * f_cells[i] * deriv_acT4_dT + (1.0 - f_cells[i]) * de_dT / dt
            
            # RHS for φ equation
            # Time derivative of φ: (1/c·Δt)φⁿ
            rhs_phi[i] = phi_prev[i] / (C_LIGHT * dt)
            
            # Explicit part of diffusion: (1-θ)∇·D∇φⁿ
            if theta < 1.0:
                # Would need to compute -(1-θ)L·φⁿ; for now assume θ=1
                pass
            
            # Coupling source term: σ_P·f(acT_★⁴ - φ̃) - (1-f)·Δe/Δt
            # Linearized: σ_P·f·acT_★⁴ - σ_P·f·φ̃ - (1-f)·Δe/Δt
            # The -σ_P·f·φ̃ term moves to LHS (diagonal), so RHS gets:
            rhs_phi[i] += sigma_P_cells[i] * f_cells[i] * acT4_star[i] 
            rhs_phi[i] -= (1.0 - f_cells[i]) * Delta_e[i] / dt
            
            # Move -σ_P·f·φ_tilde term: in Newton, φ_tilde ≈ θ·φⁿ⁺¹
            # So diagonal gets: +θ·σ_P·f
            diag_blocks[i, 0, 0] += theta * sigma_P_cells[i] * f_cells[i]
            # And RHS gets contribution from (1-θ)φⁿ part
            rhs_phi[i] += (1.0 - theta) * sigma_P_cells[i] * f_cells[i] * phi_prev[i]
            
            # === Assemble T equation (row 1 of 2x2 block) ===
            
            # Derivative of e(T) with respect to T: de/dT = ρ·c_v
            cv = self.specific_heat_func(T_k[i])
            de_dT = RHO * cv
            
            # Time derivative: (de/dT)/Δt on diagonal
            diag_blocks[i, 1, 1] = de_dT / dt
            
            # Coupling to φ through f·σ_P term (off-diagonal block element)
            # The term -f·σ_P·φ̃ appears; derivative w.r.t. φ: -θ·f·σ_P
            diag_blocks[i, 1, 0] = -theta * f_cells[i] * sigma_P_cells[i]
            
            # RHS for T equation
            # Time derivative: e(Tⁿ)/Δt
            rhs_T[i] = e_n[i] / dt
            
            # Coupling source: -f·σ_P·acT_★⁴ + f·σ_P·φ + (1-f)·Δe/Δt
            # The f·σ_P·φ term is handled via diagonal, so RHS gets:
            rhs_T[i] -= f_cells[i] * sigma_P_cells[i] * acT4_star[i]  # NEGATIVE!
            rhs_T[i] += (1.0 - f_cells[i]) * Delta_e[i] / dt  # +(1-f)·Δe/Δt
            # Add explicit φ contribution from (1-θ)φⁿ
            rhs_T[i] -= (1.0 - theta) * f_cells[i] * sigma_P_cells[i] * phi_prev[i]  # Negative from moving to RHS
        
        # Package into blocks dictionary
        A_blocks = {
            'diag': diag_blocks,
            'super': super_phi,
            'sub': sub_phi
        }
        
        return A_blocks, rhs_phi, rhs_T
    
    def assemble_system(self, Er_k, Er_prev, theta=1.0):
        """Assemble tridiagonal system for Newton iteration
        
        Theta method time discretization with Newton's method.
        Separates spatial (diffusion) and temporal (time discretization) contributions.
        
        Parameters:
        -----------
        Er_k : array
            Current Newton iterate
        Er_prev : array
            Solution at previous time step
        theta : float, optional
            Time discretization parameter (default=1.0 for implicit Euler)
            theta=1.0: Fully implicit (backward Euler)
            theta=0.5: Crank-Nicolson
            theta=0.0: Fully explicit (forward Euler)
        """
        n_cells = len(Er_k)
        
        # Get diffusion matrix at current iterate (spatial operator)
        L_tri = self.assemble_diffusion_matrix(Er_k)
        
        # Initialize system matrix: α + theta*L
        # Note: L is assembled as a loss operator (positive on diagonal)
        # System: (α + theta*L) * Er^{n+1} = RHS
        A_tri = theta * L_tri
        rhs = np.zeros(n_cells)
        
        # Evaluate time discretization coefficients
        alpha_cells = np.array([self.get_alpha_coefficient(Er_k[i], self.dt) for i in range(n_cells)])
        
        # Compute Qhat at PREVIOUS time step (should not change during Newton iterations)
        qhat_cells = np.array([self.get_u_function(Er_prev[i], self.dt) for i in range(n_cells)])
        
        # Add explicit part of diffusion operator if theta < 1
        if theta < 1.0:
            L_prev = self.assemble_diffusion_matrix(Er_prev)
            L_prev_times_Er_prev = apply_tridiagonal(L_prev, Er_prev)
        else:
            L_prev_times_Er_prev = np.zeros(n_cells)
        
        # Add time discretization terms
        for i in range(n_cells):
            # Add alpha to diagonal: (α + theta*L) * Er = RHS
            A_tri[1, i] += alpha_cells[i]
            
            # Right-hand side for theta method:
            # (α + θL)E^{n+1} = αE^n + (1-θ)LE^n - u(E^{n+1}) + u(E^n)
            # Newton linearization: RHS = αE_k - u(E_k) + Qhat + (1-θ)L(E^n)E^n
            u_k = self.get_u_function(Er_k[i], self.dt)
            qhat = qhat_cells[i]
            
            rhs[i] = alpha_cells[i] * Er_k[i] - u_k + qhat - (1.0 - theta) * L_prev_times_Er_prev[i]
            
           
        # Final matrix validation
        if np.any(~np.isfinite(A_tri)) or np.any(~np.isfinite(rhs)):
            print("Warning: Matrix contains invalid values after assembly")
            # Clean up any remaining invalid values
            A_tri[~np.isfinite(A_tri)] = 0.0
            rhs[~np.isfinite(rhs)] = 0.0
        
        return A_tri, rhs
    
    def apply_boundary_conditions_coupled(self, A_blocks, rhs_phi, rhs_T, phi_k, T_k):
        """Apply boundary conditions for coupled φ-T system
        
        Modifies the boundary blocks and RHS for φ equation.
        T equation has no spatial derivatives, so no boundary conditions needed.
        """
        n_cells = len(phi_k)
        
        # Left boundary (i = 0) - only affects φ equation
        A_bc_left, B_bc_left, C_bc_left = self.left_bc_func(phi_k[0], self.r_faces[0])
        
        if abs(B_bc_left) < 1e-14:
            # Dirichlet BC for φ
            phi_ghost = C_bc_left / A_bc_left
            T_avg = T_k[0]  # Use cell temperature for D
            D_boundary = self.get_diffusion_coefficient(T_avg)
            
            dx_half = self.r_centers[0] - self.r_faces[0]
            flux_coeff = (self.A_faces[0] * D_boundary) / (self.V_cells[0] * dx_half)
            
            # Modify φ equation diagonal and RHS
            A_blocks['diag'][0, 0, 0] += flux_coeff
            rhs_phi[0] += flux_coeff * phi_ghost
        else:
            # Robin BC
            T_avg = T_k[0]
            D_boundary = self.get_diffusion_coefficient(T_avg)
            flux_coeff = (self.A_faces[0] * D_boundary * A_bc_left) / (B_bc_left * self.V_cells[0])
            A_blocks['diag'][0, 0, 0] += flux_coeff
            rhs_phi[0] += self.A_faces[0] * D_boundary * C_bc_left / (B_bc_left * self.V_cells[0])
        
        # Right boundary (i = n_cells - 1) - only affects φ equation
        A_bc_right, B_bc_right, C_bc_right = self.right_bc_func(phi_k[-1], self.r_faces[-1])
        
        if abs(B_bc_right) < 1e-14:
            # Dirichlet BC for φ
            phi_ghost = C_bc_right / A_bc_right
            T_avg = T_k[-1]
            D_boundary = self.get_diffusion_coefficient(T_avg)
            
            dx_half = self.r_faces[-1] - self.r_centers[-1]
            flux_coeff = (self.A_faces[-1] * D_boundary) / (self.V_cells[-1] * dx_half)
            
            A_blocks['diag'][-1, 0, 0] += flux_coeff
            rhs_phi[-1] += flux_coeff * phi_ghost
        else:
            # Robin BC
            T_avg = T_k[-1]
            D_boundary = self.get_diffusion_coefficient(T_avg)
            flux_coeff = (self.A_faces[-1] * D_boundary * A_bc_right) / (B_bc_right * self.V_cells[-1])
            A_blocks['diag'][-1, 0, 0] += flux_coeff
            rhs_phi[-1] += self.A_faces[-1] * D_boundary * C_bc_right / (B_bc_right * self.V_cells[-1])
    
    def apply_boundary_conditions(self, A_tri, rhs, Er_k):
        """Apply Robin boundary conditions by modifying the boundary equations"""
        n_cells = len(Er_k)
        
        # Left boundary (i = 0)
        A_bc_left, B_bc_left, C_bc_left = self.left_bc_func(Er_k[0], self.r_faces[0])
        
        if abs(B_bc_left) < 1e-14:
            # Dirichlet BC: Implement via numerical flux instead of overwriting equation
            # Ghost value at boundary: Er_{1/2} = C_bc / A_bc
            # Flux: F = D * (Er_0 - Er_{1/2}) / (Δx/2)
            # This preserves the equation structure
            Er_ghost = C_bc_left / A_bc_left
            # Use average of cell and boundary values for D (or could use boundary value)
            Er_avg = 0.5 * (Er_k[0] + Er_ghost)
            D_boundary = self.get_diffusion_coefficient(Er_avg)
            
            # Distance from face to cell center
            dx_half = self.r_centers[0] - self.r_faces[0]
            
            # Flux: F = -D * dEr/dx = -D * (Er_0 - Er_ghost) / dx_half
            #         = D * (Er_ghost - Er_0) / dx_half
            # Flux coefficient: D * A_face / (V_cell * dx_half)
            flux_coeff = (self.A_faces[0] * D_boundary) / (self.V_cells[0] * dx_half)
            
            # Add flux contribution to diagonal and RHS
            # dEr/dt = ... + (A/V) * F = ... + (A/V) * D * (Er_ghost - Er_0) / dx
            # Moving Er_0 term to LHS: ... - flux_coeff * Er_0 = ... + flux_coeff * Er_ghost
            A_tri[1, 0] += flux_coeff
            rhs[0] += flux_coeff * Er_ghost
            
        else:
            # Standard Robin BC
            D_boundary = self.get_diffusion_coefficient(Er_k[0])
            flux_coeff = (self.A_faces[0] * D_boundary * A_bc_left) / (B_bc_left * self.V_cells[0])
            A_tri[1, 0] += flux_coeff
            rhs[0] += self.A_faces[0] * D_boundary * C_bc_left / (B_bc_left * self.V_cells[0])
        
        # Right boundary (i = n_cells - 1)
        A_bc_right, B_bc_right, C_bc_right = self.right_bc_func(Er_k[-1], self.r_faces[-1])
        
        if abs(B_bc_right) < 1e-14:
            # Dirichlet BC: Implement via numerical flux
            Er_ghost = C_bc_right / A_bc_right
            # Use average of cell and boundary values for D (or could use boundary value)
            Er_avg = 0.5 * (Er_k[-1] + Er_ghost)
            D_boundary = self.get_diffusion_coefficient(Er_avg)
            
            # Distance from cell center to face
            dx_half = self.r_faces[-1] - self.r_centers[-1]
            
            # Flux coefficient: D * A_face / (V_cell * dx_half)
            flux_coeff = (self.A_faces[-1] * D_boundary) / (self.V_cells[-1] * dx_half)
            
            # Add flux contribution to diagonal and RHS
            A_tri[1, -1] += flux_coeff
            rhs[-1] += flux_coeff * Er_ghost
            
        else:
            # Standard Robin BC
            D_boundary = self.get_diffusion_coefficient(Er_k[-1])
            flux_coeff = (self.A_faces[-1] * D_boundary * A_bc_right) / (B_bc_right * self.V_cells[-1])
            A_tri[1, -1] += flux_coeff
            rhs[-1] += self.A_faces[-1] * D_boundary * C_bc_right / (B_bc_right * self.V_cells[-1])


# =============================================================================
# TEST PROBLEMS
# =============================================================================

def manufactured_solution_test():
    """
    Test non-equilibrium radiation diffusion with simple initial conditions
    """
    print("Running non-equilibrium diffusion test...")
    
    # Define custom material properties for testing
    def custom_rosseland_opacity(T):
        """Constant Rosseland opacity"""
        return 1.0  # cm⁻¹
    
    def custom_planck_opacity(T):
        """Constant Planck opacity"""
        return 10.0  # cm⁻¹
    
    def custom_specific_heat(T):
        """Constant specific heat (per unit mass)"""
        return CV_CONST
    
    def custom_material_energy(T):
        """Linear material energy density: e(T) = ρ*cv*T"""
        return RHO * custom_specific_heat(T) * T
    
    # Define custom boundary conditions for φ
    def left_bc(phi, x):
        """Left boundary: high temperature drive"""
        T_bc = 1.0  # keV
        phi_bc = A_RAD * C_LIGHT * T_bc**4  # φ = acT^4
        return 1.0, 0.0, phi_bc  # A, B, C for Robin BC: A*φ + B*(n·∇φ) = C
    
    def right_bc(phi, x):
        """Right boundary: low temperature"""
        T_bc = 0.316  # keV
        phi_bc = A_RAD * C_LIGHT * T_bc**4
        return 1.0, 0.0, phi_bc  # Dirichlet BC
    
    # Problem setup
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.1, r_max=1.0, n_cells=50, d=0,  # planar geometry
        dt=1e-4, max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=custom_rosseland_opacity,
        planck_opacity_func=custom_planck_opacity,
        specific_heat_func=custom_specific_heat,
        material_energy_func=custom_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    # Set initial condition: start from cold temperature, equilibrium φ
    def initial_T(r):
        return 0.3 + 0.0 * r  # Cold uniform temperature
    
    solver.set_initial_condition(T_init=initial_T)
    
    # Run time evolution
    print("Time stepping...")
    solver.time_step(n_steps=5, verbose=True)
    
    return solver


def transient_test():
    """Transient test for non-equilibrium diffusion"""
    print("Running non-equilibrium transient test...")
    
    # Temperature-dependent material properties
    def nonlinear_rosseland_opacity(T):
        """Temperature-dependent Rosseland opacity"""
        return 0.5 + 0.0 * T**2
    
    def nonlinear_planck_opacity(T):
        """Temperature-dependent Planck opacity"""
        return 5.0 + 0.0 * T
    
    def temperature_dependent_cv(T):
        """Temperature-dependent specific heat"""
        return CV_CONST * (1.0 + 0.0 * T)
    
    solver = NonEquilibriumRadiationDiffusionSolver(
        r_min=0.1, r_max=1.0, n_cells=40, d=1,  # Cylindrical geometry
        dt=1e-3, max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=nonlinear_rosseland_opacity,
        planck_opacity_func=nonlinear_planck_opacity,
        specific_heat_func=temperature_dependent_cv
    )
    
    # Initial condition: hot spot in center, non-equilibrium
    def initial_T(r):
        r0 = 0.5 * (solver.r_min + solver.r_max)
        sigma = 0.15
        return 0.5 + 0.5 * np.exp(-((r - r0) / sigma)**2)
    
    def initial_phi(r):
        # Start with lower φ than equilibrium (non-equilibrium initial state)
        T = initial_T(r)
        return 0.5 * A_RAD * C_LIGHT * T**4
    
    solver.set_initial_condition(phi_init=initial_phi, T_init=initial_T)
    
    # Time evolution
    print("Time evolution...")
    n_snapshots = 20
    solutions = []
    
    for i in range(n_snapshots):
        solver.time_step(n_steps=5, verbose=False)
        r, phi, T = solver.get_solution()
        solutions.append((i * 5 * solver.dt, r.copy(), phi.copy(), T.copy()))
    
    return solutions


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_solution(solver, title="Non-Equilibrium Radiation Diffusion"):
    """Plot current solution for φ and T"""
    r, phi, T = solver.get_solution()
    Er = Er_from_phi(phi)  # Convert φ to E_r for comparison
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot temperature
    ax1.plot(r, T, 'r-', linewidth=2, label='T (material)')
    T_eq = temperature_from_phi_equilibrium(phi)
    ax1.plot(r, T_eq, 'b--', linewidth=2, label='T_eq from φ', alpha=0.7)
    ax1.set_xlabel('Radius r (cm)')
    ax1.set_ylabel('Temperature (keV)')
    ax1.set_title('Temperature')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot φ and equilibrium φ
    ax2.plot(r, phi, 'b-', linewidth=2, label='φ = E_r·c')
    phi_eq = A_RAD * C_LIGHT * T**4
    ax2.plot(r, phi_eq, 'r--', linewidth=2, label='φ_eq = acT⁴', alpha=0.7)
    ax2.set_xlabel('Radius r (cm)')
    ax2.set_ylabel('φ (GJ/cm²)')
    ax2.set_title('Radiation Energy Flux Variable')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_transient_solutions(solutions):
    """Plot multiple time snapshots for φ and T"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
    
    for i, (t, r, phi, T) in enumerate(solutions):
        # Temperature
        ax1.plot(r, T, color=colors[i], linewidth=2, alpha=0.7)
        
        # φ
        ax2.plot(r, phi, color=colors[i], linewidth=2, alpha=0.7)
        
        # Non-equilibrium measure: |φ - acT⁴|/acT⁴
        phi_eq = A_RAD * C_LIGHT * T**4
        non_eq = np.abs(phi - phi_eq) / (phi_eq + 1e-10)
        ax3.semilogy(r, non_eq, color=colors[i], linewidth=2, alpha=0.7,
                     label=f't={t:.3f}' if i % 4 == 0 else '')
        
        # E_r (radiation energy density)
        Er = Er_from_phi(phi)
        ax4.plot(r, Er, color=colors[i], linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Radius r (cm)')
    ax1.set_ylabel('Temperature T (keV)')
    ax1.set_title('Material Temperature Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Radius r (cm)')
    ax2.set_ylabel('φ (GJ/cm²)')
    ax2.set_title('Radiation Variable φ Evolution')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Radius r (cm)')
    ax3.set_ylabel('|φ - acT⁴|/acT⁴')
    ax3.set_title('Non-Equilibrium Measure')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_xlabel('Radius r (cm)')
    ax4.set_ylabel('E_r (GJ/cm³)')
    ax4.set_title('Radiation Energy Density Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Non-Equilibrium Transient Evolution')
    plt.tight_layout()
    plt.show()


# NOTE: plot_convergence_study() is commented out as it uses the old equilibrium solver
# def plot_convergence_study():
#     """Study Newton convergence and mesh refinement"""
#     # TODO: Adapt for non-equilibrium solver
#     pass


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("1D Non-Equilibrium Radiation Diffusion Finite Volume Solver")
    print("="*70)
    
    # Test 1: Basic non-equilibrium test
    print("\n" + "="*50)
    print("Test 1: Non-Equilibrium Diffusion")
    print("="*50)
    
    solver1 = manufactured_solution_test()
    plot_solution(solver1, "Non-Equilibrium Test")
    
    # Test 2: Transient evolution  
    print("\n" + "="*50)
    print("Test 2: Non-Equilibrium Transient Evolution")
    print("="*50)
    
    solutions = transient_test()
    plot_transient_solutions(solutions)
    
    print("\n" + "="*70)
    print("Configuration Options:")
    print("="*70)
    print("Material properties:")
    print("- rosseland_opacity_func(T): Rosseland mean opacity σ_R(T)")
    print("- planck_opacity_func(T): Planck mean opacity σ_P(T)")
    print("- specific_heat_func(T): Specific heat c_v(T)")
    print("- material_energy_func(T): Material energy density e(T)")
    print("\nSolver parameters:")
    print("- theta: 0=explicit, 0.5=Crank-Nicolson, 1=implicit (default)")
    print("- beta: Computed dynamically as β = 4aT_★³/C_v_★")
    print("- Use time_step() for θ-method")
    print("- Use time_step_trbdf2() for TR-BDF2 method")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

