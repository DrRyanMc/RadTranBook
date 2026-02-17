#!/usr/bin/env python3
"""
2D Finite Volume Solver for Equilibrium Radiation Diffusion Equation
using Newton iterations or Jacobian-Free Newton-Krylov (JFNK) method.

Supports both Cartesian (x-y) and cylindrical (r-z) coordinates.

PDE: ∇·(D(E_r) ∇E_r) + u(E_r) = Qhat
Time discretization: implicit Euler or TR-BDF2
Newton method with direct solve or JFNK
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from typing import Tuple, Callable, Optional
import time
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, gmres, LinearOperator


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


# =============================================================================
# MATERIAL PROPERTY FUNCTIONS (reused from 1D)
# =============================================================================

@njit
def temperature_from_Er(Er):
    """Convert radiation energy density to temperature: T = (E_r/a)^(1/4)"""
    return (Er / A_RAD) ** 0.25


@njit
def specific_heat_cv(T, coord1_val, coord2_val):
    """Specific heat capacity c_v(T, coord1, coord2). Default: constant.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    coord1_val : float
        First coordinate (x or r)
    coord2_val : float
        Second coordinate (y or z)
    """
    return CV_CONST


@njit
def material_energy_density(T, coord1_val, coord2_val):
    """Material energy density e(T, coord1, coord2) = ρ c_v(T, coord1, coord2) T
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    coord1_val : float
        First coordinate (x or r)
    coord2_val : float
        Second coordinate (y or z)
    """
    return RHO * specific_heat_cv(T, coord1_val, coord2_val) * T


@njit
def rosseland_opacity(Er, coord1_val, coord2_val):
    """Rosseland opacity σ_R(E_r, coord1, coord2). Default: constant for testing.
    
    Parameters:
    -----------
    Er : float
        Radiation energy density (GJ/cm³)
    coord1_val : float
        First coordinate (x or r)
    coord2_val : float
        Second coordinate (y or z)
    """
    return 0.1  # cm⁻¹


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def default_bc_left(Er_boundary, coord1_val, coord2_val, geometry='cartesian'):
    """Left boundary (x_min or r_min): Robin BC coefficients"""
def default_bc_left(Er_boundary, coord1_val, coord2_val, geometry='cartesian', time=0.0):
    """Left boundary (x_min or r_min): Robin BC coefficients"""
    return 1.0, 0.0, 1.0  # Dirichlet: Er = 1.0


def default_bc_right(Er_boundary, coord1_val, coord2_val, geometry='cartesian', time=0.0):
    """Right boundary (x_max or r_max): Robin BC coefficients"""
    return 1.0, 0.0, 0.1  # Dirichlet: Er = 0.1


def default_bc_bottom(Er_boundary, coord1_val, coord2_val, geometry='cartesian', time=0.0):
    """Bottom boundary (y_min or z_min): Robin BC coefficients"""
    return 0.0, 1.0, 0.0  # Neumann: dEr/dn = 0 (reflecting)


def default_bc_top(Er_boundary, coord1_val, coord2_val, geometry='cartesian', time=0.0):
    """Top boundary (y_max or z_max): Robin BC coefficients"""
    return 0.0, 1.0, 0.0  # Neumann: dEr/dn = 0 (reflecting)


# =============================================================================
# 2D GRID GENERATION AND GEOMETRY
# =============================================================================

def generate_2d_grid(coord1_min: float, coord1_max: float, n1_cells: int,
                     coord2_min: float, coord2_max: float, n2_cells: int,
                     stretch_factor1: float = 1.0, stretch_factor2: float = 1.0) -> Tuple:
    """
    Generate 2D structured grid
    
    Parameters:
    -----------
    coord1_min, coord1_max : float
        First coordinate domain boundaries (x or r)
    n1_cells : int
        Number of cells in first direction
    coord2_min, coord2_max : float
        Second coordinate domain boundaries (y or z)
    n2_cells : int
        Number of cells in second direction
    stretch_factor1, stretch_factor2 : float
        Grid stretching (1.0 = uniform)
    
    Returns:
    --------
    coord1_faces : ndarray (n1_cells + 1,)
        Face positions in first direction
    coord1_centers : ndarray (n1_cells,)
        Cell center positions in first direction
    coord2_faces : ndarray (n2_cells + 1,)
        Face positions in second direction
    coord2_centers : ndarray (n2_cells,)
        Cell center positions in second direction
    """
    # First direction
    if stretch_factor1 == 1.0:
        coord1_faces = np.linspace(coord1_min, coord1_max, n1_cells + 1)
    else:
        xi = np.linspace(0, 1, n1_cells + 1)
        coord1_faces = coord1_min + (coord1_max - coord1_min) * \
                       ((stretch_factor1**xi - 1) / (stretch_factor1 - 1))
    
    coord1_centers = 0.5 * (coord1_faces[:-1] + coord1_faces[1:])
    
    # Second direction
    if stretch_factor2 == 1.0:
        coord2_faces = np.linspace(coord2_min, coord2_max, n2_cells + 1)
    else:
        xi = np.linspace(0, 1, n2_cells + 1)
        coord2_faces = coord2_min + (coord2_max - coord2_min) * \
                       ((stretch_factor2**xi - 1) / (stretch_factor2 - 1))
    
    coord2_centers = 0.5 * (coord2_faces[:-1] + coord2_faces[1:])
    
    return coord1_faces, coord1_centers, coord2_faces, coord2_centers


def compute_2d_geometry_factors(coord1_faces, coord2_faces, geometry='cartesian'):
    """
    Compute geometry factors for 2D coordinate system
    
    Parameters:
    -----------
    coord1_faces : ndarray
        Face positions in first direction (x or r)
    coord2_faces : ndarray
        Face positions in second direction (y or z)
    geometry : str
        'cartesian' for x-y or 'cylindrical' for r-z
    
    Returns:
    --------
    A1_faces : ndarray (n1_faces, n2_cells)
        Face areas perpendicular to first direction
    A2_faces : ndarray (n1_cells, n2_faces)
        Face areas perpendicular to second direction
    V_cells : ndarray (n1_cells, n2_cells)
        Cell volumes
    """
    n1_faces = len(coord1_faces)
    n2_faces = len(coord2_faces)
    n1_cells = n1_faces - 1
    n2_cells = n2_faces - 1
    
    # Face areas and cell volumes depend on geometry
    if geometry == 'cartesian':
        # Cartesian (x-y): areas and volumes are simple products
        # A1 (perpendicular to x): area = Δy × 1 (unit depth)
        # A2 (perpendicular to y): area = Δx × 1 (unit depth)
        # V: volume = Δx × Δy × 1
        
        A1_faces = np.zeros((n1_faces, n2_cells))
        A2_faces = np.zeros((n1_cells, n2_faces))
        V_cells = np.zeros((n1_cells, n2_cells))
        
        for i in range(n1_faces):
            for j in range(n2_cells):
                dy = coord2_faces[j+1] - coord2_faces[j]
                A1_faces[i, j] = dy  # Face area perpendicular to x
        
        for i in range(n1_cells):
            for j in range(n2_faces):
                dx = coord1_faces[i+1] - coord1_faces[i]
                A2_faces[i, j] = dx  # Face area perpendicular to y
        
        for i in range(n1_cells):
            for j in range(n2_cells):
                dx = coord1_faces[i+1] - coord1_faces[i]
                dy = coord2_faces[j+1] - coord2_faces[j]
                V_cells[i, j] = dx * dy
    
    elif geometry == 'cylindrical':
        # Cylindrical (r-z): r is radial, z is axial
        # A1 (perpendicular to r): area = 2π × r × Δz
        # A2 (perpendicular to z): area = π × (r_outer² - r_inner²)
        # V: volume = π × (r_outer² - r_inner²) × Δz
        
        A1_faces = np.zeros((n1_faces, n2_cells))
        A2_faces = np.zeros((n1_cells, n2_faces))
        V_cells = np.zeros((n1_cells, n2_cells))
        
        for i in range(n1_faces):
            r = coord1_faces[i]
            for j in range(n2_cells):
                dz = coord2_faces[j+1] - coord2_faces[j]
                A1_faces[i, j] = 2.0 * np.pi * r * dz  # Cylindrical surface area
        
        for i in range(n1_cells):
            r_inner = coord1_faces[i]
            r_outer = coord1_faces[i+1]
            for j in range(n2_faces):
                A2_faces[i, j] = np.pi * (r_outer**2 - r_inner**2)  # Annular area
        
        for i in range(n1_cells):
            r_inner = coord1_faces[i]
            r_outer = coord1_faces[i+1]
            for j in range(n2_cells):
                dz = coord2_faces[j+1] - coord2_faces[j]
                V_cells[i, j] = np.pi * (r_outer**2 - r_inner**2) * dz
    
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    
    return A1_faces, A2_faces, V_cells


# =============================================================================
# 2D INDEXING UTILITIES
# =============================================================================

def cell_to_index(i, j, n1_cells, n2_cells):
    """Convert 2D cell indices (i,j) to 1D index for solution vector"""
    return i * n2_cells + j


def index_to_cell(idx, n1_cells, n2_cells):
    """Convert 1D index to 2D cell indices (i,j)"""
    i = idx // n2_cells
    j = idx % n2_cells
    return i, j


def reshape_1d_to_2d(Er_1d, n1_cells, n2_cells):
    """Reshape 1D solution vector to 2D array"""
    return Er_1d.reshape((n1_cells, n2_cells))


def reshape_2d_to_1d(Er_2d):
    """Reshape 2D array to 1D solution vector"""
    return Er_2d.flatten()


# =============================================================================
# HARMONIC AVERAGING (reused from 1D)
# =============================================================================

@njit
def harmonic_average_weighted(D_left, D_right, dx_left, dx_right):
    """Distance-weighted harmonic average for non-uniform grids"""
    if D_left <= 0 or D_right <= 0:
        return 0.0
    
    total_dx = dx_left + dx_right
    if total_dx <= 0:
        return 0.0
        
    return D_left * D_right * total_dx / (D_right * dx_left + D_left * dx_right)


# =============================================================================
# MAIN 2D SOLVER CLASS
# =============================================================================

class RadiationDiffusionSolver2D:
    """2D Finite Volume Radiation Diffusion Solver with JFNK support"""
    
    def __init__(self, coord1_min=0.1, coord1_max=1.0, n1_cells=20,
                 coord2_min=0.0, coord2_max=1.0, n2_cells=20,
                 geometry='cartesian', dt=1e-3,
                 max_newton_iter=10, newton_tol=1e-8,
                 stretch_factor1=1.0, stretch_factor2=1.0,
                 coord1_faces=None, coord2_faces=None,
                 rosseland_opacity_func=None, specific_heat_func=None,
                 material_energy_func=None,
                 left_bc_func=None, right_bc_func=None,
                 bottom_bc_func=None, top_bc_func=None,
                 theta=1.0, use_jfnk=False, gmres_tol=1e-6, gmres_maxiter=100,eps=1e-8):
        """
        Initialize 2D radiation diffusion solver
        
        Parameters:
        -----------
        coord1_min, coord1_max : float
            Domain boundaries in first direction (x or r) - ignored if coord1_faces provided
        n1_cells : int
            Number of cells in first direction - ignored if coord1_faces provided
        coord2_min, coord2_max : float
            Domain boundaries in second direction (y or z) - ignored if coord2_faces provided
        n2_cells : int
            Number of cells in second direction - ignored if coord2_faces provided
        geometry : str
            'cartesian' for x-y coordinates or 'cylindrical' for r-z
        dt : float
            Time step size
        max_newton_iter : int
            Maximum Newton iterations per time step
        newton_tol : float
            Newton convergence tolerance
        stretch_factor1, stretch_factor2 : float
            Grid stretching factors (ignored if custom faces provided)
        coord1_faces : ndarray or None
            Custom face positions in first direction. If provided, overrides coord1_min/max and n1_cells
        coord2_faces : ndarray or None
            Custom face positions in second direction. If provided, overrides coord2_min/max and n2_cells
        use_jfnk : bool
            Use Jacobian-Free Newton-Krylov method (True) or direct solve (False)
        gmres_tol : float
            GMRES tolerance for JFNK
        gmres_maxiter : int
            Maximum GMRES iterations
        """
        self.geometry = geometry
        self.dt = dt
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.theta = theta
        self.use_jfnk = use_jfnk
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        
        # Generate or use custom grid
        if coord1_faces is not None and coord2_faces is not None:
            # Use custom face arrays
            self.coord1_faces = np.array(coord1_faces)
            self.coord2_faces = np.array(coord2_faces)
            self.coord1_centers = 0.5 * (self.coord1_faces[:-1] + self.coord1_faces[1:])
            self.coord2_centers = 0.5 * (self.coord2_faces[:-1] + self.coord2_faces[1:])
            self.n1_cells = len(self.coord1_faces) - 1
            self.n2_cells = len(self.coord2_faces) - 1
            self.coord1_min = self.coord1_faces[0]
            self.coord1_max = self.coord1_faces[-1]
            self.coord2_min = self.coord2_faces[0]
            self.coord2_max = self.coord2_faces[-1]
        elif coord1_faces is not None or coord2_faces is not None:
            raise ValueError("Must provide both coord1_faces and coord2_faces, or neither")
        else:
            # Generate grid using standard method
            self.coord1_min = coord1_min
            self.coord1_max = coord1_max
            self.n1_cells = n1_cells
            self.coord2_min = coord2_min
            self.coord2_max = coord2_max
            self.n2_cells = n2_cells
            self.coord1_faces, self.coord1_centers, self.coord2_faces, self.coord2_centers = \
                generate_2d_grid(coord1_min, coord1_max, n1_cells,
                               coord2_min, coord2_max, n2_cells,
                               stretch_factor1, stretch_factor2)
        
        # Total number of cells
        self.n_total = self.n1_cells * self.n2_cells

        self.eps = eps
        
        # Total number of cells
        self.n_total = self.n1_cells * self.n2_cells
        
        # Compute geometry factors
        self.A1_faces, self.A2_faces, self.V_cells = \
            compute_2d_geometry_factors(self.coord1_faces, self.coord2_faces, geometry)
        
        # Solution arrays (stored as 1D vectors)
        self.Er = np.ones(self.n_total)
        self.Er_old = np.ones(self.n_total)
        
        # Time tracking
        self.current_time = 0.0
        
        # Material property functions
        self.rosseland_opacity_func = rosseland_opacity_func or rosseland_opacity
        self.specific_heat_func = specific_heat_func or specific_heat_cv
        self.material_energy_func = material_energy_func or material_energy_density
        
        # Boundary condition functions
        self.left_bc_func = left_bc_func or default_bc_left
        self.right_bc_func = right_bc_func or default_bc_right
        self.bottom_bc_func = bottom_bc_func or default_bc_bottom
        self.top_bc_func = top_bc_func or default_bc_top
        
        coord_names = {'cartesian': 'x-y', 'cylindrical': 'r-z'}
        print(f"Initialized 2D solver ({coord_names[geometry]} geometry)")
        print(f"  Grid: {self.n1_cells} × {self.n2_cells} cells")
        print(f"  Domain: [{self.coord1_min:.3f}, {self.coord1_max:.3f}] × [{self.coord2_min:.3f}, {self.coord2_max:.3f}]")
        if coord1_faces is not None:
            print(f"  Using custom face arrays")
        print(f"  Δt = {dt:.2e}, solver = {'JFNK' if use_jfnk else 'Direct'}")
    
    def get_diffusion_coefficient(self, Er, coord1_val, coord2_val):
        """Get diffusion coefficient D = c/(3*σ_R)"""
        return C_LIGHT / (3.0 * self.rosseland_opacity_func(Er, coord1_val, coord2_val))
    
    def get_dudEr(self, Er, coord1_val, coord2_val):
        """Get du/dEr derivative for Newton linearization"""
        T = temperature_from_Er(np.abs(Er))
        cv = self.specific_heat_func(T, coord1_val, coord2_val)
        rho_cv = RHO * cv
        term = rho_cv / (4.0 * A_RAD**0.25 * np.abs(Er)**0.75)
        return 1.0 + term
    
    def get_u_function(self, Er, coord1_val, coord2_val):
        """Get u function: u = (1/dt) * (e_mat + Er)"""
        T = temperature_from_Er(Er)
        e_mat = self.material_energy_func(T, coord1_val, coord2_val)
        return (1.0 / self.dt) * (e_mat + Er)
    
    def set_initial_condition(self, Er_init):
        """Set initial radiation energy density
        
        Parameters:
        -----------
        Er_init : float, ndarray, or callable
            Initial condition (constant, array, or function of (coord1, coord2))
        """
        if callable(Er_init):
            # Function of (coord1, coord2)
            Er_2d = np.zeros((self.n1_cells, self.n2_cells))
            for i in range(self.n1_cells):
                for j in range(self.n2_cells):
                    Er_2d[i, j] = Er_init(self.coord1_centers[i], self.coord2_centers[j])
            self.Er = reshape_2d_to_1d(Er_2d)
            self.Er_old = self.Er.copy()
        elif isinstance(Er_init, np.ndarray):
            if Er_init.shape == (self.n1_cells, self.n2_cells):
                self.Er = reshape_2d_to_1d(Er_init)
                self.Er_old = self.Er.copy()
            elif Er_init.shape == (self.n_total,):
                self.Er = Er_init.copy()
                self.Er_old = Er_init.copy()
            else:
                raise ValueError("Array shape mismatch for initial condition")
        else:
            # Constant
            self.Er = np.full(self.n_total, Er_init)
            self.Er_old = np.full(self.n_total, Er_init)
        
        if np.any(self.Er <= 0):
            raise ValueError("Initial radiation energy density must be positive everywhere")
    
    def assemble_diffusion_operator(self, Er_1d):
        """
        Assemble 2D diffusion operator in sparse matrix form
        
        Returns the sparse matrix for: L(Er) = ∇·(D ∇Er)
        """
        # Use lil_matrix for efficient construction
        L = lil_matrix((self.n_total, self.n_total))
        
        # Reshape to 2D for easier indexing
        Er_2d = reshape_1d_to_2d(Er_1d, self.n1_cells, self.n2_cells)
        
        # Evaluate diffusion coefficients at cell centers
        D_cells = np.zeros((self.n1_cells, self.n2_cells))
        for i in range(self.n1_cells):
            for j in range(self.n2_cells):
                D_cells[i, j] = self.get_diffusion_coefficient(Er_2d[i, j], 
                                                                self.coord1_centers[i],
                                                                self.coord2_centers[j])
        
        #for the faces, compute harmonic averages using D_{i-1}((T_i + T_{i-1})/2) and D_i((T_i + T_{i-1})/2)
        #averaging the termparatures across the face makes sure that a wave can propagate without getting stuck due to low diffusion coefficient on one side
        # Interior cells: assemble diffusion stencil
        for i in range(self.n1_cells):
            for j in range(self.n2_cells):
                idx_center = cell_to_index(i, j, self.n1_cells, self.n2_cells)
                V = self.V_cells[i, j]
                
                # First direction (x or r) fluxes
                # Left face (i-1/2, j)
                if i > 0:
                    idx_left = cell_to_index(i-1, j, self.n1_cells, self.n2_cells)
                    d_left = self.coord1_centers[i] - self.coord1_centers[i-1]
                    T_avg = 0.5 * (temperature_from_Er(Er_2d[i, j]) + temperature_from_Er(Er_2d[i-1, j]))
                    # Compute D at face location (average of cell centers)
                    coord1_face = self.coord1_faces[i] #0.5 * (self.coord1_centers[i] + self.coord1_centers[i-1])
                    coord2_face = self.coord2_centers[j]
                    D_left = self.get_diffusion_coefficient(A_RAD * T_avg**4, 
                                                            coord1_face-self.eps, coord2_face)
                    D_right = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face+self.eps, coord2_face)
                    D_face = harmonic_average_weighted(D_left, D_right, coord1_face-self.coord1_centers[i-1], 
                                                            self.coord1_centers[i]- coord1_face)
                    #D_face = 0.5 * (D_cells[i, j] + D_cells[i-1, j])
                    coeff = self.A1_faces[i, j] * D_face / (d_left * V)
                    L[idx_center, idx_left] = -coeff
                    L[idx_center, idx_center] += coeff
                
                # Right face (i+1/2, j)
                if i < self.n1_cells - 1:
                    idx_right = cell_to_index(i+1, j, self.n1_cells, self.n2_cells)
                    d_right = self.coord1_centers[i+1] - self.coord1_centers[i]
                    T_avg = 0.5 * (temperature_from_Er(Er_2d[i, j]) + temperature_from_Er(Er_2d[i+1, j]))
                    # Compute D at face location (average of cell centers)
                    coord1_face = self.coord1_faces[i+1]
                    coord2_face = self.coord2_centers[j]
                    D_left = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face-self.eps, coord2_face)
                    D_right = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face+self.eps, coord2_face)
                    D_face = harmonic_average_weighted(D_left, D_right, self.coord1_faces[i+1] - self.coord1_centers[i], 
                                                       self.coord1_centers[i+1] - self.coord1_faces[i+1])
                    #D_face = 0.5 * (D_cells[i, j] + D_cells[i+1, j])
                    coeff = self.A1_faces[i+1, j] * D_face / (d_right * V)
                    L[idx_center, idx_right] = -coeff
                    L[idx_center, idx_center] += coeff
                
                # Second direction (y or z) fluxes
                # Bottom face (i, j-1/2)
                if j > 0:
                    idx_bottom = cell_to_index(i, j-1, self.n1_cells, self.n2_cells)
                    d_bottom = self.coord2_centers[j] - self.coord2_centers[j-1]
                    T_avg = 0.5 * (temperature_from_Er(Er_2d[i, j]) + temperature_from_Er(Er_2d[i, j-1]))
                    # Compute D at face location (average of cell centers)
                    coord1_face = self.coord1_centers[i]
                    coord2_face = self.coord2_faces[j]
                    D_left = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face, coord2_face-self.eps)
                    D_right = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face, coord2_face+self.eps)
                    D_face = harmonic_average_weighted(D_left, D_right, 
                                                       self.coord2_faces[j] - self.coord2_centers[j-1], 
                                                       self.coord2_centers[j] - self.coord2_faces[j])
                    #D_face = 0.5 * (D_cells[i, j] + D_cells[i, j-1])
                    coeff = self.A2_faces[i, j] * D_face / (d_bottom * V)
                    L[idx_center, idx_bottom] = -coeff
                    L[idx_center, idx_center] += coeff
                
                # Top face (i, j+1/2)
                if j < self.n2_cells - 1:
                    idx_top = cell_to_index(i, j+1, self.n1_cells, self.n2_cells)
                    d_top = self.coord2_centers[j+1] - self.coord2_centers[j]
                    T_avg = 0.5 * (temperature_from_Er(Er_2d[i, j]) + temperature_from_Er(Er_2d[i, j+1]))
                    # Compute D at face location (average of cell centers)
                    coord1_face = self.coord1_centers[i]
                    coord2_face = self.coord2_faces[j+1]
                    D_left = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face, coord2_face-self.eps)
                    D_right = self.get_diffusion_coefficient(A_RAD * T_avg**4, coord1_face, coord2_face+self.eps)
                    D_face = harmonic_average_weighted(D_left, D_right, self.coord2_faces[j+1] - self.coord2_centers[j], 
                                                       self.coord2_centers[j+1] - self.coord2_faces[j+1])
                    #
                    #D_face = 0.5 * (D_cells[i, j] + D_cells[i, j+1])
                    coeff = self.A2_faces[i, j+1] * D_face / (d_top * V)
                    L[idx_center, idx_top] = -coeff
                    L[idx_center, idx_center] += coeff
        
        return L.tocsr()
    
    def apply_diffusion_operator(self, Er_1d):
        """Apply diffusion operator: y = L(Er) * Er (matrix-free)"""
        L = self.assemble_diffusion_operator(Er_1d)
        return L @ Er_1d
    
    def assemble_system(self, Er_k, Er_prev, theta=1.0, current_time=0.0):
        """
        Assemble sparse system for Newton iteration (theta method)
        
        Returns: A (sparse), rhs (vector)
        System: A * Er = rhs
        """
        # Get diffusion operator at current iterate
        L = self.assemble_diffusion_operator(Er_k)
        
        # Initialize system matrix: A = alpha - theta*L
        A = theta * L
        rhs = np.zeros(self.n_total)
        
        # Evaluate du/dEr for linearization
        dudEr = np.zeros(self.n_total)
        for idx in range(self.n_total):
            i, j = index_to_cell(idx, self.n1_cells, self.n2_cells)
            dudEr[idx] = self.get_dudEr(Er_k[idx], self.coord1_centers[i], self.coord2_centers[j])
        alpha = dudEr / self.dt
        
        # Add time discretization to diagonal
        A = A.tolil()  # Convert to lil for efficient diagonal modification
        for idx in range(self.n_total):
            A[idx, idx] += alpha[idx]
        A = A.tocsr()
        
        # Compute RHS
        u_k = np.zeros(self.n_total)
        qhat = np.zeros(self.n_total)
        for idx in range(self.n_total):
            i, j = index_to_cell(idx, self.n1_cells, self.n2_cells)
            u_k[idx] = self.get_u_function(Er_k[idx], self.coord1_centers[i], self.coord2_centers[j])
            qhat[idx] = self.get_u_function(Er_prev[idx], self.coord1_centers[i], self.coord2_centers[j])
        
        # Explicit part if theta < 1
        if theta < 1.0:
            L_prev = self.assemble_diffusion_operator(Er_prev)
            L_prev_times_Er_prev = L_prev @ Er_prev
        else:
            L_prev_times_Er_prev = np.zeros(self.n_total)
        
        for idx in range(self.n_total):
            rhs[idx] = (alpha[idx] * Er_k[idx] - u_k[idx] + qhat[idx] - 
                       (1.0 - theta) * L_prev_times_Er_prev[idx])
        
        return A, rhs
    
    def apply_boundary_conditions(self, A, rhs, Er_k):
        """
        Apply Robin boundary conditions to all four boundaries
        
        A*Er + B*(n·∇Er) = C becomes additional terms in matrix/rhs
        """
        Er_2d = reshape_1d_to_2d(Er_k, self.n1_cells, self.n2_cells)
        
        # Left boundary (i=0)
        i = 0
        for j in range(self.n2_cells):
            idx = cell_to_index(i, j, self.n1_cells, self.n2_cells)
            A_bc, B_bc, C_bc = self.left_bc_func(Er_2d[i, j], 
                                                   self.coord1_faces[0],
                                                   self.coord2_centers[j],
                                                   self.geometry,
                                                   self.current_time)
            
            if abs(B_bc) < 1e-14:  # Dirichlet
                Er_ghost = C_bc / A_bc
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j], 
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                dx_half = self.coord1_centers[i] - self.coord1_faces[i]
                flux_coeff = (self.A1_faces[i, j] * D_boundary) / \
                            (self.V_cells[i, j] * dx_half)
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += flux_coeff * Er_ghost
            else:  # Robin
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                flux_coeff = (self.A1_faces[i, j] * D_boundary * A_bc) / \
                            (B_bc * self.V_cells[i, j])
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += (self.A1_faces[i, j] * D_boundary * C_bc) / \
                           (B_bc * self.V_cells[i, j])
        
        # Right boundary (i=n1_cells-1)
        i = self.n1_cells - 1
        for j in range(self.n2_cells):
            idx = cell_to_index(i, j, self.n1_cells, self.n2_cells)
            A_bc, B_bc, C_bc = self.right_bc_func(Er_2d[i, j],
                                                    self.coord1_faces[-1],
                                                    self.coord2_centers[j],
                                                    self.geometry,
                                                    self.current_time)
            
            if abs(B_bc) < 1e-14:  # Dirichlet
                Er_ghost = C_bc / A_bc
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                dx_half = self.coord1_faces[-1] - self.coord1_centers[i]
                flux_coeff = (self.A1_faces[i+1, j] * D_boundary) / \
                            (self.V_cells[i, j] * dx_half)
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += flux_coeff * Er_ghost
            else:  # Robin
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                flux_coeff = (self.A1_faces[i+1, j] * D_boundary * A_bc) / \
                            (B_bc * self.V_cells[i, j])
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += (self.A1_faces[i+1, j] * D_boundary * C_bc) / \
                           (B_bc * self.V_cells[i, j])
        
        # Bottom boundary (j=0)
        j = 0
        for i in range(self.n1_cells):
            idx = cell_to_index(i, j, self.n1_cells, self.n2_cells)
            A_bc, B_bc, C_bc = self.bottom_bc_func(Er_2d[i, j],
                                                     self.coord1_centers[i],
                                                     self.coord2_faces[0],
                                                     self.geometry,
                                                     self.current_time)
            
            if abs(B_bc) < 1e-14:  # Dirichlet
                Er_ghost = C_bc / A_bc
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                dy_half = self.coord2_centers[j] - self.coord2_faces[j]
                flux_coeff = (self.A2_faces[i, j] * D_boundary) / \
                            (self.V_cells[i, j] * dy_half)
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += flux_coeff * Er_ghost
            else:  # Robin
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                flux_coeff = (self.A2_faces[i, j] * D_boundary * A_bc) / \
                            (B_bc * self.V_cells[i, j])
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += (self.A2_faces[i, j] * D_boundary * C_bc) / \
                           (B_bc * self.V_cells[i, j])
        
        # Top boundary (j=n2_cells-1)
        j = self.n2_cells - 1
        for i in range(self.n1_cells):
            idx = cell_to_index(i, j, self.n1_cells, self.n2_cells)
            A_bc, B_bc, C_bc = self.top_bc_func(Er_2d[i, j],
                                                  self.coord1_centers[i],
                                                  self.coord2_faces[-1],
                                                  self.geometry,
                                                  self.current_time)
            
            if abs(B_bc) < 1e-14:  # Dirichlet
                Er_ghost = C_bc / A_bc
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                dy_half = self.coord2_faces[-1] - self.coord2_centers[j]
                flux_coeff = (self.A2_faces[i, j+1] * D_boundary) / \
                            (self.V_cells[i, j] * dy_half)
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += flux_coeff * Er_ghost
            else:  # Robin
                D_boundary = self.get_diffusion_coefficient(Er_2d[i, j],
                                                             self.coord1_centers[i],
                                                             self.coord2_centers[j])
                flux_coeff = (self.A2_faces[i, j+1] * D_boundary * A_bc) / \
                            (B_bc * self.V_cells[i, j])
                A[idx, idx] = A[idx, idx] + flux_coeff
                rhs[idx] += (self.A2_faces[i, j+1] * D_boundary * C_bc) / \
                           (B_bc * self.V_cells[i, j])
    
    def residual_function(self, Er_k, Er_prev):
        """
        Compute residual F(Er_k) for Newton's method
        
        F(Er_k) = u(Er_k) - u(Er_prev) - ∇·(D(Er_k) ∇Er_k) = 0
        where u = (1/dt)*(e_mat + Er), so u already includes 1/dt factor
        """
        # Time derivative term (u already has 1/dt factor)
        u_k = np.zeros(self.n_total)
        u_prev = np.zeros(self.n_total)
        for idx in range(self.n_total):
            i, j = index_to_cell(idx, self.n1_cells, self.n2_cells)
            u_k[idx] = self.get_u_function(Er_k[idx], self.coord1_centers[i], self.coord2_centers[j])
            u_prev[idx] = self.get_u_function(Er_prev[idx], self.coord1_centers[i], self.coord2_centers[j])
        F = u_k - u_prev
        
        # Diffusion term
        L_Er = self.apply_diffusion_operator(Er_k)
        F -= L_Er
        
        return F
    
    def jfnk_jacobian_vector_product(self, Er_k, Er_prev, v, epsilon=1e-6):
        """
        Compute Jacobian-vector product J(Er_k) * v using finite differences
        
        J*v ≈ [F(Er_k + ε*v) - F(Er_k)] / ε
        """
        F_k = self.residual_function(Er_k, Er_prev)
        F_perturbed = self.residual_function(Er_k + epsilon * v, Er_prev)
        return (F_perturbed - F_k) / epsilon
    
    def newton_step_direct(self, Er_prev, verbose=False):
        """Newton iteration with direct sparse solve"""
        Er_k = self.Er.copy()
        
        for k in range(self.max_newton_iter):
            # Assemble system
            A, rhs = self.assemble_system(Er_k, Er_prev, theta=self.theta, 
                                         current_time=self.current_time)
            
            # Apply boundary conditions
            self.apply_boundary_conditions(A, rhs, Er_k)
            
            # Solve linear system
            Er_new = spsolve(A, rhs)
            
            # Check convergence
            residual = np.linalg.norm(Er_new - Er_k) / max(np.linalg.norm(Er_k), 1e-14)
            
            if verbose:
                print(f"    Newton iter {k+1}: residual = {residual:.2e}")
            
            if residual < self.newton_tol:
                if verbose:
                    print(f"    Newton converged in {k+1} iterations")
                return Er_new
            
            # Damping for negative values
            if np.any(Er_new <= 0):
                alpha = 0.5
                while np.any(Er_k + alpha * (Er_new - Er_k) <= 0) and alpha > 1e-4:
                    alpha *= 0.5
                Er_new = Er_k + alpha * (Er_new - Er_k)
                if verbose:
                    print(f"    Applied damping: alpha = {alpha:.3f}")
            
            Er_k = Er_new.copy()
        
        if verbose:
            print(f"    Newton max iterations reached, residual = {residual:.2e}")
        return Er_k
    
    def newton_step_jfnk(self, Er_prev, verbose=False):
        """Newton iteration with Jacobian-Free Newton-Krylov method"""
        Er_k = self.Er.copy()
        
        for k in range(self.max_newton_iter):
            # Compute residual
            F_k = self.residual_function(Er_k, Er_prev)
            
            # Define linear operator for GMRES
            def matvec(v):
                return self.jfnk_jacobian_vector_product(Er_k, Er_prev, v)
            
            J_op = LinearOperator((self.n_total, self.n_total), matvec=matvec)
            
            # Solve J(Er_k) * delta = -F(Er_k) using GMRES
            delta, info = gmres(J_op, -F_k, rtol=self.gmres_tol, 
                               maxiter=self.gmres_maxiter, atol=0)
            
            if info != 0 and verbose:
                print(f"    GMRES did not converge (info={info})")
            
            # Update
            Er_new = Er_k + delta
            
            # Check convergence
            residual = np.linalg.norm(delta) / max(np.linalg.norm(Er_k), 1e-14)
            
            if verbose:
                print(f"    Newton iter {k+1}: residual = {residual:.2e}, GMRES info = {info}")
            
            if residual < self.newton_tol:
                if verbose:
                    print(f"    Newton converged in {k+1} iterations")
                return Er_new
            
            # Damping for negative values
            if np.any(Er_new <= 0):
                alpha = 0.5
                while np.any(Er_k + alpha * delta <= 0) and alpha > 1e-4:
                    alpha *= 0.5
                Er_new = Er_k + alpha * delta
                if verbose:
                    print(f"    Applied damping: alpha = {alpha:.3f}")
            
            Er_k = Er_new.copy()
        
        if verbose:
            print(f"    Newton max iterations reached, residual = {residual:.2e}")
        return Er_k
    
    def time_step(self, n_steps=1, verbose=True):
        """Advance solution by n_steps time steps"""
        for step in range(n_steps):
            if verbose:
                print(f"Time step {step+1}/{n_steps}, t = {self.current_time:.4e}")
            
            Er_prev = self.Er.copy()
            
            # Choose solver
            if self.use_jfnk:
                self.Er = self.newton_step_jfnk(Er_prev, verbose=verbose)
            else:
                self.Er = self.newton_step_direct(Er_prev, verbose=verbose)
            
            self.Er_old = Er_prev.copy()
            self.current_time += self.dt
    
    def get_solution(self):
        """Return current solution as 2D array"""
        return (self.coord1_centers.copy(), self.coord2_centers.copy(),
                reshape_1d_to_2d(self.Er, self.n1_cells, self.n2_cells))


# =============================================================================
# TEST PROBLEMS
# =============================================================================

def test_2d_cartesian_gaussian():
    """Test 2D Cartesian with Gaussian initial condition"""
    print("="*60)
    print("Test: 2D Cartesian Gaussian Diffusion")
    print("="*60)
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=30,
        coord2_min=0.0, coord2_max=1.0, n2_cells=30,
        geometry='cartesian', dt=1e-3, max_newton_iter=5,
        use_jfnk=False  # Start with direct solver
    )
    
    # Gaussian initial condition
    def initial_Er(x, y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return 0.1 + np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    print("Time stepping...")
    solver.time_step(n_steps=5, verbose=True)
    
    return solver


def test_2d_cylindrical_heat_pulse():
    """Test 2D cylindrical with radial heat pulse"""
    print("="*60)
    print("Test: 2D Cylindrical Heat Pulse")
    print("="*60)
    
    solver = RadiationDiffusionSolver2D(
        coord1_min=0.1, coord1_max=1.0, n1_cells=25,
        coord2_min=0.0, coord2_max=0.5, n2_cells=20,
        geometry='cylindrical', dt=1e-3, max_newton_iter=5,
        use_jfnk=False
    )
    
    # Radial pulse
    def initial_Er(r, z):
        r0 = 0.3
        z0 = 0.25
        sigma_r, sigma_z = 0.1, 0.1
        return 0.1 + 0.5 * np.exp(-((r - r0)**2 / (2 * sigma_r**2) + 
                                    (z - z0)**2 / (2 * sigma_z**2)))
    
    solver.set_initial_condition(initial_Er)
    
    print("Time stepping...")
    solver.time_step(n_steps=5, verbose=True)
    
    return solver


def test_jfnk_performance():
    """Compare direct vs JFNK solvers"""
    print("="*60)
    print("Test: JFNK vs Direct Solver Performance")
    print("="*60)
    
    # Problem setup
    n_cells = 40
    
    # Direct solver
    print("\n--- Direct Solver ---")
    t0 = time.time()
    solver_direct = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=n_cells,
        coord2_min=0.0, coord2_max=1.0, n2_cells=n_cells,
        geometry='cartesian', dt=1e-3, use_jfnk=False
    )
    solver_direct.set_initial_condition(lambda x, y: 0.5 + 0.5*np.sin(np.pi*x)*np.sin(np.pi*y))
    solver_direct.time_step(n_steps=3, verbose=False)
    t_direct = time.time() - t0
    print(f"Direct solver time: {t_direct:.2f} seconds")
    
    # JFNK solver
    print("\n--- JFNK Solver ---")
    t0 = time.time()
    solver_jfnk = RadiationDiffusionSolver2D(
        coord1_min=0.0, coord1_max=1.0, n1_cells=n_cells,
        coord2_min=0.0, coord2_max=1.0, n2_cells=n_cells,
        geometry='cartesian', dt=1e-3, use_jfnk=True,
        gmres_tol=1e-6, gmres_maxiter=100
    )
    solver_jfnk.set_initial_condition(lambda x, y: 0.5 + 0.5*np.sin(np.pi*x)*np.sin(np.pi*y))
    solver_jfnk.time_step(n_steps=3, verbose=False)
    t_jfnk = time.time() - t0
    print(f"JFNK solver time: {t_jfnk:.2f} seconds")
    
    print(f"\nSpeedup: {t_direct/t_jfnk:.2f}x")
    
    return solver_direct, solver_jfnk


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_2d_solution(solver, title="2D Radiation Energy Density", cmap='hot'):
    """Plot 2D solution as contour plot"""
    coord1, coord2, Er_2d = solver.get_solution()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contour plot
    X, Y = np.meshgrid(coord1, coord2, indexing='ij')
    levels = 20
    contour = ax1.contourf(X, Y, Er_2d, levels=levels, cmap=cmap)
    ax1.contour(X, Y, Er_2d, levels=levels, colors='k', linewidths=0.5, alpha=0.3)
    plt.colorbar(contour, ax=ax1, label='$E_r$')
    
    coord_labels = {'cartesian': ('x', 'y'), 'cylindrical': ('r', 'z')}
    labels = coord_labels[solver.geometry]
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title(f'{title} - Contours')
    ax1.set_aspect('equal')
    
    # Surface plot (pcolormesh)
    im = ax2.pcolormesh(X, Y, Er_2d, shading='auto', cmap=cmap)
    plt.colorbar(im, ax=ax2, label='$E_r$')
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    ax2.set_title(f'{title} - Color Map')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def plot_cross_sections(solver, title="Cross Sections"):
    """Plot 1D cross sections through 2D solution"""
    coord1, coord2, Er_2d = solver.get_solution()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    coord_labels = {'cartesian': ('x', 'y'), 'cylindrical': ('r', 'z')}
    labels = coord_labels[solver.geometry]
    
    # Cross section at middle of second coordinate
    j_mid = solver.n2_cells // 2
    ax1.plot(coord1, Er_2d[:, j_mid], 'b-', linewidth=2)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel('$E_r$')
    ax1.set_title(f'Cross section at {labels[1]} = {coord2[j_mid]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Cross section at middle of first coordinate
    i_mid = solver.n1_cells // 2
    ax2.plot(coord2, Er_2d[i_mid, :], 'r-', linewidth=2)
    ax2.set_xlabel(labels[1])
    ax2.set_ylabel('$E_r$')
    ax2.set_title(f'Cross section at {labels[0]} = {coord1[i_mid]:.3f}')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("2D Radiation Diffusion Finite Volume Solver")
    print("Supports Cartesian (x-y) and Cylindrical (r-z) geometries")
    print("Direct solver and Jacobian-Free Newton-Krylov (JFNK) method")
    print("="*70)
    
    # Test 1: Cartesian Gaussian
    print("\n")
    solver1 = test_2d_cartesian_gaussian()
    plot_2d_solution(solver1, "Cartesian Gaussian Diffusion")
    plot_cross_sections(solver1, "Cartesian Gaussian - Cross Sections")
    
    # Test 2: Cylindrical heat pulse
    print("\n")
    solver2 = test_2d_cylindrical_heat_pulse()
    plot_2d_solution(solver2, "Cylindrical Heat Pulse")
    plot_cross_sections(solver2, "Cylindrical Heat Pulse - Cross Sections")
    
    # Test 3: JFNK performance
    print("\n")
    solver_direct, solver_jfnk = test_jfnk_performance()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, solver, name in zip(axes, [solver_direct, solver_jfnk], ['Direct', 'JFNK']):
        coord1, coord2, Er_2d = solver.get_solution()
        X, Y = np.meshgrid(coord1, coord2, indexing='ij')
        im = ax.contourf(X, Y, Er_2d, levels=20, cmap='hot')
        plt.colorbar(im, ax=ax, label='$E_r$')
        ax.set_title(f'{name} Solver')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    plt.suptitle('Direct vs JFNK Comparison')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("All 2D tests completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
