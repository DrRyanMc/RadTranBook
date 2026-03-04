#!/usr/bin/env python3
"""
Interface for Solving Diffusion Equations with Operator A

Solves equations of the form:
    A_g φ = b

where:
    A_g = -∇·D_g(T)∇ + (σ_a + 1/(c·Δt))
    
This provides a general interface for applying the inverse of the diffusion
operator, allowing users to specify:
- Diffusion coefficient D_g(T) (possibly flux-limited)
- Absorption coefficient σ_a
- Boundary conditions
- Right-hand side b

Example usage:
    solver = DiffusionOperatorSolver1D(
        r_min=0.0, r_max=10.0, n_cells=100,
        geometry='spherical',
        diffusion_coeff_func=my_diffusion_func,
        absorption_coeff_func=my_sigma_func,
        dt=0.1
    )
    
    # Solve A φ = b
    phi_solution = solver.solve(rhs=b, temperature=T)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple
from numba import njit

# Physical constants
C_LIGHT = 2.99792458e1  # speed of light (cm/ns)
A_RAD = 0.01372           # radiation constant (GJ/(cm³·keV⁴))


class DiffusionOperatorSolver1D:
    """
    1D Finite Volume solver for diffusion operator equations.
    
    Solves: A φ = b
    where A = -∇·D∇ + σ_total
    and σ_total = σ_a + 1/(c·Δt)
    
    Supports multiple geometries: planar, cylindrical, spherical
    """
    
    def __init__(self,
                 r_min: float,
                 r_max: float,
                 n_cells: int,
                 geometry: str = 'planar',
                 diffusion_coeff_func: Optional[Callable] = None,
                 absorption_coeff_func: Optional[Callable] = None,
                 dt: float = 1.0,
                 left_bc_func: Optional[Callable] = None,
                 right_bc_func: Optional[Callable] = None):
        """
        Initialize the diffusion operator solver.
        
        Parameters:
        -----------
        r_min, r_max : float
            Domain boundaries
        n_cells : int
            Number of cells
        geometry : str
            'planar', 'cylindrical', or 'spherical'
        diffusion_coeff_func : callable or None
            Function D(T, r, phi_left, phi_right, dx) or D(T, r)
            If None, uses constant D = 1.0
        absorption_coeff_func : callable or None
            Function σ_a(T, r) returning absorption coefficient
            If None, uses σ_a = 0.0
        dt : float
            Time step for 1/(c·Δt) term
        left_bc_func, right_bc_func : callable or None
            Boundary condition functions returning (A, B, C) coefficients
            for Robin BC: A·φ + B·∇φ = C
            Function signature: (phi_boundary, r_boundary) -> (A, B, C)
            If None, uses default reflecting BC: (0, 1, 0) for ∇φ = 0
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_cells = n_cells
        self.geometry = geometry.lower()
        self.dt = dt
        
        # Mesh generation
        self.r_faces = np.linspace(r_min, r_max, n_cells + 1)
        self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.dr_cells = np.diff(self.r_faces)
        
        # Geometric factors
        self._compute_geometry()
        
        # Material property functions
        if diffusion_coeff_func is None:
            self.diffusion_coeff_func = lambda T, r, *args: 1.0
        else:
            self.diffusion_coeff_func = diffusion_coeff_func
            
        if absorption_coeff_func is None:
            self.absorption_coeff_func = lambda T, r: 0.0
        else:
            self.absorption_coeff_func = absorption_coeff_func
        
        # Boundary condition functions (return A, B, C for Robin BC: A·φ + B·∇φ = C)
        if left_bc_func is None:
            # Default: reflecting (zero flux)
            self.left_bc_func = lambda phi, r: (0.0, 1.0, 0.0)
        else:
            self.left_bc_func = left_bc_func
            
        if right_bc_func is None:
            # Default: reflecting (zero flux)
            self.right_bc_func = lambda phi, r: (0.0, 1.0, 0.0)
        else:
            self.right_bc_func = right_bc_func
    
    def _compute_geometry(self):
        """Compute geometric factors (areas, volumes) for the mesh."""
        if self.geometry == 'planar':
            # Planar (slab): A = 1, V = Δr
            self.A_faces = np.ones(self.n_cells + 1)
            self.V_cells = self.dr_cells
            
        elif self.geometry == 'cylindrical':
            # Cylindrical: A = 2πr, V = π(r_R² - r_L²)
            self.A_faces = 2.0 * np.pi * self.r_faces
            self.V_cells = np.pi * (self.r_faces[1:]**2 - self.r_faces[:-1]**2)
            
        elif self.geometry == 'spherical':
            # Spherical: A = 4πr², V = (4π/3)(r_R³ - r_L³)
            self.A_faces = 4.0 * np.pi * self.r_faces**2
            self.V_cells = (4.0 * np.pi / 3.0) * (self.r_faces[1:]**3 - self.r_faces[:-1]**3)
            
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")
    
    def assemble_matrix(self, 
                       temperature: np.ndarray,
                       phi_guess: Optional[np.ndarray] = None) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """
        Assemble the operator matrix A and prepare for applying boundary conditions.
        
        A = -∇·D∇ + σ_total
        where σ_total = σ_a + 1/(c·Δt)
        
        Parameters:
        -----------
        temperature : ndarray
            Temperature field T(r) for evaluating D(T) and σ_a(T)
        phi_guess : ndarray or None
            Initial guess for φ (used if diffusion coeff is flux-limited)
            If None, uses zeros
            
        Returns:
        --------
        A : sparse.csr_matrix
            Operator matrix (with BCs not yet applied)
        D_faces : ndarray
            Diffusion coefficients at faces (needed for BC application)
        diag_contribution : ndarray
            Diagonal contributions before BC modification (for reference)
        """
        n = self.n_cells
        
        if phi_guess is None:
            phi_guess = np.zeros(n)
        
        # Initialize tridiagonal matrix storage
        sub_diag = np.zeros(n - 1)
        main_diag = np.zeros(n)
        super_diag = np.zeros(n - 1)
        
        # Evaluate diffusion coefficients at faces
        D_faces = np.zeros(n + 1)
        for i in range(1, n):  # Interior faces
            T_face = 0.5 * (temperature[i-1] + temperature[i])
            r_face = self.r_faces[i]
            
            # Check if diffusion coefficient function takes flux-limiting arguments
            try:
                # Try with flux-limiting interface
                phi_left = phi_guess[i-1]
                phi_right = phi_guess[i]
                dx = self.r_centers[i] - self.r_centers[i-1]
                D_faces[i] = self.diffusion_coeff_func(T_face, r_face, phi_left, phi_right, dx)
            except TypeError:
                # Fall back to simple interface
                D_faces[i] = self.diffusion_coeff_func(T_face, r_face)
        
        # Boundary faces - also handle flux-limiting interface
        # Left boundary (face 0)
        try:
            phi_left = phi_guess[0]
            phi_right = phi_guess[0]  # Use same value for boundary
            dx = self.dr_cells[0]
            D_faces[0] = self.diffusion_coeff_func(temperature[0], self.r_faces[0], phi_left, phi_right, dx)
        except TypeError:
            D_faces[0] = self.diffusion_coeff_func(temperature[0], self.r_faces[0])
        
        # Right boundary (face -1)
        try:
            phi_left = phi_guess[-1]
            phi_right = phi_guess[-1]  # Use same value for boundary
            dx = self.dr_cells[-1]
            D_faces[-1] = self.diffusion_coeff_func(temperature[-1], self.r_faces[-1], phi_left, phi_right, dx)
        except TypeError:
            D_faces[-1] = self.diffusion_coeff_func(temperature[-1], self.r_faces[-1])

        
        # Evaluate absorption coefficients
        sigma_a = np.array([self.absorption_coeff_func(temperature[i], self.r_centers[i]) 
                           for i in range(n)])
        
        # Total absorption: σ_total = σ_a + 1/(c·Δt)
        sigma_total = sigma_a + 1.0 / (C_LIGHT * self.dt)
        
        # Assemble diffusion operator: -∇·D∇
        # Discretization: -A_L·D_L·(φ_i - φ_{i-1})/Δr_L + A_R·D_R·(φ_{i+1} - φ_i)/Δr_R
        # This gives tridiagonal structure
        for i in range(n):
            V_i = self.V_cells[i]
            
            # Left face contribution
            if i > 0:
                A_left = self.A_faces[i]
                dr_left = self.r_centers[i] - self.r_centers[i-1]
                coeff = A_left * D_faces[i] / (dr_left * V_i)
                main_diag[i] += coeff
                sub_diag[i-1] = -coeff
            
            # Right face contribution
            if i < n - 1:
                A_right = self.A_faces[i+1]
                dr_right = self.r_centers[i+1] - self.r_centers[i]
                coeff = A_right * D_faces[i+1] / (dr_right * V_i)
                main_diag[i] += coeff
                super_diag[i] = -coeff
            
            # Absorption term: +σ_total on diagonal
            main_diag[i] += sigma_total[i]
        
        # Store diagonal for reference
        diag_contribution = main_diag.copy()
        
        # Create sparse matrix (before applying BCs)
        A = sparse.diags([sub_diag, main_diag, super_diag], 
                        offsets=[-1, 0, 1], 
                        shape=(n, n), 
                        format='lil')
        
        return A.tocsr(), D_faces, diag_contribution
    
    def apply_boundary_conditions(self, 
                                 A: sparse.lil_matrix, 
                                 rhs: np.ndarray,
                                 phi: np.ndarray,
                                 temperature: np.ndarray,
                                 D_faces: np.ndarray):
        """
        Apply Robin boundary conditions: A·φ + B·∇φ = C
        
        Modifies A and rhs in place to enforce boundary conditions.
        
        Parameters:
        -----------
        A : sparse matrix
            Operator matrix (will be modified)
        rhs : ndarray
            Right-hand side vector (will be modified)
        phi : ndarray
            Current φ values (for evaluating BC functions)
        temperature : ndarray
            Temperature field (for evaluating diffusion coefficient at boundary)
        D_faces : ndarray
            Diffusion coefficients at faces (computed from temperature)
        """
        n = self.n_cells
        A = A.tolil()  # Convert to lil format for efficient modification
        
        # =====================================================================
        # LEFT BOUNDARY (i=0)
        # =====================================================================
        A_bc, B_bc, C_bc = self.left_bc_func(phi[0], self.r_faces[0])
        
        if abs(B_bc) < 1e-14:
            # Dirichlet-type BC: A·φ = C → φ_boundary = C/A
            phi_boundary = C_bc / A_bc
            
            # Compute averaged phi and corresponding temperature for D_boundary
            # This is important when temperature varies rapidly at boundary
            phi_avg = 0.5 * (phi[0] + phi_boundary)
            # For radiation diffusion: T = (φ/(a·c))^{1/4}
            T_avg = (phi_avg / (A_RAD * C_LIGHT)) ** 0.25 if phi_avg > 0 else temperature[0]
            
            # Evaluate diffusion coefficient at boundary using averaged temperature
            try:
                D_boundary = self.diffusion_coeff_func(T_avg, self.r_faces[0])
            except:
                D_boundary = D_faces[0]  # Fallback to pre-computed value
            
            # Ghost cell flux discretization: flux = D * (φ_interior - φ_boundary) / dx
            dx_half = self.r_centers[0] - self.r_faces[0]  
            flux_coeff = (self.A_faces[0] * D_boundary) / (self.V_cells[0] * dx_half)
            
            A[0, 0] += flux_coeff
            rhs[0] += flux_coeff * phi_boundary
            
        else:
            # Robin/Neumann BC: A·φ + B·(∂φ/∂n) = C
            # For Marshak BC: A=1/2, B=1/(3σ_BC), C=0.5*B_g(T_b)
            # 
            # CRITICAL: For Robin BC, the diffusion coefficient should come from the BC parameter B,
            # not from the interior material! B = 1/(3σ_BC) already encodes D_BC = c/(3σ_BC) = c*B
            # 
            # Using D_faces[0] (from cold interior) would make D tiny and prevent radiation entry!
            # Instead, extract D from the BC: D = c·B (since B = 1/(3σ) and D = c/(3σ))
            D_boundary = C_LIGHT * B_bc  # Extract D from BC parameter
            
            flux_coeff = (self.A_faces[0] * D_boundary * A_bc) / (B_bc * self.V_cells[0])
            rhs_contribution = (self.A_faces[0] * D_boundary * C_bc) / (B_bc * self.V_cells[0])
            
            A[0, 0] += flux_coeff
            rhs[0] += rhs_contribution
        
        # =====================================================================
        # RIGHT BOUNDARY (i=n-1)
        # =====================================================================
        A_bc, B_bc, C_bc = self.right_bc_func(phi[-1], self.r_faces[-1])
        
        if abs(B_bc) < 1e-14:
            # Dirichlet-type BC
            phi_boundary = C_bc / A_bc
            phi_avg = 0.5 * (phi[-1] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT)) ** 0.25 if phi_avg > 0 else temperature[-1]
            
            try:
                D_boundary = self.diffusion_coeff_func(T_avg, self.r_faces[-1])
            except:
                D_boundary = D_faces[-1]
            
            dx_half = self.r_faces[-1] - self.r_centers[-1]
            flux_coeff = (self.A_faces[-1] * D_boundary) / (self.V_cells[-1] * dx_half)
            
            A[-1, -1] += flux_coeff
            rhs[-1] += flux_coeff * phi_boundary
            
        else:
            # Robin/Neumann BC: A·φ + B·(∂φ/∂n) = C
            # Extract D from BC parameter to avoid using interior material properties
            D_boundary = C_LIGHT * B_bc
            flux_coeff = (self.A_faces[-1] * D_boundary * A_bc) / (B_bc * self.V_cells[-1])
            rhs_contribution = (self.A_faces[-1] * D_boundary * C_bc) / (B_bc * self.V_cells[-1])
            
            A[-1, -1] += flux_coeff
            rhs[-1] += rhs_contribution
        
        return A.tocsr()
    
    def solve(self, 
             rhs: np.ndarray,
             temperature: np.ndarray,
             phi_guess: Optional[np.ndarray] = None,
             use_iterative: bool = False,
             max_iter: int = 10,
             tol: float = 1e-6,
             override_left_bc: Optional[Callable] = None,
             override_right_bc: Optional[Callable] = None) -> np.ndarray:
        """
        Solve A φ = b for φ.
        
        Parameters:
        -----------
        rhs : ndarray
            Right-hand side vector b
        temperature : ndarray
            Temperature field T(r) for evaluating D(T) and σ_a(T)
        phi_guess : ndarray or None
            Initial guess for φ (used if diffusion is flux-limited)
        use_iterative : bool
            If True and diffusion is flux-limited, iterate to convergence
            If False, uses phi_guess for flux limiting (default True)
        max_iter : int
            Maximum iterations for flux limiting
        tol : float
            Tolerance for flux limiting iteration
        override_left_bc : callable or None
            Temporarily override left boundary condition for this solve
        override_right_bc : callable or None
            Temporarily override right boundary condition for this solve
            
        Returns:
        --------
        phi : ndarray
            Solution to A φ = b
        """
        if phi_guess is None:
            phi_guess = np.zeros(self.n_cells)
        
        # Temporarily override boundary conditions if requested
        saved_left_bc = self.left_bc_func
        saved_right_bc = self.right_bc_func
        if override_left_bc is not None:
            self.left_bc_func = override_left_bc
        if override_right_bc is not None:
            self.right_bc_func = override_right_bc
        
        # Check if we need to iterate (for flux-limited diffusion)
        need_iteration = use_iterative
        
        phi = phi_guess.copy()
        for iteration in range(max_iter):
            # Assemble matrix
            A, D_faces, _ = self.assemble_matrix(temperature, phi)
            
            # Apply boundary conditions
            rhs_bc = rhs.copy()
            A = self.apply_boundary_conditions(A, rhs_bc, phi, temperature, D_faces)
            
            # Solve linear system
            phi_new = spsolve(A, rhs_bc)
            
            # Check convergence
            if need_iteration:
                rel_change = np.linalg.norm(phi_new - phi) / (np.linalg.norm(phi_new) + 1e-14)
                if rel_change < tol:
                    # Restore original BCs
                    self.left_bc_func = saved_left_bc
                    self.right_bc_func = saved_right_bc
                    return phi_new
                phi = phi_new
            else:
                # Restore original BCs
                self.left_bc_func = saved_left_bc
                self.right_bc_func = saved_right_bc
                return phi_new
        
        # Restore original BCs (if max iterations reached)
        self.left_bc_func = saved_left_bc
        self.right_bc_func = saved_right_bc
        
        if use_iterative:
            print(f"Warning: Flux limiting iteration did not converge after {max_iter} iterations")
        
        return phi
    
    def apply_operator(self,
                      phi: np.ndarray,
                      temperature: np.ndarray) -> np.ndarray:
        """
        Apply operator A to a given φ to compute A φ.
        
        Useful for verifying solutions or computing residuals.
        
        Parameters:
        -----------
        phi : ndarray
            Input field
        temperature : ndarray
            Temperature field for evaluating D(T) and σ_a(T)
            
        Returns:
        --------
        result : ndarray
            A φ
        """
        A, D_faces, _ = self.assemble_matrix(temperature, phi)
        
        # For applying operator, we don't modify BCs in the same way
        # We just compute the matrix-vector product
        result = A.dot(phi)
        
        return result


# =============================================================================
# EXAMPLE DIFFUSION AND ABSORPTION COEFFICIENT FUNCTIONS
# =============================================================================

def constant_diffusion(T, r, *args):
    """Constant diffusion coefficient D = 1.0"""
    return 1.0


def temperature_dependent_diffusion(T, r, *args, power=3):
    """Temperature-dependent diffusion: D = T^power"""
    return T**power


def rosseland_diffusion(T, r, *args):
    """Rosseland diffusion: D = 1/(3σ_R(T))"""
    sigma_R = 1.0 / (1.0 + T)  # Example opacity
    return 1.0 / (3.0 * sigma_R)


def flux_limited_diffusion(T, r, phi_left, phi_right, dx, lambda_func=None):
    """
    Flux-limited diffusion coefficient.
    
    D = λ(R) / σ_R
    where R = |∇φ| / (σ_R · φ) is the flux ratio
    and λ(R) is the flux limiter
    
    Parameters:
    -----------
    lambda_func : callable or None
        Flux limiter function λ(R). If None, uses Larsen limiter
    """
    # Rosseland opacity (example)
    sigma_R = 1.0 / (1.0 + T)
    
    # Compute flux ratio R
    phi_avg = 0.5 * (phi_left + phi_right) + 1e-14
    grad_phi = abs(phi_right - phi_left) / dx
    R = grad_phi / (sigma_R * phi_avg + 1e-14)
    
    # Apply flux limiter
    if lambda_func is None:
        # Larsen limiter with n=2
        lambda_val = (2.0 + R) / (6.0 + 3.0*R + R**2)
    else:
        lambda_val = lambda_func(R)
    
    return lambda_val / sigma_R


def constant_absorption(T, r):
    """Constant absorption coefficient"""
    return 1.0


def temperature_dependent_absorption(T, r, power=4):
    """Temperature-dependent absorption: σ_a = T^power"""
    return T**power


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DIFFUSION OPERATOR SOLVER - Example Usage")
    print("="*70)
    
    # Example 1: Simple diffusion with constant coefficients
    print("\nExample 1: Constant coefficients with source term")
    print("-" * 70)
    
    solver = DiffusionOperatorSolver1D(
        r_min=0.0,
        r_max=1.0,
        n_cells=50,
        geometry='planar',
        diffusion_coeff_func=lambda T, r, *args: 1.0,
        absorption_coeff_func=lambda T, r: 1.0,
        dt=1.0,
        left_bc='dirichlet',
        right_bc='dirichlet',
        left_bc_value=0.0,
        right_bc_value=0.0
    )
    
    # Define a source term (right-hand side)
    # For example: b = sin(πx)
    r = solver.r_centers
    rhs = np.sin(np.pi * r)
    temperature = np.ones_like(r)  # Constant temperature
    
    # Solve A φ = b
    phi_solution = solver.solve(rhs, temperature)
    
    print(f"Solved for {solver.n_cells} cells")
    print(f"Max φ = {phi_solution.max():.6f}")
    print(f"Min φ = {phi_solution.min():.6f}")
    
    # Example 2: Temperature-dependent diffusion
    print("\nExample 2: Temperature-dependent diffusion")
    print("-" * 70)
    
    solver2 = DiffusionOperatorSolver1D(
        r_min=0.0,
        r_max=1.0,
        n_cells=50,
        geometry='spherical',
        diffusion_coeff_func=lambda T, r, *args: T**3,
        absorption_coeff_func=lambda T, r: 0.1 * T**4,
        dt=0.1,
        left_bc='neumann',
        right_bc='dirichlet',
        left_bc_value=0.0,
        right_bc_value=1.0
    )
    
    # Temperature profile
    temperature2 = 1.0 + 0.5 * np.sin(2.0 * np.pi * solver2.r_centers)
    
    # Source term
    rhs2 = np.ones(solver2.n_cells)
    
    # Solve
    phi_solution2 = solver2.solve(rhs2, temperature2)
    
    print(f"Solved for {solver2.n_cells} cells (spherical geometry)")
    print(f"Max φ = {phi_solution2.max():.6f}")
    print(f"Min φ = {phi_solution2.min():.6f}")
    
    # Verify: compute residual A φ - b
    residual = solver2.apply_operator(phi_solution2, temperature2) - rhs2
    print(f"Max residual = {np.abs(residual).max():.2e}")
    
    print("\n" + "="*70)
    print("1D Examples completed successfully!")
    print("="*70)




# =============================================================================
# 2D DIFFUSION OPERATOR SOLVER
# =============================================================================

class DiffusionOperatorSolver2D:
    """
    2D Finite Volume solver for diffusion operator equations.
    
    Solves: A φ = b
    where A = -∇·D∇ + σ_total
    and σ_total = σ_a + 1/(c·Δt)
    
    Supports two geometries: Cartesian (x-y) and cylindrical (r-z)
    """
    
    def __init__(self,
                 x_min: float,
                 x_max: float,
                 nx_cells: int,
                 y_min: float,
                 y_max: float,
                 ny_cells: int,
                 geometry: str = 'cartesian',
                 diffusion_coeff_func: Optional[Callable] = None,
                 absorption_coeff_func: Optional[Callable] = None,
                 dt: float = 1.0,
                 left_bc: str = 'dirichlet',
                 right_bc: str = 'dirichlet',
                 bottom_bc: str = 'dirichlet',
                 top_bc: str = 'dirichlet',
                 left_bc_value: float = 0.0,
                 right_bc_value: float = 0.0,
                 bottom_bc_value: float = 0.0,
                 top_bc_value: float = 0.0):
        """
        Initialize the 2D diffusion operator solver.
        
        Parameters:
        -----------
        x_min, x_max : float
            Domain boundaries in x-direction (or r for cylindrical)
        nx_cells : int
            Number of cells in x-direction
        y_min, y_max : float
            Domain boundaries in y-direction (or z for cylindrical)
        ny_cells : int
            Number of cells in y-direction
        geometry : str
            'cartesian' for (x,y) or 'cylindrical' for (r,z)
        diffusion_coeff_func : callable or None
            Function D(T, x, y) or D(T, x, y, phi, grad_phi)
            If None, uses constant D = 1.0
        absorption_coeff_func : callable or None
            Function σ_a(T, x, y) returning absorption coefficient
            If None, uses σ_a = 0.0
        dt : float
            Time step for 1/(c·Δt) term
        left_bc, right_bc, bottom_bc, top_bc : str
            Boundary condition types: 'dirichlet' or 'neumann'
        left_bc_value, right_bc_value, bottom_bc_value, top_bc_value : float
            Boundary condition values
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.nx_cells = nx_cells
        self.ny_cells = ny_cells
        self.n_total = nx_cells * ny_cells
        self.geometry = geometry.lower()
        self.dt = dt
        
        # Mesh generation
        self.x_faces = np.linspace(x_min, x_max, nx_cells + 1)
        self.y_faces = np.linspace(y_min, y_max, ny_cells + 1)
        self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
        self.y_centers = 0.5 * (self.y_faces[:-1] + self.y_faces[1:])
        self.dx_cells = np.diff(self.x_faces)
        self.dy_cells = np.diff(self.y_faces)
        
        # Create 2D meshgrids
        self.X_centers, self.Y_centers = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Geometric factors
        self._compute_geometry()
        
        # Material property functions
        if diffusion_coeff_func is None:
            self.diffusion_coeff_func = lambda T, x, y, *args: 1.0
        else:
            self.diffusion_coeff_func = diffusion_coeff_func
            
        if absorption_coeff_func is None:
            self.absorption_coeff_func = lambda T, x, y: 0.0
        else:
            self.absorption_coeff_func = absorption_coeff_func
        
        # Boundary conditions
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.top_bc = top_bc
        self.left_bc_value = left_bc_value
        self.right_bc_value = right_bc_value
        self.bottom_bc_value = bottom_bc_value
        self.top_bc_value = top_bc_value
    
    def _compute_geometry(self):
        """Compute geometric factors (areas, volumes) for the mesh."""
        nx = self.nx_cells
        ny = self.ny_cells
        
        # Face areas for x-direction faces (nx+1, ny)
        self.Ax_faces = np.zeros((nx + 1, ny))
        # Face areas for y-direction faces (nx, ny+1)
        self.Ay_faces = np.zeros((nx, ny + 1))
        # Cell volumes (nx, ny)
        self.V_cells = np.zeros((nx, ny))
        
        if self.geometry == 'cartesian':
            # Cartesian: A_x = Δy, A_y = Δx, V = Δx·Δy
            for i in range(nx + 1):
                for j in range(ny):
                    self.Ax_faces[i, j] = self.dy_cells[j]
            
            for i in range(nx):
                for j in range(ny + 1):
                    self.Ay_faces[i, j] = self.dx_cells[i]
            
            for i in range(nx):
                for j in range(ny):
                    self.V_cells[i, j] = self.dx_cells[i] * self.dy_cells[j]
        
        elif self.geometry == 'cylindrical':
            # Cylindrical (r-z): A_r = 2πr·Δz, A_z = π(r_R² - r_L²), V = π(r_R² - r_L²)·Δz
            for i in range(nx + 1):
                r = self.x_faces[i]
                for j in range(ny):
                    self.Ax_faces[i, j] = 2.0 * np.pi * r * self.dy_cells[j]
            
            for i in range(nx):
                r_left = self.x_faces[i]
                r_right = self.x_faces[i + 1]
                for j in range(ny + 1):
                    self.Ay_faces[i, j] = np.pi * (r_right**2 - r_left**2)
            
            for i in range(nx):
                r_left = self.x_faces[i]
                r_right = self.x_faces[i + 1]
                for j in range(ny):
                    self.V_cells[i, j] = np.pi * (r_right**2 - r_left**2) * self.dy_cells[j]
        
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")
    
    def _index_2d_to_1d(self, i: int, j: int) -> int:
        """Convert 2D indices (i,j) to 1D index for flattened array."""
        return i * self.ny_cells + j
    
    def _index_1d_to_2d(self, idx: int) -> Tuple[int, int]:
        """Convert 1D index to 2D indices (i,j)."""
        i = idx // self.ny_cells
        j = idx % self.ny_cells
        return i, j
    
    def assemble_matrix(self,
                       temperature: np.ndarray,
                       phi_guess: Optional[np.ndarray] = None) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemble the operator matrix A.
        
        A = -∇·D∇ + σ_total
        where σ_total = σ_a + 1/(c·Δt)
        
        Parameters:
        -----------
        temperature : ndarray
            Temperature field T(x,y) - shape (nx, ny) or (n_total,)
        phi_guess : ndarray or None
            Initial guess for φ (used if diffusion coeff is flux-limited)
            Shape (nx, ny) or (n_total,). If None, uses zeros
            
        Returns:
        --------
        A : sparse.csr_matrix
            Operator matrix (n_total, n_total)
        diag_contribution : ndarray
            Diagonal contributions before BC modification
        """
        nx = self.nx_cells
        ny = self.ny_cells
        n_total = self.n_total
        
        # Reshape inputs to 2D if needed
        if temperature.shape == (n_total,):
            T_2d = temperature.reshape((nx, ny), order='C')
        else:
            T_2d = temperature
        
        if phi_guess is None:
            phi_2d = np.zeros((nx, ny))
        elif phi_guess.shape == (n_total,):
            phi_2d = phi_guess.reshape((nx, ny), order='C')
        else:
            phi_2d = phi_guess
        
        # Evaluate diffusion coefficients at faces
        D_x_faces = np.zeros((nx + 1, ny))
        D_y_faces = np.zeros((nx, ny + 1))
        
        # X-direction faces
        for i in range(nx + 1):
            for j in range(ny):
                if i == 0:
                    # Left boundary - use left cell
                    T_face = T_2d[0, j]
                    x_face = self.x_faces[0]
                    y_face = self.y_centers[j]
                elif i == nx:
                    # Right boundary - use right cell
                    T_face = T_2d[nx-1, j]
                    x_face = self.x_faces[-1]
                    y_face = self.y_centers[j]
                else:
                    # Interior face - average
                    T_face = 0.5 * (T_2d[i-1, j] + T_2d[i, j])
                    x_face = self.x_faces[i]
                    y_face = self.y_centers[j]
                
                D_x_faces[i, j] = self.diffusion_coeff_func(T_face, x_face, y_face)
        
        # Y-direction faces
        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    # Bottom boundary - use bottom cell
                    T_face = T_2d[i, 0]
                    x_face = self.x_centers[i]
                    y_face = self.y_faces[0]
                elif j == ny:
                    # Top boundary - use top cell
                    T_face = T_2d[i, ny-1]
                    x_face = self.x_centers[i]
                    y_face = self.y_faces[-1]
                else:
                    # Interior face - average
                    T_face = 0.5 * (T_2d[i, j-1] + T_2d[i, j])
                    x_face = self.x_centers[i]
                    y_face = self.y_faces[j]
                
                D_y_faces[i, j] = self.diffusion_coeff_func(T_face, x_face, y_face)
        
        # Evaluate absorption coefficients
        sigma_a = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                sigma_a[i, j] = self.absorption_coeff_func(T_2d[i, j], 
                                                           self.x_centers[i], 
                                                           self.y_centers[j])
        
        # Total absorption: σ_total = σ_a + 1/(c·Δt)
        sigma_total = sigma_a + 1.0 / (C_LIGHT * self.dt)
        
        # Assemble matrix in COO format
        rows = []
        cols = []
        data = []
        
        for i in range(nx):
            for j in range(ny):
                idx = self._index_2d_to_1d(i, j)
                V = self.V_cells[i, j]
                diag_val = 0.0
                
                # Left face (i-1/2, j)
                if i > 0:
                    A_left = self.Ax_faces[i, j]
                    dx = self.x_centers[i] - self.x_centers[i-1]
                    D_left = D_x_faces[i, j]
                    coeff = A_left * D_left / (dx * V)
                    
                    diag_val += coeff
                    idx_left = self._index_2d_to_1d(i-1, j)
                    rows.append(idx)
                    cols.append(idx_left)
                    data.append(-coeff)
                
                # Right face (i+1/2, j)
                if i < nx - 1:
                    A_right = self.Ax_faces[i+1, j]
                    dx = self.x_centers[i+1] - self.x_centers[i]
                    D_right = D_x_faces[i+1, j]
                    coeff = A_right * D_right / (dx * V)
                    
                    diag_val += coeff
                    idx_right = self._index_2d_to_1d(i+1, j)
                    rows.append(idx)
                    cols.append(idx_right)
                    data.append(-coeff)
                
                # Bottom face (i, j-1/2)
                if j > 0:
                    A_bottom = self.Ay_faces[i, j]
                    dy = self.y_centers[j] - self.y_centers[j-1]
                    D_bottom = D_y_faces[i, j]
                    coeff = A_bottom * D_bottom / (dy * V)
                    
                    diag_val += coeff
                    idx_bottom = self._index_2d_to_1d(i, j-1)
                    rows.append(idx)
                    cols.append(idx_bottom)
                    data.append(-coeff)
                
                # Top face (i, j+1/2)
                if j < ny - 1:
                    A_top = self.Ay_faces[i, j+1]
                    dy = self.y_centers[j+1] - self.y_centers[j]
                    D_top = D_y_faces[i, j+1]
                    coeff = A_top * D_top / (dy * V)
                    
                    diag_val += coeff
                    idx_top = self._index_2d_to_1d(i, j+1)
                    rows.append(idx)
                    cols.append(idx_top)
                    data.append(-coeff)
                
                # Absorption term on diagonal
                diag_val += sigma_total[i, j]
                
                # Add diagonal entry
                rows.append(idx)
                cols.append(idx)
                data.append(diag_val)
        
        # Create sparse matrix
        A_coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
        A_csr = A_coo.tocsr()
        
        # Store diagonal for reference
        diag_contribution = A_csr.diagonal()
        
        return A_csr, diag_contribution
    
    def apply_boundary_conditions(self, A: sparse.csr_matrix, rhs: np.ndarray):
        """
        Apply boundary conditions to the matrix and RHS.
        
        Modifies A and rhs in place.
        
        Parameters:
        -----------
        A : sparse matrix
            Operator matrix
        rhs : ndarray
            Right-hand side vector (n_total,)
        """
        nx = self.nx_cells
        ny = self.ny_cells
        
        A = A.tolil()  # Convert to lil format for efficient modification
        
        # Left boundary (i=0)
        for j in range(ny):
            idx = self._index_2d_to_1d(0, j)
            
            if self.left_bc == 'dirichlet':
                # φ_0,j = left_bc_value
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                rhs[idx] = self.left_bc_value
            
            elif self.left_bc == 'neumann':
                # ∂φ/∂x|_0,j = left_bc_value
                # Already handled by natural BC, but can add flux contribution to RHS
                pass
        
        # Right boundary (i=nx-1)
        for j in range(ny):
            idx = self._index_2d_to_1d(nx-1, j)
            
            if self.right_bc == 'dirichlet':
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                rhs[idx] = self.right_bc_value
            
            elif self.right_bc == 'neumann':
                pass
        
        # Bottom boundary (j=0)
        for i in range(nx):
            idx = self._index_2d_to_1d(i, 0)
            
            if self.bottom_bc == 'dirichlet':
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                rhs[idx] = self.bottom_bc_value
            
            elif self.bottom_bc == 'neumann':
                pass
        
        # Top boundary (j=ny-1)
        for i in range(nx):
            idx = self._index_2d_to_1d(i, ny-1)
            
            if self.top_bc == 'dirichlet':
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                rhs[idx] = self.top_bc_value
            
            elif self.top_bc == 'neumann':
                pass
        
        return A.tocsr()
    
    def solve(self,
             rhs: np.ndarray,
             temperature: np.ndarray,
             phi_guess: Optional[np.ndarray] = None,
             use_iterative: bool = False,
             max_iter: int = 10,
             tol: float = 1e-6) -> np.ndarray:
        """
        Solve A φ = b for φ.
        
        Parameters:
        -----------
        rhs : ndarray
            Right-hand side vector b - shape (nx, ny) or (n_total,)
        temperature : ndarray
            Temperature field T(x,y) - shape (nx, ny) or (n_total,)
        phi_guess : ndarray or None
            Initial guess for φ (used if diffusion is flux-limited)
            Shape (nx, ny) or (n_total,)
        use_iterative : bool
            If True and diffusion is flux-limited, iterate to convergence
        max_iter : int
            Maximum iterations for flux limiting
        tol : float
            Tolerance for flux limiting iteration
            
        Returns:
        --------
        phi : ndarray
            Solution to A φ = b - shape (nx, ny)
        """
        nx = self.nx_cells
        ny = self.ny_cells
        n_total = self.n_total
        
        # Flatten inputs if needed
        if rhs.shape == (nx, ny):
            rhs_1d = rhs.flatten(order='C')
        else:
            rhs_1d = rhs
        
        if phi_guess is None:
            phi_1d = np.zeros(n_total)
        elif phi_guess.shape == (nx, ny):
            phi_1d = phi_guess.flatten(order='C')
        else:
            phi_1d = phi_guess
        
        # Iterative solution for flux-limited diffusion
        for iteration in range(max_iter):
            # Assemble matrix
            A, _ = self.assemble_matrix(temperature, phi_1d)
            
            # Apply boundary conditions
            rhs_bc = rhs_1d.copy()
            A = self.apply_boundary_conditions(A, rhs_bc)
            
            # Solve linear system
            phi_new = spsolve(A, rhs_bc)
            
            # Check convergence
            if use_iterative:
                rel_change = np.linalg.norm(phi_new - phi_1d) / (np.linalg.norm(phi_new) + 1e-14)
                if rel_change < tol:
                    return phi_new.reshape((nx, ny), order='C')
                phi_1d = phi_new
            else:
                return phi_new.reshape((nx, ny), order='C')
        
        if use_iterative:
            print(f"Warning: Flux limiting iteration did not converge after {max_iter} iterations")
        
        return phi_1d.reshape((nx, ny), order='C')
    
    def apply_operator(self,
                      phi: np.ndarray,
                      temperature: np.ndarray) -> np.ndarray:
        """
        Apply operator A to a given φ to compute A φ.
        
        Useful for verifying solutions or computing residuals.
        
        Parameters:
        -----------
        phi : ndarray
            Input field - shape (nx, ny) or (n_total,)
        temperature : ndarray
            Temperature field - shape (nx, ny) or (n_total,)
            
        Returns:
        --------
        result : ndarray
            A φ - shape (nx, ny)
        """
        nx = self.nx_cells
        ny = self.ny_cells
        
        # Flatten if needed
        if phi.shape == (nx, ny):
            phi_1d = phi.flatten(order='C')
        else:
            phi_1d = phi
        
        A, _ = self.assemble_matrix(temperature, phi_1d)
        result_1d = A.dot(phi_1d)
        
        return result_1d.reshape((nx, ny), order='C')


# =============================================================================
# EXAMPLE USAGE FOR 2D SOLVER
# =============================================================================

# 2D examples are part of the main execution block above
def run_2d_examples():
    """Run 2D solver examples"""
    print("\n" + "="*70)
    print("2D DIFFUSION OPERATOR SOLVER - Example Usage")
    print("="*70)
    
    # Example 1: 2D Cartesian with constant coefficients
    print("\nExample 1: 2D Cartesian - Constant coefficients")
    print("-" * 70)
    
    solver2d = DiffusionOperatorSolver2D(
        x_min=0.0, x_max=1.0, nx_cells=30,
        y_min=0.0, y_max=1.0, ny_cells=30,
        geometry='cartesian',
        diffusion_coeff_func=lambda T, x, y, *args: 1.0,
        absorption_coeff_func=lambda T, x, y: 1.0,
        dt=1.0,
        left_bc='dirichlet', right_bc='dirichlet',
        bottom_bc='dirichlet', top_bc='dirichlet',
        left_bc_value=0.0, right_bc_value=0.0,
        bottom_bc_value=0.0, top_bc_value=0.0
    )
    
    # Create a 2D source term
    X, Y = solver2d.X_centers, solver2d.Y_centers
    rhs_2d = np.sin(np.pi * X) * np.sin(np.pi * Y)
    temperature_2d = np.ones_like(X)
    
    # Solve
    phi_solution_2d = solver2d.solve(rhs_2d, temperature_2d)
    
    print(f"Solved for {solver2d.nx_cells} × {solver2d.ny_cells} = {solver2d.n_total} cells")
    print(f"Max φ = {phi_solution_2d.max():.6f}")
    print(f"Min φ = {phi_solution_2d.min():.6f}")
    
    # Example 2: 2D Cylindrical with temperature-dependent diffusion
    print("\nExample 2: 2D Cylindrical - Temperature-dependent diffusion")
    print("-" * 70)
    
    solver2d_cyl = DiffusionOperatorSolver2D(
        x_min=0.1, x_max=1.0, nx_cells=20,  # r direction
        y_min=0.0, y_max=2.0, ny_cells=30,  # z direction
        geometry='cylindrical',
        diffusion_coeff_func=lambda T, r, z, *args: T**3,
        absorption_coeff_func=lambda T, r, z: 0.1 * T**4,
        dt=0.1,
        left_bc='neumann', right_bc='dirichlet',
        bottom_bc='dirichlet', top_bc='neumann',
        right_bc_value=1.0, bottom_bc_value=0.5
    )
    
    # Temperature profile
    R, Z = solver2d_cyl.X_centers, solver2d_cyl.Y_centers
    temperature_cyl = 1.0 + 0.3 * np.sin(np.pi * R) * np.cos(np.pi * Z / 2.0)
    
    # Source term
    rhs_cyl = np.ones_like(R) * 0.5
    
    # Solve
    phi_solution_cyl = solver2d_cyl.solve(rhs_cyl, temperature_cyl)
    
    print(f"Solved for {solver2d_cyl.nx_cells} × {solver2d_cyl.ny_cells} cells (cylindrical)")
    print(f"Max φ = {phi_solution_cyl.max():.6f}")
    print(f"Min φ = {phi_solution_cyl.min():.6f}")
    
    # Verify: compute residual
    residual_cyl = solver2d_cyl.apply_operator(phi_solution_cyl, temperature_cyl) - rhs_cyl
    print(f"Max residual = {np.abs(residual_cyl).max():.2e}")
    
    print("\n" + "="*70)
    print("2D Examples completed successfully!")
    print("="*70)


# Run 2D examples when file is executed directly
if __name__ == "__main__":
    # Note: 1D examples already ran above in the first main block
    # Now run 2D examples
    run_2d_examples()

