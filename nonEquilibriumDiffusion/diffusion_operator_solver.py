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
from scipy.sparse.linalg import spsolve, splu
from typing import Callable, Optional, Tuple
from numba import njit, prange
import numba

# Physical constants
C_LIGHT = 2.99792458e1  # speed of light (cm/ns)
A_RAD = 0.01372           # radiation constant (GJ/(cm³·keV⁴))


@njit(parallel=True, fastmath=True, cache=True)
def _eval_func_flat_parallel(func, T_flat, X_flat, Y_flat):
    """Evaluate func(T,x,y) over flat arrays in parallel (Numba JIT path).
    func must itself be a @njit-compiled function.
    Returns a 1-D float64 array of the same length.
    """
    n = T_flat.shape[0]
    out = np.empty(n, dtype=np.float64)
    for k in prange(n):
        out[k] = func(T_flat[k], X_flat[k], Y_flat[k])
    return out


def _is_njit(f):
    """Return True if f is a Numba CPUDispatcher (i.e. decorated with @njit)."""
    try:
        return isinstance(f, numba.core.registry.CPUDispatcher)
    except AttributeError:
        return False


def _probe_array_callable(func) -> bool:
    """Return True if *func* accepts and returns numpy arrays element-wise.

    Calls the function with two 2-element float64 arrays.  If it returns
    an array of the same shape, the function supports vectorised numpy
    input and can be called directly instead of through np.vectorize.
    """
    T_probe = np.array([1.0, 2.0])
    x_probe = np.array([0.1, 0.2])
    try:
        result = func(T_probe, x_probe, x_probe)
        return (isinstance(result, np.ndarray) and
                result.shape == T_probe.shape)
    except Exception:
        return False


def _make_callable(func):
    """Return *func* if it accepts array inputs, else wrap with np.vectorize.

    Use this in hot loops (Newton iterations, matrix assembly) to avoid
    per-element Python overhead when the underlying function already supports
    numpy array semantics.
    """
    if _probe_array_callable(func):
        return func
    return np.vectorize(func, otypes=[np.float64])


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

        # Optional callbacks for evaluating the effective diffusion coefficient
        # used in Robin boundary closures. Signature:
        #   f(T_cell, phi_cell, r_boundary) -> D_boundary
        self.left_boundary_diffusion_func = None
        self.right_boundary_diffusion_func = None
    
    def apply_operator(self, phi, T):
        """
        Applies A·phi without solving.
        """
        A = self.assemble_matrix(T)
        return A @ phi


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
            # Default to the boundary-face coefficient from matrix assembly, but
            # allow problem setups to override it so the Robin closure and the
            # effective face coefficient are evaluated consistently.
            D_boundary = D_faces[0]
            if self.left_boundary_diffusion_func is not None:
                try:
                    D_candidate = self.left_boundary_diffusion_func(temperature[0], phi[0], self.r_faces[0])
                    if np.isfinite(D_candidate) and D_candidate > 0.0:
                        D_boundary = D_candidate
                except Exception:
                    pass
            
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
            D_boundary = D_faces[-1]
            if self.right_boundary_diffusion_func is not None:
                try:
                    D_candidate = self.right_boundary_diffusion_func(temperature[-1], phi[-1], self.r_faces[-1])
                    if np.isfinite(D_candidate) and D_candidate > 0.0:
                        D_boundary = D_candidate
                except Exception:
                    pass
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
                 x_min: float = 0.0,
                 x_max: float = 1.0,
                 nx_cells: int = 10,
                 y_min: float = 0.0,
                 y_max: float = 1.0,
                 ny_cells: int = 10,
                 geometry: str = 'cartesian',
                 diffusion_coeff_func: Optional[Callable] = None,
                 absorption_coeff_func: Optional[Callable] = None,
                 dt: float = 1.0,
                 x_faces: Optional[np.ndarray] = None,
                 y_faces: Optional[np.ndarray] = None,
                 left_bc_func: Optional[Callable] = None,
                 right_bc_func: Optional[Callable] = None,
                 bottom_bc_func: Optional[Callable] = None,
                 top_bc_func: Optional[Callable] = None,
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
            Domain boundaries in x-direction (or r for cylindrical) - ignored if x_faces provided
        nx_cells : int
            Number of cells in x-direction - ignored if x_faces provided
        y_min, y_max : float
            Domain boundaries in y-direction (or z for cylindrical) - ignored if y_faces provided
        ny_cells : int
            Number of cells in y-direction - ignored if y_faces provided
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
        x_faces, y_faces : ndarray or None
            Custom face positions. If provided, override x_min/max, nx_cells, y_min/max, ny_cells
        left_bc_func, right_bc_func, bottom_bc_func, top_bc_func : callable or None
            Boundary condition functions returning (A, B, C) for Robin BC: A·φ + B·∇φ = C
            If None, uses string-based BCs
        left_bc, right_bc, bottom_bc, top_bc : str
            Boundary condition types: 'dirichlet' or 'neumann' (used if BC funcs are None)
        left_bc_value, right_bc_value, bottom_bc_value, top_bc_value : float
            Boundary condition values (used if BC funcs are None)
        """
        self.geometry = geometry.lower()
        self.dt = dt
        
        # Matrix cache for fixed-temperature solves (reduces floating point error accumulation)
        self._cached_T = None
        self._cached_A = None
        self._cached_D_x_faces = None
        self._cached_D_y_faces = None
        # LU factorization cache — factorize once per Newton iter, reuse across all GMRES back-subs
        self._cached_LU = None
        
        # Generate or use custom grid
        if x_faces is not None and y_faces is not None:
            # Use custom face arrays
            self.x_faces = np.array(x_faces)
            self.y_faces = np.array(y_faces)
            self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
            self.y_centers = 0.5 * (self.y_faces[:-1] + self.y_faces[1:])
            self.nx_cells = len(self.x_faces) - 1
            self.ny_cells = len(self.y_faces) - 1
            self.x_min = self.x_faces[0]
            self.x_max = self.x_faces[-1]
            self.y_min = self.y_faces[0]
            self.y_max = self.y_faces[-1]
        elif x_faces is not None or y_faces is not None:
            raise ValueError("Must provide both x_faces and y_faces, or neither")
        else:
            # Generate uniform mesh
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            self.nx_cells = nx_cells
            self.ny_cells = ny_cells
            self.x_faces = np.linspace(x_min, x_max, nx_cells + 1)
            self.y_faces = np.linspace(y_min, y_max, ny_cells + 1)
            self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
            self.y_centers = 0.5 * (self.y_faces[:-1] + self.y_faces[1:])
        
        self.n_total = self.nx_cells * self.ny_cells
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
        
        # Boundary condition functions (for Robin BCs)
        self.left_bc_func = left_bc_func
        self.right_bc_func = right_bc_func
        self.bottom_bc_func = bottom_bc_func
        self.top_bc_func = top_bc_func
        
        # Boundary conditions (string-based, for backward compatibility)
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.top_bc = top_bc
        self.left_bc_value = left_bc_value
        self.right_bc_value = right_bc_value
        self.bottom_bc_value = bottom_bc_value
        self.top_bc_value = top_bc_value

        # Fast-path: if both material functions are @njit, use the parallel
        # Numba kernel in assemble_matrix instead of np.vectorize.
        self._use_numba_eval = (
            _is_njit(self.diffusion_coeff_func) and
            _is_njit(self.absorption_coeff_func)
        )

        # Medium fast-path: if the functions accept numpy array inputs
        # (return a same-shaped array for array T/x/y), call them directly
        # instead of through np.vectorize.  Direct array calls run entirely
        # in numpy's C layer → GIL released → true thread parallelism when
        # multiple groups are assembled concurrently.
        self._use_array_call = False
        if not self._use_numba_eval:
            self._use_array_call = (
                _probe_array_callable(self.diffusion_coeff_func) and
                _probe_array_callable(self.absorption_coeff_func)
            )

        # Precompute fixed COO stencil structure and geometric ratios.
        self._precompute_stencil()

    @staticmethod
    def _probe_array_callable(func) -> bool:
        """Delegate to module-level helper (kept for external callers)."""
        return _probe_array_callable(func)

    def _precompute_stencil(self):
        """Precompute fixed row/col arrays and geometric ratio arrays for the
        5-point finite-volume stencil.  Called once at construction; only the
        *coefficients* (which depend on D and σ_a) need recomputing each solve.
        """
        nx = self.nx_cells
        ny = self.ny_cells

        # ---- center-to-center distances for interior x-faces ----
        # _dx_int[k]   = x_centers[k+1] - x_centers[k],  k in [0, nx-2]
        # shape (nx-1, 1) for broadcasting against (nx-1, ny) arrays
        dx_int = (self.x_centers[1:] - self.x_centers[:-1])[:, np.newaxis]

        # ---- center-to-center distances for interior y-faces ----
        # _dy_int[k]   = y_centers[k+1] - y_centers[k],  k in [0, ny-2]
        # shape (1, ny-1) for broadcasting against (nx, ny-1) arrays
        dy_int = (self.y_centers[1:] - self.y_centers[:-1])[np.newaxis, :]

        # Geometric ratios (constant for a fixed mesh):
        #   geom_xl[k,j] = Ax[k+1,j] / (dx_int[k] * V[k+1,j])  → left  contribution to cell k+1
        #   geom_xr[k,j] = Ax[k+1,j] / (dx_int[k] * V[k  ,j])  → right contribution to cell k
        #   geom_yb[i,k] = Ay[i,k+1] / (dy_int[k] * V[i,k+1])  → bottom contribution to cell (i,k+1)
        #   geom_yt[i,k] = Ay[i,k+1] / (dy_int[k] * V[i,k  ])  → top   contribution to cell (i,k)
        self._geom_xl = self.Ax_faces[1:nx, :] / (dx_int * self.V_cells[1:,    :])  # (nx-1, ny)
        self._geom_xr = self.Ax_faces[1:nx, :] / (dx_int * self.V_cells[:nx-1, :])  # (nx-1, ny)
        self._geom_yb = self.Ay_faces[:, 1:ny] / (dy_int * self.V_cells[:,  1:   ])  # (nx, ny-1)
        self._geom_yt = self.Ay_faces[:, 1:ny] / (dy_int * self.V_cells[:, :ny-1 ])  # (nx, ny-1)

        # ---- Fixed COO row/col index arrays (int32) ----
        # x-left  off-diagonal: row=(k+1,j), col=(k,j),   k in [0,nx-2]
        ii_xl, jj_xl = np.meshgrid(np.arange(1,  nx,  dtype=np.int32),
                                    np.arange(ny,       dtype=np.int32), indexing='ij')
        self._rows_xl = (ii_xl      * ny + jj_xl).ravel()
        self._cols_xl = ((ii_xl-1)  * ny + jj_xl).ravel()

        # x-right off-diagonal: row=(k,j),   col=(k+1,j), k in [0,nx-2]
        ii_xr, jj_xr = np.meshgrid(np.arange(nx-1, dtype=np.int32),
                                    np.arange(ny,   dtype=np.int32), indexing='ij')
        self._rows_xr = (ii_xr      * ny + jj_xr).ravel()
        self._cols_xr = ((ii_xr+1)  * ny + jj_xr).ravel()

        # y-bottom off-diagonal: row=(i,k+1), col=(i,k),   k in [0,ny-2]
        ii_yb, jj_yb = np.meshgrid(np.arange(nx,    dtype=np.int32),
                                    np.arange(1, ny, dtype=np.int32), indexing='ij')
        self._rows_yb = (ii_yb * ny + jj_yb    ).ravel()
        self._cols_yb = (ii_yb * ny + jj_yb - 1).ravel()

        # y-top   off-diagonal: row=(i,k),   col=(i,k+1), k in [0,ny-2]
        ii_yt, jj_yt = np.meshgrid(np.arange(nx,    dtype=np.int32),
                                    np.arange(ny-1,  dtype=np.int32), indexing='ij')
        self._rows_yt = (ii_yt * ny + jj_yt    ).ravel()
        self._cols_yt = (ii_yt * ny + jj_yt + 1).ravel()

        # diagonal: all cells
        all_idx = np.arange(self.n_total, dtype=np.int32)
        self._rows_diag = all_idx
        self._cols_diag = all_idx

        # Precomputed 2-D meshes for face positions used when evaluating D(T,x,y)
        # X-face mesh: (nx+1, ny) — Xf_2d[i,j]=x_faces[i], Yc_xf[i,j]=y_centers[j]
        self._Xf_2d  = np.broadcast_to(self.x_faces[:, np.newaxis],
                                        (nx + 1, ny)).copy()
        self._Yc_xf  = np.broadcast_to(self.y_centers[np.newaxis, :],
                                        (nx + 1, ny)).copy()
        # Y-face mesh: (nx, ny+1) — Xc_yf[i,j]=x_centers[i], Yf_2d[i,j]=y_faces[j]
        self._Xc_yf  = np.broadcast_to(self.x_centers[:, np.newaxis],
                                        (nx, ny + 1)).copy()
        self._Yf_2d  = np.broadcast_to(self.y_faces[np.newaxis, :],
                                        (nx, ny + 1)).copy()
        # Cell-center flat arrays for σ_a evaluation
        self._Xc_flat = self.X_centers.ravel()
        self._Yc_flat = self.Y_centers.ravel()

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
            # Vectorized with broadcasting (no loops needed)
            self.Ax_faces[:] = self.dy_cells[np.newaxis, :]          # (nx+1, ny)
            self.Ay_faces[:] = self.dx_cells[:, np.newaxis]           # (nx, ny+1)
            self.V_cells[:]  = self.dx_cells[:, np.newaxis] * self.dy_cells[np.newaxis, :]

        elif self.geometry == 'cylindrical':
            # Cylindrical (r-z): A_r = 2πr·Δz, A_z = π(r_R²-r_L²), V = π(r_R²-r_L²)·Δz
            # Ax_faces[i, j] = 2π * x_faces[i] * dy_cells[j]
            self.Ax_faces[:] = (2.0 * np.pi * self.x_faces[:, np.newaxis]
                                * self.dy_cells[np.newaxis, :])
            # Ay_faces[i, j] = π * (x_faces[i+1]² - x_faces[i]²)
            r_sq_diff = self.x_faces[1:]**2 - self.x_faces[:-1]**2   # (nx,)
            self.Ay_faces[:] = np.pi * r_sq_diff[:, np.newaxis]       # broadcast over j
            # V_cells[i, j] = π * (r_R² - r_L²) * dy_cells[j]
            self.V_cells[:]  = np.pi * r_sq_diff[:, np.newaxis] * self.dy_cells[np.newaxis, :]
        
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

        # ---- Vectorised face temperatures (no conditionals inside the loop) ----
        # X-faces: shape (nx+1, ny)
        T_x = np.empty((nx + 1, ny))
        T_x[0]    = T_2d[0]           # left  boundary: use first cell
        T_x[nx]   = T_2d[nx - 1]      # right boundary: use last  cell
        T_x[1:nx] = 0.5 * (T_2d[:-1] + T_2d[1:])   # interior: arithmetic mean

        # Y-faces: shape (nx, ny+1)
        T_y = np.empty((nx, ny + 1))
        T_y[:, 0]    = T_2d[:, 0]          # bottom boundary
        T_y[:, ny]   = T_2d[:, ny - 1]     # top    boundary
        T_y[:, 1:ny] = 0.5 * (T_2d[:, :-1] + T_2d[:, 1:])

        # ---- Diffusion coefficients at faces ----
        _D = self.diffusion_coeff_func
        T_1d = T_2d.ravel()
        if self._use_numba_eval:
            D_x_faces = _eval_func_flat_parallel(
                _D, T_x.ravel(), self._Xf_2d.ravel(), self._Yc_xf.ravel()
            ).reshape(nx + 1, ny)
            D_y_faces = _eval_func_flat_parallel(
                _D, T_y.ravel(), self._Xc_yf.ravel(), self._Yf_2d.ravel()
            ).reshape(nx, ny + 1)
            sigma_a = _eval_func_flat_parallel(
                self.absorption_coeff_func, T_1d, self._Xc_flat, self._Yc_flat
            )
        elif self._use_array_call:
            # Direct array call: the function supports numpy array inputs.
            # This runs entirely in numpy's C layer, releases the GIL, and
            # avoids the per-element Python overhead of np.vectorize.
            _D = self.diffusion_coeff_func
            D_x_faces = np.asarray(_D(T_x, self._Xf_2d, self._Yc_xf),
                                   dtype=np.float64)
            D_y_faces = np.asarray(_D(T_y, self._Xc_yf, self._Yf_2d),
                                   dtype=np.float64)
            sigma_a = np.asarray(
                self.absorption_coeff_func(T_1d, self._Xc_flat, self._Yc_flat),
                dtype=np.float64
            )
        else:
            _vD = np.vectorize(_D, otypes=[np.float64])
            D_x_faces = _vD(T_x, self._Xf_2d, self._Yc_xf)      # (nx+1, ny)
            D_y_faces = _vD(T_y, self._Xc_yf, self._Yf_2d)      # (nx, ny+1)
            sigma_a = np.vectorize(self.absorption_coeff_func, otypes=[np.float64])(
                T_1d, self._Xc_flat, self._Yc_flat
            )

        # ---- Absorption coefficients at cell centres ----
        sigma_a_2d = sigma_a.reshape((nx, ny))

        # σ_total = σ_a + 1/(c·Δt)
        sigma_total = sigma_a_2d + 1.0 / (C_LIGHT * self.dt)

        # ---- Vectorised COO assembly (no Python inner loop) ----
        # Off-diagonal coefficients: coeff = (precomputed geom ratio) × D_face
        coeff_xl = self._geom_xl * D_x_faces[1:nx,  :]   # (nx-1, ny)
        coeff_xr = self._geom_xr * D_x_faces[1:nx,  :]   # (nx-1, ny)
        coeff_yb = self._geom_yb * D_y_faces[:,  1:ny]    # (nx, ny-1)
        coeff_yt = self._geom_yt * D_y_faces[:,  1:ny]    # (nx, ny-1)

        # Diagonal = σ_total + sum of all neighbour coupling coefficients
        diag = sigma_total.copy()
        diag[1:,    :] += coeff_xl      # left-face contribution  to cell i (i≥1)
        diag[:nx-1, :] += coeff_xr      # right-face contribution to cell i (i≤nx-2)
        diag[:,  1:]   += coeff_yb      # bottom-face contribution to cell j (j≥1)
        diag[:, :ny-1] += coeff_yt      # top-face   contribution to cell j (j≤ny-2)

        # Build COO using pre-allocated index arrays (constructed once at init)
        rows = np.concatenate([self._rows_xl, self._rows_xr,
                                self._rows_yb, self._rows_yt, self._rows_diag])
        cols = np.concatenate([self._cols_xl, self._cols_xr,
                                self._cols_yb, self._cols_yt, self._cols_diag])
        data = np.concatenate([-coeff_xl.ravel(), -coeff_xr.ravel(),
                                -coeff_yb.ravel(), -coeff_yt.ravel(),
                                 diag.ravel()])

        A_coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
        A_csr = A_coo.tocsr()

        # Store diagonal before BC modification (for reference / diagnostics)
        diag_contribution = diag.ravel()

        return A_csr, D_x_faces, D_y_faces, diag_contribution
    
    def apply_boundary_conditions(self, A: sparse.csr_matrix, rhs: np.ndarray,
                                 phi: np.ndarray, temperature: np.ndarray,
                                 D_x_faces: np.ndarray, D_y_faces: np.ndarray,
                                 bc_time: float = 0.0,
                                 skip_matrix: bool = False,
                                 skip_rhs: bool = False):
        """
        Apply boundary conditions to the matrix and/or RHS.
        Supports Robin BCs: A·φ + B·∇φ = C

        Parameters
        ----------
        skip_matrix : bool
            If True, skip diagonal modifications to A (only update rhs).
            Use when the LU factorization of A is already cached.
        skip_rhs : bool
            If True, skip source contributions to rhs (only update A).
            Use when building the matrix for LU factorization.
        """
        nx = self.nx_cells
        ny = self.ny_cells

        if not skip_matrix:
            A = A.tolil()  # LIL allows efficient diagonal modification

        # Convert phi and temperature to 2D for easier indexing
        phi_2d = phi.reshape(nx, ny)
        T_2d = temperature.reshape(nx, ny)

        # Left boundary (i=0, x=x_min)
        for j in range(ny):
            idx = self._index_2d_to_1d(0, j)
            y_pos = self.y_centers[j]

            if self.left_bc_func is not None:
                A_bc, B_bc, C_bc = self.left_bc_func(phi_2d[0, j], (self.x_faces[0], y_pos), bc_time)

                if abs(B_bc) < 1e-14:
                    phi_boundary = C_bc / A_bc
                    T_avg = T_2d[0, j]
                    D_boundary = self.diffusion_coeff_func(T_avg, self.x_faces[0], y_pos)
                    dx_half = self.x_centers[0] - self.x_faces[0]
                    flux_coeff = (self.Ax_faces[0, j] * D_boundary) / (self.V_cells[0, j] * dx_half)
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += flux_coeff * phi_boundary
                else:
                    D_boundary = D_x_faces[0, j]
                    flux_coeff = (self.Ax_faces[0, j] * D_boundary * A_bc) / (B_bc * self.V_cells[0, j])
                    rhs_contribution = (self.Ax_faces[0, j] * D_boundary * C_bc) / (B_bc * self.V_cells[0, j])
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += rhs_contribution
            else:
                if self.left_bc == 'dirichlet':
                    if not skip_matrix:
                        A[idx, :] = 0.0
                        A[idx, idx] = 1.0
                    if not skip_rhs:
                        rhs[idx] = self.left_bc_value
                elif self.left_bc == 'neumann':
                    pass

        # Right boundary (i=nx-1, x=x_max)
        for j in range(ny):
            idx = self._index_2d_to_1d(nx-1, j)
            y_pos = self.y_centers[j]

            if self.right_bc_func is not None:
                A_bc, B_bc, C_bc = self.right_bc_func(phi_2d[nx-1, j], (self.x_faces[-1], y_pos), bc_time)

                if abs(B_bc) < 1e-14:
                    phi_boundary = C_bc / A_bc
                    T_avg = T_2d[nx-1, j]
                    D_boundary = self.diffusion_coeff_func(T_avg, self.x_faces[-1], y_pos)
                    dx_half = self.x_faces[-1] - self.x_centers[nx-1]
                    flux_coeff = (self.Ax_faces[nx, j] * D_boundary) / (self.V_cells[nx-1, j] * dx_half)
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += flux_coeff * phi_boundary
                else:
                    D_boundary = D_x_faces[nx, j]
                    flux_coeff = (self.Ax_faces[nx, j] * D_boundary * A_bc) / (B_bc * self.V_cells[nx-1, j])
                    rhs_contribution = (self.Ax_faces[nx, j] * D_boundary * C_bc) / (B_bc * self.V_cells[nx-1, j])
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += rhs_contribution
            else:
                if self.right_bc == 'dirichlet':
                    if not skip_matrix:
                        A[idx, :] = 0.0
                        A[idx, idx] = 1.0
                    if not skip_rhs:
                        rhs[idx] = self.right_bc_value
                elif self.right_bc == 'neumann':
                    pass

        # Bottom boundary (j=0, y=y_min)
        for i in range(nx):
            idx = self._index_2d_to_1d(i, 0)
            x_pos = self.x_centers[i]

            if self.bottom_bc_func is not None:
                A_bc, B_bc, C_bc = self.bottom_bc_func(phi_2d[i, 0], (x_pos, self.y_faces[0]), bc_time)

                if abs(B_bc) < 1e-14:
                    phi_boundary = C_bc / A_bc
                    T_avg = T_2d[i, 0]
                    D_boundary = self.diffusion_coeff_func(T_avg, x_pos, self.y_faces[0])
                    dy_half = self.y_centers[0] - self.y_faces[0]
                    flux_coeff = (self.Ay_faces[i, 0] * D_boundary) / (self.V_cells[i, 0] * dy_half)
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += flux_coeff * phi_boundary
                else:
                    D_boundary = D_y_faces[i, 0]
                    flux_coeff = (self.Ay_faces[i, 0] * D_boundary * A_bc) / (B_bc * self.V_cells[i, 0])
                    rhs_contribution = (self.Ay_faces[i, 0] * D_boundary * C_bc) / (B_bc * self.V_cells[i, 0])
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += rhs_contribution
            else:
                if self.bottom_bc == 'dirichlet':
                    if not skip_matrix:
                        A[idx, :] = 0.0
                        A[idx, idx] = 1.0
                    if not skip_rhs:
                        rhs[idx] = self.bottom_bc_value
                elif self.bottom_bc == 'neumann':
                    pass

        # Top boundary (j=ny-1, y=y_max)
        for i in range(nx):
            idx = self._index_2d_to_1d(i, ny-1)
            x_pos = self.x_centers[i]

            if self.top_bc_func is not None:
                A_bc, B_bc, C_bc = self.top_bc_func(phi_2d[i, ny-1], (x_pos, self.y_faces[-1]), bc_time)

                if abs(B_bc) < 1e-14:
                    phi_boundary = C_bc / A_bc
                    T_avg = T_2d[i, ny-1]
                    D_boundary = self.diffusion_coeff_func(T_avg, x_pos, self.y_faces[-1])
                    dy_half = self.y_faces[-1] - self.y_centers[ny-1]
                    flux_coeff = (self.Ay_faces[i, ny] * D_boundary) / (self.V_cells[i, ny-1] * dy_half)
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += flux_coeff * phi_boundary
                else:
                    D_boundary = D_y_faces[i, ny]
                    flux_coeff = (self.Ay_faces[i, ny] * D_boundary * A_bc) / (B_bc * self.V_cells[i, ny-1])
                    rhs_contribution = (self.Ay_faces[i, ny] * D_boundary * C_bc) / (B_bc * self.V_cells[i, ny-1])
                    if not skip_matrix:
                        A[idx, idx] += flux_coeff
                    if not skip_rhs:
                        rhs[idx] += rhs_contribution
            else:
                if self.top_bc == 'dirichlet':
                    if not skip_matrix:
                        A[idx, :] = 0.0
                        A[idx, idx] = 1.0
                    if not skip_rhs:
                        rhs[idx] = self.top_bc_value
                elif self.top_bc == 'neumann':
                    pass

        if skip_matrix:
            # A was never converted to LIL and was never modified; the return
            # value is discarded by all skip_matrix=True callers.
            return None
        result = A.tocsr()
        del A  # free the LIL matrix immediately, before returning to the caller
        return result
    
    def solve(self,
             rhs: np.ndarray,
             temperature: np.ndarray,
             phi_guess: Optional[np.ndarray] = None,
             use_iterative: bool = False,
             max_iter: int = 10,
             tol: float = 1e-6,
             bc_time: float = 0.0,
             override_left_bc: Optional[Callable] = None,
             override_right_bc: Optional[Callable] = None,
             override_bottom_bc: Optional[Callable] = None,
             override_top_bc: Optional[Callable] = None,
             skip_bc_rhs: bool = False) -> np.ndarray:
        """
        Solve A φ = b for φ.

        Parameters
        ----------
        skip_bc_rhs : bool
            When True the boundary-condition *source* terms (C_bc) are not
            added to the RHS.  Use this for homogeneous-BC solves (operator B
            in the GMRES matvec) where C_bc = 0 for every boundary cell.
            Avoids all per-cell BC function calls on cache-hit paths, which
            is the dominant cost for large meshes with many GMRES iterations.
        """
        nx = self.nx_cells
        ny = self.ny_cells
        n_total = self.n_total
        
        # Temporarily override boundary conditions if requested
        saved_left_bc = self.left_bc_func
        saved_right_bc = self.right_bc_func
        saved_bottom_bc = self.bottom_bc_func
        saved_top_bc = self.top_bc_func
        if override_left_bc is not None:
            self.left_bc_func = override_left_bc
        if override_right_bc is not None:
            self.right_bc_func = override_right_bc
        if override_bottom_bc is not None:
            self.bottom_bc_func = override_bottom_bc
        if override_top_bc is not None:
            self.top_bc_func = override_top_bc
        
        # Flatten inputs if needed
        if rhs.shape == (nx, ny):
            rhs_1d = rhs.flatten(order='C')
        else:
            rhs_1d = rhs
        
        if temperature.shape == (nx, ny):
            T_1d = temperature.flatten(order='C')
        else:
            T_1d = temperature
        
        if phi_guess is None:
            phi_1d = np.zeros(n_total)
        elif phi_guess.shape == (nx, ny):
            phi_1d = phi_guess.flatten(order='C')
        else:
            phi_1d = phi_guess
        
        # Iterative solution for flux-limited diffusion
        for iteration in range(max_iter):
            # ----------------------------------------------------------------
            # LU-caching fast path
            # Factorize A once per Newton iteration (when T changes), then
            # reuse the factorization for every GMRES back-substitution.
            # This eliminates ~5000 redundant factorizations per time step.
            # ----------------------------------------------------------------
            T_cached = (self._cached_T is not None and
                        self._cached_A is not None and
                        self._cached_LU is not None and
                        np.array_equal(T_1d, self._cached_T))

            if T_cached:
                if skip_bc_rhs:
                    # Homogeneous BC solve (GMRES operator B matvec):
                    # C_bc = 0 everywhere → no RHS modifications needed.
                    # Just do a back-substitution with the cached factorization.
                    phi_new = self._cached_LU.solve(rhs_1d)
                else:
                    # Same T, inhomogeneous RHS: apply BC source contributions.
                    D_x_faces = self._cached_D_x_faces
                    D_y_faces = self._cached_D_y_faces
                    rhs_bc = rhs_1d.copy()
                    self.apply_boundary_conditions(
                        None, rhs_bc, phi_1d, T_1d,
                        D_x_faces, D_y_faces, bc_time=bc_time,
                        skip_matrix=True
                    )
                    phi_new = self._cached_LU.solve(rhs_bc)
            else:
                # T changed (or first call): assemble and factorize.
                A, D_x_faces, D_y_faces, _ = self.assemble_matrix(T_1d, phi_1d)

                # Build full matrix (interior + BC diagonal terms).
                # When skip_bc_rhs=True we still need the BC matrix terms
                # (A_bc/B_bc diagonal flux coupling) but not the RHS source.
                rhs_bc = rhs_1d.copy()
                A_full = self.apply_boundary_conditions(
                    A, rhs_bc, phi_1d, T_1d, D_x_faces, D_y_faces,
                    bc_time=bc_time, skip_rhs=skip_bc_rhs
                )

                lu = splu(A_full.tocsc())
                del A_full  # free post-BC matrix; LU factorization is all we need
                if not use_iterative:
                    self._cached_T = T_1d.copy()
                    self._cached_A = A  # reference only — no copy needed; A goes out of scope after this block
                    self._cached_D_x_faces = D_x_faces
                    self._cached_D_y_faces = D_y_faces
                    self._cached_LU = lu

                phi_new = lu.solve(rhs_bc)
            
            # Check convergence
            if use_iterative:
                rel_change = np.linalg.norm(phi_new - phi_1d) / (np.linalg.norm(phi_new) + 1e-14)
                if rel_change < tol:
                    # Restore original BCs
                    self.left_bc_func = saved_left_bc
                    self.right_bc_func = saved_right_bc
                    self.bottom_bc_func = saved_bottom_bc
                    self.top_bc_func = saved_top_bc
                    return phi_new.reshape((nx, ny), order='C')
                phi_1d = phi_new
            else:
                # Restore original BCs
                self.left_bc_func = saved_left_bc
                self.right_bc_func = saved_right_bc
                self.bottom_bc_func = saved_bottom_bc
                self.top_bc_func = saved_top_bc
                return phi_new.reshape((nx, ny), order='C')
        
        # Restore original BCs (if max iterations reached)
        self.left_bc_func = saved_left_bc
        self.right_bc_func = saved_right_bc
        self.bottom_bc_func = saved_bottom_bc
        self.top_bc_func = saved_top_bc
        
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
        
        A, _, _, _ = self.assemble_matrix(temperature, phi_1d)
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

