#!/usr/bin/env python3
"""
1D Finite Volume Solver for Equilibrium Radiation Diffusion Equation
using Newton iterations in terms of radiation energy density E_r.

PDE: ∇·(D(E_r) ∇E_r) + u(E_r) = Qhat
Time discretization: implicit Euler (quasi-steady at each time step)
Newton method in "next-iterate" operator form
"""

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


# =============================================================================
# MATERIAL PROPERTY FUNCTIONS
# =============================================================================

@njit
def temperature_from_Er(Er):
    """Convert radiation energy density to temperature: T = (E_r/a)^(1/4)"""
    return (Er / A_RAD) ** 0.25


@njit
def specific_heat_cv(T):
    """Specific heat capacity c_v(T). Default: constant."""
    return CV_CONST


@njit
def material_energy_density(T):
    """Material energy density e(T) = ρ c_v T"""
    return RHO * specific_heat_cv(T) * T


@njit
def rosseland_opacity(Er):
    """Rosseland opacity σ_R(E_r). Default: constant for testing."""
    return 0.1  # cm⁻¹


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

@njit
def robin_bc_coefficients(Er, x, is_left=True):
    """
    Robin boundary condition: A_bc * E_r + B_bc(E_r) * (n·∇E_r) = C_bc
    Returns (A_bc, B_bc, C_bc)
    
    For testing: use simple Dirichlet-like conditions
    """
    if is_left:
        # Left boundary (r = r_min)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet: E_r = C_bc
        C_bc = 1.0  # E_r(r_min) = 1.0
    else:
        # Right boundary (r = r_max)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet: E_r = C_bc
        C_bc = 0.1  # E_r(r_max) = 0.1
    
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


@njit
def interpolate_face_value(E_left, E_right, r_left, r_center_left, 
                          r_center_right, r_right):
    """
    Distance-weighted interpolation for face value E_{i+1/2}
    Used for nonlinear correction term
    """
    # Distance from face to left and right cell centers
    d_left = abs(0.5 * (r_left + r_right) - r_center_left)
    d_right = abs(r_center_right - 0.5 * (r_left + r_right))
    
    if d_left + d_right == 0:
        return 0.5 * (E_left + E_right)
    
    # Distance-weighted average
    w_left = d_right / (d_left + d_right)
    w_right = d_left / (d_left + d_right)
    
    return w_left * E_left + w_right * E_right


# =============================================================================
# TRIDIAGONAL SOLVER (THOMAS ALGORITHM)
# =============================================================================

@njit
def solve_tridiagonal(A_tri, rhs):
    """
    Solve tridiagonal system using Thomas algorithm
    
    Parameters:
    -----------
    A_tri : ndarray (3, n)
        Tridiagonal matrix with format:
        A_tri[0, :] = sub-diagonal (a_i)
        A_tri[1, :] = main diagonal (b_i) 
        A_tri[2, :] = super-diagonal (c_i)
    rhs : ndarray (n,)
        Right-hand side vector
    
    Returns:
    --------
    x : ndarray (n,)
        Solution vector
    """
    n = len(rhs)
    x = np.zeros(n)
    
    # Work on copies to avoid modifying input arrays
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Extract diagonals
    a = A_tri[0, :]  # sub-diagonal
    b = A_tri[1, :]  # main diagonal  
    c = A_tri[2, :]  # super-diagonal
    
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





# =============================================================================
# MAIN SOLVER CLASS
# =============================================================================

class RadiationDiffusionSolver:
    """1D Finite Volume Radiation Diffusion Solver"""
    
    def __init__(self, r_min=0.1, r_max=1.0, n_cells=50, d=0, dt=1e-3,
                 max_newton_iter=10, newton_tol=1e-8, stretch_factor=1.0,
                 rosseland_opacity_func=None, specific_heat_func=None, 
                 material_energy_func=None, left_bc_func=None, right_bc_func=None):
        
        self.r_min = r_min
        self.r_max = r_max  
        self.n_cells = n_cells
        self.d = d  # Coordinate system
        self.dt = dt
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        
        # Generate grid and geometry
        self.r_faces, self.r_centers = generate_grid(r_min, r_max, n_cells, stretch_factor)
        self.A_faces, self.V_cells = compute_geometry_factors(self.r_faces, d)
        
        # Solution arrays
        self.Er = np.ones(n_cells)  # Current solution
        self.Er_old = np.ones(n_cells)  # Previous time step
        
        # Material property functions (use defaults if not provided)
        self.rosseland_opacity_func = rosseland_opacity_func or rosseland_opacity
        self.specific_heat_func = specific_heat_func or specific_heat_cv
        self.material_energy_func = material_energy_func or material_energy_density
        
        # Boundary condition functions (use defaults if not provided)
        self.left_bc_func = left_bc_func or (lambda Er, x: robin_bc_coefficients(Er, x, True))
        self.right_bc_func = right_bc_func or (lambda Er, x: robin_bc_coefficients(Er, x, False))
        

        
        # Solver options
        self.use_nonlinear_correction = True
        self.use_secant_derivative = True
        self.max_newton_iter_per_step = 1  # For one-step method, set to 1
        self.nonlinear_skip_boundary_cells = 0  # Skip nonlinear correction for first N cells
        self.nonlinear_limiter = 1.0  # Limit NL correction to this fraction of linear term (1.0 = no limit)
        
        print(f"Initialized solver: {n_cells} cells, r ∈ [{r_min:.3f}, {r_max:.3f}]")
        print(f"Coordinate system d={d}, Δt={dt:.2e}")
    
    def get_diffusion_coefficient(self, Er):
        """Get diffusion coefficient using the configured opacity function"""
        return C_LIGHT / (3.0 * self.rosseland_opacity_func(Er))
    
    def get_diffusion_coefficient_derivative(self, Er, Er_prev=None, dEr=1e-5):
        """Get derivative of diffusion coefficient"""
        if Er_prev is not None:
            # Secant approximation
            dE = Er - Er_prev
            if abs(dE) > 1e-10 * max(abs(Er), abs(Er_prev), 1e-14):
                return (self.get_diffusion_coefficient(Er) - self.get_diffusion_coefficient(Er_prev)) / dE
            else:
                # Fall back to finite difference if change is too small
                return (self.get_diffusion_coefficient(Er + dEr) - self.get_diffusion_coefficient(Er - dEr)) / (2 * dEr)
        else:
            # Finite difference
            return (self.get_diffusion_coefficient(Er + dEr) - self.get_diffusion_coefficient(Er - dEr)) / (2 * dEr)
    
    def get_alpha_coefficient(self, Er, dt):
        """Get Newton linearization coefficient using configured material functions
        
        Note: expects specific_heat_func(T) to return cv (per unit mass), then
        computes ρ*cv internally. For α = (1/dt)*(1 + ρ*cv/(4*a^(1/4)*Er^(3/4)))
        """
        T = temperature_from_Er(np.abs(Er))
        cv = self.specific_heat_func(T)
        rho_cv = RHO * cv
        term = rho_cv / (4.0 * A_RAD**0.25 * np.abs(Er)**0.75)
        return (1.0 / dt) * (1.0 + term)
    
    def get_u_function(self, Er, dt):
        """Get u function using configured material energy function"""
        T = temperature_from_Er(Er)
        e_mat = self.material_energy_func(T)
        return (1.0 / dt) * (e_mat + Er)
    
    def set_initial_condition(self, Er_init):
        """Set initial radiation energy density"""
        if callable(Er_init):
            self.Er = Er_init(self.r_centers)
            self.Er_old = Er_init(self.r_centers)
        else:
            self.Er = np.full(self.n_cells, Er_init)
            self.Er_old = np.full(self.n_cells, Er_init)
        
        # Validate positive energy density
        if np.any(self.Er <= 0):
            raise ValueError("Initial radiation energy density must be positive everywhere")
    
    def newton_step(self, Er_prev_timestep, verbose=False):
        """Perform Newton iterations for one time step"""
        Er_k = self.Er.copy()  # Initial guess for this time step
        
        for k in range(self.max_newton_iter):
            # Assemble linear system
            A_tri, rhs = self.assemble_system(
                Er_k, Er_prev_timestep, 
                self.use_nonlinear_correction, self.use_secant_derivative
            )
            
            # Apply boundary conditions (including nonlinear correction if enabled)
            self.apply_boundary_conditions(A_tri, rhs, Er_k, self.use_nonlinear_correction)
            
            # Solve linear system
            Er_new = solve_tridiagonal(A_tri, rhs)
            
            # Comprehensive solution validation and recovery
            solution_valid = True
            
            # Check for NaN/Inf values
            if np.any(~np.isfinite(Er_new)):
                if verbose:
                    print(f"    Newton iteration produced invalid values!")
                    print(f"    Residual: {np.linalg.norm(Er_new - Er_k):.2e}")
                    print(f"    Er_k min/max: {Er_k.min():.2e}/{Er_k.max():.2e}")
                    print(f"    Er_new min/max: {Er_new.min():.2e}/{Er_new.max():.2e}")
                
                # Fallback: use very conservative update
                delta_Er = Er_new - Er_k
                # Only use finite components
                finite_mask = np.isfinite(Er_new)
                if np.any(finite_mask):
                    Er_new = Er_k.copy()
                    Er_new[finite_mask] = Er_k[finite_mask] + 0.01 * (Er_new[finite_mask] - Er_k[finite_mask])
                else:
                    # Complete failure - use previous iterate
                    Er_new = Er_k.copy()
                solution_valid = False
            
            # Check for negative values
            elif np.any(Er_new < 0):
                # Line search: find largest alpha such that Er_k + alpha*(Er_new - Er_k) > 0
                delta_Er = Er_new - Er_k
                alpha = 1.0
                
                # Find minimum alpha to keep all values positive
                for i in range(len(Er_k)):
                    if delta_Er[i] < 0 and Er_k[i] + alpha * delta_Er[i] <= 0:
                        # Require Er_k[i] + alpha * delta_Er[i] >= epsilon
                        epsilon = max(1e-12, 0.001 * abs(Er_k[i]))
                        alpha_needed = (epsilon - Er_k[i]) / delta_Er[i] if delta_Er[i] != 0 else 0.1
                        alpha = min(alpha, abs(alpha_needed))
                
                # Apply conservative damping
                alpha = max(0.01, min(alpha, 0.9))  # More conservative range
                Er_new = Er_k + alpha * delta_Er
                
                if verbose:
                    print(f"    Applied Newton damping: alpha = {alpha:.3f}")
            
            # Additional safety check after damping
            if np.any(Er_new <= 0) or np.any(~np.isfinite(Er_new)):
                # Emergency fallback: minimal update
                Er_new = Er_k + 1e-6 * np.abs(Er_k)
                solution_valid = False
                if verbose:
                    print(f"    Applied emergency fallback update")
                Er_new = Er_k + alpha * delta_Er
                print(f"    Applied Newton damping: alpha = {alpha:.3f}")
            
            # Check convergence
            norm_k = np.linalg.norm(Er_k)
            if norm_k < 1e-14:
                norm_k = 1.0  # Avoid division by zero
            residual = np.linalg.norm(Er_new - Er_k) / norm_k
            
            # Check for NaN or invalid values
            if np.isnan(residual) or np.any(np.isnan(Er_new)):
                print("    Newton iteration produced invalid values!")
                print(f"    Residual: {residual}")
                print(f"    Er_k min/max: {Er_k.min():.2e}/{Er_k.max():.2e}")
                print(f"    Er_new min/max: {Er_new.min():.2e}/{Er_new.max():.2e}")
                #make a plot of the energy density
                plt.plot(Er_new)
                plt.show()
                raise ValueError("Invalid values in Newton iteration")
            
            if residual < self.newton_tol:
                if verbose:
                    print(f"    Newton converged in {k+1} iterations, residual={residual:.2e}")
                return Er_new
            
            Er_k = Er_new.copy()
            
            # For one-step method, exit after first iteration
            if self.max_newton_iter_per_step == 1:
                break
        
        if verbose:
            print(f"    Newton max iterations reached, residual={residual:.2e}")
        return Er_k
    
    def adaptive_nonlinear_limiter(self, Er_k, current_limiter):
        """Adaptively adjust nonlinear limiter based on solution stability"""
        
        # Check for rapid temperature variations (sign of instability)
        T_k = np.array([temperature_from_Er(Er) for Er in Er_k])
        
        if len(T_k) > 2:
            # Compute temperature gradient magnitude
            dT_dr = np.abs(np.diff(T_k))
            max_gradient = np.max(dT_dr) if len(dT_dr) > 0 else 0.0
            
            # If gradients are very large, reduce limiter
            if max_gradient > 10.0:  # Large temperature gradient (keV/cell)
                return min(current_limiter, 0.1)
            elif max_gradient > 1.0:
                return min(current_limiter, 0.3)
        
        return current_limiter
    
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

    def time_step(self, n_steps=1, verbose=True):
        """Advance solution by n_steps time steps"""
        
        for step in range(n_steps):
            if verbose:
                print(f"Time step {step+1}/{n_steps}")
            
            # Store previous solution
            Er_prev = self.Er.copy()
            
            # Newton iterations
            self.Er = self.newton_step(Er_prev, verbose=verbose)
            
            # Update for next time step
            self.Er_old = Er_prev.copy()
    
    def get_solution(self):
        """Return current solution"""
        return self.r_centers.copy(), self.Er.copy()
    
    def assemble_system(self, Er_k, Er_prev, use_nonlinear_correction=True, use_secant_derivative=True):
        """Assemble tridiagonal system for Newton iteration using configured material functions"""
        n_cells = len(Er_k)
        n_faces = len(self.r_faces)
        
        # Initialize tridiagonal matrix and RHS
        A_tri = np.zeros((3, n_cells))  # [sub, main, super]
        rhs = np.zeros(n_cells)
        
        # Evaluate diffusion coefficients at cell centers
        D_cells = np.array([self.get_diffusion_coefficient(Er_k[i]) for i in range(n_cells)])
        DE_cells = np.zeros(n_cells)  # ∂D/∂E_r
        alpha_cells = np.array([self.get_alpha_coefficient(Er_k[i], self.dt) for i in range(n_cells)])
        
        # Compute Qhat at PREVIOUS time step (should not change during Newton iterations)
        qhat_cells = np.array([self.get_u_function(Er_prev[i], self.dt) for i in range(n_cells)])
        
        for i in range(n_cells):
            if use_secant_derivative and Er_prev is not None:
                DE_cells[i] = self.get_diffusion_coefficient_derivative(Er_k[i], Er_prev[i])
            else:
                DE_cells[i] = self.get_diffusion_coefficient_derivative(Er_k[i])
        
        # Face diffusion coefficients (evaluate at averaged temperature, then harmonic mean)
        D_faces = np.zeros(n_faces)
        DE_faces = np.zeros(n_faces)
        
        # Interior faces
        for i in range(1, n_faces - 1):
            # Distance from face to left and right cell centers
            dx_left = self.r_faces[i] - self.r_centers[i-1]  # r_{i+1/2} - r_i
            dx_right = self.r_centers[i] - self.r_faces[i]   # r_{i+1} - r_{i+1/2}
            
            # Average temperature at face, then evaluate D at that temperature
            T_left = temperature_from_Er(Er_k[i-1])
            T_right = temperature_from_Er(Er_k[i])
            T_face = 0.5 * (T_left + T_right)
            Er_face = A_RAD * T_face**4
            
            # Evaluate D at the face temperature
            D_at_face_temp = self.get_diffusion_coefficient(Er_face)
            D_faces[i] = D_at_face_temp
            
            # For derivative, use harmonic average (less critical)
            DE_faces[i] = harmonic_average_weighted(DE_cells[i-1], DE_cells[i], dx_left, dx_right)
        
        # Boundary faces
        D_faces[0] = D_cells[0]
        D_faces[-1] = D_cells[-1]
        DE_faces[0] = DE_cells[0]
        DE_faces[-1] = DE_cells[-1]
        
        # Assemble interior cells
        for i in range(n_cells):
            # Cell volume and face areas
            V_i = self.V_cells[i]
            A_left = self.A_faces[i]
            A_right = self.A_faces[i + 1]
            
            # Grid spacing
            if i == 0:
                dr_left = self.r_centers[i] - self.r_faces[i]
            else:
                dr_left = self.r_centers[i] - self.r_centers[i - 1]
                
            if i == n_cells - 1:
                dr_right = self.r_faces[i + 1] - self.r_centers[i]
            else:
                dr_right = self.r_centers[i + 1] - self.r_centers[i]
            
            # Diffusive flux contributions (L^(k) operator)
            if i > 0:  # Interior left face
                coeff_left = A_left * D_faces[i] / (dr_left * V_i)
                A_tri[0, i] = -coeff_left  # sub-diagonal (E_{i-1})
                A_tri[1, i] += coeff_left  # main diagonal (E_i)
                
            if i < n_cells - 1:  # Interior right face
                coeff_right = A_right * D_faces[i + 1] / (dr_right * V_i)
                A_tri[2, i] = -coeff_right  # super-diagonal (E_{i+1})
                A_tri[1, i] += coeff_right  # main diagonal (E_i)
            
            # Volume term: α^(k) * φ
            A_tri[1, i] += alpha_cells[i]
            
            # Right-hand side: N^(k)[E_r^(k)] + α^(k) * E_r^(k) - u(E_r^(k)) + Qhat
            # where Qhat = u(E_r^n) is evaluated at PREVIOUS time step
            u_k = self.get_u_function(Er_k[i], self.dt)
            qhat = qhat_cells[i]  # Use value from previous time step (frozen during Newton)
            
            rhs[i] = alpha_cells[i] * Er_k[i] - u_k + qhat
            
            # Nonlinear correction term N^(k) (optional)
            # Skip for first few cells near boundaries if requested (for testing/debugging)
            skip_this_cell = i < self.nonlinear_skip_boundary_cells or i >= n_cells - self.nonlinear_skip_boundary_cells
            
            # Also skip if neighboring cells are in skip zone (for consistency at faces)
            if not skip_this_cell and i > 0:
                skip_this_cell = skip_this_cell or (i-1 < self.nonlinear_skip_boundary_cells)
            if not skip_this_cell and i < n_cells - 1:
                skip_this_cell = skip_this_cell or (i+1 >= n_cells - self.nonlinear_skip_boundary_cells)
            
            if use_nonlinear_correction and not skip_this_cell:
                # Enhanced nonlinear correction with robustness checks
                try:
                    # Validate D_E values are finite and reasonable
                    if i > 0 and not np.isfinite(DE_faces[i]):
                        continue  # Skip this correction
                    if i < n_cells - 1 and not np.isfinite(DE_faces[i + 1]):
                        continue  # Skip this correction
                    
                    # N^(k)[E_r^(k+1)] contributions to MATRIX (LHS)
                    # N^(k)[φ] = -∇·(D_E^(k) φ ∇E_r^(k))
                    # Discretized as: -(1/V) * [A_right * F_right - A_left * F_left]
                    
                    # Compute linear diffusion diagonal coefficient for limiting
                    linear_diag_coeff = 0.0
                    if i > 0:
                        linear_diag_coeff += A_left * D_faces[i] / (dr_left * V_i)
                    if i < n_cells - 1:
                        linear_diag_coeff += A_right * D_faces[i + 1] / (dr_right * V_i)
                    
                    # Safety check for linear coefficient
                    if not np.isfinite(linear_diag_coeff) or linear_diag_coeff <= 0:
                        continue  # Skip if linear term is problematic
                
                    # Left face contribution to matrix
                    if i > 0:
                        grad_Er_k = (Er_k[i] - Er_k[i-1]) / dr_left
                        
                        # Check gradient is reasonable
                        if np.isfinite(grad_Er_k) and abs(grad_Er_k) < 1e10:
                            # Flux contribution with safety checks
                            de_face = DE_faces[i]
                            if np.isfinite(de_face) and abs(de_face) < 1e10:
                                coeff_nl = 0.5 * A_left * de_face * grad_Er_k / V_i
                                
                                # Enhanced limiter with multiple checks
                                max_nl_coeff = self.nonlinear_limiter * linear_diag_coeff
                                
                                # Additional limit based on matrix conditioning
                                max_matrix_coeff = 0.1 * abs(A_tri[1, i])  # Don't overwhelm diagonal
                                max_nl_coeff = min(max_nl_coeff, max_matrix_coeff)
                                
                                if abs(coeff_nl) > max_nl_coeff:
                                    coeff_nl = np.sign(coeff_nl) * max_nl_coeff
                                
                                # Final safety check
                                if np.isfinite(coeff_nl):
                                    A_tri[0, i] += coeff_nl  # coefficient of E_{i-1}^(k+1) 
                                    A_tri[1, i] += coeff_nl  # coefficient of E_i^(k+1)
                
                    # Right face contribution to matrix with safety checks
                    if i < n_cells - 1:
                        grad_Er_k = (Er_k[i+1] - Er_k[i]) / dr_right
                        
                        # Check gradient is reasonable
                        if np.isfinite(grad_Er_k) and abs(grad_Er_k) < 1e10:
                            de_face = DE_faces[i + 1]
                            if np.isfinite(de_face) and abs(de_face) < 1e10:
                                coeff_nl = 0.5 * A_right * de_face * grad_Er_k / V_i
                                
                                # Apply enhanced limiter
                                max_nl_coeff = self.nonlinear_limiter * linear_diag_coeff
                                max_matrix_coeff = 0.1 * abs(A_tri[1, i])
                                max_nl_coeff = min(max_nl_coeff, max_matrix_coeff)
                                
                                if abs(coeff_nl) > max_nl_coeff:
                                    coeff_nl = np.sign(coeff_nl) * max_nl_coeff
                                
                                # Final safety check
                                if np.isfinite(coeff_nl):
                                    A_tri[1, i] -= coeff_nl   # coefficient of E_i^(k+1)
                                    A_tri[2, i] -= coeff_nl   # coefficient of E_{i+1}^(k+1)
                                    
                                    # Final safety check
                                    if np.isfinite(coeff_nl):
                                        A_tri[1, i] -= coeff_nl   # coefficient of E_i^(k+1)
                                    A_tri[2, i] -= coeff_nl   # coefficient of E_{i+1}^(k+1)
                
                except Exception as e:
                    # If nonlinear matrix correction fails, skip it for this cell
                    if hasattr(self, '_nl_matrix_error_count'):
                        self._nl_matrix_error_count += 1
                    else:
                        self._nl_matrix_error_count = 1
                        print(f"Warning: Nonlinear matrix correction error at cell {i}: {e}")
                    # Continue to RHS corrections
                
                # N^(k)[E_r^(k)] contributions to RHS with enhanced safety
                try:
                        # Left face contribution
                        if i > 0:
                            Er_face = interpolate_face_value(Er_k[i-1], Er_k[i], 
                                                           self.r_faces[i], self.r_centers[i-1], 
                                                           self.r_centers[i], self.r_faces[i+1])
                            grad_Er = (Er_k[i] - Er_k[i-1]) / dr_left
                            
                            # Safety checks
                            if np.isfinite(Er_face) and np.isfinite(grad_Er) and np.isfinite(DE_faces[i]):
                                flux_nl = -DE_faces[i] * Er_face * grad_Er
                                
                                # Limit flux magnitude relative to linear diffusive flux
                                linear_flux = -D_faces[i] * grad_Er if np.isfinite(D_faces[i]) else 0.0
                                max_nl_flux = self.nonlinear_limiter * abs(linear_flux) if linear_flux != 0 else abs(flux_nl) * 0.1
                                
                                if abs(flux_nl) > max_nl_flux:
                                    flux_nl = np.sign(flux_nl) * max_nl_flux
                            
                            # Final safety check and application
                            if np.isfinite(flux_nl):
                                rhs[i] += A_left * flux_nl / V_i
                
                        # Right face contribution with same safety checks
                        if i < n_cells - 1:
                            Er_face = interpolate_face_value(Er_k[i], Er_k[i+1],
                                                           self.r_faces[i+1], self.r_centers[i], 
                                                           self.r_centers[i+1], self.r_faces[i+2] if i+1 < n_cells-1 else self.r_faces[i+1])
                            grad_Er = (Er_k[i+1] - Er_k[i]) / dr_right
                            
                            # Safety checks
                            if np.isfinite(Er_face) and np.isfinite(grad_Er) and np.isfinite(DE_faces[i+1]):
                                flux_nl = -DE_faces[i+1] * Er_face * grad_Er
                                
                                # Limit flux magnitude relative to linear diffusive flux
                                linear_flux = -D_faces[i+1] * grad_Er if np.isfinite(D_faces[i+1]) else 0.0
                                max_nl_flux = self.nonlinear_limiter * abs(linear_flux) if linear_flux != 0 else abs(flux_nl) * 0.1
                                
                                if abs(flux_nl) > max_nl_flux:
                                    flux_nl = np.sign(flux_nl) * max_nl_flux
                                
                                # Final safety check and application
                                if np.isfinite(flux_nl):
                                    rhs[i] -= A_right * flux_nl / V_i
                
                except Exception as e:
                    # If nonlinear correction fails, skip it for this cell
                    if hasattr(self, '_nl_error_count'):
                        self._nl_error_count += 1
                    else:
                        self._nl_error_count = 1
                        print(f"Warning: Nonlinear correction error at cell {i}: {e}")
                    continue
        
        # Final matrix validation
        if np.any(~np.isfinite(A_tri)) or np.any(~np.isfinite(rhs)):
            print("Warning: Matrix contains invalid values after assembly")
            # Clean up any remaining invalid values
            A_tri[~np.isfinite(A_tri)] = 0.0
            rhs[~np.isfinite(rhs)] = 0.0
        
        return A_tri, rhs
    
    def apply_boundary_conditions(self, A_tri, rhs, Er_k, use_nonlinear_correction=False):
        """Apply Robin boundary conditions by modifying the boundary equations
        
        With nonlinear correction, the boundary flux includes γ^(k) term:
        F = D * [C + γ^(k)*(∂E_r^(k)/∂r)*(E_r^(k+1) - E_r^(k)) - A*E_r^(k+1)] / B
        """
        n_cells = len(Er_k)
        
        # Left boundary (i = 0)
        A_bc_left, B_bc_left, C_bc_left = self.left_bc_func(Er_k[0], self.r_faces[0])
        
        if abs(B_bc_left) < 1e-14:
            # Dirichlet BC: Implement via numerical flux instead of overwriting equation
            # Ghost value at boundary: Er_{1/2} = C_bc / A_bc
            # Flux: F = D * (Er_0 - Er_{1/2}) / (Δx/2)
            # This preserves the equation structure and allows nonlinear corrections
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
            # General Robin BC with optional nonlinear correction
            D_boundary = self.get_diffusion_coefficient(Er_k[0])
            
            if use_nonlinear_correction and n_cells > 1:
                # Include nonlinear correction γ^(k) = D_E * ∂E_r^(k)/∂r
                D_E_boundary = self.get_diffusion_coefficient_derivative(Er_k[0])
                dr = self.r_centers[0] - self.r_faces[0]
                if n_cells > 1:
                    # Approximate gradient at boundary face
                    grad_Er_k = (Er_k[0] - Er_k[0]) / dr  # Zero at boundary face for now
                    # Better: use one-sided difference
                    dr_neighbor = self.r_centers[1] - self.r_centers[0]
                    grad_Er_k = (Er_k[1] - Er_k[0]) / dr_neighbor
                else:
                    grad_Er_k = 0.0
                
                gamma_k = D_E_boundary * grad_Er_k
                
                # Modified Robin BC: flux includes γ^(k) term
                # F = D * [C + γ^(k)*(∂E_r^(k)/∂r)*(E_r^(k+1) - E_r^(k)) - A*E_r^(k+1)] / B
                flux_coeff = (self.A_faces[0] * D_boundary * A_bc_left) / (B_bc_left * self.V_cells[0])
                gamma_coeff = (self.A_faces[0] * D_boundary * gamma_k) / (B_bc_left * self.V_cells[0])
                
                A_tri[1, 0] += flux_coeff - gamma_coeff
                rhs[0] += self.A_faces[0] * D_boundary * (C_bc_left - gamma_k * Er_k[0]) / (B_bc_left * self.V_cells[0])
            else:
                # Standard Robin BC without nonlinear correction
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
            # General Robin BC with optional nonlinear correction
            D_boundary = self.get_diffusion_coefficient(Er_k[-1])
            
            if use_nonlinear_correction and n_cells > 1:
                D_E_boundary = self.get_diffusion_coefficient_derivative(Er_k[-1])
                # Approximate gradient at right boundary
                dr_neighbor = self.r_centers[-1] - self.r_centers[-2]
                grad_Er_k = (Er_k[-1] - Er_k[-2]) / dr_neighbor
                
                gamma_k = D_E_boundary * grad_Er_k
                
                flux_coeff = (self.A_faces[-1] * D_boundary * A_bc_right) / (B_bc_right * self.V_cells[-1])
                gamma_coeff = (self.A_faces[-1] * D_boundary * gamma_k) / (B_bc_right * self.V_cells[-1])
                
                A_tri[1, -1] += flux_coeff - gamma_coeff
                rhs[-1] += self.A_faces[-1] * D_boundary * (C_bc_right - gamma_k * Er_k[-1]) / (B_bc_right * self.V_cells[-1])
            else:
                # Standard Robin BC without nonlinear correction
                flux_coeff = (self.A_faces[-1] * D_boundary * A_bc_right) / (B_bc_right * self.V_cells[-1])
                A_tri[1, -1] += flux_coeff
                rhs[-1] += self.A_faces[-1] * D_boundary * C_bc_right / (B_bc_right * self.V_cells[-1])


# =============================================================================
# TEST PROBLEMS
# =============================================================================

def manufactured_solution_test():
    """
    Manufactured solution test with known analytical solution
    
    For constant D and linear source, steady solution is quadratic
    """
    print("Running manufactured solution test...")
    
    # Define custom material properties for testing
    def custom_opacity(Er):
        """Constant opacity for simple test case"""
        return 100  # cm⁻¹
    
    def custom_specific_heat(T):
        """Constant specific heat (per unit mass)"""
        return CV_CONST
    
    def custom_material_energy(T):
        """Linear material energy density: e(T) = ρ*cv*T"""
        return RHO * custom_specific_heat(T) * T
    
    # Define custom boundary conditions
    def left_bc(Er, x):
        """Left boundary: blackbody at 1 keV (T=1 keV -> Er = a*T^4)"""
        sigmaR = custom_opacity(Er)
        T_bc = 1.0  # keV
        Er_bc = A_RAD * T_bc**4
        return 1.0, 1/(3*sigmaR), Er_bc  # A, B, C for Robin BC: A*Er + B*(n·∇Er) = C
    
    def right_bc(Er, x):
        """Right boundary: low energy E_r (T ≈ 0.316 keV -> Er = a*T^4)"""
        sigmaR = custom_opacity(Er)
        T_bc = 0.316  # keV (corresponds to Er ≈ 0.001*A_RAD)
        Er_bc = A_RAD * T_bc**4
        return 1.0, 1/(3*sigmaR), Er_bc  # A, B, C
    
    # Problem setup with custom material functions and boundary conditions
    solver = RadiationDiffusionSolver(
        r_min=0.1, r_max=1.0, n_cells=100, d=0,  # planar geometry
        dt=1e-5, max_newton_iter=5, newton_tol=1e-10,
        rosseland_opacity_func=custom_opacity,
        specific_heat_func=custom_specific_heat,
        material_energy_func=custom_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    solver.use_nonlinear_correction = False  # Disable nonlinear correction for this test
    solver.use_secant_derivative = False     # Use finite difference for D_E

    # Set initial condition (linear profile)
    def initial_Er(r):
        return r*0 + 1e-1 +(r>0.25)*(r<0.75)*1.0 # initial bump in the center
    
    solver.set_initial_condition(initial_Er)
    
    # Run to steady state
    print("Time stepping to steady state...")
    solver.time_step(n_steps=2, verbose=False)
    
    return solver


def transient_test():
    """Transient test problem"""
    print("Running transient test...")
    
    # Example with nonlinear material properties
    def nonlinear_opacity(Er):
        """Temperature-dependent opacity"""
        T = temperature_from_Er(Er)
        return 10 + 0.0 * T  # Opacity increases with temperature
    
    def temperature_dependent_cv(T):
        """Temperature-dependent specific heat"""
        return CV_CONST * (1.0 + 0.0 * T)  # Slightly increasing with T
    
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=0.5, n_cells=50, d=1,  # Cylindrical geometry
        dt=1e-3, max_newton_iter=3, newton_tol=1e-8,
        rosseland_opacity_func=nonlinear_opacity,
        specific_heat_func=temperature_dependent_cv
        # Use default material_energy_func
    )
    
    # Initial condition: Gaussian-like profile
    def initial_Er(r):
        r0 = 0.5 * (solver.r_min + solver.r_max)
        sigma = 0.1
        return 1.0 + 0.5 * np.exp(-((r - r0) / sigma)**2)
    
    solver.set_initial_condition(initial_Er)
    
    # Time evolution
    print("Time evolution...")
    n_snapshots = 50
    solutions = []
    
    for i in range(n_snapshots):
        solver.time_step(n_steps=20, verbose=False)
        r, Er = solver.get_solution()
        solutions.append((i * 20 * solver.dt, r.copy(), Er.copy()))
    
    return solutions


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_solution(solver, title="Radiation Energy Density"):
    """Plot current solution"""
    r, Er = solver.get_solution()
    
    plt.figure(figsize=(10, 6))
    plt.plot(r, Er, 'b-', linewidth=2, label='E_r')
    plt.xlabel('Radius r')
    plt.ylabel('Radiation Energy Density E_r')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_transient_solutions(solutions):
    """Plot multiple time snapshots"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
    
    for i, (t, r, Er) in enumerate(solutions):
        plt.plot(r, Er, color=colors[i], linewidth=2, 
                label=f't = {t:.4f}')
    
    plt.xlabel('Radius r')
    plt.ylabel('Radiation Energy Density E_r')
    plt.title('Transient Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_convergence_study():
    """Study Newton convergence and mesh refinement"""
    print("Convergence study...")
    
    # Mesh refinement study
    n_cells_list = [25, 50, 100, 200]
    errors = []
    
    # Reference solution (finest mesh)
    solver_ref = RadiationDiffusionSolver(n_cells=400, dt=1e-3)
    solver_ref.set_initial_condition(1.0)
    solver_ref.time_step(n_steps=50, verbose=False)
    r_ref, Er_ref = solver_ref.get_solution()
    
    for n_cells in n_cells_list:
        solver = RadiationDiffusionSolver(n_cells=n_cells, dt=1e-3)
        solver.set_initial_condition(1.0)  
        solver.time_step(n_steps=50, verbose=False)
        r, Er = solver.get_solution()
        
        # Interpolate reference solution to current mesh
        Er_ref_interp = np.interp(r, r_ref, Er_ref)
        error = np.linalg.norm(Er - Er_ref_interp) / np.linalg.norm(Er_ref_interp)
        errors.append(error)
        print(f"  n_cells={n_cells:3d}, error={error:.2e}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(n_cells_list, errors, 'o-', linewidth=2, markersize=8)
    plt.loglog(n_cells_list, 1e-2 * np.array(n_cells_list,dtype=float)**(-2), '--', 
               label='2nd order', alpha=0.7)
    plt.xlabel('Number of cells')
    plt.ylabel('Relative error')
    plt.title('Mesh Convergence Study')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("1D Radiation Diffusion Finite Volume Solver")
    print("="*60)
    
    # Test 1: Manufactured solution (steady state)
    print("\n" + "="*40)
    print("Test 1: Manufactured Solution")
    print("="*40)
    
    solver1 = manufactured_solution_test()
    plot_solution(solver1, "Manufactured Solution Test (Steady State)")
    
    # Test 2: Transient evolution  
    print("\n" + "="*40)
    print("Test 2: Transient Evolution")
    print("="*40)
    
    solutions = transient_test()
    plot_transient_solutions(solutions)
    
    # Test 3: Convergence study
    print("\n" + "="*40)  
    print("Test 3: Convergence Study")
    print("="*40)
    
    plot_convergence_study()
    
    # Configuration options demonstration
    print("\n" + "="*40)
    print("Configuration Options:")
    print("="*40)
    print("To toggle approximations, modify solver attributes:")
    print("- solver.use_nonlinear_correction = False  # Drop N^(k) term")
    print("- solver.use_secant_derivative = False     # Use finite differences for D_E")  
    print("- solver.max_newton_iter_per_step = 1      # One-step method (k=0 only)")
    print("\nTo modify material properties, edit functions:")
    print("- rosseland_opacity(Er)")
    print("- specific_heat_cv(T)")
    print("- robin_bc_coefficients(Er, x, is_left)")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

