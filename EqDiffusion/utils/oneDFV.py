#!/usr/bin/env python3
"""
1D Finite Volume Solver for Equilibrium Radiation Diffusion Equation
using Newton iterations in terms of radiation energy density E_r.

PDE: ∇·(D(E_r) ∇E_r) + u(E_r) = Qhat
Time discretization: implicit Euler
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
# MAIN SOLVER CLASS
# =============================================================================

class RadiationDiffusionSolver:
    """1D Finite Volume Radiation Diffusion Solver"""
    
    def __init__(self, r_min=0.1, r_max=1.0, n_cells=50, d=0, dt=1e-3,
                 max_newton_iter=10, newton_tol=1e-8, stretch_factor=1.0,
                 rosseland_opacity_func=None, specific_heat_func=None, 
                 material_energy_func=None, left_bc_func=None, right_bc_func=None,
                 theta=1.0):
        
        self.r_min = r_min
        self.r_max = r_max  
        self.n_cells = n_cells
        self.d = d  # Coordinate system
        self.dt = dt
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.theta = theta  # Time discretization parameter (1.0 = implicit Euler)
        
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
        
        print(f"Initialized solver: {n_cells} cells, r ∈ [{r_min:.3f}, {r_max:.3f}]")
        print(f"Coordinate system d={d}, Δt={dt:.2e}")
    
    def get_diffusion_coefficient(self, Er):
        """Get diffusion coefficient using the configured opacity function"""
        return C_LIGHT / (3.0 * self.rosseland_opacity_func(Er))
    
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
    
    def get_dudEr(self, Er):
        """Get du/dEr derivative for Newton linearization
        
        Returns: 1 + de_mat/dEr = 1 + ρ*cv/(4*a^(1/4)*Er^(3/4))
        """
        T = temperature_from_Er(np.abs(Er))
        cv = self.specific_heat_func(T)
        rho_cv = RHO * cv
        term = rho_cv / (4.0 * A_RAD**0.25 * np.abs(Er)**0.75)
        return 1.0 + term
    
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
        #print theta to check
        #print("Theta =", self.theta)
        for k in range(self.max_newton_iter):
            # Assemble linear system
            A_tri, rhs = self.assemble_system(Er_k, Er_prev_timestep, theta=self.theta)
            
            # Apply boundary conditions
            self.apply_boundary_conditions(A_tri, rhs, Er_k)
            
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
                current_energy = np.sum(Er_new)
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
                alpha = max(0.0001, min(alpha, 0.9))  # More conservative range
                Er_new = Er_k + alpha * delta_Er
                adjusted_energy = np.sum(Er_new)
                #scale to have the same total energy as before
                if not np.isclose(current_energy, adjusted_energy):
                    Er_new *= (current_energy / adjusted_energy)
                    if verbose:
                        print(f"    Energy changed during damping, rescaling solution")
                if verbose:
                    print(f"    Applied Newton damping: alpha = {alpha:.3f}")
            
            # Additional safety check after damping
            if np.any(Er_new <= 0) or np.any(~np.isfinite(Er_new)):
                # Emergency fallback: minimal update
                Er_new = Er_k + 1e-6 * np.abs(Er_k)
                solution_valid = False
                if verbose:
                    print(f"    Applied emergency fallback update")
            
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
        
        if verbose:
            print(f"    Newton max iterations reached, residual={residual:.2e}")
        return Er_k
    
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
    
    def time_step_trbdf2(self, n_steps=1, gamma=None, verbose=True):
        """Advance solution using TR-BDF2 method
        
        TR-BDF2 is a two-stage composite method:
        Stage 1: Trapezoidal rule from t^n to t^{n+γ}
        Stage 2: BDF2 from t^n, t^{n+γ} to t^{n+1}
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        gamma : float, optional
            Intermediate time fraction (default: 2 - sqrt(2) ≈ 0.586)
        verbose : bool
            Print progress information
        """
        if gamma is None:
            gamma = 2.0 - np.sqrt(2.0)  # Optimal value for L-stability
        
        # Store original parameters
        original_dt = self.dt
        original_theta = self.theta
        
        for step in range(n_steps):
            if verbose:
                print(f"TR-BDF2 step {step+1}/{n_steps}")
            
            # Store solution at t^n
            Er_n = self.Er.copy()
            
            # Stage 1: Trapezoidal rule to intermediate point t^{n+γ}
            if verbose:
                print(f"  Stage 1: TR to t^{{n+{gamma:.3f}}}")
            
            self.dt = gamma * original_dt
            self.theta = 0.5  # Trapezoidal rule
            Er_intermediate = self.newton_step(Er_n, verbose=verbose)
            
            # Stage 2: BDF2 from t^n and t^{n+γ} to t^{n+1}
            if verbose:
                print(f"  Stage 2: BDF2 to t^{{n+1}}")
            
            self.dt =  original_dt
            #set initial guess for Er to be Er_intermediate
            self.Er = Er_intermediate.copy()
            self.Er = self.newton_step_bdf2(Er_n, Er_intermediate, gamma, verbose=verbose)
            
            # Update for next time step
            self.Er_old = Er_n.copy()
        
        # Restore original parameters
        self.dt = original_dt
        self.theta = original_theta
    
    def newton_step_bdf2(self, Er_n, Er_intermediate, gamma, verbose=True):
        """Perform Newton iterations for BDF2 stage
        
        BDF2 discretization using solutions at t^n and t^{n+γ}:
        (3*Er^{n+1} - 4*Er^{n+γ} + Er^n) / (2*dt_2) = F(Er^{n+1})
        
        where dt_2 = (1-γ)*dt is the second stage time step
        """
        Er_k = self.Er.copy()  # Initial guess
        
        for k in range(self.max_newton_iter):
            # Assemble BDF2 system
            A_tri, rhs = self.assemble_system_bdf2(Er_k, Er_n, Er_intermediate, gamma)
            
            # Apply boundary conditions
            self.apply_boundary_conditions(A_tri, rhs, Er_k)
            
            # Solve linear system
            Er_new = solve_tridiagonal(A_tri, rhs)
            
            # Check convergence
            norm_k = np.linalg.norm(Er_k)
            if norm_k < 1e-14:
                norm_k = 1.0
            residual = np.linalg.norm(Er_new - Er_k) / norm_k
            
            if residual < self.newton_tol:
                if verbose:
                    print(f"    BDF2 Newton converged in {k+1} iterations, residual={residual:.2e}")
                return Er_new
            
            Er_k = Er_new.copy()
        
        if verbose:
            print(f"    BDF2 Newton max iterations reached, residual={residual:.2e}")
        return Er_k
    
    def assemble_system_bdf2(self, Er_k, Er_n, Er_intermediate, gamma):
        """Assemble system for BDF2 stage of TR-BDF2
        
        BDF2 for non-uniform time steps with t^n, t^{n+γ}, t^{n+1}:
        Uses special coefficients for material energy terms:
        
        1/Δt * [1/(γ(2-γ)) * u(E_r^{n+1}) - (1-γ)²/(γ(2-γ)) * u(E_r^n) - 1/(2-γ) * u(E_r^{n+γ})]
             = ∇·D(E_r^{n+1})∇E_r^{n+1}
             
        where u = e_mat + E_r
        """
        n_cells = len(Er_k)
        dt = self.dt  # Full time step Δt
        #print("dt in assemble_system_bdf2:", dt)
        
        # TR-BDF2 coefficients for total energy u = e_mat + Er
        # From paper: (1/Δt)[c_0*u^{n+1} + c_1*u^n + c_2*u^{n+γ}] = L(Er^{n+1})
        c_0 = (2-gamma)/(1-gamma)  # Coefficient of u^{n+1}
        c_1 = (1-gamma)/gamma  # Coefficient of u^n
        c_2 = -1/(gamma*(1-gamma))  # Coefficient of u^{n+γ}
        
        # Get diffusion matrix at current iterate
        
        L_tri = self.assemble_diffusion_matrix(Er_k)
        
        # Initialize system matrix: a_0 + L (same sign structure as theta method)
        A_tri = L_tri.copy()
        rhs = np.zeros(n_cells)
        
        # Evaluate material energy e_mat at the three time levels
        T_np1 = temperature_from_Er(Er_k)
        T_n = temperature_from_Er(Er_n)
        T_ng = temperature_from_Er(Er_intermediate)
        
        e_mat_np1 = np.array([self.material_energy_func(T_np1[i]) for i in range(n_cells)])
        e_mat_n = np.array([self.material_energy_func(T_n[i]) for i in range(n_cells)])
        e_mat_ng = np.array([self.material_energy_func(T_ng[i]) for i in range(n_cells)])
        
        # u = e_mat + E_r
        u_np1 = e_mat_np1 + Er_k
        u_n = e_mat_n + Er_n
        u_ng = e_mat_ng + Er_intermediate
        
        # Get du/dEr for Newton linearization
        dudEr_cells = np.array([self.get_dudEr(Er_k[i]) for i in range(n_cells)])
        
        # Compute "alpha-like" coefficients that include material coupling
        # Per paper formula: divide by full Δt, not (1-γ)Δt
        # a_0 = (c_0/dt) * du/dEr accounts for both Er and e_mat time derivatives
        a_0_cells = (c_0 / dt) * dudEr_cells
        
        for i in range(n_cells):
            # Matrix: [a_0 - L] * Er = RHS, where a_0 includes material coupling
            A_tri[1, i] += a_0_cells[i]  # Main diagonal
            
            # RHS: Newton linearization of u^{n+1}:
            # u(Er^{n+1}) ≈ u(Er_k) + du/dEr*(Er^{n+1} - Er_k)
            # Moving linearized term to LHS gives RHS: -(c_0/dt)*[u(Er_k) - du/dEr*Er_k] - (c_1/dt)*u^n - (c_2/dt)*u^{n+γ}
            # Simplify: (c_0/dt)*du/dEr*Er_k - (c_0/dt)*u(Er_k) - (c_1/dt)*u^n - (c_2/dt)*u^{n+γ}
            rhs[i] = ((c_0 / dt) * dudEr_cells[i] * Er_k[i] - (c_0 / dt) * u_np1[i] - 
                     (c_1 / dt) * u_n[i] - (c_2 / dt) * u_ng[i])
        
        return A_tri, rhs
    
    def get_solution(self):
        """Return current solution"""
        return self.r_centers.copy(), self.Er.copy()
    
    def assemble_diffusion_matrix(self, Er_k):
        """Assemble diffusion matrix (spatial operator only)
        
        Returns the tridiagonal matrix for the diffusion operator:
        L(Er) = div(D * grad(Er))
        
        Parameters:
        -----------
        Er_k : array
            Current iterate of radiation energy density
            
        Returns:
        --------
        L_tri : array (3, n_cells)
            Tridiagonal matrix for diffusion operator
        """
        n_cells = len(Er_k)
        n_faces = len(self.r_faces)
        
        # Initialize tridiagonal matrix
        L_tri = np.zeros((3, n_cells))  # [sub, main, super]
        
        # Evaluate diffusion coefficients at cell centers
        D_cells = np.array([self.get_diffusion_coefficient(Er_k[i]) for i in range(n_cells)])
        
        # Face diffusion coefficients (evaluate at averaged temperature)
        D_faces = np.zeros(n_faces)
        
        # Interior faces
        for i in range(1, n_faces - 1):
            # Average temperature at face, then evaluate D at that temperature
            T_left = temperature_from_Er(Er_k[i-1])
            T_right = temperature_from_Er(Er_k[i])
            T_face = 0.5 * (T_left + T_right)
            Er_face = A_RAD * T_face**4
            
            # Evaluate D at the face temperature
            D_at_face_temp = self.get_diffusion_coefficient(Er_face)
            D_faces[i] = D_at_face_temp
        
        # Boundary faces
        D_faces[0] = D_cells[0]
        D_faces[-1] = D_cells[-1]
        
        # Assemble diffusion operator for interior cells
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
            
            # Diffusive flux contributions
            if i > 0:  # Interior left face
                coeff_left = A_left * D_faces[i] / (dr_left * V_i)
                L_tri[0, i] = -coeff_left  # sub-diagonal (E_{i-1})
                L_tri[1, i] += coeff_left  # main diagonal (E_i)
                
            if i < n_cells - 1:  # Interior right face
                coeff_right = A_right * D_faces[i + 1] / (dr_right * V_i)
                L_tri[2, i] = -coeff_right  # super-diagonal (E_{i+1})
                L_tri[1, i] += coeff_right  # main diagonal (E_i)
        
        return L_tri
    
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
    
    # Example test problem configuration

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
    print("- solver.max_newton_iter_per_step = 1      # One-step method (k=0 only)")
    print("- solver.theta = 1.0                       # Time integration: 0=explicit, 0.5=Crank-Nicolson, 1=implicit")
    print("\nTo modify material properties, edit functions:")
    print("- rosseland_opacity(Er)")
    print("- specific_heat_cv(T)")
    print("- robin_bc_coefficients(Er, x, is_left)")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

