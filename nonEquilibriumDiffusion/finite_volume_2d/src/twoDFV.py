#!/usr/bin/env python3
"""
2D Finite Volume Solver for Non-Equilibrium Radiation Diffusion Equation
using Newton iterations with coupled φ(x,y,t) and T(x,y,t) variables.

Supports two geometries:
  - Cartesian (x-y): Standard 2D Cartesian coordinates
  - Cylindrical (r-z): Axisymmetric cylindrical coordinates

Variables:
  φ(x,y,t) or φ(r,z,t) = E_r * c   (radiation energy density × speed of light)
  T(x,y,t) or T(r,z,t)              (material temperature)

Coupled PDEs:
  (1/c) ∂φ/∂t - ∇·(D∇φ) = σ_P·f(acT_★^4 - φ̃) - (1-f)·Δe/Δt
  ∂e/∂t = f·σ_P(φ̃ - acT_★^4) + (1-f)·Δe/Δt
  
Time discretization: θ-method or TR-BDF2
Newton method for nonlinear coupling
Sparse matrix solver for 2D diffusion operator
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
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
def specific_heat_cv(T, x, y):
    """Specific heat capacity c_v(T, x, y). Default: constant.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm)
    
    Returns:
    --------
    c_v : float
        Specific heat capacity (erg/g/keV)
    """
    return CV_CONST


@njit
def material_energy_density(T, x, y):
    """Material energy density e(T, x, y) = ρ c_v(T, x, y) T
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm)
    
    Returns:
    --------
    e : float
        Material energy density (erg/cm³)
    """
    return RHO * specific_heat_cv(T, x, y) * T


def inverse_material_energy_density(e, x, y):
    """Inverse: T from e. Default assumes e = ρ·c_v·T => T = e/(ρ·c_v)
    
    Parameters:
    -----------
    e : float
        Material energy density (erg/cm³)
    x, y : float
        Spatial coordinates (cm)
    
    Returns:
    --------
    T : float
        Temperature (keV)
    """
    return e / (RHO * CV_CONST)


@njit
def planck_opacity(T, x, y):
    """Planck mean opacity σ_P(T, x, y). Default: constant for testing.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm)
    
    Returns:
    --------
    sigma_P : float
        Planck opacity (cm⁻¹)
    """
    return 1.0  # cm⁻¹


@njit
def rosseland_opacity(T, x, y):
    """Rosseland opacity σ_R(T, x, y). Default: constant for testing.
    
    Parameters:
    -----------
    T : float
        Temperature (keV)
    x, y : float
        Spatial coordinates (cm)
    
    Returns:
    --------
    sigma_R : float
        Rosseland opacity (cm⁻¹)
    """
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


@njit
def _apply_flux_limiter_standard(R):
    """Numba-compiled standard flux limiter"""
    if R < 1e-10:
        return 1.0 / 3.0
    coth_R = 1.0 / np.tanh(R)
    return coth_R / R - 1.0 / (R * R)


@njit
def _apply_flux_limiter_lp(R):
    """Numba-compiled Levermore-Pomraning flux limiter"""
    numerator = 2.0 + R
    denominator = 6.0 + 3.0*R + R*R
    return numerator / denominator


@njit
def _apply_flux_limiter_larsen(R, n=2.0):
    """Numba-compiled Larsen flux limiter"""
    return (3.0**n + R**n)**(-1.0/n)


@njit
def _apply_flux_limiter_sum(R):
    """Numba-compiled sum flux limiter"""
    return 1.0 / (3.0 + R)


@njit
def _apply_flux_limiter_max(R):
    """Numba-compiled max flux limiter"""
    return 1.0 / max(3.0, R)


def compute_face_diffusion_coefficients_x(
    nx, ny, T_star_2d, phi_star_2d, sigma_R_2d,
    x_centers, y_centers, x_faces, rosseland_opacity_func,
    boundary_funcs, current_time, A_RAD, C_LIGHT,
    flux_limiter_type=0, flux_limiter_n=2.0
):
    """Compute diffusion coefficients at x-direction faces
    
    Parameters:
    -----------
    nx, ny : int
        Number of cells in x and y directions
    T_star_2d : ndarray (nx, ny)
        Temperature at cell centers
    phi_star_2d : ndarray (nx, ny)
        Phi at cell centers
    sigma_R_2d : ndarray (nx, ny)
        Rosseland opacity at cell centers (used only for initial estimate)
    x_centers, y_centers : ndarray
        Cell center coordinates
    x_faces : ndarray
        Face coordinates in x direction
    rosseland_opacity_func : callable
        Function to evaluate Rosseland opacity at face (T, x, y)
    boundary_funcs : dict
        Dictionary of boundary functions
    current_time : float
        Current simulation time
    A_RAD, C_LIGHT : float
        Radiation constants
    flux_limiter_type : int
        0=standard, 1=LP, 2=Larsen, 3=sum, 4=max
    flux_limiter_n : float
        Parameter for Larsen flux limiter
    
    Returns:
    --------
    D_x_faces : ndarray (nx+1, ny)
        Diffusion coefficients at x-direction faces
    """
    D_x_faces = np.zeros((nx + 1, ny))
    
    for i in range(nx + 1):
        for j in range(ny):
            if i == 0:
                # Left boundary - average cell and boundary temperatures
                T_cell = T_star_2d[0, j]
                phi_cell = phi_star_2d[0, j]
                pos = (x_faces[0], y_centers[j])
                A_bc, B_bc, C_bc = boundary_funcs['left'](phi_cell, pos, current_time)
                
                if abs(B_bc) < 1e-14:  # Dirichlet
                    phi_boundary = C_bc / A_bc
                    T_boundary = (phi_boundary / (A_RAD * C_LIGHT))**0.25
                    T_avg = 0.5 * (T_cell + T_boundary)
                else:  # Robin or Neumann - use cell temperature
                    T_avg = T_cell
                
                sigma_R_face = rosseland_opacity_func(T_avg, x_faces[0], y_centers[j])
                D_x_faces[i, j] = 1.0 / (3.0 * sigma_R_face)
            elif i == nx:
                # Right boundary - average cell and boundary temperatures
                T_cell = T_star_2d[nx-1, j]
                phi_cell = phi_star_2d[nx-1, j]
                pos = (x_faces[-1], y_centers[j])
                A_bc, B_bc, C_bc = boundary_funcs['right'](phi_cell, pos, current_time)
                
                if abs(B_bc) < 1e-14:  # Dirichlet
                    phi_boundary = C_bc / A_bc
                    T_boundary = (phi_boundary / (A_RAD * C_LIGHT))**0.25
                    T_avg = 0.5 * (T_cell + T_boundary)
                else:  # Robin or Neumann - use cell temperature
                    T_avg = T_cell
                
                sigma_R_face = rosseland_opacity_func(T_avg, x_faces[-1], y_centers[j])
                D_x_faces[i, j] = 1.0 / (3.0 * sigma_R_face)
            else:
                # Interior face - harmonic mean with flux limiter
                T_left = T_star_2d[i-1, j]
                T_right = T_star_2d[i, j]
                phi_left = phi_star_2d[i-1, j]
                phi_right = phi_star_2d[i, j]
                
                # Average temperature for opacity evaluation
                T_avg = 0.5 * (T_left + T_right)
                
                # Evaluate opacity at FACE locations using average temperature
                x_left = x_centers[i-1]
                x_right = x_centers[i]
                x_face = x_faces[i]
                y_face = y_centers[j]
                sigma_R_left = rosseland_opacity_func(T_avg, x_left, y_face)
                sigma_R_right = rosseland_opacity_func(T_avg, x_right, y_face)
                
                # Compute flux limiter parameters
                dx_left = x_face - x_left
                dx_right = x_right - x_face
                dx_total = dx_left + dx_right
                
                grad_phi_mag = abs(phi_right - phi_left) / dx_total
                phi_avg = 0.5 * (phi_left + phi_right)
                
                # Compute diffusion coefficients on left and right
                if phi_avg < 1e-30:
                    D_left = 1.0 / (3.0 * sigma_R_left)
                    D_right = 1.0 / (3.0 * sigma_R_right)
                else:
                    R_left = grad_phi_mag / (sigma_R_left * phi_avg)
                    R_right = grad_phi_mag / (sigma_R_right * phi_avg)
                    
                    # Apply flux limiter
                    if flux_limiter_type == 0:
                        lambda_left = _apply_flux_limiter_standard(R_left)
                        lambda_right = _apply_flux_limiter_standard(R_right)
                    elif flux_limiter_type == 1:
                        lambda_left = _apply_flux_limiter_lp(R_left)
                        lambda_right = _apply_flux_limiter_lp(R_right)
                    elif flux_limiter_type == 2:
                        lambda_left = _apply_flux_limiter_larsen(R_left, flux_limiter_n)
                        lambda_right = _apply_flux_limiter_larsen(R_right, flux_limiter_n)
                    elif flux_limiter_type == 3:
                        lambda_left = _apply_flux_limiter_sum(R_left)
                        lambda_right = _apply_flux_limiter_sum(R_right)
                    else:  # flux_limiter_type == 4
                        lambda_left = _apply_flux_limiter_max(R_left)
                        lambda_right = _apply_flux_limiter_max(R_right)
                    
                    D_left = lambda_left / sigma_R_left
                    D_right = lambda_right / sigma_R_right
                
                # Harmonic mean with safeguard
                denom = dx_left/max(D_left, 1e-30) + dx_right/max(D_right, 1e-30)
                D_x_faces[i, j] = dx_total / max(denom, 1e-30)
    
    return D_x_faces


def compute_face_diffusion_coefficients_y(
    nx, ny, T_star_2d, phi_star_2d, sigma_R_2d,
    x_centers, y_centers, y_faces, rosseland_opacity_func,
    boundary_funcs, current_time, A_RAD, C_LIGHT,
    flux_limiter_type=0, flux_limiter_n=2.0
):
    """Compute diffusion coefficients at y-direction faces
    
    Parameters:
    -----------
    nx, ny : int
        Number of cells in x and y directions
    T_star_2d : ndarray (nx, ny)
        Temperature at cell centers
    phi_star_2d : ndarray (nx, ny)
        Phi at cell centers
    sigma_R_2d : ndarray (nx, ny)
        Rosseland opacity at cell centers (used only for initial estimate)
    x_centers, y_centers : ndarray
        Cell center coordinates
    y_faces : ndarray
        Face coordinates in y direction
    rosseland_opacity_func : callable
        Function to evaluate Rosseland opacity at face (T, x, y)
    boundary_funcs : dict
        Dictionary of boundary functions
    current_time : float
        Current simulation time
    A_RAD, C_LIGHT : float
        Radiation constants
    flux_limiter_type : int
        0=standard, 1=LP, 2=Larsen, 3=sum, 4=max
    flux_limiter_n : float
        Parameter for Larsen flux limiter
    
    Returns:
    --------
    D_y_faces : ndarray (nx, ny+1)
        Diffusion coefficients at y-direction faces
    """
    D_y_faces = np.zeros((nx, ny + 1))
    
    for i in range(nx):
        for j in range(ny + 1):
            if j == 0:
                # Bottom boundary - average cell and boundary temperatures
                T_cell = T_star_2d[i, 0]
                phi_cell = phi_star_2d[i, 0]
                pos = (x_centers[i], y_faces[0])
                A_bc, B_bc, C_bc = boundary_funcs['bottom'](phi_cell, pos, current_time)
                
                if abs(B_bc) < 1e-14:  # Dirichlet
                    phi_boundary = C_bc / A_bc
                    T_boundary = (phi_boundary / (A_RAD * C_LIGHT))**0.25
                    T_avg = 0.5 * (T_cell + T_boundary)
                else:  # Robin or Neumann - use cell temperature
                    T_avg = T_cell
                
                sigma_R_face = rosseland_opacity_func(T_avg, x_centers[i], y_faces[0])
                D_y_faces[i, j] = 1.0 / (3.0 * sigma_R_face)
            elif j == ny:
                # Top boundary - average cell and boundary temperatures
                T_cell = T_star_2d[i, ny-1]
                phi_cell = phi_star_2d[i, ny-1]
                pos = (x_centers[i], y_faces[-1])
                A_bc, B_bc, C_bc = boundary_funcs['top'](phi_cell, pos, current_time)
                
                if abs(B_bc) < 1e-14:  # Dirichlet
                    phi_boundary = C_bc / A_bc
                    T_boundary = (phi_boundary / (A_RAD * C_LIGHT))**0.25
                    T_avg = 0.5 * (T_cell + T_boundary)
                else:  # Robin or Neumann - use cell temperature
                    T_avg = T_cell
                
                sigma_R_face = rosseland_opacity_func(T_avg, x_centers[i], y_faces[-1])
                D_y_faces[i, j] = 1.0 / (3.0 * sigma_R_face)
            else:
                # Interior face - harmonic mean with flux limiter
                T_bottom = T_star_2d[i, j-1]
                T_top = T_star_2d[i, j]
                phi_bottom = phi_star_2d[i, j-1]
                phi_top = phi_star_2d[i, j]
                
                # Average temperature for opacity evaluation
                T_avg = 0.5 * (T_bottom + T_top)
                
                # Evaluate opacity at FACE locations using average temperature
                x_face = x_centers[i]
                y_bottom = y_centers[j-1]
                y_top = y_centers[j]
                y_face = y_faces[j]
                sigma_R_bottom = rosseland_opacity_func(T_avg, x_face, y_bottom)
                sigma_R_top = rosseland_opacity_func(T_avg, x_face, y_top)
                
                # Compute flux limiter parameters
                dy_bottom = y_face - y_bottom
                dy_top = y_top - y_face
                dy_total = dy_bottom + dy_top
                
                grad_phi_mag = abs(phi_top - phi_bottom) / dy_total
                phi_avg = 0.5 * (phi_bottom + phi_top)
                
                # Compute diffusion coefficients on bottom and top
                if phi_avg < 1e-30:
                    D_bottom = 1.0 / (3.0 * sigma_R_bottom)
                    D_top = 1.0 / (3.0 * sigma_R_top)
                else:
                    R_bottom = grad_phi_mag / (sigma_R_bottom * phi_avg)
                    R_top = grad_phi_mag / (sigma_R_top * phi_avg)
                    
                    # Apply flux limiter
                    if flux_limiter_type == 0:
                        lambda_bottom = _apply_flux_limiter_standard(R_bottom)
                        lambda_top = _apply_flux_limiter_standard(R_top)
                    elif flux_limiter_type == 1:
                        lambda_bottom = _apply_flux_limiter_lp(R_bottom)
                        lambda_top = _apply_flux_limiter_lp(R_top)
                    elif flux_limiter_type == 2:
                        lambda_bottom = _apply_flux_limiter_larsen(R_bottom, flux_limiter_n)
                        lambda_top = _apply_flux_limiter_larsen(R_top, flux_limiter_n)
                    elif flux_limiter_type == 3:
                        lambda_bottom = _apply_flux_limiter_sum(R_bottom)
                        lambda_top = _apply_flux_limiter_sum(R_top)
                    else:  # flux_limiter_type == 4
                        lambda_bottom = _apply_flux_limiter_max(R_bottom)
                        lambda_top = _apply_flux_limiter_max(R_top)
                    
                    D_bottom = lambda_bottom / sigma_R_bottom
                    D_top = lambda_top / sigma_R_top
                
                # Harmonic mean with safeguard
                denom = dy_bottom/max(D_bottom, 1e-30) + dy_top/max(D_top, 1e-30)
                D_y_faces[i, j] = dy_total / max(denom, 1e-30)
    
    return D_y_faces


@njit
def assemble_implicit_matrix_coo(nx, ny, dt, theta, C_LIGHT,
                                   D_x_faces, D_y_faces,
                                   Ax_faces, Ay_faces,
                                   x_centers, y_centers,
                                   V_cells, sigma_P, f):
    """
    Assemble the implicit matrix in COO format for efficient sparse matrix construction.
    
    Returns:
        rows: row indices
        cols: column indices  
        data: matrix values
        n_entries: number of non-zero entries
    """
    # Estimate max entries: diagonal + 4 neighbors per cell
    max_entries = nx * ny * 5
    rows = np.zeros(max_entries, dtype=np.int32)
    cols = np.zeros(max_entries, dtype=np.int32)
    data = np.zeros(max_entries, dtype=np.float64)
    
    entry_idx = 0
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j  # Column-major indexing
            V = V_cells[i, j]
            
            # Diagonal: time derivative + coupling term
            diag_val = 1.0 / (C_LIGHT * dt) + theta * sigma_P[idx] * f[idx]
            
            # X-direction contribution to diagonal and off-diagonal
            if i > 0:  # Left face
                A_left = Ax_faces[i, j]
                dx_left = x_centers[i] - x_centers[i-1]
                coeff = theta * A_left * D_x_faces[i, j] / (dx_left * V)
                diag_val += coeff
                
                # Off-diagonal: left neighbor
                idx_left = (i-1) * ny + j
                rows[entry_idx] = idx
                cols[entry_idx] = idx_left
                data[entry_idx] = -coeff
                entry_idx += 1
            
            if i < nx - 1:  # Right face
                A_right = Ax_faces[i+1, j]
                dx_right = x_centers[i+1] - x_centers[i]
                coeff = theta * A_right * D_x_faces[i+1, j] / (dx_right * V)
                diag_val += coeff
                
                # Off-diagonal: right neighbor
                idx_right = (i+1) * ny + j
                rows[entry_idx] = idx
                cols[entry_idx] = idx_right
                data[entry_idx] = -coeff
                entry_idx += 1
            
            # Y-direction contribution to diagonal and off-diagonal
            if j > 0:  # Bottom face
                A_bottom = Ay_faces[i, j]
                dy_bottom = y_centers[j] - y_centers[j-1]
                coeff = theta * A_bottom * D_y_faces[i, j] / (dy_bottom * V)
                diag_val += coeff
                
                # Off-diagonal: bottom neighbor
                idx_bottom = i * ny + (j-1)
                rows[entry_idx] = idx
                cols[entry_idx] = idx_bottom
                data[entry_idx] = -coeff
                entry_idx += 1
            
            if j < ny - 1:  # Top face
                A_top = Ay_faces[i, j+1]
                dy_top = y_centers[j+1] - y_centers[j]
                coeff = theta * A_top * D_y_faces[i, j+1] / (dy_top * V)
                diag_val += coeff
                
                # Off-diagonal: top neighbor
                idx_top = i * ny + (j+1)
                rows[entry_idx] = idx
                cols[entry_idx] = idx_top
                data[entry_idx] = -coeff
                entry_idx += 1
            
            # Diagonal entry
            rows[entry_idx] = idx
            cols[entry_idx] = idx
            data[entry_idx] = diag_val
            entry_idx += 1
    
    return rows[:entry_idx], cols[:entry_idx], data[:entry_idx]


@njit
def assemble_trbdf2_matrix_coo(nx, ny, dt, c_np1, C_LIGHT,
                                 D_x_faces, D_y_faces,
                                 Ax_faces, Ay_faces,
                                 x_centers, y_centers,
                                 V_cells, sigma_P, f_TB):
    """
    Assemble the TR-BDF2 implicit matrix in COO format.
    
    TR-BDF2 uses c_np1/(c*dt) instead of 1/(c*dt) for time derivative.
    Fully implicit (no theta parameter).
    
    Returns:
        rows: row indices
        cols: column indices  
        data: matrix values
    """
    # Estimate max entries: diagonal + 4 neighbors per cell
    max_entries = nx * ny * 5
    rows = np.zeros(max_entries, dtype=np.int32)
    cols = np.zeros(max_entries, dtype=np.int32)
    data = np.zeros(max_entries, dtype=np.float64)
    
    entry_idx = 0
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            V = V_cells[i, j]
            
            # Diagonal: time derivative + coupling term
            diag_val = c_np1 / (C_LIGHT * dt) + f_TB[idx] * sigma_P[idx]
            
            # X-direction contribution to diagonal and off-diagonal
            if i > 0:  # Left face
                A_left = Ax_faces[i, j]
                dx_left = x_centers[i] - x_centers[i-1]
                coeff = A_left * D_x_faces[i, j] / (dx_left * V)
                diag_val += coeff
                
                # Off-diagonal: left neighbor
                idx_left = (i-1) * ny + j
                rows[entry_idx] = idx
                cols[entry_idx] = idx_left
                data[entry_idx] = -coeff
                entry_idx += 1
            
            if i < nx - 1:  # Right face
                A_right = Ax_faces[i+1, j]
                dx_right = x_centers[i+1] - x_centers[i]
                coeff = A_right * D_x_faces[i+1, j] / (dx_right * V)
                diag_val += coeff
                
                # Off-diagonal: right neighbor
                idx_right = (i+1) * ny + j
                rows[entry_idx] = idx
                cols[entry_idx] = idx_right
                data[entry_idx] = -coeff
                entry_idx += 1
            
            # Y-direction contribution to diagonal and off-diagonal
            if j > 0:  # Bottom face
                A_bottom = Ay_faces[i, j]
                dy_bottom = y_centers[j] - y_centers[j-1]
                coeff = A_bottom * D_y_faces[i, j] / (dy_bottom * V)
                diag_val += coeff
                
                # Off-diagonal: bottom neighbor
                idx_bottom = i * ny + (j-1)
                rows[entry_idx] = idx
                cols[entry_idx] = idx_bottom
                data[entry_idx] = -coeff
                entry_idx += 1
            
            if j < ny - 1:  # Top face
                A_top = Ay_faces[i, j+1]
                dy_top = y_centers[j+1] - y_centers[j]
                coeff = A_top * D_y_faces[i, j+1] / (dy_top * V)
                diag_val += coeff
                
                # Off-diagonal: top neighbor
                idx_top = i * ny + (j+1)
                rows[entry_idx] = idx
                cols[entry_idx] = idx_top
                data[entry_idx] = -coeff
                entry_idx += 1
            
            # Diagonal entry
            rows[entry_idx] = idx
            cols[entry_idx] = idx
            data[entry_idx] = diag_val
            entry_idx += 1
    
    return rows[:entry_idx], cols[:entry_idx], data[:entry_idx]


@njit
def compute_rhs_vector(nx, ny, dt, theta, C_LIGHT,
                       phi_prev, sigma_P, f, acT4_star, Delta_e):
    """
    Compute the RHS vector for the radiation diffusion equation.
    
    Returns:
        rhs: RHS vector (flattened in column-major order)
    """
    n_total = nx * ny
    rhs = np.zeros(n_total, dtype=np.float64)
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            
            # φ^n/(c·Δt)
            rhs[idx] = phi_prev[idx] / (C_LIGHT * dt)
            
            # σ_P·f·acT★⁴
            rhs[idx] += sigma_P[idx] * f[idx] * acT4_star[idx]
            
            # -(1-f)·Δe/Δt
            rhs[idx] -= (1.0 - f[idx]) * Delta_e[idx] / dt
            
            # -(1-θ)·σ_P·f·φ^n
            rhs[idx] -= (1.0 - theta) * sigma_P[idx] * f[idx] * phi_prev[idx]
    
    return rhs


@njit
def compute_explicit_diffusion(nx, ny, phi_prev_2d,
                                 D_x_faces, D_y_faces,
                                 Ax_faces, Ay_faces,
                                 x_centers, y_centers,
                                 V_cells):
    """
    Compute the explicit diffusion term for theta < 1.
    
    Returns:
        explicit_term: contribution to RHS (flattened in column-major order)
    """
    n_total = nx * ny
    explicit_term = np.zeros(n_total, dtype=np.float64)
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            V = V_cells[i, j]
            explicit_diffusion = 0.0
            
            # X-direction
            if i > 0:
                A_left = Ax_faces[i, j]
                dx_left = x_centers[i] - x_centers[i-1]
                flux_left = -D_x_faces[i, j] * (phi_prev_2d[i, j] - phi_prev_2d[i-1, j]) / dx_left
                explicit_diffusion += A_left * flux_left / V
            
            if i < nx - 1:
                A_right = Ax_faces[i+1, j]
                dx_right = x_centers[i+1] - x_centers[i]
                flux_right = -D_x_faces[i+1, j] * (phi_prev_2d[i+1, j] - phi_prev_2d[i, j]) / dx_right
                explicit_diffusion -= A_right * flux_right / V
            
            # Y-direction
            if j > 0:
                A_bottom = Ay_faces[i, j]
                dy_bottom = y_centers[j] - y_centers[j-1]
                flux_bottom = -D_y_faces[i, j] * (phi_prev_2d[i, j] - phi_prev_2d[i, j-1]) / dy_bottom
                explicit_diffusion += A_bottom * flux_bottom / V
            
            if j < ny - 1:
                A_top = Ay_faces[i, j+1]
                dy_top = y_centers[j+1] - y_centers[j]
                flux_top = -D_y_faces[i, j+1] * (phi_prev_2d[i, j+1] - phi_prev_2d[i, j]) / dy_top
                explicit_diffusion -= A_top * flux_top / V
            
            explicit_term[idx] = explicit_diffusion
    
    return explicit_term


@njit
def apply_bc_modifications(nx, ny, phi_2d, T_2d,
                           bc_types, bc_values,  # Pre-evaluated BCs
                           Ax_faces, Ay_faces,
                           x_centers, y_centers, x_faces, y_faces,
                           V_cells, sigma_R_2d, C_LIGHT, A_RAD):
    """
    Compute boundary condition modifications in COO format.
    
    bc_types: (4,) array with int codes: 0=Dirichlet, 1=Robin, 2=Reflecting
    bc_values: (4, max_boundary_size, 3) array with [A_bc, B_bc, C_bc] for each boundary
    
    Returns:
        diag_rows: indices for diagonal modifications
        diag_vals: values for diagonal modifications
        rhs_adds: additions to RHS
    """
    # Estimate size: 4 boundaries, max(nx, ny) cells per boundary
    max_bc_cells = 2 * (nx + ny)
    diag_rows = np.zeros(max_bc_cells, dtype=np.int32)
    diag_vals = np.zeros(max_bc_cells, dtype=np.float64)
    rhs_adds = np.zeros(max_bc_cells, dtype=np.float64)
    rhs_rows = np.zeros(max_bc_cells, dtype=np.int32)
    
    entry_idx = 0
    
    # Left boundary (i=0)
    bc_type = bc_types[0]
    for j in range(ny):
        idx = 0 * ny + j
        A_bc = bc_values[0, j, 0]
        B_bc = bc_values[0, j, 1]
        C_bc = bc_values[0, j, 2]
        
        if abs(B_bc) < 1e-14:  # Dirichlet
            phi_boundary = C_bc / max(abs(A_bc), 1e-30)
            phi_avg = 0.5 * (phi_2d[0, j] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
            D_boundary = 1.0 / (3.0 * sigma_R_2d[0, j])  # Simplified
            
            dx_half = x_centers[0] - x_faces[0]
            V = V_cells[0, j]
            flux_coeff = (Ax_faces[0, j] * D_boundary) / (V * dx_half)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = flux_coeff * phi_boundary
            entry_idx += 1
        else:  # Robin
            D_boundary = 1.0 / (3.0 * sigma_R_2d[0, j])
            V = V_cells[0, j]
            flux_coeff = (Ax_faces[0, j] * D_boundary * A_bc) / (B_bc * V)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = Ax_faces[0, j] * D_boundary * C_bc / (B_bc * V)
            entry_idx += 1
    
    # Right boundary (i=nx-1)
    for j in range(ny):
        idx = (nx-1) * ny + j
        A_bc = bc_values[1, j, 0]
        B_bc = bc_values[1, j, 1]
        C_bc = bc_values[1, j, 2]
        
        if abs(B_bc) < 1e-14:  # Dirichlet
            phi_boundary = C_bc / max(abs(A_bc), 1e-30)
            phi_avg = 0.5 * (phi_2d[nx-1, j] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
            D_boundary = 1.0 / (3.0 * sigma_R_2d[nx-1, j])
            
            dx_half = x_faces[-1] - x_centers[-1]
            V = V_cells[nx-1, j]
            flux_coeff = (Ax_faces[-1, j] * D_boundary) / (V * dx_half)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = flux_coeff * phi_boundary
            entry_idx += 1
        else:  # Robin
            D_boundary = 1.0 / (3.0 * sigma_R_2d[nx-1, j])
            V = V_cells[nx-1, j]
            flux_coeff = (Ax_faces[-1, j] * D_boundary * A_bc) / (B_bc * V)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = Ax_faces[-1, j] * D_boundary * C_bc / (B_bc * V)
            entry_idx += 1
    
    # Bottom boundary (j=0)
    for i in range(nx):
        idx = i * ny + 0
        A_bc = bc_values[2, i, 0]
        B_bc = bc_values[2, i, 1]
        C_bc = bc_values[2, i, 2]
        
        if abs(B_bc) < 1e-14:  # Dirichlet
            phi_boundary = C_bc / max(abs(A_bc), 1e-30)
            phi_avg = 0.5 * (phi_2d[i, 0] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
            D_boundary = 1.0 / (3.0 * sigma_R_2d[i, 0])
            
            dy_half = y_centers[0] - y_faces[0]
            V = V_cells[i, 0]
            flux_coeff = (Ay_faces[i, 0] * D_boundary) / (V * dy_half)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = flux_coeff * phi_boundary
            entry_idx += 1
        else:  # Robin
            D_boundary = 1.0 / (3.0 * sigma_R_2d[i, 0])
            V = V_cells[i, 0]
            flux_coeff = (Ay_faces[i, 0] * D_boundary * A_bc) / (B_bc * V)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = Ay_faces[i, 0] * D_boundary * C_bc / (B_bc * V)
            entry_idx += 1
    
    # Top boundary (j=ny-1)
    for i in range(nx):
        idx = i * ny + (ny-1)
        A_bc = bc_values[3, i, 0]
        B_bc = bc_values[3, i, 1]
        C_bc = bc_values[3, i, 2]
        
        if abs(B_bc) < 1e-14:  # Dirichlet
            phi_boundary = C_bc / max(abs(A_bc), 1e-30)
            phi_avg = 0.5 * (phi_2d[i, ny-1] + phi_boundary)
            T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
            D_boundary = 1.0 / (3.0 * sigma_R_2d[i, ny-1])
            
            dy_half = y_faces[-1] - y_centers[-1]
            V = V_cells[i, ny-1]
            flux_coeff = (Ay_faces[i, ny-1] * D_boundary) / (V * dy_half)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = flux_coeff * phi_boundary
            entry_idx += 1
        else:  # Robin
            D_boundary = 1.0 / (3.0 * sigma_R_2d[i, ny-1])
            V = V_cells[i, ny-1]
            flux_coeff = (Ay_faces[i, ny-1] * D_boundary * A_bc) / (B_bc * V)
            
            diag_rows[entry_idx] = idx
            diag_vals[entry_idx] = flux_coeff
            rhs_rows[entry_idx] = idx
            rhs_adds[entry_idx] = Ay_faces[i, ny-1] * D_boundary * C_bc / (B_bc * V)
            entry_idx += 1
    
    return (diag_rows[:entry_idx], diag_vals[:entry_idx], 
            rhs_rows[:entry_idx], rhs_adds[:entry_idx])


# =============================================================================
# GRID GENERATION AND GEOMETRY
# =============================================================================

def generate_2d_grid(x_min: float, x_max: float, nx_cells: int,
                     y_min: float, y_max: float, ny_cells: int,
                     x_stretch: float = 1.0, y_stretch: float = 1.0,
                     geometry: str = 'cartesian') -> Tuple[np.ndarray, np.ndarray, 
                                                            np.ndarray, np.ndarray]:
    """
    Generate 2D structured grid
    
    Parameters:
    -----------
    x_min, x_max : float
        First coordinate domain boundaries (x for Cartesian, r for cylindrical)
    nx_cells : int
        Number of cells in first coordinate direction
    y_min, y_max : float
        Second coordinate domain boundaries (y for Cartesian, z for cylindrical)
    ny_cells : int
        Number of cells in second coordinate direction
    x_stretch, y_stretch : float
        Grid stretching factors (1.0 = uniform)
    geometry : str
        'cartesian' for (x,y) or 'cylindrical' for (r,z)
    
    Returns:
    --------
    x_faces : ndarray (nx_cells + 1,)
        Face positions in first coordinate
    y_faces : ndarray (ny_cells + 1,)
        Face positions in second coordinate
    x_centers : ndarray (nx_cells,)
        Cell center positions in first coordinate
    y_centers : ndarray (ny_cells,)
        Cell center positions in second coordinate
        
    Notes:
    ------
    For cylindrical geometry, x corresponds to r and must have x_min >= 0
    """
    if geometry == 'cylindrical' and x_min < 0:
        raise ValueError("For cylindrical geometry, r_min (x_min) must be >= 0")
    
    # First coordinate (x or r)
    if x_stretch == 1.0:
        x_faces = np.linspace(x_min, x_max, nx_cells + 1)
    else:
        xi = np.linspace(0, 1, nx_cells + 1)
        x_faces = x_min + (x_max - x_min) * ((x_stretch**xi - 1) / (x_stretch - 1))
    
    # Second coordinate (y or z)
    if y_stretch == 1.0:
        y_faces = np.linspace(y_min, y_max, ny_cells + 1)
    else:
        yi = np.linspace(0, 1, ny_cells + 1)
        y_faces = y_min + (y_max - y_min) * ((y_stretch**yi - 1) / (y_stretch - 1))
    
    # Cell centers
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    
    return x_faces, y_faces, x_centers, y_centers


def compute_2d_geometry_factors(x_faces, y_faces, geometry='cartesian'):
    """
    Compute geometry factors for 2D finite volumes
    
    Parameters:
    -----------
    x_faces : ndarray (nx+1,)
        Face positions in first coordinate (x or r)
    y_faces : ndarray (ny+1,)
        Face positions in second coordinate (y or z)
    geometry : str
        'cartesian' for (x,y) or 'cylindrical' for (r,z)
    
    Returns:
    --------
    Ax_faces : ndarray (nx+1, ny)
        Face areas for x-direction (or r-direction) faces
    Ay_faces : ndarray (nx, ny+1)
        Face areas for y-direction (or z-direction) faces
    V_cells : ndarray (nx, ny)
        Cell volumes
        
    Notes:
    ------
    Cartesian (x-y):
      - Ax_faces[i,j]: area of face perpendicular to x at position x_faces[i]
                       between y_centers[j-1] and y_centers[j]
                       Ax = Δy_j
      - Ay_faces[i,j]: area of face perpendicular to y at position y_faces[j]
                       between x_centers[i-1] and x_centers[i]
                       Ay = Δx_i
      - V_cells[i,j]: volume of cell (i,j)
                      V = Δx_i * Δy_j
    
    Cylindrical (r-z):
      - Ar_faces[i,j]: area of face perpendicular to r at radius r_faces[i]
                       between z_centers[j-1] and z_centers[j]
                       Ar = 2π * r_faces[i] * Δz_j
      - Az_faces[i,j]: area of face perpendicular to z at position z_faces[j]
                       between r_centers[i-1] and r_centers[i]
                       Az = π * (r_faces[i+1]² - r_faces[i]²)
      - V_cells[i,j]: volume of cell (i,j)
                      V = π * (r_faces[i+1]² - r_faces[i]²) * Δz_j
    """
    nx = len(x_faces) - 1
    ny = len(y_faces) - 1
    
    if geometry == 'cartesian':
        # Cartesian geometry
        # Face areas in x-direction: Ax = Δy
        Ax_faces = np.zeros((nx + 1, ny))
        for j in range(ny):
            dy = y_faces[j+1] - y_faces[j]
            Ax_faces[:, j] = dy
        
        # Face areas in y-direction: Ay = Δx
        Ay_faces = np.zeros((nx, ny + 1))
        for i in range(nx):
            dx = x_faces[i+1] - x_faces[i]
            Ay_faces[i, :] = dx
        
        # Cell volumes: V = Δx * Δy
        V_cells = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                dx = x_faces[i+1] - x_faces[i]
                dy = y_faces[j+1] - y_faces[j]
                V_cells[i, j] = dx * dy
    
    elif geometry == 'cylindrical':
        # Cylindrical geometry (r-z)
        # Face areas in r-direction: Ar = 2π * r * Δz
        Ax_faces = np.zeros((nx + 1, ny))  # Ar_faces
        for i in range(nx + 1):
            r = x_faces[i]
            for j in range(ny):
                dz = y_faces[j+1] - y_faces[j]
                Ax_faces[i, j] = 2.0 * np.pi * r * dz
        
        # Face areas in z-direction: Az = π * (r_outer² - r_inner²)
        Ay_faces = np.zeros((nx, ny + 1))  # Az_faces
        for i in range(nx):
            r_inner = x_faces[i]
            r_outer = x_faces[i+1]
            area_z = np.pi * (r_outer**2 - r_inner**2)
            Ay_faces[i, :] = area_z
        
        # Cell volumes: V = π * (r_outer² - r_inner²) * Δz
        V_cells = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                r_inner = x_faces[i]
                r_outer = x_faces[i+1]
                dz = y_faces[j+1] - y_faces[j]
                V_cells[i, j] = np.pi * (r_outer**2 - r_inner**2) * dz
    
    else:
        raise ValueError(f"Unknown geometry: {geometry}. Use 'cartesian' or 'cylindrical'")
    
    return Ax_faces, Ay_faces, V_cells


def index_2d_to_1d(i, j, nx, ny):
    """Convert 2D cell index (i,j) to 1D array index
    Uses C-order (row-major) indexing to match order='C'"""
    return i * ny + j


def index_1d_to_2d(idx, nx, ny):
    """Convert 1D array index to 2D cell index (i,j)
    Uses C-order (row-major) indexing to match order='C'"""
    i = idx // ny
    j = idx % ny
    return i, j


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def robin_bc_coefficients_2d(phi, pos, t, boundary='left', geometry='cartesian'):
    """
    Robin boundary condition: A_bc * φ + B_bc * (n·∇φ) = C_bc
    Returns (A_bc, B_bc, C_bc)
    
    Parameters:
    -----------
    phi : float
        Value of φ at boundary
    pos : tuple (x, y) or (r, z)
        Position at boundary
    t : float
        Current time (ns)
    boundary : str
        'left', 'right', 'bottom', 'top'
    geometry : str
        'cartesian' or 'cylindrical'
    
    Returns:
    --------
    A_bc, B_bc, C_bc : float
        Boundary condition coefficients
    """
    if boundary == 'left':
        # Left boundary (x=x_min or r=r_min)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet
        T_bc = 1.0  # Temperature at boundary (keV)
        phi_bc = A_RAD * T_bc**4 * C_LIGHT
        C_bc = phi_bc
    elif boundary == 'right':
        # Right boundary (x=x_max or r=r_max)
        A_bc = 1.0
        B_bc = 0.0  # Dirichlet
        T_bc = 0.316
        phi_bc = A_RAD * T_bc**4 * C_LIGHT
        C_bc = phi_bc
    elif boundary == 'bottom':
        # Bottom boundary (y=y_min or z=z_min)
        A_bc = 0.0
        B_bc = 1.0  # Neumann (zero flux)
        C_bc = 0.0
    elif boundary == 'top':
        # Top boundary (y=y_max or z=z_max)
        A_bc = 0.0
        B_bc = 1.0  # Neumann (zero flux)
        C_bc = 0.0
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
    
    return A_bc, B_bc, C_bc


# =============================================================================
# MAIN SOLVER CLASS
# =============================================================================

class NonEquilibriumRadiationDiffusionSolver2D:
    """2D Finite Volume Non-Equilibrium Radiation Diffusion Solver
    
    Solves coupled system for φ(x,y,t) and T(x,y,t) or φ(r,z,t) and T(r,z,t)
    """
    
    def __init__(self, x_min=0.0, x_max=1.0, nx_cells=20,
                 y_min=0.0, y_max=1.0, ny_cells=20,
                 geometry='cartesian', dt=1e-3,
                 max_newton_iter=10, newton_tol=1e-8,
                 x_stretch=1.0, y_stretch=1.0,
                 x_faces=None, y_faces=None,
                 rosseland_opacity_func=None, planck_opacity_func=None,
                 specific_heat_func=None, material_energy_func=None,
                 inverse_material_energy_func=None,
                 boundary_funcs=None,
                 theta=1.0, flux_limiter_func=None):
        """
        Initialize 2D solver
        
        Parameters:
        -----------
        x_min, x_max : float
            First coordinate domain (x for Cartesian, r for cylindrical) - ignored if x_faces provided
        nx_cells : int
            Number of cells in first coordinate - ignored if x_faces provided
        y_min, y_max : float
            Second coordinate domain (y for Cartesian, z for cylindrical) - ignored if y_faces provided
        ny_cells : int
            Number of cells in second coordinate - ignored if y_faces provided
        geometry : str
            'cartesian' for (x,y) or 'cylindrical' for (r,z)
        dt : float
            Time step size
        max_newton_iter : int
            Maximum Newton iterations
        newton_tol : float
            Newton convergence tolerance
        x_stretch, y_stretch : float
            Grid stretching factors (ignored if custom faces provided)
        x_faces : ndarray or None
            Custom face positions in first direction. If provided, overrides x_min/max and nx_cells
        y_faces : ndarray or None
            Custom face positions in second direction. If provided, overrides y_min/max and ny_cells
        theta : float
            Time discretization parameter (1.0 = implicit Euler)
        flux_limiter_func : callable
            Flux limiter function λ(R)
        boundary_funcs : dict
            Dictionary with keys 'left', 'right', 'bottom', 'top' mapping to
            boundary condition functions
        """
        self.geometry = geometry
        self.dt = dt
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.theta = theta
        
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
            # Generate grid using standard method
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            self.nx_cells = nx_cells
            self.ny_cells = ny_cells
            self.x_faces, self.y_faces, self.x_centers, self.y_centers = \
                generate_2d_grid(x_min, x_max, nx_cells, y_min, y_max, ny_cells,
                               x_stretch, y_stretch, geometry)
        
        self.n_total = self.nx_cells * self.ny_cells
        
        self.Ax_faces, self.Ay_faces, self.V_cells = \
            compute_2d_geometry_factors(self.x_faces, self.y_faces, geometry)
        
        # Create 2D mesh grids for visualization
        self.X_centers, self.Y_centers = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Solution arrays (stored as 1D arrays of length n_total)
        self.phi = np.ones(self.n_total)
        self.T = np.ones(self.n_total)
        self.phi_old = np.ones(self.n_total)
        self.T_old = np.ones(self.n_total)
        
        # Time tracking
        self.current_time = 0.0
        
        # Material property functions
        self.rosseland_opacity_func = rosseland_opacity_func or rosseland_opacity
        self.planck_opacity_func = planck_opacity_func or planck_opacity
        self.specific_heat_func = specific_heat_func or specific_heat_cv
        self.material_energy_func = material_energy_func or material_energy_density
        self.inverse_material_energy_func = inverse_material_energy_func or inverse_material_energy_density
        
        # Flux limiter function
        self.flux_limiter_func = flux_limiter_func or flux_limiter_standard
        
        # Detect flux limiter type for Numba acceleration
        self._detect_flux_limiter_type()
        
        # Boundary condition functions
        if boundary_funcs is None:
            self.boundary_funcs = {
                'left': lambda phi, pos, t: robin_bc_coefficients_2d(phi, pos, t, 'left', geometry),
                'right': lambda phi, pos, t: robin_bc_coefficients_2d(phi, pos, t, 'right', geometry),
                'bottom': lambda phi, pos, t: robin_bc_coefficients_2d(phi, pos, t, 'bottom', geometry),
                'top': lambda phi, pos, t: robin_bc_coefficients_2d(phi, pos, t, 'top', geometry)
            }
        else:
            self.boundary_funcs = boundary_funcs
        
        # Print initialization info
        coord_names = {'cartesian': 'x-y', 'cylindrical': 'r-z'}
        print(f"Initialized 2D non-equilibrium solver ({coord_names[geometry]} geometry)")
        print(f"  Grid: {self.nx_cells} × {self.ny_cells} = {self.n_total} cells")
        print(f"  Domain: [{self.x_min:.3f}, {self.x_max:.3f}] × [{self.y_min:.3f}, {self.y_max:.3f}]")
        if x_faces is not None:
            print(f"  Using custom face arrays")
        print(f"  Δt={dt:.2e}, θ={theta}")
    
    def _detect_flux_limiter_type(self):
        """Detect which flux limiter is being used for Numba acceleration"""
        import functools
        
        # Get the actual function (unwrap if it's a partial)
        func = self.flux_limiter_func
        if isinstance(func, functools.partial):
            base_func = func.func
            # Check if it's Larsen with specific n
            if base_func.__name__ == 'flux_limiter_larsen':
                self._flux_limiter_type = 2  # Larsen
                self._flux_limiter_n = func.keywords.get('n', 2.0)
                return
        else:
            base_func = func
        
        # Match by function name
        func_name = base_func.__name__
        if 'standard' in func_name:
            self._flux_limiter_type = 0
            self._flux_limiter_n = 2.0
        elif 'levermore' in func_name or 'pomraning' in func_name or func_name == 'flux_limiter_lp':
            self._flux_limiter_type = 1
            self._flux_limiter_n = 2.0
        elif 'larsen' in func_name:
            self._flux_limiter_type = 2
            self._flux_limiter_n = 2.0
        elif 'sum' in func_name:
            self._flux_limiter_type = 3
            self._flux_limiter_n = 2.0
        elif 'max' in func_name:
            self._flux_limiter_type = 4
            self._flux_limiter_n = 2.0
        else:
            # Default to standard if unknown
            self._flux_limiter_type = 0
            self._flux_limiter_n = 2.0
    
    def get_phi_2d(self):
        """Return φ as 2D array (nx, ny)"""
        return self.phi.reshape((self.nx_cells, self.ny_cells), order='C')
    
    def get_T_2d(self):
        """Return T as 2D array (nx, ny)"""
        return self.T.reshape((self.nx_cells, self.ny_cells), order='C')
    
    def set_phi_2d(self, phi_2d):
        """Set φ from 2D array (nx, ny)"""
        self.phi = phi_2d.flatten(order='C')
    
    def set_T_2d(self, T_2d):
        """Set T from 2D array (nx, ny)"""
        self.T = T_2d.flatten(order='C')
    
    def get_diffusion_coefficient(self, T, x, y, phi_left=None, phi_right=None, dx=None):
        """Get flux-limited diffusion coefficient D = λ(R)/σ_R
        
        Parameters:
        -----------
        T : float
            Temperature at the face
        x, y : float
            Spatial coordinates at the face
        phi_left : float, optional
            φ value on left side of face
        phi_right : float, optional
            φ value on right side of face
        dx : float, optional
            Distance between cell centers
        
        Returns:
        --------
        D : float
            Diffusion coefficient
        """
        sigma_R = self.rosseland_opacity_func(T, x, y)
        
        if phi_left is None or phi_right is None or dx is None:
            return 1.0 / (3.0 * sigma_R)
        
        grad_phi_mag = abs(phi_right - phi_left) / dx
        phi_face = 0.5 * (phi_left + phi_right)
        
        if phi_face < 1e-30:
            return 1.0 / (3.0 * sigma_R)
        
        R = grad_phi_mag / (sigma_R * phi_face)
        lambda_R = self.flux_limiter_func(R)
        
        return lambda_R / sigma_R
    
    def get_harmonic_diffusion_coefficient(self, T_left, T_right, x_left, y_left, 
                                           x_right, y_right, phi_left, phi_right, 
                                           dx_left, dx_right):
        """Get harmonic mean diffusion coefficient for heterogeneous materials
        
        Computes D at left and right cell centers using average temperature,
        then returns harmonic mean weighted by mesh spacing.
        
        Parameters:
        -----------
        T_left, T_right : float
            Temperatures in left and right cells
        x_left, y_left : float
            Spatial coordinates of left cell center
        x_right, y_right : float
            Spatial coordinates of right cell center
        phi_left, phi_right : float
            φ values in left and right cells
        dx_left, dx_right : float
            Distance from left cell center to face, and face to right cell center
        
        Returns:
        --------
        D_harmonic : float
            Harmonic mean diffusion coefficient
        """
        # Average temperature (used for both sides)
        T_avg = 0.5 * (T_left + T_right)
        
        # Compute diffusion coefficient on left side (at left cell center)
        sigma_R_left = self.rosseland_opacity_func(T_avg, x_left, y_left)
        
        # Compute flux limiter on left side
        dx_total = dx_left + dx_right
        grad_phi_mag = abs(phi_right - phi_left) / dx_total
        phi_avg = 0.5 * (phi_left + phi_right)
        
        if phi_avg < 1e-30:
            D_left = 1.0 / (3.0 * sigma_R_left)
        else:
            R_left = grad_phi_mag / (sigma_R_left * phi_avg)
            lambda_left = self.flux_limiter_func(R_left)
            D_left = lambda_left / sigma_R_left
        
        # Compute diffusion coefficient on right side (at right cell center)
        sigma_R_right = self.rosseland_opacity_func(T_avg, x_right, y_right)
        
        if phi_avg < 1e-30:
            D_right = 1.0 / (3.0 * sigma_R_right)
        else:
            R_right = grad_phi_mag / (sigma_R_right * phi_avg)
            lambda_right = self.flux_limiter_func(R_right)
            D_right = lambda_right / sigma_R_right
        
        # Harmonic mean weighted by mesh spacing
        # D_harmonic = (dx_left + dx_right) / (dx_left/D_left + dx_right/D_right)
        D_harmonic = dx_total / (dx_left/D_left + dx_right/D_right)
        
        return D_harmonic
    
    def get_beta(self, T_star, x, y):
        """Compute coupling parameter β = 4aT_★³/C_v_★
        
        Parameters:
        -----------
        T_star : float
            Temperature (keV)
        x, y : float
            Spatial coordinates (cm)
        
        Returns:
        --------
        beta : float
            Coupling parameter (1/keV)
        """
        cv_star = self.specific_heat_func(T_star, x, y)
        return (4.0 * A_RAD * T_star**3) / (RHO * cv_star)
    
    def get_f_factor(self, T_star, x, y, dt, theta):
        """Compute linearization factor f = 1 / (1 + β·σ_P·c·θ·Δt)
        
        Parameters:
        -----------
        T_star : float
            Temperature (keV)
        x, y : float
            Spatial coordinates (cm)
        dt : float
            Time step (ns)
        theta : float
            Time discretization parameter
        
        Returns:
        --------
        f : float
            Linearization factor (dimensionless)
        """
        beta = self.get_beta(T_star, x, y)
        sigma_P = self.planck_opacity_func(T_star, x, y)
        return 1.0 / (1.0 + beta * sigma_P * C_LIGHT * theta * dt)
    
    def get_f_factor_trbdf2(self, T_star, x, y, dt, Lambda):
        """Compute linearization factor f_TB for TR-BDF2 from equation (8.62)
        
        f_TB = 1 / (1 + [(1-Λ)/(2-Λ)] · β·σ_P·c·Δt)
        where β = 4aT_★³/C_v_★
        
        Parameters:
        -----------
        T_star : float
            Temperature (keV)
        x, y : float
            Spatial coordinates (cm)
        dt : float
            Time step (ns)
        Lambda : float
            Intermediate time fraction
        
        Returns:
        --------
        f_TB : float
            Linearization factor (dimensionless)
        """
        beta = self.get_beta(T_star, x, y)
        sigma_P = self.planck_opacity_func(T_star, x, y)
        coeff = (1.0 - Lambda) / (2.0 - Lambda)
        return 1.0 / (1.0 + coeff * beta * sigma_P * C_LIGHT * dt)
    
    def set_initial_condition(self, phi_init=None, T_init=None):
        """Set initial conditions for φ and T
        
        Parameters:
        -----------
        phi_init : callable, float, or ndarray
            Initial φ. If callable: phi_init(x, y) or phi_init(r, z)
        T_init : callable, float, or ndarray
            Initial temperature. If callable: T_init(x, y) or T_init(r, z)
        """
        # Handle T initialization
        if T_init is not None:
            if callable(T_init):
                T_2d = np.zeros((self.nx_cells, self.ny_cells))
                for i in range(self.nx_cells):
                    for j in range(self.ny_cells):
                        T_2d[i, j] = T_init(self.x_centers[i], self.y_centers[j])
                self.T = T_2d.flatten(order='C')
            elif isinstance(T_init, np.ndarray):
                if T_init.shape == (self.nx_cells, self.ny_cells):
                    self.T = T_init.flatten(order='C')
                elif T_init.shape == (self.n_total,):
                    self.T = T_init.copy()
                else:
                    raise ValueError(f"T_init shape {T_init.shape} doesn't match grid")
            else:
                self.T = np.full(self.n_total, float(T_init))
        
        # Handle φ initialization
        if phi_init is None:
            # Default: equilibrium with T
            T_2d = self.get_T_2d()
            phi_2d = A_RAD * C_LIGHT * T_2d**4
            self.phi = phi_2d.flatten(order='C')
        else:
            if callable(phi_init):
                phi_2d = np.zeros((self.nx_cells, self.ny_cells))
                for i in range(self.nx_cells):
                    for j in range(self.ny_cells):
                        phi_2d[i, j] = phi_init(self.x_centers[i], self.y_centers[j])
                self.phi = phi_2d.flatten(order='C')
            elif isinstance(phi_init, np.ndarray):
                if phi_init.shape == (self.nx_cells, self.ny_cells):
                    self.phi = phi_init.flatten(order='C')
                elif phi_init.shape == (self.n_total,):
                    self.phi = phi_init.copy()
                else:
                    raise ValueError(f"phi_init shape {phi_init.shape} doesn't match grid")
            else:
                self.phi = np.full(self.n_total, float(phi_init))
        
        # If only phi given and T not set, compute T from equilibrium
        if phi_init is not None and T_init is None:
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
        """Return current solution as 2D arrays"""
        return (self.x_centers.copy(), self.y_centers.copy(),
                self.get_phi_2d(), self.get_T_2d())
    
    def compute_material_properties_vectorized(self, T_array, phi_array=None):
        """Compute material properties for all cells using vectorized operations
        
        Parameters:
        -----------
        T_array : ndarray (n_total,)
            Temperature at each cell
        phi_array : ndarray (n_total,) or None
            Phi at each cell (only needed for some computations)
        
        Returns:
        --------
        dict with keys:
            'sigma_P' : ndarray (n_total,) - Planck opacity
            'sigma_R' : ndarray (n_total,) - Rosseland opacity (if needed)
            'cv' : ndarray (n_total,) - Specific heat
            'beta' : ndarray (n_total,) - Coupling parameter
            'f' : ndarray (n_total,) - Linearization factor
            'e' : ndarray (n_total,) - Material energy
        """
        # Create 2D meshgrids for coordinates
        X_mesh, Y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Reshape T to 2D
        T_2d = T_array.reshape((self.nx_cells, self.ny_cells), order='C')
        
        # Compute properties using vectorized calls
        sigma_P_2d = self.planck_opacity_func(T_2d, X_mesh, Y_mesh)
        cv_2d = self.specific_heat_func(T_2d, X_mesh, Y_mesh)
        e_2d = self.material_energy_func(T_2d, X_mesh, Y_mesh)
        
        # Handle scalar returns (constant properties)
        if np.isscalar(sigma_P_2d):
            sigma_P_2d = np.full_like(T_2d, sigma_P_2d)
        if np.isscalar(cv_2d):
            cv_2d = np.full_like(T_2d, cv_2d)
        if np.isscalar(e_2d):
            e_2d = np.full_like(T_2d, e_2d)
        
        # Compute derived quantities
        beta_2d = (4.0 * A_RAD * T_2d**3) / (RHO * cv_2d)
        f_2d = 1.0 / (1.0 + beta_2d * sigma_P_2d * C_LIGHT * self.theta * self.dt)
        
        # Flatten back to 1D arrays
        result = {
            'sigma_P': sigma_P_2d.flatten(order='C'),
            'cv': cv_2d.flatten(order='C'),
            'beta': beta_2d.flatten(order='C'),
            'f': f_2d.flatten(order='C'),
            'e': e_2d.flatten(order='C')
        }
        
        return result
    
    def compute_temperature_from_energy_vectorized(self, e_array):
        """Compute temperature from energy for all cells using vectorized operations
        
        Parameters:
        -----------
        e_array : ndarray (n_total,)
            Material energy at each cell
        
        Returns:
        --------
        T_array : ndarray (n_total,)
            Temperature at each cell
        """
        # Create 2D meshgrids for coordinates
        X_mesh, Y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        # Reshape e to 2D
        e_2d = e_array.reshape((self.nx_cells, self.ny_cells), order='C')
        
        # Compute T using vectorized call
        T_2d = self.inverse_material_energy_func(e_2d, X_mesh, Y_mesh)
        
        # Flatten back to 1D
        T_array = T_2d.flatten(order='C')
        
        return T_array
    
    def assemble_phi_equation(self, phi_star, T_star, phi_prev, T_prev, theta=1.0, source=None, timing=False):
        """Assemble sparse matrix for φ equation
        
        Equation 8.59a (2D version):
        (φ^{n+1} - φ^n)/(c·Δt) + θ∇·D∇φ^{n+1} + (1-θ)∇·D∇φ^n = 
            σ_P·f(acT★⁴ - φ̃) - (1-f)·Δe/Δt + S
        
        Returns:
        --------
        A_sparse : scipy.sparse matrix (n_total, n_total)
            Sparse matrix for φ^{n+1}
        rhs : ndarray (n_total,)
            Right-hand side vector
        """
        import time
        t_start = time.time() if timing else None
        
        n_total = self.n_total
        nx = self.nx_cells
        ny = self.ny_cells
        dt = self.dt
        
        # Reshape arrays to 2D for easier indexing
        phi_star_2d = phi_star.reshape((nx, ny), order='C')
        T_star_2d = T_star.reshape((nx, ny), order='C')
        phi_prev_2d = phi_prev.reshape((nx, ny), order='C')
        T_prev_2d = T_prev.reshape((nx, ny), order='C')
        
        t0 = time.time() if timing else None
        # Compute material properties using vectorized operations
        props_star = self.compute_material_properties_vectorized(T_star, phi_star)
        props_prev = self.compute_material_properties_vectorized(T_prev, phi_prev)
        
        e_star = props_star['e']
        e_n = props_prev['e']
        sigma_P = props_star['sigma_P']
        f = props_star['f']
        
        Delta_e = e_star - e_n
        acT4_star = A_RAD * C_LIGHT * T_star**4
        t1 = time.time() if timing else None
        if timing:
            time_props = t1 - t0
        
        t0 = time.time() if timing else None
        # Compute Rosseland opacity at all cell centers for diffusion coefficients
        # Create meshgrid for vectorized evaluation
        X_mesh, Y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        sigma_R_2d = self.rosseland_opacity_func(T_star_2d, X_mesh, Y_mesh)
        
        # Handle scalar returns
        if np.isscalar(sigma_R_2d):
            sigma_R_2d = np.full_like(T_star_2d, sigma_R_2d)
        
        # Compute diffusion coefficients (now without Numba to allow opacity evaluation at face)
        D_x_faces = compute_face_diffusion_coefficients_x(
            nx, ny, T_star_2d, phi_star_2d, sigma_R_2d,
            self.x_centers, self.y_centers, self.x_faces,
            self.rosseland_opacity_func,
            self.boundary_funcs, self.current_time, A_RAD, C_LIGHT,
            self._flux_limiter_type, self._flux_limiter_n
        )
        
        D_y_faces = compute_face_diffusion_coefficients_y(
            nx, ny, T_star_2d, phi_star_2d, sigma_R_2d,
            self.x_centers, self.y_centers, self.y_faces,
            self.rosseland_opacity_func,
            self.boundary_funcs, self.current_time, A_RAD, C_LIGHT,
            self._flux_limiter_type, self._flux_limiter_n
        )
        
        t1 = time.time() if timing else None
        if timing:
            time_diffusion = t1 - t0
        
        t0 = time.time() if timing else None
        # Assemble matrix using Numba-accelerated function (COO format)
        rows, cols, data = assemble_implicit_matrix_coo(
            nx, ny, dt, theta, C_LIGHT,
            D_x_faces, D_y_faces,
            self.Ax_faces, self.Ay_faces,
            self.x_centers, self.y_centers,
            self.V_cells, sigma_P, f
        )
        
        # Build sparse matrix from COO format
        from scipy.sparse import coo_matrix
        A_csr = coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()
        
        # Compute RHS using Numba-accelerated function
        rhs = compute_rhs_vector(nx, ny, dt, theta, C_LIGHT,
                                  phi_prev, sigma_P, f, acT4_star, Delta_e)
        
        # Add explicit diffusion term if θ < 1
        if theta < 1.0:
            explicit_term = compute_explicit_diffusion(
                nx, ny, phi_prev_2d,
                D_x_faces, D_y_faces,
                self.Ax_faces, self.Ay_faces,
                self.x_centers, self.y_centers,
                self.V_cells
            )
            rhs += (1.0 - theta) * explicit_term
        
        # Add external source if provided
        if source is not None:
            if isinstance(source, np.ndarray):
                if source.shape == (nx, ny):
                    rhs += source.flatten(order='C')
                elif source.shape == (n_total,):
                    rhs += source
            else:
                rhs += source
        
        t1 =time.time() if timing else None
        if timing:
            time_assembly = t1 - t0
        
        # CSR conversion already done above
        t0 = time.time() if timing else None
        t1 = time.time() if timing else None
        if timing:
            time_csr = 0.0  # Already included in assembly time
            time_total = time_props + time_diffusion + time_assembly + time_csr
            print(f"        [assemble_phi_equation timing]")
            print(f"          Material properties:  {time_props*1000:.1f} ms ({100*time_props/time_total:.1f}%)")
            print(f"          Diffusion coeffs:     {time_diffusion*1000:.1f} ms ({100*time_diffusion/time_total:.1f}%)")
            print(f"          Matrix assembly:      {time_assembly*1000:.1f} ms ({100*time_assembly/time_total:.1f}%)")
        
        return A_csr, rhs
    
    def apply_boundary_conditions_phi(self, A, rhs, phi):
        """Apply boundary conditions to φ equation
        
        Modifies matrix and RHS for boundary cells
        
        Parameters:
        -----------
        A : scipy.sparse matrix
            Sparse matrix (will be modified in-place if possible)
        rhs : ndarray
            Right-hand side vector (modified in-place)
        phi : ndarray
            Current φ values (for evaluating BC)
        """
        nx = self.nx_cells
        ny = self.ny_cells
        
        # Convert to LIL format for efficient modification
        A_lil = A.tolil()
        
        phi_2d = phi.reshape((nx, ny), order='C')
        T_2d = self.T.reshape((nx, ny), order='C')
        
        # Pre-compute diffusion coefficients at boundaries to avoid repeated scalar calls
        # Left boundary
        for j in range(ny):
            idx = index_2d_to_1d(0, j, nx, ny)
            pos = (self.x_faces[0], self.y_centers[j])
            A_bc, B_bc, C_bc = self.boundary_funcs['left'](phi_2d[0, j], pos, self.current_time)
            
            if abs(B_bc) < 1e-14:
                # Dirichlet BC
                phi_boundary = C_bc / A_bc
                # Average phi to get temperature for diffusion coefficient
                # This prevents using the cold interior T, which gives huge opacity
                phi_avg = 0.5 * (phi_2d[0, j] + phi_boundary)
                T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
                sigma_R = self.rosseland_opacity_func(T_avg, self.x_centers[0], self.y_centers[j])
                D_boundary = 1.0 / (3.0 * sigma_R)
                
                dx_half = self.x_centers[0] - self.x_faces[0]
                V = self.V_cells[0, j]
                flux_coeff = (self.Ax_faces[0, j] * D_boundary) / (V * dx_half)
                
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += flux_coeff * phi_boundary
            else:
                # Robin BC
                sigma_R = self.rosseland_opacity_func(T_2d[0, j], self.x_centers[0], self.y_centers[j])
                D_boundary = 1.0 / (3.0 * sigma_R)
                V = self.V_cells[0, j]
                flux_coeff = (self.Ax_faces[0, j] * D_boundary * A_bc) / (B_bc * V)
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += self.Ax_faces[0, j] * D_boundary * C_bc / (B_bc * V)
        
        # Right boundary
        for j in range(ny):
            idx = index_2d_to_1d(nx-1, j, nx, ny)
            pos = (self.x_faces[-1], self.y_centers[j])
            A_bc, B_bc, C_bc = self.boundary_funcs['right'](phi_2d[-1, j], pos, self.current_time)
            
            if abs(B_bc) < 1e-14:
                # Dirichlet BC
                phi_boundary = C_bc / A_bc
                # Average phi to get temperature for diffusion coefficient
                phi_avg = 0.5 * (phi_2d[-1, j] + phi_boundary)
                T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
                sigma_R = self.rosseland_opacity_func(T_avg, self.x_centers[-1], self.y_centers[j])
                D_boundary = 1.0 / (3.0 * sigma_R)
                
                dx_half = self.x_faces[-1] - self.x_centers[-1]
                V = self.V_cells[-1, j]
                flux_coeff = (self.Ax_faces[-1, j] * D_boundary) / (V * dx_half)
                
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += flux_coeff * phi_boundary
            else:
                # Robin BC
                sigma_R = self.rosseland_opacity_func(T_2d[-1, j], self.x_centers[-1], self.y_centers[j])
                D_boundary = 1.0 / (3.0 * sigma_R)
                V = self.V_cells[-1, j]
                flux_coeff = (self.Ax_faces[-1, j] * D_boundary * A_bc) / (B_bc * V)
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += self.Ax_faces[-1, j] * D_boundary * C_bc / (B_bc * V)
        
        # Bottom boundary
        for i in range(nx):
            idx = index_2d_to_1d(i, 0, nx, ny)
            pos = (self.x_centers[i], self.y_faces[0])
            A_bc, B_bc, C_bc = self.boundary_funcs['bottom'](phi_2d[i, 0], pos, self.current_time)
            
            if abs(B_bc) < 1e-14:
                # Dirichlet BC
                phi_boundary = C_bc / A_bc
                # Average phi to get temperature for diffusion coefficient
                phi_avg = 0.5 * (phi_2d[i, 0] + phi_boundary)
                T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
                sigma_R = self.rosseland_opacity_func(T_avg, self.x_centers[i], self.y_centers[0])
                D_boundary = 1.0 / (3.0 * sigma_R)
                
                dy_half = self.y_centers[0] - self.y_faces[0]
                V = self.V_cells[i, 0]
                flux_coeff = (self.Ay_faces[i, 0] * D_boundary) / (V * dy_half)
                
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += flux_coeff * phi_boundary
            else:
                # Robin BC
                sigma_R = self.rosseland_opacity_func(T_2d[i, 0], self.x_centers[i], self.y_centers[0])
                D_boundary = 1.0 / (3.0 * sigma_R)
                V = self.V_cells[i, 0]
                flux_coeff = (self.Ay_faces[i, 0] * D_boundary * A_bc) / (B_bc * V)
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += self.Ay_faces[i, 0] * D_boundary * C_bc / (B_bc * V)
        
        # Top boundary
        for i in range(nx):
            idx = index_2d_to_1d(i, ny-1, nx, ny)
            pos = (self.x_centers[i], self.y_faces[-1])
            A_bc, B_bc, C_bc = self.boundary_funcs['top'](phi_2d[i, -1], pos, self.current_time)
            
            if abs(B_bc) < 1e-14:
                # Dirichlet BC
                phi_boundary = C_bc / A_bc
                # Average phi to get temperature for diffusion coefficient
                phi_avg = 0.5 * (phi_2d[i, -1] + phi_boundary)
                T_avg = (phi_avg / (A_RAD * C_LIGHT))**0.25
                sigma_R = self.rosseland_opacity_func(T_avg, self.x_centers[i], self.y_centers[-1])
                D_boundary = 1.0 / (3.0 * sigma_R)
                
                dy_half = self.y_faces[-1] - self.y_centers[-1]
                V = self.V_cells[i, -1]
                flux_coeff = (self.Ay_faces[i, -1] * D_boundary) / (V * dy_half)
                
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += flux_coeff * phi_boundary
            else:
                # Robin BC
                sigma_R = self.rosseland_opacity_func(T_2d[i, -1], self.x_centers[i], self.y_centers[-1])
                D_boundary = 1.0 / (3.0 * sigma_R)
                V = self.V_cells[i, -1]
                flux_coeff = (self.Ay_faces[i, -1] * D_boundary * A_bc) / (B_bc * V)
                A_lil[idx, idx] += flux_coeff
                rhs[idx] += self.Ay_faces[i, -1] * D_boundary * C_bc / (B_bc * V)
        
        # Convert back to CSR
        return A_lil.tocsr(), rhs
    
    def solve_T_equation(self, phi_np1, T_star, phi_prev, T_prev, theta=1.0, timing=False):
        """Solve T equation cell-by-cell (no spatial coupling)
        
        Equation 8.59b:
        (e(T_{n+1}) - e(T_n))/Δt = f·σ_P(φ̃ - acT★⁴) + (1-f)·Δe/Δt
        
        where φ̃ = θ·φ^{n+1} + (1-θ)·φ^n
        
        Returns:
        --------
        T_np1 : ndarray (n_total,)
            Solution T_{n+1}
        """
        import time
        t_start = time.time() if timing else None
        
        n_total = self.n_total
        dt = self.dt
        
        # Compute φ̃
        phi_tilde = theta * phi_np1 + (1.0 - theta) * phi_prev
        
        t0 = time.time() if timing else None
        # Compute material properties using vectorized operations
        props_star = self.compute_material_properties_vectorized(T_star, T_star)
        props_prev = self.compute_material_properties_vectorized(T_prev, T_prev)
        
        e_star = props_star['e']
        e_n = props_prev['e']
        sigma_P = props_star['sigma_P']
        f = props_star['f']
        t1 = time.time() if timing else None
        if timing:
            time_props = t1 - t0
        
        Delta_e = e_star - e_n
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        t0 = time.time() if timing else None
        # Vectorized solve for e^{n+1}
        e_np1 = e_n + dt * f * sigma_P * (phi_tilde - acT4_star) + (1.0 - f) * Delta_e
        
        # Vectorized solve for T^{n+1} from e^{n+1}
        T_np1 = self.compute_temperature_from_energy_vectorized(e_np1)
        t1 = time.time() if timing else None
        if timing:
            time_solve = t1 - t0
            time_total = time_props + time_solve
            print(f"        [solve_T_equation timing]")
            print(f"          Material properties:  {time_props*1000:.1f} ms ({100*time_props/time_total:.1f}%)")
            print(f"          Solve for T:          {time_solve*1000:.1f} ms ({100*time_solve/time_total:.1f}%)")
        
        return T_np1
    
    def newton_step(self, phi_prev_timestep, T_prev_timestep, source=None, verbose=False, timing=False):
        """Perform Newton iterations for coupled φ-T system
        
        Parameters:
        -----------
        phi_prev_timestep : ndarray
            φ^n from previous time step
        T_prev_timestep : ndarray
            T^n from previous time step
        source : ndarray or None
            Source term for phi equation
        verbose : bool
            Print iteration info
        timing : bool
            Print detailed timing breakdown
        
        Returns:
        --------
        phi_np1, T_np1 : ndarray
            Solutions φ^{n+1} and T_{n+1}
        """
        import time
        
        # Initial guess
        phi_star = phi_prev_timestep.copy()
        T_star = T_prev_timestep.copy()
        
        # Timing accumulators
        time_assemble = 0.0
        time_bc = 0.0
        time_solve_phi = 0.0
        time_solve_T = 0.0
        time_other = 0.0
        
        for k in range(self.max_newton_iter):
            t_iter_start = time.time()
            
            # Step 1: Solve φ equation
            t0 = time.time()
            A_phi, rhs_phi = self.assemble_phi_equation(
                phi_star, T_star, phi_prev_timestep, T_prev_timestep, 
                theta=self.theta, source=source, timing=timing)
            t1 = time.time()
            time_assemble += (t1 - t0)
            
            # Apply boundary conditions
            t0 = time.time()
            A_phi, rhs_phi = self.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
            t1 = time.time()
            time_bc += (t1 - t0)
            
            # Solve sparse system
            t0 = time.time()
            phi_np1 = spsolve(A_phi, rhs_phi)
            t1 = time.time()
            time_solve_phi += (t1 - t0)
            
            # Step 2: Solve T equation
            t0 = time.time()
            T_np1 = self.solve_T_equation(
                phi_np1, T_star, phi_prev_timestep, T_prev_timestep, theta=self.theta, timing=timing)
            t1 = time.time()
            time_solve_T += (t1 - t0)
            
            # Check validity and compute residuals
            t0 = time.time()
            if np.any(~np.isfinite(phi_np1)) or np.any(~np.isfinite(T_np1)):
                if verbose:
                    print(f"    Newton iteration {k+1} produced invalid values!")
                phi_np1 = phi_star + 0.01 * (phi_np1 - phi_star)
                T_np1 = T_star + 0.01 * (T_np1 - T_star)
            
            # Check for negative values and apply damping
            if np.any(phi_np1 <= 0) or np.any(T_np1 <= 0):
                alpha = 1.0
                for idx in range(len(phi_star)):
                    if phi_np1[idx] <= 0:
                        alpha = min(alpha, 0.9 * phi_star[idx] / (phi_star[idx] - phi_np1[idx]))
                    if T_np1[idx] <= 0:
                        alpha = min(alpha, 0.9 * T_star[idx] / (T_star[idx] - T_np1[idx]))
                
                alpha = max(0.001, min(alpha, 0.9))
                phi_np1 = phi_star + alpha * (phi_np1 - phi_star)
                T_np1 = T_star + alpha * (T_np1 - T_star)
                
                if verbose:
                    print(f"    Applied damping: alpha = {alpha:.3f}")
            
            # Compute residuals
            r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
            r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
            t1 = time.time()
            time_other += (t1 - t0)
            
            if verbose:
                print(f"    Newton iteration {k+1}: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
            
            if r_phi < self.newton_tol and r_T < self.newton_tol:
                if verbose:
                    print(f"    Newton converged in {k+1} iterations")
                if timing:
                    total_time = time_assemble + time_bc + time_solve_phi + time_solve_T + time_other
                    print(f"\n    Timing breakdown ({k+1} iterations, {total_time:.3f}s total):")
                    print(f"      Assemble phi matrix: {time_assemble:.3f}s ({100*time_assemble/total_time:.1f}%)")
                    print(f"      Apply BC:            {time_bc:.3f}s ({100*time_bc/total_time:.1f}%)")
                    print(f"      Solve phi (sparse):  {time_solve_phi:.3f}s ({100*time_solve_phi/total_time:.1f}%)")
                    print(f"      Solve T equation:    {time_solve_T:.3f}s ({100*time_solve_T/total_time:.1f}%)")
                    print(f"      Other (residuals):   {time_other:.3f}s ({100*time_other/total_time:.1f}%)")
                return phi_np1, T_np1
            
            # Update for next iteration
            phi_star = phi_np1.copy()
            T_star = T_np1.copy()
        
        if verbose:
            print(f"    Newton max iterations reached: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
        if timing:
            total_time = time_assemble + time_bc + time_solve_phi + time_solve_T + time_other
            print(f"\n    Timing breakdown (max {self.max_newton_iter} iterations, {total_time:.3f}s total):")
            print(f"      Assemble phi matrix: {time_assemble:.3f}s ({100*time_assemble/total_time:.1f}%)")
            print(f"      Apply BC:            {time_bc:.3f}s ({100*time_bc/total_time:.1f}%)")
            print(f"      Solve phi (sparse):  {time_solve_phi:.3f}s ({100*time_solve_phi/total_time:.1f}%)")
            print(f"      Solve T equation:    {time_solve_T:.3f}s ({100*time_solve_T/total_time:.1f}%)")
            print(f"      Other (residuals):   {time_other:.3f}s ({100*time_other/total_time:.1f}%)")
        return phi_star, T_star
    
    def time_step(self, n_steps=1, source=None, verbose=True, timing=False):
        """Advance solution by n_steps time steps
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        source : ndarray, callable, or None
            Source term for phi equation
        verbose : bool
            Print progress
        timing : bool
            Print detailed timing breakdown
        """
        for step in range(n_steps):
            if verbose:
                print(f"Time step {step+1}/{n_steps}, t = {self.current_time:.6e} ns")
            
            phi_prev = self.phi.copy()
            T_prev = self.T.copy()
            
            # Evaluate source at current time
            if callable(source):
                source_at_t = source(self.current_time)
            else:
                source_at_t = source
            
            # Newton iterations
            self.phi, self.T = self.newton_step(phi_prev, T_prev, source=source_at_t, verbose=verbose, timing=timing)
            
            # Update old values
            self.phi_old = phi_prev.copy()
            self.T_old = T_prev.copy()
            
            # Update current time
            self.current_time += self.dt
    
    def time_step_trbdf2(self, n_steps=1, Lambda=None, source=None, verbose=True):
        """Advance solution using TR-BDF2 method
        
        TR-BDF2 is a two-stage composite method:
        Stage 1: Trapezoidal rule from t^n to t^{n+Λ}
        Stage 2: BDF2 from t^n, t^{n+Λ} to t^{n+1}
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        Lambda : float, optional
            Intermediate time fraction (default: 2 - sqrt(2) ≈ 0.586)
        source : ndarray, callable, or None, optional
            Source term for phi equation
        verbose : bool
            Print progress
        """
        if Lambda is None:
            Lambda = 2.0 - np.sqrt(2.0)  # Optimal value for L-stability
        
        # Store original parameters
        original_dt = self.dt
        original_theta = self.theta
        
        for step in range(n_steps):
            if verbose:
                print(f"TR-BDF2 step {step+1}/{n_steps}, t = {self.current_time:.6e} ns")
            
            # Store solution at t^n
            phi_n = self.phi.copy()
            T_n = self.T.copy()
            
            # Evaluate source at current time if it's a function
            if callable(source):
                source_stage1 = source(self.current_time)
                source_stage2 = source(self.current_time)
            else:
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
            self.phi, self.T = self.newton_step_bdf2(
                phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=source_stage2, verbose=verbose)
            
            # Update old values for next time step
            self.phi_old = phi_n.copy()
            self.T_old = T_n.copy()
            
            # Update current time
            self.current_time += original_dt
        
        # Restore original parameters
        self.dt = original_dt
        self.theta = original_theta
    
    def newton_step_bdf2(self, phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=None, verbose=False):
        """Perform Newton iterations for BDF2 stage of TR-BDF2
        
        Implements equations (8.61a) and (8.61b) for the second stage
        
        Parameters:
        -----------
        phi_n, T_n : ndarray
            Solutions at time level n
        phi_Lambda, T_Lambda : ndarray
            Solutions at intermediate time level n+Λ
        Lambda : float
            TR-BDF2 parameter
        source : ndarray or None
            Source term
        verbose : bool
            Print iteration info
        
        Returns:
        --------
        phi_np1, T_np1 : ndarray
            Solutions at time level n+1
        """
        # Initial guess: use intermediate solution
        phi_star = self.phi.copy()
        T_star = self.T.copy()
        
        for k in range(self.max_newton_iter):
            # Step 1: Solve equation 8.61a for φ^{n+1}
            A_phi, rhs_phi = self.assemble_phi_equation_trbdf2(
                phi_star, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source=source)
            
            # Apply boundary conditions
            A_phi, rhs_phi = self.apply_boundary_conditions_phi(A_phi, rhs_phi, phi_star)
            
            # Solve sparse system
            phi_np1 = spsolve(A_phi, rhs_phi)
            
            # Step 2: Solve equation 8.61b for T_{n+1}
            T_np1 = self.solve_T_equation_trbdf2(
                phi_np1, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda)
            
            # Check validity
            if np.any(~np.isfinite(phi_np1)) or np.any(~np.isfinite(T_np1)):
                if verbose:
                    print(f"    BDF2 Newton iteration {k+1} produced invalid values!")
                phi_np1 = phi_star + 0.01 * (phi_np1 - phi_star)
                T_np1 = T_star + 0.01 * (T_np1 - T_star)
            
            # Check for negative values and apply damping
            if np.any(phi_np1 <= 0) or np.any(T_np1 <= 0):
                alpha = 1.0
                for idx in range(len(phi_star)):
                    if phi_np1[idx] <= 0:
                        alpha = min(alpha, 0.9 * phi_star[idx] / (phi_star[idx] - phi_np1[idx]))
                    if T_np1[idx] <= 0:
                        alpha = min(alpha, 0.9 * T_star[idx] / (T_star[idx] - T_np1[idx]))
                
                alpha = max(0.1, min(alpha, 0.9))
                phi_np1 = phi_star + alpha * (phi_np1 - phi_star)
                T_np1 = T_star + alpha * (T_np1 - T_star)
                
                if verbose:
                    print(f"    BDF2 damping: alpha = {alpha:.3f}")
            
            # Compute residuals
            r_phi = np.linalg.norm(phi_np1 - phi_star) / (np.linalg.norm(phi_star) + 1e-14)
            r_T = np.linalg.norm(T_np1 - T_star) / (np.linalg.norm(T_star) + 1e-14)
            
            if verbose:
                print(f"    BDF2 Newton iteration {k+1}: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
            
            if r_phi < self.newton_tol and r_T < self.newton_tol:
                if verbose:
                    print(f"    BDF2 Newton converged in {k+1} iterations")
                return phi_np1, T_np1
            
            # Update for next iteration
            phi_star = phi_np1.copy()
            T_star = T_np1.copy()
        
        if verbose:
            print(f"    BDF2 Newton max iterations: r_φ={r_phi:.2e}, r_T={r_T:.2e}")
        return phi_star, T_star
    
    def assemble_phi_equation_trbdf2(self, phi_star, T_star, phi_n, T_n, 
                                     phi_Lambda, T_Lambda, Lambda, source=None):
        """Assemble sparse matrix for φ equation in TR-BDF2 BDF2 stage
        
        Equation 8.61a:
        (1/c·Δt)·[(2-Λ)/(1-Λ)·φⁿ⁺¹ - 1/(Λ(1-Λ))·φⁿ⁺ᴧ + (1-Λ)/Λ·φⁿ] + ∇·D∇φⁿ⁺¹ =
            f_TB·σ_P(acT_★⁴ - φⁿ⁺¹) - (1-f_TB)·Δe/Δt + S
        
        Returns:
        --------
        A_sparse : scipy.sparse matrix
            Sparse matrix for φ^{n+1}
        rhs : ndarray
            Right-hand side vector
        """
        n_total = self.n_total
        nx = self.nx_cells
        ny = self.ny_cells
        dt = self.dt
        
        # BDF2 coefficients from equation (8.62)
        c_np1 = (2.0 - Lambda) / (1.0 - Lambda)  # Coefficient of solution at n+1
        c_nL = -1.0 / (Lambda * (1.0 - Lambda))   # Coefficient of solution at n+Λ
        c_n = (1.0 - Lambda) / Lambda             # Coefficient of solution at n
        
        # Reshape arrays to 2D
        phi_star_2d = phi_star.reshape((nx, ny), order='C')
        T_star_2d = T_star.reshape((nx, ny), order='C')
        phi_n_2d = phi_n.reshape((nx, ny), order='C')
        phi_Lambda_2d = phi_Lambda.reshape((nx, ny), order='C')
        
        # Evaluate material energies using vectorized operations
        X_mesh, Y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        T_star_2d_full = T_star.reshape((nx, ny), order='C')
        T_Lambda_2d = T_Lambda.reshape((nx, ny), order='C')
        T_n_2d = T_n.reshape((nx, ny), order='C')
        
        e_star_2d = self.material_energy_func(T_star_2d_full, X_mesh, Y_mesh)
        e_nL_2d = self.material_energy_func(T_Lambda_2d, X_mesh, Y_mesh)
        e_n_2d = self.material_energy_func(T_n_2d, X_mesh, Y_mesh)
        sigma_P_2d = self.planck_opacity_func(T_star_2d_full, X_mesh, Y_mesh)
        
        # Handle scalar returns
        if np.isscalar(e_star_2d):
            e_star_2d = np.full((nx, ny), e_star_2d)
        if np.isscalar(e_nL_2d):
            e_nL_2d = np.full((nx, ny), e_nL_2d)
        if np.isscalar(e_n_2d):
            e_n_2d = np.full((nx, ny), e_n_2d)
        if np.isscalar(sigma_P_2d):
            sigma_P_2d = np.full((nx, ny), sigma_P_2d)
        
        # Compute f_TB for each cell
        f_TB_2d = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                f_TB_2d[i, j] = self.get_f_factor_trbdf2(T_star_2d_full[i, j], 
                                                         self.x_centers[i], self.y_centers[j], dt, Lambda)
        
        # Flatten to 1D
        e_star = e_star_2d.flatten(order='C')
        e_nL = e_nL_2d.flatten(order='C')
        e_n = e_n_2d.flatten(order='C')
        sigma_P = sigma_P_2d.flatten(order='C')
        f_TB = f_TB_2d.flatten(order='C')
        
        # Compute Δe from equation (8.62)
        Delta_e = c_np1 * e_star + c_nL * e_nL + c_n * e_n
        
        # Compute acT⁴
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Compute diffusion coefficients at all faces
        # X-direction faces
        D_x_faces = np.zeros((nx + 1, ny))
        for i in range(nx + 1):
            for j in range(ny):
                y_face = self.y_centers[j]
                if i == 0 or i == nx:
                    # Boundary face - use simple approach
                    if i == 0:
                        x_face = self.x_faces[0]
                        T_face = T_star_2d[0, j]
                    else:
                        x_face = self.x_faces[-1]
                        T_face = T_star_2d[-1, j]
                    D_x_faces[i, j] = self.get_diffusion_coefficient(T_face, x_face, y_face)
                else:
                    # Interior face - use harmonic mean for heterogeneous materials
                    T_left = T_star_2d[i-1, j]
                    T_right = T_star_2d[i, j]
                    x_left = self.x_centers[i-1]
                    x_right = self.x_centers[i]
                    y_left = self.y_centers[j]
                    y_right = self.y_centers[j]
                    phi_left = phi_star_2d[i-1, j]
                    phi_right = phi_star_2d[i, j]
                    x_face = self.x_faces[i]
                    dx_left = x_face - x_left
                    dx_right = x_right - x_face
                    D_x_faces[i, j] = self.get_harmonic_diffusion_coefficient(
                        T_left, T_right, x_left, y_left, x_right, y_right,
                        phi_left, phi_right, dx_left, dx_right)
        
        # Y-direction faces
        D_y_faces = np.zeros((nx, ny + 1))
        for i in range(nx):
            for j in range(ny + 1):
                x_face = self.x_centers[i]
                if j == 0 or j == ny:
                    # Boundary face - use simple approach
                    if j == 0:
                        y_face = self.y_faces[0]
                        T_face = T_star_2d[i, 0]
                    else:
                        y_face = self.y_faces[-1]
                        T_face = T_star_2d[i, -1]
                    D_y_faces[i, j] = self.get_diffusion_coefficient(T_face, x_face, y_face)
                else:
                    # Interior face - use harmonic mean for heterogeneous materials
                    T_left = T_star_2d[i, j-1]
                    T_right = T_star_2d[i, j]
                    x_left = self.x_centers[i]
                    x_right = self.x_centers[i]
                    y_left = self.y_centers[j-1]
                    y_right = self.y_centers[j]
                    phi_left = phi_star_2d[i, j-1]
                    phi_right = phi_star_2d[i, j]
                    y_face = self.y_faces[j]
                    dy_left = y_face - y_left
                    dy_right = y_right - y_face
                    D_y_faces[i, j] = self.get_harmonic_diffusion_coefficient(
                        T_left, T_right, x_left, y_left, x_right, y_right,
                        phi_left, phi_right, dy_left, dy_right)
        
        # Assemble matrix using Numba-accelerated function (COO format)
        rows, cols, data = assemble_trbdf2_matrix_coo(
            nx, ny, dt, c_np1, C_LIGHT,
            D_x_faces, D_y_faces,
            self.Ax_faces, self.Ay_faces,
            self.x_centers, self.y_centers,
            self.V_cells, sigma_P, f_TB
        )
        
        # Build sparse matrix from COO format
        from scipy.sparse import coo_matrix
        A_csr = coo_matrix((data, (rows, cols)), shape=(n_total, n_total)).tocsr()
        
        # Compute RHS vector
        rhs = np.zeros(n_total)
        rhs = (-c_nL * phi_Lambda - c_n * phi_n) / (C_LIGHT * dt)
        rhs += f_TB * sigma_P * acT4_star
        rhs -= (1.0 - f_TB) * Delta_e / dt
        
        # Add external source if provided
        if source is not None:
            if isinstance(source, np.ndarray):
                if source.shape == (nx, ny):
                    rhs += source.flatten(order='C')
                elif source.shape == (n_total,):
                    rhs += source
            else:
                rhs += source
        
        return A_csr, rhs
    
    def solve_T_equation_trbdf2(self, phi_np1, T_star, phi_n, T_n, phi_Lambda, T_Lambda, Lambda):
        """Solve T equation for TR-BDF2 BDF2 stage
        
        Equation 8.61b:
        (1/Δt)·[(2-Λ)/(1-Λ)·e(T_{n+1}) - 1/(Λ(1-Λ))·e(T_{n+Λ}) + (1-Λ)/Λ·e(T_n)] =
            f_TB·σ_P(φ^{n+1} - acT_★⁴) + (1-f_TB)·Δe/Δt
        
        Returns:
        --------
        T_np1 : ndarray
            Solution T_{n+1}
        """
        n_total = self.n_total
        dt = self.dt
        T_np1 = np.zeros(n_total)
        
        # BDF2 coefficients
        c_np1 = (2.0 - Lambda) / (1.0 - Lambda)
        c_nL = -1.0 / (Lambda * (1.0 - Lambda))
        c_n = (1.0 - Lambda) / Lambda
        
        # Evaluate material energies using vectorized operations
        X_mesh, Y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        
        T_Lambda_2d = T_Lambda.reshape((self.nx_cells, self.ny_cells), order='C')
        T_n_2d = T_n.reshape((self.nx_cells, self.ny_cells), order='C')
        T_star_2d = T_star.reshape((self.nx_cells, self.ny_cells), order='C')
        
        e_nL_2d = self.material_energy_func(T_Lambda_2d, X_mesh, Y_mesh)
        e_n_2d = self.material_energy_func(T_n_2d, X_mesh, Y_mesh)
        e_star_2d = self.material_energy_func(T_star_2d, X_mesh, Y_mesh)
        sigma_P_2d = self.planck_opacity_func(T_star_2d, X_mesh, Y_mesh)
        
        # Handle scalar returns
        if np.isscalar(e_nL_2d):
            e_nL_2d = np.full((self.nx_cells, self.ny_cells), e_nL_2d)
        if np.isscalar(e_n_2d):
            e_n_2d = np.full((self.nx_cells, self.ny_cells), e_n_2d)
        if np.isscalar(e_star_2d):
            e_star_2d = np.full((self.nx_cells, self.ny_cells), e_star_2d)
        if np.isscalar(sigma_P_2d):
            sigma_P_2d = np.full((self.nx_cells, self.ny_cells), sigma_P_2d)
        
        # Compute f_TB for each cell
        f_TB_2d = np.zeros((self.nx_cells, self.ny_cells))
        for i in range(self.nx_cells):
            for j in range(self.ny_cells):
                f_TB_2d[i, j] = self.get_f_factor_trbdf2(T_star_2d[i, j], 
                                                         self.x_centers[i], self.y_centers[j], dt, Lambda)
        
        # Flatten to 1D
        e_nL = e_nL_2d.flatten(order='C')
        e_n = e_n_2d.flatten(order='C')
        e_star = e_star_2d.flatten(order='C')
        sigma_P = sigma_P_2d.flatten(order='C')
        f_TB = f_TB_2d.flatten(order='C')
        
        Delta_e = c_np1 * e_star + c_nL * e_nL + c_n * e_n
        acT4_star = A_RAD * C_LIGHT * T_star**4
        
        # Solve for T_{n+1} at each cell (vectorized)
        rhs_T = (-c_nL * e_nL - c_n * e_n) / dt
        rhs_T += f_TB * sigma_P * (phi_np1 - acT4_star)
        rhs_T += (1.0 - f_TB) * Delta_e / dt
        
        # e_{n+1} = (Δt/c_np1) * rhs_T
        e_np1 = (dt / c_np1) * rhs_T
        
        # Invert material energy to get T (vectorized)
        T_np1 = self.compute_temperature_from_energy_vectorized(e_np1)
        
        return T_np1
    
    def plot_solution(self, figsize=(12, 5), save_path=None, show=False):
        """Plot current solution φ and T
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            If provided, save figure to this path
        show : bool
            If True, call plt.show() (default: False for non-interactive use)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        phi_2d = self.get_phi_2d()
        T_2d = self.get_T_2d()
        
        # Plot φ
        im1 = axes[0].pcolormesh(self.x_centers, self.y_centers, phi_2d.T, 
                                  shading='auto', cmap='viridis')
        axes[0].set_xlabel('x' if self.geometry == 'cartesian' else 'r')
        axes[0].set_ylabel('y' if self.geometry == 'cartesian' else 'z')
        axes[0].set_title('φ = E_r × c')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot T
        im2 = axes[1].pcolormesh(self.x_centers, self.y_centers, T_2d.T,
                                  shading='auto', cmap='hot')
        axes[1].set_xlabel('x' if self.geometry == 'cartesian' else 'r')
        axes[1].set_ylabel('y' if self.geometry == 'cartesian' else 'z')
        axes[1].set_title('Temperature T (keV)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("2D Non-Equilibrium Radiation Diffusion Solver - Example")
    print("="*70)
    
    # Example 1: Cartesian geometry with hot spot
    print("\nExample 1: Cartesian (x-y) geometry with hot spot")
    print("-" * 70)
    
    solver_cart = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=0.0, x_max=2.0, nx_cells=30,
        y_min=0.0, y_max=2.0, ny_cells=30,
        geometry='cartesian',
        dt=0.01,
        theta=1.0,
        max_newton_iter=20,
        newton_tol=1e-6
    )
    
    # Initial condition: hot spot in center
    def T_init_hot_spot(x, y):
        r2 = (x - 1.0)**2 + (y - 1.0)**2
        return 0.3 + 0.7 * np.exp(-20.0 * r2)
    
    solver_cart.set_initial_condition(T_init=T_init_hot_spot)
    
    print("Initial condition set. Running 5 time steps...")
    solver_cart.time_step(n_steps=5, verbose=True)
    
    # Get solution statistics
    x, y, phi_2d, T_2d = solver_cart.get_solution()
    print(f"\nSolution statistics:")
    print(f"  φ range: [{phi_2d.min():.4e}, {phi_2d.max():.4e}]")
    print(f"  T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
    
    print("\nSaving plot to 'cartesian_example.png'...")
    solver_cart.plot_solution(save_path='cartesian_example.png', show=False)
    
    # Example 2: Cylindrical geometry
    print("\n" + "="*70)
    print("Example 2: Cylindrical (r-z) geometry")
    print("-" * 70)
    
    solver_cyl = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=0.1, x_max=2.0, nx_cells=25,
        y_min=0.0, y_max=3.0, ny_cells=30,
        geometry='cylindrical',
        dt=0.01,
        theta=1.0,
        max_newton_iter=20,
        newton_tol=1e-6
    )
    
    # Initial condition: hot region at small r, mid-z
    def T_init_cyl(r, z):
        return 0.3 + 0.7 * np.exp(-2.0 * r**2) * np.exp(-2.0 * (z - 1.5)**2)
    
    solver_cyl.set_initial_condition(T_init=T_init_cyl)
    
    print("Initial condition set. Running 5 time steps...")
    solver_cyl.time_step(n_steps=5, verbose=True)
    
    # Get solution statistics
    x, y, phi_2d, T_2d = solver_cyl.get_solution()
    print(f"\nSolution statistics:")
    print(f"  φ range: [{phi_2d.min():.4e}, {phi_2d.max():.4e}]")
    print(f"  T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
    
    print("\nSaving plot to 'cylindrical_example.png'...")
    solver_cyl.plot_solution(save_path='cylindrical_example.png', show=False)
    
    # Example 3: TR-BDF2 time integration
    print("\n" + "="*70)
    print("Example 3: TR-BDF2 time integration (Cartesian)")
    print("-" * 70)
    
    solver_trbdf2 = NonEquilibriumRadiationDiffusionSolver2D(
        x_min=0.0, x_max=2.0, nx_cells=25,
        y_min=0.0, y_max=2.0, ny_cells=25,
        geometry='cartesian',
        dt=0.02,  # Larger time step with TR-BDF2
        theta=1.0,
        max_newton_iter=20,
        newton_tol=1e-6
    )
    
    # Initial condition: same hot spot as Example 1
    solver_trbdf2.set_initial_condition(T_init=T_init_hot_spot)
    
    print("Initial condition set. Running 5 TR-BDF2 steps...")
    solver_trbdf2.time_step_trbdf2(n_steps=5, verbose=True)
    
    # Get solution statistics
    x, y, phi_2d, T_2d = solver_trbdf2.get_solution()
    print(f"\nSolution statistics:")
    print(f"  φ range: [{phi_2d.min():.4e}, {phi_2d.max():.4e}]")
    print(f"  T range: [{T_2d.min():.4f}, {T_2d.max():.4f}] keV")
    
    print("\nSaving plot to 'trbdf2_example.png'...")
    solver_trbdf2.plot_solution(save_path='trbdf2_example.png', show=False)
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
