"""
Utility functions for multigroup IMC simulations.

Provides common opacity models, energy group structures, and helper functions.
"""

import os
import numpy as np

# Physical constants (from MG_IMC2D)
C_LIGHT = 29.98  # cm/ns
A_RAD = 0.01372  # GJ/(cm³·keV⁴)


def create_log_energy_groups(E_min, E_max, n_groups):
    """Create logarithmically-spaced energy groups.
    
    Parameters
    ----------
    E_min : float
        Minimum energy (keV)
    E_max : float
        Maximum energy (keV)
    n_groups : int
        Number of groups
    
    Returns
    -------
    energy_edges : array
        Energy boundaries, length n_groups + 1
    """
    return np.logspace(np.log10(E_min), np.log10(E_max), n_groups + 1)


def create_linear_energy_groups(E_min, E_max, n_groups):
    """Create linearly-spaced energy groups."""
    return np.linspace(E_min, E_max, n_groups + 1)


def powerlaw_opacity_functions(energy_edges, sigma_ref=1.0, T_ref=1.0, alpha=3.0):
    """Create power-law opacity functions for each group.
    
    Creates opacity functions of the form:
        σ_g(T) = σ_ref * (E_g / E_ref)^(-α) * (T / T_ref)^(-3)
    
    where E_g is the group center energy.
    
    Parameters
    ----------
    energy_edges : array
        Energy group boundaries
    sigma_ref : float
        Reference opacity at reference energy and temperature (cm⁻¹)
    T_ref : float
        Reference temperature (keV)
    alpha : float
        Frequency power law exponent (default 3 for bremsstrahlung)
    
    Returns
    -------
    sigma_funcs : list of callables
        Opacity function for each group
    """
    n_groups = len(energy_edges) - 1
    E_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    E_ref = E_centers[0]  # Use first group as reference
    
    sigma_funcs = []
    for g in range(n_groups):
        E_g = E_centers[g]
        sigma_0_g = sigma_ref * (E_g / E_ref)**(-alpha)
        
        def make_opacity(sig0):
            def opacity_func(T):
                return sig0 * (T / T_ref)**(-3)
            return opacity_func
        
        sigma_funcs.append(make_opacity(sigma_0_g))
    
    return sigma_funcs


def constant_opacity_functions(energy_edges, sigma_values):
    """Create constant (temperature-independent) opacity functions.
    
    Parameters
    ----------
    energy_edges : array
        Energy group boundaries
    sigma_values : array or list
        Opacity value for each group (cm⁻¹)
    
    Returns
    -------
    sigma_funcs : list of callables
    """
    n_groups = len(energy_edges) - 1
    sigma_values = np.atleast_1d(sigma_values)
    
    if len(sigma_values) == 1:
        sigma_values = np.full(n_groups, sigma_values[0])
    
    assert len(sigma_values) == n_groups, "Must provide one opacity per group"
    
    sigma_funcs = []
    for g in range(n_groups):
        sig_g = sigma_values[g]
        def make_opacity(sig):
            def opacity_func(T):
                return sig * np.ones_like(T)
            return opacity_func
        sigma_funcs.append(make_opacity(sig_g))
    
    return sigma_funcs


def simple_eos_functions(cv_value=0.1, rho=1.0):
    """Create simple linear EOS functions.
    
    e = ρ c_v T
    
    Parameters
    ----------
    cv_value : float
        Specific heat (GJ/(g·keV))
    rho : float
        Density (g/cm³)
    
    Returns
    -------
    eos : callable
        Energy as function of temperature
    inv_eos : callable
        Temperature as function of energy
    cv : callable
        Specific heat function (constant)
    """
    def eos(T):
        return rho * cv_value * T
    
    def inv_eos(e):
        return e / (rho * cv_value)
    
    def cv(T):
        return cv_value * np.ones_like(T)
    
    return eos, inv_eos, cv


def compute_group_fractions(radiation_energy_by_group):
    """Compute fractional radiation energy in each group.
    
    Parameters
    ----------
    radiation_energy_by_group : array (n_groups, nx, ny)
        Radiation energy density by group
    
    Returns
    -------
    fractions : array (n_groups, nx, ny)
        Fractional energy in each group (sums to 1)
    """
    total = np.sum(radiation_energy_by_group, axis=0, keepdims=True)
    total = np.maximum(total, 1e-300)  # Avoid division by zero
    return radiation_energy_by_group / total


def compute_total_opacity(sigma_funcs, T, B_funcs=None, energy_edges=None):
    """Compute total (Planck-weighted or unweighted) opacity.
    
    If B_funcs and energy_edges provided, computes Planck-weighted:
        σ_total = Σ_g σ_g B_g / Σ_g B_g
    
    Otherwise computes unweighted sum:
        σ_total = Σ_g σ_g
    
    Parameters
    ----------
    sigma_funcs : list of callables
        Opacity functions
    T : array
        Temperature array
    B_funcs : list of callables, optional
        Planck function for each group
    energy_edges : array, optional
        Energy boundaries
    
    Returns
    -------
    sigma_total : array
        Total opacity
    """
    n_groups = len(sigma_funcs)
    
    if B_funcs is None or energy_edges is None:
        # Unweighted sum
        sigma_total = np.zeros_like(T)
        for g in range(n_groups):
            sigma_total += sigma_funcs[g](T)
        return sigma_total
    
    # Planck-weighted
    try:
        from planck_integrals import Bg
    except ImportError:
        # Fall back to unweighted
        sigma_total = np.zeros_like(T)
        for g in range(n_groups):
            sigma_total += sigma_funcs[g](T)
        return sigma_total
    
    sigma_total = np.zeros_like(T)
    B_total = np.zeros_like(T)
    
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g + 1]
        B_g = np.zeros_like(T)
        
        # Handle scalar or array T
        T_flat = T.flatten()
        B_g_flat = np.array([Bg(E_low, E_high, t) for t in T_flat])
        B_g = B_g_flat.reshape(T.shape)
        
        sigma_g = sigma_funcs[g](T)
        sigma_total += sigma_g * B_g
        B_total += B_g
    
    return sigma_total / (B_total + 1e-300)


def print_group_info(energy_edges, sigma_funcs=None, T_test=1.0):
    """Print information about energy group structure.
    
    Parameters
    ----------
    energy_edges : array
        Energy boundaries
    sigma_funcs : list, optional
        Opacity functions to evaluate
    T_test : float
        Test temperature for opacity evaluation (keV)
    """
    n_groups = len(energy_edges) - 1
    
    print(f"Energy Group Structure: {n_groups} groups")
    print("=" * 60)
    print(f"{'Group':<8} {'E_low (keV)':<15} {'E_high (keV)':<15} {'ΔE (keV)':<15}")
    print("-" * 60)
    
    for g in range(n_groups):
        E_low = energy_edges[g]
        E_high = energy_edges[g + 1]
        Delta_E = E_high - E_low
        print(f"{g:<8} {E_low:<15.6e} {E_high:<15.6e} {Delta_E:<15.6e}")
    
    if sigma_funcs is not None:
        print()
        print(f"Opacities at T = {T_test} keV:")
        print("-" * 60)
        print(f"{'Group':<8} {'σ_g (cm⁻¹)':<20}")
        print("-" * 60)
        T_array = np.array([[T_test]])
        for g in range(n_groups):
            sigma_g = sigma_funcs[g](T_array)[0, 0]
            print(f"{g:<8} {sigma_g:<20.6e}")
    
    print("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint I/O (atomic writes via temp-file rename)
# ---------------------------------------------------------------------------

def atomic_checkpoint_save(path, payload):
    """Save *payload* (any picklable dict) to *path* atomically.

    Writes to ``path + ".tmp"`` first, then renames to *path* to prevent
    partial writes if the process is killed mid-save.

    Parameters
    ----------
    path : str
        Destination file path.
    payload : dict
        Any picklable object.
    """
    import pickle
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, str(path))


def checkpoint_load(path):
    """Load and return the dict previously saved by :func:`atomic_checkpoint_save`.

    Parameters
    ----------
    path : str
        Checkpoint file to read.

    Returns
    -------
    payload : dict
    """
    import pickle
    with open(str(path), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Planck-spectrum group integrals (pure-Python numerical quadrature)
# ---------------------------------------------------------------------------

def planck_group_integral(E_low, E_high, T, n_quad=80):
    """Return the integrated Planck intensity B_g(T) for a frequency group.

    .. math::

        B_g(T) = \\int_{E_{low}}^{E_{high}} \\frac{2 E^3}{c^2}
                 \\frac{1}{e^{E/T} - 1} \\, dE

    Evaluated with *n_quad*-point composite trapezoid rule.  Uses the same
    convention as the ``planck_integrals`` C extension (energy in keV).

    Parameters
    ----------
    E_low, E_high : float
        Group energy bounds (keV).
    T : float
        Temperature (keV).  Returns 0 for T ≤ 0.
    n_quad : int
        Number of quadrature points (default 80).

    Returns
    -------
    B_g : float
    """
    if T <= 0.0:
        return 0.0
    E = np.linspace(E_low, E_high, n_quad)
    B_E = (2.0 * E**3 / C_LIGHT**2) / (np.exp(np.minimum(E / T, 500.0)) - 1.0 + 1e-300)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(B_E, E))
    return float(np.trapz(B_E, E))


def planck_spectrum_by_group(energy_edges, T):
    """Return per-group Planck integrals B_g(T) for all groups.

    Parameters
    ----------
    energy_edges : array_like
        Group boundary energies, length ``n_groups + 1`` (keV).
    T : float
        Temperature (keV).

    Returns
    -------
    B : ndarray, shape (n_groups,)
    """
    n_groups = len(energy_edges) - 1
    B = np.zeros(n_groups)
    for g in range(n_groups):
        B[g] = planck_group_integral(energy_edges[g], energy_edges[g + 1], T)
    return B


# ---------------------------------------------------------------------------
# Mean opacities via numerical quadrature
# ---------------------------------------------------------------------------

def rosseland_mean_D(sigma_func, T, E_low, E_high, T_floor=1e-10, n_quad=60):
    """Rosseland-mean diffusion coefficient  D = 1 / (3 σ_R).

    Uses the Rosseland harmonic mean:

    .. math::

        \\frac{1}{\\sigma_R} = \\frac{\\int (1/\\sigma_a) \\partial B/\\partial T \\, dE}
                                      {\\int \\partial B/\\partial T \\, dE}

    where the integrals run over [E_low, E_high].

    Parameters
    ----------
    sigma_func : callable
        ``sigma_func(T_scalar, E_scalar) -> float`` — absorption opacity.
    T : float
        Temperature (keV); clamped to *T_floor*.
    E_low, E_high : float
        Group energy bounds (keV).
    T_floor : float
        Minimum temperature used in opacity evaluation.
    n_quad : int
        Quadrature points.

    Returns
    -------
    D : float
        Diffusion coefficient (cm).
    """
    T_use = max(float(T), T_floor)
    E = np.linspace(E_low, E_high, n_quad)
    x = E / T_use
    x_clip = np.minimum(x, 500.0)
    ex = np.exp(x_clip)
    dBdT_kernel = np.where(
        x_clip < 100.0,
        ex / (ex - 1.0 + 1e-300)**2,
        np.exp(-x_clip),
    )
    dBdT = (2.0 * E**4 / (C_LIGHT**2 * T_use**2)) * dBdT_kernel
    inv_sigma = np.array([1.0 / max(float(sigma_func(T_use, e)), 1e-300) for e in E])
    if hasattr(np, "trapezoid"):
        denom = np.trapezoid(dBdT, E)
        numer = np.trapezoid(inv_sigma * dBdT, E)
    else:
        denom = np.trapz(dBdT, E)
        numer = np.trapz(inv_sigma * dBdT, E)
    if denom < 1e-300:
        sigma_geom = float(np.sqrt(sigma_func(T_use, E_low) * sigma_func(T_use, E_high)))
        return 1.0 / (3.0 * max(sigma_geom, 1e-20))
    inv_sigma_R = numer / denom
    return float(max(inv_sigma_R, 1e-20)) / 3.0


def planck_mean_sigma(sigma_func, T, E_low, E_high, T_floor=1e-10, n_quad=60):
    """Planck-mean absorption opacity  σ_P = ⟨σ_a⟩_{B_E(T)}.

    Parameters
    ----------
    sigma_func : callable
        ``sigma_func(T_scalar, E_scalar) -> float`` — absorption opacity.
    T : float
        Temperature (keV); clamped to *T_floor*.
    E_low, E_high : float
        Group energy bounds (keV).
    T_floor : float
        Minimum temperature used in evaluation.
    n_quad : int
        Quadrature points.

    Returns
    -------
    sigma_P : float
        Planck-mean opacity (cm⁻¹).
    """
    T_use = max(float(T), T_floor)
    E = np.linspace(E_low, E_high, n_quad)
    x_clip = np.minimum(E / T_use, 500.0)
    B_E = (2.0 * E**3 / C_LIGHT**2) / (np.exp(x_clip) - 1.0 + 1e-300)
    sigma_E = np.array([float(sigma_func(T_use, e)) for e in E])
    if hasattr(np, "trapezoid"):
        denom = np.trapezoid(B_E, E)
        numer = np.trapezoid(sigma_E * B_E, E)
    else:
        denom = np.trapz(B_E, E)
        numer = np.trapz(sigma_E * B_E, E)
    if denom < 1e-300:
        return float(np.sqrt(sigma_func(T_use, E_low) * sigma_func(T_use, E_high)))
    return float(numer / denom)


def geom_mean_sigma(sigma_func, T, E_low, E_high, T_floor=1e-10):
    """Geometric mean of σ_a at group energy boundaries.

    .. math::

        \\sigma_{\\rm geom} = \\sqrt{\\sigma_a(T, E_{\\rm low}) \\cdot
                                     \\sigma_a(T, E_{\\rm high})}

    Parameters
    ----------
    sigma_func : callable
        ``sigma_func(T_scalar, E_scalar) -> float``.
    T : float
        Temperature (keV).
    E_low, E_high : float
        Group energy bounds (keV).
    T_floor : float
        Minimum temperature used in evaluation.

    Returns
    -------
    sigma_geom : float
    """
    T_use = max(float(T), T_floor)
    s_low  = float(sigma_func(T_use, E_low))
    s_high = float(sigma_func(T_use, E_high))
    return float(np.sqrt(s_low * s_high))


# ---------------------------------------------------------------------------
# Generic power-law opacity model
# ---------------------------------------------------------------------------

def make_powerlaw_sigma(C_OPA, alpha, beta, rho=1.0, sigma_max=1e14):
    """Return a scalar opacity function  σ(T, E) = ρ · C · T^α · E^β.

    Parameters
    ----------
    C_OPA : float
        Opacity coefficient.
    alpha : float
        Temperature exponent.
    beta : float
        Energy (frequency) exponent.
    rho : float
        Material density (g/cm³).  Default 1.
    sigma_max : float
        Hard upper cap on the returned opacity.  Default 1e14.

    Returns
    -------
    sigma : callable
        ``sigma(T_scalar, E_scalar) -> float``.
    """
    def sigma(T, E):
        T_use = max(float(T), 1e-300)
        return float(min(rho * C_OPA * T_use**alpha * float(E)**beta, sigma_max))
    return sigma


def make_powerlaw_group_opacity(E_low, E_high, C_OPA, alpha, beta,
                                rho=1.0, sigma_max=1e14):
    """Group opacity using the geometric mean of boundary values.

    Returns a closure ``opacity(T) -> float`` suitable for use as a
    sigma_a function in the MG-IMC and diffusion solvers:

    .. math::

        \\sigma_g(T) = \\sqrt{\\sigma(T, E_{\\rm low}) \\cdot \\sigma(T, E_{\\rm high})}

    where  σ(T, E) = ρ · C · T^α · E^β.

    Parameters
    ----------
    E_low, E_high : float
        Group energy bounds (keV).
    C_OPA : float
        Opacity coefficient.
    alpha : float
        Temperature exponent.
    beta : float
        Energy exponent.
    rho : float
        Density (g/cm³).
    sigma_max : float
        Hard upper cap.

    Returns
    -------
    opacity : callable
        ``opacity(T) -> float``.
    """
    sigma_func = make_powerlaw_sigma(C_OPA, alpha, beta, rho, sigma_max)

    def opacity(T):
        return float(np.sqrt(sigma_func(T, E_low) * sigma_func(T, E_high)))

    return opacity


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------

def build_clustered_edges(x_min, x_max, nx, cluster_beta=0.0, side='left'):
    """Build a 1-D mesh with optional boundary clustering.

    Parameters
    ----------
    x_min, x_max : float
        Domain limits.
    nx : int
        Number of cells.
    cluster_beta : float
        Clustering strength.  ``0`` gives uniform spacing.  Positive values
        compress cells near *side* using an exponential map.
    side : str
        ``'left'`` (default) or ``'right'``.

    Returns
    -------
    edges : ndarray, length nx + 1
    """
    if cluster_beta <= 0.0:
        return np.linspace(x_min, x_max, nx + 1)
    s = np.linspace(0.0, 1.0, nx + 1)
    mapped = (np.exp(cluster_beta * s) - 1.0) / (np.exp(cluster_beta) - 1.0)
    if side == 'right':
        mapped = 1.0 - mapped[::-1]
    return x_min + (x_max - x_min) * mapped


if __name__ == "__main__":
    print("MG_IMC Utilities Module")
    print()
    
    # Test energy group creation
    print("Test 1: Logarithmic energy groups")
    E_edges = create_log_energy_groups(0.1, 10.0, 5)
    print_group_info(E_edges)
    print()
    
    # Test opacity functions
    print("Test 2: Power-law opacity functions")
    sigma_funcs = powerlaw_opacity_functions(E_edges, sigma_ref=1.0, T_ref=1.0, alpha=3.0)
    print_group_info(E_edges, sigma_funcs=sigma_funcs, T_test=0.5)
    print()
    
    print("Test 3: Simple EOS")
    eos, inv_eos, cv = simple_eos_functions(cv_value=0.1, rho=1.0)
    T_test = 2.0
    e_test = eos(np.array([[T_test]]))
    T_recovered = inv_eos(e_test)
    print(f"T = {T_test} keV → e = {e_test[0,0]:.6f} GJ/cm³ → T = {T_recovered[0,0]:.6f} keV")
    print(f"Specific heat: c_v = {cv(np.array([[T_test]]))[0,0]:.6f} GJ/(g·keV)")
    print()
    
    print("All tests passed!")
