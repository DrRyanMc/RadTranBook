"""
Utility functions for multigroup IMC simulations.

Provides common opacity models, energy group structures, and helper functions.
"""

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
