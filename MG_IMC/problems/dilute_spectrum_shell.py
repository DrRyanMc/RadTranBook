"""
Dilute Spectrum Shell — shared problem parameters and setup utilities.

Physical setup
--------------
A blackbody source of radius R_S = 1 cm at temperature T_S = 1 keV sits at
the centre.  Surrounding it is an optically-thin cavity [R_S, R_1] filled with
near-vacuum material (ρ = 1e-8 g/cm³).  A dense, optically-thick shell
occupies [R_1, R_2] (ρ = 2 g/cm³).  Beyond R_2 out to the outer computational
boundary R_OUT radiation escapes freely.

The opacity is a power-law in both temperature and photon energy:

    σ_{a,g}(T) = ρ² · C · T^A · ν̄_g^B

with  C = 10 cm²/g,  A = −0.5,  B = −3  and ν̄_g = √(ν_{g−½} · ν_{g+½}).

Because the cavity is so transparent, radiation from the source free-streams
across it with roughly geometric dilution:

    E_r(r) ≈ (a c / 4) · T_S^4 · (R_S / r)²

The shell imposes a frequency-dependent optical depth, selectively absorbing
certain groups and re-emitting at the local shell temperature.  The result is
a non-Planckian (diluted, hardened) spectrum inside the shell — the diagnostic
of interest.

Units: distance cm, time ns, temperature keV, energy GJ.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (must agree with MG_IMC1D)
# ---------------------------------------------------------------------------
C_LIGHT = 29.98   # cm / ns
A_RAD   = 0.01372 # GJ / (cm³ · keV⁴)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
R_S   = 1.0   # cm  inner (source) radius
R_1   = 25.0  # cm  cavity / shell interface
R_2   = 27.0  # cm  shell / outer interface
R_OUT = 35.0  # cm  outer computational boundary

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
T_S = 1.0  # keV  blackbody temperature of the inner source

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
RHO_CAVITY = 1.0e-8  # g/cm³  near-vacuum in the cavity
RHO_SHELL  = 2.0     # g/cm³  dense shell
CV_SPEC    = 0.05    # GJ / (g · keV)  specific heat (mass-based)
T_INIT     = 0.02    # keV   initial material temperature everywhere
T_FLOOR    = 0.01    # keV   hard lower bound in opacity evaluations

# ---------------------------------------------------------------------------
# Opacity model:  σ_{a,g} = ρ · C_OPA · T^A_OPA · ν̄_g^B_OPA
# ---------------------------------------------------------------------------
C_OPA   = 10.0   # cm²/g
A_OPA   = -0.5   # temperature exponent
B_OPA   = -3.0   # frequency exponent
# Opacity cap (prevents absurdly small mean free paths in lowest groups)
SIGMA_MAX = 1.0e12  # cm⁻¹

# ---------------------------------------------------------------------------
# Energy groups (default)
# ---------------------------------------------------------------------------
N_GROUPS_DEFAULT = 32
NU_MIN = 1.0e-2  # keV  lower group edge
NU_MAX = 30.0    # keV  upper group edge

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
DT_DEFAULT = 0.01  # ns  nominal timestep
T_FINAL   = 4.0  # ns  end of simulation

# Analytic estimates for reference:
#   light travel from source to shell:  (R_1 - R_S) / C_LIGHT ≈ 0.80 ns
#   free-streaming T_r at R_1 from geometric dilution:
#       T_r ≈ T_S · sqrt(R_S / (2·R_1)) ≈ 0.14 keV

DUMP_TIMES = [0.25, 0.50, 0.75, 0.90,.91,.92,.93,.94,.95,.96,.97,.98,.99, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00]  # ns


# ===========================================================================
# Mesh construction
# ===========================================================================

def make_mesh(mode="standard"):
    """Build the non-uniform radial mesh.

    Four regions with different spacing:
      [R_S, 23 cm]  — coarse (most of the cavity)
      [23, R_1]     — fine approach to shell inner face
      [R_1, R_2]    — finest (inside the shell itself)
      [R_2, R_OUT]  — coarser outer buffer

    Parameters
    ----------
    mode : {'quick', 'standard', 'publication'}
        Controls cell count.  'standard' and above are identical to the
        reference grid; 'quick' uses half the cells.

    Returns
    -------
    mesh : ndarray, shape (n_cells, 2)
        Each row is [r_inner, r_outer].
    r_centers : ndarray, shape (n_cells,)
    rho_per_cell : ndarray, shape (n_cells,)
    """
    factor = 2 if mode == "quick" else 1

    # Build edges for each region using linspace so that the shared boundary
    # points (23.0, R_1=25.0, R_2=27.0) are exactly the same float in both
    # adjacent arrays.  np.arange with a floating-point step accumulates
    # rounding error and can produce a last element that differs from the
    # intended stop by ~1e-14, creating a near-zero-width ghost cell after
    # np.unique that leaves T_r = 0 at those interfaces.
    def _ls(start, stop, step):
        n = int(round((stop - start) / step)) + 1
        return np.linspace(start, stop, n)

    edges_coarse = _ls(R_S,   23.0,  0.25 * factor)
    edges_fine   = _ls(23.0,  R_1,   0.1  * factor)
    edges_shell  = _ls(R_1,   R_2,   0.05 * factor)
    edges_outer  = _ls(R_2,   R_OUT, 0.25 * factor)

    # Merge edges without duplicating interior boundaries
    all_edges = np.unique(np.concatenate([
        edges_coarse, edges_fine, edges_shell, edges_outer
    ]))

    mesh = np.column_stack([all_edges[:-1], all_edges[1:]])
    r_centers = 0.5 * (mesh[:, 0] + mesh[:, 1])

    # Material density per cell
    rho_per_cell = np.where(
        (r_centers >= R_1) & (r_centers < R_2),
        RHO_SHELL,
        RHO_CAVITY,
    )

    return mesh, r_centers, rho_per_cell


# ===========================================================================
# Energy groups
# ===========================================================================

def make_energy_edges(n_groups=N_GROUPS_DEFAULT, nu_min=NU_MIN, nu_max=NU_MAX):
    """Return log-spaced energy group edges (keV)."""
    return np.logspace(np.log10(nu_min), np.log10(nu_max), n_groups + 1)


# ===========================================================================
# Opacity functions
# ===========================================================================

def make_sigma_a_funcs(energy_edges, rho_per_cell):
    """Build the list of per-group opacity callables expected by MG_IMC1D.step().

    Each callable: sigma_g(T) → ndarray, shape (n_cells,)
    where T is the current material temperature array.

    The opacity is:
        σ_{a,g}(T) = ρ_i**2 · C_OPA · max(T_i, T_FLOOR)^A_OPA · ν̄_g^B_OPA

    capped at SIGMA_MAX.
    """
    n_groups = len(energy_edges) - 1
    funcs = []
    for g in range(n_groups):
        nu_bar = np.sqrt(energy_edges[g] * energy_edges[g + 1])
        # Capture nu_bar and rho_per_cell by value in the closure
        def _make(nb, rho):
            def sigma_g(T):
                T_use = np.maximum(T, T_FLOOR)
                raw = rho * C_OPA * T_use ** A_OPA * nb ** B_OPA
                return np.minimum(raw, SIGMA_MAX)
            return sigma_g
        funcs.append(_make(nu_bar, rho_per_cell))
    return funcs


# ===========================================================================
# Equation of state (spatially varying density)
# ===========================================================================

def make_eos_functions(rho_per_cell):
    """Build EOS callables for the spatially inhomogeneous material.

    Linear EOS:  u(T) = ρ_i · CV_SPEC · T_i   (GJ / cm³)

    Returns
    -------
    eos      : T_array → u_array
    inv_eos  : u_array → T_array
    cv_func  : T_array → cv_array  (GJ / cm³ / keV, spatially varying)
    """
    #make cv_vol rho*CV_SPEC when rho >1 else make cv_vol = 1 
    cv_vol = np.where(rho_per_cell > 1, rho_per_cell * CV_SPEC, 1.0)  # GJ / (cm³ · keV), shape (n_cells,)

    def eos(T):
        return cv_vol * T

    def inv_eos(u):
        # Guard against vacuum cells (rho=0 → cv_vol=0).  Those cells have
        # sigma_a=0 so u never changes from 0; return T_INIT as a safe value.
        safe_cv = np.where(cv_vol > 0, cv_vol, 1.0)
        return np.where(cv_vol > 0, u / safe_cv, T_INIT)

    def cv_func(T):
        # Vacuum cells (rho=0 → cv_vol=0): return inf so that the Fleck-factor
        # beta = 4aT³/cv → 0 instead of inf, avoiding a NaN when multiplied by
        # sigma_P=0.  The actual Fleck factor value is irrelevant for these cells.
        cv_out = np.where(cv_vol > 0, cv_vol, np.inf)
        return cv_out * np.ones_like(T)

    return eos, inv_eos, cv_func


# ===========================================================================
# Diagnostics
# ===========================================================================

def print_optical_depth_audit(mesh, energy_edges, rho_per_cell, T_ref=T_INIT):
    """Print a summary of optical depth through each material region.

    Evaluates at the reference temperature T_ref for representative groups.
    """
    n_groups = len(energy_edges) - 1
    n_cells  = mesh.shape[0]
    dr       = mesh[:, 1] - mesh[:, 0]

    # Which cells belong to each region
    rc = 0.5 * (mesh[:, 0] + mesh[:, 1])
    mask_cav   = rc < R_1
    mask_shell = (rc >= R_1) & (rc < R_2)

    print("=" * 70)
    print("Optical depth audit at T_ref = {:.3f} keV".format(T_ref))
    print(f"  Cavity:  {np.sum(mask_cav)} cells, ρ = {RHO_CAVITY:.1e} g/cm³,"
          f" Δr_tot = {np.sum(dr[mask_cav]):.1f} cm")
    print(f"  Shell:   {np.sum(mask_shell)} cells, ρ = {RHO_SHELL:.1f} g/cm³,"
          f" Δr_tot = {np.sum(dr[mask_shell]):.1f} cm")
    print("")
    print(f"  {'Group':>5}  {'E_low':>8}  {'E_high':>8}  {'ν̄_g':>8}"
          f"  {'τ_cav':>10}  {'τ_shell':>10}")
    print("  " + "-" * 60)

    T_arr = np.full(n_cells, T_ref)
    sigmas = make_sigma_a_funcs(energy_edges, rho_per_cell)

    # Print every 4th group to keep output concise
    for g in range(0, n_groups, max(1, n_groups // 8)):
        nu_bar = np.sqrt(energy_edges[g] * energy_edges[g + 1])
        sig    = sigmas[g](T_arr)
        tau_cav   = float(np.sum(sig[mask_cav]   * dr[mask_cav]))
        tau_shell = float(np.sum(sig[mask_shell] * dr[mask_shell]))
        print(f"  {g:>5}  {energy_edges[g]:>8.2e}  {energy_edges[g+1]:>8.2e}"
              f"  {nu_bar:>8.2e}  {tau_cav:>10.2e}  {tau_shell:>10.2e}")
    print("=" * 70)


def free_streaming_Tr(r, R_S=R_S, T_S=T_S):
    """Analytic free-streaming radiation temperature under geometric dilution.

    T_r(r) = T_S · (R_S / (2r))^{1/2}

    Valid in the optically-thin cavity well after the light-crossing time.
    """
    return T_S * np.sqrt(R_S / (2.0 * r))


def free_streaming_Er(r, R_S=R_S, T_S=T_S):
    """Analytic free-streaming radiation energy density (GJ/cm³).

    E_r(r) = (a / 4) · T_S^4 · (R_S / r)^2

    Derived from the free-streaming flux F(r) = sigma_SB * T_S^4 * (R_S/r)^2
    and E_r = F / c, giving E_r = (a/4) * T_S^4 * (R_S/r)^2.
    """
    return (A_RAD / 4.0) * T_S**4 * (R_S / r)**2
