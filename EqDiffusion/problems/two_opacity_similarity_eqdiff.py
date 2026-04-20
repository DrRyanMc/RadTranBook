#!/usr/bin/env python3
"""
Two-opacity similarity problem for EqDiffusion (2D Cartesian).

This script sets up the Hammer-Rosen style two-region problem with:
- Region 1 (y < 0)
- Region 2 (y >= 0)
- Interface at y = 0
- Left boundary (x = 0): blackbody source at T = T_source
- Right boundary (x = x_max): zero-gradient outflow
- Top/bottom boundaries (y = +/-L): zero-gradient (large-L approximation)

Requested parameter set:
- gamma = 0.25
- beta = 0.25
- n = 3

For this tabulated case we use:
- eta_max = 24.6
- xi_max = 1.120

The driver chooses the transverse half-width L and derives:
- t_final = (A1 * L / eta_max)^2
- x_max = (xi_max / eta_max) * L

Definitions used:
- beta  = (kappa0_1 * rho1) / (kappa0_2 * rho2)
- gamma = (kappa0_1 * cv1  * rho1) / (kappa0_2 * cv2  * rho2)

For gamma = beta, cv1 = cv2 when rho1 = rho2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from twoDFV import RadiationDiffusionSolver2D, temperature_from_Er, A_RAD, C_LIGHT


# -----------------------------------------------------------------------------
# Problem controls
# -----------------------------------------------------------------------------
N_OPACITY = 3
BETA_TARGET = 0.25
GAMMA_TARGET = 0.25

RHO_1 = 1.0
RHO_2 = 1.0
CV_1 = 0.30      # volumetric cv proxy used by this solver style (GJ/(cm^3 keV))
KAPPA0_2 = 300.0

# Derive region-1 coefficients from target beta/gamma
KAPPA0_1 = BETA_TARGET * KAPPA0_2 * (RHO_2 / RHO_1)
CV_2 = (KAPPA0_1 * CV_1 * RHO_1) / (GAMMA_TARGET * KAPPA0_2 * RHO_2)

# Check achieved values
BETA_ACHIEVED = (KAPPA0_1 * RHO_1) / (KAPPA0_2 * RHO_2)
GAMMA_ACHIEVED = (KAPPA0_1 * CV_1 * RHO_1) / (KAPPA0_2 * CV_2 * RHO_2)

T_SOURCE = 1.0
T_INIT = 0.01
T_FLOOR = 0.001

ETA_MAX = 24.6
XI_MAX = 1.120
DEFAULT_L = 3.0


def similarity_diffusion_scale():
    """Return the Marshak-style similarity scale K for region 1.

    This follows the same volumetric-coefficient convention used elsewhere in
    EqDiffusion, where opacity is in cm^-1 and c_v is volumetric.
    """
    return (8.0 * A_RAD * C_LIGHT) * (T_SOURCE ** N_OPACITY) / (
        3.0 * (N_OPACITY + 4.0) * KAPPA0_1 * CV_1
    )


def similarity_A1():
    """Return A1 defined by xi = x * A1 / sqrt(t)."""
    return 1.0 / np.sqrt(similarity_diffusion_scale())


def derive_similarity_domain(L):
    """Compute x extent and final time from L, eta_max, and xi_max."""
    A1 = similarity_A1()
    t_final = (A1 * L / ETA_MAX) ** 2
    x_max = (XI_MAX / ETA_MAX) * L
    return A1, x_max, t_final


def create_log_spacing_one_sided(start, end, n_cells, from_left=True):
    """Create logarithmically graded faces across a single interval."""
    if n_cells <= 1:
        return [end]

    width = end - start
    max_ratio = 5.0
    growth = max_ratio ** (1.0 / (n_cells - 1)) if n_cells > 1 else 1.0

    if abs(growth - 1.0) < 1e-12:
        first_width = width / n_cells
        cell_widths = [first_width] * n_cells
    else:
        first_width = width * (growth - 1.0) / (growth ** n_cells - 1.0)
        cell_widths = [first_width * growth ** i for i in range(n_cells)]

    if not from_left:
        cell_widths = cell_widths[::-1]

    faces = []
    current_pos = start
    for cell_width in cell_widths:
        current_pos += cell_width
        faces.append(current_pos)

    faces[-1] = end
    return faces


def create_log_spacing_around_interface(left, right, interface_location, n_cells):
    """Create graded faces in a coarse cell with finest spacing at the interface."""
    if interface_location <= left:
        return create_log_spacing_one_sided(left, right, n_cells, from_left=True)
    if interface_location >= right:
        return create_log_spacing_one_sided(left, right, n_cells, from_left=False)

    n_left = max(1, int(n_cells * (interface_location - left) / (right - left)))
    n_right = n_cells - n_left
    if n_right == 0:
        n_right = 1
        n_left = max(1, n_cells - 1)

    faces_left = create_log_spacing_one_sided(left, interface_location, n_left, from_left=False)
    faces_right = create_log_spacing_one_sided(interface_location, right, n_right, from_left=True)
    return faces_left + faces_right


def generate_refined_faces(domain_min, domain_max, interface_locations, n_refine, n_coarse, refine_width):
    """Generate faces with local logarithmic refinement around target interfaces."""
    coarse_faces = np.linspace(domain_min, domain_max, n_coarse + 1)
    refine_info = {}

    for interface_location in sorted(interface_locations):
        for coarse_index in range(n_coarse):
            cell_left = coarse_faces[coarse_index]
            cell_right = coarse_faces[coarse_index + 1]
            overlaps_refinement_zone = (
                cell_left <= interface_location + refine_width
                and cell_right >= interface_location - refine_width
            )
            if not overlaps_refinement_zone:
                continue

            cell_center = 0.5 * (cell_left + cell_right)
            if coarse_index not in refine_info:
                refine_info[coarse_index] = interface_location
            elif abs(interface_location - cell_center) < abs(refine_info[coarse_index] - cell_center):
                refine_info[coarse_index] = interface_location

    face_list = [domain_min]
    for coarse_index in range(n_coarse):
        cell_left = coarse_faces[coarse_index]
        cell_right = coarse_faces[coarse_index + 1]
        if coarse_index in refine_info:
            refined_faces = create_log_spacing_around_interface(
                cell_left,
                cell_right,
                refine_info[coarse_index],
                n_refine,
            )
            face_list.extend(refined_faces)
        else:
            face_list.append(cell_right)

    return np.array(face_list)


@njit
def in_region_1(x, y):
    return y < 0.0


@njit
def two_opacity_kappa(Er, x, y):
    """Power-law opacity kappa = kappa0 * (T_source/T)^n with two regions."""
    T = temperature_from_Er(Er)
    if T < T_FLOOR:
        T = T_FLOOR

    if in_region_1(x, y):
        kappa0 = KAPPA0_1
    else:
        kappa0 = KAPPA0_2

    return kappa0 * (T_SOURCE / T) ** N_OPACITY


@njit
def two_region_cv(T, x, y):
    if in_region_1(x, y):
        return CV_1
    return CV_2


@njit
def two_region_material_energy(T, x, y):
    return two_region_cv(T, x, y) * T


def bc_left_source(Er_boundary, x_val, y_val, geometry="cartesian", time=0.0):
    """Marshak-style source at x=0 with diffusion-consistent Robin form."""
    Er_src = A_RAD * T_SOURCE ** 4
    sigma = two_opacity_kappa(Er_boundary, x_val, y_val)
    D_boundary = C_LIGHT / (3.0 * sigma)
    return 0.5, D_boundary, 0.5 * Er_src


def bc_right_open(Er_boundary, x_val, y_val, geometry="cartesian", time=0.0):
    """Zero normal gradient at x=x_max."""
    return 0.0, 1.0, 0.0


def bc_bottom_farfield(Er_boundary, x_val, y_val, geometry="cartesian", time=0.0):
    """Large-L approximation to far-field condition at y=-L."""
    return 0.0, 1.0, 0.0


def bc_top_farfield(Er_boundary, x_val, y_val, geometry="cartesian", time=0.0):
    """Large-L approximation to far-field condition at y=+L."""
    return 0.0, 1.0, 0.0


def save_temperature_colormap(x_centers, y_centers, T_2d, time_value):
    """Save a temperature colormap for a single snapshot time."""
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.pcolormesh(X, Y, T_2d, shading="auto", cmap="plasma")
    #ax.axhline(0.0, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(
        f"Two-Opacity Similarity Test: n={N_OPACITY}, beta={BETA_ACHIEVED:.2f}, gamma={GAMMA_ACHIEVED:.2f}, t={time_value:.3f} ns"
    )
    plt.colorbar(im, ax=ax, label="T (keV)")
    plt.tight_layout()
    filename = f"two_opacity_similarity_eqdiff_t_{time_value:.3f}ns.png"
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def save_mesh_plot(x_faces, y_faces, use_refined_mesh):
    """Save a plot of the computational mesh."""
    fig, ax = plt.subplots(figsize=(10, 3.6))

    x_min = x_faces[0]
    x_max = x_faces[-1]
    y_min = y_faces[0]
    y_max = y_faces[-1]

    for y_face in y_faces:
        ax.plot([x_min, x_max], [y_face, y_face], color="black", linewidth=0.25, alpha=0.5)
    for x_face in x_faces:
        ax.plot([x_face, x_face], [y_min, y_max], color="black", linewidth=0.25, alpha=0.5)

    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("auto")
    ax.set_title("Two-Opacity Mesh" + (" (refined at y=0)" if use_refined_mesh else " (uniform)"))
    plt.tight_layout()

    filename = "two_opacity_similarity_eqdiff_mesh_refined.png" if use_refined_mesh else "two_opacity_similarity_eqdiff_mesh_uniform.png"
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def run_two_opacity_similarity(
    output_times=None,
    nx=210,
    ny=80,
    dt=5e-4,
    L=DEFAULT_L,
    use_refined_mesh=True,
    n_refine=10,
    refine_width=0.05,
):
    print("=" * 72)
    print("Two-Opacity Similarity Test (EqDiffusion)")
    print("=" * 72)
    print(f"n = {N_OPACITY}")
    print(f"beta target   = {BETA_TARGET:.6f}, achieved = {BETA_ACHIEVED:.6f}")
    print(f"gamma target  = {GAMMA_TARGET:.6f}, achieved = {GAMMA_ACHIEVED:.6f}")
    print(f"kappa0_1      = {KAPPA0_1:.6f}")
    print(f"kappa0_2      = {KAPPA0_2:.6f}")
    print(f"cv_1          = {CV_1:.6f}")
    print(f"cv_2          = {CV_2:.6f}")

    A1, x_max, t_final = derive_similarity_domain(L)
    print(f"eta_max       = {ETA_MAX:.6f}")
    print(f"xi_max        = {XI_MAX:.6f}")
    print(f"L             = {L:.6f} cm")
    print(f"A1            = {A1:.6f} ns^(-1/2) cm^(-1)")
    print(f"derived x_max = {x_max:.6f} cm")
    print(f"derived t_fin = {t_final:.6f} ns")
    print(f"refined mesh  = {use_refined_mesh}")

    # Domain and mesh
    x_min = 0.0
    y_min, y_max = -L, L

    x_faces = np.linspace(x_min, x_max, nx + 1)
    if use_refined_mesh:
        y_faces = generate_refined_faces(
            y_min,
            y_max,
            interface_locations=[0.0],
            n_refine=n_refine,
            n_coarse=ny,
            refine_width=refine_width,
        )
        print(f"y-interface refinement width = {refine_width:.6f} cm")
        print(f"mesh cells = {len(x_faces) - 1} x {len(y_faces) - 1}")
        print(f"minimum dy = {np.min(np.diff(y_faces)):.6e} cm")
    else:
        y_faces = np.linspace(y_min, y_max, ny + 1)
        print(f"mesh cells = {len(x_faces) - 1} x {len(y_faces) - 1}")

    save_mesh_plot(x_faces, y_faces, use_refined_mesh)

    # Sample the transient up to the similarity-limited final time.
    if output_times is None:
        output_times = [0.05 * t_final, 0.1 * t_final, 0.25 * t_final, 0.5 * t_final, t_final]

    solver = RadiationDiffusionSolver2D(
        coord1_faces=x_faces,
        coord2_faces=y_faces,
        geometry="cartesian",
        dt=dt,
        max_newton_iter=25,
        newton_tol=1e-7,
        rosseland_opacity_func=two_opacity_kappa,
        specific_heat_func=two_region_cv,
        material_energy_func=two_region_material_energy,
        left_bc_func=bc_left_source,
        right_bc_func=bc_right_open,
        bottom_bc_func=bc_bottom_farfield,
        top_bc_func=bc_top_farfield,
        theta=1.0,
        use_jfnk=False,
    )

    Er_bg = A_RAD * T_INIT ** 4
    solver.set_initial_condition(Er_bg)

    snapshots = []
    t = 0.0

    for t_target in output_times:
        while t < t_target:
            if t + dt > t_target:
                solver.dt = t_target - t
            else:
                solver.dt = dt
            solver.time_step(n_steps=1, verbose=False)
            t += solver.dt

        x_centers, y_centers, Er_2d = solver.get_solution()
        T_2d = temperature_from_Er(Er_2d)
        snapshots.append((t, x_centers.copy(), y_centers.copy(), Er_2d.copy(), T_2d.copy()))
        save_temperature_colormap(x_centers, y_centers, T_2d, t)
        print(f"t = {t:8.4f} ns, Tmin = {T_2d.min():.5f} keV, Tmax = {T_2d.max():.5f} keV")

    # Save final solution and history
    t_final, x_centers, y_centers, Er_2d, T_2d = snapshots[-1]
    np.savez(
        f"two_opacity_similarity_eqdiff_{nx}x{ny}.npz",
        x_centers=x_centers,
        y_centers=y_centers,
        Er_2d=Er_2d,
        T_2d=T_2d,
        t_final=t_final,
        L=L,
        eta_max=ETA_MAX,
        xi_max=XI_MAX,
        A1=A1,
        beta=BETA_ACHIEVED,
        gamma=GAMMA_ACHIEVED,
        n=N_OPACITY,
        kappa0_1=KAPPA0_1,
        kappa0_2=KAPPA0_2,
        cv_1=CV_1,
        cv_2=CV_2,
    )

    # Also save a stable final alias for convenience.
    save_temperature_colormap(x_centers, y_centers, T_2d, t_final)
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.pcolormesh(X, Y, T_2d, shading="auto", cmap="plasma")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(
        f"Two-Opacity Similarity Test: n={N_OPACITY}, beta={BETA_ACHIEVED:.2f}, gamma={GAMMA_ACHIEVED:.2f}, t={t_final:.2f} ns"
    )
    plt.colorbar(im, ax=ax, label="T (keV)")
    plt.tight_layout()
    fig.savefig("two_opacity_similarity_eqdiff_final.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Plot centerline profiles in each region (just off the interface)
    j_lower = np.argmin(np.abs(y_centers - (-0.2)))
    j_upper = np.argmin(np.abs(y_centers - (0.2)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t_snap, x_s, y_s, Er_s, T_s in snapshots:
        ax.plot(x_s, T_s[:, j_lower], linewidth=2.0, alpha=0.9, label=f"y~ -0.2, t={t_snap:.1f}")
        ax.plot(x_s, T_s[:, j_upper], linewidth=1.5, alpha=0.9, linestyle="--", label=f"y~ +0.2, t={t_snap:.1f}")

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("T (keV)")
    ax.set_title("Region-wise Profiles (below/above interface)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig("two_opacity_similarity_eqdiff_profiles.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Saved: two_opacity_similarity_eqdiff_final.png")
    print("Saved: two_opacity_similarity_eqdiff_profiles.png")
    print(f"Saved: two_opacity_similarity_eqdiff_{nx}x{ny}.npz")

    return solver, snapshots


if __name__ == "__main__":
    run_two_opacity_similarity(nx=200, ny=20, dt=1e-2, L=1.0, use_refined_mesh=True, n_refine=20, refine_width=0.05)
