"""1-D M1 solver for gray non-equilibrium radiation transport.

This module implements a first-order finite-volume discretization of the M1
system with a Fleck-Cummings implicit linearization and optional 1-D
curvilinear geometry through A(s)=s^d (d=0 slab, d=1 cylindrical, d=2
spherical).  The interface mirrors the S_N, P_N, diffusion, and IMC solvers
in this package: material properties are supplied as callables.
"""

import numpy as np


C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm^3 keV^4)


def _safe_abs(x, eps=1e-14):
    return np.maximum(np.abs(x), eps)


def closure_p1(f_abs):
    """P1 closure (constant Eddington factor)."""
    return np.ones_like(f_abs) / 3.0


def closure_kershaw(f_abs):
    """Kershaw closure: chi = 1/3 + 2/3 f^2."""
    f2 = np.clip(f_abs, 0.0, 1.0) ** 2
    return (1.0 / 3.0) + (2.0 / 3.0) * f2


def closure_levermore(f_abs):
    """Levermore closure used in M1 models."""
    f2 = np.clip(f_abs, 0.0, 1.0) ** 2
    denom = 2.0 + np.sqrt(np.maximum(4.0 - 3.0 * f2, 1e-14))
    return (1.0 / 3.0) + (2.0 * f2) / denom


def closure_minerbo_poly(f_abs):
    """Polynomial Minerbo approximation exact at f=1.

    chi = 1/3 + (1/15) * (6f^2 - 2f + 6) f^2
    """
    f = np.clip(f_abs, 0.0, 1.0)
    return (1.0 / 3.0) + (1.0 / 15.0) * (6.0 * f * f - 2.0 * f + 6.0) * f * f


def closure_minerbo_rational(f_abs):
    """Rational Minerbo fit used historically."""
    f = np.clip(f_abs, 0.0, 1.0)
    num = 0.01932 * f + 0.2694 * f * f
    den = 1.0 - 0.5953 * f + 0.02625 * f * f
    return (1.0 / 3.0) + num / np.maximum(den, 1e-14)


class M1Solver1D:
    """Generic first-order 1-D M1 solver with Fleck-Cummings implicit coupling.

    Solves the two-moment M1 radiation equations coupled to a material
    described by user-supplied opacity and equation-of-state callables,
    mirroring the interface of the S_N, diffusion, and IMC solvers.

    Parameters
    ----------
    x_min, x_max : float
        Domain extent.
    n_cells : int
        Number of spatial cells.
    d : int
        Geometry: 0 = slab, 1 = cylindrical, 2 = spherical.
    sigma_func : callable  T -> ndarray
        Absorption opacity σ_a(T) [cm^{-1}].  Defaults to σ_a = 1.
    scat_func : callable  T -> ndarray
        Scattering opacity σ_s(T) [cm^{-1}].  Defaults to zero.
    EOS : callable  T -> ndarray
        Material energy density e(T).  Defaults to e = a T^4.
    invEOS : callable  e -> ndarray
        Temperature from material energy density.  Must be consistent with EOS.
    dt : float
        Time step [ns].
    closure_func : callable
        Eddington factor closure chi(|f|).  Defaults to Levermore.
    c_light : float
        Speed of light [cm/ns].
    a_rad : float
        Radiation constant a [GJ cm^{-3} keV^{-4}].
    left_bc_mode : str
        Left boundary: ``"reflect"`` (default), ``"marshak"``,
        ``"free_stream"``, ``"copy"``, or ``"incident"``.
    right_bc_mode : str
        Right boundary: ``"marshak"`` (default), ``"reflect"``,
        ``"free_stream"``, ``"copy"``, or ``"incident"``.
    left_bc_state : tuple (Er, F) or None
        Fixed ghost-cell state ``(Er_bc, F_bc)`` when
        ``left_bc_mode == "incident"``.
    right_bc_state : tuple (Er, F) or None
        Fixed ghost-cell state ``(Er_bc, F_bc)`` when
        ``right_bc_mode == "incident"``.
    max_nonlinear_iter : int
        Maximum Picard iterations per time step (for nonlinear closure).
    nonlinear_tol : float
        Convergence tolerance for the Picard iteration.
    relaxation : float
        Under-relaxation factor (1.0 = none).
    """

    def __init__(
        self,
        x_min=0.0,
        x_max=20.0,
        n_cells=400,
        d=0,
        sigma_func=None,
        scat_func=None,
        EOS=None,
        invEOS=None,
        dt=1e-4,
        closure_func=closure_levermore,
        c_light=C_LIGHT,
        a_rad=A_RAD,
        left_bc_mode="reflect",
        right_bc_mode="marshak",
        left_bc_state=None,
        right_bc_state=None,
        max_nonlinear_iter=40,
        nonlinear_tol=1e-8,
        relaxation=1.0,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.n_cells = int(n_cells)
        self.d = int(d)
        if self.d not in (0, 1, 2):
            raise ValueError("d must be 0 (slab), 1 (cylindrical), or 2 (spherical)")
        self.dt = float(dt)
        self.c_light = float(c_light)
        self.a_rad = float(a_rad)
        self.closure_func = closure_func
        self.left_bc_mode = str(left_bc_mode)
        self.right_bc_mode = str(right_bc_mode)
        self.left_bc_state = left_bc_state
        self.right_bc_state = right_bc_state
        _valid_bc = ("reflect", "marshak", "free_stream", "copy", "incident")
        if self.left_bc_mode not in _valid_bc:
            raise ValueError(f"left_bc_mode must be one of {_valid_bc}")
        if self.right_bc_mode not in _valid_bc:
            raise ValueError(f"right_bc_mode must be one of {_valid_bc}")
        if self.left_bc_mode == "incident" and left_bc_state is None:
            raise ValueError("left_bc_state must be provided when left_bc_mode='incident'")
        if self.right_bc_mode == "incident" and right_bc_state is None:
            raise ValueError("right_bc_state must be provided when right_bc_mode='incident'")
        self.max_nonlinear_iter = int(max_nonlinear_iter)
        self.nonlinear_tol = float(nonlinear_tol)
        self.relaxation = float(relaxation)

        # Material property callables.  Defaults reproduce the radiation-
        # dominated (e = aT^4) material used in many textbook benchmarks.
        _a = float(a_rad)
        self.sigma_func = sigma_func if sigma_func is not None else (
            lambda T: np.ones_like(np.asarray(T, dtype=float))
        )
        self.scat_func = scat_func if scat_func is not None else (
            lambda T: np.zeros_like(np.asarray(T, dtype=float))
        )
        self.EOS = EOS if EOS is not None else (
            lambda T: _a * np.maximum(np.asarray(T, dtype=float), 1e-8) ** 4
        )
        self.invEOS = invEOS if invEOS is not None else (
            lambda e: np.maximum(
                np.maximum(np.asarray(e, dtype=float), 1e-30) / _a, 1e-30
            ) ** 0.25
        )

        self.x_faces = np.linspace(x_min, x_max, self.n_cells + 1)
        self.x_centers = 0.5 * (self.x_faces[1:] + self.x_faces[:-1])
        self.dx = self.x_faces[1:] - self.x_faces[:-1]

        # Area terms for 1-D curvilinear form A(s) = s^d.
        if self.d == 0:
            self.A_faces = np.ones_like(self.x_faces)
            self.A_centers = np.ones_like(self.x_centers)
        else:
            self.A_faces = np.maximum(self.x_faces, 1e-12) ** self.d
            self.A_centers = np.maximum(self.x_centers, 1e-12) ** self.d

        self.Er = np.zeros(self.n_cells)
        self.F = np.zeros(self.n_cells)
        self.T = np.zeros(self.n_cells)

    def initialize(self, T_init=1e-3):
        """Initialize with near-cold equilibrium state."""
        self.T[:] = T_init
        self.Er[:] = self.a_rad * T_init ** 4
        self.F[:] = 0.0

    def closure_eval(self, Er, F):
        """Compute reduced flux and Eddington factor in each cell."""
        denom = self.c_light * np.maximum(Er, 1e-20)
        f = np.clip(F / denom, -1.0, 1.0)
        chi = self.closure_func(np.abs(f))
        return f, chi

    def physical_flux(self, U):
        """Physical flux G(U) for slab M1 equations."""
        Er = U[:, 0]
        F = U[:, 1]
        _, chi = self.closure_eval(Er, F)
        G1 = F
        G2 = (self.c_light ** 2) * Er * chi
        return np.column_stack((G1, G2))

    def _extend_with_ghosts(self, U):
        """Apply left and right boundary ghost states."""
        U_ext = np.zeros((self.n_cells + 2, 2))
        U_ext[1:-1, :] = U

        # Left ghost (index 0).
        if self.left_bc_mode == "incident":
            U_ext[0, 0] = self.left_bc_state[0]
            U_ext[0, 1] = self.left_bc_state[1]
        else:
            U_ext[0, 0] = U[0, 0]
            if self.left_bc_mode == "reflect":
                U_ext[0, 1] = -U[0, 1]
            elif self.left_bc_mode == "marshak":
                U_ext[0, 1] = -0.5 * self.c_light * U[0, 0]
            elif self.left_bc_mode == "copy":
                U_ext[0, 1] = U[0, 1]
            else:  # free_stream
                U_ext[0, 1] = -self.c_light * U[0, 0]

        # Right ghost (index -1).
        if self.right_bc_mode == "incident":
            U_ext[-1, 0] = self.right_bc_state[0]
            U_ext[-1, 1] = self.right_bc_state[1]
        else:
            U_ext[-1, 0] = U[-1, 0]
            if self.right_bc_mode == "reflect":
                U_ext[-1, 1] = -U[-1, 1]
            elif self.right_bc_mode == "marshak":
                U_ext[-1, 1] = 0.5 * self.c_light * U[-1, 0]
            elif self.right_bc_mode == "copy":
                U_ext[-1, 1] = U[-1, 1]
            else:  # free_stream
                U_ext[-1, 1] = self.c_light * U[-1, 0]
        return U_ext

    def numerical_fluxes(self, U):
        """Rusanov flux with max characteristic speed lambda=c."""
        U_ext = self._extend_with_ghosts(U)
        GL = self.physical_flux(U_ext[:-1, :])
        GR = self.physical_flux(U_ext[1:, :])
        lam = self.c_light
        return 0.5 * (GL + GR) - 0.5 * lam * (U_ext[1:, :] - U_ext[:-1, :])

    def _transport_divergence(self, U):
        """Compute FV transport divergence (A G)_{i+1/2} - (A G)_{i-1/2}."""
        num_flux = self.numerical_fluxes(U)
        area_flux = self.A_faces[:, None] * num_flux
        return (area_flux[1:, :] - area_flux[:-1, :]) / (self.A_centers[:, None] * self.dx[:, None])

    def _geometric_momentum_term(self, Er, F):
        """Compute c^2 * d/s * ((3chi-1)/2) * Er for d>0."""
        if self.d == 0:
            return np.zeros_like(Er)
        _, chi = self.closure_eval(Er, F)
        s = np.maximum(self.x_centers, 1e-12)
        return (self.c_light ** 2) * (self.d / s) * 0.5 * (3.0 * chi - 1.0) * Er

    def _sweep_gs_one(self, Er_it, F_it, Er_old, F_old,
                      sigma_eff, sigma_tot, U_n, source, forward):
        """One directional cell-implicit Gauss-Seidel sweep (modifies in-place).

        Each cell is solved implicitly: the Rusanov flux at both faces of cell i
        is evaluated with ``Er_it[i]`` and ``F_it[i]`` as implicit unknowns.
        Collecting these terms onto the LHS gives a 2×1 linear system that is
        solved explicitly.  Because each cell is updated immediately, already-
        updated neighbours are used for subsequent cells, so boundary information
        propagates across the entire domain in a single forward + backward pass.
        The scheme is unconditionally stable (no CFL condition).

        Parameters
        ----------
        forward : bool
            ``True`` for left-to-right sweep; ``False`` for right-to-left.
        """
        c  = self.c_light
        dt = self.dt
        n  = self.n_cells

        # Ghost states (computed from the current iterate).
        U_ext      = self._extend_with_ghosts(np.column_stack((Er_it, F_it)))
        Er_ghost_L = U_ext[0,   0];  F_ghost_L = U_ext[0,   1]
        Er_ghost_R = U_ext[n+1, 0];  F_ghost_R = U_ext[n+1, 1]

        # Geometric source at current iterate (explicit; updated each sweep).
        geo = self._geometric_momentum_term(Er_it, F_it)

        cell_range = range(n) if forward else range(n - 1, -1, -1)

        for i in cell_range:
            A_L = self.A_faces[i]
            A_R = self.A_faces[i + 1]
            Vc  = self.A_centers[i] * self.dx[i]

            # Left and right neighbour states — in-place updates mean the
            # "already-swept" side always holds the most recent value.
            Er_L = Er_ghost_L if i == 0     else Er_it[i - 1]
            F_L  = F_ghost_L  if i == 0     else F_it[i - 1]
            Er_R = Er_ghost_R if i == n - 1 else Er_it[i + 1]
            F_R  = F_ghost_R  if i == n - 1 else F_it[i + 1]

            # Closure at neighbours.
            f_abs_L = float(np.clip(abs(F_L) / (c * max(Er_L, 1e-20)), 0.0, 1.0))
            f_abs_R = float(np.clip(abs(F_R) / (c * max(Er_R, 1e-20)), 0.0, 1.0))
            chi_L   = float(self.closure_func(np.array([f_abs_L]))[0])
            chi_R   = float(self.closure_func(np.array([f_abs_R]))[0])

            # Implicit coefficient: same for Er and F (from Rusanov dissipation
            # at both faces when the current cell is the implicit unknown).
            coeff_c = 0.5 * c * (A_L + A_R) / Vc

            # ── Er update ──────────────────────────────────────────────────
            er_num = (Er_old[i] / dt
                      + (A_L * (F_L + c * Er_L) + A_R * (c * Er_R - F_R))
                        / (2.0 * Vc)
                      + sigma_eff[i] * c * U_n[i]
                      + source[i])
            er_den = 1.0 / dt + coeff_c + sigma_eff[i] * c
            Er_it[i] = er_num / er_den

            # ── F update ───────────────────────────────────────────────────
            # For curvilinear geometry (d>0) the chi_i term in the pressure
            # tensor divergence does not cancel (A_R ≠ A_L); linearise with
            # the current-iterate closure value at cell i.
            if self.d != 0:
                f_abs_i = float(np.clip(
                    abs(F_it[i]) / (c * max(Er_it[i], 1e-20)), 0.0, 1.0))
                chi_i_cur = float(self.closure_func(np.array([f_abs_i]))[0])
                chi_cur_correction = (0.5 * c**2 * (A_R - A_L) / Vc
                                      * Er_it[i] * chi_i_cur)
            else:
                chi_cur_correction = 0.0

            f_num = (F_old[i] / dt
                     + (A_L * (c**2 * chi_L * Er_L + c * F_L)
                        + A_R * (c * F_R - c**2 * chi_R * Er_R))
                       / (2.0 * Vc)
                     - geo[i]
                     - chi_cur_correction)
            f_den = 1.0 / dt + coeff_c + sigma_tot[i] * c
            F_it[i] = f_num / f_den
            # Enforce realizability: the GS chi-linearisation can produce
            # |F|/(cEr) slightly > 1 for mixed-beam cells with nonlinear
            # closures.  Clamp here — this is a physics constraint, not
            # a convergence mask.
            max_F = c * Er_it[i]
            if abs(F_it[i]) > max_F:
                F_it[i] = max_F if F_it[i] > 0 else -max_F

    def step(self, source=None, verbose=False):
        """Advance one time step with Fleck-Cummings implicit linearization.

        The material-radiation coupling is handled via the Fleck-Cummings
        scheme: a single linearisation at T^n decouples the radiation solve
        from the material update, allowing an arbitrary equation of state.

        The nonlinear M1 closure is handled by a symmetric Gauss-Seidel (SGS)
        iteration: each outer iteration sweeps all cells once left-to-right then
        right-to-left.  Because each cell is updated implicitly using the most
        recent neighbour values (in-place updates), boundary information
        propagates across the entire domain in a single pass and the scheme is
        unconditionally stable with no CFL condition.

        Raises
        ------
        ValueError
            If the state is non-physical at entry or after the step.
        """
        if source is None:
            source = np.zeros(self.n_cells)

        Er_old = self.Er.copy()
        F_old  = self.F.copy()
        T_old  = self.T.copy()

        if np.any(T_old <= 0.0):
            raise ValueError(
                f"Non-positive temperature before step: min T = {T_old.min():.3e}"
            )
        if np.any(Er_old < 0.0):
            raise ValueError(
                f"Negative radiation energy before step: min Er = {Er_old.min():.3e}"
            )

        c  = self.c_light
        dt = self.dt

        # ── Fleck-Cummings linearisation at T_old ─────────────────────────
        sigma     = self.sigma_func(T_old)
        scat      = self.scat_func(T_old)
        sigma_tot = sigma + scat

        # beta = d(aT^4)/de  via centred numerical derivative of EOS.
        h_cv = T_old * 1e-4 + 1e-12
        Cv   = (self.EOS(T_old + h_cv) - self.EOS(T_old - h_cv)) / (2.0 * h_cv)
        if np.any(Cv <= 0.0):
            raise ValueError(
                f"Non-positive heat capacity from EOS: min Cv = {Cv.min():.3e}"
            )
        beta      = 4.0 * self.a_rad * T_old ** 3 / Cv
        f         = 1.0 / (1.0 + beta * c * sigma * dt)
        sigma_eff = f * sigma

        U_n   = self.a_rad * T_old ** 4
        e_old = self.EOS(T_old)

        # ── Symmetric Gauss-Seidel iteration for nonlinear closure ────────
        Er_it = Er_old.copy()
        F_it  = F_old.copy()

        converged = False
        for k in range(self.max_nonlinear_iter):
            Er_prev = Er_it.copy()
            F_prev  = F_it.copy()

            # One forward + one backward sweep per SGS iteration.
            self._sweep_gs_one(Er_it, F_it, Er_old, F_old,
                               sigma_eff, sigma_tot, U_n, source, forward=True)
            self._sweep_gs_one(Er_it, F_it, Er_old, F_old,
                               sigma_eff, sigma_tot, U_n, source, forward=False)

            if self.relaxation < 1.0:
                w      = self.relaxation
                Er_it  = (1.0 - w) * Er_prev + w * Er_it
                F_it   = (1.0 - w) * F_prev  + w * F_it

            dEr = np.max(np.abs(Er_it - Er_prev) / np.maximum(np.abs(Er_it), 1e-16))
            dF  = np.max(np.abs(F_it  - F_prev)  / np.maximum(np.abs(F_it),  1e-16))
            err = max(dEr, dF)

            if err < self.nonlinear_tol:
                converged = True
                break

        if verbose and not converged:
            print(
                f"Warning: step did not reach tol={self.nonlinear_tol:.1e} "
                f"in {self.max_nonlinear_iter} iterations"
            )

        # ── Fleck-Cummings material energy update ──────────────────────────
        e_new = e_old + f * c * sigma * dt * (Er_it - U_n)
        T_new = self.invEOS(e_new)

        # Validate post-step state — raise rather than silently fix.
        if np.any(Er_it < 0.0):
            raise ValueError(
                f"Negative radiation energy after step: min Er = {Er_it.min():.3e}"
            )
        if np.any(np.abs(F_it) > c * Er_it * (1.0 + 1e-10)):
            worst = (np.abs(F_it) / (c * np.abs(Er_it) + 1e-300) - 1.0).max()
            raise ValueError(
                f"Realizability violated after step: max(|F|/(cEr) - 1) = {worst:.3e}"
            )
        if np.any(e_new < 0.0):
            raise ValueError(
                f"Negative material energy after step: min e = {e_new.min():.3e}"
            )
        if np.any(T_new <= 0.0):
            raise ValueError(
                f"Non-positive temperature after step: min T = {T_new.min():.3e}"
            )

        self.Er[:] = Er_it
        self.F[:]  = F_it
        self.T[:]  = T_new

    def run(self, n_steps, source_func=None, output_times=None):
        """Run simulation and optionally save snapshots when crossing times."""
        if output_times is None:
            output_times = []

        output_times = list(output_times)
        out_idx = 0
        snapshots = {}

        t = 0.0
        for istep in range(n_steps):
            t_old = t
            t = (istep + 1) * self.dt

            if source_func is None:
                src = np.zeros(self.n_cells)
            else:
                src = source_func(t_old)

            self.step(source=src)

            if out_idx < len(output_times) and t >= output_times[out_idx]:
                tau_key = output_times[out_idx]
                snapshots[tau_key] = {
                    "x": self.x_centers.copy(),
                    "Er": self.Er.copy(),
                    "F": self.F.copy(),
                    "T": self.T.copy(),
                }
                out_idx += 1

        return snapshots
