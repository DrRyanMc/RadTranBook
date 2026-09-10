"""2-D M1 solver for gray non-equilibrium radiation transport.

Cartesian (x, z) or axisymmetric cylindrical (r, z) geometry on a
uniform tensor-product grid.

The three-component state is (E_r, F_1, F_2) where F_1 and F_2 are the
radiation flux components in the x/r and z directions respectively.  The
radiation pressure tensor is closed by

    P = E_r [ (1-χ)/2 · I₃  +  (3χ-1)/2 · n̂⊗n̂ ]

with n̂ = F/|F| and χ = χ(f), f = |F|/(c E_r).  The physical flux
vectors are

    x1-direction:  G = (F₁,  c²P₁₁,  c²P₁₂)
    x2-direction:  G = (F₂,  c²P₁₂,  c²P₂₂)

Cartesian equations (13.46–13.48):
    ∂E_r/∂t + ∂F₁/∂x + ∂F₂/∂z = S_E − c σ E_r
    (1/c)∂F₁/∂t + c(∂P₁₁/∂x + ∂P₁₂/∂z) + σF₁ = S_F₁
    (1/c)∂F₂/∂t + c(∂P₁₂/∂x + ∂P₂₂/∂z) + σF₂ = S_F₂

Cylindrical equations (13.50–13.52) with P_φφ = P₃₃ = E_r(1−χ)/2:
    ∂E_r/∂t + (1/r)∂(rF₁)/∂r + ∂F₂/∂z = S_E − c σ E_r
    (1/c)∂F₁/∂t + c[1/r ∂(rP₁₁)/∂r + ∂P₁₂/∂z − P₃₃/r] + σF₁ = S_F₁
    (1/c)∂F₂/∂t + c[1/r ∂(rP₁₂)/∂r + ∂P₂₂/∂z] + σF₂ = S_F₂

Time integration uses Fleck–Cummings implicit linearisation of the
radiation–material coupling and symmetric Gauss-Seidel (SGS) iteration for
the nonlinear M₁ closure.

Stability
---------
Each cell is solved implicitly for its own Rusanov self-dissipation
coefficient (the stiff term that otherwise imposes a CFL restriction),
while a forward + backward sweep over the whole domain propagates
boundary information in a single pass.  The scheme is therefore
unconditionally stable — **no CFL condition** is required, and no
characteristic direction needs to be known in advance.

Interface
---------
Mirrors M1Solver1D in m1_1d.py; key differences are the additional
spatial dimension, a third flux component, and the 'geometry' selector.
"""

import numpy as np

from .m1_1d import (
    closure_p1,
    closure_kershaw,
    closure_levermore,
    closure_minerbo_poly,
    closure_minerbo_rational,
    C_LIGHT,
    A_RAD,
)

__all__ = ["M1Solver2D"]


class M1Solver2D:
    """Finite-volume 2-D M1 solver with Fleck–Cummings implicit coupling.

    Parameters
    ----------
    x1_min, x1_max : float
        Bounds of the first coordinate (x for Cartesian, r for cylindrical).
    n1 : int
        Number of cells in the x1 direction.
    x2_min, x2_max : float
        Bounds of the second coordinate z.
    n2 : int
        Number of cells in the x2 direction.
    geometry : {'cartesian', 'cylindrical'}
        Coordinate system.  'cylindrical' assumes axisymmetry about the
        axis x1 = 0.
    sigma_func : callable  T -> ndarray (n1, n2), optional
        Absorption opacity σ_a(T) [cm⁻¹].  Defaults to σ_a = 1.
    scat_func : callable  T -> ndarray (n1, n2), optional
        Scattering opacity σ_s(T) [cm⁻¹].  Defaults to zero.
    EOS : callable  T -> ndarray (n1, n2), optional
        Material energy density e(T).  Defaults to e = aT⁴.
    invEOS : callable  e -> ndarray (n1, n2), optional
        Inverse EOS T(e).  Must be consistent with EOS.
    dt : float
        Time step [ns].
    closure_func : callable
        Eddington factor χ(f).  Defaults to Levermore.
    c_light, a_rad : float
        Speed of light and radiation constant.
    bc_x1_lo, bc_x1_hi : str
        Left / right (or inner / outer) boundary modes:
        ``'reflect'``, ``'marshak'``, ``'free_stream'``,
        ``'copy'``, or ``'incident'``.
    bc_x2_lo, bc_x2_hi : str
        Bottom / top boundary modes (same choices).
    bc_x{1,2}_{lo,hi}_state : tuple (Er, F1, F2) or None
        Fixed ghost state for ``'incident'`` boundaries.
    max_nonlinear_iter : int
        Maximum number of forward+backward SGS sweep pairs per time step.
    nonlinear_tol : float
        Convergence tolerance for (Er, F1, F2) relative change.
    relaxation : float
        Under-relaxation factor (1.0 = none).
    """

    _VALID_BC = ("reflect", "marshak", "free_stream", "copy", "incident", "callable")

    def __init__(
        self,
        x1_min, x1_max, n1,
        x2_min, x2_max, n2,
        geometry="cartesian",
        sigma_func=None,
        scat_func=None,
        EOS=None,
        invEOS=None,
        dt=1e-4,
        closure_func=closure_levermore,
        c_light=C_LIGHT,
        a_rad=A_RAD,
        bc_x1_lo="reflect",
        bc_x1_hi="marshak",
        bc_x2_lo="reflect",
        bc_x2_hi="marshak",
        bc_x1_lo_state=None,
        bc_x1_hi_state=None,
        bc_x2_lo_state=None,
        bc_x2_hi_state=None,
        max_nonlinear_iter=20,
        nonlinear_tol=1e-6,
        relaxation=1.0,
        x1_faces=None,
        x2_faces=None,
    ):
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.geometry = geometry.lower()
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError("geometry must be 'cartesian' or 'cylindrical'")

        self.dt = float(dt)
        self.c_light = float(c_light)
        self.a_rad = float(a_rad)
        self.closure_func = closure_func
        self.max_nonlinear_iter = int(max_nonlinear_iter)
        self.nonlinear_tol = float(nonlinear_tol)
        self.relaxation = float(relaxation)

        # ── Grid ────────────────────────────────────────────────────────────
        if x1_faces is not None:
            self.x1_faces = np.asarray(x1_faces, dtype=float)
            if len(self.x1_faces) != n1 + 1:
                raise ValueError(
                    f"x1_faces length {len(self.x1_faces)} != n1+1={n1+1}")
        else:
            self.x1_faces = np.linspace(x1_min, x1_max, n1 + 1)
        if x2_faces is not None:
            self.x2_faces = np.asarray(x2_faces, dtype=float)
            if len(self.x2_faces) != n2 + 1:
                raise ValueError(
                    f"x2_faces length {len(self.x2_faces)} != n2+1={n2+1}")
        else:
            self.x2_faces = np.linspace(x2_min, x2_max, n2 + 1)
        self.x1_c = 0.5 * (self.x1_faces[1:] + self.x1_faces[:-1])  # (n1,)
        self.x2_c = 0.5 * (self.x2_faces[1:] + self.x2_faces[:-1])  # (n2,)
        # Per-cell widths — shape (n1,) and (n2,), work for both uniform and
        # non-uniform meshes.  Broadcast as dx1[:,None] or dx2[None,:] in
        # the divergence operator.
        self.dx1 = np.diff(self.x1_faces)   # (n1,)
        self.dx2 = np.diff(self.x2_faces)   # (n2,)

        # 2-D cell-centre coordinate meshes, shape (n1, n2)
        self.X1, self.X2 = np.meshgrid(self.x1_c, self.x2_c, indexing="ij")

        # Area scale factors for the x1-direction faces.
        # Cartesian: uniform (all 1).  Cylindrical: r_face (the radial position).
        if self.geometry == "cartesian":
            self.A1_faces = np.ones(n1 + 1)
        else:
            self.A1_faces = np.maximum(self.x1_faces, 1e-12)  # r_face, (n1+1,)
            self.r_c = np.maximum(self.X1, 1e-12)             # r at cell centres

        # Per-face "external" flags for the Gauss-Seidel sweep: True means
        # the face carries full Rusanov dissipation (a genuine cell-to-cell
        # boundary); False means a purely centred, non-dissipative coupling
        # (used by the Simple-Corner-Balance subclass M1SolverSCB2D for its
        # internal sub-cell faces).  Every face is external by default.
        self.is_ext_x1 = np.ones(n1 + 1, dtype=bool)
        self.is_ext_x2 = np.ones(n2 + 1, dtype=bool)

        # ── Boundary conditions ─────────────────────────────────────────────
        for mode, name in [
            (bc_x1_lo, "bc_x1_lo"), (bc_x1_hi, "bc_x1_hi"),
            (bc_x2_lo, "bc_x2_lo"), (bc_x2_hi, "bc_x2_hi"),
        ]:
            if mode not in self._VALID_BC:
                raise ValueError(f"{name}={mode!r} must be one of {self._VALID_BC}")

        self.bc_x1_lo = bc_x1_lo;  self.bc_x1_lo_state = bc_x1_lo_state
        self.bc_x1_hi = bc_x1_hi;  self.bc_x1_hi_state = bc_x1_hi_state
        self.bc_x2_lo = bc_x2_lo;  self.bc_x2_lo_state = bc_x2_lo_state
        self.bc_x2_hi = bc_x2_hi;  self.bc_x2_hi_state = bc_x2_hi_state

        for mode, state, name in [
            (bc_x1_lo, bc_x1_lo_state, "x1_lo"),
            (bc_x1_hi, bc_x1_hi_state, "x1_hi"),
            (bc_x2_lo, bc_x2_lo_state, "x2_lo"),
            (bc_x2_hi, bc_x2_hi_state, "x2_hi"),
        ]:
            if mode in ("incident", "callable") and state is None:
                raise ValueError(
                    f"bc_{name}_state must be given when mode='incident'/'callable'")

        # ── Material callables ───────────────────────────────────────────────
        _a = float(a_rad)
        self.sigma_func = sigma_func or (
            lambda T: np.ones_like(np.asarray(T, dtype=float)))
        self.scat_func  = scat_func  or (
            lambda T: np.zeros_like(np.asarray(T, dtype=float)))
        self.EOS    = EOS    or (
            lambda T: _a * np.maximum(np.asarray(T, dtype=float), 1e-8) ** 4)
        self.invEOS = invEOS or (
            lambda e: np.maximum(
                np.maximum(np.asarray(e, dtype=float), 1e-30) / _a, 1e-30
            ) ** 0.25)

        # ── State arrays, shape (n1, n2) ─────────────────────────────────────
        self.Er = np.zeros((n1, n2))
        self.F1 = np.zeros((n1, n2))
        self.F2 = np.zeros((n1, n2))
        self.T  = np.zeros((n1, n2))

    # ────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ────────────────────────────────────────────────────────────────────────

    def initialize(self, T_init=1e-3):
        """Set cold-equilibrium initial conditions.

        Parameters
        ----------
        T_init : float or ndarray (n1, n2)
            Initial temperature [keV].  A scalar is broadcast.
        """
        T = np.broadcast_to(np.asarray(T_init, dtype=float),
                             (self.n1, self.n2)).copy()
        self.T[:, :]  = T
        self.Er[:, :] = self.a_rad * T ** 4
        self.F1[:, :] = 0.0
        self.F2[:, :] = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Closure and pressure tensor
    # ────────────────────────────────────────────────────────────────────────

    def _closure_eval(self, Er, F1, F2):
        """Reduced flux f, Eddington factor χ, and unit-vector components."""
        mag = np.sqrt(F1 ** 2 + F2 ** 2)
        denom = self.c_light * np.maximum(Er, 1e-20)
        f = np.clip(mag / denom, 0.0, 1.0)
        chi = self.closure_func(f)
        safe = np.maximum(mag, 1e-300)
        n1 = np.where(mag > 1e-30, F1 / safe, 0.0)
        n2 = np.where(mag > 1e-30, F2 / safe, 0.0)
        return f, chi, n1, n2

    def _pressure_tensor(self, Er, F1, F2):
        """Return P₁₁, P₁₂, P₂₂, P₃₃ from the 3-D M₁ closure."""
        _, chi, n1, n2 = self._closure_eval(Er, F1, F2)
        iso   = 0.5 * (1.0 - chi)
        aniso = 0.5 * (3.0 * chi - 1.0)
        P11 = Er * (iso + aniso * n1 ** 2)
        P12 = Er * aniso * n1 * n2
        P22 = Er * (iso + aniso * n2 ** 2)
        P33 = Er * iso          # = P_φφ (azimuthal / out-of-plane)
        return P11, P12, P22, P33

    # ────────────────────────────────────────────────────────────────────────
    # Physical flux vectors
    # ────────────────────────────────────────────────────────────────────────

    def _flux_x1(self, Er, F1, F2):
        """Physical flux in x1-direction: (F₁, c²P₁₁, c²P₁₂)."""
        P11, P12, _, _ = self._pressure_tensor(Er, F1, F2)
        c2 = self.c_light ** 2
        return F1.copy(), c2 * P11, c2 * P12

    def _flux_x2(self, Er, F1, F2):
        """Physical flux in x2-direction: (F₂, c²P₁₂, c²P₂₂)."""
        _, P12, P22, _ = self._pressure_tensor(Er, F1, F2)
        c2 = self.c_light ** 2
        return F2.copy(), c2 * P12, c2 * P22

    # ────────────────────────────────────────────────────────────────────────
    # Ghost-cell extension
    # ────────────────────────────────────────────────────────────────────────

    def _extend_with_ghosts(self, Er, F1, F2):
        """Pad interior arrays with one layer of ghost cells on all 4 faces.

        Returns three (n1+2, n2+2) arrays.  The interior occupies
        [1:n1+1, 1:n2+1].  Ghost rows/columns encode the boundary conditions.
        """
        n1, n2 = self.n1, self.n2
        c = self.c_light

        Eg = np.zeros((n1 + 2, n2 + 2))
        G1 = np.zeros((n1 + 2, n2 + 2))
        G2 = np.zeros((n1 + 2, n2 + 2))

        Eg[1:n1+1, 1:n2+1] = Er
        G1[1:n1+1, 1:n2+1] = F1
        G2[1:n1+1, 1:n2+1] = F2

        # x1_lo ghost row (index 0)
        Er_e = Er[0, :];  F1_e = F1[0, :];  F2_e = F2[0, :]
        mode = self.bc_x1_lo
        if mode == "incident":
            er_bc, f1_bc, f2_bc = self.bc_x1_lo_state
            Eg[0, 1:n2+1] = er_bc;  G1[0, 1:n2+1] = f1_bc;  G2[0, 1:n2+1] = f2_bc
        elif mode == "callable":
            eg, g1, g2 = self.bc_x1_lo_state(Er_e, F1_e, F2_e)
            Eg[0, 1:n2+1] = eg;     G1[0, 1:n2+1] = g1;     G2[0, 1:n2+1] = g2
        elif mode == "reflect":
            Eg[0, 1:n2+1] = Er_e;   G1[0, 1:n2+1] = -F1_e;  G2[0, 1:n2+1] = F2_e
        elif mode == "marshak":
            Eg[0, 1:n2+1] = Er_e;   G1[0, 1:n2+1] = -0.5*c*Er_e; G2[0,1:n2+1] = F2_e
        elif mode == "free_stream":
            Eg[0, 1:n2+1] = Er_e;   G1[0, 1:n2+1] = -c*Er_e;     G2[0,1:n2+1] = F2_e
        else:
            Eg[0, 1:n2+1] = Er_e;   G1[0, 1:n2+1] = F1_e;         G2[0,1:n2+1] = F2_e

        # x1_hi (right ghost, i=n1+1)
        Er_e = Er[-1, :]; F1_e = F1[-1, :]; F2_e = F2[-1, :]
        mode = self.bc_x1_hi
        if mode == "incident":
            er_bc, f1_bc, f2_bc = self.bc_x1_hi_state
            Eg[n1+1,1:n2+1]=er_bc; G1[n1+1,1:n2+1]=f1_bc; G2[n1+1,1:n2+1]=f2_bc
        elif mode == "callable":
            eg, g1, g2 = self.bc_x1_hi_state(Er_e, F1_e, F2_e)
            Eg[n1+1,1:n2+1]=eg;    G1[n1+1,1:n2+1]=g1;    G2[n1+1,1:n2+1]=g2
        elif mode == "reflect":
            Eg[n1+1,1:n2+1]=Er_e;  G1[n1+1,1:n2+1]=-F1_e; G2[n1+1,1:n2+1]=F2_e
        elif mode == "marshak":
            Eg[n1+1,1:n2+1]=Er_e;  G1[n1+1,1:n2+1]=0.5*c*Er_e; G2[n1+1,1:n2+1]=F2_e
        elif mode == "free_stream":
            Eg[n1+1,1:n2+1]=Er_e;  G1[n1+1,1:n2+1]=c*Er_e;     G2[n1+1,1:n2+1]=F2_e
        else:
            Eg[n1+1,1:n2+1]=Er_e;  G1[n1+1,1:n2+1]=F1_e;        G2[n1+1,1:n2+1]=F2_e

        # x2_lo (bottom ghost, j=0)
        Er_e = Er[:, 0]; F1_e = F1[:, 0]; F2_e = F2[:, 0]
        mode = self.bc_x2_lo
        if mode == "incident":
            er_bc, f1_bc, f2_bc = self.bc_x2_lo_state
            Eg[1:n1+1,0]=er_bc; G1[1:n1+1,0]=f1_bc; G2[1:n1+1,0]=f2_bc
        elif mode == "callable":
            eg, g1, g2 = self.bc_x2_lo_state(Er_e, F1_e, F2_e)
            Eg[1:n1+1,0]=eg;    G1[1:n1+1,0]=g1;    G2[1:n1+1,0]=g2
        elif mode == "reflect":
            Eg[1:n1+1,0]=Er_e;  G1[1:n1+1,0]=F1_e;  G2[1:n1+1,0]=-F2_e
        elif mode == "marshak":
            Eg[1:n1+1,0]=Er_e;  G1[1:n1+1,0]=F1_e;  G2[1:n1+1,0]=-0.5*c*Er_e
        elif mode == "free_stream":
            Eg[1:n1+1,0]=Er_e;  G1[1:n1+1,0]=F1_e;  G2[1:n1+1,0]=-c*Er_e
        else:
            Eg[1:n1+1,0]=Er_e;  G1[1:n1+1,0]=F1_e;  G2[1:n1+1,0]=F2_e

        # x2_hi (top ghost, j=n2+1)
        Er_e = Er[:,-1]; F1_e = F1[:,-1]; F2_e = F2[:,-1]
        mode = self.bc_x2_hi
        if mode == "incident":
            er_bc, f1_bc, f2_bc = self.bc_x2_hi_state
            Eg[1:n1+1,n2+1]=er_bc; G1[1:n1+1,n2+1]=f1_bc; G2[1:n1+1,n2+1]=f2_bc
        elif mode == "callable":
            eg, g1, g2 = self.bc_x2_hi_state(Er_e, F1_e, F2_e)
            Eg[1:n1+1,n2+1]=eg;    G1[1:n1+1,n2+1]=g1;    G2[1:n1+1,n2+1]=g2
        elif mode == "reflect":
            Eg[1:n1+1,n2+1]=Er_e;  G1[1:n1+1,n2+1]=F1_e;  G2[1:n1+1,n2+1]=-F2_e
        elif mode == "marshak":
            Eg[1:n1+1,n2+1]=Er_e;  G1[1:n1+1,n2+1]=F1_e;  G2[1:n1+1,n2+1]=0.5*c*Er_e
        elif mode == "free_stream":
            Eg[1:n1+1,n2+1]=Er_e;  G1[1:n1+1,n2+1]=F1_e;  G2[1:n1+1,n2+1]=c*Er_e
        else:
            Eg[1:n1+1,n2+1]=Er_e;  G1[1:n1+1,n2+1]=F1_e;  G2[1:n1+1,n2+1]=F2_e

        return Eg, G1, G2

    # ────────────────────────────────────────────────────────────────────────
    # Rusanov numerical fluxes
    # ────────────────────────────────────────────────────────────────────────

    def _rusanov_x1(self, Eg, G1g, G2g):
        """Rusanov fluxes at all (n1+1)×n2 x1-faces.

        Returns three arrays of shape (n1+1, n2).
        """
        c = self.c_light
        n1, n2 = self.n1, self.n2

        # Left and right states at each x1-face
        ErL = Eg[0:n1+1, 1:n2+1]; F1L = G1g[0:n1+1, 1:n2+1]; F2L = G2g[0:n1+1, 1:n2+1]
        ErR = Eg[1:n1+2, 1:n2+1]; F1R = G1g[1:n1+2, 1:n2+1]; F2R = G2g[1:n1+2, 1:n2+1]

        gEr_L, gF1_L, gF2_L = self._flux_x1(ErL, F1L, F2L)
        gEr_R, gF1_R, gF2_R = self._flux_x1(ErR, F1R, F2R)

        GEr = 0.5*(gEr_L + gEr_R) - 0.5*c*(ErR - ErL)
        GF1 = 0.5*(gF1_L + gF1_R) - 0.5*c*(F1R - F1L)
        GF2 = 0.5*(gF2_L + gF2_R) - 0.5*c*(F2R - F2L)
        return GEr, GF1, GF2   # (n1+1, n2)

    def _rusanov_x2(self, Eg, G1g, G2g):
        """Rusanov fluxes at all n1×(n2+1) x2-faces.

        Returns three arrays of shape (n1, n2+1).
        """
        c = self.c_light
        n1, n2 = self.n1, self.n2

        ErL = Eg[1:n1+1, 0:n2+1]; F1L = G1g[1:n1+1, 0:n2+1]; F2L = G2g[1:n1+1, 0:n2+1]
        ErR = Eg[1:n1+1, 1:n2+2]; F1R = G1g[1:n1+1, 1:n2+2]; F2R = G2g[1:n1+1, 1:n2+2]

        gEr_L, gF1_L, gF2_L = self._flux_x2(ErL, F1L, F2L)
        gEr_R, gF1_R, gF2_R = self._flux_x2(ErR, F1R, F2R)

        GEr = 0.5*(gEr_L + gEr_R) - 0.5*c*(ErR - ErL)
        GF1 = 0.5*(gF1_L + gF1_R) - 0.5*c*(F1R - F1L)
        GF2 = 0.5*(gF2_L + gF2_R) - 0.5*c*(F2R - F2L)
        return GEr, GF1, GF2   # (n1, n2+1)

    # ────────────────────────────────────────────────────────────────────────
    # Transport divergence and geometric source
    # ────────────────────────────────────────────────────────────────────────

    def _transport_divergence(self, Er, F1, F2):
        """FV divergence of the M₁ flux, shape (n1, n2) for each component."""
        Eg, G1g, G2g = self._extend_with_ghosts(Er, F1, F2)

        Gx_Er, Gx_F1, Gx_F2 = self._rusanov_x1(Eg, G1g, G2g)  # (n1+1, n2)
        Gz_Er, Gz_F1, Gz_F2 = self._rusanov_x2(Eg, G1g, G2g)  # (n1, n2+1)

        dx1 = self.dx1[:, None]   # (n1, 1) for broadcasting
        dx2 = self.dx2[None, :]   # (1, n2) for broadcasting
        if self.geometry == "cartesian":
            # Standard cell-centred finite-volume divergence.
            div_Er = (Gx_Er[1:,:] - Gx_Er[:-1,:]) / dx1 \
                   + (Gz_Er[:,1:] - Gz_Er[:,:-1]) / dx2
            div_F1 = (Gx_F1[1:,:] - Gx_F1[:-1,:]) / dx1 \
                   + (Gz_F1[:,1:] - Gz_F1[:,:-1]) / dx2
            div_F2 = (Gx_F2[1:,:] - Gx_F2[:-1,:]) / dx1 \
                   + (Gz_F2[:,1:] - Gz_F2[:,:-1]) / dx2
        else:
            # Cylindrical: (1/r) ∂(r G_r)/∂r + ∂G_z/∂z
            # Area factors A1_faces = r_faces; cell "volume" ∝ r_c·Δr
            r_R = self.A1_faces[1:, None]    # (n1, 1)  r at right radial face
            r_L = self.A1_faces[:-1, None]   # (n1, 1)  r at left  radial face
            rc  = self.r_c                   # (n1, n2) r at cell centres

            div_Er = (r_R*Gx_Er[1:,:] - r_L*Gx_Er[:-1,:]) / (rc * dx1) \
                   + (Gz_Er[:,1:] - Gz_Er[:,:-1]) / dx2
            div_F1 = (r_R*Gx_F1[1:,:] - r_L*Gx_F1[:-1,:]) / (rc * dx1) \
                   + (Gz_F1[:,1:] - Gz_F1[:,:-1]) / dx2
            div_F2 = (r_R*Gx_F2[1:,:] - r_L*Gx_F2[:-1,:]) / (rc * dx1) \
                   + (Gz_F2[:,1:] - Gz_F2[:,:-1]) / dx2

        return div_Er, div_F1, div_F2

    def _geo_source_F1(self, Er, F1, F2):
        """Geometric source c² P_φφ/r in the r-momentum equation (cylindrical).

        This term arises from ∇·P in cylindrical coordinates:
          (∇·P)_r = (1/r)∂(rP_rr)/∂r + ∂P_rz/∂z − P_φφ/r
        The FV divergence handles the first two terms; the −P_φφ/r term,
        when brought to the right-hand side of the momentum equation,
        contributes +c²P_φφ/r to the flux update.

        Returns zero for Cartesian geometry.
        """
        if self.geometry == "cartesian":
            return np.zeros_like(Er)
        _, _, _, P33 = self._pressure_tensor(Er, F1, F2)
        return self.c_light**2 * P33 / self.r_c   # shape (n1, n2)

    # ────────────────────────────────────────────────────────────────────────
    # Symmetric Gauss-Seidel sweep (unconditionally stable, no CFL limit)
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # Face-dissipation weights (hook for SCB opacity-dependent weighting)
    # ────────────────────────────────────────────────────────────────────────

    def _get_ext_weights(self, sigma_tot):
        """Return (ext_x1_full, ext_x2_full): full 2-D face-dissipation
        weight arrays (1.0 = full Rusanov dissipation, 0.0 = purely
        centred/no dissipation).

        Base implementation: static, from ``self.is_ext_x1`` /
        ``self.is_ext_x2`` (broadcast across the other direction) —
        every face is external (weight 1.0) by default.  Subclasses
        (e.g. ``M1SolverSCB2D``) override this to make the *internal*
        face weight depend on the local opacity ``sigma_tot``.

        Returns
        -------
        ext_x1_full : ndarray, shape (n1+1, n2)
        ext_x2_full : ndarray, shape (n1, n2+1)
        """
        n1, n2 = self.n1, self.n2
        ext_x1_full = np.broadcast_to(
            self.is_ext_x1.astype(float)[:, None], (n1 + 1, n2))
        ext_x2_full = np.broadcast_to(
            self.is_ext_x2.astype(float)[None, :], (n1, n2 + 1))
        return ext_x1_full, ext_x2_full

    def _sweep_gs_2d_one(self, Er_it, F1_it, F2_it, Er_old, F1_old, F2_old,
                         sigma_eff, sigma_tot, U_n, source, forward):
        """One directional cell-implicit Gauss-Seidel sweep (in-place).

        Mirrors ``M1Solver1D._sweep_gs_one``, generalised to two spatial
        dimensions and three transported moments (Er, F1, F2).  Each cell
        is solved implicitly for its own Rusanov self-dissipation
        coefficient — the stiff term responsible for the CFL restriction
        in a Jacobi (fully-explicit) scheme — while already-updated
        neighbours (earlier in the *same* pass) supply the streaming terms
        explicitly.  A forward pass (low-i/low-j → high-i/high-j) followed
        by a backward pass propagates boundary information across the
        whole domain in one sweep pair, giving an unconditionally stable
        scheme with no CFL restriction (no characteristic direction needs
        to be known in advance).

        Each of the 4 faces bordering a cell is independently "external"
        (``self.is_ext_x1``/``self.is_ext_x2`` True — full Rusanov
        dissipation, a genuine cell-to-cell boundary) or "internal" (False
        — purely centred coupling, no dissipation).  Every face is
        external by default, giving the plain first-order FV scheme; the
        ``M1SolverSCB2D`` subclass sets some faces internal to implement a
        Simple-Corner-Balance discretisation (sub-cell resolution without
        adding numerical dissipation at the sub-cell scale — this is what
        gives SCB improved accuracy in optically thick regions).

        The nonlinear M1-closure pressure-tensor terms are evaluated once
        per pass from the iterate state at the *start* of the pass (a
        single vectorised call) — this mirrors how the Jacobi scheme's
        geometric-curvature source is already handled, and only affects
        cylindrical geometry: in Cartesian geometry A1_lo = A1_hi so these
        correction terms vanish identically.

        Parameters
        ----------
        forward : bool
            ``True`` for a bottom-left → top-right sweep (i, j ascending);
            ``False`` for the reverse (i, j descending).
        """
        n1, n2 = self.n1, self.n2
        c  = self.c_light
        dt = self.dt

        # ── Pass-start snapshot: ghost extension + pressure tensor ───────
        Eg, G1g, G2g = self._extend_with_ghosts(Er_it, F1_it, F2_it)
        P11g, P12g, P22g, _ = self._pressure_tensor(Eg, G1g, G2g)
        geo_F1 = self._geo_source_F1(Er_it, F1_it, F2_it)          # (n1, n2)

        # ── Geometry-dependent coefficients (mesh-only) ──────────────────
        A1_lo = self.A1_faces[:-1]        # (n1,)
        A1_hi = self.A1_faces[1:]         # (n1,)
        if self.geometry == "cartesian":
            denom1_2d = np.broadcast_to(self.dx1[:, None], (n1, n2))
        else:
            denom1_2d = self.r_c * self.dx1[:, None]              # (n1, n2)

        # Local pressure-tensor asymmetry corrections (cylindrical only;
        # exactly zero for Cartesian geometry since A1_lo = A1_hi there).
        # Unconditional: this term comes from the always-present centred
        # flux average, independent of the external/internal face flags.
        P11_local = P11g[1:n1+1, 1:n2+1]
        P12_local = P12g[1:n1+1, 1:n2+1]
        dA1 = (A1_hi - A1_lo)[:, None]
        corr1 = 0.5 * c**2 * dA1 / denom1_2d * P11_local
        corr2 = 0.5 * c**2 * dA1 / denom1_2d * P12_local

        # ── Per-face external/internal weights (1.0 = full dissipation,
        #    0.0 = purely centred).  May vary over the full 2-D grid for
        #    opacity-dependent subclasses (M1SolverSCB2D). ────────────────
        ext_x1_full, ext_x2_full = self._get_ext_weights(sigma_tot)
        # (n1+1, n2) and (n1, n2+1)

        # Self-dissipation coefficient — identical structure for Er, F1, F2.
        # Only external faces contribute (internal faces carry no, or
        # reduced, dissipation, hence no/reduced self-coupling).
        coeff_c = (0.5 * c * (ext_x1_full[:-1, :] * A1_lo[:, None]
                              + ext_x1_full[1:, :] * A1_hi[:, None])
                   / denom1_2d
                   + 0.5 * c * (ext_x2_full[:, :-1] + ext_x2_full[:, 1:])
                   / self.dx2[None, :])                            # (n1, n2)

        i_seq = range(n1) if forward else range(n1 - 1, -1, -1)
        j_seq = range(n2) if forward else range(n2 - 1, -1, -1)

        for i in i_seq:
            A_L = A1_lo[i]; A_R = A1_hi[i]
            for j in j_seq:
                dx2_j = self.dx2[j]
                den1  = denom1_2d[i, j]
                eLx1 = ext_x1_full[i, j];   eRx1 = ext_x1_full[i+1, j]
                eBx2 = ext_x2_full[i, j];   eTx2 = ext_x2_full[i, j+1]

                # ── Neighbour raw states (live array; ghost at edges) ────
                Er_L = Er_it[i-1, j] if i > 0      else Eg[0,      j+1]
                F1_L = F1_it[i-1, j] if i > 0      else G1g[0,     j+1]
                F2_L = F2_it[i-1, j] if i > 0      else G2g[0,     j+1]

                Er_R = Er_it[i+1, j] if i < n1 - 1 else Eg[n1+1,   j+1]
                F1_R = F1_it[i+1, j] if i < n1 - 1 else G1g[n1+1,  j+1]
                F2_R = F2_it[i+1, j] if i < n1 - 1 else G2g[n1+1,  j+1]

                Er_B = Er_it[i, j-1] if j > 0      else Eg[i+1,    0]
                F1_B = F1_it[i, j-1] if j > 0      else G1g[i+1,   0]
                F2_B = F2_it[i, j-1] if j > 0      else G2g[i+1,   0]

                Er_T = Er_it[i, j+1] if j < n2 - 1 else Eg[i+1,    n2+1]
                F1_T = F1_it[i, j+1] if j < n2 - 1 else G1g[i+1,   n2+1]
                F2_T = F2_it[i, j+1] if j < n2 - 1 else G2g[i+1,   n2+1]

                # ── Neighbour pressure-tensor snapshot (padded indices) ──
                P11_L = P11g[i,   j+1];  P11_R = P11g[i+2, j+1]
                P12_L = P12g[i,   j+1];  P12_R = P12g[i+2, j+1]
                P12_B = P12g[i+1, j  ];  P12_T = P12g[i+1, j+2]
                P22_B = P22g[i+1, j  ];  P22_T = P22g[i+1, j+2]

                # ── Er update (implicit self term only; the local F1/F2
                #    flux terms cancel exactly in the divergence, matching
                #    the 1-D precedent).  The "other flux component"
                #    (F1/F2) terms are always centred (unconditional); the
                #    same-variable (Er) terms carry dissipation only at
                #    external faces (eLx1, eRx1, eBx2, eTx2) ─────────────
                er_num = (Er_old[i, j] / dt
                          + 0.5 * (A_L * F1_L - A_R * F1_R) / den1
                          + 0.5 * (A_L * eLx1 * c * Er_L
                                   + A_R * eRx1 * c * Er_R) / den1
                          + 0.5 * (F2_B - F2_T) / dx2_j
                          + 0.5 * (eBx2 * c * Er_B + eTx2 * c * Er_T) / dx2_j
                          + sigma_eff[i, j] * c * U_n[i, j] + source[i, j])
                er_den = 1.0 / dt + coeff_c[i, j] + sigma_eff[i, j] * c
                Er_it[i, j] = er_num / er_den

                # ── F1 update ─────────────────────────────────────────────
                f1_num = (F1_old[i, j] / dt
                          + 0.5 * (A_L * c**2 * P11_L - A_R * c**2 * P11_R) / den1
                          + 0.5 * (A_L * eLx1 * c * F1_L
                                   + A_R * eRx1 * c * F1_R) / den1
                          + 0.5 * (c**2 * P12_B - c**2 * P12_T) / dx2_j
                          + 0.5 * (eBx2 * c * F1_B + eTx2 * c * F1_T) / dx2_j
                          - corr1[i, j] + geo_F1[i, j])
                f1_den = 1.0 / dt + coeff_c[i, j] + sigma_tot[i, j] * c
                F1_it[i, j] = f1_num / f1_den

                # ── F2 update ─────────────────────────────────────────────
                f2_num = (F2_old[i, j] / dt
                          + 0.5 * (A_L * c**2 * P12_L - A_R * c**2 * P12_R) / den1
                          + 0.5 * (A_L * eLx1 * c * F2_L
                                   + A_R * eRx1 * c * F2_R) / den1
                          + 0.5 * (c**2 * P22_B - c**2 * P22_T) / dx2_j
                          + 0.5 * (eBx2 * c * F2_B + eTx2 * c * F2_T) / dx2_j
                          - corr2[i, j])
                f2_den = 1.0 / dt + coeff_c[i, j] + sigma_tot[i, j] * c
                F2_it[i, j] = f2_num / f2_den

                # ── Realizability clamp: |F| <= c Er ─────────────────────
                # Only meaningful for Er > 0; a negative Er_it (tolerated
                # transient, see step()'s post-step validation) makes
                # max_F < 0, which must NOT trigger the clamp (it would
                # flip F1/F2 to a nonsensical negative-scaled value).
                max_F = c * Er_it[i, j]
                if max_F > 0.0:
                    mag = (F1_it[i, j]**2 + F2_it[i, j]**2) ** 0.5
                    if mag > max_F:
                        scale = max_F / mag
                        F1_it[i, j] *= scale
                        F2_it[i, j] *= scale

    # ────────────────────────────────────────────────────────────────────────
    # Time step
    # ────────────────────────────────────────────────────────────────────────

    def step(self, source=None, verbose=False):
        """Advance one time step with Fleck–Cummings and symmetric Gauss–Seidel.

        Parameters
        ----------
        source : ndarray (n1, n2) or None
            External isotropic source entering the energy equation [GJ cm⁻³ ns⁻¹].
        verbose : bool
            Print a warning if the SGS loop does not converge.

        Raises
        ------
        ValueError
            If the state is non-physical at entry or after the step.

        Notes
        -----
        The nonlinear M1 closure is handled by a symmetric Gauss-Seidel (SGS)
        iteration: each outer iteration sweeps every cell once from
        bottom-left to top-right, then once from top-right to bottom-left.
        Because each cell is updated implicitly using the most recent
        neighbour values (in-place updates), boundary information
        propagates across the entire domain in a single pass and the
        scheme is unconditionally stable with **no CFL condition** — no
        characteristic direction needs to be known in advance.
        """
        if source is None:
            source = np.zeros((self.n1, self.n2))

        Er_old = self.Er.copy()
        F1_old = self.F1.copy()
        F2_old = self.F2.copy()
        T_old  = self.T.copy()

        if np.any(T_old <= 0.0):
            raise ValueError(
                f"Non-positive temperature before step: min T = {T_old.min():.3e}")
        # Negative Er before step is tolerated (e.g. P1 oscillations).
        if np.any(Er_old < 0.0) and verbose:
            print(f"  [m1_2d] negative Er at step start: min={Er_old.min():.3e}")

        c  = self.c_light
        dt = self.dt

        # ── Fleck–Cummings linearisation at T_old ────────────────────────────
        sigma     = self.sigma_func(T_old)
        scat      = self.scat_func(T_old)
        sigma_tot = sigma + scat

        h_cv = T_old * 1e-4 + 1e-12
        Cv = (self.EOS(T_old + h_cv) - self.EOS(T_old - h_cv)) / (2.0 * h_cv)
        if np.any(Cv <= 0.0):
            raise ValueError(
                f"Non-positive heat capacity from EOS: min Cv = {Cv.min():.3e}")
        beta      = 4.0 * self.a_rad * T_old**3 / Cv
        f_FC      = 1.0 / (1.0 + beta * c * sigma * dt)   # Fleck factor
        sigma_eff = f_FC * sigma

        U_n   = self.a_rad * T_old**4
        e_old = self.EOS(T_old)

        # ── Symmetric Gauss-Seidel iteration for nonlinear closure ───────────
        Er_it = Er_old.copy()
        F1_it = F1_old.copy()
        F2_it = F2_old.copy()

        converged = False
        for _ in range(self.max_nonlinear_iter):
            Er_prev = Er_it.copy()
            F1_prev = F1_it.copy()
            F2_prev = F2_it.copy()

            # One forward + one backward sweep per SGS iteration.
            self._sweep_gs_2d_one(Er_it, F1_it, F2_it, Er_old, F1_old, F2_old,
                                  sigma_eff, sigma_tot, U_n, source, forward=True)
            self._sweep_gs_2d_one(Er_it, F1_it, F2_it, Er_old, F1_old, F2_old,
                                  sigma_eff, sigma_tot, U_n, source, forward=False)

            if self.relaxation < 1.0:
                w      = self.relaxation
                Er_it  = (1.0 - w) * Er_prev + w * Er_it
                F1_it  = (1.0 - w) * F1_prev + w * F1_it
                F2_it  = (1.0 - w) * F2_prev + w * F2_it

            dEr = np.max(np.abs(Er_it - Er_prev) / np.maximum(np.abs(Er_it), 1e-16))
            dF1 = np.max(np.abs(F1_it - F1_prev) / np.maximum(np.abs(F1_it), 1e-16))
            dF2 = np.max(np.abs(F2_it - F2_prev) / np.maximum(np.abs(F2_it), 1e-16))

            if max(dEr, dF1, dF2) < self.nonlinear_tol:
                converged = True
                break

        if verbose and not converged:
            print(
                f"Warning: step did not reach tol={self.nonlinear_tol:.1e} "
                f"in {self.max_nonlinear_iter} iterations")

        # ── Fleck–Cummings material energy update ─────────────────────────────
        e_new = e_old + f_FC * c * sigma * dt * (Er_it - U_n)
        T_new = self.invEOS(np.maximum(e_new, 1e-30))

        # ── Post-step validation ───────────────────────────────────────────────
        # The SGS scheme and certain closures (e.g. P1) can produce small
        # negative Er values.  We do NOT clip the state here because clipping
        # Er while leaving F unchanged creates |F|/(c Er) >> 1, which causes
        # the next step's Rusanov divergence to blow up.  Instead, negative Er
        # is handled inside _closure_eval via  np.maximum(Er, 1e-20)  in the
        # denominator, so the closure remains well-defined.  Only a truly
        # catastrophic negative temperature signals a real divergence.

        er_min = float(Er_it.min())
        if er_min < 0.0 and verbose:
            print(f"  [m1_2d] negative Er "
                  f"(min={er_min:.3e}, max={Er_it.max():.3e})")

        if np.any(T_new <= 0.0):
            raise ValueError(
                f"Non-positive temperature after step: min T = {T_new.min():.3e}")

        self.Er[:] = Er_it
        self.F1[:] = F1_it
        self.F2[:] = F2_it
        self.T[:]  = T_new

    def run(self, n_steps, source_func=None, output_times=None):
        """Run for ``n_steps`` steps, saving snapshots at ``output_times`` [ns].

        Parameters
        ----------
        source_func : callable t -> ndarray (n1, n2), optional
            External isotropic source as a function of time.
        output_times : list of float, optional
            Physical times [ns] at which to save snapshots.

        Returns
        -------
        snapshots : dict
            Keyed by the requested output time [ns].  Each value is a dict
            with keys ``'x1', 'x2', 'Er', 'F1', 'F2', 'T'``.
        """
        if output_times is None:
            output_times = []
        output_times = list(output_times)
        out_idx = 0
        snapshots = {}

        t = 0.0
        for istep in range(n_steps):
            t_old = t
            t = (istep + 1) * self.dt
            src = (np.zeros((self.n1, self.n2)) if source_func is None
                   else source_func(t_old))
            self.step(source=src)
            if out_idx < len(output_times) and t >= output_times[out_idx]:
                snapshots[output_times[out_idx]] = {
                    "x1": self.x1_c.copy(),
                    "x2": self.x2_c.copy(),
                    "Er": self.Er.copy(),
                    "F1": self.F1.copy(),
                    "F2": self.F2.copy(),
                    "T":  self.T.copy(),
                }
                out_idx += 1
        return snapshots

    # ────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ────────────────────────────────────────────────────────────────────────

    @property
    def T_rad(self):
        """Radiation temperature (aT⁴ = Er) as a 2-D array, shape (n1, n2)."""
        return (np.maximum(self.Er, 0.0) / self.a_rad) ** 0.25

    @property
    def reduced_flux(self):
        """Reduced flux magnitude |F|/(c Er), shape (n1, n2)."""
        mag = np.sqrt(self.F1**2 + self.F2**2)
        return mag / (self.c_light * np.maximum(self.Er, 1e-20))

    @property
    def cfl_number(self):
        """Informational CFL number max c·Δt·(1/Δx₁ᵢ + 1/Δx₂ⱼ).

        Not a stability requirement — the symmetric Gauss-Seidel scheme is
        unconditionally stable — but useful as an accuracy/resolution guide.
        """
        return float(self.c_light * self.dt
                     * np.max(1.0 / self.dx1[:, None] + 1.0 / self.dx2[None, :]))
