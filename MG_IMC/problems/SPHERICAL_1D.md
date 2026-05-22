# 1-D Spherical Multigroup IMC — Gray-Equivalent Test

A convergence/equivalence test that verifies the multigroup 1-D spherical
IMC (`MG_IMC1D`) against the gray 1-D spherical IMC (`IMC1D`).  When all
groups carry the same constant opacity the two codes must agree within
Monte Carlo noise.

## Physical Setup

Solid sphere of radius R = 10 cm with 20 uniform radial zones.

| Parameter         | Value             |
|-------------------|-------------------|
| Geometry          | 1-D spherical     |
| R (outer radius)  | 10 cm             |
| N_cells           | 20                |
| BC (outer)        | Vacuum            |

## Governing Equations

1-D spherical multigroup transport coupled to material energy:

$$\frac{1}{c}\frac{\partial E_g}{\partial t}
  + \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 F_g)
  = c\,\sigma_{a,g}(a T^4 b_g - E_g)$$

$$\rho\,c_v\frac{\partial T}{\partial t}
  = c\sum_g \sigma_{a,g}(E_g - aT^4 b_g)$$

## Opacity Model

**Constant (temperature- and frequency-independent):**

$$\sigma_{a,g} = \sigma_0 = 10\ \text{cm}^{-1} \quad \text{for all groups } g$$

Because σ is the same for every group, the multigroup simulation must
produce statistically equivalent results to a single-group (gray) run.

## Equation of State

Linear:

$$u = c_v\,T, \qquad c_v = 1.0\ \text{GJ/(cm³·keV)}\ \text{(volumetric)}$$

## Initial Conditions

| Quantity      | Value     |
|---------------|-----------|
| T_mat(r, 0)   | 1.0 keV   |
| T_rad(r, 0)   | 0.1 keV   |

The system starts out of equilibrium (T_mat ≠ T_rad).

## Boundary Conditions

**Outer boundary (r = R):** Vacuum — no incoming radiation.

## Energy Groups

| Parameter   | Value                              |
|-------------|------------------------------------|
| N_groups    | 4                                  |
| ν range     | [0, 40] keV (equal-width groups)   |
| Spacing     | Uniform                            |

## Expected Behavior

Both gray and multigroup simulations should produce identical T_mat(r) and
T_rad(r) profiles at the final time within Monte Carlo noise (~1–2%).  The
radiation temperature starts at 0.1 keV and rises as material energy is
transferred to the radiation field; T_mat drops from 1 keV and the two
temperatures converge toward equilibrium.

Even though all groups carry the same constant opacity, the multigroup run
still uses temperature-dependent Planck-weighted emission fractions
$b_g(T) = B_g(T)/\sum_{g'} B_{g'}(T)$ (from `planck_integrals`).  With
constant σ, the Planck mean opacity equals σ_0 regardless of how the
fractions are distributed, so the gray and multigroup evolutions agree.

Because the sphere is finite and the outer BC is vacuum, energy leaks out
over the diffusion timescale τ ≈ R²/(c λ_mfp) ≈ R² σ/c.

## Files

| File                        | Purpose                                    |
|-----------------------------|--------------------------------------------|
| `test_mg_imc1d_spherical.py` | Multigroup and gray comparison, generates `test_mg_vs_gray.pdf` |
