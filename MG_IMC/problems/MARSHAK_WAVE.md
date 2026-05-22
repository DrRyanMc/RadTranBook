# Marshak Wave — Multigroup Power-Law Opacity

A 1-D slab radiation-transport benchmark in which a hot left boundary
drives a radiation wave through a cold, optically-thick medium.  The
power-law opacity creates a spectrally-complex wave that cannot be
captured accurately by gray methods.

## Physical Setup

Semi-infinite slab geometry (treated as finite 1-D with vacuum right BC):

```
T_bc(t) →  [  cold medium  ]  vacuum
x = 0                        x = 7 cm
```

| Parameter           | Value             |
|---------------------|-------------------|
| Domain length       | 7 cm              |
| ρ                   | 0.01 g/cm³        |
| c_v (specific)      | 0.05 GJ/(g·keV)   |
| T_init              | 0.005 keV         |

## Governing Equations

1-D multigroup IMC with slab geometry:

$$\frac{\partial E_g}{\partial t} + c\,\frac{\partial F_g}{\partial x}
  = c\,\sigma_{a,g}(acT^4\,b_g - E_g)$$

$$\rho\,c_v\frac{\partial T}{\partial t}
  = c \sum_g \sigma_{a,g}(E_g - acT^4 b_g)$$

where the Planck-weighted emission fractions are:

$$b_g(T) = \frac{B_g(T)}{\sum_{g'} B_{g'}(T)}, \qquad
  B_g(T) = \int_{\nu_{g-1}}^{\nu_g} B(\nu,T)\,d\nu$$

computed via the `planck_integrals` library at each time step.  The left
boundary injects radiation distributed across groups as
$\chi_g = B_g(T_{\rm bc}) / \sum_{g'} B_{g'}(T_{\rm bc})$.

## Opacity Model

$$\sigma_{a,g}(T) = 10\,\rho\,T^{-1/2}\,\bar{\nu}_g^{-3} \qquad [\text{cm}^{-1}]$$

using the geometric mean group energy:

$$\sigma_{a,g}(T) = \sqrt{\sigma_a(T,\nu_{g-1/2})\,\sigma_a(T,\nu_{g+1/2})}$$

Hard cap: σ_max = 10¹⁴ cm⁻¹.

## Equation of State

$$u = \rho\,c_v\,T, \qquad c_v = 0.05\ \text{GJ/(g·keV)}$$

## Boundary Conditions

**Left boundary (x = 0):** Time-dependent blackbody source with linear ramp:

$$T_{\rm bc}(t) = \begin{cases}
  T_{\rm start} + (T_{\rm end} - T_{\rm start})\,t/t_{\rm ramp} & t < t_{\rm ramp}\\
  T_{\rm end} & t \geq t_{\rm ramp}
\end{cases}$$

Default parameters: T_start = 0.05 keV, T_end = 0.25 keV, t_ramp = 5 ns.
A static version (constant T_end = 0.25 keV) is also available.

**Right boundary (x = 7 cm):** Vacuum (zero incoming intensity).

## Initial Conditions

$$T(x,0) = T_r(x,0) = T_{\rm init} = 0.005\ \text{keV}$$

## Energy Groups

| Parameter        | Default  |
|------------------|----------|
| Number of groups | 10 (variable, typically 2–50) |
| ν_min            | 0.01 keV |
| ν_max            | 10 keV   |
| Spacing          | Logarithmic |

## Mesh

Default: 140 cells, uniform or left-clustered (optional β-parameter).
Left-clustering via exponential map:

$$x_i = x_{\rm min} + (x_{\rm max} - x_{\rm min})\,
       \frac{e^{\beta s_i} - 1}{e^\beta - 1}, \quad s_i = i/N$$

## Simulation Parameters

| Parameter    | Default   |
|--------------|-----------|
| dt           | 0.01 ns   |
| T_final      | 10 ns     |
| N_particles  | 200 000 target, 100 000 boundary |
| Diagnostics  | T_mat(x), T_rad(x) at t = 1, 2, 5, 10 ns |

## Expected Behavior

The radiation wave front advances with a speed that depends on the opacity.
With power-law opacity σ ∝ ν⁻³, soft (low-energy) photons are highly
absorbed; the wave spectrum is harder than blackbody.  Gray models
significantly underestimate the penetration depth at early times and
overestimate at late times compared to multigroup.

## Files

| File                               | Purpose                               |
|------------------------------------|---------------------------------------|
| `test_marshak_wave_multigroup_powerlaw.py` | IMC runner + problem setup    |
| `../visualization/plot_marshak_wave_multigroup_powerlaw_imc_compare.py` | Multi-G comparison plot |
| `../visualization/plot_marshak_wave_multigroup_powerlaw_do_imc_compare.py` | IMC vs S_N comparison |
