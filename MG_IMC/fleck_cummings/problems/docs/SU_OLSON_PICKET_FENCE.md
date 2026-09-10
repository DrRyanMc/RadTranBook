# Su-Olson Picket Fence Benchmark (Case A)

A two-group multigroup benchmark derived from the Su & Olson (1997) picket-
fence problem.  One group is optically thin and the other optically thick,
creating a severe test for multigroup methods and their handling of different
optical depths simultaneously.

## Physical Setup

1-D slab geometry:

```
reflecting BC   |  source region Q  |  pure transport  |  vacuum BC
x = 0           x = 0.5 cm                              x = 20 cm
```

| Parameter    | Value                               |
|--------------|-------------------------------------|
| Domain       | 0 – 20 cm                           |
| Source width | 0 – 0.5 cm (Q_g = 0.5 per group)   |
| ρ            | 1 g/cm³                             |
| BC at x = 0  | Reflecting                          |
| BC at x = 20 | Vacuum                              |

## Governing Equations

Two-group non-equilibrium radiation-material system with internal source:

$$\frac{1}{c}\frac{\partial E_g}{\partial t} + \frac{\partial F_g}{\partial x}
  = \sigma_g\left(\frac{acT^4}{2} - E_g\right) + Q_g$$

$$\frac{\partial u}{\partial t}
  = c\sum_g \sigma_g\left(E_g - \frac{acT^4}{2}\right)$$

where equal emission fractions b_g = 0.5 are assumed for both groups.

## Opacity Model

**Constant (temperature-independent) opacities:**

| Group | Energy range | σ_g (cm⁻¹)  | Physical character |
|-------|--------------|-------------|--------------------|
| 0     | "thin"       | 2/11 ≈ 0.182 | optically thin    |
| 1     | "thick"      | 20/11 ≈ 1.818 | optically thick  |

Total Planck mean: σ_P = 0.5 × 2/11 + 0.5 × 20/11 = 1.0 cm⁻¹.

## Equation of State

Radiation-dominated EOS (Stefan-Boltzmann):

$$u = a\,T^4, \qquad c_v = 4aT^3$$

## Initial Conditions

$$T(x,0) = 0, \qquad E_g(x,0) = 0$$

## Boundary Conditions

| Boundary | Condition                    |
|----------|------------------------------|
| x = 0    | Reflecting (zero net flux)   |
| x = 20   | Vacuum (no incoming photons) |

## Source

Isotropic internal source in 0 ≤ x ≤ 0.5 cm:

$$Q_g = 0.5\ \text{for}\ g = 0,1$$

## Dimensionless Variables (Su-Olson rescaling)

Following Su & Olson (1997), define:

$$\tau = c\,\sigma_P\,t, \qquad \xi = \sigma_P\,x$$

Reference data from **Table 2** of Su & Olson (1997) gives the dimensionless
radiation energy $U_1(\xi, \tau)$ for the thin group at τ = 0.1, 0.3, 1, 3.

## Expected Behavior

The thin group (g = 0) penetrates far ahead of the thick group.  At early
times the two groups have very different spatial distributions; they
equilibrate only over long times.  Gray methods using the Planck-mean opacity
cannot capture the spectral separation.

## Files

| File                        | Purpose           |
|-----------------------------|-------------------|
| `test_su_olson_picket_fence.py` | IMC runner and reference-data comparison |

## References

- Su, B., & Olson, G. L. (1997). Non-equilibrium diffusion problems
  with a piecewise-constant Planck function.  *Journal of Quantitative
  Spectroscopy and Radiative Transfer*, 57(3), 329–345.
