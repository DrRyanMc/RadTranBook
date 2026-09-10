# Crooked Pipe — 2-D Cylindrical Multigroup IMC

A 2-D cylindrical (r-z) radiation-transport benchmark that tests the ability
of a solver to correctly route radiation through a U-shaped (crooked) channel.
The geometry creates a strongly non-trivial streaming path that challenges
diffusion approximations.

## Physical Setup

**Domain:** r ∈ [0, 2] cm, z ∈ [0, 3.5] cm (cylindrical symmetry about r = 0).

### Geometry — U-shaped (reversed) crooked pipe

```
z
3.5 |  ################## outer wall ##################
    |  # thick ##########|<--bend-->|################## #
2.5 |  # wall  # inner   |          |  outer leg       #
    |  # 0.5≤r<1.0 leg   |          |  1.0≤r<1.5       #
    |  # (thick)  r<0.5  |          |                  #
    |  ########  (thin)  |          |################## #
0.0 |  ########  inlet   |(thick 0.5-1.0)  outlet      #
     -------+------+--------+---------+-----------+-----
           0.0   0.5       1.0       1.5         2.0  r
```

| Region          | r range (cm)  | z range (cm) | ρ (g/cm³) |
|-----------------|---------------|--------------|-----------|
| Inner leg (thin)| 0 – 0.5       | 0 – 3.5      | 0.01      |
| Thick wall      | 0.5 – 1.0     | 0 – 2.5      | 2.0       |
| Bend (thin)     | 0 – 1.5       | 2.5 – 3.5    | 0.01      |
| Outer leg (thin)| 1.0 – 1.5     | 0 – 2.5      | 0.01      |
| Outer wall      | ≥ 1.5         | all z        | 2.0       |

## Governing Equations

2-D multigroup IMC in cylindrical (r-z) geometry with photon streaming and
material energy exchange:

$$\frac{1}{c}\frac{\partial E_g}{\partial t}
  + \nabla \cdot \mathbf{F}_g
  = c\,\sigma_{a,g}(\rho,T)\left(acT^4 b_g - E_g\right)$$

$$\rho\,c_v\frac{\partial T}{\partial t}
  = c\sum_g \sigma_{a,g}(E_g - acT^4 b_g)$$

## Opacity Model

Power-law, matching the diffusion reference case:

$$\sigma_{a,g}(T) = 10\,\rho(r,z)\,T^{-1/2}\,\bar{\nu}_g^{-3} \qquad [\text{cm}^{-1}]$$

Group opacity uses the geometric mean of boundary values:

$$\sigma_{a,g}(T) = \sqrt{\sigma_a(T,\nu_{g-1/2})\,\sigma_a(T,\nu_{g+1/2})}$$

Hard cap: σ_max = 10¹⁴ cm⁻¹.

## Equation of State

$$u = \rho\,c_v\,T, \qquad c_v = C_{V,\rm mass}\,\rho, \qquad C_{V,\rm mass} = 0.05\ \text{GJ/(g·keV)}$$

## Initial Conditions

$$T(r,z,0) = T_r(r,z,0) = T_{\rm init} = 0.05\ \text{keV everywhere}$$

## Boundary Conditions

| Boundary                       | Condition                                        |
|--------------------------------|--------------------------------------------------|
| z = 0, r < 0.5 (inlet)        | Isotropic blackbody source, T_bc(t) (see below)  |
| z = 0, r ≥ 0.5                 | Reflecting                                       |
| z = 3.5 (top)                  | Reflecting                                       |
| r = 0 (axis)                   | Reflecting (cylindrical symmetry)                |
| r = 2.0 (outer)                | Reflecting                                       |
| z = 0, 1.0 ≤ r < 1.5 (outlet) | Vacuum (no incoming photons)                     |

**Left inlet source** (time-dependent ramp):

$$T_{\rm bc}(t) = \begin{cases}
  T_{\rm start} + (T_{\rm end} - T_{\rm start})\,t/t_{\rm ramp} & t < t_{\rm ramp}\\
  T_{\rm end} & t \geq t_{\rm ramp}
\end{cases}$$

Default: T_start = 0.05 keV, T_end = 0.5 keV, t_ramp = 5 ns.

## Fiducial Monitor Points

Five spatial probes for temperature history (T_mat(t), T_rad(t)):

| Point | r (cm) | z (cm) | Location                     |
|-------|--------|--------|------------------------------|
| P1    | 0.0    | 0.25   | inlet, inner leg             |
| P2    | 0.0    | 2.25   | inner leg near bend          |
| P3    | 0.75   | 3.05   | inside bend / thick wall     |
| P4    | 1.25   | 2.25   | outer leg near bend          |
| P5    | 1.25   | 0.25   | outlet, outer leg            |

## Energy Groups

| Parameter       | Default (refined run) |
|-----------------|-----------------------|
| Number of groups| 10 (also 2, 4, 50)    |
| ν_min           | (problem dependent)   |
| Spacing         | Logarithmic           |

## Expected Behavior

Radiation entering the inner leg (r < 0.5) must turn through the bend at
z = 2.5 cm to reach the outlet at z = 0, r ≈ 1.25 cm.  The thick wall
(ρ = 2 g/cm³) strongly attenuates diffusive leakage.  Transport methods
show a clear wave front; diffusion models tend to smear it.  The fiducial
points track the radiation "ringing" as the wave traverses the pipe.

## Files

| File                                  | Purpose                          |
|---------------------------------------|----------------------------------|
| `crooked_pipe_multigroup_imc.py`      | IMC runner (2-D MG-IMC)          |
| `../visualization/plot_crooked_pipe_mg_imc_fiducial.py` | Fiducial plots |

## References

- Morel, J. E., et al. (1996). A cell-centered Lagrangian-mesh diffusion
  differencing scheme.  *Journal of Computational Physics*, 103, 286–299.
- Densmore, J. D., et al. (2012). Multigroup discrete-diffusion Monte Carlo.
