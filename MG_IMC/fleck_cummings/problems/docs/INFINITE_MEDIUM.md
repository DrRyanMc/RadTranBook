# Infinite-Medium Equilibration

A 0-D (single-cell with all-reflecting boundaries) benchmark that tests the
multigroup IMC material-radiation equilibration.  The exact solution is
thermodynamic equilibrium at a unique temperature determined by total energy
conservation; the ODE system has an analytic solution via `scipy.integrate`.

## Physical Setup

A single spatial cell with all boundaries reflecting — equivalent to an
infinite, homogeneous medium.

| Parameter        | Value           |
|------------------|-----------------|
| Cell size        | 1 cell (0-D)    |
| All boundaries   | Reflecting      |
| ρ                | 1 g/cm³         |

## Governing Equations (0-D limit)

The radiation-material system reduces to two coupled ODEs:

$$\frac{d E_g}{d t} = c\,\sigma_{a,g}(T)\left(a c T^4\,b_g(T) - E_g\right)$$

$$\frac{d u}{d t} = -\sum_g \frac{d E_g}{d t}$$

where $u = \rho\,c_v\,T$ and $b_g(T)$ are the Planck-weighted emission fractions:

$$b_g(T) = \frac{B_g(T)}{\sum_{g'} B_{g'}(T)}, \qquad
  B_g(T) = \int_{\nu_{g-1}}^{\nu_g} B(\nu, T)\,d\nu$$

The fractions $b_g(T)$ are temperature-dependent and computed from the actual
Planck integral over each group's frequency band.

## Opacity Model

The infinite-medium test uses either:

1. **Constant (gray-equivalent):** σ_g = σ_0 = 1 cm⁻¹ for all groups.  
   In this case all groups equilibrate at the same rate and the multigroup
   result must match the gray IMC result within MC noise.

2. **Exponential-band (frequency-selective):**  The underlying monochromatic
   opacity is

$$\sigma'(\nu, T) = \sigma_0\,\frac{1 - e^{-\nu/T}}{\nu^3\,\sqrt{T}}$$

   `compare_infinite_medium_imc_vs_scipy.py` uses this form directly on a fine
   continuous energy grid.  `test_infinite_medium_multigroup_expband.py` uses
   the Planck-mean over each group band.  The Planck-mean is the exact
   opacity-weighted average:

$$\sigma_g(T) = \frac{\displaystyle\int_{\nu_{\rm low}}^{\nu_{\rm high}}
               \sigma'(\nu,T)\,B(\nu,T)\,d\nu}
               {\displaystyle\int_{\nu_{\rm low}}^{\nu_{\rm high}} B(\nu,T)\,d\nu}$$

   The numerator simplifies exactly because $B(\nu,T)\propto\nu^3/(e^{\nu/T}-1)$
   and the product with $\sigma'$ telescopes via the identity
   $(1-e^{-\nu/T})\,/\,(e^{\nu/T}-1)=e^{-\nu/T}$, giving:

$$\sigma_g(T) = \frac{\sigma_0\,(e^{-\nu_{\rm low}/T} - e^{-\nu_{\rm high}/T})}
               {\sqrt{T}\,N(\nu_{\rm low},\nu_{\rm high},T)}$$

   where $N(\nu_1,\nu_2,T) \propto \int_{\nu_1}^{\nu_2} B(\nu,T)\,d\nu$
   is evaluated via polylogarithm (Li₂, Li₃, Li₄) sums.  This is an
   **exact** closed form — no quadrature approximation is involved.  The
   opacity is strongly peaked in a particular frequency band and tests the
   group-resolved equilibration.

## Initial Conditions

| Quantity              | Value                                      |
|-----------------------|--------------------------------------------|
| T_mat                 | 0.4 keV (or problem-specific)              |
| T_rad (initial)       | 1.0 keV (set per group from Planck spectrum) |
| E_g (initial)         | Set to Planck spectrum at T_rad = 0.5 keV  |

## Boundary Conditions

All boundaries reflecting → no energy loss.

## Expected Behavior

Total energy (material + radiation) is exactly conserved.  The system
equilibrates toward a unique final temperature $T_\infty$ satisfying:

$$\rho\,c_v\,T_\infty + a\,T_\infty^4 = \rho\,c_v\,T_{\rm mat,0} + a\,T_{\rm rad,0}^4$$

The exact trajectory is given by the coupled ODE system integrated with
`scipy.solve_ivp` at high accuracy, which serves as the reference solution.

## Files

| File                                       | Purpose                              |
|--------------------------------------------|--------------------------------------|
| `test_infinite_medium_multigroup_expband.py` | MG-IMC run (exponential-band opacity) |
| `test_0D_energy_conservation.py`           | Energy-conservation unit test        |
| `test_0D_multigroup_compare.py`            | Gray vs multigroup comparison        |
| `compare_infinite_medium_imc_vs_scipy.py`  | IMC vs scipy reference overlay       |
