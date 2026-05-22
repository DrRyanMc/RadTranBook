# Dilute Spectrum Shell

A 1-D spherical benchmark that produces a non-Planckian (diluted, spectrally
hardened) radiation field inside and beyond a dense shell.  The problem tests
multigroup methods in the free-streaming limit and their ability to represent
a spectrum that departs significantly from a blackbody.

## Physical Setup

```
|<-- cavity -->|<-shell->|<-- outer buffer -->|
R_S=1          R_1=25    R_2=27              R_OUT=30   (cm)
```

| Region              | Radii (cm)    | ρ (g/cm³) |
|---------------------|---------------|-----------|
| Source surface      | r = R_S = 1   | —         |
| Cavity (near-vacuum)| 1 – 25        | 1×10⁻⁸   |
| Dense shell         | 25 – 27       | 2.0       |
| Outer buffer        | 27 – 30       | 1×10⁻⁸   |

## Governing Equations

Multigroup radiation transport (IMC/diffusion) coupled to material energy:

$$\frac{1}{c}\frac{\partial E_g}{\partial t} + \nabla \cdot \mathbf{F}_g
  = c\,\sigma_{a,g}(aT^4 b_g - E_g)$$

$$\rho\,c_v\frac{\partial T}{\partial t}
  = c \sum_g \sigma_{a,g}(E_g - aT^4 b_g)$$

where $b_g = B_g(T) / (acT^4/4\pi)$ is the Planck emission fraction.

## Opacity Model

Power-law in temperature and photon energy:

$$\sigma_{a,g}(T) = \rho \cdot C \cdot T^{A} \cdot \bar{\nu}_g^{B}$$

| Parameter  | Value         | Description                        |
|------------|---------------|------------------------------------|
| C          | 10 cm²/g      | opacity coefficient                |
| A          | −0.5          | temperature exponent               |
| B          | −3.0          | photon-energy exponent             |
| σ_max      | 10¹² cm⁻¹   | hard upper cap (prevents tiny mfp) |
| T_floor    | 0.01 keV      | minimum T in opacity evaluation    |

Group opacity uses the geometric mean of boundary values:

$$\sigma_{a,g}(T) = \sqrt{\sigma_a(T,\nu_{g-1/2})\,\sigma_a(T,\nu_{g+1/2})}$$

## Equation of State

Linear in temperature:

$$u = \rho\,c_v\,T, \qquad c_v = 0.05\ \text{GJ/(g·keV)}$$

## Energy Groups

| Parameter        | Value        |
|------------------|--------------|
| ν_min            | 1×10⁻⁴ keV  |
| ν_max            | 20 keV       |
| N_groups default | 32           |
| Spacing          | Logarithmic  |

## Initial Conditions

| Quantity           | Value    |
|--------------------|----------|
| T_mat everywhere   | 0.02 keV |
| T_rad everywhere   | 0.02 keV |

## Boundary Conditions

| Boundary                  | Condition                                  |
|---------------------------|--------------------------------------------|
| Inner (r = R_S = 1 cm)    | Isotropic blackbody source, T_S = 1.0 keV |
| Outer (r = R_OUT = 30 cm) | Vacuum (no incoming radiation)             |

## Diagnostics

**Free-streaming reference** (geometric dilution from source):

$$E_r(r) \approx \frac{ac}{4}\,T_S^4\,\left(\frac{R_S}{r}\right)^2, \qquad
T_r(r) \approx T_S\,\sqrt{\frac{R_S}{2r}}$$

Light travel time from source to shell inner face:
$$t_{\rm travel} = (R_1 - R_S)/c \approx 0.80\ \text{ns}$$

Free-streaming radiation temperature at r = R_1:
$$T_r(R_1) \approx T_S\sqrt{R_S/(2R_1)} \approx 0.14\ \text{keV}$$

The shell imposes a frequency-dependent optical depth that selectively
absorbs low-energy photons and re-emits at the local shell temperature,
producing a hardened (non-Planckian) spectrum in the cavity.

## Simulation Parameters

| Parameter       | Value       |
|-----------------|-------------|
| T_final         | 4.0 ns      |
| dt (nominal)    | 0.01 ns     |
| Dump times (ns) | 0.25, 0.50, 0.75, 0.90, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00 |

## Files

| File                               | Purpose                              |
|------------------------------------|--------------------------------------|
| `dilute_spectrum_shell.py`         | Shared parameters and mesh/EOS setup |
| `run_dilute_spectrum_shell.py`     | MG-IMC run script                    |
| `run_dilute_spectrum_shell_comparison.py` | Diffusion + gray IMC comparison |
| `../visualization/plot_dilute_spectrum_shell.py` | Figure generation      |

