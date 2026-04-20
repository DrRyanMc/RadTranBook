# MG_IMC: Multigroup Implicit Monte Carlo in 2D

This directory contains a multigroup extension of the IMC2D.py code for solving multigroup radiation transport problems using the Implicit Monte Carlo method.

## Overview

**MG_IMC2D.py** extends IMC2D.py to support multigroup radiation transport. Each Monte Carlo particle carries:
- A **group index** (g = 0, 1, ..., G-1)
- A **frequency** within that group (sampled from appropriate distributions)
- All standard particle attributes (position, direction, weight, time)

## Key Features

### 1. Multigroup Transport

The code implements the multigroup IMC equations with the Fleck factor:

```
f = 1 / (1 + β c σ_P Δt)
```

where σ_P is the Planck-weighted opacity summed over all groups.

The transport equation for group g is:

```
(1/c ∂/∂t + Ω·∇ + σ_g) I_g = (f σ_g b_g★ c a T_n^4) / 4π 
                                + ((1-f) σ_g b_g★ / σ_P) Σ_g' σ_g' φ_g' / 4π + Q_g / 4π
```

### 2. Energy Group Sampling

The code implements two different sampling strategies depending on the particle source:

#### A. **Mixture of Gammas** (Initial Conditions & Boundary Sources)

For particles emitted from boundaries or initial equilibrium conditions, frequencies are sampled from the Planck spectrum using a mixture of Gamma distributions (equations 10.23-10.27):

```
p(x) = (15/π⁴) x³ e^(-x)  where x = ν/T
     = (90/π⁴) Σ_n (1/n⁴) Gamma(k=4, r=n)
```

**Algorithm:**
1. Sample integer n with probability ∝ 1/n⁴
2. Sample x from Gamma(k=4, r=n): x = -(1/n) log(r₁r₂r₃r₄)
3. Convert to frequency: ν = T·x
4. Determine group g containing ν

This is implemented in `_sample_planck_spectrum_mixture_of_gammas()`.

#### B. **Piecewise Constant** (Material Emission & Effective Scatter)

For particles emitted from material or undergoing effective scattering, the group is sampled from a discrete distribution (equation 10.18):

```
P_g = (σ_a,g b_g★) / σ_P
```

where b_g★ = B_g(T★) is the Planck function integral over group g at the linearization temperature T★.

This is implemented in `_sample_group_piecewise_constant()`.

### 3. Group-Dependent Opacities

Each energy group has its own absorption opacity function:
- `sigma_a_funcs`: List of callables, one per group
- Each function takes temperature array and returns opacity array
- Allows for frequency-dependent opacity models

### 4. Planck Function Integrals

The code uses the external `planck_integrals` library to compute group-integrated Planck functions:
- `Bg(E_low, E_high, T)`: Integrated Planck function for group g
- `dBgdT(E_low, E_high, T)`: Temperature derivative

If the library is unavailable, gray approximations are used.

### 5. Geometry Support

Like IMC2D.py, supports both:
- **Cartesian (xy)**: Rectangular 2D slab
- **Cylindrical (rz)**: Axisymmetric cylindrical geometry

### 6. JIT Compilation

The transport kernel `_transport_particles_2d_mg()` is JIT-compiled with Numba for performance, with parallel execution over particles.

## File Structure

```
MG_IMC/
├── MG_IMC2D.py              # Main multigroup IMC module
├── test_2group_marshak.py   # Example: 2-group Marshak wave
└── README.md                # This file
```

## Usage

### Basic Example

```python
from MG_IMC2D import run_simulation
import numpy as np

# Define energy groups (keV)
energy_edges = np.array([0.1, 1.0, 10.0])  # 2 groups
n_groups = len(energy_edges) - 1

# Define opacity functions for each group
def sigma_a_group_0(T):
    return 1.0 * T**(-3)  # Group 0 opacity

def sigma_a_group_1(T):
    return 0.5 * T**(-3)  # Group 1 opacity

sigma_a_funcs = [sigma_a_group_0, sigma_a_group_1]

# Spatial mesh
x_edges = np.linspace(0, 5, 51)
y_edges = np.array([0, 1])  # 1D slab

# Initial conditions
Tinit = np.full((50, 1), 0.01)  # keV
Tr_init = Tinit.copy()

# Material properties
def eos(T):
    return 0.1 * T  # GJ/cm³

def inv_eos(e):
    return e / 0.1

def cv(T):
    return 0.1 * np.ones_like(T)

# Run simulation
history, state = run_simulation(
    Ntarget=10000,
    Nboundary=5000,
    Nsource=0,
    Nmax=50000,
    Tinit=Tinit,
    Tr_init=Tr_init,
    T_boundary=(1.0, 0.0, 0.0, 0.0),
    dt=0.1,
    edges1=x_edges,
    edges2=y_edges,
    energy_edges=energy_edges,
    sigma_a_funcs=sigma_a_funcs,
    eos=eos,
    inv_eos=inv_eos,
    cv=cv,
    source=0.0,
    final_time=1.0,
    geometry="xy",
)
```

## API Reference

### Main Functions

#### `init_simulation()`
Initialize multigroup simulation state.

**New Parameters:**
- `energy_edges`: Array of energy group boundaries (keV)

**Returns:**
- `SimulationState2DMG`: State object with multigroup data

#### `step()`
Advance simulation by one time step.

**New Parameters:**
- `energy_edges`: Energy group boundaries
- `sigma_a_funcs`: List of opacity functions, one per group

**Returns:**
- `state`: Updated state
- `info`: Dictionary with step information including `radiation_energy_by_group`

#### `run_simulation()`
Run complete simulation.

**New Parameters:**
- `energy_edges`: Energy group boundaries
- `sigma_a_funcs`: List of opacity functions

**Returns:**
- `history`: List of info dictionaries
- `state`: Final state

### Data Structures

#### `SimulationState2DMG`
Extends `SimulationState2D` with multigroup data:

**New Fields:**
- `groups`: Array of group indices for each particle
- `radiation_energy_by_group`: Array of shape `(n_groups, nx, ny)` containing radiation energy density by group

### Helper Functions

#### `_sample_planck_spectrum_mixture_of_gammas(n, T, energy_edges)`
Sample frequencies from Planck spectrum using mixture of Gammas.

#### `_sample_group_piecewise_constant(n, probabilities)`
Sample group indices from discrete distribution.

## Physical Units

Consistent with IMC2D.py:
- **Distance**: cm
- **Time**: ns
- **Temperature**: keV
- **Energy**: GJ
- **Frequency/Energy**: keV (photon energy)

## Notes and Limitations

1. **Planck Integrals**: Requires `planck_integrals` library for accurate group-integrated Planck functions. Falls back to gray approximations if unavailable.

2. **Group Sampling**: The mixture-of-Gammas sampling is exact for Planck spectrum but requires careful implementation. The code truncates the infinite series at n=100.

3. **Opacity Models**: Users must provide appropriately normalized opacity functions for each group. Total opacity should integrate correctly over the spectrum.

4. **Performance**: Multigroup calculations are more expensive than gray. Use JIT compilation (Numba) for production runs.

5. **Boundary Sources**: Currently supports uniform temperature boundaries. Position-dependent boundary sources can be added via `boundary_source_func`.

## References

The implementation follows the multigroup IMC formulation in:
- Section 10.5: "Multigroup IMC" in the textbook
- Equations 10.10-10.14: Multigroup transport with Fleck factor
- Equations 10.18: Group sampling for emission/scatter
- Equations 10.23-10.27: Planck spectrum sampling via mixture of Gammas

## Testing

Run the example test:

```bash
cd MG_IMC
python test_2group_marshak.py
```

This will run a 2-group Marshak wave problem and generate diagnostic plots.

## Extending the Code

To add new features:

1. **More energy groups**: Simply provide longer `energy_edges` array
2. **Multigroup sources**: Provide 3D source array of shape `(n_groups, nx, ny)`
3. **Group-dependent scatter**: Modify transport kernel to handle group-to-group scattering
4. **Custom sampling**: Replace sampling functions with problem-specific distributions

## Comparison with IMC2D.py

**Similarities:**
- Same geometry support (xy, rz)
- Same particle transport mechanics
- Same boundary conditions and reflections
- Same material coupling and Fleck factor approach

**Differences:**
- Particles carry group index
- Multiple opacity functions (one per group)
- Group-specific energy deposition and tracking
- Planck spectrum sampling for IC/boundaries
- Piecewise constant sampling for emission/scatter
- Radiation energy tracked by group

## Future Enhancements

Possible extensions:
- [ ] Group-to-group scattering matrix
- [ ] Frequency-dependent boundary sources
- [ ] Adaptive energy group structure
- [ ] Variance reduction techniques for rare groups
- [ ] Parallel MPI support for large multigroup problems
- [ ] Checkpoint/restart capability with group data

## Contact

For questions or issues, refer to the main RadTranBook repository.
