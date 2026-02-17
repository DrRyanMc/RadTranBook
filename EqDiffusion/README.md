# Equilibrium Radiation Diffusion Solver

A finite volume solver for the equilibrium radiation diffusion equation with nonlinear material properties. This code is designed for radiation hydrodynamics applications where radiation and matter are in local thermodynamic equilibrium (LTE).

## Overview

This solver implements numerical methods for the time-dependent, nonlinear diffusion equation:

```
∂E_r/∂t = ∇·(D(E_r) ∇E_r) + source terms
```

where:
- `E_r` is the radiation energy density (GJ/cm³)
- `D(E_r) = c/(3σ_R(T))` is the nonlinear diffusion coefficient
- `σ_R(T)` is the Rosseland mean opacity (cm⁻¹)
- `c` is the speed of light
- Temperature `T = (E_r/a)^(1/4)` via the Stefan-Boltzmann law

### Physical Constants

- Speed of light: `c = 2.998 × 10¹⁰ cm/s = 2.998 × 10¹ cm/ns`
- Radiation constant: `a = 0.01372 GJ/(cm³·keV⁴)`
- Stefan-Boltzmann constant: `σ_SB = ac/4`

## Code Structure

```
EqDiffusion/
├── utils/              # Core solver modules
│   ├── oneDFV.py       # 1D finite volume solver
│   ├── twoDFV.py       # 2D finite volume solver (Cartesian/cylindrical)
│   └── plotfuncs.py    # Plotting utilities
├── problems/           # Problem setup scripts
│   ├── marshak_wave.py           # Classic Marshak wave test
│   ├── zeldovich_wave.py         # Zeldovich self-similar wave
│   ├── crooked_pipe_test.py      # 2D complex geometry test
│   ├── refined_zoning.py         # Mesh refinement study
│   └── ...
├── tests/              # Unit and verification tests
├── analysis/           # Efficiency studies and analysis scripts
├── verification/       # Verification against analytical solutions
├── debug/              # Debugging and diagnostic scripts
├── plots/              # Generated figures
├── data/               # Solution data files (.npz)
└── refined_solutions/  # High-resolution reference solutions
```

## Numerical Methods

### Spatial Discretization
- **Finite volume method** with cell-centered unknowns
- **1D**: Planar, cylindrical (r), or spherical coordinates
- **2D**: Cartesian (x-y) or cylindrical (r-z) coordinates
- Harmonic averaging of diffusion coefficients at cell faces
- Conservative flux formulation

### Time Integration
- **Implicit Euler**: First-order accurate, unconditionally stable
- **TR-BDF2**: Second-order accurate L-stable method
  - Stage 1: Trapezoidal rule to midpoint
  - Stage 2: BDF2 to next time level
- Adaptive time stepping based on iteration convergence

### Nonlinear Solver
- **Newton-Raphson method** with analytic Jacobian
- **JFNK (Jacobian-Free Newton-Krylov)** for large 2D problems
- Sparse linear system solvers (direct or iterative GMRES)
- Nonlinear correction terms for improved convergence

## Benchmark Problems

### 1. Marshak Wave (`marshak_wave.py`)
Classic radiative heat wave propagating into cold material.

**Setup:**
- Domain: 0 to 10 cm (1D planar)
- Left boundary: Incoming flux from blackbody at T = 1 keV
- Right boundary: Zero radiation
- Material: σ_R = 300 T⁻³ cm⁻¹, c_v = 0.3 GJ/(cm³·keV)
- Initial condition: Cold material (T ~ 0.01 keV)

**Physics:** Tests nonlinear diffusion with strong temperature dependence and moving thermal fronts.

### 2. Zeldovich Wave (`zeldovich_wave.py`)
Self-similar radiative wave with analytical solution.

**Setup:**
- Domain: 0 to 10 cm (1D planar, cylindrical, or spherical)
- Initial condition: Concentrated energy pulse in first cell
- Boundary: Reflecting (zero flux) at boundaries
- Material: Same as Marshak wave

**Physics:** Self-similar wave propagation with known analytical solutions for verification.

### 3. Crooked Pipe (`crooked_pipe_test.py`)
Complex 2D geometry with spatially-varying material properties.

**Setup:**
- Domain: r ∈ [0, 2] cm, z ∈ [0, 7] cm (cylindrical)
- Geometry: "Crooked pipe" with thick/thin regions
- Thick regions: σ_R = 200 cm⁻¹, c_v = 0.5 GJ/(cm³·keV)
- Thin regions: σ_R = 0.2 cm⁻¹, c_v = 0.0005 GJ/(cm³·keV)
- Source: T = 0.3 keV at bottom center

**Physics:** Tests handling of material discontinuities, complex geometries, and 2D transport.

### 4. Refined Zoning (`refined_zoning.py`)
Mesh refinement convergence study.

**Setup:** Similar to crooked pipe with progressively refined meshes to verify spatial convergence.

### 5. Linear Gaussian Problems
Linearized test problems with Gaussian analytical solutions for code verification.

## Usage

### Running a Problem

```bash
cd problems/
python marshak_wave.py
```

Output:
- Figures saved to `../plots/`
- Solution data saved to `../data/` as `.npz` files

### Basic Solver Usage (1D)

```python
from oneDFV import RadiationDiffusionSolver

# Define material properties
def opacity(Er):
    T = (Er / A_RAD) ** 0.25
    return 300.0 * T**(-3)

def specific_heat(T):
    return 0.3

# Create solver
solver = RadiationDiffusionSolver(
    x_edges=np.linspace(0, 10, 101),
    opacity_func=opacity,
    cv_func=specific_heat,
    geometry='planar'  # or 'cylindrical', 'spherical'
)

# Set initial condition and boundary conditions
solver.set_initial_condition(Er_init)
solver.set_left_bc('dirichlet', value=A_RAD * T_left**4)
solver.set_right_bc('neumann', value=0.0)

# Time integration
t_final = 50.0  # ns
dt = 0.01
solver.solve_to_time(t_final, dt, method='implicit_euler')
```

### Basic Solver Usage (2D)

```python
from twoDFV import RadiationDiffusionSolver2D

# Define spatially-varying properties
def opacity(Er, x, y):
    # Can depend on position
    return sigma_R

def specific_heat(T, x, y):
    return cv

# Create solver
solver = RadiationDiffusionSolver2D(
    x_edges=x_edges,
    y_edges=y_edges,
    opacity_func=opacity,
    cv_func=specific_heat,
    coordinate_system='cartesian'  # or 'cylindrical'
)

# Set boundary conditions and solve
# ... (similar to 1D)
```

## Tests and Verification

### Unit Tests (`tests/`)
- `test_linear_gaussian.py` - Linearized diffusion with analytical solution
- `test_marshak_wave_speed.py` - Verify wave speed against theory
- `test_energy_conservation.py` - Energy conservation checks
- `test_theta_method.py` - Time integration scheme verification
- And many more...

Run tests:
```bash
cd tests/
python test_linear_gaussian.py
```

### Verification Scripts (`verification/`)
- Comparison against analytical solutions
- Convergence studies
- Method-of-manufactured-solutions verification

## Efficiency Studies (`analysis/`)

Performance analysis comparing different time integration methods:
- `marshak_efficiency_study.py` - Compare IE, CN, TR-BDF2
- `linear_gaussian_efficiency_study.py` - Efficiency for linear problems
- Cost vs. accuracy trade-offs
- Optimal time step selection

## Features

### Solver Capabilities
- ✅ Multiple coordinate systems (1D: planar/cylindrical/spherical, 2D: Cartesian/cylindrical)
- ✅ Nonlinear material properties (temperature-dependent opacity, specific heat)
- ✅ Spatially-varying material properties
- ✅ Multiple time integration schemes (IE, TR-BDF2)
- ✅ Adaptive time stepping
- ✅ Direct and iterative linear solvers
- ✅ JFNK for large systems
- ✅ Various boundary conditions (Dirichlet, Neumann, Robin, time-dependent)
- ✅ Energy conservation monitoring
- ✅ Diagnostic output and visualization

### Material Models
- Temperature-dependent opacity: σ_R(T) (power-law, tables, etc.)
- Temperature-dependent specific heat: c_v(T)
- Spatially-varying properties
- Interface conditions and material discontinuities

## Dependencies

- Python 3.7+
- NumPy
- SciPy (sparse linear algebra)
- Matplotlib (visualization)
- Numba (JIT compilation for performance)

## References

This code implements methods for radiation diffusion relevant to:
1. Inertial Confinement Fusion (ICF)
2. Astrophysical radiative transfer
3. High-energy-density physics

### Key Papers
- Marshak wave: Marshak, R.E. (1958). "Effect of Radiation on Shock Wave Behavior"
- Zeldovich wave: Zel'dovich, Ya.B. & Raizer, Yu.P. (1966). "Physics of Shock Waves"
- Finite volume methods for nonlinear diffusion

## Author

Ryan McClarren (and collaborators)

## License

[Specify license here]

---

*This code is part of research for a radiation transport textbook.*
