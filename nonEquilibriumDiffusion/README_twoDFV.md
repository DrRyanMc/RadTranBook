# 2D Non-Equilibrium Radiation Diffusion Solver

## Overview

`twoDFV.py` extends the 1D non-equilibrium radiation diffusion solver (`oneDFV.py`) to two dimensions. It solves the coupled radiation-material equations for φ(x,y,t) and T(x,y,t) (or φ(r,z,t) and T(r,z,t) in cylindrical coordinates).

## Features

### Supported Geometries

1. **Cartesian (x-y)**: Standard 2D Cartesian coordinates
   - Domain: [x_min, x_max] × [y_min, y_max]
   - Uniform or stretched grids in both dimensions

2. **Cylindrical (r-z)**: Axisymmetric cylindrical coordinates
   - Domain: [r_min, r_max] × [z_min, z_max] with r_min ≥ 0
   - Proper geometric factors accounting for cylindrical symmetry

### Numerical Methods

- **Spatial Discretization**: Finite volume method with 5-point stencil
- **Sparse Matrix Storage**: Efficient CSR format using scipy.sparse
- **Time Integration**: 
  - θ-method (implicit Euler, Crank-Nicolson, or general θ)
  - TR-BDF2 (composite method: Trapezoidal Rule + BDF2)
- **Nonlinear Solver**: Newton iteration for φ-T coupling
- **Flux Limiting**: Support for various flux limiters (Levermore-Pomraning, Larsen, etc.)

### Key Differences from 1D Code

1. **Grid Structure**: 2D structured grids with (nx, ny) cells
2. **Matrix Format**: Sparse matrices instead of tridiagonal
3. **Index Mapping**: Helper functions to convert between 2D (i,j) and 1D indices
4. **Geometry Factors**: 
   - Cartesian: Ax = Δy, Ay = Δx, V = Δx·Δy
   - Cylindrical: Ar = 2πr·Δz, Az = π(r²_outer - r²_inner), V = π(r²_outer - r²_inner)·Δz
5. **Boundary Conditions**: Four boundaries (left, right, bottom, top) instead of two

## Usage Example

### Cartesian Geometry

```python
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Create solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=2.0, nx_cells=30,
    y_min=0.0, y_max=2.0, ny_cells=30,
    geometry='cartesian',
    dt=0.01,
    theta=1.0,  # Implicit Euler
    max_newton_iter=20,
    newton_tol=1e-6
)

# Set initial condition with a hot spot
def T_init(x, y):
    r2 = (x - 1.0)**2 + (y - 1.0)**2
    return 0.3 + 0.7 * np.exp(-20.0 * r2)

solver.set_initial_condition(T_init=T_init)

# Run simulation
solver.time_step(n_steps=100, verbose=True)

# Get and plot solution
x, y, phi_2d, T_2d = solver.get_solution()
solver.plot_solution(save_path='result.png')
```

### Cylindrical Geometry

```python
# Create cylindrical solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.1, x_max=2.0, nx_cells=25,  # r-direction
    y_min=0.0, y_max=3.0, ny_cells=30,  # z-direction
    geometry='cylindrical',
    dt=0.01,
    theta=1.0
)

# Set initial condition
def T_init_cyl(r, z):
    return 0.3 + 0.7 * np.exp(-2.0 * r**2) * np.exp(-2.0 * (z - 1.5)**2)

solver.set_initial_condition(T_init=T_init_cyl)

# Run simulation
solver.time_step(n_steps=100, verbose=True)

# Plot
solver.plot_solution(save_path='cylindrical_result.png')
```

### Using TR-BDF2 Time Integration

For problems requiring larger time steps or higher accuracy, use TR-BDF2:

```python
# Create solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=2.0, nx_cells=30,
    y_min=0.0, y_max=2.0, ny_cells=30,
    geometry='cartesian',
    dt=0.02,  # Can use larger time steps with TR-BDF2
    theta=1.0
)

# Set initial condition
solver.set_initial_condition(T_init=T_init_func)

# Use TR-BDF2 instead of standard time_step
solver.time_step_trbdf2(
    n_steps=100, 
    Lambda=None,  # Default: 2 - sqrt(2) ≈ 0.586
    verbose=True
)

# Get solution
x, y, phi_2d, T_2d = solver.get_solution()
```

**TR-BDF2 Advantages:**
- L-stable (good for stiff problems)
- Second-order accurate in time
- Can use larger time steps than implicit Euler
- Better for transient problems with sharp gradients

**TR-BDF2 Notes:**
- Each TR-BDF2 step has two stages (TR and BDF2)
- May require more Newton iterations in Stage 1 (TR)
- Overall can be more efficient for problems requiring high accuracy

## Class Structure

### Main Class: `NonEquilibriumRadiationDiffusionSolver2D`

#### Initialization Parameters

- `x_min`, `x_max`, `nx_cells`: First coordinate domain and resolution (x or r)
- `y_min`, `y_max`, `ny_cells`: Second coordinate domain and resolution (y or z)
- `geometry`: 'cartesian' or 'cylindrical'
- `dt`: Time step size
- `theta`: Time discretization parameter (0.5=Crank-Nicolson, 1.0=implicit Euler)
- `max_newton_iter`: Maximum Newton iterations per time step
- `newton_tol`: Newton convergence tolerance
- `x_stretch`, `y_stretch`: Grid stretching factors (1.0 = uniform)
- `rosseland_opacity_func`, `planck_opacity_func`: Material opacity functions
- `specific_heat_func`, `material_energy_func`: Material property functions
- `flux_limiter_func`: Flux limiter function λ(R)
- `boundary_funcs`: Dictionary of boundary condition functions

#### Key Methods

- `set_initial_condition(phi_init=None, T_init=None)`: Set initial φ and T
  - Accepts callables, arrays, or scalars
  - For callables: `phi_init(x, y)` or `T_init(r, z)`

- `time_step(n_steps=1, source=None, verbose=True)`: Advance solution using θ-method
  - `source`: Optional source term (array or callable)
  - Newton iteration at each time step

- `time_step_trbdf2(n_steps=1, Lambda=None, source=None, verbose=True)`: Advance using TR-BDF2
  - `Lambda`: Intermediate time fraction (default: 2 - √2 ≈ 0.586)
  - Two-stage method: Trapezoidal Rule then BDF2
  - Better stability and accuracy for stiff problems

- `get_solution()`: Return (x, y, phi_2d, T_2d)
  - Arrays are 2D with shape (nx, ny)

- `plot_solution(figsize=(12,5), save_path=None, show=False)`: Visualize solution
  - Creates side-by-side plots of φ and T
  - Can save to file and/or display

- `get_phi_2d()`, `get_T_2d()`: Get 2D arrays of current solution
- `set_phi_2d(phi_2d)`, `set_T_2d(T_2d)`: Set solution from 2D arrays

## Grid and Geometry

### Grid Generation

The solver uses structured grids with cell-centered unknowns:
- Face positions: `x_faces[i]` for i = 0, ..., nx
- Cell centers: `x_centers[i]` for i = 0, ..., nx-1
- Similar for y/z direction

### Geometry Factors

**Cartesian:**
- Face area (x-normal): A_x[i,j] = Δy_j
- Face area (y-normal): A_y[i,j] = Δx_i
- Cell volume: V[i,j] = Δx_i · Δy_j

**Cylindrical:**
- Face area (r-normal): A_r[i,j] = 2πr_i · Δz_j
- Face area (z-normal): A_z[i,j] = π(r²_{i+1} - r²_i)
- Cell volume: V[i,j] = π(r²_{i+1} - r²_i) · Δz_j

### Index Mapping

Internal storage is 1D (length nx·ny):
- `index_2d_to_1d(i, j, nx)`: Convert (i,j) → k
- `index_1d_to_2d(k, nx)`: Convert k → (i,j)
- Ordering: k = i + j·nx (row-major, i varies fastest)

## Sparse Matrix Structure

The diffusion operator produces a sparse matrix with:
- Main diagonal: time derivative + diffusion diagonal + coupling
- Off-diagonals: spatial coupling in x and y directions
- 5-point stencil: connects each cell to (i±1,j) and (i,j±1) neighbors

Typical sparsity pattern for nx=ny=n:
- Non-zeros per row: ~5
- Total non-zeros: ~5n²
- Matrix size: n² × n²

Stored in CSR (Compressed Sparse Row) format for efficient solving.

## Boundary Conditions

Default boundary conditions (can be customized):
- **Left/Right (x or r boundaries)**: Dirichlet with specified temperatures
- **Bottom/Top (y or z boundaries)**: Neumann (zero flux)

Custom boundary conditions via `boundary_funcs` dictionary:
```python
def my_bc(phi, pos):
    """Return (A_bc, B_bc, C_bc) for: A·φ + B·(n·∇φ) = C"""
    # Example: Dirichlet with φ = phi_boundary
    return (1.0, 0.0, phi_boundary)

boundary_funcs = {
    'left': my_bc,
    'right': my_bc,
    'bottom': lambda phi, pos: (0.0, 1.0, 0.0),  # Zero flux
    'top': lambda phi, pos: (0.0, 1.0, 0.0)
}

solver = NonEquilibriumRadiationDiffusionSolver2D(
    ..., boundary_funcs=boundary_funcs
)
```

## Material Properties

All material property functions from the 1D code are supported:
- `rosseland_opacity(T)`: σ_R(T)
- `planck_opacity(T)`: σ_P(T)
- `specific_heat_cv(T)`: c_v(T)
- `material_energy_density(T)`: e(T) = ρc_v(T)T
- `inverse_material_energy_density(e)`: T from e

Custom functions can be provided during initialization.

## Flux Limiters

Available flux limiter functions:
- `flux_limiter_standard`: λ = 1/3 (no limiting)
- `flux_limiter_levermore_pomraning`: λ^LP = (2+R)/(6+3R+R²)
- `flux_limiter_larsen`: λ^L = (3^n + R^n)^(-1/n)
- `flux_limiter_sum`: λ = 1/(3+R)
- `flux_limiter_max`: λ = 1/max(3, R)

where R = |∇φ|/(σ_R·φ)

## Performance Considerations

### Memory Usage

For an nx × ny grid:
- Solution storage: ~2·nx·ny floating point numbers (φ and T)
- Sparse matrix: ~5·nx·ny non-zeros
- Total: O(nx·ny)

### Computational Cost

Per time step:
- Matrix assembly: O(nx·ny)
- Sparse solve: O(nx·ny) to O((nx·ny)^{1.5}) depending on solver
- Newton iterations: typically 2-5 per time step

### Recommended Grid Sizes

- Small problems: 20×20 to 50×50 (< 1 second per time step)
- Medium problems: 50×50 to 100×100 (1-10 seconds per time step)
- Large problems: 100×100 to 500×500 (10-300 seconds per time step)

## Verification and Testing

Run the built-in examples:
```bash
python3 twoDFV.py
```

This will:
1. Run Cartesian example with hot spot diffusion
2. Run cylindrical example with axisymmetric diffusion
3. Save plots to `cartesian_example.png` and `cylindrical_example.png`
4. Print solution statistics and convergence information

## Comparison with 1D Code

| Feature | 1D (oneDFV.py) | 2D (twoDFV.py) |
|---------|----------------|----------------|
| Geometries | Planar, cylindrical, spherical | Cartesian (x-y), cylindrical (r-z) |
| Grid | 1D array | 2D structured |
| Matrix type | Tridiagonal | Sparse (CSR) |
| Stencil | 3-point | 5-point |
| Solver | Thomas algorithm | scipy.sparse.linalg.spsolve |
| Time integration | θ-method, TR-BDF2 | θ-method, TR-BDF2 |
| Memory | O(n) | O(n²) |
| Time/step | O(n) | O(n²) to O(n³) |

## Future Extensions

Possible extensions to consider:
- Adaptive mesh refinement
- Parallel sparse solvers (PETSc, Trilinos)
- Higher-order spatial discretization
- Unstructured grids
- 3D extension (r-θ-z cylindrical or x-y-z Cartesian)
- Adaptive time stepping

## References

See the 1D code (`oneDFV.py`) and associated documentation for:
- Mathematical formulation
- Physics of non-equilibrium radiation diffusion
- Coupling equations (8.59a, 8.59b)
- Linearization strategy
