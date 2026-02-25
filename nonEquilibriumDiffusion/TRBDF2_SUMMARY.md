# TR-BDF2 Implementation Summary for twoDFV.py

## What Was Added

TR-BDF2 (Trapezoidal Rule - Backward Differentiation Formula 2) time integration has been successfully added to the 2D non-equilibrium radiation diffusion solver. This provides an L-stable, second-order accurate time integration method as an alternative to the θ-method.

## New Methods

### 1. `get_f_factor_trbdf2(T_star, dt, Lambda)`
Computes the linearization factor for TR-BDF2:
```
f_TB = 1 / (1 + [(1-Λ)/(2-Λ)] · β·σ_P·c·Δt)
```
where β = 4aT_★³/C_v_★

### 2. `time_step_trbdf2(n_steps, Lambda, source, verbose)`
Main time stepping routine using TR-BDF2:
- **Stage 1**: Trapezoidal rule from t^n to t^{n+Λ}
- **Stage 2**: BDF2 from t^n and t^{n+Λ} to t^{n+1}
- **Lambda**: Default value is 2 - √2 ≈ 0.586 (optimal for L-stability)

### 3. `newton_step_bdf2(phi_n, T_n, phi_Lambda, T_Lambda, Lambda, source, verbose)`
Newton iteration for the BDF2 stage (Stage 2)
- Solves equations (8.61a) and (8.61b) sequentially
- Uses intermediate solution from Stage 1 as initial guess

### 4. `assemble_phi_equation_trbdf2(...)`
Assembles the sparse matrix system for φ in BDF2 stage
- Uses BDF2 coefficients: c_np1, c_nL, c_n
- Fully implicit spatial discretization
- Returns CSR sparse matrix and RHS vector

### 5. `solve_T_equation_trbdf2(...)`
Solves the temperature equation for BDF2 stage
- Cell-by-cell solution (no spatial coupling)
- Uses BDF2 time derivative formula

## Usage Example

```python
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Create solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=2.0, nx_cells=30,
    y_min=0.0, y_max=2.0, ny_cells=30,
    geometry='cartesian',
    dt=0.02,  # Can use larger time steps
    theta=1.0
)

# Set initial condition
def T_init(x, y):
    r2 = (x - 1.0)**2 + (y - 1.0)**2
    return 0.3 + 0.7 * np.exp(-20.0 * r2)

solver.set_initial_condition(T_init=T_init)

# Use TR-BDF2 time integration
solver.time_step_trbdf2(n_steps=100, verbose=True)

# Get solution
x, y, phi_2d, T_2d = solver.get_solution()
```

## Advantages of TR-BDF2

1. **L-Stable**: Excellent for stiff problems with vastly different time scales
2. **Second-Order Accurate**: More accurate than implicit Euler (first-order)
3. **Larger Time Steps**: Can use larger Δt while maintaining accuracy
4. **No Ringing**: Unlike Crank-Nicolson, doesn't produce oscillations

## Comparison with θ-method

| Property | θ-method (θ=1) | TR-BDF2 |
|----------|----------------|---------|
| Order of accuracy | 1st order | 2nd order |
| Stability | A-stable | L-stable |
| Cost per step | 1 Newton solve | 2 Newton solves |
| Time step size | Smaller | Larger possible |
| Best for | Simple problems | Stiff/transient problems |

## Test Results

The implementation was tested on a 15×15 grid with a hot spot initial condition:
- Both methods (implicit Euler and TR-BDF2) produced consistent results
- Maximum relative difference: ~3.5% in φ, ~0.9% in T
- TR-BDF2 converged successfully in both stages

Example run (from `test_trbdf2_2d.py`):
```
Test 1: Implicit Euler (θ=1.0)
Final φ range: [1.7060e-02, 3.9648e-01]
Final T range: [0.3013, 0.9806] keV

Test 2: TR-BDF2
Final φ range: [1.5416e-02, 3.9780e-01]
Final T range: [0.3005, 0.9829] keV

✓ TR-BDF2 and θ-method solutions are consistent
```

## Implementation Notes

### Convergence Behavior

- **Stage 1 (TR)**: May require more Newton iterations, especially for cold initial conditions transitioning to hot regions
- **Stage 2 (BDF2)**: Typically converges quickly in 2-3 iterations
- Overall, TR-BDF2 can be more efficient for problems requiring high temporal accuracy

### Damping Strategy

Both stages use adaptive damping when negative values are encountered:
- Line search to ensure φ > 0 and T > 0
- Starts with α = 0.9 and reduces if needed
- Prevents non-physical solutions during Newton iterations

### Matrix Structure

Same sparse structure as θ-method:
- 5-point stencil in 2D
- CSR format for efficient solving
- Same boundary condition handling

## Files Modified

1. **twoDFV.py**: Added 5 new methods for TR-BDF2
2. **README_twoDFV.md**: Updated documentation
3. **test_trbdf2_2d.py**: New test script (created)

## References

See the 1D code (`oneDFV.py`) for mathematical formulation and equations (8.61a, 8.61b, 8.62).

## Future Work

Potential improvements:
- Adaptive Lambda parameter based on local truncation error
- More sophisticated line search in Newton iteration
- Preconditioner for sparse solver to improve convergence
- Adaptive time stepping based on error estimates
