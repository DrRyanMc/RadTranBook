# M1 1-D Solver (Su-Olson)

This folder contains a first-order finite-volume solver for the 1-D M1 moment equations with a fully implicit nonlinear backward-Euler step and non-equilibrium material coupling.

## Files

- `m1_1d.py`: solver and closure definitions
- `su_olson_m1_closure_comparison.py`: Su-Olson driver that compares closures and writes plots

## Implemented Closures

- P1 (`chi = 1/3`)
- Kershaw
- Levermore
- Minerbo polynomial approximation
- Minerbo rational approximation

## Run

From repo root:

```bash
python M1/su_olson_m1_closure_comparison.py
```

## Geometry option

Set `d` in `M1SuOlsonSolver1D`:

- `d=0`: slab
- `d=1`: cylindrical
- `d=2`: spherical

The momentum equation includes the geometric term
`c^2 (d/s) ((3 chi - 1)/2) E_r` when `d > 0`, and flux divergence is evaluated in conservative area form with `A(s)=s^d`.

## Boundary options

Set `right_bc_mode` in `M1SuOlsonSolver1D`:

- `copy`: simple outflow copy state
- `marshak`: P1/Marshak-like vacuum relation `F = (c/2) E_r` at right boundary
- `free_stream`: outgoing free-stream `F = c E_r`

For Su-Olson slab comparisons, `marshak` is typically the most appropriate choice.

Outputs:

- `su_olson_m1_closures_early_times.pdf`
- `su_olson_m1_closures_late_times.pdf`

## Notes

- Spatial discretization: first-order finite volume in `A(s)`-conservative form.
- Time discretization: fully implicit backward Euler with nonlinear Picard iteration.
- Numerical flux: Rusanov with `lambda = c`.
- Left boundary: reflecting (flux sign flip in ghost cell).
- Right boundary: selectable (`copy`, `marshak`, `free_stream`).
- Local source coupling (`E_r` and `aT^4`) is solved implicitly via a 2x2 BE system each nonlinear iteration.
