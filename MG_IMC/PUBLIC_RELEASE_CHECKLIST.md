# Public Release Checklist

## Private Material

- Store unpublished derivations, LaTeX manuscript sources, manuscript tooling,
  one-off diagnostics, and generated outputs in `MG_IMC/private/`. This
  directory is ignored by Git.
- Public MG_IMC contains no LaTeX source, bibliography, or LaTeX build files.
- Do not add derivations or manuscript-oriented explanations to public problem
  drivers; retain those materials in the private archive.

## Public Layout

```text
MG_IMC/
  fleck_cummings/
    src/
    problems/
    tests/
  shared/
    visualization/
  private/                 # ignored; unpublished material only
```

Import the solver through `MG_IMC.fleck_cummings.src.MG_IMC2D` or the public
`MG_IMC` package. Problem and test drivers can be run directly or as modules.

## Dependencies

- Core solvers require `numpy`.
- Plotting drivers require `matplotlib`.
- `numba` and `planck_integrals` are optional; the solver provides fallback
  implementations when unavailable.

## Release-Blocking Markers

Run this command before publishing:

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' MG_IMC \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

No such markers remain in the public source tree.

## Exclude From Public Release

Review files named `_*.py`, `benchmark_*.py`, `debug_*.py`, `diagnose_*.py`,
and `*_diagnostic.py` before publishing. Generated `npz`, `png`, `pdf`, `log`,
cache, and Compton-table files belong in `MG_IMC/private/` unless intentionally
curated as release data.