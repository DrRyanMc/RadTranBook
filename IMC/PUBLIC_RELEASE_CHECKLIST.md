# Public Release Checklist

## Private Material

- Place unpublished derivations, manuscript sources, research notes, one-off
  diagnostics, and generated outputs in `IMC/private/`. This directory is
  ignored by Git.
- This directory contains no LaTeX sources or LaTeX-style markup.
- Keep implementation documentation concise. Move any new derivations or
  manuscript-oriented explanations to the private archive.

## Public Layout

```text
IMC/
  fleck_cummings/
    src/
    problems/
    tests/
  carter_forest/
    src/
    problems/
    tests/
  shared/
    problems/
    visualization/
  private/                 # ignored; unpublished material only
```

Import solver modules through their package paths, for example
`IMC.fleck_cummings.src.IMC1D`. Problem and test drivers can be run directly or
as modules.

## Dependencies

- All public solvers require `numpy`; plotting drivers additionally require
  `matplotlib`.
- `numba` is optional: serial solvers provide a Python fallback.
- MPI solvers require `mpi4py` and an MPI runtime.

## Release-Blocking Markers

Run this command before publishing:

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' IMC \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

No such markers remain in the public IMC source tree.

## Exclude From Public Release

Keep profiling, scratch, and diagnostic files out of public commits. Review
files named `_*.py`, `benchmark_*.py`, `monitor_*.py`, `debug_*.py`, and
`*_diagnostic.py` before publication. Generated `png`, `pdf`, `npz`, `pkl`,
`csv`, `json`, `log`, `out`, and cache artifacts belong in `IMC/private/` unless
they are explicitly curated release data.