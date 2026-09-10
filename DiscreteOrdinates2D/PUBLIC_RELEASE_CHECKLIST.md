# Public Release Checklist

## Private Material

Keep unpublished derivations, LaTeX reports, scratch diagnostics, and generated
results in `DiscreteOrdinates2D/private/`, which is ignored by Git. The public
component contains no LaTeX source or build artifacts.

## Public Layout

```text
DiscreteOrdinates2D/
  src/             # Cartesian and cylindrical gray/multigroup 2-D S_N solvers
  problems/        # benchmark drivers
  tests/           # reusable regression tests and smoke runner
  visualization/   # analysis and comparison plots
  private/         # ignored
```

Import solvers through `DiscreteOrdinates2D.src`, for example
`DiscreteOrdinates2D.src.sn_solver_2d`.

## Release Audit

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' DiscreteOrdinates2D \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

Review new `_*.py`, `debug_*.py`, `check_*.py`, and `diagnose_*.py` files before
publishing. Generated `npz`, `png`, `pdf`, log, and bytecode artifacts belong in
the private archive unless explicitly curated as release data.