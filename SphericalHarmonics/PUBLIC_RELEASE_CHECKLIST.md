# Public Release Checklist

## Private Material

Keep unpublished derivations, LaTeX writeups, performance studies, and generated
outputs in the ignored top-level `private/SphericalHarmonics/` archive. Public
P_N and SP_N source contains no LaTeX files.

## Public Layout

```text
SphericalHarmonics/
  src/             # P_N solvers and required Jacobian CSV data
  problems/        # P_N benchmark drivers
  tests/           # P_N regression tests
  visualization/   # P_N analysis tools
  SPN/             # simplified P_N method with the same structure
```

The Jacobian CSV files under `src/Jacobians/` are required solver data and must
remain public. Import P_N solvers via `SphericalHarmonics.src` and SP_N solvers
via `SphericalHarmonics.SPN.src`.

## Release Audit

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' SphericalHarmonics \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

Review new `debug_*`, `benchmark_*`, `diagnose_*`, and generated artifact files
before publication.