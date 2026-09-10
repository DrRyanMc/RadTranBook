# Public Release Checklist

## Private Material

Keep unpublished derivations, LaTeX writeups, result archives, and generated
figures in the ignored top-level `private/M1/` archive. Public M1 source contains
no LaTeX files.

## Public Layout

```text
M1/
  src/             # one-dimensional, two-dimensional, and SCB M1 solvers
  problems/        # benchmark drivers
  tests/           # regression tests
  visualization/   # comparison and analysis tools
```

Import the public 1-D API with `from M1 import M1Solver1D`; 2-D solvers are in
`M1.src`.

## Release Audit

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' M1 \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

Review new `debug_*`, `benchmark_*`, `diagnose_*`, and generated artifact files
before publication.