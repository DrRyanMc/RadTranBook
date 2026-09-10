# Public Release Checklist

## Private Material

Place unpublished derivations, scratch scripts, legacy implementations, and
generated outputs in `DiscreteOrdinates/private/`, which is ignored by Git.
The public component contains no LaTeX sources.

## Public Layout

```text
DiscreteOrdinates/
  src/             # gray and multigroup 1-D S_N solvers and physics updates
  problems/        # benchmark drivers
  tests/           # reusable regression tests
  visualization/   # analysis and comparison plots
  private/         # ignored
```

Import solvers through `DiscreteOrdinates.src`, for example
`DiscreteOrdinates.src.sn_solver`.

## Release Audit

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' DiscreteOrdinates \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

Review new `_*.py`, `debug_*.py`, `check_*.py`, and `diagnose_*.py` files before
publishing. Generated `npz`, `png`, `pdf`, log, and bytecode artifacts belong in
the private archive unless explicitly curated as release data.