# Public Release Checklist

## Private Material

- Place unpublished derivations, manuscript sources, and research notes in
  `nonEquilibriumDiffusion/private/`. This directory is ignored by Git.
- Do not place private material in `problems/`, solver modules, or committed
  documentation.
- This directory currently contains no LaTeX sources or LaTeX-style markup.
- `TRBDF2_SUMMARY.md` is a prose implementation summary. Review it before a
  release if it is intended to remain private.

## Release-Blocking Markers

Run this command before a public release:

```sh
rg -n -i '\\b(todo|fixme|xxx|hack)\\b' nonEquilibriumDiffusion \
  -g '*.py' -g '*.md' -g '*.rst' -g '!PUBLIC_RELEASE_CHECKLIST.md'
```

No such markers remain after the obsolete commented convergence-study stub was
removed from `finite_volume_1d/src/oneDFV.py`.

## Dependencies

- The finite-volume solvers require `numpy`, `scipy`, `matplotlib`, and `numba`.
- The multigroup solvers additionally require the external `planck_integrals`
  package. Install or publish that package before releasing a runnable
  multigroup workflow.
- The checked-in `.venv31213` has a broken Python symlink to a former local
  micromamba installation and is not a portable release environment.

## Directory Migration Status

All public solver implementations, regression tests, and problem drivers live
under their owning method directories:

```text
nonEquilibriumDiffusion/
  finite_volume_1d/
    src/
    problems/
    tests/
  finite_volume_2d/
    src/
    problems/
    tests/
  multigroup_1d/
    src/
    problems/
    tests/
  multigroup_2d/
    src/
    problems/
    tests/
  shared/
    src/
    problems/
    tests/
  private/                 # ignored; unpublished material only
```

Import solvers through their package paths, for example
`nonEquilibriumDiffusion.finite_volume_1d.src.oneDFV`. Scripts can be run
directly or as modules.

## Exclude From Public Release

Development-only scripts should be removed or kept only in the ignored private
area before publishing. Candidates include files named `debug_*`, `diagnose_*`,
`check_*`, `quick_*`, `*_diagnostic.py`, `*_performance.py`, `minimal_test.py`,
and benchmark/profile scripts. Each candidate still needs a maintainer review:
some `test_*` scripts are physics regressions, not disposable experiments.