"""
MG_IMC: Multigroup Implicit Monte Carlo in 2D

A multigroup extension of IMC2D for solving frequency-dependent
radiation transport problems.

Sub-packages
------------
problems
    Benchmark problem definitions (geometry, opacities, BCs, run scripts).
visualization
    Figure generation scripts for all benchmark problems.
"""

from .MG_IMC2D import (
    # Main simulation functions
    init_simulation,
    step,
    run_simulation,

    # State class
    SimulationState2DMG,

    # Constants
    __c as C_LIGHT,
    __a as A_RAD,

    # Utility functions (exported for advanced users)
    _sample_planck_spectrum_mixture_of_gammas,
    _sample_group_piecewise_constant,
)

from .mg_utils import (
    # Energy group creation
    create_log_energy_groups,
    create_linear_energy_groups,

    # Opacity models
    powerlaw_opacity_functions,
    constant_opacity_functions,
    make_powerlaw_sigma,
    make_powerlaw_group_opacity,

    # Material models
    simple_eos_functions,

    # Analysis tools
    compute_group_fractions,
    compute_total_opacity,
    print_group_info,

    # Mean opacities
    rosseland_mean_D,
    planck_mean_sigma,
    geom_mean_sigma,

    # Planck integrals
    planck_group_integral,
    planck_spectrum_by_group,

    # Checkpoint I/O
    atomic_checkpoint_save,
    checkpoint_load,

    # Mesh utilities
    build_clustered_edges,
)

# Sub-packages (imported lazily so that missing optional deps don't break the
# top-level import).
from . import problems      # noqa: F401
from . import visualization # noqa: F401

__version__ = "1.0.0"
__author__ = "Based on IMC2D by Ryan McClarren"

__all__ = [
    # Main API
    "init_simulation",
    "step",
    "run_simulation",
    "SimulationState2DMG",

    # Constants
    "C_LIGHT",
    "A_RAD",

    # Utilities
    "create_log_energy_groups",
    "create_linear_energy_groups",
    "powerlaw_opacity_functions",
    "constant_opacity_functions",
    "make_powerlaw_sigma",
    "make_powerlaw_group_opacity",
    "simple_eos_functions",
    "compute_group_fractions",
    "compute_total_opacity",
    "print_group_info",
    "rosseland_mean_D",
    "planck_mean_sigma",
    "geom_mean_sigma",
    "planck_group_integral",
    "planck_spectrum_by_group",
    "atomic_checkpoint_save",
    "checkpoint_load",
    "build_clustered_edges",

    # Sub-packages
    "problems",
    "visualization",
]
