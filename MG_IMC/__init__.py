"""
MG_IMC: Multigroup Implicit Monte Carlo in 2D

A multigroup extension of IMC2D for solving frequency-dependent 
radiation transport problems.
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
    
    # Material models
    simple_eos_functions,
    
    # Analysis tools
    compute_group_fractions,
    compute_total_opacity,
    print_group_info,
)

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
    "simple_eos_functions",
    "compute_group_fractions",
    "compute_total_opacity",
    "print_group_info",
]
