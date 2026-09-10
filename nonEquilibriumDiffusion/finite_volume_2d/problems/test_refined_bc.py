#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Quick test of refined zoning BC visualization
"""
import sys
sys.path.insert(0, '..')

from refined_zoning_noneq import main

# Run just to first output
solver = main(output_times=[0.001], dt_max=1.0, use_refined_mesh=False)
print("\nTest complete!")
