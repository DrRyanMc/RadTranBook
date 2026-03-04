#!/usr/bin/env python3
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import (
    flux_limiter_levermore_pomraning,
    flux_limiter_sum, 
    flux_limiter_larsen,
    flux_limiter_max
)

# Test that different limiters give different results
R_vals = [0.1, 1.0, 5.0, 10.0]
print("Testing flux limiters:")
print(f"{'R':<10} {'LP':<12} {'Larsen':<12} {'Sum':<12} {'Max':<12}")
for R in R_vals:
    lp = flux_limiter_levermore_pomraning(R)
    larsen = flux_limiter_larsen(R)
    sum_lim = flux_limiter_sum(R)
    max_lim = flux_limiter_max(R)
    print(f"{R:<10.2f} {lp:<12.6f} {larsen:<12.6f} {sum_lim:<12.6f} {max_lim:<12.6f}")

print("\nDifferences at R=5.0:")
R = 5.0
lp = flux_limiter_levermore_pomraning(R)
larsen = flux_limiter_larsen(R)
sum_lim = flux_limiter_sum(R)
max_lim = flux_limiter_max(R)
print(f"LP - Larsen: {lp - larsen:.6f}")
print(f"LP - S um: {lp - sum_lim:.6f}")
print(f"LP - Max: {lp - max_lim:.6f}")
