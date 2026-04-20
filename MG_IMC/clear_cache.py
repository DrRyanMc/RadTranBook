#!/usr/bin/env python3
"""
Clear Numba JIT cache for MG_IMC2D module.

Run this if you get ModuleNotFoundError when loading cached functions,
especially when switching between Python versions.
"""

import os
import glob
import sys

cache_dir = "__pycache__"

if not os.path.exists(cache_dir):
    print(f"No cache directory found: {cache_dir}")
    sys.exit(0)

# Find all Numba cache files
cache_files = []
for ext in ['*.nbc', '*.nbi', '*.pyc']:
    cache_files.extend(glob.glob(os.path.join(cache_dir, ext)))

if not cache_files:
    print(f"No Numba cache files found in {cache_dir}")
    sys.exit(0)

print(f"Found {len(cache_files)} Numba cache files in {cache_dir}")
print("Clearing cache...")

removed = 0
for f in cache_files:
    try:
        os.remove(f)
        removed += 1
    except Exception as e:
        print(f"Warning: Could not remove {f}: {e}")

print(f"✓ Removed {removed} cache files")
print("\nCache cleared! You can now run your test.")
