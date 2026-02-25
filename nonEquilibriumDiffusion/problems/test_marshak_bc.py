#!/usr/bin/env python3
"""Test boundary condition application for Marshak wave"""
import sys
sys.path.insert(0, '..')
import numpy as np

# Test the opacity function
from marshak_wave_2d import marshak_rosseland_opacity

# Test with scalar
T_scalar = 0.1
sigma_scalar = marshak_rosseland_opacity(T_scalar, 0.0, 0.0)
print(f"Scalar test: T={T_scalar}, sigma={sigma_scalar}, type={type(sigma_scalar)}")

# Test with arrays (meshgrid)
x = np.array([0.0, 0.1, 0.2])
y = np.array([0.0, 0.1])
X, Y = np.meshgrid(x, y, indexing='ij')
T_array = 0.1 * np.ones_like(X)
print(f"\nArray shapes: X.shape={X.shape}, Y.shape={Y.shape}, T.shape={T_array.shape}")

sigma_array = marshak_rosseland_opacity(T_array, X, Y)
print(f"Array test: sigma.shape={sigma_array.shape if hasattr(sigma_array, 'shape') else 'scalar'}")
print(f"sigma_array type: {type(sigma_array)}")
if hasattr(sigma_array, 'shape'):
    print(f"sigma_array values:\n{sigma_array}")

# Test boundary condition function
from marshak_wave_2d import bc_blackbody_incoming
A, B, C = bc_blackbody_incoming(1.0, (0.0, 0.0), 0.0)
print(f"\nBoundary condition: A={A}, B={B}, C={C}")
print(f"abs(B) < 1e-14: {abs(B) < 1e-14}")
phi_boundary = C / A
print(f"phi_boundary = {phi_boundary}")
