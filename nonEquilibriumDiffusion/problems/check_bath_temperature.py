#!/usr/bin/env python3
"""
Diagnostic: Check T_bath vs T_surface for converging Marshak wave tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import converging_marshak_wave_marshak as test1
import converging_marshak_wave_test3_marshak as test3


def check_test1():
    print("=" * 70)
    print("TEST 1: Checking T_bath vs T_surface")
    print("=" * 70)
    
    times = test1.OUTPUT_TIMES_NS
    R = test1.R
    
    for t_ns in times:
        # Surface temperature (interior solution at r=R)
        T_surf = test1._surface_T_keV(t_ns)
        T_surf_HeV = T_surf * test1.T_HEV_PER_KEV
        
        # Bath temperature (includes correction via Λ)
        T_bath = test1._bath_T_keV(t_ns)
        T_bath_HeV = T_bath * test1.T_HEV_PER_KEV
        
        # Check xi_R and Λ
        xi_R = (R / (R / 10.0)) / (-t_ns) ** test1.DELTA
        W_xi_R = test1._Wxsi_scalar(xi_R)
        V_xi_R = test1._Vxsi_scalar(xi_R)
        Lambda_R = test1._Lambda(xi_R)
        
        # Correction factor
        correction = 1.0 + 0.103502 * Lambda_R * (-t_ns) ** (-0.541423)
        correction = correction ** 0.25
        
        print(f"\nTime: {t_ns:.4f} ns")
        print(f"  ξ_R = {xi_R:.6f}")
        print(f"  W(ξ_R) = {W_xi_R:.6f}")
        print(f"  V(ξ_R) = {V_xi_R:.6f}")
        print(f"  Λ(ξ_R) = ξ_R V W^{{-1.5}} = {Lambda_R:.6f}")
        print(f"  Correction = [1 + 0.103502 Λ (-t)^{{-0.541423}}]^{{1/4}} = {correction:.6f}")
        print(f"  T_surf = {T_surf_HeV:.6f} HeV  ({T_surf:.6f} keV)")
        print(f"  T_bath = {T_bath_HeV:.6f} HeV  ({T_bath:.6f} keV)")
        print(f"  T_bath / T_surf = {T_bath/T_surf:.6f}")
        print(f"  Expected ratio = {correction:.6f}")


def check_test3():
    print("\n" + "=" * 70)
    print("TEST 3: Checking T_bath vs T_surface")
    print("=" * 70)
    
    times = test3.OUTPUT_TIMES_NS
    R = test3.R
    
    for t_ns in times:
        # Surface temperature (interior solution at r=R)
        T_surf = test3._surface_T_keV(t_ns)
        T_surf_HeV = T_surf * test3.T_HEV_PER_KEV
        
        # Bath temperature (includes correction via Λ)
        T_bath = test3._bath_T_keV(t_ns)
        T_bath_HeV = T_bath * test3.T_HEV_PER_KEV
        
        # Check xi_R and Λ
        xi_R = (R / 1e-4) / (-t_ns) ** test3.DELTA
        W_xi_R = test3._Wxsi_scalar(xi_R)
        V_xi_R = test3._Vxsi_scalar(xi_R)
        Lambda_R = test3._Lambda(xi_R)
        
        # Correction factor (Test 3 constants from equation 8.88)
        correction = 1.0 + 0.075821 * Lambda_R * (-t_ns) ** (-0.316092)
        correction = correction ** 0.25
        
        print(f"\nTime: {t_ns:.4f} ns")
        print(f"  ξ_R = {xi_R:.6f}")
        print(f"  W(ξ_R) = {W_xi_R:.6f}")
        print(f"  V(ξ_R) = {V_xi_R:.6f}")
        print(f"  Λ(ξ_R) = ξ_R^0.6625 V W^{{-1}} = {Lambda_R:.6f}")
        print(f"  Correction = [1 + 0.075821 Λ (-t)^{{-0.316092}}]^{{1/4}} = {correction:.6f}")
        print(f"  T_surf = {T_surf_HeV:.6f} HeV  ({T_surf:.6f} keV)")
        print(f"  T_bath = {T_bath_HeV:.6f} HeV  ({T_bath:.6f} keV)")
        print(f"  T_bath / T_surf = {T_bath/T_surf:.6f}")
        print(f"  Expected ratio = {correction:.6f}")


def check_bc_sign():
    print("\n" + "=" * 70)
    print("BOUNDARY CONDITION SIGN CHECK")
    print("=" * 70)
    print("\nStandard Marshak BC at outer boundary r=R:")
    print("  φ/2 + (c/3k) * |dφ/dr| = (a c T_bath^4) / 2")
    print("\nOr equivalently (no factor of 1/2):")
    print("  φ + 2(c/3k) * |dφ/dr| = a c T_bath^4")
    print("\nIn solver BC form: α φ + β (dφ/dr) = γ")
    print("\nCurrent implementation:")
    print("  α = 1,  β = c/(3k),  γ = a c T_bath^4")
    print("\nPossible corrections:")
    print("  1) Missing factor of 1/2 on φ term:     α = 1/2, β = c/(3k),   γ = a c T_bath^4 / 2")
    print("  2) Missing factor of 2 on gradient:    α = 1,   β = 2c/(3k),  γ = a c T_bath^4")
    print("  3) Wrong sign (gradient should be +):  α = 1,   β = -c/(3k),  γ = a c T_bath^4")


if __name__ == '__main__':
    check_test1()
    check_test3()
    check_bc_sign()
