#!/usr/bin/env python3
"""Small smoke test for IMC2D geometry modes."""

import numpy as np

import IMC2D as imc2d


def _run_case(geometry):
    if geometry == "xy":
        e1 = np.linspace(0.0, 0.1, 6)   # x edges
        e2 = np.linspace(0.0, 0.1, 5)   # y edges
        reflect = (True, True, True, True)
    else:
        e1 = np.linspace(0.0, 0.1, 6)   # r edges
        e2 = np.linspace(0.0, 0.1, 5)   # z edges
        reflect = (True, False, True, True)

    nx = len(e1) - 1
    ny = len(e2) - 1

    T0 = np.full((nx, ny), 0.5)
    Tr0 = np.full((nx, ny), 0.5)

    sigma_a = lambda T: 50.0 + 0.0 * T
    cv_val = 0.1
    eos = lambda T: cv_val * T
    inv_eos = lambda u: u / cv_val
    cv = lambda T: cv_val + 0.0 * T

    source = np.zeros((nx, ny))
    T_boundary = (0.0, 0.0, 0.0, 0.0)

    times, Tr_hist, T_hist = imc2d.run_simulation(
        Ntarget=2000,
        Nboundary=0,
        Nsource=0,
        Nmax=10000,
        Tinit=T0,
        Tr_init=Tr0,
        T_boundary=T_boundary,
        dt=0.005,
        edges1=e1,
        edges2=e2,
        sigma_a_func=sigma_a,
        eos=eos,
        inv_eos=inv_eos,
        cv=cv,
        source=source,
        final_time=0.02,
        reflect=reflect,
        output_freq=1,
        geometry=geometry,
    )

    print(f"[{geometry}] ran {len(times)-1} steps; final mean T={T_hist[-1].mean():.6f} keV")


def main():
    _run_case("xy")
    _run_case("rz")


if __name__ == "__main__":
    main()
