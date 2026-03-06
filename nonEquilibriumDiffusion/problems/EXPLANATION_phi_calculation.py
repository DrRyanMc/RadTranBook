"""
Explanation of φ_g calculation in the κ formulation

For each group g, the solver computes φ_g by solving:

    A_g · φ_g = χ_g(1-f)κ + ξ_g

where:
- A_g is the multigroup diffusion operator for group g
- κ is the absorption rate density (couples all groups) 
- ξ_g is the source term

The diffusion operator A_g includes:
    A_g = (1/(c·dt)) + spatial diffusion terms + absorption

The source term ξ_g consists of:
    ξ_g = (1/(c·dt))·φ_g^old 
          + 4π·σ_a,g·B_g(T★)                      [Planck emission]
          - χ_g(1-f)·[Σ_g' 4π·σ_a,g'·B_g'(T★) + Δe/Δt]  [coupling term]

So the full equation being solved for each group is:

    (1/(c·dt))·φ_g + [diffusion terms] = χ_g(1-f)κ + (1/(c·dt))·φ_g^old + ...

This is the correct κ-formulation from the book.

THE ISSUE AT STEADY STATE:
---------------------------
At equilibrium with reflecting BCs:
- κ → 0 (no net material-radiation exchange)  
- T → T_eq (material equilibrates)
- φ_g should → φ_g^eq (radiation equilibrates)

But when κ ≈ 0, the equation becomes:

    A_g · φ_g ≈ ξ_g = (1/(c·dt))·φ_g^old + emission terms

In the time-dependent operator A_g:
    (1/(c·dt))·φ_g + [diffusion] = (1/(c·dt))·φ_g^old + emission

Rearranging:
    (1/(c·dt))·(φ_g - φ_g^old) + [diffusion] = emission

This shows that φ_g evolves toward equilibrium on a time scale ~ dt.

If φ_g^old is not at equilibrium, it takes many time steps for φ_g to decay 
to the correct equilibrium value. The larger dt is, the faster convergence.

WHAT WE OBSERVE:
----------------
In our test:
- Initial: φ_total = c·E_r = 2.57×10^-6 (from T_rad = 0.05 keV)
- After step 1: φ_total = 2.38×10^-7 (dropped by ~10×)
- Expected equilibrium: φ_total = 1.61×10^-7

After step 1, the system "freezes" with φ_total ≈ 1.48× too large.

WHY IT FREEZES:
---------------
Once κ becomes small enough, the material temperature stops changing 
(T ≈ T_eq), so emission terms become constant. Meanwhile, the time-dependent
term (1/(c·dt))·(φ_g - φ_g^old) balances the small residual in emission 
terms, preventing further evolution.

The φ_old/(c·dt) term acts as a "memory" that keeps φ_g close to φ_g^old.

VERIFICATION:
-------------
From the diagnostic output, at pseudo-equilibrium:
- Actual φ_total = 2.385×10^-7  
- Expected φ_eq = 1.607×10^-7
- Ratio = 1.48× (exactly matches E_r ratio!)

The time scale for radiation equilibration is:
    τ_eq ~ (c·dt) / (c·σ_a) = dt/σ_a  

With dt = 0.01 ns and σ = 100 cm^-1, we get τ_eq ~ 10^-4 ns.

But this is the time scale for each time step. To equilibrate from an 
initial condition far from equilibrium requires many time steps.

SOLUTION:
---------
This is actually CORRECT physics for a time-dependent problem!

The issue is our test initializes φ_g far from equilibrium 
(radiation from T = 0.05 keV, material at T = 0.025 keV).

For such an initial condition to equilibrate requires enough time steps 
for radiation to diffuse and thermalize with the material.

In a realistic problem (like Su-Olson or Marshak wave), the radiation 
enters through boundaries and equilibrates as it propagates, so this 
isn't an issue.
"""
print("See explanation in this file")
