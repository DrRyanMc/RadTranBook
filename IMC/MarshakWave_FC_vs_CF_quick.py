"""
Quick test version of Marshak wave comparison with reduced particle counts
"""

import numpy as np
import matplotlib.pyplot as plt
import IMCSlab as imc
import IMC1D_CarterForest as imccf

# Physical constants
a = 0.01372
c = 299.792458

def run_fleck_cummings(dt, t_final, mesh, t_init, tr_init, eos, inv_eos,
                       sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary):
    """Run Fleck-Cummings IMC"""
    state = imc.State(mesh, t_init, tr_init, eos, inv_eos,
                      sigma_a_f, sigma_a_f, cv, source)
    t = 0.0
    step_num = 0
    
    while t < t_final:
        state, info = imc.step(state, n_target, n_boundary, n_source, n_max,
                               dt, (False, False), t_boundary)
        t += dt
        step_num += 1
    
    return state

def run_carter_forest(dt, t_final, mesh, t_init, tr_init, eos, inv_eos,
                      sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary):
    """Run Carter-Forest IMC"""
    state = imccf.State(mesh, t_init, tr_init, eos, inv_eos,
                        sigma_a_f, sigma_a_f, cv, source)
    t = 0.0
    step_num = 0
    
    while t < t_final:
        state, info = imccf.step(state, n_target, n_boundary, n_source, n_max,
                                 dt, (False, False), t_boundary)
        t += dt
        step_num += 1
        if step_num % 10 == 0:
            print(f"  CF: t={t:.4f} N={info['n_census']} E={info['total_energy']:.6f} dE={info['delta_energy']:.3e}", flush=True)
    
    return state

def main():
    # --- Problem parameters (REDUCED for quick test) ---
    n_target = 1000  # Fewer particles
    n_boundary = 1000
    n_max = 5000
    n_source = 0

    t_finals = [1.0]  # Just one time for quick test
    dt = 0.05  # Larger timestep

    l_domain = 0.2
    i_cells = 25  # Fewer cells
    mesh = np.zeros((i_cells, 2))
    dx = l_domain / i_cells
    for i in range(i_cells):
        mesh[i] = [i * dx, (i + 1) * dx]
    mesh_midpoints = 0.5 * (mesh[:, 0] + mesh[:, 1])

    t_init = np.zeros(i_cells) + 1e-4
    tr_init = np.zeros(i_cells) + 1e-4
    t_boundary = (1.0, 0)
    source = np.zeros(i_cells)

    sigma_a_f = lambda t: 300 * t**-3
    cv_val = 0.3
    eos = lambda t: cv_val * t
    inv_eos = lambda u: u / cv_val
    cv = lambda t: cv_val

    # --- Self-similar solution parameters ---
    xi_max = 1.11305
    omega = 0.05989
    rho = 1.0
    t_bc = t_boundary[0]
    sigma_0 = sigma_a_f(t_boundary[0])
    k_const = 8 * a * c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)

    # --- Run both methods ---
    print("Running Fleck-Cummings...")
    np.random.seed(12345)
    state_fc = run_fleck_cummings(
        dt, t_finals[0], mesh, t_init, tr_init, eos, inv_eos,
        sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary
    )
    print(f"FC Final: T_rad_max={state_fc.radiation_temperature.max():.4f} keV")
    
    print("\nRunning Carter-Forest...")
    np.random.seed(12345)
    state_cf = run_carter_forest(
        dt, t_finals[0], mesh, t_init, tr_init, eos, inv_eos,
        sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary
    )
    print(f"CF Final: T_rad_max={state_cf.radiation_temperature.max():.4f} keV")

    # --- Quick plot ---
    xi_vals = np.linspace(0.0, xi_max, 400)
    shape = np.where(
        xi_vals < xi_max,
        (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
        1e-30,
    )
    t_ss = t_bc * np.power(shape, 1.0 / 6.0)
    r_ss = xi_vals * np.sqrt(k_const * t_finals[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # FC
    ax1.plot(mesh_midpoints, state_fc.temperature, 'b-', lw=2, label='Material T')
    ax1.plot(mesh_midpoints, state_fc.radiation_temperature, 'r--', lw=2, label='Radiation T')
    ax1.plot(r_ss, t_ss, 'k:', lw=1.5, alpha=0.5, label='Self-similar')
    ax1.set_xlim([0.0, l_domain])
    #ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('position (cm)')
    ax1.set_ylabel('temperature (keV)')
    ax1.set_title('Fleck-Cummings IMC')
    ax1.legend()
    
    # CF
    ax2.plot(mesh_midpoints, state_cf.temperature, 'b-', lw=2, label='Material T')
    ax2.plot(mesh_midpoints, state_cf.radiation_temperature, 'r--', lw=2, label='Radiation T')
    ax2.plot(r_ss, t_ss, 'k:', lw=1.5, alpha=0.5, label='Self-similar')
    ax2.set_xlim([0.0, l_domain])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('position (cm)')
    ax2.set_ylabel('temperature (keV)')
    ax2.set_title('Carter-Forest IMC')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('marshak_wave_quick_test.pdf', dpi=300)
    print("\nSaved: marshak_wave_quick_test.pdf")


if __name__ == '__main__':
    main()
