import IMCSlab as imc
import IMC1D_CarterForest as imc_cf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
})

a = imc.__a
c = imc.__c


def run_fleck_cummings(dt, t_final, mesh, t_init, tr_init, eos, inv_eos, sigma_a_f, cv, source,
                       n_target, n_boundary, n_source, n_max, t_boundary):
    """Run Fleck-Cummings IMC."""
    state = imc.init_simulation(n_target, t_init, tr_init, mesh, eos, inv_eos)
    step = 0
    while state.time < t_final - 1e-12:
        step_dt = min(dt, t_final - state.time)
        state, info = imc.step(state, n_target, n_boundary, n_source, n_max,
                               t_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
                               reflect=(False, True))
        if step % 20 == 0 or state.time >= t_final - 1e-12:
            print(f"  FC: t={info['time']:.4f} N={info['N_particles']:5d} "
                  f"E={info['total_energy']:.6f} dE={info['energy_loss']:.3e}")
        step += 1
    return state


def run_carter_forest(dt, t_final, mesh, t_init, tr_init, eos, inv_eos, sigma_a_f, cv, source,
                      n_target, n_boundary, n_source, n_max, t_boundary):
    """Run Carter-Forest IMC."""
    state = imc_cf.init_simulation(n_target, t_init, tr_init, mesh, eos, inv_eos)
    step = 0
    while state.time < t_final - 1e-12:
        step_dt = min(dt, t_final - state.time)
        state, info = imc_cf.step(state, n_target, n_boundary, n_source, n_max,
                                  t_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
                                  reflect=(False, True))
        if step % 20 == 0 or state.time >= t_final - 1e-12:
            print(f"  CF: t={info['time']:.4f} N={info['N_particles']:5d} "
                  f"E={info['total_energy']:.6f} dE={info['energy_loss']:.3e}")
        step += 1
    return state


def main():
    
    # --- Problem parameters ---
    n_target = 10000
    n_boundary = 10000
    n_max = 4 * 10**4
    n_source = 0

    t_finals = [1.0, 5.0, 10.0]  # ns - three different times
    dt = 0.025  # ns - single timestep for all runs

    l_domain = 0.2
    i_cells = 50
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

    # --- Run both methods for each time ---
    results_fc_temp = {}
    results_fc_radtemp = {}
    results_cf_temp = {}
    results_cf_radtemp = {}
    
    for t_final in t_finals:
        print(f"\n{'='*60}")
        print(f"Running dt = {dt:.3f} ns to t = {t_final:.1f} ns")
        print(f"{'='*60}")
        
        # Fleck-Cummings
        print(f"\nFleck-Cummings IMC:")
        np.random.seed(12345)
        state_fc = run_fleck_cummings(
            dt, t_final, mesh, t_init, tr_init, eos, inv_eos,
            sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary
        )
        results_fc_temp[t_final] = state_fc.temperature.copy()
        results_fc_radtemp[t_final] = state_fc.radiation_temperature.copy()
        
        # Carter-Forest
        print(f"\nCarter-Forest IMC:")
        np.random.seed(12345)
        state_cf = run_carter_forest(
            dt, t_final, mesh, t_init, tr_init, eos, inv_eos,
            sigma_a_f, cv, source, n_target, n_boundary, n_source, n_max, t_boundary
        )
        results_cf_temp[t_final] = state_cf.temperature.copy()
        results_cf_radtemp[t_final] = state_cf.radiation_temperature.copy()

    # --- Create plots with all times on same plot ---
    my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_styles = ['-', '--', '-.']
    
    # Fleck-Cummings material temperature
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t_final in enumerate(t_finals):
        # Compute self-similar solution for this time
        xi_vals = np.linspace(0.0, xi_max, 400)
        shape = np.where(
            xi_vals < xi_max,
            (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
            1e-30,
        )
        t_ss = t_bc * np.power(shape, 1.0 / 6.0)
        r_ss = xi_vals * np.sqrt(k_const * t_final)
        
        color = my_colors[idx % len(my_colors)]
        ax.plot(
            mesh_midpoints,
            results_fc_temp[t_final],
            color=color,
            linestyle=line_styles[idx],
            linewidth=2.0,
            alpha=0.9,
            label=f"t = {t_final:.0f} ns",
            zorder=2
        )
        
        # Self-similar solution
        ax.plot(r_ss, t_ss, color=color, linestyle=':', linewidth=1.5, 
                alpha=0.5, zorder=1)
    
    ax.set_xlim([0.0, l_domain])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('temperature (keV)')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    out_pdf = 'marshak_wave_fc_material_temp.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved: {out_pdf}")
    plt.close()
    
    # Carter-Forest material temperature
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t_final in enumerate(t_finals):
        # Compute self-similar solution for this time
        xi_vals = np.linspace(0.0, xi_max, 400)
        shape = np.where(
            xi_vals < xi_max,
            (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
            1e-30,
        )
        t_ss = t_bc * np.power(shape, 1.0 / 6.0)
        r_ss = xi_vals * np.sqrt(k_const * t_final)
        
        color = my_colors[idx % len(my_colors)]
        ax.plot(
            mesh_midpoints,
            results_cf_temp[t_final],
            color=color,
            linestyle=line_styles[idx],
            linewidth=2.0,
            alpha=0.9,
            label=f"t = {t_final:.0f} ns",
            zorder=2
        )
        
        # Self-similar solution
        ax.plot(r_ss, t_ss, color=color, linestyle=':', linewidth=1.5, 
                alpha=0.5, zorder=1)
    
    ax.set_xlim([0.0, l_domain])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('temperature (keV)')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    out_pdf = 'marshak_wave_cf_material_temp.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved: {out_pdf}")
    plt.close()
    
    # Fleck-Cummings radiation temperature
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t_final in enumerate(t_finals):
        # Compute self-similar solution for this time
        xi_vals = np.linspace(0.0, xi_max, 400)
        shape = np.where(
            xi_vals < xi_max,
            (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
            1e-30,
        )
        t_ss = t_bc * np.power(shape, 1.0 / 6.0)
        r_ss = xi_vals * np.sqrt(k_const * t_final)
        
        color = my_colors[idx % len(my_colors)]
        ax.plot(
            mesh_midpoints,
            results_fc_radtemp[t_final],
            color=color,
            linestyle=line_styles[idx],
            linewidth=2.0,
            alpha=0.9,
            label=f"t = {t_final:.0f} ns",
            zorder=2
        )
        
        # Self-similar solution
        ax.plot(r_ss, t_ss, color=color, linestyle=':', linewidth=1.5, 
                alpha=0.5, zorder=1)
    
    ax.set_xlim([0.0, l_domain])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('radiation temperature (keV)')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    out_pdf = 'marshak_wave_fc_radiation_temp.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved: {out_pdf}")
    plt.close()
    
    # Carter-Forest radiation temperature
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, t_final in enumerate(t_finals):
        # Compute self-similar solution for this time
        xi_vals = np.linspace(0.0, xi_max, 400)
        shape = np.where(
            xi_vals < xi_max,
            (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
            1e-30,
        )
        t_ss = t_bc * np.power(shape, 1.0 / 6.0)
        r_ss = xi_vals * np.sqrt(k_const * t_final)
        
        color = my_colors[idx % len(my_colors)]
        ax.plot(
            mesh_midpoints,
            results_cf_radtemp[t_final],
            color=color,
            linestyle=line_styles[idx],
            linewidth=2.0,
            alpha=0.9,
            label=f"t = {t_final:.0f} ns",
            zorder=2
        )
        
        # Self-similar solution
        ax.plot(r_ss, t_ss, color=color, linestyle=':', linewidth=1.5, 
                alpha=0.5, zorder=1)
    
    ax.set_xlim([0.0, l_domain])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('radiation temperature (keV)')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    out_pdf = 'marshak_wave_cf_radiation_temp.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved: {out_pdf}")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"All plots saved successfully")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
