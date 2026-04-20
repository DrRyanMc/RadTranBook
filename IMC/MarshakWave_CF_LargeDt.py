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

a = imc_cf.__a
c = imc_cf.__c


def run_case(dt, t_final, mesh, t_init, tr_init, eos, inv_eos, sigma_a_f, cv, source,
             n_target, n_boundary, n_source, n_max, t_boundary):
    state = imc_cf.init_simulation(n_target, t_init, tr_init, mesh, eos, inv_eos)
    step = 0
    while state.time < t_final - 1e-12:
        step_dt = min(dt, t_final - state.time)
        state, info = imc_cf.step(state, n_target, n_boundary, n_source, n_max,
                                  t_boundary, step_dt, mesh, sigma_a_f, inv_eos, cv, source,
                                  reflect=(False, True))
        if dt >= 0.1 or step % 20 == 0 or state.time >= t_final - 1e-12:
            print(f"dt={dt:.2f}\t"
                  f"t={info['time']:.6f}\t"
                  f"N={info['N_particles']}\t"
                  f"E={info['total_energy']:.6f}\t"
                  f"dE={info['energy_loss']:.6e}")
        step += 1
    return state


def main():
    # --- Problem parameters ---
    n_target = 10000
    n_boundary = 10000
    n_max = 4 * 10**4
    n_source = 0

    t_finals = [1.0, 5.0, 10.0]  # ns
    dt_values = [0.01, 0.025, 0.05]  # ns

    l_domain = 0.2  # cm
    i_cells = 50
    mesh = np.zeros((i_cells, 2))
    dx = l_domain / i_cells
    for i in range(i_cells):
        mesh[i] = [i * dx, (i + 1) * dx]
    mesh_midpoints = 0.5 * (mesh[:, 0] + mesh[:, 1])

    t_init = np.zeros(i_cells) + 1e-4  # keV
    tr_init = np.zeros(i_cells) + 1e-4  # keV
    t_boundary = (1.0, 0)  # Left: 1 keV, Right: vacuum
    source = np.zeros(i_cells)

    # Marshak wave parameters
    sigma_a_f = lambda t: 300 * t**-3  # cm^-1
    cv_val = 0.3  # GJ/(cm^3 keV)
    eos = lambda t: cv_val * t
    inv_eos = lambda u: u / cv_val
    cv = lambda t: cv_val

    # --- Self-similar parameters (for each t_final) ---
    xi_max = 1.11305
    omega = 0.05989
    rho = 1.0
    t_bc = t_boundary[0]

    # --- Run each timestep case to each t_final ---
    results = {dt: {} for dt in dt_values}
    results_rad = {dt: {} for dt in dt_values}
    for t_final in t_finals:
        sigma_0 = sigma_a_f(t_boundary[0])
        k_const = 8 * a * c / ((4 + 3) * 3 * sigma_0 * rho * cv_val)
        xi_vals = np.linspace(0.0, xi_max, 400)
        shape = np.where(
            xi_vals < xi_max,
            (1.0 - xi_vals / xi_max) * (1.0 + omega * xi_vals / xi_max),
            1e-30,
        )
        t_ss = t_bc * np.power(shape, 1.0 / 6.0)
        r_ss = xi_vals * np.sqrt(k_const * t_final)

        for dt in dt_values:
            np.random.seed(12345)
            final_state = run_case(
                dt,
                t_final,
                mesh,
                t_init,
                tr_init,
                eos,
                inv_eos,
                sigma_a_f,
                cv,
                source,
                n_target,
                n_boundary,
                n_source,
                n_max,
                t_boundary,
            )
            results[dt][t_final] = final_state.temperature.copy()
            results_rad[dt][t_final] = final_state.radiation_temperature.copy()

        # Store self-similar solution for each t_final
        if t_final == t_finals[0]:
            r_ss_1, t_ss_1 = r_ss, t_ss
        elif t_final == t_finals[1]:
            r_ss_5, t_ss_5 = r_ss, t_ss
        elif t_final == t_finals[2]:
            r_ss_10, t_ss_10 = r_ss, t_ss

    # --- Plot material temperature comparison ---
    my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    line_styles = ['-', '--', '-.']
    time_colors = ['#888888', '#444444', '#000000']  # faded to dark for time overlays

    # Plot all times for each dt, but only label the first time for legend
    for idx, (dt, line_style) in enumerate(zip(dt_values, line_styles)):
        color = my_colors[idx % len(my_colors)]
        for t_idx, t_final in enumerate(t_finals):
            label = (r"$\Delta t=$" + f"{dt:.3f} ns") if t_idx == 0 else None
            ax.plot(
                mesh_midpoints,
                results[dt][t_final],
                color=color,
                linestyle=line_style,
                linewidth=1.0 if t_idx == 0 else 1.0,
                alpha=1.0,
                label=label,
                zorder=2
            )

    # Plot self-similar solutions for each time
    ax.plot(r_ss_1, t_ss_1, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
    #check if t_finals has more than one element
    try:
        if  t_final >= t_finals[1]:
            ax.plot(r_ss_5, t_ss_5, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
            if  t_final >= t_finals[2]:
                ax.plot(r_ss_10, t_ss_10, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
    except:
        pass
    ax.set_xlim([0.0, l_domain])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('temperature (keV)')
    ax.legend(fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()

    out_pdf = f'marshak_wave_carter_forest_large_dt_comparison_multi.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved figure: {out_pdf}")
    plt.show()
    # --- Plot radiation temperature comparison ---
    my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    line_styles = ['-', '--', '-.']
    time_colors = ['#888888', '#444444', '#000000']  # faded to dark for time overlays

    # Plot all times for each dt, but only label the first time for legend
    for idx, (dt, line_style) in enumerate(zip(dt_values, line_styles)):
        color = my_colors[idx % len(my_colors)]
        for t_idx, t_final in enumerate(t_finals):
            label = (r"$\Delta t=$" + f"{dt:.3f} ns") if t_idx == 0 else None
            ax.plot(
                mesh_midpoints,
                results_rad[dt][t_final],
                color=color,
                linestyle=line_style,
                linewidth=1.0 if t_idx == 0 else 1.0,
                alpha=1.0,
                label=label,
                zorder=2
            )

    # Plot self-similar solutions for each time
    ax.plot(r_ss_1, t_ss_1, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
    #check if t_finals has more than one element
    try:
        if  t_final >= t_finals[1]:
            ax.plot(r_ss_5, t_ss_5, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
            if  t_final >= t_finals[2]:
                ax.plot(r_ss_10, t_ss_10, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=0)
    except:
        pass
    ax.set_xlim([0.0, l_domain])
    ax.set_xlabel('position (cm)')
    ax.set_ylabel('radiation temperature (keV)')
    ax.legend(fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()

    out_pdf = f'marshak_wave_carter_forest_large_dt_comparison_multi_rad.pdf'
    plt.savefig(out_pdf, dpi=600)
    print(f"Saved figure: {out_pdf}")
    plt.show()

if __name__ == '__main__':
    main()
