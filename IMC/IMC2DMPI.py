"""IMC2DMPI.py - Hybrid MPI + thread-parallel 2D IMC wrapper.

This module mirrors IMC2D high-level APIs for geometry='xy' and geometry='rz':
- init_simulation
- step
- run_simulation

Parallelization strategy:
- Particles are split across ranks (each rank owns a disjoint subset).
- Material fields are identical on all ranks.
- Per-step deposited energy and scalar intensity are reduced with Allreduce.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import IMC2D as imc2d


@dataclass
class SimulationState2DMPI:
    weights: np.ndarray
    dir1: np.ndarray
    dir2: np.ndarray
    times: np.ndarray
    pos1: np.ndarray
    pos2: np.ndarray
    cell_i: np.ndarray
    cell_j: np.ndarray

    internal_energy: np.ndarray
    temperature: np.ndarray
    radiation_temperature: np.ndarray

    time: float
    previous_total_energy: float

    comm: object
    rank: int
    size: int
    count: int = 0


def _local_count(global_n, size):
    if global_n <= 0:
        return 0
    return max(1, int(global_n) // int(size))


def init_simulation(
    Ntarget,
    Tinit,
    Tr_init,
    edges1,
    edges2,
    eos,
    inv_eos,
    Ntarget_ic=None,
    geometry="xy",
    comm=None,
):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nx, ny = imc2d._shape_from_edges(edges1, edges2)
    volumes = imc2d._cell_volumes(edges1, edges2, geometry)

    Ntarget_local = _local_count(Ntarget, size)
    if Ntarget_ic is None:
        N_ic_local = Ntarget_local
    else:
        N_ic_local = _local_count(Ntarget_ic, size)

    internal_energy = eos(Tinit)
    temperature = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    if geometry == "xy":
        p = imc2d._equilibrium_sample_xy(N_ic_local, Tr_init, edges1, edges2)
    elif geometry == "rz":
        p = imc2d._equilibrium_sample_rz(N_ic_local, Tr_init, edges1, edges2)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    weights, dir1, dir2, times, pos1, pos2, cell_i, cell_j = p

    flat = imc2d._flatten_index(cell_i, cell_j, nx)
    local_rad = np.bincount(flat, weights=weights, minlength=nx * ny).reshape(nx, ny)
    global_rad = np.zeros((nx, ny))
    comm.Allreduce(local_rad, global_rad, op=MPI.SUM)
    radiation_temperature = (global_rad / volumes / imc2d.__a) ** 0.25

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad = float(comm.allreduce(np.sum(weights), op=MPI.SUM))
    prev_total = total_internal + total_rad
    total_particles = int(comm.allreduce(len(weights), op=MPI.SUM))

    if rank == 0:
        print(
            "Time",
            "N",
            "Total Energy",
            "Total Internal Energy",
            "Total Radiation Energy",
            "Boundary Emission",
            "Lost Energy",
            sep="\t",
        )
        print("=" * 111)
        print(
            "{:.6f}".format(0.0),
            total_particles,
            "{:.6f}".format(prev_total),
            "{:.6f}".format(total_internal),
            "{:.6f}".format(total_rad),
            "{:.6f}".format(0.0),
            "{:.6f}".format(0.0),
            sep="\t",
        )

    return SimulationState2DMPI(
        weights=weights,
        dir1=dir1,
        dir2=dir2,
        times=times,
        pos1=pos1,
        pos2=pos2,
        cell_i=cell_i,
        cell_j=cell_j,
        internal_energy=internal_energy,
        temperature=temperature,
        radiation_temperature=radiation_temperature,
        time=0.0,
        previous_total_energy=prev_total,
        comm=comm,
        rank=rank,
        size=size,
        count=0,
    )


def step(
    state,
    Ntarget,
    Nboundary,
    Nsource,
    Nmax,
    T_boundary,
    dt,
    edges1,
    edges2,
    sigma_a_func,
    inv_eos,
    cv,
    source,
    reflect=(False, False, False, False),
    theta=1.0,
    use_scalar_intensity_Tr=True,
    conserve_comb_energy=False,
    geometry="xy",
    rz_linear_source=True,
    max_events_per_particle=100,
):
    comm = state.comm
    rank = state.rank
    size = state.size

    Ntarget_local = _local_count(Ntarget, size)
    Nboundary_local = _local_count(Nboundary, size)
    Nsource_local = _local_count(Nsource, size)
    Nmax_local = _local_count(Nmax, size)

    nx, ny = imc2d._shape_from_edges(edges1, edges2)
    volumes = imc2d._cell_volumes(edges1, edges2, geometry)

    weights = state.weights
    dir1 = state.dir1
    dir2 = state.dir2
    times = state.times
    pos1 = state.pos1
    pos2 = state.pos2
    cell_i = state.cell_i
    cell_j = state.cell_j
    internal_energy = state.internal_energy
    temperature = state.temperature

    sigma_a_true = sigma_a_func(temperature)
    beta = 4.0 * imc2d.__a * temperature**3 / cv(temperature)
    f = 1.0 / (1.0 + theta * beta * sigma_a_true * imc2d.__c * dt)
    f = np.clip(f, 0.0, 1.0)
    sigma_s = sigma_a_true * (1.0 - f)
    sigma_a = sigma_a_true * f

    b_left = imc2d._boundary_temperature_value(T_boundary[0], state.time)
    b_right = imc2d._boundary_temperature_value(T_boundary[1], state.time)
    b_bottom = imc2d._boundary_temperature_value(T_boundary[2], state.time)
    b_top = imc2d._boundary_temperature_value(T_boundary[3], state.time)

    boundary_emission_local = 0.0

    if Nboundary_local > 0:
        if geometry == "xy":
            for side, Tb in (
                ("left", b_left),
                ("right", b_right),
                ("bottom", b_bottom),
                ("top", b_top),
            ):
                s = imc2d._sample_boundary_xy(Nboundary_local, side, Tb, dt, edges1, edges2)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
                ci, cj = imc2d._locate_indices(p1, p2, edges1, edges2)
                weights = np.concatenate((weights, w))
                dir1 = np.concatenate((dir1, d1))
                dir2 = np.concatenate((dir2, d2))
                times = np.concatenate((times, t))
                pos1 = np.concatenate((pos1, p1))
                pos2 = np.concatenate((pos2, p2))
                cell_i = np.concatenate((cell_i, ci))
                cell_j = np.concatenate((cell_j, cj))
                boundary_emission_local += float(np.sum(w))
        else:
            for side, Tb in (
                ("rmin", b_left),
                ("rmax", b_right),
                ("zmin", b_bottom),
                ("zmax", b_top),
            ):
                s = imc2d._sample_boundary_rz(Nboundary_local, side, Tb, dt, edges1, edges2)
                if s is None:
                    continue
                w, d1, d2, t, p1, p2 = s
                ci, cj = imc2d._locate_indices(p1, p2, edges1, edges2)
                valid = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
                if np.any(valid):
                    weights = np.concatenate((weights, w[valid]))
                    dir1 = np.concatenate((dir1, d1[valid]))
                    dir2 = np.concatenate((dir2, d2[valid]))
                    times = np.concatenate((times, t[valid]))
                    pos1 = np.concatenate((pos1, p1[valid]))
                    pos2 = np.concatenate((pos2, p2[valid]))
                    cell_i = np.concatenate((cell_i, ci[valid]))
                    cell_j = np.concatenate((cell_j, cj[valid]))
                    boundary_emission_local += float(np.sum(w[valid]))

    source_emission_local = 0.0
    if Nsource_local > 0 and np.max(source) > 0.0:
        if geometry == "xy":
            s = imc2d._sample_source_xy(Nsource_local, source, dt, edges1, edges2)
        else:
            if rz_linear_source:
                s = imc2d._sample_source_rz_linear(Nsource_local, source, temperature, dt, edges1, edges2)
            else:
                s = imc2d._sample_source_rz_uniform(Nsource_local, source, dt, edges1, edges2)
        w, d1, d2, t, p1, p2, ci, cj = s
        if len(w) > 0:
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))
            source_emission_local = float(np.sum(w))

    emitted_energies = (
        imc2d.__a * imc2d.__c * np.maximum(temperature, 0.0) ** 4 * sigma_a * dt * volumes
    )

    E_emit = float(np.sum(emitted_energies))
    if E_emit > 0.0 and Ntarget_local > 0:
        emiss_power = emitted_energies / (dt * volumes + 1e-300)
        if geometry == "xy":
            p = imc2d._sample_source_xy(Ntarget_local, emiss_power, dt, edges1, edges2)
        else:
            if rz_linear_source:
                p = imc2d._sample_source_rz_linear(
                    Ntarget_local, emiss_power, temperature, dt, edges1, edges2
                )
            else:
                p = imc2d._sample_source_rz_uniform(Ntarget_local, emiss_power, dt, edges1, edges2)

        w, d1, d2, t, p1, p2, ci, cj = p
        if len(w) > 0:
            weights = np.concatenate((weights, w))
            dir1 = np.concatenate((dir1, d1))
            dir2 = np.concatenate((dir2, d2))
            times = np.concatenate((times, t))
            pos1 = np.concatenate((pos1, p1))
            pos2 = np.concatenate((pos2, p2))
            cell_i = np.concatenate((cell_i, ci))
            cell_j = np.concatenate((cell_j, cj))

    geometry_code = imc2d._GEOM_XY if geometry == "xy" else imc2d._GEOM_RZ
    dep_cell_local, si_cell_local, boundary_loss_local = imc2d._transport_particles_2d(
        weights,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        cell_i,
        cell_j,
        edges1,
        edges2,
        sigma_a,
        sigma_s,
        volumes,
        dt,
        reflect,
        max_events_per_particle,
        geometry_code,
    )

    dep_cell_global = np.zeros_like(dep_cell_local)
    si_cell_global = np.zeros_like(si_cell_local)
    comm.Allreduce(dep_cell_local, dep_cell_global, op=MPI.SUM)
    comm.Allreduce(si_cell_local, si_cell_global, op=MPI.SUM)

    internal_energy = internal_energy + dep_cell_global - emitted_energies / volumes
    temperature = inv_eos(internal_energy)

    if use_scalar_intensity_Tr:
        radiation_temperature = (si_cell_global / imc2d.__a / imc2d.__c) ** 0.25
    else:
        valid = weights > 0.0
        flat = imc2d._flatten_index(cell_i[valid], cell_j[valid], nx)
        local_rad = np.bincount(flat, weights=weights[valid], minlength=nx * ny).reshape(nx, ny)
        global_rad = np.zeros((nx, ny))
        comm.Allreduce(local_rad, global_rad, op=MPI.SUM)
        radiation_temperature = (global_rad / volumes / imc2d.__a) ** 0.25

    (
        weights,
        cell_i,
        cell_j,
        dir1,
        dir2,
        times,
        pos1,
        pos2,
        comb_disc,
    ) = imc2d._comb(weights, cell_i, cell_j, dir1, dir2, times, pos1, pos2, Nmax_local, nx, ny)

    if conserve_comb_energy:
        local_disc = comb_disc.reshape(ny, nx).T
        global_disc = np.zeros_like(local_disc)
        comm.Allreduce(local_disc, global_disc, op=MPI.SUM)
        internal_energy = internal_energy + global_disc / volumes
        temperature = inv_eos(internal_energy)

    times = np.zeros_like(times)

    total_internal = float(np.sum(internal_energy * volumes))
    total_rad = float(comm.allreduce(np.sum(weights), op=MPI.SUM))
    total_energy = total_internal + total_rad

    total_boundary_emission = float(comm.allreduce(boundary_emission_local, op=MPI.SUM))
    total_boundary_loss = float(comm.allreduce(boundary_loss_local, op=MPI.SUM))
    total_source_emission = float(comm.allreduce(source_emission_local, op=MPI.SUM))
    total_N = int(comm.allreduce(len(weights), op=MPI.SUM))

    energy_loss = (
        total_energy
        - state.previous_total_energy
        - total_boundary_emission
        + total_boundary_loss
        - total_source_emission
    )

    state.weights = weights
    state.dir1 = dir1
    state.dir2 = dir2
    state.times = times
    state.pos1 = pos1
    state.pos2 = pos2
    state.cell_i = cell_i
    state.cell_j = cell_j

    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
    state.time += dt
    state.previous_total_energy = total_energy
    state.count += 1

    info = {
        "time": state.time,
        "temperature": temperature,
        "radiation_temperature": radiation_temperature,
        "N_particles": total_N,
        "total_energy": total_energy,
        "total_internal_energy": total_internal,
        "total_radiation_energy": total_rad,
        "boundary_emission": total_boundary_emission,
        "boundary_loss": total_boundary_loss,
        "source_emission": total_source_emission,
        "energy_loss": energy_loss,
    }

    if rank == 0:
        pass

    return state, info


def run_simulation(
    Ntarget,
    Nboundary,
    Nsource,
    Nmax,
    Tinit,
    Tr_init,
    T_boundary,
    dt,
    edges1,
    edges2,
    sigma_a_func,
    eos,
    inv_eos,
    cv,
    source,
    final_time,
    reflect=(False, False, False, False),
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=True,
    Ntarget_ic=None,
    conserve_comb_energy=False,
    geometry="xy",
    rz_linear_source=True,
    max_events_per_particle=100,
    comm=None,
):
    state = init_simulation(
        Ntarget,
        Tinit,
        Tr_init,
        edges1,
        edges2,
        eos,
        inv_eos,
        Ntarget_ic=Ntarget_ic,
        geometry=geometry,
        comm=comm,
    )

    t_hist = [0.0]
    Tr_hist = [state.radiation_temperature.copy()]
    T_hist = [state.temperature.copy()]

    while state.time < final_time - 1e-15:
        step_dt = min(dt, final_time - state.time)
        state, info = step(
            state,
            Ntarget,
            Nboundary,
            Nsource,
            Nmax,
            T_boundary,
            step_dt,
            edges1,
            edges2,
            sigma_a_func,
            inv_eos,
            cv,
            source,
            reflect=reflect,
            theta=theta,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            geometry=geometry,
            rz_linear_source=rz_linear_source,
            max_events_per_particle=max_events_per_particle,
        )

        if (state.count - 1) % output_freq == 0 or (final_time - state.time) < 1e-12:
            t_hist.append(info["time"])
            Tr_hist.append(state.radiation_temperature.copy())
            T_hist.append(state.temperature.copy())
            if state.rank == 0:
                print(
                    "{:.6f}".format(info["time"]),
                    info["N_particles"],
                    "{:.6f}".format(info["total_energy"]),
                    "{:.6f}".format(info["total_internal_energy"]),
                    "{:.6f}".format(info["total_radiation_energy"]),
                    "{:.6f}".format(info["boundary_emission"]),
                    "{:.6e}".format(info["energy_loss"]),
                    sep="\t",
                )

    return np.array(t_hist), np.array(Tr_hist), np.array(T_hist)
