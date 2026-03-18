"""IMC1DMPI.py - Hybrid MPI + Numba-threads wrapper for IMC1D.

This module mirrors the high-level API in IMC1D.py (init_simulation, step,
run_simulation) but distributes particles across MPI ranks and uses Allreduce
for global material/radiation updates.

Supports both geometry='slab' and geometry='spherical'.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

import IMC1D as imc


@dataclass
class SimulationState:
    """Mutable state passed between time steps (MPI-distributed particles)."""

    weights: np.ndarray
    mus: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    cell_indices: np.ndarray
    internal_energy: np.ndarray
    temperature: np.ndarray
    radiation_temperature: np.ndarray
    time: float
    previous_total_energy: float
    comm: object
    rank: int
    size: int
    count: int = 0


def _local_count(global_count, size):
    """Per-rank particle count for approximately global_count total particles."""
    if global_count <= 0:
        return 0
    return max(1, int(global_count) // int(size))


def init_simulation(
    Ntarget,
    Tinit,
    Tr_init,
    mesh,
    eos,
    inv_eos,
    Ntarget_ic=None,
    geometry="slab",
    comm=None,
):
    """Initialize per-rank particle arrays and shared material state."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    I = mesh.shape[0]
    volumes = imc._cell_volumes(mesh, geometry)

    Ntarget_local = _local_count(Ntarget, size)
    if Ntarget_ic is None:
        N_ic_local = Ntarget_local
    else:
        N_ic_local = _local_count(Ntarget_ic, size)

    internal_energy = eos(Tinit)
    temperature = Tinit.copy()
    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse EOS failed"

    if geometry == "slab":
        p = imc.equilibrium_sample(N_ic_local, Tr_init, mesh)
    else:
        p = imc.equilibrium_sample_spherical(N_ic_local, Tr_init, mesh)

    weights, mus, times, positions, cell_indices = p

    local_rad_weights = np.bincount(cell_indices, weights=weights, minlength=I)
    global_rad_weights = np.zeros(I)
    comm.Allreduce(local_rad_weights, global_rad_weights, op=MPI.SUM)
    radiation_temperature = (global_rad_weights / volumes / imc.__a) ** 0.25

    total_internal_energy = np.sum(internal_energy * volumes)
    total_radiation_energy = float(comm.allreduce(np.sum(weights), op=MPI.SUM))
    previous_total_energy = total_internal_energy + total_radiation_energy
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
            "{:.6f}".format(previous_total_energy),
            "{:.6f}".format(total_internal_energy),
            "{:.6f}".format(total_radiation_energy),
            "{:.6f}".format(0.0),
            "{:.6f}".format(0.0),
            sep="\t",
        )

    return SimulationState(
        weights=weights,
        mus=mus,
        times=times,
        positions=positions,
        cell_indices=cell_indices,
        internal_energy=internal_energy,
        temperature=temperature,
        radiation_temperature=radiation_temperature,
        time=0.0,
        previous_total_energy=previous_total_energy,
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
    NMax,
    T_boundary,
    dt,
    mesh,
    sigma_a_func,
    inv_eos,
    cv,
    source,
    reflect=(False, False),
    theta=1.0,
    use_scalar_intensity_Tr=True,
    conserve_comb_energy=False,
    geometry="slab",
):
    """Advance the simulation by one time step (MPI + Numba threads)."""
    comm = state.comm
    size = state.size

    Ntarget_local = _local_count(Ntarget, size)
    Nboundary_local = _local_count(Nboundary, size)
    Nsource_local = _local_count(Nsource, size)
    NMax_local = _local_count(NMax, size)

    I = mesh.shape[0]
    volumes = imc._cell_volumes(mesh, geometry)

    weights = state.weights
    mus = state.mus
    times = state.times
    positions = state.positions
    cell_indices = state.cell_indices
    internal_energy = state.internal_energy
    temperature = state.temperature

    sigma_a_true = sigma_a_func(temperature)
    beta = 4.0 * imc.__a * temperature**3 / cv(temperature)
    f = 1.0 / (1.0 + theta * beta * sigma_a_true * imc.__c * dt)
    assert np.all(f >= 0.0) and np.all(f <= 1.0), "Fleck factor out of bounds"
    sigma_s = sigma_a_true * (1.0 - f)
    sigma_a = sigma_a_true * f

    T_left = T_boundary[0](state.time) if callable(T_boundary[0]) else T_boundary[0]
    T_right = T_boundary[1](state.time) if callable(T_boundary[1]) else T_boundary[1]

    boundary_emission_local = 0.0

    if T_left > 0.0 and Nboundary_local > 0:
        if geometry == "slab":
            bs = imc.create_boundary(Nboundary_local, T_left, dt)
        else:
            bs = imc.create_boundary_spherical(
                Nboundary_local, T_left, dt, mesh[0, 0], outward=True
            )
        weights = np.concatenate((weights, bs[0]))
        mus = np.concatenate((mus, bs[1]))
        times = np.concatenate((times, bs[2]))
        positions = np.concatenate((positions, bs[3]))
        cell_indices = np.concatenate((cell_indices, bs[4]))
        boundary_emission_local += float(np.sum(bs[0]))

    if T_right > 0.0 and Nboundary_local > 0:
        if geometry == "slab":
            bs = imc.create_boundary(Nboundary_local, T_right, dt)
            bs_pos = np.full(len(bs[0]), mesh[-1, 1])
            bs_idx = np.full(len(bs[0]), I - 1, dtype=int)
            bs_mu = -bs[1]
        else:
            bs = imc.create_boundary_spherical(
                Nboundary_local, T_right, dt, mesh[-1, 1], outward=False
            )
            bs_pos = np.full(len(bs[0]), mesh[-1, 1])
            bs_idx = np.full(len(bs[0]), I - 1, dtype=int)
            bs_mu = bs[1]

        weights = np.concatenate((weights, bs[0]))
        mus = np.concatenate((mus, bs_mu))
        times = np.concatenate((times, bs[2]))
        positions = np.concatenate((positions, bs_pos))
        cell_indices = np.concatenate((cell_indices, bs_idx))
        boundary_emission_local += float(np.sum(bs[0]))

    source_emission_local = 0.0
    if np.max(source) > 0.0 and Nsource_local > 0:
        if geometry == "slab":
            sp = imc.sample_source(Nsource_local, source, mesh, dt)
        else:
            sp = imc.sample_source_spherical(Nsource_local, source, mesh, dt)

        weights = np.concatenate((weights, sp[0]))
        mus = np.concatenate((mus, sp[1]))
        times = np.concatenate((times, sp[2]))
        positions = np.concatenate((positions, sp[3]))
        cell_indices = np.concatenate((cell_indices, sp[4]))
        source_emission_local = float(np.sum(sp[0]))

    if Ntarget_local > 0:
        if geometry == "slab":
            internal_source = imc.emitted_particles(
                Ntarget_local, temperature, dt, mesh, sigma_a
            )
        else:
            internal_source = imc.emitted_particles_spherical(
                Ntarget_local, temperature, dt, mesh, sigma_a
            )

        weights = np.concatenate((weights, internal_source[0]))
        mus = np.concatenate((mus, internal_source[1]))
        times = np.concatenate((times, internal_source[2]))
        positions = np.concatenate((positions, internal_source[3]))
        cell_indices = np.concatenate((cell_indices, internal_source[4]))
        emitted_energies = internal_source[5]
    else:
        emitted_energies = np.zeros(I)

    weight_floor = 1e-10 * np.sum(weights) / max(len(weights), 1)
    if geometry == "slab":
        deposited, scalar_intensity = imc.move_particles(
            weights,
            mus,
            times,
            positions,
            cell_indices,
            mesh,
            sigma_a,
            sigma_s,
            dt,
            reflect,
            weight_floor,
        )
    else:
        deposited, scalar_intensity = imc.move_particles_spherical(
            weights,
            mus,
            times,
            positions,
            cell_indices,
            mesh,
            sigma_a,
            sigma_s,
            dt,
            reflect,
            weight_floor,
        )

    global_deposited = np.zeros_like(deposited)
    global_scalar_intensity = np.zeros_like(scalar_intensity)
    comm.Allreduce(deposited, global_deposited, op=MPI.SUM)
    comm.Allreduce(scalar_intensity, global_scalar_intensity, op=MPI.SUM)

    internal_energy = internal_energy + global_deposited - emitted_energies / volumes
    temperature = inv_eos(internal_energy)

    new_time = state.time + dt
    if use_scalar_intensity_Tr:
        radiation_temperature = (global_scalar_intensity / imc.__a / imc.__c) ** 0.25
    else:
        valid = (cell_indices >= 0) & (cell_indices < I)
        local_weights = np.bincount(
            cell_indices[valid], weights=weights[valid], minlength=I
        )
        global_weights = np.zeros(I)
        comm.Allreduce(local_weights, global_weights, op=MPI.SUM)
        radiation_temperature = (global_weights / volumes / imc.__a) ** 0.25

    boundary_loss_local = 0.0

    mask = cell_indices < 0
    if reflect[0]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = 0
    else:
        boundary_loss_local += float(np.sum(weights[mask]))
        keep = ~mask
        weights = weights[keep]
        mus = mus[keep]
        times = times[keep]
        positions = positions[keep]
        cell_indices = cell_indices[keep]

    mask = cell_indices >= I
    if reflect[1]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = I - 1
    else:
        boundary_loss_local += float(np.sum(weights[mask]))
        keep = ~mask
        weights = weights[keep]
        mus = mus[keep]
        times = times[keep]
        positions = positions[keep]
        cell_indices = cell_indices[keep]

    weights, cell_indices, mus, times, positions, comb_discrepancy = imc.comb(
        weights, cell_indices, mus, times, positions, NMax_local, I
    )

    if conserve_comb_energy:
        global_comb_discrepancy = np.zeros(I)
        comm.Allreduce(comb_discrepancy, global_comb_discrepancy, op=MPI.SUM)
        internal_energy = internal_energy + global_comb_discrepancy / volumes
        temperature = inv_eos(internal_energy)

    times = np.zeros(times.shape)

    total_internal_energy = np.sum(internal_energy * volumes)
    total_radiation_energy = float(comm.allreduce(np.sum(weights), op=MPI.SUM))

    total_boundary_emission = float(comm.allreduce(boundary_emission_local, op=MPI.SUM))
    total_boundary_loss = float(comm.allreduce(boundary_loss_local, op=MPI.SUM))
    total_source_emission = float(comm.allreduce(source_emission_local, op=MPI.SUM))
    total_N = int(comm.allreduce(len(weights), op=MPI.SUM))

    total_energy = total_internal_energy + total_radiation_energy
    energy_loss = (
        total_energy
        - state.previous_total_energy
        - total_boundary_emission
        + total_boundary_loss
        - total_source_emission
    )

    state.weights = weights
    state.mus = mus
    state.times = times
    state.positions = positions
    state.cell_indices = cell_indices
    state.internal_energy = internal_energy
    state.temperature = temperature
    state.radiation_temperature = radiation_temperature
    state.time = new_time
    state.previous_total_energy = total_energy
    state.count += 1

    info = {
        "time": new_time,
        "radiation_temperature": radiation_temperature,
        "temperature": temperature,
        "N_particles": total_N,
        "total_energy": total_energy,
        "total_internal_energy": total_internal_energy,
        "total_radiation_energy": total_radiation_energy,
        "boundary_emission": total_boundary_emission,
        "boundary_loss": total_boundary_loss,
        "source_emission": total_source_emission,
        "energy_loss": energy_loss,
    }
    return state, info


def run_simulation(
    Ntarget,
    Nboundary,
    Nsource,
    NMax,
    Tinit,
    Tr_init,
    T_boundary,
    dt,
    mesh,
    sigma_a_func,
    eos,
    inv_eos,
    cv,
    source,
    final_time,
    reflect=(False, False),
    output_freq=1,
    theta=1.0,
    use_scalar_intensity_Tr=True,
    Ntarget_ic=None,
    conserve_comb_energy=False,
    geometry="slab",
):
    """Run full simulation; returns history arrays on all ranks."""
    state = init_simulation(
        Ntarget,
        Tinit,
        Tr_init,
        mesh,
        eos,
        inv_eos,
        Ntarget_ic=Ntarget_ic,
        geometry=geometry,
    )

    radiation_temperatures = [state.radiation_temperature.copy()]
    temperatures = [state.temperature.copy()]
    time_values = [0.0]

    while state.time < final_time:
        step_dt = min(dt, final_time - state.time)
        state, info = step(
            state,
            Ntarget,
            Nboundary,
            Nsource,
            NMax,
            T_boundary,
            step_dt,
            mesh,
            sigma_a_func,
            inv_eos,
            cv,
            source,
            reflect,
            theta=theta,
            use_scalar_intensity_Tr=use_scalar_intensity_Tr,
            conserve_comb_energy=conserve_comb_energy,
            geometry=geometry,
        )

        if (state.count - 1) % output_freq == 0 or (info["time"] - final_time) < step_dt:
            radiation_temperatures.append(state.radiation_temperature.copy())
            temperatures.append(state.temperature.copy())
            time_values.append(info["time"])
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

    return np.array(time_values), np.array(radiation_temperatures), np.array(temperatures)
