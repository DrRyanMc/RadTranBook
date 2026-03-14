import numpy as np
import matplotlib.pyplot as plt
import math
import random
from dataclasses import dataclass
from numba import jit, prange, get_thread_id, get_num_threads

__c = 29.98 #cm/ns
__a = 0.01372

#this function uses numba and takes a particle and moves it to the next cell or the end of time step
@jit(nopython=True, cache=True)
def move_particle(weight, mu, position, cell_l, cell_r,sigma_a, sigma_s, distance_to_census):
    dx = cell_r - cell_l
    if mu>0:
        distance_to_boundary = (cell_r - position)/mu
    else:
        distance_to_boundary = (cell_l - position)/mu
    #sample distance to next scatter
    if (sigma_s > 1e-10):
        distance_to_scatter = -math.log(1-random.random())/sigma_s
    else:
        distance_to_scatter = 1e10
    if (sigma_s*dx > 10000):
        distance_to_scatter = -math.log(1-random.random())/(10000/dx)
    
    #find which distance is smaller
    distance_to_next_event = min(distance_to_boundary, distance_to_scatter, distance_to_census)
    #assert positive distance
    assert distance_to_next_event > 0, "Negative distance"
    dx = cell_r - cell_l
    #variable to hold where particle moves, -1 to left, 0 to census, 1 to right
    new_location = 0
    #which event is next
    #move particle to the collision site
    position = position + mu*distance_to_next_event
    #check if the position is close to the boundary
    if math.fabs(distance_to_boundary - distance_to_next_event) < 1e-10:
        #move particle to the boundary
        if mu>0:
            assert(np.isclose(position,cell_r)), "Position not close to boundary"
            position = cell_r
            new_location = 1
        else:
            assert(np.isclose(position,cell_l)), "Position not close to boundary"
            position = cell_l
            new_location = -1
    elif distance_to_next_event == distance_to_scatter:
        #sample new mu
        mu = random.uniform(-1,1)
    else:
        #census
        True
    #check that particle is in the cell
    if position < cell_l or position > cell_r:
        #print out a lot of detail
        print("Particle not in cell - after move")
        print("Particle position: ", position)
        print("Cell position: ", cell_l, cell_r)
        print("Particle weight: ", weight)
    assert position >= cell_l and position <= cell_r, "Particle not in cell position"
    #update distance to census
    distance_to_census = distance_to_census - distance_to_next_event
    weight_factor =  math.exp(-sigma_a*distance_to_next_event)
    #compute deposited weight, this is the integral of the weight over the distance traveled
    if (sigma_a > 1e-10):
        deposited_intensity = weight*(1-weight_factor)/sigma_a/dx
    else:
        deposited_intensity = weight*distance_to_next_event/dx
    deposited_weight = deposited_intensity*sigma_a
    weight = weight*weight_factor

    
    return weight, mu, position, new_location, deposited_weight, deposited_intensity, distance_to_census
    
#this function loops over all particles and moves them to the next cell or census
@jit(nopython=True, parallel=True)
def move_particles(weights, mus, times, positions, cell_indices, mesh, sigma_a, sigma_s, dt, refl):
    N = len(weights)
    n_cells = len(sigma_a)
    # Per-thread accumulation arrays avoid race conditions on deposition
    n_threads = get_num_threads()
    dep_threads = np.zeros((n_threads, n_cells))
    si_threads  = np.zeros((n_threads, n_cells))

    for i in prange(N):
        tid = get_thread_id()
        distance_to_census = (dt - times[i]) * __c

        #move particle until it reaches census
        while distance_to_census > 0:
            loc = int(cell_indices[i])
            #check that particle is in the cell
            if positions[i] < mesh[loc][0] or positions[i] > mesh[loc][1]:
                print("Particle not in cell")
                print("Particle position: ", positions[i])
                print("Cell position: ", mesh[loc][0], mesh[loc][1])
                print("Cell index: ", loc)
                print("Particle index: ", i)
                print("Particle weight: ", weights[i])
            output = move_particle(weights[i], mus[i], positions[i], mesh[loc][0], mesh[loc][1], sigma_a[loc], sigma_s[loc], distance_to_census)
            weights[i]   = output[0]
            mus[i]       = output[1]
            positions[i] = output[2]
            if output[3] == 1:
                cell_indices[i] = cell_indices[i] + 1
            elif output[3] == -1:
                cell_indices[i] = cell_indices[i] - 1
            dep_threads[tid, loc] += output[4]
            si_threads[tid, loc]  += output[5] / dt
            distance_to_census = output[6]
            if cell_indices[i] == len(mesh):
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] = cell_indices[i] - 1
                else:
                    distance_to_census = 0.0
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] = cell_indices[i] + 1
                else:
                    distance_to_census = 0.0
            if weights[i] < 1e-14:
                loc = int(cell_indices[i])
                if (loc > 0) and (loc < len(sigma_a) - 1):
                    dep_threads[tid, loc] += weights[i] / (mesh[loc][1] - mesh[loc][0])
                    si_threads[tid, loc]  += weights[i] / dt / sigma_a[loc] / (mesh[loc][1] - mesh[loc][0])
                weights[i] = 0.0
                distance_to_census = 0.0

    deposited_weights = dep_threads.sum(axis=0)
    scalar_intensity  = si_threads.sum(axis=0)
    return deposited_weights, scalar_intensity

def comb(weights, cell_indices, mus, times, positions, Ntarget, n_cells):
    """
    Per-cell stochastic comb 
    For each particle: numcomb = floor(w / ew_target(cell) + xi), where
    ew_target(cell) = ecen(cell) / ncen_desired(cell) and ncen_desired is
    proportional to each cell's energy fraction of the global target.
    """
    ecen = np.bincount(cell_indices, weights=weights, minlength=n_cells)
    total_ecen = np.sum(ecen)
    if total_ecen == 0:
        return weights, cell_indices, mus, times, positions, np.zeros(n_cells)

    # Desired particles per cell proportional to energy; at least 1 if cell has energy
    ncen_desired = np.where(ecen > 0,
                            np.maximum(1, np.round(Ntarget * ecen / total_ecen).astype(int)),
                            0)

    # Target energy weight per cell
    ew_cen = np.where(ncen_desired > 0, ecen / ncen_desired, 0.0)

    # Stochastic rounding per particle
    new_weights      = []
    new_mus          = []
    new_times        = []
    new_positions    = []
    new_cell_indices = []

    for i in range(len(weights)):
        cell      = int(cell_indices[i])
        ew_target = ew_cen[cell]
        if ew_target <= 0:
            continue
        numcomb = int(weights[i] / ew_target + random.random())
        for _ in range(numcomb):
            new_weights.append(ew_target)
            new_mus.append(mus[i])
            new_times.append(times[i])
            new_positions.append(positions[i])
            new_cell_indices.append(cell)

    new_weights_arr      = np.array(new_weights)
    new_cell_indices_arr = np.array(new_cell_indices, dtype=int)
    if len(new_weights_arr) > 0:
        ecen_after = np.bincount(new_cell_indices_arr, weights=new_weights_arr, minlength=n_cells)
    else:
        ecen_after = np.zeros(n_cells)
    energy_discrepancy = ecen - ecen_after  # positive = energy removed from radiation by combing
    return (new_weights_arr, new_cell_indices_arr,
            np.array(new_mus), np.array(new_times), np.array(new_positions),
            energy_discrepancy)

#function for sampling surface source
@jit(nopython=True, cache=True)
def create_boundary(N,T,dt):
    assert (N > 0)
    total_emission = __a*__c*T**4/4*dt
    weights = np.zeros(N)+total_emission/N
    mus = np.sqrt(np.random.uniform(0,1,N))
    times = np.random.uniform(0,dt,N)
    positions = np.zeros(N)+1e-8
    cell_indices = np.zeros(N,dtype="int")
    return weights, mus, times, positions, cell_indices

@jit(nopython=True, cache=True)
def sample_linear_density(dx, s, T, N, adjust_slope=True):
    """
    Samples N points from a probability density function P(x) ~ s*x + (a - s*dx/2),
    ensuring that density is maximum at f(dx/2) = a and follows slope s.

    Parameters:
        dx (float): Size of the cell (range [0, dx]).
        s (float): Desired slope of the density function.
        N (int): Number of samples to generate.
        adjust_slope (bool): Whether to automatically adjust invalid slopes.

    Returns:
        np.array: Array of sampled positions.
    """
    # Compute valid slope limits
    if (math.fabs(s) < 1e-6):
        return np.random.uniform(0, dx, N)
    s_min = -2 *T/ dx**2
    s_max = 2 *T/ dx**2

    # Validate or adjust the slope
    if s < s_min or s > s_max:
        if adjust_slope:
            s = max(s_min, min(s, s_max))  # Clamp s within valid range
            print(f"Warning: Slope adjusted to {s} to stay within valid limits.")
        else:
            raise ValueError(f"Invalid slope {s}. Must be within [{s_min}, {s_max}].")

    # Compute the points
    xis = np.random.uniform(0, 1, N)
    #assert sqrt is positive
    assert (s/(2*T) - 1/dx)**2 + (2*s)/(T*dx) >= 0, "{T} {dx} {s} {xis} gives negative square root"
    x_samples = (-2*T + s*dx + 2*T*dx*np.sqrt((s/(2.*T) - 1/dx)**2 + (2*s*xis)/(T*dx)))/(2.*s)
    
    return x_samples
#function for sampling during time step
@jit(nopython=True, cache=True)
def equilibrium_sample(N, T, mesh):
    #T is the temperature of the cell
    #The energy density in each cell needs to be a*T**4
    #Therefore, each particle will be born with energy (a*T**4)/sum(a*T**4)*N
    #The particles will be born with a random position in the cell
    #The particles will be born with a random direction
    #The particles will be born with time 0
    #The particles will be born with weight (a*T**4)/sum(a*T**4)*N/N
    #The particles will be born with cell index i
    I = int(mesh.shape[0])
    dx = mesh[:,1] - mesh[:,0]
    total_emitted = np.sum(__a*T**4*dx)
    energy_per_zone = __a*T**4*dx
    emitted_per_zone = np.ceil(energy_per_zone/total_emitted*N).astype("int")
    N = np.sum(emitted_per_zone)
    weights      = np.empty(N)
    positions    = np.empty(N)
    cell_indices = np.empty(N, dtype=np.int64)
    offset = 0
    for i in range(I):
        n = emitted_per_zone[i]
        weights[offset:offset+n]      = energy_per_zone[i] / n
        positions[offset:offset+n]    = np.random.uniform(mesh[i, 0], mesh[i, 1], n)
        cell_indices[offset:offset+n] = i
        offset += n
    mus   = np.random.uniform(-1, 1, N)
    times = np.zeros(N)
    #make sure all lists are the same length
    #print out information if they are not
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices

@jit(nopython=True, cache=True)
def emitted_particles(Ntarget, Temperatures, dt, mesh, sigma_a):
    #compute slopes
    slopes = np.zeros(Temperatures.shape)
    dx = mesh[:,1] - mesh[:,0]

    
    slopes[1:-1] = (Temperatures[2:] - Temperatures[:-2])/(dx[2:]+dx[:-2])*2
    slopes[0] = (Temperatures[1] - Temperatures[0])/(dx[1] + dx[0])*2
    slopes[-1] = (Temperatures[-1] - Temperatures[-2])/(dx[-1] + dx[-2])*2
    #check that the interpolants are non-negative at the cell boundaries
    #evaluate T(x) = s*x + (a - s*dx/2) at the cell boundaries
    #if the interpolant is negative, set the slope to 0
    #do it withouth a for loop
    left_vals = slopes*0 + (Temperatures - slopes*dx/2)
    right_vals = slopes*dx + (Temperatures - slopes*dx/2)
    mask = left_vals < 0
    slopes[mask] = 0
    mask = right_vals < 0
    slopes[mask] = 0
    
    #assert that all temperatures are positive
    assert np.all(Temperatures > 0), "Negative temperature"
    #compute emitted energy
    emitted_energies = (__a*__c*Temperatures**4)*sigma_a*dt*dx
    total_emission = np.sum(emitted_energies)
    # Pre-compute per-cell counts to avoid reflected-list overhead
    Ns = np.empty(len(emitted_energies), dtype=np.int64)
    for i in range(len(emitted_energies)):
        Ns[i] = int(math.ceil(Ntarget * emitted_energies[i] / total_emission))
    N_total      = np.sum(Ns)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(len(emitted_energies)):
        n = Ns[i]
        weights[offset:offset+n]      = emitted_energies[i] / n
        mus[offset:offset+n]          = np.random.uniform(-1, 1, n)
        times[offset:offset+n]        = np.random.uniform(0, dt, n)
        positions[offset:offset+n]    = sample_linear_density(dx[i], slopes[i], Temperatures[i], n) + mesh[i, 0]
        cell_indices[offset:offset+n] = i
        offset += n
    #print("Total emitted energy: ", total_emission, "sum of weights: ", np.sum(weights))
    #assert that the total emitted energy is equal to the sume of the weights
    assert np.allclose(total_emission,np.sum(weights)), "Total emitted energy not equal to sum of weights"
    return weights, mus, times, positions, cell_indices, emitted_energies

@jit(nopython=True, cache=True)
def sample_source(N,source,mesh,dt):
    dxs = mesh[:,1] - mesh[:,0]
    N_per_zone   = np.ceil(N*source*dt*dxs/np.sum(source*dt*dxs)).astype(np.int64)
    N_total      = np.sum(N_per_zone)
    weights      = np.empty(N_total)
    mus          = np.empty(N_total)
    times        = np.empty(N_total)
    positions    = np.empty(N_total)
    cell_indices = np.empty(N_total, dtype=np.int64)
    offset = 0
    for i in range(mesh.shape[0]):
        if N_per_zone[i] > 0:
            n = N_per_zone[i]
            weights[offset:offset+n]      = source[i]*dt*dxs[i]/n
            mus[offset:offset+n]          = np.random.uniform(-1, 1, n)
            times[offset:offset+n]        = np.random.uniform(0, 1, n)
            positions[offset:offset+n]    = np.random.uniform(mesh[i,0], mesh[i,1], n)
            cell_indices[offset:offset+n] = i
            offset += n
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices

@dataclass
class SimulationState:
    """Mutable state passed between time steps."""
    weights:               np.ndarray
    mus:                   np.ndarray
    times:                 np.ndarray
    positions:             np.ndarray
    cell_indices:          np.ndarray
    internal_energy:       np.ndarray
    temperature:           np.ndarray
    radiation_temperature: np.ndarray
    time:                  float
    previous_total_energy: float
    count:                 int = 0


def init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos, Ntarget_ic=None):
    """Initialise particle arrays and material state; return a SimulationState at t=0.

    Ntarget_ic independently controls the number of particles used for the initial
    equilibrium sample.  Defaults to Ntarget when not specified.
    """
    I    = mesh.shape[0]
    dxs  = mesh[:, 1] - mesh[:, 0]
    internal_energy = eos(Tinit)
    temperature     = Tinit.copy()

    assert np.allclose(inv_eos(internal_energy), Tinit), "Inverse equation of state not working"

    # Sample initial particles from equilibrium
    N_ic = Ntarget_ic if Ntarget_ic is not None else Ntarget
    p            = equilibrium_sample(N_ic, Tr_init, mesh)
    weights      = p[0]
    mus          = p[1]
    times        = p[2]
    positions    = p[3]
    cell_indices = p[4]

    # Compute initial radiation temperature
    radiation_temperature = (np.bincount(cell_indices, weights=weights, minlength=I) / dxs / __a) ** 0.25

    total_internal_energy  = np.sum(internal_energy * dxs)
    total_radiation_energy = np.sum(weights)
    previous_total_energy  = total_internal_energy + total_radiation_energy

    # Print header and t=0 row
    print("Time", "N", "Total Energy", "Total Internal Energy", "Total Radiation Energy",
          "Boundary Emission", "Lost Energy", sep='\t')
    print("===============================================================================================================")
    print("{:.6f}".format(0.0), len(weights),
          "{:.6f}".format(previous_total_energy),
          "{:.6f}".format(total_internal_energy),
          "{:.6f}".format(total_radiation_energy),
          "{:.6f}".format(0.0), "{:.6f}".format(0.0), sep='\t')

    return SimulationState(
        weights=weights, mus=mus, times=times, positions=positions,
        cell_indices=cell_indices, internal_energy=internal_energy,
        temperature=temperature, radiation_temperature=radiation_temperature,
        time=0.0, previous_total_energy=previous_total_energy, count=0,
    )


def step(state, Ntarget, Nboundary, Nsource, NMax, T_boundary, dt, mesh,
         sigma_a_func, inv_eos, cv, source, reflect=(False, False), theta=1.0,
         use_scalar_intensity_Tr=True, conserve_comb_energy=False):
    """Advance the simulation by one time step dt.

    theta is the implicitness factor (default 1 = fully implicit).  It scales
    the Fleck-factor denominator: f = 1 / (1 + theta*beta*sigma_a*c*dt).
    theta=1 recovers standard IMC; theta=0 gives an explicit update.

    use_scalar_intensity_Tr (default True): estimate radiation temperature from
    the time-averaged scalar intensity returned by move_particles,
    T_r = (scalar_intensity / (a*c))^0.25.  If False, use the weight bincount
    of surviving particles instead.

    Mutates *state* in-place and also returns it for convenience, along with a
    diagnostics dict containing: time, radiation_temperature, temperature,
    N_particles, total_energy, total_internal_energy, total_radiation_energy,
    boundary_emission, boundary_loss, source_emission, energy_loss.
    """
    I   = mesh.shape[0]
    dxs = mesh[:, 1] - mesh[:, 0]

    weights      = state.weights
    mus          = state.mus
    times        = state.times
    positions    = state.positions
    cell_indices = state.cell_indices
    internal_energy = state.internal_energy
    temperature     = state.temperature

    # --- Fleck-factor cross sections ---
    sigma_a_true = sigma_a_func(temperature)
    beta = 4 * __a * temperature**3 / cv(temperature)
    f    = 1 / (1 + theta * beta * sigma_a_true * __c * dt)
    assert np.all(f >= 0) and np.all(f <= 1), "Fleck factor out of bounds"
    sigma_s = sigma_a_true * (1 - f)
    sigma_a = sigma_a_true * f

    # --- Boundary sources ---
    # Support scalar or callable (time-dependent) boundary temperatures
    T_left  = T_boundary[0](state.time) if callable(T_boundary[0]) else T_boundary[0]
    T_right = T_boundary[1](state.time) if callable(T_boundary[1]) else T_boundary[1]
    boundary_emission = 0.0
    if T_left > 0:
        bs = create_boundary(Nboundary, T_left, dt)
        weights      = np.concatenate((weights,      bs[0]))
        mus          = np.concatenate((mus,           bs[1]))
        times        = np.concatenate((times,         bs[2]))
        positions    = np.concatenate((positions,     bs[3]))
        cell_indices = np.concatenate((cell_indices,  bs[4]))
        boundary_emission += np.sum(bs[0])
    if T_right > 0:
        bs = create_boundary(Nboundary, T_right, dt)
        weights      = np.concatenate((weights,      bs[0]))
        mus          = np.concatenate((mus,          -bs[1]))
        times        = np.concatenate((times,         bs[2]))
        positions    = np.concatenate((positions,     np.full(len(bs[0]), mesh[-1, 1])))
        cell_indices = np.concatenate((cell_indices,  np.full(len(bs[0]), I - 1, dtype=int)))
        boundary_emission += np.sum(bs[0])

    # --- Fixed source ---
    source_emission = 0.0
    if np.max(source) > 0 and Nsource > 0:
        sp = sample_source(Nsource, source, mesh, dt)
        weights      = np.concatenate((weights,      sp[0]))
        mus          = np.concatenate((mus,           sp[1]))
        times        = np.concatenate((times,         sp[2]))
        positions    = np.concatenate((positions,     sp[3]))
        cell_indices = np.concatenate((cell_indices,  sp[4]))
        source_emission = np.sum(sp[0])

    # --- Internal emission ---
    internal_source = emitted_particles(Ntarget, temperature, dt, mesh, sigma_a)
    weights      = np.concatenate((weights,      internal_source[0]))
    mus          = np.concatenate((mus,           internal_source[1]))
    times        = np.concatenate((times,         internal_source[2]))
    positions    = np.concatenate((positions,     internal_source[3]))
    cell_indices = np.concatenate((cell_indices,  internal_source[4]))

    # --- Transport ---
    deposited, scalar_intensity = move_particles(
        weights, mus, times, positions, cell_indices, mesh, sigma_a, sigma_s, dt, reflect)

    # --- Update material state ---
    emitted_energies = internal_source[5]
    internal_energy  = internal_energy + deposited - emitted_energies / dxs
    temperature      = inv_eos(internal_energy)

    # --- Radiation temperature ---
    new_time = state.time + dt
    if use_scalar_intensity_Tr:
        radiation_temperature = (scalar_intensity / __a / __c) ** 0.25
    else:
        valid = (cell_indices >= 0) & (cell_indices < I)
        radiation_temperature = (
            np.bincount(cell_indices[valid], weights=weights[valid], minlength=I) / dxs / __a
        ) ** 0.25

    # --- Remove escaped particles ---
    boundary_loss = 0.0
    mask = cell_indices < 0
    if reflect[0]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = 0
    else:
        boundary_loss = np.sum(weights[mask])
        keep = ~mask
        weights = weights[keep]; mus = mus[keep]; times = times[keep]
        positions = positions[keep]; cell_indices = cell_indices[keep]

    mask = cell_indices >= I
    if reflect[1]:
        mus[mask] = -mus[mask]
        cell_indices[mask] = I - 1
    else:
        boundary_loss += np.sum(weights[mask])
        keep = ~mask
        weights = weights[keep]; mus = mus[keep]; times = times[keep]
        positions = positions[keep]; cell_indices = cell_indices[keep]

    # --- Particle combing ---
    weights, cell_indices, mus, times, positions, comb_discrepancy = comb(
        weights, cell_indices, mus, times, positions, NMax, I)

    # Optional: deposit per-cell comb energy discrepancy back into the material
    if conserve_comb_energy:
        internal_energy = internal_energy + comb_discrepancy / dxs
        temperature     = inv_eos(internal_energy)

    # Reset particle times to start of new step
    times = np.zeros(times.shape)

    # --- Energy diagnostics ---
    total_internal_energy  = np.sum(internal_energy * dxs)
    total_radiation_energy = np.sum(weights)
    total_energy = total_internal_energy + total_radiation_energy
    energy_loss  = (total_energy - state.previous_total_energy
                    - boundary_emission + boundary_loss - source_emission)

    # --- Update state in-place ---
    state.weights               = weights
    state.mus                   = mus
    state.times                 = times
    state.positions             = positions
    state.cell_indices          = cell_indices
    state.internal_energy       = internal_energy
    state.temperature           = temperature
    state.radiation_temperature = radiation_temperature
    state.time                  = new_time
    state.previous_total_energy = total_energy
    state.count                += 1

    info = {
        'time':                   new_time,
        'radiation_temperature':  radiation_temperature,
        'temperature':            temperature,
        'N_particles':            len(weights),
        'total_energy':           total_energy,
        'total_internal_energy':  total_internal_energy,
        'total_radiation_energy': total_radiation_energy,
        'boundary_emission':      boundary_emission,
        'boundary_loss':          boundary_loss,
        'source_emission':        source_emission,
        'energy_loss':            energy_loss,
    }
    return state, info


#function to run simulation
def run_simulation(Ntarget,Nboundary,Nsource,NMax, Tinit,Tr_init, T_boundary, dt, mesh, 
                   sigma_a_func,eos,inv_eos,cv,source, final_time, reflect=(False,False),
                   output_freq=1, theta=1.0, use_scalar_intensity_Tr=True, Ntarget_ic=None,
                   conserve_comb_energy=False):
    state = init_simulation(Ntarget, Tinit, Tr_init, mesh, eos, inv_eos, Ntarget_ic=Ntarget_ic)

    radiation_temperatures = [state.radiation_temperature.copy()]
    temperatures           = [state.temperature.copy()]
    time_values            = [0.0]

    while state.time < final_time:
        step_dt = min(dt, final_time - state.time)
        state, info = step(state, Ntarget, Nboundary, Nsource, NMax, T_boundary, step_dt,
                           mesh, sigma_a_func, inv_eos, cv, source, reflect, theta=theta,
                           use_scalar_intensity_Tr=use_scalar_intensity_Tr,
                           conserve_comb_energy=conserve_comb_energy)

        # output on the requested frequency, and always on the final step
        if (state.count - 1) % output_freq == 0 or (info['time'] - final_time) < step_dt:
            radiation_temperatures.append(state.radiation_temperature.copy())
            temperatures.append(state.temperature.copy())
            time_values.append(info['time'])
            print("{:.6f}".format(info['time']), info['N_particles'],
                  "{:.6f}".format(info['total_energy']),
                  "{:.6f}".format(info['total_internal_energy']),
                  "{:.6f}".format(info['total_radiation_energy']),
                  "{:.6f}".format(info['boundary_emission']),
                  "{:.6e}".format(info['energy_loss']), sep='\t')

    return np.array(time_values), np.array(radiation_temperatures), np.array(temperatures)

if __name__ == "__main__":


    #set parameters
    Ntarget = 20000
    Nboundary = 0
    NSource = 0
    NMax = 100000
    dt = 0.01
    L = 0.1#0.5 #length of slab
    I = 2 #number of cells
    mesh = np.zeros((I,2))
    dx = L/I
    for i in range(I):
        mesh[i] = [i*dx,(i+1)*dx]
    Tinit = np.zeros(I) + 0.5
    #Tinit[0] = 1.0
    Trinit = np.zeros(I) + .5
    T_boundary = (0.0,0)
    sigma_a_f = lambda T: 1e2+0*T ##300*T**-3
    source = np.zeros(I)
    cv_val = 0.01
    eos = lambda T: cv_val*T
    inv_eos = lambda u: (u/cv_val)
    cv = lambda T: cv_val #4*cv_val*T**3
    final_time = dt*10
    #run simulation
    times, radiation_temperatures, temperatures = run_simulation(Ntarget,Nboundary,NSource, NMax,Tinit,Trinit,
                                                                 T_boundary, dt, mesh, sigma_a_f,
                                                                 eos,inv_eos,cv,source, final_time, reflect=(True,True))
    print(temperatures.shape)
    #plot temperatures in cell 0 over time
    plt.plot(times,temperatures[:,0])
    plt.plot(times,radiation_temperatures[:,0])
    plt.plot(times,times*0+0.5)
    plt.legend(["Temperature","Radiation Temperature"])
    plt.show()
    #print("Internal Energy", internal_energy)
    #print("Temperature", temperature)
    #print("Radiation Temperature", radiation_temperature)
    #plot results
    #plt.plot(mesh[:,0],temperature,"o-")
    #plt.plot(mesh[:,0],radiation_temperature,"^--")
    #xplot = np.linspace(0,L,1000)
    #true solution is E2(sigma_a*x)
    from scipy.special import expn
    #plt.plot(xplot,(a*c*expn(2,10*xplot)))
    #plt.legend(["Temperature","Radiation Temperature"])
    #plt.show()