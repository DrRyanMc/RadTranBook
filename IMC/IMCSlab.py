import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numba import jit

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
@jit(nopython=True, cache=True)
def move_particles(weights, mus, times, positions,cell_indices, mesh, sigma_a, sigma_s,dt,refl):
    #cell_indices is a list of the indices of the cells that the particles are in
    #mesh is a 2-D array of length I where I is the number of cells, each row is the left and right boundary of the cell
    #assert that all arrays are the same length

    if not(len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)):
        print("Length of weights: ", len(weights))
        print("Length of mus: ", len(mus))
        print("Length of times: ", len(times))
        print("Length of positions: ", len(positions))
        print("Length of cell_indices: ", len(cell_indices))
    assert len(weights) == len(mus) == len(times) == len(positions)
    #initialize array to hold deposited weight
    deposited_weights = np.zeros(len(sigma_a))
    scalar_intensity = np.zeros(len(sigma_a))
    #initialize array to hold new positions
    N = len(weights) #number of particles
    #compute sum of weights in each zone
    zone_weights = np.zeros(len(sigma_a))
    for i in range(N):
        loc = int(cell_indices[i])
        zone_weights[loc] += weights[i]
    
    #weights *= 0.0
    #dxs = mesh[:,1] - mesh[:,0]
    #return zone_weights/dxs, scalar_intensity
    #loop over particles
    for i in range(N):
        distance_to_census = (dt-times[i])*__c
        
        #move particle until it reaches census
        while distance_to_census > 0:
            loc = int(cell_indices[i])
            #check that particle is in the cell
            if positions[i] < mesh[loc][0] or positions[i] > mesh[loc][1]:
                #print out a lot of detail
                print("Particle not in cell")
                print("Particle position: ", positions[i])
                print("Cell position: ", mesh[loc][0], mesh[loc][1])
                print("Cell index: ", loc)
                print("Particle index: ", i)
                print("Particle weight: ", weights[i])
            assert positions[i] >= mesh[loc][0] and positions[i] <= mesh[loc][1], "Particle not in cell position"
            #move particle
            output = move_particle(weights[i], mus[i], positions[i], mesh[loc][0], mesh[loc][1], sigma_a[loc], sigma_s[loc], distance_to_census)
            #update arrays
            weights[i] = output[0]
            mus[i] = output[1]
            positions[i] = output[2]
            #check to see if particle is in a new cell
            if output[3] == 1:
                cell_indices[i] = cell_indices[i] + 1
            elif output[3] == -1:
                cell_indices[i] = cell_indices[i] - 1
            #update deposited weight
            #assert output[4] >= 0, "Negative deposited weight"
            deposited_weights[loc] += output[4]
            #update scalar intensity
            scalar_intensity[loc] += output[5]/dt
            #update distance to census
            distance_to_census = output[6]
            #check to see if particle has exited the slab
            if cell_indices[i] == len(mesh):
                if refl[1]:
                    mus[i] = -mus[i]
                    cell_indices[i] = cell_indices[i] - 1
                else:
                    distance_to_census = 0
            elif cell_indices[i] == -1:
                if refl[0]:
                    mus[i] = -mus[i]
                    cell_indices[i] = cell_indices[i] + 1
                else:
                    distance_to_census = 0
            #if weight < 1e-10, kill particle and add energy to deposited_weights
            if weights[i] < 1e-109:
                loc = int(cell_indices[i])
                if (loc>0) and (loc<len(sigma_a)-1):
                    deposited_weights[loc] += weights[i]/(mesh[loc][1]-mesh[loc][0])
                    scalar_intensity[loc] += weights[i]/dt/sigma_a[loc]/(mesh[loc][1]-mesh[loc][0])
                    weights[i] = 0
                    distance_to_census = 0

    return deposited_weights, scalar_intensity

#simple comb function to control particle number
@jit(nopython=True, cache=True)
def comb(Ntarget, N):
    if (N < Ntarget):
        return np.arange(N)
    spacing = N/Ntarget
    xi = random.random()
    comb_vals = np.arange(0,Ntarget)*spacing+xi
    mask = np.floor(comb_vals).astype("int")
    return mask

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
    weights = np.empty(0)
    positions = np.empty(0)
    cell_indices = np.empty(0).astype("int")
    for i in range(I):
        temp_weights = np.zeros(emitted_per_zone[i]) + energy_per_zone[i]/emitted_per_zone[i]
        weights = np.concatenate((weights,temp_weights))
        temp_positions = np.random.uniform(mesh[i,0],mesh[i,1],emitted_per_zone[i])
        temp_cell_indices = np.zeros(emitted_per_zone[i],dtype="int") + i
        positions = np.concatenate((positions,temp_positions))
        cell_indices = np.concatenate((cell_indices,temp_cell_indices))
    mus = np.random.uniform(-1,1,N)
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
    #loop over cells and sample particles
    weights = []
    mus = []
    times = []
    positions = []
    cell_indices = []
    for i in range(len(emitted_energies)):
        N = int(math.ceil(Ntarget*emitted_energies[i]/total_emission))
        temp_weights = np.zeros(N) + emitted_energies[i]/N
        temp_mus = np.random.uniform(-1,1,N)
        temp_times = np.random.uniform(0,dt,N)
        temp_positions = sample_linear_density(dx[i],slopes[i],Temperatures[i],N)+mesh[i,0] #np.random.uniform(mesh[i,0],mesh[i,1],N) #
        temp_cell_indices = np.zeros(N, dtype="int") + i
        weights.extend(temp_weights)
        mus.extend(temp_mus)
        times.extend(temp_times)
        positions.extend(temp_positions)# + mesh[i,0])
        cell_indices.extend(temp_cell_indices)
        """ old way of sampling
        for j in range(N):
            weights.append(emitted_energies[i]/N)
            mus.append(random.uniform(-1,1))
            times.append(random.uniform(0,dt))
            positions.append(sample_linear_density(dx[i],slopes[i],Temperatures[i],1)[0] + mesh[i,0])
            cell_indices.append(i)"
        """
    weights = np.array(weights)
    mus = np.array(mus)
    times = np.array(times)
    positions = np.array(positions)
    cell_indices = np.array(cell_indices, dtype="int")
    #print("Total emitted energy: ", total_emission, "sum of weights: ", np.sum(weights))
    #assert that the total emitted energy is equal to the sume of the weights
    assert np.allclose(total_emission,np.sum(weights)), "Total emitted energy not equal to sum of weights"
    return weights, mus, times, positions, cell_indices, emitted_energies

@jit(nopython=True, cache=True)
def sample_source(N,source,mesh,dt):
    #set up lists to hold particles
    weights = np.empty(0)
    mus = np.empty(0)
    times = np.empty(0)
    positions = np.empty(0)
    cell_indices = np.empty(0).astype("int")
    dxs = mesh[:,1] - mesh[:,0]
    N_per_zone = np.ceil(N*source*dt*dxs/np.sum(source*dt*dxs)).astype("int")
    #loop over cells
    for i in range(mesh.shape[0]):
        #sample N_per_zone[i] particles from the source
        if N_per_zone[i] > 0:
            temp_weights = np.zeros(N_per_zone[i]) + source[i]*dt*dxs[i]/N_per_zone[i]
            temp_mus = np.random.uniform(-1,1,N_per_zone[i])
            temp_times = np.random.uniform(0,1,N_per_zone[i])
            temp_positions = np.random.uniform(mesh[i,0],mesh[i,1],N_per_zone[i])
            temp_cell_indices = np.zeros(N_per_zone[i],dtype="int") + i
            #add the particles to the appropriate lists
            weights = np.concatenate((weights,temp_weights))
            mus = np.concatenate((mus,temp_mus))
            times = np.concatenate((times,temp_times))
            positions = np.concatenate((positions,temp_positions))
            cell_indices = np.concatenate((cell_indices,temp_cell_indices))
    #make sure all lists are the same length
    assert len(weights) == len(mus) == len(times) == len(positions) == len(cell_indices)
    return weights, mus, times, positions, cell_indices
#function to run simulation
def run_simulation(Ntarget,Nboundary,Nsource,NMax, Tinit,Tr_init, T_boundary, dt, mesh, 
                   sigma_a_func,eos,inv_eos,cv,source, final_time, reflect=(False,False),
                   output_freq=1):
    #T_boundary is a tuple with the left and right boundary temperatures
    I = mesh.shape[0]
    deposited_energies = np.zeros(mesh.shape[0])
    internal_energy = eos(Tinit)
    radiation_temperature = np.zeros(mesh.shape[0])
    temperature = Tinit.copy()
    time = 0
    weights = np.empty(0)
    mus = np.empty(0)
    times = np.empty(0)
    positions = np.empty(0)
    cell_indices = np.empty(0, dtype=int)
    #sample initial particles from equilibrium with the initial temperature
    initial_particles = equilibrium_sample(Ntarget, Tr_init, mesh)
    weights = np.concatenate((weights,initial_particles[0]))
    mus = np.concatenate((mus,initial_particles[1]))
    times = np.concatenate((times,initial_particles[2]))
    positions = np.concatenate((positions,initial_particles[3]))
    cell_indices = np.concatenate((cell_indices,initial_particles[4]))

    dxs = mesh[:,1] - mesh[:,0]
    boundary_loss = 0.0
    boundary_emission = 0.0

    #create variables to hold radiation temperature and material temperature over time
    radiation_temperatures = []
    temperatures = []
    time_values = []

    initial_radiation_temperature = np.zeros(I)
    for i in range(I):
        #mask particles in cell i
        mask = cell_indices == i
        #compute radiation temperature in cell i
        initial_radiation_temperature[i] = (np.sum(weights[mask]/dxs[i])/__a)**0.25
    radiation_temperatures.append(initial_radiation_temperature)
    temperatures.append(Tinit)
    time_values.append(0.0)
    #make sure inv_eos returns temperature using np.all_close
    assert np.allclose(inv_eos(internal_energy),Tinit), "Inverse equation of state not working"
    #print the table header
    print("Time", "N", "Total Energy", "Total Internal Energy", "Total Radiation Energy", "Boundary Emission","Lost Energy", sep='\t')
    print("===============================================================================================================")
    count = 0


    #now print the values at time 0
    total_internal_energy = np.sum(internal_energy*dxs)
    total_radiation_energy = np.sum(weights) #np.sum(a*radiation_temperature**4*dxs)
    total_energy = total_internal_energy + total_radiation_energy
    previous_total_energy = total_energy
    #no emission or loss at time 0
    boundary_emission = 0.
    energy_loss = 0.
    print("{:.6f}".format(time), len(weights), "{:.6f}".format(total_energy), "{:.6f}".format(total_internal_energy),
          "{:.6f}".format(total_radiation_energy), "{:.6f}".format(boundary_emission), "{:.6f}".format(energy_loss), sep='\t')
    while time < final_time:
        if time + dt > final_time:
            dt = final_time - time
        #compute sigma_a
        sigma_a_true = sigma_a_func(temperature)
        #compute beta
        beta = 4*__a*temperature**3/cv(temperature)
        #compute Fleck factor
        f = 1/(1+beta*sigma_a_true*__c*dt)
        #check that each f is between 0 and 1 using np.all
        assert np.all(f >= 0) and np.all(f <= 1), "Fleck factor out of bounds"

        #compute sigma_s
        sigma_s = sigma_a_true*(1-f)
        sigma_a = sigma_a_true*f
        #sample boundary source
        #sample from the left boundary?
        boundary_emission = 0
        if (T_boundary[0] > 0):
            boundary_source = create_boundary(Nboundary,T_boundary[0],dt)
            #add the particles to the appropriate lists
            weights = np.concatenate((weights,boundary_source[0]))
            mus = np.concatenate((mus,boundary_source[1]))
            times = np.concatenate((times,boundary_source[2]))
            positions = np.concatenate((positions,boundary_source[3]))
            cell_indices = np.concatenate((cell_indices,boundary_source[4]))
            boundary_emission += np.sum(boundary_source[0])
            #print(np.sum(boundary_source[0])/c,0.5*a*dt*(T_boundary[0]**4))
        #sample from the right boundary?
        if (T_boundary[1] > 0):
            boundary_source = create_boundary(Nboundary,T_boundary[1],dt)
            #add the particles to the appropriate lists
            weights = np.concatenate((weights,boundary_source[0]))
            mus = np.concatenate((mus,-boundary_source[1]))
            times = np.concatenate((times,boundary_source[2]))
            positions = np.concatenate((positions,mesh[-1,1]))
            cell_indices = np.concatenate((cell_indices,I-1))
            boundary_emission += np.sum(boundary_source[0])
        #sample from fixed source
        source_emission = 0
        if np.max(source) > 0 and Nsource > 0:
            source_particles = sample_source(Nsource,source,mesh,dt)
            #add the particles to the appropriate lists
            weights = np.concatenate((weights,source_particles[0]))
            mus = np.concatenate((mus,source_particles[1]))
            times = np.concatenate((times,source_particles[2]))
            positions = np.concatenate((positions,source_particles[3]))
            cell_indices = np.concatenate((cell_indices,source_particles[4]))
            source_emission = np.sum(source_particles[0])
        #sample from internal source
        internal_source = emitted_particles(Ntarget, temperature, dt, mesh, sigma_a)
        #add the particles to the appropriate lists
        weights = np.concatenate((weights,internal_source[0]))
        mus = np.concatenate((mus,internal_source[1]))
        times = np.concatenate((times,internal_source[2]))
        positions = np.concatenate((positions,internal_source[3]))
        cell_indices = np.concatenate((cell_indices,internal_source[4]))
        #move particles
        deposited, scalar_intensity = move_particles(weights, mus, times, positions, cell_indices, mesh, sigma_a, sigma_s, dt, reflect)
        #print(np.sum(deposited/dxs),np.sum(0.5*a*dt*(T_boundary[0]**4)), sum(scalar_intensity/c),sum(weights)/c)
        #update internal energy
        emitted_energies = internal_source[5]
        #print("sum of emitted_energies: ", np.sum(emitted_energies))
        internal_energy += deposited - emitted_energies/dxs
        #update temperature
        temperature = inv_eos(internal_energy)
        #update radiation temperature   
        radiation_temperature = (scalar_intensity/__a/__c)**.25

        radiation_temperature = np.zeros(I)
        for i in range(I):
            #mask particles in cell i
            mask = cell_indices == i
            #compute radiation temperature in cell i
            radiation_temperature[i] = (np.sum(weights[mask]))
        radiation_temperature /= dxs
        radiation_temperature = (radiation_temperature/__a)**0.25
        #update time
        time += dt
        #clean up particles that left the slab and store their weights in boundary_loss
        #check left boundary
        mask = cell_indices < 0


        #now check right boundary
        mask = cell_indices < 0
        boundary_loss = 0.
        if (reflect[0]): #reflecting boundary
            mus[mask] = -mus[mask]
            cell_indices[mask] = 0
        else:
            boundary_loss = np.sum(weights[mask])
            weights = weights[~mask]
            mus = mus[~mask]
            times = times[~mask]
            positions = positions[~mask]
            cell_indices = cell_indices[~mask]
        #now check right boundary
        mask = cell_indices >= I
        if (reflect[1]): #reflecting boundary
            mus[mask] = -mus[mask]
            cell_indices[mask] = I-1
        else:
            boundary_loss += np.sum(weights[mask])
            weights = weights[~mask]
            mus = mus[~mask]
            times = times[~mask]
            positions = positions[~mask]
            cell_indices = cell_indices[~mask]

        
        #apply particle combing
        if (len(weights) > NMax):
            Ncurr = len(weights)
            mask = comb(NMax, len(weights))
            previous_weights = np.sum(weights)
            lost_weight = np.sum(weights[~mask])
            weights = weights[mask]
            current_weights = np.sum(weights)
            weights *= previous_weights/current_weights
            #check that the resultings weights are the same
            assert np.allclose(np.sum(weights),previous_weights), "Particle combing failed"
            mus = mus[mask]
            times = times[mask]
            positions = positions[mask]
            cell_indices = cell_indices[mask]

        #set all particle times to 0
        times = np.zeros(times.shape)
        total_internal_energy = np.sum(internal_energy*dxs)
        total_radiation_energy = np.sum(weights) #np.sum(a*radiation_temperature**4*dxs)
        total_energy = total_internal_energy + total_radiation_energy
        energy_loss = total_energy-previous_total_energy - boundary_emission + boundary_loss - source_emission
        previous_total_energy = total_energy
        #print("total internal energy from eos: ", np.sum(eos(temperature)*dxs))

        #store the radiation and material temperatures
        if count % output_freq == 0 or (time - final_time) < dt:
            radiation_temperatures.append(radiation_temperature.copy())
            temperatures.append(temperature.copy())
            time_values.append(time)
            #print to the screen a row in a table containing the time, number of particles, the total energy, the total internal energy, and the total radiation energy, the boundary emission, energy_loss with tabs between each value
            #also limit every number to 6 decimal places
            print("{:.6f}".format(time), len(weights), "{:.6f}".format(total_energy), "{:.6f}".format(total_internal_energy), 
                "{:.6f}".format(total_radiation_energy), "{:.6f}".format(boundary_emission), "{:.6e}".format(energy_loss), sep='\t') 

        #increment count
        count += 1      

    #make times, radiation_temperatures and temperatures into numpy arrays
    time_values = np.array(time_values)
    radiation_temperatures = np.array(radiation_temperatures)
    temperatures = np.array(temperatures)
    return time_values,radiation_temperatures, temperatures

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