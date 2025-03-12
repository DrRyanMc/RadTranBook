import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numba import jit

c = 29.98 #cm/ns
a = 0.01372

#this function uses numba and takes a particle and moves it to the next cell or the end of time step
@jit(nopython=True, cache=True)
def move_particle(weight, mu, position, cell_l, cell_r,sigma_a, sigma_s, distance_to_census):
    if mu>0:
        distance_to_boundary = (cell_r - position)/mu
    else:
        distance_to_boundary = (cell_l - position)/mu
    #sample distance to next scatter
    distance_to_scatter = random.expovariate(sigma_s)
    #find which distance is smaller
    distance_to_next_event = min(distance_to_boundary, distance_to_scatter, distance_to_census)
    dx = cell_r - cell_l
    #variable to hold where particle moves, -1 to left, 0 to census, 1 to right
    new_location = 0
    #which event is next
    #move particle to the collision site
    position = position + mu*distance_to_next_event
    if distance_to_next_event == distance_to_boundary:
        if mu>0:
            position = cell_r
            new_location = 1
        else:
            position = cell_l
            new_location = -1
    elif distance_to_next_event == distance_to_scatter:
        #scatter
        weight = weight * sigma_s/(sigma_s + sigma_a)
        #sample new mu
        mu = random.uniform(-1,1)
    else:
        #census
        True
    #update distance to census
    distance_to_census = distance_to_census - distance_to_next_event
    #update weight with implicit caputure
    weight = weight * math.exp(-sigma_a*distance_to_next_event)
    #compute deposited weight, this is the integral of the weight over the distance traveled
    deposited_intensity = weight*(1-math.exp(-sigma_a*distance_to_next_event))/sigma_a/dx
    deposited_weight = deposited_intensity*sigma_a
    return weight, mu, position, new_location, deposited_weight, deposited_intensity/dt, distance_to_census
    
#this function loops over all particles and moves them to the next cell or census
@jit(nopython=True, cache=True)
def move_particles(weights, mus, times, positions,cell_indices, mesh, sigma_a, sigma_s,dt):
    #cell_indices is a list of the indices of the cells that the particles are in
    #mesh is a 2-D array of length I where I is the number of cells, each row is the left and right boundary of the cell
    #assert that all arrays are the same length
    assert len(weights) == len(mus) == len(times) == len(positions)
    #initialize array to hold deposited weight
    deposited_weights = np.zeros(len(sigma_a))
    scalar_intensity = np.zeros(len(sigma_a))
    #initialize array to hold new positions
    N = len(weights) #number of particles
    
    #loop over particles
    for i in range(N):
        distance_to_census = (dt-times[i])*c
        
        #move particle until it reaches census
        while distance_to_census > 0:
            loc = cell_indices[i]
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
            deposited_weights[loc] += output[4]
            #update scalar intensity
            scalar_intensity[loc] += output[5]
            #update distance to census
            distance_to_census = output[6]
            #check to see if particle has exited the slab
            if cell_indices[i] == len(mesh):
                distance_to_census = 0
            elif cell_indices[i] == -1:
                distance_to_census = 0

    return deposited_weights, scalar_intensity

#function for sampling surface source
def create_boundary(N,T,dt):
    assert (N > 0)
    total_emission = a*c*T**4/2*dt
    weights = np.zeros(N)+total_emission/N
    mus = np.sqrt(np.random.uniform(0,1,N))
    times = np.random.uniform(0,dt,N)
    positions = np.zeros(N)
    cell_indices = np.zeros(N,dtype=int)
    return weights, mus, times, positions, cell_indices

#function for sampling during time step
def source_particles(Ntarget, Temperatures, dt, mesh, sigma_a):
    dx = mesh[:,1] - mesh[:,0]
    emitted_energies = a*c*Temperatures**4*dt*dx*sigma_a
    total_emission = np.sum(emitted_energies)
    #loop over cells and sample particles
    weights = []
    mus = []
    times = []
    positions = []
    cell_indices = []
    for i in range(len(emitted_energies)):
        N = math.ceil(Ntarget*emitted_energies[i]/total_emission)
        for j in range(N):
            weights.append(emitted_energies[i]/N)
            mus.append(np.random.uniform(-1,1))
            times.append(np.random.uniform(0,dt))
            positions.append(np.random.uniform(mesh[i,0],mesh[i,1]))
            cell_indices.append(i)
    weights = np.array(weights)
    mus = np.array(mus)
    times = np.array(times)
    positions = np.array(positions)
    cell_indices = np.array(cell_indices, dtype=int)
    return weights, mus, times, positions, cell_indices

#function to run simulation
def run_simulation(Ntarget,Nboundary, Tinit, T_boundary, dt, mesh, sigma_a_func,eos,inv_eos,cv, final_time):
    #T_boundary is a tuple with the left and right boundary temperatures
    I = mesh.shape[0]
    deposited_energies = np.zeros_like(mesh.shape[0])
    internal_energy = eos(Tinit)
    radiation_temperature = np.zeros_like(mesh.shape[0])
    temperature = Tinit.copy()
    time = 0
    weights = np.empty(0)
    mus = np.empty(0)
    times = np.empty(0)
    positions = np.empty(0)
    cell_indices = np.empty(0, dtype=int)
    dxs = mesh[:,1] - mesh[:,0]
    boundary_loss = 0.0
    boundary_emission = 0.0
    #print the table header
    print("Time", "N", "Total Energy", "Total Internal Energy", "Total Radiation Energy", "Boundary Emission","Lost Energy", sep='\t')
    print("===============================================================================================================")
    while time < final_time:
        sigma_a_true = sigma_a_func(temperature)
        #compute beta
        beta = 4*a*temperature**3/cv(temperature)
        #compute Fleck factor
        f = 1/(1+beta*sigma_a_true*c*dt)
        #compute sigma_s
        sigma_s = sigma_a_true*(1-f)
        sigma_a = sigma_a_true*f
        #sample boundary source
        #sample from the left boundary?
        if (T_boundary[0] > 0):
            boundary_source = create_boundary(Nboundary,T_boundary[0],dt)
            #add the particles to the appropriate lists
            weights = np.concatenate((weights,boundary_source[0]))
            mus = np.concatenate((mus,boundary_source[1]))
            times = np.concatenate((times,boundary_source[2]))
            positions = np.concatenate((positions,boundary_source[3]))
            cell_indices = np.concatenate((cell_indices,boundary_source[4]))
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
        #sample from internal source
        internal_source = source_particles(Ntarget, Tinit, dt, mesh, sigma_a)
        #add the particles to the appropriate lists
        weights = np.concatenate((weights,internal_source[0]))
        mus = np.concatenate((mus,internal_source[1]))
        times = np.concatenate((times,internal_source[2]))
        positions = np.concatenate((positions,internal_source[3]))
        cell_indices = np.concatenate((cell_indices,internal_source[4]))
        #move particles
        deposited, scalar_intensity = move_particles(weights, mus, times, positions, cell_indices, mesh, sigma_a, sigma_s, dt)
        #print(np.sum(deposited/dxs),np.sum(0.5*a*dt*(T_boundary[0]**4)), sum(scalar_intensity/c),sum(weights)/c)
        #update internal energy
        emitted = a*c*sigma_a*(temperature**4)*dt
        internal_energy += deposited - emitted
        #update temperature
        temperature = inv_eos(internal_energy)
        #update radiation temperature
        radiation_temperature = (scalar_intensity/a/c)**.25
        #update time
        time += dt
        #clean up particles that left the slab and store their weights in boundary_loss
        mask = cell_indices >= 0
        boundary_loss += np.sum(weights[~mask])/c
        weights = weights[mask]
        mus = mus[mask]
        times = times[mask]
        positions = positions[mask]
        cell_indices = cell_indices[mask]
        mask = cell_indices < I
        boundary_loss += np.sum(weights[~mask])/c
        weights = weights[mask]
        mus = mus[mask]
        times = times[mask]
        positions = positions[mask]
        cell_indices = cell_indices[mask]
        #set all particle times to 0
        times = np.zeros(times.shape)
        total_internal_energy = np.sum(internal_energy*dxs)
        total_radiation_energy = np.sum(a*radiation_temperature**4*dxs)
        total_energy = total_internal_energy + total_radiation_energy
        boundary_emission += 0.5*a*dt*(T_boundary[0]**4+T_boundary[1]**4)
        energy_loss = total_energy - boundary_emission + boundary_loss
        #print to the screen a row in a table containing the time, number of particles, the total energy, the total internal energy, and the total radiation energy, the boundary emission, energy_loss with tabs between each value
        #also limit every number to 6 decimal places
        print("{:.6f}".format(time), len(weights), "{:.6f}".format(total_energy), "{:.6f}".format(total_internal_energy), 
              "{:.6f}".format(total_radiation_energy), "{:.6f}".format(boundary_emission), "{:.6f}".format(energy_loss), sep='\t')       
    return internal_energy, temperature, radiation_temperature

if __name__ == "__main__":
    #set parameters
    Ntarget = 1000
    Nboundary = 1000
    dt = 0.01
    L = 0.5 #length of slab
    I = 50 #number of cells
    mesh = np.zeros((I,2))
    dx = L/I
    for i in range(I):
        mesh[i] = [i*dx,(i+1)*dx]
    Tinit = np.zeros(I) + 1e-1
    Tinit[0] = 1.0
    T_boundary = (1.0,0)
    sigma_a = lambda T: 300*T**-3
    eos = lambda T: 0.3*T
    inv_eos = lambda u: u/0.3
    cv = lambda T:T*0+0.3
    final_time = dt*5
    #run simulation
    internal_energy, temperature, radiation_temperature = run_simulation(Ntarget,Nboundary, Tinit, T_boundary, dt, mesh, sigma_a,eos,inv_eos,cv, final_time)
    #print("Internal Energy", internal_energy)
    #print("Temperature", temperature)
    #print("Radiation Temperature", radiation_temperature)
    #plot results
    plt.plot(mesh[:,0],temperature)
    plt.plot(mesh[:,0],radiation_temperature)
    xplot = np.linspace(0,L,1000)
    #true solution is E2(sigma_a*x)
    from scipy.special import expn
    #plt.plot(xplot,(a*c*expn(2,10*xplot)))
    plt.legend(["Temperature","Radiation Temperature"])
    plt.show()