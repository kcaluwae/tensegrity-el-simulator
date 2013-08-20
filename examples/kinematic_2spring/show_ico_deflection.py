'''
	Ken Caluwaerts
	<ken@caluwaerts.eu>
	
	Starting from a recorded motion capture state,
	the rest length of 2 springs is modifief every 5s.
'''
import numpy as np
import ico_15072013 as ico
import mass_matrix
import mplab_gui
import simulator_v1_floor as simulator
import scipy.io as sio

data = sio.loadmat('2013-07-17-16-53-16_short_resampled_50Hz.mat')
strut_len = ico.bar_lengths.sum(1)
ico.N = data['X_t'][0]*0.001#np.zeros((12,3))
#flip Z and Y
dummy = ico.N.copy()
ico.N[:,1] = dummy[:,2]
ico.N[:,2] = dummy[:,1]

#compute total strut lengths
strut_lengths = ico.bar_lengths.sum(1)

#compute spring+strings eq. length
spring_l0 = ico.string_lengths.sum(1)+ico.spring_lengths

#attachments
constrained_nodes = np.zeros(ico.N.shape[0]).astype(bool)

#spring velocity damping
spring_d = np.ones(ico.C.shape[0])*0.2

#computed center of mass (to decouple rotational and translational eq. of motion)
bar_sigma = [1,0,None,None,None,1]
bar_massmatrix = [] #strut mass matrices 
for i in xrange(ico.B.shape[0]):
	b, bar_sigma[i], dummy = mass_matrix.compute_mass_matrix(length = strut_lengths[i], sigma=bar_sigma[i], density=
		lambda x: 0.5*ico.bar_mass[i]*(strut_lengths[i]/ico.bar_lengths[i,0] if x<ico.bar_lengths[i,0]/strut_lengths[i] else strut_lengths[i]/ico.bar_lengths[i,1]))
	bar_massmatrix.append(b)
bar_massmatrix = np.array(bar_massmatrix)

#compute nodal forces due to gravity

external_forces = np.zeros(ico.N.shape)
force_distribution = lambda x: (0,0,-9.81) 
for i in xrange(ico.B.shape[0]):
	#this is pretty complicated for something really simple...
	f0,f1 = mass_matrix.compute_consistent_nodal_forces_vector(length=strut_lengths[i],density=lambda x: 0.5*ico.bar_mass[i]*(strut_lengths[i]/ico.bar_lengths[i,0] if x<ico.bar_lengths[i,0]/strut_lengths[i] else strut_lengths[i]/ico.bar_lengths[i,1]),force_distribution=force_distribution)
	from_ = ico.B[i].argmin()
	to_ = ico.B[i].argmax()
	external_forces[from_] = f0
	external_forces[to_] = f1


#these springs show up blue in the GUI (easier to spot the actuators)
#ico.spring_
tagged_springs = np.zeros(ico.C.shape[0])
tagged_springs[-6:] = 1
constrained_nodes[:] = 0

constrained_nodes[1] = 1
constrained_nodes[2] = 1
constrained_nodes[11] = 1


sim = simulator.Simulator(ico.N, ico.N*0, constrained_nodes, ico.B, bar_massmatrix, bar_sigma, ico.C, ico.spring_rates, spring_d, spring_l0, nodal_damping=0.00001, two_way_spring=False,external_nodal_forces=lambda t: external_forces[:12])
gui = mplab_gui.Visualization(sim,draw_node_indices = True,tagged_springs=tagged_springs)
sim.add_event_listener(gui.callback,"time_step")
sim.initialize_ode(control_dt=50000)
times = data['t_all']
l0 = spring_l0.copy()
X_save = []
for i in xrange(times.shape[0]):
	sim.simulate()
	sim.spring_l0_offset[24:] = l0[24:]-np.abs((data["motor_cnt_t"][i][:6]-2**15)*0.0002158)
	print sim.spring_l0_offset[24:]
	print sim.spring_forces[24:]
	print "%d/%d"%(i,times.shape[0])
	X_save.append(sim.nodes_eucl.copy())

