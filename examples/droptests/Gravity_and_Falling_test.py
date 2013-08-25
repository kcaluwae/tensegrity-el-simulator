'''
	Original Coded by:
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>

	Edited by:
	Jonathan Bruce 2013
	<jbruce@soe.ucsc.edu>
'''

import simulator_v2 as simulator
import scipy.io as sio
import equilibrium_util as eu
import mass_matrix
import numpy as np
from matplotlib import pylab as plt
plt.ion()

#create structure
#import icosahedron
#import icosahedron_payload
import icosahedron_payload_test
import mplab_gui

enable_gui = True 
rodMass = 0

#################################################################################
#################### Use these to change simulation parameters ##################
########################### All units are in Metric #############################

rod_height = 4.0
payload_radius = 0.3

rod_mass = 5.0
payload_mass = 70.0

spring_outer_springConstant = 44000
spring_inner_springConstant = 10000
spring_damping = 25.0

default_gravity = False # The default is Earth's and will ignore user input for gravity
gravity = 1.352 # Titan's Gravity, change this value for something other than default

simulation_time = 250 # Simulation time in milliseconds

#################################################################################

#N,B,C = icosahedron.create_icosahedron()
#N,B,C = icosahedron_payload.create_icosahedron()
N,B,C = icosahedron_payload_test.create_icosahedron(height=rod_height,payloadR=payload_radius)
if B.shape[0]==7:
	rodMass = 1


N = N.T
B = -B
numInnerSprings = 12

#bar_sigma defines how to translate from generalized coordinates to Euclidean coordinates
#this has to be consistent with the point of reference for the mass matrix (see below)
#I always use 0
bar_sigma = np.zeros(B.shape[0])
constrained_nodes = np.zeros(N.shape[0]).astype(bool)
#note: you can't constrain two ends of a bar, although you can add external fixations!
constrained_nodes[2] = 0 #fix a number of strut endpoints
constrained_nodes[8] = 1 #fix a number of strut endpoints
constrained_nodes[10] = 1 #fix a number of strut endpoints
spring_k_1 = np.ones(C.shape[0]) #spring constants 
for i in range (spring_k_1.shape[0]):
	if i < spring_k_1.shape[0]-numInnerSprings:
		spring_k_1[i] = spring_outer_springConstant #Outer Springs
	else:
		spring_k_1[i] = spring_inner_springConstant #Inner Springs
print 'Spring Constant Matrix'
print spring_k_1
spring_d = np.ones(C.shape[0])*spring_damping #spring damping
bar_lengths,spring_lengths = eu.compute_lengths(B,C,N.T)

#spring equilibrium length
spring_l0 = spring_lengths-0.04

#compute mass matrices
density_1 = lambda x:rod_mass
density_2 = lambda x:payload_mass
if rodMass == 1:
	bar_mass = np.array ([ mass_matrix.compute_mass_matrix (length = bar_lengths.ravel()[i], density = density_1, sigma=0.)[0] for i in range (B.shape[0]) ])
	bar_mass[-1] = np.array([mass_matrix.compute_mass_matrix (length = bar_lengths.ravel()[i], density = density_2, sigma=0.)[0] ])
else:
	bar_mass = np.array ([ mass_matrix.compute_mass_matrix (length = bar_lengths.ravel()[i], density = density_1, sigma=None)[0] for i in range (B.shape[0]) ])

#compute external forces (gravity :)
external_forces_1 = np.zeros(N.shape)

if default_gravity == True:
	force_distribution_gravity = lambda x: (0,0,-9.81) # Earth's gravity
else:
	force_distribution_gravity = lambda x: (0,0,-gravity) # Titan's Gravity

for i in xrange(B.shape[0]):
	#this is pretty complicated for something really simple...
	if i<(B.shape[0]-1):
		f1_1,f2_1 = mass_matrix.compute_consistent_nodal_forces_vector(length=bar_lengths.ravel()[i],density=density_1,force_distribution=force_distribution_gravity)
		from_ = B[i].argmin()
		to_ = B[i].argmax()
		external_forces_1[from_] = f1_1
		external_forces_1[to_] = f2_1
	else:	
		f1_1,f2_1 = mass_matrix.compute_consistent_nodal_forces_vector(length=bar_lengths.ravel()[i],density=density_2,force_distribution=force_distribution_gravity)
		from_ = B[i].argmin()
		to_ = B[i].argmax()
		external_forces_1[from_] = f1_1
		external_forces_1[to_] = f2_1

print 'Gravity Forces'
print external_forces_1

external_forces_1_func = lambda t: (1*t*external_forces_1) #gravity is applied

#initialize the simulator
#the second argument is the initial nodal velocity
initVel = np.zeros(N.shape)
initVel[:][:,2] = -11.4
print 'Initial Velocity of Nodes'
print initVel
sim = simulator.Simulator(N, initVel, constrained_nodes, B, bar_mass, bar_sigma, C, spring_k_1, spring_d, spring_l0, nodal_damping=0.05, two_way_spring=False,external_nodal_forces=external_forces_1_func)
if(enable_gui):
    gui = mplab_gui.Visualization(sim)
    sim.add_event_listener(gui.callback,"time_step")

sim.initialize_ode()#rtol=1e-5,atol=1e-5)#integrator='rk4',control_dt=50000,internal_dt=2000,dt=0.002)

sim.simulate()
offsets = []
pos1 = [] 
l0 = sim.spring_l0.copy()

nForce1 = []
sForce = []
simVel = []
simAccel = []

#FIRST SIMULATION: random forces applied along the springs.
#The random forces are recorded for retesting later.

print 'Bar Mass Matrix'
print bar_mass

print 'Simulation Time'
for j in xrange(simulation_time):
	sim.simulate()
	plt.savefig('Viz_%d'%j)
	pos1.append(sim.nodes_eucl.copy())
	nForce1.append(sim.nodal_forces_reduced)
	sForce.append(sim.spring_forces)
	simVel.append(sim.nodes_dot)
	if (j%10 == 0):
	    print '%d'%j
	'''
	if (j >= 15):
	    sim.spring_l0_offset = ((np.random.random(sim.spring_connections.shape[0])*0.2+0.2))
	    print sim.time*1e-6
	    for i in xrange(50):
		sim.simulate()
		pos1.append(sim.nodes_eucl.copy())
		nForce1.append(sim.nodal_forces_reduced)'''

#plot and save the results

iterations = (np.arange(0.,j+1.)/1000)[np.newaxis].T

plt.figure(3)
print np.array(pos1).shape
plt.plot(np.array(pos1)[:,:,2],'b')
plt.title('Z Direction Position')
plt.ylabel('vertical position nodes')
plt.xlabel('time (s)')
plt.savefig('Z_Direction_Position')
np.savetxt('Z_Position.txt',np.column_stack((iterations,np.array(pos1)[:,:,2])),delimiter=',',fmt='%f')
np.savetxt('Y_Position.txt',np.column_stack((iterations,np.array(pos1)[:,:,1])),delimiter=',',fmt='%f')
np.savetxt('X_Position.txt',np.column_stack((iterations,np.array(pos1)[:,:,0])),delimiter=',',fmt='%f')

plt.figure(4)
print np.array(nForce1).shape
for i in xrange(1):
    plt.subplot(3,1,i+1)
    plt.title('Nodal Forces (X direction) %d'%(i+1))
    plt.plot(np.array(nForce1)[:,:,0])
    plt.ylabel('Forces (N)')
    #plt.xlabel('time (s)')
for i in xrange(1):
    plt.subplot(3,1,i+2)
    plt.title('Nodal Forces (Y direction) %d'%(i+1))
    plt.plot(np.array(nForce1)[:,:,1])
    plt.ylabel('Force (N)')
    #plt.xlabel('time (s)')
for i in xrange(1):
    plt.subplot(3,1,i+3)
    plt.title('Nodal Forces (Z direction) %d'%(i+1))
    plt.plot(np.array(nForce1)[:,:,2])
    plt.ylabel('Force (N)')
    #plt.xlabel('time (s)')
plt.savefig('Nodal_Forces')
maxSpringTension = 0
'''
for i in range(np.array(sForce).shape[1]):
	for j in range(np.array(sForce).shape[0]):
		if (maxSpringTension < np.array(sForce)[j,i]):
			maxSpringTension = np.array(sForce)[j,i]
'''
print np.max(sForce)

for i in range(10):
	plt.figure(5)
	plt.plot(np.array(sForce)[:,i],label=str(i))
	plt.legend(loc='best')
	plt.savefig('String_Forces_0-9')
for i in range(10):
	plt.figure(6)
	plt.plot(np.array(sForce)[:,i+10],label=str(i+10))
	plt.legend(loc='best')
	plt.savefig('String_Forces_10-19')
for i in range(10):
	plt.figure(7)
	plt.plot(np.array(sForce)[:,i+20],label=str(i+20))
	plt.legend(loc='best')
	plt.savefig('String_Forces_20-29')
for i in range(np.array(sForce).shape[1]-30):
	plt.figure(8)
	plt.plot(np.array(sForce)[:,i+30],label=str(i+30))
	plt.legend(loc='best')
	plt.savefig('String_Forces_30-35')
plt.title('Spring Forces')
plt.ylabel('Force (N)')
plt.xlabel('Spring')
np.savetxt('Spring_Forces.txt',np.column_stack((iterations,np.array(sForce))),delimiter=',',fmt='%f')

plt.figure(9)
plt.plot(np.array(simVel)[:,13])
plt.savefig('Payload_Velcity')
np.savetxt('Payload_Velocity.txt',np.column_stack((iterations,np.array(simVel)[:,13])),delimiter=',',fmt='%f')

