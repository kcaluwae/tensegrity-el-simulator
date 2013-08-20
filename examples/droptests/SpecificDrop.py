'''
	Original Coded by:
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>

	Edited by:
	Jonathan Bruce 2013
	<jbruce@soe.ucsc.edu>
	
	Edited by:
	Kyle Morse summer 2013
	<mors9075@vandals.uidaho.edu>
'''

import simulator_v2 as simulator
import equilibrium_util as eu
import mass_matrix
import numpy as np
from matplotlib import pylab as plt
plt.ion()

#create structure
import icosahedron_payload_test
import mplab_gui

enable_gui = True 
rodMass = 0

#################################################################################
#################### Use these to change simulation parameters ##################
########################### All units are in Metric #############################

rod_height = 4. #length of strut in meters

spring_outer_springConstant = 40000 #N/m
spring_inner_springConstant = 7000 #N/m

ko =  spring_outer_springConstant #can use ko and ki for conversions if desired (1 lb/in = 175.1 N/m)
ki = spring_inner_springConstant

#either drop_height or v_o is used to define velocity not both
#drop_height = 5.0 #drop height in m
#v_o = -np.sqrt(2*9.81*drop_height) #velocity on earth in m/s in terms of height

v_o = -11.4 #m/s #velocity in m/s predefined by user

#orientation angle based on rotation about y axis for structure
angle = 0#-0.36486

payload_mass =70. #mass of accelerometer payload kg
payload_radius = 0.2

D_o = 1.25 #outer diameter in inches
D_i = 1.18 #inner diameter of strut in inches
#################################################################################
#Leave these parameters constant
Area_b = 3.14*.25*(D_o**2-D_i**2)*0.000645 #Area of base converted to square meters
Volume_strut = Area_b*rod_height
densityAl = 2780 #density of Al 2014 in kg/m^3
rod_mass = densityAl*Volume_strut #dimension based on 1 1/4 inch diameter aluminum strut

spring_damping = 0.10
stretch_out = 0.000001 #stretch for pretension in meters
stretch_in = 0.000001 #stretch for pretension of inner springs in meters

default_gravity = True # The default is Earth's and will ignore user input for gravity
gravity = 1.352 # Titan's Gravity, change this value for something other than default

simulation_time = 150 # Simulation time in milliseconds
#################################################################################
#Outputs for any result
print 'Rod Length: {}'.format(rod_height)
print 'K_out: {}'.format(ko)
print 'K_in: {}'.format(ki)
#print 'Velocity: {}'.format(v_o)
#print 'Strut Length: {}'.format(rod_height)
#print 'Rod mass: {}'.format(rod_mass)

#create sets of springs for force calculations
sList=np.zeros(60).reshape((6,10))
sList[0] = [0,1,2,3,24,4,5,6,7,25]
sList[1] = [8,9,10,11,26,12,13,14,15,27]
sList[2] = [16,1,4,12,30,18,19,5,13,29]
sList[3] = [20,21,0,8,28,22,23,1,9,31]
sList[4] = [10,14,18,22,32,2,6,19,23,33]
sList[5] = [11,15,16,20,34,3,7,17,21,35]

#create icosahedron N is nodes, B are bars, C refers to springs
N,B,C = icosahedron_payload_test.create_icosahedron(height=rod_height,payloadR=payload_radius,angleRot=angle)
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
#struts 2, 8, and 10 work best for constrained nodes
constrained_nodes[2] = 0 #fix a number of strut endpoints
constrained_nodes[8] = 1 #fix a number of strut endpoints
constrained_nodes[10] = 1 #fix a number of strut endpoints

spring_k_1 = np.ones(C.shape[0]) #spring constants 
for i in range (spring_k_1.shape[0]):
	if i < spring_k_1.shape[0]-numInnerSprings:
		spring_k_1[i] = ko #Outer Springs
	else:
		spring_k_1[i] = ki #Inner Springs

spring_d = np.ones(C.shape[0])*spring_damping #spring damping
bar_lengths,spring_lengths = eu.compute_lengths(B,C,N.T)

#spring equilibrium length: spring_l0 = spring_lengths-stretch
spring_l0 = np.zeros(C.shape[0])
for i in range(spring_l0.shape[0]):
	if i<spring_l0.shape[0]-numInnerSprings:
		spring_l0[i]=spring_lengths[i]-stretch_out
	else:
		spring_l0[i]=spring_lengths[i]-stretch_in

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
external_forces_1_func = lambda t: (1*t*external_forces_1) #gravity is applied

#the second argument is the initial nodal velocity
initVel = np.zeros(N.shape)
initVel[:][:,2] = v_o

#initialize the simulator
sim = simulator.Simulator(N, initVel, constrained_nodes, B, bar_mass, bar_sigma, C, spring_k_1, spring_d, spring_l0, nodal_damping=0.05, two_way_spring=False,external_nodal_forces=external_forces_1_func)
#prints view of structure during deformation (only prints last frame in Windows)
#if(enable_gui):
#    gui = mplab_gui.Visualization(sim)
#    sim.add_event_listener(gui.callback,"time_step")

sim.initialize_ode()#rtol=1e-5,atol=1e-5)#integrator='rk4',control_dt=50000,internal_dt=2000,dt=0.002)

sim.simulate()

offsets = []
pos1 = [] 
l0 = sim.spring_l0.copy()
nForce1 = []
sForce = []
simVel = []
simAccel = []

endVel = 0.
posDiff = np.ones(simulation_time)
numMin = 0.
#FIRST SIMULATION: random forces applied along the springs.
#The random forces are recorded for retesting later.
for j in xrange(simulation_time):
	sim.simulate()
	pos1.append(sim.nodes_eucl.copy())
	sForce.append(sim.spring_forces)
	simVel.append(sim.nodes_dot)
	simAccel.append(sim.Q_dot_dot)     
	
	#run simulation until velocity is 10% of the absolute value of the original velocity
	#this stops the simulation just after the maximum deflection
	if (endVel <= -0.1*v_o):							
		#Determine state of payload and nodes as well as forces
		aAccel = np.array(simAccel)
		payAccel = aAccel[:,6]
		aPos = np.array(pos1)
		CorrectPos = aPos[:,13]-aPos[0,13]*np.ones_like(aPos[:,13])
		nodePos = aPos[:,0:11,2]-aPos[0,0:11,2]
		aForce = np.array(sForce)
		outerForce = aForce[:,0:24]
		innerForce = aForce[:,24:36]
		aVel = np.array(simVel)
		endVel = aVel[j,13,2]
		
		#Get set of spring forces
		listForce = np.zeros(60).reshape((6,10))
		arraySize = 6*simulation_time
		sumForces = np.zeros(arraySize).reshape((simulation_time,6))
		for r in range(sList.shape[0]):
			for c in range(sList.shape[1]):
				listForce[r,c] = aForce[j,sList[r,c]]
			sumForces[j,r] = np.sum(listForce[r,:])
		
		#determine if payload impacts strut below it
		#MUST RECALCULATE SECOND TERM IF NOT TWO POINT CONTACT
		posDiff[j] = aPos[j,13,2] - np.min(aPos[j,2,2])#time dependent only for 2 pt contact ie aPos[m,node,2]
		if posDiff[j] <= 0:
			num = np.abs(posDiff[j])
			if num > numMin:
				numMin = num
	else:
		break

if any(posDiff <= 0):
	print('0')	
	print(np.max(payAccel/9.81))
	print(numMin)				
else:
	gap = np.min(posDiff)	
	print('Max Force Outer: {}'.format(np.max(outerForce)/4.448))
	print('Max Force Inner: {}'.format(np.max(innerForce)/4.448))
	print('Acceleration: {}'.format(np.max(payAccel/9.81)))
	print('Space: {}'.format(np.max(gap)))
	print('Payload Deflection: {}'.format(np.min(CorrectPos)))
	print('Total spring max force: {}'.format(np.max(sumForces)/4.448))
	print('Time: {}'.format(j))
#######can copy and modify code below to save files of data if desired
	#np.savetxt('Payload_Velocity.txt',np.column_stack((iterations,np.array(simVel)[:,13])),delimiter=',',fmt='%f')
	
#	plt.figure(4)
#	plt.plot(np.array(simAccel)[:,6])
#
#	plt.figure(5)
#	plt.plot(CorrectPos[:,2])
#
#	plt.figure(6)
#	plt.plot(np.array(simVel)[:,13])