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

import datetime
from pytz import timezone

rodMass = 0

#################################################################################
#################### Use these to change simulation parameters ##################
########################### All units are in Metric #############################

rod_lengthMax = 4.5 #length of strut in meters
rod_lengthMin = 3.5 #length of strut in meters
rod_lengthStep = .5 #length of strut in meters

spring_outer_springConstantMax = 1600000  #N/m
spring_inner_springConstantMax = 40000  #N/m
spring_outer_springConstantMin = 1400000  #N/m
spring_inner_springConstantMin = 30000  #N/m
spring_outer_springConstantStep = 100000  #N/m
spring_inner_springConstantStep = 10000  #N/m

#drop_height = 5.0 #drop height in m/s
#v_o = -np.sqrt(2*9.81*drop_height) #velocity on earth in m/s
v_o = -11.4

D_o = 1.25 #outer diameter in inches
D_i = 1.18 #inner diameter of strut in inches

Area_b = 3.14*.25*(D_o**2-D_i**2)*0.000645 #Area of base converted to square meters
densityAl = 2780 #density of Al 2014 in kg/m^3
payload_mass = 70. #mass of IMU payload kg
payload_radius = .2
spring_damping = .10

default_gravity = True # The default is Earth's and will ignore user input for gravity
gravity = 1.352 # Titan's Gravity, change this value for something other than default

simulation_time = 150 # Simulation time in milliseconds

stretch_out = .000001 #stretch for pretension in meters
stretch_in = .000001 #stretch for pretension of inner springs in meters

runs = 1
for rod_height in np.arange(rod_lengthMin,rod_lengthMax,rod_lengthStep):
	for ko in range(spring_outer_springConstantMin,spring_outer_springConstantMax,spring_outer_springConstantStep):
		for ki in range(spring_inner_springConstantMin,spring_inner_springConstantMax,spring_inner_springConstantStep):

			Volume_strut = Area_b*rod_height
			rod_mass = densityAl*Volume_strut			
			
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
			#nodes 2,8,and 10 work best for constrained endpoints if more than 2 are constrained			
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
			
			#spring equilibrium length
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
			sim.initialize_ode()#rtol=1e-5,atol=1e-5)#integrator='rk4',control_dt=50000,internal_dt=2000,dt=0.002)
			sim.simulate()
			
			offsets = []
			pos1 = [] 
			l0 = sim.spring_l0.copy()
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

				#run simulation until velocity is 25% of the absolute value of the original velocity
				#this stops the simulation just after the maximum deflection
				if (endVel <= -0.25*v_o):							
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
				c = open('ContactRodMass.txt','a')
				c.write('\r\n')
				c.write(str(datetime.datetime.now(timezone('US/Pacific-New'))))
				c.write('\nLength of rod: {}'.format(rod_height))				
				c.write('\nK_out: {}'.format(ko))
				c.write('\nK_in: {}'.format(ki))
				c.write('\nMax Force: {}'.format(np.max(sForce)))
				c.write('\nMax Acceleration:{}'.format(np.amax(payAccel)/9.81))
				c.write('\nOvershoot: {}'.format(numMin))
				c.write('\nTime: {}'.format(j))
				c.close()					
			else:
				gap = np.min(posDiff)	
				n = open('No_ContactTest.txt','a')
				n.write('\r\n')
				n.write(str(datetime.datetime.now(timezone('US/Pacific-New'))))
				n.write('\nK_out: {}'.format(ko))
				n.write('\nK_in: {}'.format(ki))
				n.write('\nPayload Mass: {}'.format(payload_mass))
				n.write('\nLength of rod: {}'.format(rod_height))
				n.write('\nMax Force Outer: {}'.format(np.max(outerForce)))
				n.write('\nMax Force Inner: {}'.format(np.max(innerForce)))
				n.write('\nMax Acceleration: {}'.format(np.amax(payAccel)/9.81))
				n.write('\nPayload Deflection: {}'.format(np.min(CorrectPos)))
				n.write('\nDistance from payload to strut: {}'.format(gap))
				n.write('\nTime: {}'.format(j))					
				n.write('\r\n')
				n.close()
			print(runs)
			runs += 1