import numpy as np
import equilibrium_util as eu
import mass_matrix

'''
	This file contains a python description of the robot as per 29/05/2013
	Numbering is the same as in the spreadsheat/robot, except that numbering starts from 0!!!
'''
#lengths of the struts (node 1 -> COM -> node 2) in meter
bar_lengths = np.array([
	[0.47,0.445],
	[0.51,0.55],
	[0.47,0.44],
	[0.5,0.5],
	[0.5,0.5],
	[0.5,0.5]
	])

#mass of each spring in kg
spring_mass = np.zeros(30)
spring_mass[:24] = 0.00462
spring_mass[24:30] = 0.007698

	
#total mass of each strut in kg
bar_mass = np.array([
	0.303+3*0.005,
	0.296+3*0.005,
	0.279+3*0.005,
	0.036+2*0.005,
	0.036+2*0.005,
	0.036+2*0.005
	])
bar_mass += np.sum(spring_mass)/6.+0.05

#initial coordinates of the nodes in meter
N = np.array([
	[-0.25,0,bar_lengths[0,0]],
	[-0.25,0,-bar_lengths[0,1]],
	[0,bar_lengths[1,0],-0.25],
	[0,-bar_lengths[1,1],-0.25],
	[bar_lengths[2,0],-0.25,0],
	[-bar_lengths[2,1],-0.25,0],
	[0,-bar_lengths[3,0],0.25],
	[0,bar_lengths[3,1],0.25],
	[0.25,0,bar_lengths[4,0]],
	[0.25,0,-bar_lengths[4,1]],
	[bar_lengths[5,0],0.25,0],
	[-bar_lengths[5,1],0.25,0],
	])

#strut connections
B = np.array([
	[-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0],
	])
	
B = B[:,:12]
#spring connections
C = np.zeros((30,N.shape[0]))
i=0
#passive springs
C[i,(0,5)] = (-1,1);i+=1
C[i,(0,6)] = (-1,1);i+=1
C[i,(0,7)] = (-1,1);i+=1
C[i,(0,11)] = (-1,1);i+=1
C[i,(1,2)] = (-1,1);i+=1
C[i,(1,3)] = (-1,1);i+=1
C[i,(1,5)] = (-1,1);i+=1
C[i,(1,11)] = (-1,1);i+=1
C[i,(2,9)] = (-1,1);i+=1
C[i,(2,10)] = (-1,1);i+=1
C[i,(2,11)] = (-1,1);i+=1
C[i,(3,4)] = (-1,1);i+=1 #broken
C[i,(3,5)] = (-1,1);i+=1
C[i,(3,9)] = (-1,1);i+=1
C[i,(4,6)] = (-1,1);i+=1
C[i,(4,8)] = (-1,1);i+=1
C[i,(4,9)] = (-1,1);i+=1
C[i,(5,6)] = (-1,1);i+=1
C[i,(6,8)] = (-1,1);i+=1
C[i,(7,8)] = (-1,1);i+=1
C[i,(7,10)] = (-1,1);i+=1
C[i,(7,11)] = (-1,1);i+=1
C[i,(8,10)] = (-1,1);i+=1
C[i,(9,10)] = (-1,1);i+=1
#actuated springs
C[i,(0,10)] = (-1,1);i+=1
C[i,(1,6)] = (-1,1);i+=1
C[i,(2,8)] = (-1,1);i+=1
C[i,(3,11)] = (-1,1);i+=1
C[i,(4,7)] = (-1,1);i+=1
C[i,(5,9)] = (-1,1);i+=1

#lengths of the strings attached to the springs in meter
#[ node 1->spring ], [spring -> node 2]
string_lengths = np.array([
	[0.22,	0.055],
	[0.33,	0.05],
	[0.20,	0.055],
	[0.34,	0.04],
	[0.33,	0.045],
	[0.02,	0.31],
	[0.095,	0.29],
	[0.25,	0.035],
	[0.23,	0.055],
	[0.29,	0.05],
	[0.34,	0.035],
	[0.305,	0.035],
	[0.055,	0.265],
	[0.13,	0.11],
	[0.28,	0.065],
	[0.275,	0.05],
	[0.30,	0.05],
	[0.36,	0.04],
	[0.05,	0.335],
	[0.07,	0.235],
	[0.035,	0.375],
	[0.10,	0.24],
	[0.30,	0.03],
	[0.33,	0.045],
	[0.755,	0.025], #actuated springs (i.e. these change)
	[0.54,	0.10],
	[0.79,	0.015],
	[0.72,	0.045],
	[0.83,	0.04],
	[0.78,	0.02],
	])

#string_lengths[:-6]+=0.02
#string_lengths[-6:,1]+=0.02
#string_lengths[11]+=0.02
#spring rate in N/m
spring_rates = np.zeros(C.shape[0])
spring_rates[:24] = 28.466
spring_rates[24:30] = 81.193

#equilibrium lengths of the springs in meter
spring_lengths = np.zeros(C.shape[0])
spring_lengths[:24] = 0.064
spring_lengths[24:30] = 0.07
#spring_lengths[11] += 0.03
