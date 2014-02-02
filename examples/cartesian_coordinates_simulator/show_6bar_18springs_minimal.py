'''
	Ken Caluwaerts
	<ken@caluwaerts.eu>
	
	Example of a minimal tensegrity with 6 bars and 18 springs.
	See Calladine 1978 for a discussion of the properties of this structure.
'''
import matplotlib.pylab as plt
plt.ion()
import numpy as np
import mplab_gui
import simulator_springmass_stiff as simulator

N = np.array([[-0.25,  0.  ,  0.5 ],
       [ 0.  ,  0.5 , -0.25],
       [ 0.5 , -0.25,  0.  ],
       [-0.5 , -0.25,  0.  ],
       [ 0.  , -0.5 ,  0.25],
       [ 0.25,  0.  ,  0.5 ],
       [ 0.25,  0.  , -0.5 ],
       [-0.25,  0.  , -0.5 ],
       [ 0.5 ,  0.25,  0.  ],
       [-0.5 ,  0.25,  0.  ],
       [ 0.  ,  0.5 ,  0.25],
       [ 0.  , -0.5 , -0.25]])

#spring velocity damping
spring_d = np.ones(18)*0.2 #high damping, we only want to find the equilibrium position
spring_l0 = np.zeros(18)*0+0.2 #some value (doesn't really matter for this example)
node_mass = np.zeros(12)+.3 #some value for this example
spring_rates = np.ones(18)*100. #some value for this example

#this matrix contains a 1 for a spring and a 2 for a bar
A = np.array([[0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2],
       [1, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0],
       [0, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1],
       [2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1],
       [1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1],
       [0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]])
C = np.zeros((18,12))
B = np.zeros((6,12))

ci = 0
bi = 0
for i in xrange(12):
    for j in xrange(i,12):
        if(A[i,j]==1):
            C[ci,i] = 1
            C[ci,j] = -1
            ci+=1
        if(A[i,j]==2):
            B[bi,i] = 1
            B[bi,j] = -1
            bi+=1
            
#these springs show up blue in the GUI (easier to spot the actuators)
tagged_springs = np.zeros(C.shape[0])

sim = simulator.Simulator(N, B, node_mass, C, spring_rates, spring_d, spring_l0, nodal_damping=0.0001)
gui = mplab_gui.Visualization(sim,draw_node_indices = True,tagged_springs=tagged_springs)
sim.add_event_listener(gui.callback,"time_step")
sim.initialize_ode(control_dt=50000)
l0 = spring_l0.copy()
X_save = []
for i in xrange(1000):
	sim.simulate()
	#you can actuate the structure here
	print "%d"%(i)
	X_save.append(sim.nodes_eucl.copy())

