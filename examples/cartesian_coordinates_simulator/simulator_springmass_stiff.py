'''
	Ken Caluwaerts 2014
	<ken@caluwaerts.eu>
	
	Tensegrity spring-mass model with Cartesian coordinates instead of Skelton's
	generalized coordinates.
	
'''
import numpy as np
from scipy.integrate import ode as sode

class Simulator(object):
	def __init__(self,nodes,bar_connections,node_mass,spring_connections,spring_k,spring_d,spring_l0,nodal_damping=0,bar_stiffness=100000., external_nodal_forces=None):
		'''
			Automatically computed constants:
			N: number of nodes
			B: number of bars
			S: number of springs
			C: number of constrained nodes
			
			Parameters:
			nodes: initial nodal coordinate matrix: Nx3
			bar_connections: bar connection matrix: BxN
			node_mass: masses of the nodes: N 
			spring_connections: spring connection matrix: SxN
			spring_k: vector of spring constants: S
			spring_d: vector of spring damping coefficients: S
			spring_l0: vector of spring equilibrium lengths: S
			nodal_damping: nodal damping coefficient (global)
			bar_stiffness: spring constants of the bars	
			external_nodal_forces: a function of time returning the external forces applied to the nodes (or None to disable external forces). Can be used to model e.g. gravity.

			Returns:
			real_node_velocities: the nodal velocities are projected on the bars and this component is removed as well as the velocities of fixed points		
		'''
		self.nodes = nodes
		self.nodes_eucl = self.nodes.copy() #a copy, because the user might change self.nodes to change the position of the fixed nodes (ALL nodes)
		self.bar_connections = bar_connections
		self.node_mass = node_mass.ravel()
		self.spring_connections = spring_connections
		self.spring_k = spring_k
		self.spring_d = spring_d
		self.spring_l0 = spring_l0
		self.spring_l0_offset = self.spring_l0.copy()#np.zeros(spring_l0.shape) #the offset (motor) of spring_l0 (a constant parameter), so the actual eq. length is spring_l0+spring_l0_offset. This vector is computed internally, by interpolating spring_l0_offset_target == the offset target length at the next time step
		self.nodal_damping = nodal_damping
		self.bar_stiffness = bar_stiffness
		self.external_nodal_forces = external_nodal_forces 

		#Check if the bars and spring connection matrices are valid
		#if(np.where(np.abs(self.bar_connections).sum(axis=0)>1)[0].shape[0]>0):
			#raise Exception("Invalid bar connection matrix: two bars have the same endpoint")
		if(not np.abs(self.bar_connections.sum(axis=1)).sum() == 0):
			raise Exception("Invalid bar connection matrix: each bar needs two endpoints")
		if(not np.abs(self.spring_connections.sum(axis=1)).sum() == 0):
			raise Exception("Invalid spring connection matrix: each spring needs two endpoints")
		
		#constants
		self.N = self.nodes.shape[0]
		self.B = self.bar_connections.shape[0]
		self.S = self.spring_connections.shape[0]
	
		self.compute_bar_lengths()
		self.bar_l0 = self.L_bar.copy()

		#compute the real initial velocities (projections onto the bars)
		#self.compute_initial_velocities()

		#allow hooks
		self.init_event_listeners()

	def compute_bar_lengths(self):
		'''
			Computes the bar lengths, which is needed to compute Phi
			sets self.L_bar: B
		'''
		self.L_bar = np.sqrt(np.sum(np.dot(self.bar_connections,self.nodes_eucl)**2,1))

	def compute_spring_lengths(self):
		'''
			Computes the spring lengths
			sets self.L_spring: S
		'''
		self.L_spring = np.sqrt(np.sum(np.dot(self.spring_connections,self.nodes_eucl)**2,1))

	def compute_spring_velocities(self):
		'''
			Computes the velocity of two endpoints projected onto normalized vector connecting them (if there's a spring between them)
			sets self.L_spring_dot: S
		'''
		self.L_spring_dot = np.sqrt(np.sum(np.dot(self.spring_connections,self.nodes_dot)**2,1))

	def compute_nodal_forces(self,t,mix):
		'''
			Computes the forces (Euclidean) on the nodes (using self.nodes_eucl, self.nodes_dot) and stores them in self.nodal_forces
		'''
		#compute spring_forces
		self.compute_spring_lengths()
		self.compute_bar_lengths()
		self.compute_spring_velocities()
		self.spring_forces = self.spring_k*(self.L_spring-(self.spring_state*(1.-mix)+mix*self.spring_l0_offset)) #force acting along the strings (in Newton): k(l-l0)
		
		#handle slack springs if required
		self.spring_forces = np.where(self.spring_forces>0,self.spring_forces,0)

		#compute force density
		self.spring_force_densities = self.spring_forces/self.L_spring
				
		#compute nodal forces 
		self.K = -np.dot(self.spring_connections.T*self.spring_force_densities,self.spring_connections) #C^T k' C
		#self.K += -np.dot(self.bar_connections.T*self.bar_force_densities,self.bar_connections)
		self.nodal_forces = np.dot(self.K,self.nodes_eucl)

		#add external forces
		if(not self.external_nodal_forces is None):
                    self.nodal_forces += self.external_nodal_forces(t)
				
		#ground contacts
		#self.nodal_forces[:,2] += np.where(self.nodes_eucl[:,2]<0.,-self.nodes_eucl[:,2]*1000,0)#np.exp(-self.nodes_eucl[:,2]),0)
		
		#compute damping
		self.nodal_forces += -(self.nodes_dot.T*self.nodal_damping).T #nodal damping
		#TODO: compute this damping matrix once
		self.nodal_forces += -np.dot(np.dot(self.spring_connections.T*self.spring_d,self.spring_connections),self.nodes_dot) #spring damping
		

	def compute_potential_energy(self):
		'''
			Computes the potential energy of the structure.
			compute_spring_lengths needs to be called first.
		'''
		self.potential_energy = np.sum(self.spring_k*(self.L_spring-(self.spring_l0+self.spring_l0_offset))**2)/2
		return self.potential_energy

	def initialize_ode(self,integrator="vode",control_dt = 20000,internal_dt=1000,**kwargs):
		'''
			Initializes the integrator properties.
			Parameters: 
				integrator: which integrator (vode/euler/dopri5/rk4/dopri853)
				control_dt: the control timestep in microseconds (integer!) == multiple of internal_dt
				internal_dt: the internal timestep in microseconds (time step of the integrator)
				kwargs: any other arguments are passed directly to the set_integrator method of scipy.integrate.ode
		'''
		self.control_dt = control_dt
		self.internal_dt = internal_dt
		self.time = 0 #total simulation time in milliseconds since start of simulation
		self.time_steps = 0 #==self.time/control_dt
		self.num_steps = 0
		#TODO: check if control_dt is a multiple of internal_dt

		#compute initial system state
		self.Q_reduced = self.nodes_eucl
		self.Q_dot_reduced = self.nodes_eucl*0
		#vectorize
		self.initial_state = np.hstack((self.Q_reduced.ravel(),self.Q_dot_reduced.ravel()))

		#self.precompute_matrices()

		if(integrator=="vode"):
        		self.ode = sode(self._simu_callback) 
        		self.ode.set_integrator('vode',**kwargs)#,max_step=1e-4)
		elif(integrator=="dopri5"):
			self.ode = sode(self._simu_callback) 
        		self.ode.set_integrator('dopri5',**kwargs)#,max_step=1e-4)
		elif(integrator=="rk4"):
			import rk4
			self.ode = rk4.RK4Integration(self._simu_callback,self.initial_state.shape[0],**kwargs)
		elif(integrator=="euler"):
			import euler
			self.ode = euler.EulerIntegration(self._simu_callback,self.initial_state.shape[0],**kwargs)

		self.ode.set_initial_value(self.initial_state)
		self.spring_state = self.spring_l0_offset

		#TODO: add support for other integrators

	def init_event_listeners(self):
		self.event_listeners = {"time_step":[]}

	def add_event_listener(self,callback,event):
		self.event_listeners[event].append(callback)

	def fire_event(self,event):
		for e in self.event_listeners[event]:
			e(event)

	def simulate(self):
		'''
			Performs a control step (advances the simulation control_dt microseconds)
		'''
		
		for i in range(1,self.control_dt/self.internal_dt+1):
			state = self.ode.integrate(1e-6*(self.time+self.internal_dt*i))
		
		self.compute_state(state)
		self.time += self.control_dt
		self.time_steps += 1
		self.spring_state = self.spring_l0_offset
		self.fire_event("time_step")

	def compute_state(self,state):
		'''
			Computes the state based on the information from the integrator
		'''
		q = state[:state.shape[0]/2]
		q_dot = state[state.shape[0]/2:]
		#convert this to nodal Euclidean coordinates
		self.nodes_eucl = q.reshape((self.N,3))
		self.nodes_dot = q_dot.reshape((self.N,3))

	def _simu_callback(self,t,y):
		'''
			Simulation callback.
			t: time at which to evaluate the derivative 
			y: system state at time t
		
			We are integrating a second order system so
			y takes the following form:
			n n'
			where n are the euclidean coordinates stacked in a single row vector
		'''
		self.num_steps += 1
		self.compute_state(y)
		
		#compute nodal forces
		self.compute_nodal_forces(t,(t-self.time*1e-6)/(self.control_dt*1e-6))
		            		
		self.nodes_dot_dot = (self.nodal_forces.T/self.node_mass).T
		n1_n2 = np.sqrt(np.sum(self.bar_connections.dot(self.nodes_eucl)**2,1))
		n1_n2_dot = np.sqrt(np.sum(self.bar_connections.dot(self.nodes_dot)**2,1))
		n1_n2_nodes_dot_dot = np.sum(self.bar_connections.dot(self.nodes_eucl)*self.bar_connections.dot(self.nodes_dot_dot),1)
		
		#compute the constraint force (Lagrange multiplier) in Cartesian coordinates
		#This is equivalent to the result on page 177 in Skelton's book
		#the second term just removes any force (due to springs, friction and external impacts) in the direction of a bar
		#the first term accounts for the velocity in the direction of a bar (needs to be zero) and is typically small
		self.bar_force_densities = -(n1_n2_dot**2+n1_n2_nodes_dot_dot)/(2*n1_n2**2)
		
		''' This is the same as the line below
		for i in xrange(self.B):
		    bi = self.bar_connections[i]
		    _start = bi.argmin()
		    _stop = bi.argmax()
		    f = self.bar_force_densities[i]
		    self.nodes_dot_dot[_start] += f*(self.nodes_eucl[_start]-self.nodes_eucl[_stop])
		    self.nodes_dot_dot[_stop]  -= f*(self.nodes_eucl[_start]-self.nodes_eucl[_stop])
		 '''
		
		self.nodes_dot_dot += np.dot(self.bar_connections.T*self.bar_force_densities,self.bar_connections).dot(self.nodes_eucl)
		
		self.new_state = np.hstack((self.nodes_dot.flat,self.nodes_dot_dot.flat))
		
		return self.new_state

		
		
		
