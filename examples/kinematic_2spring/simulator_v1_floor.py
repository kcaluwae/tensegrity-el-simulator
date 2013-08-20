'''
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>
'''
import numpy as np
from scipy.integrate import ode as sode

class Simulator(object):
	def __init__(self,nodes,node_velocities,constrained_nodes,bar_connections,bar_mass,bar_sigma,spring_connections,spring_k,spring_d,spring_l0,nodal_damping=0,two_way_spring=False, external_nodal_forces=None):
		'''
			Automatically computed constants:
			N: number of nodes
			B: number of bars
			S: number of springs
			C: number of constrained nodes
			
			Parameters:
			nodes: initial nodal coordinate matrix: Nx3
			node_velocities: initial nodal velocities (Euclidean) (see Returns): Nx3
			constrained_nodes: indicates nodes which don't move (boolean): N
			bar_connections: bar connection matrix: BxN
			bar_mass: mass matrices of the bars @see mass_matrix: Bx2x2 
			bar_sigma: sigma values used for each bar: B
			spring_connections: spring connection matrix: SxN
			spring_k: vector of spring constants: S
			spring_d: vector of spring damping coefficients: S
			spring_l0: vector of spring equilibrium lengths: S
			nodal_damping: nodal damping coefficient (global)
			two_way_spring: if True, springs don't go slack when shorter than their equilibrium length	
			external_nodal_forces: a function of time returning the external forces applied to the nodes (or None to disable external forces). Can be used to model e.g. gravity.

			Returns:
			real_node_velocities: the nodal velocities are projected on the bars and this component is removed as well as the velocities of fixed points		
		'''
		self.nodes = nodes
		self.nodes_eucl = self.nodes.copy() #a copy, because the user might change self.nodes to change the position of the fixed nodes (ALL nodes)
		self.node_velocities = node_velocities
		self.constrained_nodes = constrained_nodes
		self.bar_connections = bar_connections
		self.bar_mass = bar_mass
		self.bar_sigma = bar_sigma
		self.spring_connections = spring_connections
		self.spring_k = spring_k
		self.spring_d = spring_d
		self.spring_l0 = spring_l0
		self.spring_l0_offset = self.spring_l0.copy()#np.zeros(spring_l0.shape) #the offset (motor) of spring_l0 (a constant parameter), so the actual eq. length is spring_l0+spring_l0_offset. This vector is computed internally, by interpolating spring_l0_offset_target == the offset target length at the next time step
		self.spring_l0_offset_target = np.zeros(spring_l0.shape)
		self.nodal_damping = nodal_damping
		self.two_way_spring = two_way_spring
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
		self.C = self.constrained_nodes.sum()
		self.nodes_bar_indices = np.where(np.sum(np.abs(self.bar_connections),0))[0]#contains the indices of self.nodes which are endpoints of bars (the only ones of interest to the simulation): 2B
		self.nodes_all_to_bar_nodes = np.eye(self.N)[self.nodes_bar_indices].astype(int)#selects only the B rows from self.nodes that have bars attached to them 
		self.nodes_bar_nodes_to_all = np.zeros((self.N,2*self.B)) #converts the matrix containing only Euclidean coordinates of bar endpoints to the full shape of self.nodes, so it can be used to reconstruct the full state matrix: Nx2B
		self.nodes_bar_nodes_to_all[self.nodes_bar_indices] = np.eye(2*self.B)
		self.bar_connections_reduced = self.bar_connections[:,self.nodes_bar_indices] #bar_connections with all columns removed corresponding to nodes not being an endpoint of a bar: Bx2B
		self.nodes_fixed_indices = np.where(self.constrained_nodes)[0] #contains the indices (in self.nodes) of the constrained nodes: C
		self.nodes_fixed = self.nodes[self.nodes_fixed_indices] #contains the Euclidean coordinates of the constrained nodes: Cx3
		self.nodes_fixed_internal_indices = np.intersect1d(self.nodes_fixed_indices,self.nodes_bar_indices) #fixed bar nodes: <=B
		self.nodes_fixed_external_indices = np.lib.setdiff1d(self.nodes_fixed_indices,self.nodes_fixed_internal_indices) #fixed nodes not attached to a bar <= C
		self.nodes_fixed_to_full = np.zeros((self.N,self.C))#converts the nodes_fixed matrix to the full shape of self.nodes so you can use it to reconstruct the full state matrix: NxC
		self.nodes_fixed_to_full[self.nodes_fixed_indices] = np.eye(self.C)
		self.compute_Y_matrix()
	
		#compute the Phi matrix to convert between generalized and Euclidean coordinates
		self.compute_bar_lengths()
		self.compute_Phi()

		#compute initial Q matrix (generalized coordinates)
		self.Q = np.dot(np.linalg.inv(self.Phi),np.dot(self.nodes_all_to_bar_nodes,self.nodes))

		#compute the real initial velocities (projections onto the bars)
		self.compute_initial_velocities()

		#compute the fixed part of Q (R values that don't change because of a fixed bar endpoint)
		self.compute_Q_fixed_matrix()	

		#compute the reduced version of Q and Q_dot: removed fixed R and R' == 0
		#These are the indices in Q and Q_dot that don't change (i.e. non zero elements of Q_fixed).
		#These are obtained by creating a vector with 1s on the location of fixed internal points (bar endpoints) and then converting this to generalized coordinates. This will result in a vector with ones at the location of fixed Rs.
		self.Q_fixed_indices = np.where(np.dot(np.linalg.inv(self.Phi),np.dot(self.nodes_all_to_bar_nodes,np.array([i in self.nodes_fixed_internal_indices for i in range(self.N)]).astype(int).reshape((-1,1))))[:self.B]>0)[0]
		self.Q_dot[:self.B][self.Q_fixed_indices] = 0 #all generalized velocities of constrained endpoints are zero. This is already the case without this line, but there might be some rounding errors.
		self.Q_full_to_reduced = np.eye(self.B*2)[np.lib.setdiff1d(range(self.B*2),self.Q_fixed_indices)].astype(int)#This matrix removes rows from Q which are fixed (or 0 in Q_dot)		
		self.Q_reduced_to_full = np.zeros((self.B*2,self.B*2-self.nodes_fixed_internal_indices.shape[0])).astype(int)#adds zero rows at locations in Q/Q_dot which are constant, so Q = np.dot(Q_reduced_to_full,Q_reduc) + Q_fixed
		self.Q_reduced_to_full[np.lib.setdiff1d(range(self.B*2),self.Q_fixed_indices)] = np.eye(self.B*2-self.nodes_fixed_internal_indices.shape[0])

		#initialize integration
		#current position of the nodes
		#self.nodes_eucl #current position (Euclidean) of ALL nodes		
		#self.nodes_dot #current velocity of the nodes (Euclidean) (ALL nodes)
		#self.Q #current generalized coordinates (of all nodes attached to a bar)
		#self.Q_dot # current generalized velocities (of all nodes attached to a bar)
		#self.Q_reduc # current generalized coordinates with all constant Rs removed
		#self.Q_dot_reduc #  current generalized velocities with all zero R_dots removed

		#allow hooks
		self.init_event_listeners()


	def compute_Y_matrix(self):
		'''
			Computes the constant offset matrix for the conversion from generalized to Euclidean coordinates.
			I've put this in a function, because you might want to call this every simulation step when you move the fixed points kinematically during the simulation.
		'''
		self.Y = np.dot(self.nodes_fixed_to_full,self.nodes_fixed) #Matrix containing only the fixed nodes' Euclidean coordinates: Nx3
		self.Y_external = self.Y.copy() #This is self.Y with the fixed bar endpoints removed. Otherwise we count them twice as we will fix a part of Q
		self.Y_external[self.nodes_fixed_internal_indices] = 0 

	def compute_Q_fixed_matrix(self):
		'''
			Computes the constant part of Q.
			The reason for this is given in compute_Y_matrix
		'''
		#all Q parts corresponding to bars with a fixed node
		self.Q_fixed = np.dot(np.linalg.inv(self.Phi),np.dot(self.nodes_all_to_bar_nodes,self.Y))
		self.Q_fixed[self.B:] = 0 #only fix R part

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

	def compute_Phi(self):
		'''
			Computes the matrix which allows you to convert between generalized coordinates and Euclidean coordinates of the BAR ENDPOINTS:
			np.dot(self.nodes_all_to_bar_nodes,self.N)
			i.e. Phi ignores external fixed nodes
			
		'''
		#Q: 2Bx3 with the top rows of Q = R and the bottom rows = B
		#Phi converts Generalized bar coordinates (FULL) to reduced (bar nodes only) Euclidean coordinates:
		#self.N = np.dot(self.nodes_bar_nodes_to_all,np.dot(self.Phi,self.Q))+self.Y_external holds
		#self.Q = np.dot(np.linalg.inv(self.Phi),np.dot(self.nodes_all_to_bar_nodes,(self.nodes))) holds
		#another simple check is to verify that all b's are of unit length np.sum(self.Q[self.B:]**2,1) should give all ones.
		self.Phi = np.zeros((self.B*2,self.B*2))

		for i in range(self.B):
			bar_from = self.bar_connections_reduced[i].argmin() #the bar originates from this node (-1)
			bar_to = self.bar_connections_reduced[i].argmax() #the bar ends at this node (1)
			self.Phi[bar_from,i] = 1
			self.Phi[bar_to,i] = 1
			self.Phi[bar_from,i+self.B] = -self.bar_sigma[i]*self.L_bar[i] 
			self.Phi[bar_to,i+self.B] = (1-self.bar_sigma[i])*self.L_bar[i]

	def compute_initial_velocities(self):
		'''
			Computes the real initial velocities based on the initial velocities passed as nodes_velocities in the constructor.
			This is done by setting all velocities of the fixed nodes to 0 and removing the velocity component of the bars along the bar (i.e. the length of a bar is fixed).
			The result is stored in self.nodes_dot and self.Q_dot
		'''
		self.nodes_dot = self.node_velocities.copy()
		self.nodes_dot[self.nodes_fixed_indices] = 0 #set fixed node velocities to zero
		#compute generalized velocities
		Q_dot = np.dot(np.linalg.inv(self.Phi),np.dot(self.nodes_all_to_bar_nodes,(self.nodes_dot)))
		B_dot_projection = np.sum(self.Q[self.B:]*Q_dot[self.B:],1) #project B_dot (velocity) on B (unit length!)
		Q_dot[self.B:] -= B_dot_projection.reshape((-1,1))*self.Q[self.B:]
		self.Q_dot = Q_dot
		self.nodes_dot = np.dot(self.nodes_bar_nodes_to_all,np.dot(self.Phi,self.Q_dot))

	def compute_nodal_forces(self,t,mix):
		'''
			Computes the forces (Euclidean) on the nodes (using self.nodes_eucl, self.nodes_dot) and stores them in self.nodal_forces
		'''
		#compute spring_forces
		self.compute_spring_lengths()
		self.compute_spring_velocities()
		#self.spring_forces = self.spring_k*(self.L_spring-(self.spring_l0+self.spring_l0_offset)) #force acting along the strings (in Newton): k(l-l0)
		self.spring_forces = self.spring_k*(self.L_spring-(self.spring_state*(1.-mix)+mix*self.spring_l0_offset)) #force acting along the strings (in Newton): k(l-l0)
		#print mix

		#handle slack springs if required
		if(not self.two_way_spring):
			self.spring_forces = np.where(self.spring_forces>0,self.spring_forces,0)

		#compute force density
		self.spring_force_densities = self.spring_forces/self.L_spring
		
		#compute nodal forces 
		self.K = -np.dot(self.spring_connections.T*self.spring_force_densities,self.spring_connections) #C^T k' C
		self.nodal_forces = np.dot(self.K,self.nodes_eucl)

		#add external forces
		if(not self.external_nodal_forces is None):
			self.nodal_forces += self.external_nodal_forces(t)
		
		#floor contacts
		self.nodal_forces[:,2] += np.where(self.nodes_eucl[:,2]<0.,-self.nodes_eucl[:,2]*1000,0)#np.exp(-self.nodes_eucl[:,2]),0)
		
		#compute damping
		self.nodal_forces += -(self.nodes_dot.T*self.nodal_damping).T #nodal damping
		#TODO: compute this damping matrix once
		self.nodal_forces += -np.dot(np.dot(self.spring_connections.T*self.spring_d,self.spring_connections),self.nodes_dot) #spring damping
		self.nodal_forces_reduced = np.dot(self.nodes_all_to_bar_nodes,self.nodal_forces) #with external fixed points removed
		self.nodal_forces_reduced[self.nodes_fixed_internal_indices,:] = 0
		

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
		self.Q_reduced = np.dot(self.Q_full_to_reduced,self.Q)
		self.Q_dot_reduced = np.dot(self.Q_full_to_reduced,self.Q_dot)
		#vectorize
		self.initial_state = np.hstack((self.Q_reduced.ravel(),self.Q_dot_reduced.ravel()))

		self.precompute_matrices()

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
		#compute q_reduced and q_dot_reduced
		q = state[:state.shape[0]/2]
		q_dot = state[state.shape[0]/2:]
		self.Q_reduced = q.reshape((q.shape[0]/3,3)) 
		self.Q_dot_reduced = q_dot.reshape((q_dot.shape[0]/3,3))
		#convert it back to full q and q_dot
		self.Q = np.dot(self.Q_reduced_to_full,self.Q_reduced)+self.Q_fixed
		self.Q_dot = np.dot(self.Q_reduced_to_full,self.Q_dot_reduced)
		#convert this to nodal Euclidean coordinates
		self.nodes_eucl = np.dot(self.nodes_bar_nodes_to_all,np.dot(self.Phi,self.Q))+self.Y_external
		self.nodes_dot = np.dot(self.nodes_bar_nodes_to_all,np.dot(self.Phi,self.Q_dot))

	def precompute_matrices(self):
		'''
			Precompute some of the matrices used in _simu_callback (kron is slow)
		'''
		self.m = self.bar_mass[:,0,0] #a vector containing the masses of the bars: B 
		self.f = self.bar_mass[:,0,1] #a vector containing the coupling between rotational and translational motion: B
		self.j = self.bar_mass[:,1,1] #a vector containing the inertia terms: B
		self.j_inv = 1./self.j
		self.M_start = np.kron(np.diag(self.m),np.diag(np.array([1,1,1,0,0,0]))) #mI terms filled in
		self.M_start_2 = np.kron(np.eye(self.B),np.diag(np.array([0,0,0,1,1,1])))
		self.P_tmp = np.kron(np.eye(self.B),np.ones((3,3)))
		self.P_tmp_2 = np.eye(3*self.B)
		self.j_inv_P_start = np.kron(np.diag(self.j_inv),np.ones((3,3)))
		self.f_j_inv_P_start = np.kron(np.diag(self.f),np.ones((3,3)))

	def _simu_callback(self,t,y):
		'''
			Simulation callback.
			t: time at which to evaluate the derivative 
			y: system state at time t
		
			We are integrating a second order system so
			y takes the following form:
			q q'
			where q is the vector form of the reduced generalized coordinates q
		'''
		self.num_steps += 1
		self.compute_state(y)
		
		#compute nodal forces
		self.compute_nodal_forces(t,(t-self.time*1e-6)/(self.control_dt*1e-6))
		
		#convert forces to generalized forces
		self.Fq = np.dot(self.nodal_forces_reduced.T,self.Phi) #3x2B
		#self.Fq[:,0] = 0
		#self.Fq[:,2] = 0
		#self.Fq[:] = 0 
		#self.Fr = self.Fq[:3] #split up the generalized forces
		#self.Fb = self.Fq[3:]
		#self.Fq[:,2:] = 0
		
		R = self.Q[:self.B] #only translational generalized coordinates
		B = self.Q[self.B:] #only rotational generalized coordinates
		#print np.sum(B**2,1)
		#B = B/np.sqrt(np.sum(B**2,1)).reshape((-1,1))
		#print np.sum(B**2,1)

		#Because the rotational and translational motions are not decoupled
		#We will need to use the vector form of the equations
		#So we will create a vector of the following form
		#r1_x
		#r1_y
		#r1_z
		#b1_x
		#b1_y
		#b1_z
		#r2_x
		#r2_y
		#r2_z
		#b2_x
		#b2_y
		#b2_z
		#...
		B_flat = B.ravel() #b1_x b1_y b1_z b2_x b2_y b2_z ...

		#compute all projection matrices
		#we compute all projection matrices at once
		#np.kron(np.eye(self.B),np.ones((3,3))) selects 3x3 blocks on the diagonal
		#so P is 3Bx3B 
		self.b_norm_square = np.sum(B**2,1) #this should be a vector of ones ideally!!!
		self.b_dot_norm_square = np.sum(self.Q_dot[self.B:]**2,1)
		self.P = self.P_tmp_2-np.outer(B_flat/np.kron(self.b_norm_square,np.array([1,1,1])),B_flat)*self.P_tmp

		#compute mass matrix M
		#The mass matrix has size 6Bx6B
				
		self.M = self.M_start + self.M_start_2 #I terms filled in
		self.j_inv_P = self.j_inv_P_start*self.P
		self.f_j_inv_P = self.f_j_inv_P_start*self.j_inv_P
		self.f_f_j_inv_P = self.f_j_inv_P_start*self.f_j_inv_P
		for i in range(self.B):
			#TODO: this can be done without a loop
			self.M[i*6:i*6+3,i*6:i*6+3] -= self.f_f_j_inv_P[i*3:(i+1)*3,i*3:(i+1)*3]
			self.M[i*6+3:(i+1)*6,i*6:i*6+3] += self.f_j_inv_P[i*3:(i+1)*3,i*3:(i+1)*3]
	
		#compute the H matrix
		self.H = np.zeros((6*self.B,6*self.B))
		for i in range(self.B):
			#TODO: this can be done without a loop
			self.H[i*6:i*6+3,i*6:i*6+3] = np.eye(3)
			self.H[i*6:i*6+3,i*6+3:(i+1)*6] = -self.f_j_inv_P[i*3:(i+1)*3,i*3:(i+1)*3]
			self.H[i*6+3:(i+1)*6,i*6+3:(i+1)*6] = self.j_inv_P[i*3:(i+1)*3,i*3:(i+1)*3]
		  
		#compute the g vector
		self.g = (np.hstack((-B*self.f.reshape((-1,1)),B))*(self.b_dot_norm_square/self.b_norm_square).reshape((-1,1))).ravel()
		#tmp_g = self.g.reshape((-1,1)).reshape((self.B*2,3))
		#self.g = np.hstack((tmp_g[:self.B],tmp_g[self.B:])).ravel()
		
		#we have
		#q'' = M^-1(HF-g)
		self.M_inv = np.linalg.inv(self.M) #TODO: this is stupid, you can simply invert the blocks of M or compute it by hand!!!
		

		tmp_Fq = np.hstack((self.Fq.T[:self.B],self.Fq.T[self.B:])).ravel()#Fq needs to be rearranged 
		#self.Q_dot_dot_scrambled = np.dot(self.M_inv,(np.dot(self.H,self.Fq.T.ravel().reshape((-1,1)))-self.g.reshape((-1,1)))) 
		self.Q_dot_dot_scrambled = np.dot(self.M_inv,(np.dot(self.H,tmp_Fq.reshape((-1,1)))-self.g.reshape((-1,1)))) 				
		#rearrange Q_dot_dot_scrambled to:
		#r1_x
		#r1_y
		#r1_z
		#r2_x
		#r2_y
		#r2_z
		#b1_x
		#b1_y
		#b1_z
		#b2_x
		#b2_y
		#b2_z
		#...
		self.Q_dot_dot = np.vstack((self.Q_dot_dot_scrambled.reshape((self.B,6))[:,:3],self.Q_dot_dot_scrambled.reshape((self.B,6))[:,3:]))

		#compute Q_dot_dot
		#self.Q_dot_dot = np.zeros((self.Q.shape))
		#remove constant terms from Q_dot_dot 
		self.Q_dot_dot_reduced = np.dot(self.Q_full_to_reduced,self.Q_dot_dot)
		#vectorize
		self.new_state = np.hstack((self.Q_dot_reduced.ravel(),self.Q_dot_dot_reduced.ravel()))
		#vectorize 
		return self.new_state

		
		
		
