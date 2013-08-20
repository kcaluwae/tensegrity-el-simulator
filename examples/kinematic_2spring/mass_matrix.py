'''
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>
	Juan Pablo Carbajal
	<ajuanpi@gmail.com>
'''
import numpy as np
import scipy.integrate as si

def compute_mass_matrix(length=1,density=lambda x: 1,sigma=None):
	'''
	Mass matrix of one dimensional rod in 3D.
	Let q be the configuration vector of the rod, with the first three elements of
	q being the spatial coordinates (e.g. x,y,z) and the second three elements of
	q the rotiational coordinates (e.g. Euler angles), then the kinetical energy
	of the rod is given by
	T = 1/2 (dqdt)^T kron(J,eye(3)) dqdt

	@var{sigma} is between 0 and 1. Corresponds to the point in the rod that is
	being used to indicate the position of the rod in space.
	If @var{sigma} is None then the value corresponding to the center of mass
	of the rod is used. This makes @var{J} a diagonal matrix.
	
	@var{density} is a function handle to the density of the rod defined in the
	interval 0,1. The integral of this density equals the mass and is stored in
	@code{@var{J}(1,1)}. If omitted, the default is a uniform rod with unit mass.

	@var{l} is the length if the rod. If omitted the rod hasunit length.

	@return: mass matrix, sigma (useful when the center of mass is computed by this function)
	'''	
	length *= 1. #make sure it's a float

	if(sigma is None):
		#compute center of mass
		if(density is None):
			sigma = 0.5
		else:
			#integrate
			sigma = si.quad(lambda x: density(x)*x,0,1)[0]/si.quad(lambda x: density(x),0,1)[0]
	print sigma
	print length
	u = np.array([-sigma*length,(1-sigma)*length])

	m = si.quad(lambda x: density(sigma+x/length), u[0],u[1])[0]/length
	f = si.quad(lambda x: density(sigma+x/length)*x, u[0],u[1])[0]/length
	j = si.quad(lambda x: density(sigma+x/length)*x**2, u[0],u[1])[0]/length

	J = np.array([[m,f],[f,j]])

	return J, sigma, density


def compute_consistent_nodal_forces(length=1.,density=lambda x: 1,force_distribution=lambda x: 1):
	'''
		Density: mass distribution along the longitudinal axis of the bar.
		see compute_mass_matrix for more details.
		length: length of the strut
		force distribution: the distribution of the force (scalar!!!) along the longitudinal axis of the strut (rescaled to lie within [0,1] like the density).
		If you want to compute the effect of gravity, just use lambda x: 1.
		This function can also be used to compute unequal loading (e.g. a collision). 
		In that case you use the correct force_distribution and then apply a force with a magnitude equal to the result of this function in the direction of the force.
		returns: 
			consistent nodal force coefficients for both ends of the strut 
			E.g. for gravity acting on a bar of length 1 and constant density (total mass = 1), each end takes half of the total force.

	'''
	#length is just a scaling factor
	f1 = si.quad(lambda x: (1-x)*force_distribution(x)*density(x),0,1)[0]*length
	f2 = si.quad(lambda x: x*force_distribution(x)*density(x),0,1)[0]*length
	return f1,f2

def compute_consistent_nodal_forces_vector(length=1.,density=lambda x: 1,force_distribution=lambda x: (1,1,1)):
	'''
		Density: mass distribution along the longitudinal axis of the bar.
		see compute_mass_matrix for more details.
		length: length of the strut
		force distribution: the distribution of the force (vector in world coordinates) along the longitudinal axis of the strut (rescaled to lie within [0,1] like the density).
		If you want to compute the effect of gravity, just use lambda x: 1.
		This function can also be used to compute unequal loading (e.g. a collision). 
		In that case you use the correct force_distribution and then apply a force with a magnitude equal to the result of this function in the direction of the force.
		returns: 
			consistent nodal forces for both ends of the strut 
			E.g gravity: 
				mass_matrix.compute_consistent_nodal_forces_vector(density=lambda x: 1, force_distribution= lambda x: (0,0,-9.81))
				gives you
				(array([ 0.   ,  0.   , -4.905]), array([ 0.   ,  0.   , -4.905]))
				these are the nodal forces (-g/2) for gravity of a bar of mass 1 with equal mass distribution along its length

	'''
	#length is just a scaling factor
	f1v = np.zeros(3)
	f2v = np.zeros(3)
	for i in xrange(3):			
		f1 = si.quad(lambda x: (1-x)*force_distribution(x)[i]*density(x),0,1)[0]*length
		f2 = si.quad(lambda x: x*force_distribution(x)[i]*density(x),0,1)[0]*length
		f1v[i] = f1
		f2v[i] = f2
	return f1v,f2v
