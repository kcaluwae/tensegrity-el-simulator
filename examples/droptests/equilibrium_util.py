'''
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>
	
'''
import numpy as np

def compute_lengths(B,C,N):
	'''
		Computes the bar and spring lengths for a configuration.
	'''
	bar_lengths = np.sqrt(np.sum(np.dot(B,N.T)**2,1))
	spring_lengths = np.sqrt(np.sum(np.dot(C,N.T)**2,1))
	return bar_lengths, spring_lengths

def compute_spring_offsets(V_spring,c,l,k,eq_length):
	'''
		V_spring: spring nullspace
		c: coefficients for the nullspace (make sure they're valid ;)
		l: spring lengths in the NEW configuration
		k: spring constants (in the new configuration)
		eq_length: spring eq. length (set this to zero if you want to define the initial eq. lengths
	'''
	a = -np.dot(V_spring,c.reshape((-1,1)))*l/k+l-eq_length
	return a

def compute_spring_constants(V_spring,c,l,eq_length,eq_offset):
	'''
		V_spring: spring nullspace
		c: coefficients for the nullspace (make sure they're valid ;)
		l: spring lengths in the NEW configuration
		eq_length: spring eq. length (set this to zero if you want to define the initial eq. lengths
		eq_offset: in the current config.
	'''
	k = np.dot ( V_spring, c.reshape((-1,1)) ) *l / (l-(eq_length+eq_offset) )
	return k


