'''
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>
'''
import matplotlib.pylab as plt
plt.ion()
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import threading
import numpy as np

class Visualization(object):
	'''
		A simple 3D visualization for tensegrity structures using the new Matplotlib animation interface (version 1.1.1)
		Based on: http://matplotlib.sourceforge.net/trunk-docs/examples/animation/simple_3danim.py
	'''
	def __init__(self, simulator,draw_node_indices=True,tagged_springs = None):
		self.simulator = simulator
		self.changed = False #Do we need to redraw
		self.iteration=0
		self.draw_node_indices = draw_node_indices
		self.tagged_springs = tagged_springs
		self.create_window()
		
		#self.animation = animation.FuncAnimation(self.fig,self.update,25,init_func=self.init_plot, blit=False)
		self.init_plot()

	def create_window(self):
		'''
			Creates a simple 3D plotting window
		'''
		self.fig = plt.figure()
		self.ax = p3.Axes3D(self.fig)


	def init_plot(self):
		'''
			This function initializes the plotting window.
		'''
		#get the structure for the first time from the simulator
		nodes = self.simulator.nodes
		#plot the structure
		self.bars_plot = []
		for i in range(self.simulator.B):
			_from = self.simulator.bar_connections[i].argmin()
			_to = self.simulator.bar_connections[i].argmax()
			self.bars_plot.append(self.ax.plot(nodes[(_from,_to),0],nodes[(_from,_to),1],nodes[(_from,_to),2],c='r',linewidth=3)[0])
		self.springs_plot = []
		for i in range(self.simulator.S):
			_from = self.simulator.spring_connections[i].argmin()
			_to = self.simulator.spring_connections[i].argmax()
			if(self.tagged_springs is None):
				c = 'g'
			else:
				c = 'g' if self.tagged_springs[i]==0 else 'b'
			self.springs_plot.append(self.ax.plot(nodes[(_from,_to),0],nodes[(_from,_to),1],nodes[(_from,_to),2],c=c)[0])

		#draw fixed nodes
		#fixed_nodes = nodes[self.simulator.nodes_fixed_indices]
		#if(fixed_nodes.shape[0]>0):
		#	self.ax.plot(fixed_nodes[:,0],fixed_nodes[:,1],fixed_nodes[:,2],'b+')
		self.nodes_plot = self.ax.plot(self.simulator.nodes_eucl[:,0],self.simulator.nodes_eucl[:,1],self.simulator.nodes_eucl[:,2],'yo')
		
		#draw node indices
		
		self.indices_plot = []
		if(self.draw_node_indices):
			for i in xrange(self.simulator.nodes_eucl.shape[0]):
				self.indices_plot.append(self.ax.text(self.simulator.nodes_eucl[i,0],self.simulator.nodes_eucl[i,1],self.simulator.nodes_eucl[i,2],i))
		#axes properties
		self.ax.set_xlim3d([-.5, .5])
		self.ax.set_xlabel('X')
		self.ax.set_ylim3d([-.5, .5])
		self.ax.set_ylabel('Y')
		self.ax.set_zlim3d([0, 1])
		self.ax.set_zlabel('Z')

		self.last_time = 0
		return self.bars_plot+self.springs_plot

	def update(self,iteration):
		'''
			This function refreshes the plot
		'''
		if self.changed:
			for i in range(self.simulator.B):
				nodes = self.simulator.nodes_eucl
				_from = self.simulator.bar_connections[i].argmin()
				_to = self.simulator.bar_connections[i].argmax()
				bar_plot = self.bars_plot[i]
				# NOTE: there is no .set_data() for 3 dim data...
				bar_plot.set_data(nodes[(_from,_to),0],nodes[(_from,_to),1])
				bar_plot.set_3d_properties(nodes[(_from,_to),2])
			for i in range(self.simulator.S):
				nodes = self.simulator.nodes_eucl
				_from = self.simulator.spring_connections[i].argmin()
				_to = self.simulator.spring_connections[i].argmax()
				springs_plot = self.springs_plot[i]
				# NOTE: there is no .set_data() for 3 dim data...
				springs_plot.set_data(nodes[(_from,_to),0],nodes[(_from,_to),1])
				springs_plot.set_3d_properties(nodes[(_from,_to),2])
				#springs_plot.set_linewidth(self.simulator.spring_forces[i])
			self.nodes_plot[0].set_data(self.simulator.nodes_eucl[:,0],self.simulator.nodes_eucl[:,1])
			self.nodes_plot[0].set_3d_properties(self.simulator.nodes_eucl[:,2])
			self.changed = False
			if(self.draw_node_indices):
				for i in xrange(self.simulator.nodes_eucl.shape[0]):
					self.indices_plot[i]._position3d = np.array((self.simulator.nodes_eucl[i,0],self.simulator.nodes_eucl[i,1],self.simulator.nodes_eucl[i,2]))
			#if(iteration%10==0):
			#	self.ax.plot(self.simulator.nodes_eucl[:,0],self.simulator.nodes_eucl[:,1],self.simulator.nodes_eucl[:,2],'bo')
			return self.bars_plot+self.springs_plot+self.nodes_plot+self.indices_plot
		else:
			return []

	def callback(self,event):
		self.changed = True
		self.last_time+=1
		self.iteration+=1
		if(self.last_time>10):
			self.last_time=0
			self.update(0)
			plt.draw()
		#if(self.iteration%2==0):
		#	plt.savefig('img/video_%.5d.png'%self.iteration)
