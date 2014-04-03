'''
Created on Aug 4, 2011

@author: kcaluwae
'''
import mpl_toolkits.mplot3d.axes3d as p3 #3d plots
import numpy as np
from matplotlib import pylab as plt
plt.ion()

def plot_tensegrity(points,bars,springs,actuators=None,springprops=None,barprops=None,tagged_actuators=None,ax=None,show_nodes=True,spring_color='g'):
    '''
    Creates a 3d plot of a tensegrity structure
    '''
    #plot the robot
    #if you want to debug the springs, plot them!
    num_springs = springs.shape[0]
    num_bars = bars.shape[0]
    if(actuators is None):
        num_actuators = 0
    else:
        num_actuators = actuators.shape[0]
    if(ax is None):
        ax = p3.Axes3D(plt.figure())
    ax.set_aspect('equal')
    #ax.scatter3D(points[0,:],points[1,:],points[2,:],c='yellow')
    for j in range(num_springs):
        p_a = points[:,springs[j].argmin()]
        p_b = points[:,springs[j].argmax()]
        line = np.vstack((p_a,p_b))
        if(springprops is None):
            ax.plot(line[:,0],line[:,1],line[:,2],c=spring_color)
        elif(type(springprops) is np.ndarray):
            ax.plot(line[:,0],line[:,1],line[:,2],'--',c=spring_color,linewidth=springprops[j])
        else:
            ax.plot(line[:,0],line[:,1],line[:,2],'--',c=spring_color)

    for j in range(num_bars):
        p_a = points[:,bars[j].argmin()]
        p_b = points[:,bars[j].argmax()]
        line = np.vstack((p_a,p_b))
        #ax.plot(line[:,0],line[:,1],line[:,2],c='red',linewidth=3)
        if(barprops is None):
            ax.plot(line[:,0],line[:,1],line[:,2],c='red',linewidth=3)
        elif(barprops == True):
            ax.plot(line[:,0],line[:,1],line[:,2],c='blue',linewidth=3)
        else:
            ax.plot(line[:,0],line[:,1],line[:,2],c='cyan',linewidth=3)
        
    for j in range(num_actuators):
        p_a = points[:,actuators[j].argmin()]
        p_b = points[:,actuators[j].argmax()]
        line = np.vstack((p_a,p_b))
        if(not tagged_actuators is None and tagged_actuators[j]):
            ax.plot(line[:,0],line[:,1],line[:,2],'b--',linewidth=3)
        else:
            ax.plot(line[:,0],line[:,1],line[:,2],c='blue')
    if(show_nodes):
        for j in xrange(points.T.shape[0]):
            ax.text3D(points.T[j,0],points.T[j,1],points.T[j,2],'%d'%j)
    ax.set_aspect('equal')    
    return ax
