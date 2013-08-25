'''
	Ken Caluwaerts 2012-2013
	<ken@caluwaerts.eu>
'''
import numpy as np
import scipy.io as sio

def create_icosahedron(height=1.,payloadR=0.3):
    '''
    Creates a tensegrity icosahedron
    '''
    #create points

    points = np.zeros((3,2*6+2))
    phi = (1+np.sqrt(5))*0.5
    offest = 0.792
    points[:,0] = (-phi,0,1*offest)
    points[:,1] = (phi,0,1*offest)
    points[:,2] = (-phi,0,-1*offest)
    points[:,3] = (phi,0,-1*offest)
    points[:,4] = (1*offest,-phi,0)
    points[:,5] = (1*offest,phi,0)
    points[:,6] = (-1*offest,-phi,0)
    points[:,7] = (-1*offest,phi,0)
    points[:,8] = (0,1*offest,-phi)
    points[:,9] = (0,1*offest,phi)
    points[:,10] = (0,-1*offest,-phi)
    points[:,11] = (0,-1*offest,phi)
    points[:,12] = (0,0,(payloadR/(height*0.62*0.5)))
    points[:,13] = (0,0,(-payloadR/(height*0.62*0.5)))
    
    #theta = -0.36486 #theta=-20.905 degrees
    #yRot = np.array([[np.cos(theta),0.,np.sin(theta)],[0.,1.,0.],[-np.sin(theta),0.,np.cos(theta)]])
    #testPoints = np.dot(yRot,points)
    #print points
    #print testPoints
    #points = testPoints
     
    points *= (height*0.62)*0.5

    #scale the tensegrity and center it

    #create bars
    bars = np.zeros((6+1,6*2+2))
    bars[:,::2] = np.eye(6+1)
    bars[:,1::2] = -np.eye(6+1)

    #create springs
    springs = np.zeros((6*6,2*6+2))
    springs[0,(0,6)] = (1,-1)
    springs[1,(0,7)] = (1,-1)
    springs[2,(0,9)] = (1,-1)
    springs[3,(0,11)] = (1,-1)
    springs[4,(1,4)] = (1,-1)
    springs[5,(1,5)] = (1,-1)
    springs[6,(1,9)] = (1,-1)
    springs[7,(1,11)] = (1,-1)
    springs[8,(2,6)] = (1,-1)
    springs[9,(2,7)] = (1,-1)
    springs[10,(2,8)] = (1,-1)
    springs[11,(2,10)] = (1,-1)
    springs[12,(3,4)] = (1,-1)
    springs[13,(3,5)] = (1,-1)
    springs[14,(3,8)] = (1,-1)
    springs[15,(3,10)] = (1,-1)
    springs[16,(4,10)] = (1,-1)
    springs[17,(4,11)] = (1,-1)
    springs[18,(5,8)] = (1,-1)
    springs[19,(5,9)] = (1,-1)
    springs[20,(6,10)] = (1,-1)
    springs[21,(6,11)] = (1,-1)
    springs[22,(7,8)] = (1,-1)
    springs[23,(7,9)] = (1,-1)
    springs[24,(0,12)] = (1,-1)
    springs[25,(1,12)] = (1,-1)
    springs[26,(2,13)] = (1,-1)
    springs[27,(3,13)] = (1,-1)
    springs[28,(6,13)] = (1,-1)
    springs[29,(5,13)] = (1,-1)
    springs[30,(4,12)] = (1,-1)
    springs[31,(7,12)] = (1,-1)
    springs[32,(8,13)] = (1,-1)
    springs[33,(9,12)] = (1,-1)
    springs[34,(10,13)] = (1,-1)
    springs[35,(11,12)] = (1,-1)
#    springs[:num_struts,:num_struts] = -np.eye(num_struts)+np.roll(np.eye(num_struts),1,1) #ground layer
#    springs[num_struts:2*num_struts,num_struts:] = -np.eye(num_struts)+np.roll(np.eye(num_struts),1,1) #top layer
#    springs[2*num_struts:,:num_struts] = -np.eye(num_struts) #connections between layers
#    springs[2*num_struts:,num_struts:] = np.roll(np.eye(num_struts),1,1)
    return points,bars,springs


