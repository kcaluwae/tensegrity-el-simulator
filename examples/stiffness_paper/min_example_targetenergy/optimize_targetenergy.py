'''
	Ken Caluwaerts
	<ken@caluwaerts.eu>
	Juan Pablo Carbajal
	<ajuanpi@gmail.com>
	Ghent University 2013-2014
	
	Reproduces Figure 5 from
	"Energy Conserving Constant Shape Optimization of Tensegrity Structures"
	Ken Caluwaerts, Juan Pablo Carbajal 2014
	
	36 tensile members are added to a minimal 6 strut tensegrity (18 springs
	in the original configuration).
	Starting from a random force density assignment, we tune the elastic potential
	energy in a target deformation of the structure subject to the constraint that
	the tuning requires zero net mechanical work.	
	
	See Calladine 1978 for a discussion of the properties of the minimal structure.
'''
import numpy as np
import scipy.io as sio
import scipy.optimize as so
data = sio.loadmat("gd_nullspace_targetenergy_data.mat")
do_makeplots = True #use matplotlib to plot the results?
do_plot3D_result = False #make a 3D plot of the structures (not very informative)
q_lower_limit = 5 #N/m soft lower limit of force densities during optimization

'''
    Code example from: http://wiki.scipy.org/Cookbook/RankNullspace
'''
def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = np.max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


#initial spring rest lengths
spring_l0 = data["spring_l0"].ravel() 
#spring stiffnesses (k)
spring_rates = data["spring_rates"].ravel() #np.ones(18)*100.+np.random.randn(18)*5 #some value for this example

#this matrix contains a 1 for a spring and a 2 for a bar (connection pattern of the minimal 6 strut tensegrity)
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
#Build the bar and spring connectivity matrices based on A
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
            
l0 = spring_l0.copy()
C = data["C"]
#Nodal coordinates (spatial configuration of the structure)
sim_nodes_eucl = data["sim_nodes_eucl"] 

Ax = C.T.dot(np.diag(C.dot(sim_nodes_eucl[:,0:1]).ravel()))
Ay = C.T.dot(np.diag(C.dot(sim_nodes_eucl[:,1:2]).ravel()))
Az = C.T.dot(np.diag(C.dot(sim_nodes_eucl[:,2:3]).ravel()))
A = np.vstack((Ax,Ay,Az))
B = data["sim_bar_connections"]
sim_bar_connections = B.copy()
#CB contains the expanded connectivity matrix (6struts+54 springs)
CB = data["CB"]
       
#Compute the equilibrium matrix
Abx = CB.T.dot(np.diag(CB.dot(sim_nodes_eucl[:,0:1]).ravel()))
Aby = CB.T.dot(np.diag(CB.dot(sim_nodes_eucl[:,1:2]).ravel()))
Abz = CB.T.dot(np.diag(CB.dot(sim_nodes_eucl[:,2:3]).ravel()))
Ab = np.vstack((Abx,Aby,Abz))

#load the initial assignment of force densities (expanded)
q0 = data["q0"] 

#Find the nullspace
V = nullspace(Ab) #FULL V (with bars)
if(V.shape[1]<2):
    print "Nullspace has dimension 0 or 1"
    
#simple test to verify that q0 is indeed in the nullspace
c0 = np.linalg.lstsq(V,q0.ravel())[0].reshape((-1,1))
if(np.linalg.norm(V.dot(c0)-q0)>1e-8):
    print "Cannot find a good c0 for the given force densities: weird!"


#compute current lengths of the members
l = np.apply_along_axis(np.linalg.norm,1,CB.dot(sim_nodes_eucl)).reshape((-1,1))
l0 = data["l0"]
#assign stiffnesses to the (extended) members
k = q0*l/(l-l0)

#Load the target configuration (nodal coordinates)
target = data["target"]

#everything below disregards the potential energy stored in the bars
ks = k[sim_bar_connections.shape[0]:]
ls = l[sim_bar_connections.shape[0]:]
Vs = V[sim_bar_connections.shape[0]:] #disregard bars for potential energy
alpha = ls**2/(2*ks)
alpha_diag = np.diag(alpha.ravel())
B = Vs.T.dot(alpha_diag).dot(Vs) #REDEFINED! (This is B from the paper)

#compute lengths in the target configuration
ltarget = np.apply_along_axis(np.linalg.norm,1,CB.dot(target)).reshape((-1,1))
g=((ltarget-l)/l)[sim_bar_connections.shape[0]:] #modified stiffness
a = 0.5*g.T.dot(np.diag((ks*ls**2).ravel())).dot(g)[0,0]
s = Vs.T.dot(np.diag((ls**2).ravel())).dot(g)

U_original = lambda c: c.T.dot(B).dot(c)[0,0] #Potential energy in the original configuration
U_target = lambda c: (a+s.T.dot(c)+c.T.dot(B).dot(c))[0,0]  #potential energy in the target configuration
uc0 = U_original(c0)
uc0sqrt = np.sqrt(uc0)
utargetc0 = U_target(c0)
#Equation 37
y = lambda c, mu: c+mu*s+2*mu*B.dot(c) 
x = lambda y, _lambda: np.linalg.inv(np.eye(B.shape[0])+_lambda*B).dot(y) #replace with solve or lstsq if needed!
#you can check that it works by e.g. using x(y(c0,0),0) = c0
assert np.linalg.norm(x(y(c0,0),0)-c0)<1e-10

#eigen decomposition of B to speed up things (Eq. 33)
d,P = np.linalg.eigh(B)
d = d.reshape((-1,1))
diag_d = np.diag(d.ravel())
diag_sqrt_d = np.diag(np.sqrt(d.ravel()))
diag_sqrt_d_PT = diag_sqrt_d.dot(P.T)

#Eq. 33-35
z = lambda y: diag_sqrt_d_PT.dot(y)
lambda_err_ynorm = lambda y,_lambda: np.linalg.norm(z(y)/(1+_lambda*d))
lambda_err = lambda y, _lambda: (lambda_err_ynorm(y,_lambda)-uc0sqrt)**2  #this should be plenty fast
lambda_err_grad = lambda y, _lambda: -2*(1-uc0sqrt/lambda_err_ynorm(y,_lambda))*z(y).T.dot(z(y)*d/(1+_lambda*d)**3)[0] #needs to be a vector for scipy optimize
lambda_err_and_grad = lambda y, _lambda: (lambda_err(y,_lambda),lambda_err_grad(y,_lambda))

#line search for the optimal Langrange multiplier value (lambda)
def find_lambda(y):
    err_f = lambda _lambda: lambda_err_and_grad(y,_lambda)
    res = so.minimize(err_f,[0],tol=1e-12,jac=True,method='TNC') #Truncated Newton is fastest in practice
    if(not res.success):
            print "WARNING: lambda optimization not successful!"
            
    return res.x[0], res.fun
    
def next_c(c_t,mu,debug=True):
    '''
        Compute c_t+1, based on current c_t
    '''
    #compute y, correct force densities below the lower limit (q_lower_limit N/m)
    #This is a soft lower limit 
    #Note that if it slightly violates the lower limit constraint in this iteration,
    #then it will be corrected in the next.
    _y = y(c_t,mu)
    _y = np.linalg.lstsq(Vs,np.where(Vs.dot(_y)<q_lower_limit,q_lower_limit,Vs.dot(_y)))[0] 
    #compute lambda
    _lambda, fun = find_lambda(_y)
    if(fun>1e-12):
        print "warning, error tolerance not met"
    _x = x(_y,_lambda) #c_t+1
    c_t_plus_1 = _x
    #do some checks
    #pot energy in c_t
    if(debug):
        Uc_t = U_target(c_t) 
        Ux = U_target(_x)
        Ux0 = U_original(_x)
        print "Utarget(c0): %f\tUtarget(c_t): %f\tUtarget(c_t+1): %f\tU0(c_t): %f\tU0(c_t+1): %f"%(utargetc0,Uc_t,Ux,uc0,Ux0)
    #pot energy in c_t+1
    return _x

#run gradient ascent/descent (main loop)
_iter = 0
c_t = c0
c_t_hist = []
c_t_min = 0
c_t_max = 0
utarget_min = U_target(c0)
utarget_max = utarget_min
while(_iter<150000):
    if(_iter<50000):
        mu = 5  #gradient ascent (max)
    elif(_iter<100000):
        mu = -5#gradient descent (min)
    else:
        mu = 2 #gradient ascent with slower learning
    debug = (_iter%10==0) 
    utarget_c_t = U_target(c_t) #the optimization goal: elastic energy in the target configuration
    if(utarget_c_t>utarget_max):
        c_t_max = c_t
        utarget_max = utarget_c_t
    elif(utarget_c_t<utarget_min):
        c_t_min = c_t
        utarget_min = utarget_c_t
    if(debug):
        c_t_hist.append((_iter,c_t,utarget_c_t))
    c_t = next_c(c_t,mu,debug)
    _iter+=1
    
#compute new spring properties (final)
#assign stiffnesses to the members
q0_t = V.dot(c_t)
k_t = k.copy()
l0_t = l*(1-q0_t/k_t) #just a choice 
ks_t = k_t[sim_bar_connections.shape[0]:]
l0s_t = l0_t[sim_bar_connections.shape[0]:]

numB = sim_bar_connections.shape[0]

if do_plot3D_result:
    import plotrobot
    ax1 = plotrobot.plot_tensegrity(sim_nodes_eucl.T,sim_bar_connections,CB[numB:],springprops=(ls*Vs.dot(c_t_min)).ravel()/50) #minimal in original config
    ax2 = plotrobot.plot_tensegrity(target.T,sim_bar_connections,CB[numB:],springprops=(ltarget[numB:]*Vs.dot(c_t_min)).ravel()/50) #minimal in target config
    ax3 = plotrobot.plot_tensegrity(target.T,sim_bar_connections,CB[numB:],springprops=(ltarget[numB:]*Vs.dot(c_t_max)).ravel()/50) #maximal in target config
    ax4 = plotrobot.plot_tensegrity(sim_nodes_eucl.T,sim_bar_connections,CB[numB:],springprops=(ls*Vs.dot(c_t_max)).ravel()/50) #maximal in original config
    
    axs = [ax1,ax2,ax3,ax4]
    for ax in axs:
        ax.set_xlim(-.4,.4)
        ax.set_ylim(-.4,.4)
        ax.set_zlim(-.4,.4)
        #stupid matplotlib aspect ratio...
        ax.set_aspect('equal')
        MAX = .8
        for direction in (-1, 1):
            for point in np.diag(direction * MAX * np.array([1,1,1])):
                ax.plot([point[0]], [point[1]], [point[2]], 'w')
            
#show the behavior of the springs during optimization
utarget_evol = np.array([n[2] for n in c_t_hist])
c_evol =  np.array([n[1].ravel() for n in c_t_hist])
qs_evol = Vs.dot(c_evol.T).T
fs_evol = qs_evol*ls.T
f0s = q0[sim_bar_connections.shape[0]:]*ls
l0s_evol = (1-qs_evol/ks.T)*ls.T
iters = l0s_evol.shape[0]
iters_range = np.arange(iters)*10

if do_makeplots:
    import matplotlib.pylab as plt
    plt.ion()
    
    plt.figure()
    plt.imshow((((l0s_evol[::]-ls.T)**2)*ks.T*0.5).T,interpolation='nearest',cmap=plt.cm.gray_r,aspect='auto')
    plt.colorbar()
    plt.ylabel("spring number")
    plt.xlabel("iteration/10 ($t$)")
    plt.title("Evolution of spring potential energy (J)")
    
    plt.figure()
    plt.title("Evolution of rest length, spring force and potential energy")
    
    plt.plot(iters_range,np.mean(l0s_evol-l0[sim_bar_connections.shape[0]:].T,1)*100,'b--',lw=3,label='$\overline{l_0(t)-l_0(0)}$ (cm)')
    plt.plot(iters_range,np.mean(fs_evol-f0s.T,1),'r-',lw=3,label='$\overline{f(t)-f(0)}$ (N)')
    plt.plot(iters_range,utarget_evol-uc0,'k',lw=3,label='$U_{target}(t)-U_{orig}$ (J)')
    plt.xlabel("iteration ($t$)")
    plt.ylabel("black: J, blue: cm, red: N")
    plt.legend(loc=4)
