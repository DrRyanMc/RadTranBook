#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#this next line is only needed in iPython notebooks
#get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.interpolate import BPoly
import math
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc', size = 20)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def hide_spines(intx=False,inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font) 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
           # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font)
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font)
            #ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel(), fontproperties = font)
            ax.set_ylabel(ax.get_ylabel(), fontproperties = font)
            ax.set_title(ax.get_title(), fontproperties = font)
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
def show(nm,a=0,b=0):
    hide_spines(a,b)
    #ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    #plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    if (len(nm)>0):
        plt.savefig(nm+".pdf",bbox_inches='tight');
    plt.show()
    
    

c = 300.0 # speed of light
a = 0.01372 #radiation constant
ac = a*c
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import math
from tabulate import tabulate
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
def sweep1D_bern1(I,hx,q,sigma_t,mu,boundary):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
    Outputs:
        psi:             value of angular flux in each zone
    """
    
    
    psi = np.zeros((I,2))
    ihx = 1/hx
    h = hx
    M = np.array([[1./9,1./18],[1./18,4./63]])*h
    K = np.array([[-0.5,-0.5],[0.5,0.5]])
    if (mu > 0): 
        psi_left = boundary
        for i in range(I):
            iminhalf = np.array([0,1])
            iplushalf = np.array([0,1])
            rhs = np.dot(M,q[i,:]) + mu*np.dot(np.array([[0,1],[0,0]]),psi_left)
            lhs = M*sigma_t[i] - mu*K + np.diag(iplushalf)*mu
            tmp = np.linalg.solve(lhs,rhs)
            psi[i,:] = tmp.reshape(2) 
            psi_left = tmp
    else:
        psi_right = boundary
        iminhalf = np.array([1,0])
        iplushalf = np.array([1,0])
        for i in reversed(range(I)):
            rhs =  - mu*np.dot(np.array([[0,0],[1,0]]),psi_right) + np.dot(M,q[i,:])
            lhs = M*sigma_t[i] - mu*K - np.diag(iminhalf)*mu
            tmp = np.linalg.solve(lhs,rhs)
            #print(rhs,lhs,tmp)
            psi[i,:] = tmp.reshape(2)
            psi_right = tmp
    return psi


# In[3]:


from numba import jit, int32, float64
@jit(float64[:,:](int32,float64,float64[:,:],float64[:,:],float64, float64[:],int32,int32), nopython=True)
def sweep1D_bern(I,hx,q,sigma_t,mu,boundary,order=1,fix=0):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
    Outputs:
        psi:             value of angular flux in each zone
    """
    
    lumped = 1
    psi = np.zeros((I,order+1))
    ihx = 1/hx
    h = hx
    if (order==1):
        M = np.array([[0.333333, 0.166667],[0.166667, 0.333333]])*h
        K = np.array([[-0.5,-0.5],[0.5,0.5]])
    elif (order == 2):
        M = np.array([[0.2, 0.1, 0.0333333], 
                      [0.1, 0.133333, 0.1], 
                      [0.0333333, 0.1, 0.2]])*h
        K = np.array([[-0.5, -0.333333, -0.166667], 
                      [0.333333, 0., -0.333333], 
                      [0.166667, 0.333333, 0.5]])
    
    elif (order == 3):
        M = np.array([[0.142857, 0.0714286, 0.0285714, 0.00714286], 
                      [0.0714286, 0.0857143,   0.0642857, 0.0285714], 
                      [0.0285714, 0.0642857, 0.0857143,   0.0714286], 
                      [0.00714286, 0.0285714, 0.0714286, 0.142857]])*h
        K = np.array([[-0.5, -0.3, -0.15, -0.05], [0.3, 0., -0.15, -0.15], [0.15, 0.15, 0., -0.3], [0.05, 0.15, 0.3, 0.5]])
    elif (order == 4):
        M = np.array([[0.111111, 0.0555556, 0.0238095, 0.00793651, 0.0015873],
                     [0.0555556, 0.0634921, 0.047619, 0.0253968, 0.00793651],
                     [0.0238095, 0.047619, 0.0571429, 0.047619, 0.0238095],
                     [0.00793651, 0.0253968, 0.047619, 0.0634921, 0.0555556],
                     [0.0015873, 0.00793651, 0.0238095, 0.0555556, 0.111111]])*h
        K = np.array([[-0.5, -0.285714, -0.142857, -0.0571429, -0.0142857], 
                      [0.285714, 0., -0.114286, -0.114286, -0.0571429],
                      [0.142857, 0.114286, 0., -0.114286, -0.142857],
                      [0.0571429, 0.114286, 0.114286, 0., -0.285714],
                      [0.0142857, 0.0571429, 0.142857, 0.285714, 0.5]])
    elif (order == 5):
        M = np.array([[0.0909091, 0.0454545, 0.020202, 0.00757576, 0.0021645, 0.00036075],
                     [0.0454545, 0.0505051, 0.0378788, 0.021645, 0.00901876, 0.0021645],
                     [0.020202, 0.0378788, 0.04329, 0.036075, 0.021645, 0.00757576],
                     [0.00757576, 0.021645, 0.036075, 0.04329, 0.0378788, 0.020202],
                     [0.0021645, 0.00901876, 0.021645, 0.0378788, 0.0505051, 0.0454545],
                     [0.00036075, 0.0021645, 0.00757576, 0.020202, 0.0454545, 0.0909091]])*h
        K = np.array([[-0.5, -0.277778, -0.138889, -0.0595238, -0.0198413, -0.00396825], 
                      [0.277778, 0., -0.0992063, -0.0992063, -0.0595238, -0.0198413],
                      [0.138889, 0.0992063, 0., -0.0793651, -0.0992063, -0.0595238],
                      [0.0595238, 0.0992063, 0.0793651, 0., -0.0992063, -0.138889],
                      [0.0198413, 0.0595238, 0.0992063, 0.0992063, 0., -0.277778],
                      [0.00396825, 0.0198413, 0.0595238, 0.138889, 0.277778, 0.5]])
    elif (order == 6):
        M = np.array([[0.0769231, 0.0384615, 0.0174825, 0.00699301, 0.002331, 0.000582751,   0.0000832501], 
                      [0.0384615, 0.041958, 0.0314685, 0.018648,   0.00874126, 0.002997, 0.000582751], 
                      [0.0174825, 0.0314685, 0.034965,   0.0291375, 0.0187313, 0.00874126, 0.002331], 
                      [0.00699301, 0.018648,   0.0291375, 0.0333, 0.0291375, 0.018648, 0.00699301], 
                      [0.002331,   0.00874126, 0.0187313, 0.0291375, 0.034965, 0.0314685,   0.0174825], 
                      [0.000582751, 0.002997, 0.00874126, 0.018648, 0.0314685,   0.041958, 0.0384615], 
                      [0.0000832501, 0.000582751, 0.002331,   0.00699301, 0.0174825, 0.0384615, 0.0769231]])*h
        K = np.array([[-0.5, -0.272727, -0.136364, -0.0606061, -0.0227273, -0.00649351,-0.00108225],
                      [0.272727,   0., -0.0909091, -0.0909091, -0.0584416, -0.025974, -0.00649351],
                      [0.136364, 0.0909091,   0., -0.0649351, -0.0811688, -0.0584416, -0.0227273], 
                      [0.0606061,   0.0909091, 0.0649351,   0., -0.0649351, -0.0909091, -0.0606061], 
                      [0.0227273, 0.0584416,   0.0811688, 0.0649351, 0., -0.0909091, -0.136364], 
                      [0.00649351,   0.025974, 0.0584416, 0.0909091, 0.0909091,   0., -0.272727], 
                      [0.00108225, 0.00649351, 0.0227273, 0.0606061, 0.136364, 0.272727, 0.5]])
    if ( lumped > 0):
        M_new = M*0
        K_new = K*0
        for i in range(order+1):
            M_new[i,i] = np.sum(M[i,:])
            K_new[i,i] = np.sum(K[i,:])
        M = M_new
        #K = K_new
    if (mu > 0): 
        psi_left = boundary
        for i in range(I):
            if (order == 4):
                iminhalf = np.array([0,0,0,0,1.0])
                iplushalf = np.array([0,0,0,0,1.0])
            elif (order == 1):
                iminhalf = np.array([0,1.])
                iplushalf = np.array([0,1.])
            else:
                iminhalf = np.zeros(order+1)
                iminhalf[-1] = 1
                iplushalf = np.zeros(order+1)
                iplushalf[-1] = 1
                      
            rhs = np.dot(M,q[i,:]) 
            rhs[0] += mu*psi_left[-1] 
            lhs = M*np.diag(sigma_t[i]) - mu*K + np.diag(iplushalf)*mu
            tmp = np.linalg.solve(lhs,rhs)
            if (fix >0) and (np.min(tmp)<0):
                tmpZ = (tmp>0)*tmp
                tmp = tmpZ*(np.sum(rhs)/(np.sum(np.dot(lhs,tmpZ))+1e-14))
            psi[i,:] = tmp.reshape(order+1) 
            psi_left = tmp
    else:
        psi_right = boundary
        for i in range(I):
            if (order == 4):
                iminhalf = np.array([1.0,0,0,0,0])
                iplushalf = np.array([1.0,0,0,0,0,0])
            elif (order == 1):
                iminhalf = np.array([1.0,0])
                iplushalf = np.array([1.0,0])
            else:
                iminhalf = np.zeros(order+1)
                iminhalf[0] = 1
                iplushalf = np.zeros(order+1)
                iplushalf[0] = 1
        #iminhalf = np.array([1,0])
        #iplushalf = np.array([1,0])
        for i in range(I-1,-1,-1):
            rhs =   np.dot(M,q[i,:])
            rhs[-1] += - mu*psi_right[0]
            lhs = M*np.diag(sigma_t[i]) - mu*K - np.diag(iminhalf)*mu
            tmp = np.linalg.solve(lhs,rhs)
            #print(lhs,rhs,tmp)
            if (fix >0) and (np.min(tmp)<0):
                tmpZ = (tmp>0)*tmp
                tmp = tmpZ*(np.sum(rhs)/(np.sum(np.dot(lhs,tmpZ))+1e-14))
            psi[i,:] = tmp.reshape(order+1)
            psi_right = tmp
    return psi


# In[4]:


def source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, phi=np.zeros(1),
                     Linf_tol = 1.0e-5, tolerance = 1.0e-8,maxits = 100, 
                     LOUD=False, plot = False, DSA=False, order=4, fix=0 ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi:             value of scalar flux in each zone
    """
    if (phi.size != I*(order+1)):
        phi = np.zeros((I,order+1))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    W /= np.sum(W)
    psi = np.zeros((I,N,order+1))
    psi[0,:,:] = phi[0,:]/np.sum(W)
    psi[-1,:,:] = phi[-1,:]/np.sum(W)
    iteration = 1
    if plot:
        plotvar = np.zeros(1)
        plotvar[0] = np.mean(phi_old)
    while not(converged):
        phi = np.zeros((I,order+1))
        #sweep over each direction
        for n in range(N):
            #deal with reflecting BC
            if ((BCs[n,0] < 0) and (n>=N//2)):
                tmpBC = psi[-1,n-N//2,:]
            elif ((BCs[n,0] < 0) and (n<N//2)):
                tmpBC = psi[0,n+N//2,:]
            else:
                tmpBC = BCs[n,:]
            source = q.copy()
            for node in range(order+1):
                source[:,n,node] += phi_old[:,node]*sigma_s[:,node] 
            tmp_psi = sweep1D_bern(I,hx,np.array(source[:,n,:]),np.array(sigma_t),MU[n],tmpBC,order=order,fix=fix)
            psi[:,n,:] = tmp_psi.copy()
            phi += tmp_psi*W[n]
        if plot:
            plotvar = np.append(plotvar,np.mean(phi))
        #check convergence
        
        L2err = np.sum((phi_old.reshape(I*(order+1))/phi.reshape(I*(order+1))  - 1)**2/math.sqrt(I))
        if (iteration != 1):
            change = np.append(change,L2err)
        else:
            change = np.zeros(1)+L2err
        Linferr = np.max(np.abs(phi_old/phi-1))
        converged = ((L2err < tolerance) and (Linferr < Linf_tol)) or (iteration > maxits)
        if (LOUD>0) or (converged and LOUD<0):
            print("Iteration",iteration,": Relative Change =",L2err,Linferr)
        if (iteration > maxits) and ( not(DSA)):
            print("Warning: Source Iteration did not converge")
        iteration += 1
        phi_old = phi.copy()
    x = np.linspace(hx/2,I*hx-hx/2,I)
    if DSA:
        return x, phi, psi, iteration-1
    if plot:
        return x,phi,iteration-1,plotvar
    
                
                
        
    return x, phi, iteration-1,change


# In[5]:


def temp_solve(I,hx,q,sigma_func,N,BCs, Cv, phi,psi, T,
               dt_val = 0.001, tfinal = 1.0,
                     Linf_tol = 1.0e-5, tolerance = 1.0e-8,maxits = 1000, 
                     LOUD=False, plot = False,order=4, fix=0 ):
    t_current = 0.0
    c = 300.0 # speed of light
    a = 0.01372 #radiation constant
    ac = a*c
    phis = []
    Ts = []
    phi_old = phi.copy()
    print(psi.shape)
    psi_old = psi.copy()
    T_old = T.copy()
    phis.append(phi_old)
    Ts.append(T_old)
    print("|", end = '')
    curr_step = 0
    iterations = 0
    while (t_current < tfinal):
        dt = np.min([dt_val,tfinal-t_current])
        t_current += dt
        if (int(10*t_current/tfinal) > curr_step):
            curr_step += 1
            print(curr_step, end = '')
        icdt = 1.0/(c*dt)
        beta = 4*a*T_old**3/Cv
        sigma = sigma_func(T_old)
        f = 1.0/(1+beta*c*dt*sigma)
        sigma_a = f*sigma
        sigma_s = (1-f)*sigma
        source = q + (sigma_a*ac*T_old**4)[:,None,:] + icdt*psi_old 
        sigma_t = sigma+icdt
        """print("f = ",f)
        print("sigma_t = ",sigma_t)
        print("sigma_s = ",sigma_s)
        print("source = ",source)
        """
        x, phi, psi, iteration = source_iteration(I,hx,source,sigma_t,sigma_s,N,BCs, phi=phi_old,
                     Linf_tol = Linf_tol, tolerance = tolerance,maxits = maxits, 
                     LOUD=LOUD, plot = plot, DSA=True, order=order, fix=fix )
        iterations += iteration
        T = T_old + sigma_a*dt*(phi-ac*T_old**4)/Cv
        #print(T)
        T_old = T.copy()
        phi_old = phi.copy()
        psi_old = psi.copy()
        phis.append(phi_old)
        Ts.append(T_old)
    return x, phis, Ts, iterations


# In[6]:


def DMD_prec(matvec, b, K = 10, steady = 0, x = np.zeros(1), step_size = 10, GM = 0, res=1):
    res = np.min([1.0e-6,res])
    res = np.max([res,1e-11])
    #print(res)
    N = b.size
    linf = 0
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    x_new = x*0
    x_orig = x.copy()
    x_0 = x.copy()
    #perform K iterations of matvec
    Yplus = np.zeros((N,K-1))
    Yminus = np.zeros((N,K-1))
    
    for k in range(K):
        x_new = matvec(x) + b 
        L2err = np.sum((x/x_new  - 1)**2/math.sqrt(N))
        Linferr = np.max(np.abs(x/x_new-1))
        if (k == 0):
            change = np.zeros(1) + L2err
            change_linf= np.zeros(1) + Linferr
        else:
            change = np.append(change,L2err)
            change_linf = np.append(change,Linferr)
        if (k < K-1):
            Yminus[:,k] = x_new - x
            
            x_0 = x_new.copy()
        if (k>0):
            Yplus[:,k-1] = x_new-x
        
        x = x_new.copy()
    #now perform update
    
    #compute svd
    [u,s,v] = np.linalg.svd(Yminus,full_matrices=False)
    #print("U shape =", u.shape, "V shape =", v.shape)
    #find the non-zero singular values
    if (x.size > 1) and (s[(1-np.cumsum(s)/np.sum(s)) > (1.e-3)*res].size >= 1):
        spos = s[(1-np.cumsum(s)/np.sum(s)) > (1.e-3)*res].copy()
    else:
        spos = s[s>0].copy()
    #create diagonal matrix
    mat_size = np.min([K,len(spos)])
    S = np.zeros((mat_size,mat_size))
    #select the u and v that correspond with the nonzero singular values
    unew = 1.0*u[:,0:mat_size]
    vnew = 1.0*v[0:mat_size,:]
    #S will be the inverse of the singular value diagonal matrix 
    S[np.diag_indices(mat_size)] = 1/spos
    #not sure we need this
    En = u.copy() 

    #the approximate A operator is Ut A U = Ut Y+ V S
    part1 = np.dot(np.matrix(unew).getH(),Yplus)
    part2 = np.dot(part1,np.matrix(vnew).getH())
    Atilde = np.dot(part2,np.matrix(S).getH())
    if (Atilde.shape[0] > 0):
        try:
            [eigsN,vsN] = np.linalg.eig(Atilde)
            if (np.max(np.abs(eigsN))>1):
                #an eigenvalue is too big
                print("*****Warning*****  The number of steps may be too small")
                eigsN[np.abs(eigsN) > 1] = 0
            eigsN = np.real(eigsN)
            #change Atilde to only have the right eigenvalues
            Atilde = np.real(np.dot(np.dot(vsN,np.diag(eigsN)),np.linalg.inv(vsN)))

            if steady:
                Z = np.dot(unew,vsN)
                Zdagger = np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
                rhs = np.dot(np.matrix(unew).getH(),Yplus[:,-1])
                delta_y = np.linalg.solve(np.identity(Atilde.shape[0]) - Atilde,np.transpose(rhs))
                x_old = - (Yplus[:,K-1-1] - x)
                steady_update = x_old + np.transpose(np.dot(unew,delta_y))
                return steady_update, change, change_linf, Atilde,Yplus,Yminus
            else:

                Z = np.dot(unew,vsN)
                Zdagger = np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
                rhs = x
                step_1 = np.dot(Zdagger,Yplus[:,-1]).getH()
                step_2 = np.linalg.solve(np.identity(Atilde.shape[0])- np.diag(eigsN), step_1)
                step_3 = np.dot(np.identity(Atilde.shape[0])- np.diag(eigsN**step_size), step_2)
                step_4 = np.dot(Z,step_3)
                
                x_old = - (Yplus[:,K-1-1] - x)
                nonsteady = np.zeros(N)
                nonsteady[0:N] = x_old + np.transpose(step_4)
                return nonsteady, change
        except Exception as e:
            print("There is an unexpected problem",e)
            return x, change, linf
    else:
        print(spos)


# In[7]:


def one_incSVD(u,W,sigma,V,r,k,eps=1e-18, eps_sv = 1e-12):
    #Use first column of A to start, but keep going until we find a large enough column
    found_basis = r;
    rows = W.shape[0]
    if len(sigma)==0:
        sigma = np.zeros(1)
    if (r==0):
        while not(found_basis):
            #if (k>0):
                #print("Warning: Too small")
            sigma_tmp = np.array(np.linalg.norm(u))
            if (sigma_tmp > eps):
                found_basis = 1;
                sigma[0] = sigma_tmp
                #print((u/sigma_tmp).shape,W[:,0].shape)
                W[:,0] = u/sigma_tmp
                V = np.ones(1)
            k += 1
        r = r+1
    else:
        #now we have found a starting place, let's add on
        ell = np.dot(W[:,0:r].transpose(),u)
       
        p = 0
        if (np.dot(u,u) - np.dot(ell,ell)) > eps:
            p = np.sqrt(np.dot(u,u) - np.dot(ell,ell))
            
        j = (u-np.dot(W[:,0:r],ell))
        if (p > eps):
            j /= p
        Q = np.zeros((r+1,r+1))
        Q[0:r,0:r] = np.diag(sigma[0:r])
        Q[0:r,r] = ell
        if (p > eps):
            Q[r,r] = p
        [Wbar, Sbar, Vbar] = np.linalg.svd(Q, full_matrices=True)
        Vbar = Vbar.transpose()
        if (p < eps):
            #print("Warning")
            #print(Vbar.shape)
            W = np.dot(W,Wbar[0:r,0:r])
            sigma = Sbar[0:r]
            Vnew = np.zeros((k+1, r+1))
            Vnew[0:k,0:r] = V.copy()
            Vnew[-1,-1] = 1
            V = np.dot(Vnew,Vbar[:,0:r])
            k += 1
        else:
            Wnew = np.zeros((W.shape[0],r+1))
            Wnew[:,0:r] = W.copy()
            Wnew[:,r] = j
            W = np.dot(Wnew, Wbar)
            sigma = Sbar.copy()
            Vnew = np.zeros((k+1, r+1))
            Vnew[0:k,0:r] = V.copy()
            Vnew[-1,-1] = 1
            #print("Vnew=",Vnew)
            V = np.dot(Vnew,Vbar)
            r += 1
            k += 1
        
        #truncate if small
        if (sigma[-1] < eps_sv):
            print("Truncating")
            #print(V,W,sigma)
            r -= 1
            V = V[:,0:r]
            W = W[:,0:r]
            sigma = sigma[0:r]
    
    #orthogonalize
    orth = np.dot(W[:,0].transpose(),W[:,-1])
    #if (orth > np.min([eps,1e-14*rows])) and (r>1):
        #print("Orthogonalizing")
        #print("*****",orth)
        #tmp = np.dot(np.dot(W,np.diag(sigma)),V.T)
        #[W,sigma,V] = np.linalg.svd(tmp,full_matrices=False)
        #[Q,R] = np.linalg.qr(W)
        #W = Q
        #[Q,R] = np.linalg.qr(V)
        #V = Q
    return W,sigma,V, r, k


# In[8]:


def DMD_prec_inc(matvec, b, K = 10, steady = 0, x = np.zeros(1), step_size = 10, GM = 0, res=1):
    res = np.min([1.0e-6,res])
    res = np.max([res,1e-11])
    #print(res)
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    x_new = x*0
    x_orig = x.copy()
    x_0 = x.copy()
    #perform K iterations of matvec
    Yplus = np.zeros((N,K-1))
    Yminus = np.zeros((N,K-1))
    r = 0
    k_val = 0
    
    update_old = x.copy()
    steady_update = x.copy()
    u = np.zeros((N,1))
    s = []
    v = [0]
    for k in range(K):
        x_new = matvec(x) + b 
        L2err = np.sum((x/x_new  - 1)**2/math.sqrt(N))
        Linferr = np.max(np.abs(x/x_new-1))
        if (k == 0):
            change = np.zeros(1) + L2err
            change_linf= np.zeros(1) + Linferr
        else:
            change = np.append(change,L2err)
            change_linf = np.append(change,Linferr)
        if (k < K-1):
            Yminus[:,k] = x_new - x
            
            x_0 = x_new.copy()
        if (k>0):
            Yplus[:,k-1] = x_new-x
            
            [u,s,v,r,k_val] = one_incSVD(Yminus[:,k-1],u,s,v,r=r,k=k_val, eps=res*1e-14, eps_sv=res*1e-14)
            #print("V shape",v.shape)
        
        x = x_new.copy()
        
        #now perform update
        if (k>1):
            vT = (v.T).copy()
            
            #[U,S,V] = np.linalg.svd(Yminus)
            #[u,s,vT] = np.linalg.svd(Yminus[:,0:k],full_matrices=False); r=1;k_val=1
            #find the non-zero singular values
            if (x.size > 1) and (s[(1-np.cumsum(s)/np.sum(s)) >= (1.e-6)*res].size >= 1):
                spos = s[(1-np.cumsum(s)/np.sum(s)) >= (1.e-6)*res].copy()
            else:
                spos = s[s>0].copy()
            #create diagonal matrix
            #print(spos)
            mat_size = np.min([K,len(spos)])
            S = np.zeros((mat_size,mat_size))

            #select the u and v that correspond with the nonzero singular values
            unew = 1.0*u[:,0:mat_size]
            vnew = 1.0*vT[0:mat_size,0:k]
            #S will be the inverse of the singular value diagonal matrix 
            S[np.diag_indices(mat_size)] = 1/spos
            #not sure we need this
            En = u.copy() 

            #the approximate A operator is Ut A U = Ut Y+ V S
            #print("unew size =", u.shape, "v size =", vnew.shape, "Y+ size =", Yplus[:,0:k].shape)
            part1 = np.dot(np.matrix(unew).getH(),Yplus[:,0:k])
            part2 = np.dot(part1,np.matrix(vnew).getH())
            Atilde = np.dot(part2,np.matrix(S).getH())
            if (Atilde.shape[0] > 0):
                try:
                    [eigsN,vsN] = np.linalg.eig(Atilde)
                    if (np.max(np.abs(eigsN))>1):
                        #an eigenvalue is too big
                        #print(spos)
                        print("*****Warning*****  The number of steps may be too small")
                        eigsN[np.abs(eigsN) > 1] = 0
                    eigsN = np.real(eigsN)
                    #change Atilde to only have the right eigenvalues
                    Atilde = np.real(np.dot(np.dot(vsN,np.diag(eigsN)),np.linalg.inv(vsN)))

                    #always steady here
                    Z = np.dot(unew,vsN)
                    Zdagger = np.linalg.solve(np.dot(Z.getH(),Z),Z.getH())
                    rhs = np.dot(np.matrix(unew).getH(),Yplus[:,k-1])
                    delta_y = np.linalg.solve(np.identity(Atilde.shape[0]) - Atilde,np.transpose(rhs))
                    x_old = - (Yplus[:,k-1] - x)
                    steady_update = x_old + np.transpose(np.dot(unew,delta_y))
                    #print("k =", k, "change =",np.linalg.norm(steady_update - update_old), "r =",r,"k_val =",k_val)
                    update_old = steady_update.copy()


                except Exception as e:
                    print("There is an unexpected problem",e)
                    print(spos)
                    return x, change, change_linf, Atilde, Yplus, Yminus
            if (k > r + 1):
                return steady_update, change, change_linf, Atilde, Yplus, Yminus
        
    return steady_update, change, change_linf, Atilde, Yplus, Yminus


# In[9]:


def single_source_iteration(I,hx,source,sigma_t,N,BCs, order=4, fix=0):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        phi:             value of scalar flux in each zone
    """
    phi = np.zeros((I,order+1))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    W /= np.sum(W)
    #sweep over each direction
    for n in range(N):
        #deal with reflecting BC
        if ((BCs[n,0] < 0) and MU[n]<0):
            tmpBC = psi[-1,n-N//2,:]
        elif ((BCs[n,0] < 0) and (MU[n]>0)):
            mu_arg = np.argmin(np.abs(-MU[n]-MU))
            #print("Refl BC: MU[%i]=%f, gets MU[%i]=%f" %(n,MU[n],n+N//2,MU[mu_arg]))
            tmpBC = psi[0,mu_arg,:]
        else:
            tmpBC = BCs[n,:]
        #print(phi.shape)
        #print(I,hx,tmpBC,MU[n],source.reshape((I,N,order+1))[:,n,:],sigma_t)
        tmp_psi = sweep1D_bern(I,hx,source.reshape((I,N,order+1))[:,n,:],sigma_t,MU[n],tmpBC,order=order,fix=fix)
        #print(phi.shape,tmp_psi.shape)
        phi += tmp_psi.reshape(phi.shape)*W[n]
    return phi

def single_source_iteration_psi(I,hx,source,sigma_t,N,BCs,order,fix=0):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        phi:             value of scalar flux in each zone
    """
    psi = np.zeros((I,N,order+1))
    phi = np.zeros((I,order+1))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    #sweep over each direction
    for n in range(N):
        #deal with reflecting BC
        if ((BCs[n,0] < 0) and (n>=N//2)):
            tmpBC = psi[-1,n-N//2,:]
        elif ((BCs[n,0] < 0) and (n<N//2)):
            tmpBC = psi[0,n+N//2,:]
        else:
            tmpBC = BCs[n,:]
        
        tmp_psi = sweep1D_bern(I,hx,source.reshape((I,N,order+1))[:,n,:],sigma_t,MU[n],tmpBC,order=order,fix=fix)
        psi[:,n,:] = tmp_psi
        phi += tmp_psi*W[n]
    return psi


def single_source_iteration_psi_phi(I,hx,psi,sigma_t,N,BCs,order,fix=0):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        phi:             value of scalar flux in each zone
    """
    #print(psi.shape)
    phi_old = np.zeros((I,order+1))
    MU, W = np.polynomial.legendre.leggauss(N)
    for n in range(N):
        phi_old += psi[:,n,:]*W[n]
    psi = np.zeros((I,N,order+1))
    phi = phi_old*0
    converged = False
    #sweep over each direction
    for n in range(N):
        #deal with reflecting BC
        if ((BCs[n,0] < 0) and (n>=N//2)):
            tmpBC = psi[-1,n-N//2,:]
        elif ((BCs[n,0] < 0) and (n<N//2)):
            tmpBC = psi[0,n+N//2,:]
        else:
            tmpBC = BCs[n,:]
        tmp_psi = sweep1D_bern(I,hx,phi_old.reshape((I,order+1)),sigma_t,MU[n],tmpBC,order=order,fix=fix)
        psi[:,n,:] = tmp_psi
        phi += tmp_psi*W[n]
    return psi


# In[10]:


def solver_with_dmd(matvec, b, K = 10, Rits = 2, steady = 1, x = np.zeros(1), step_size = 10, 
                    L2_tol = 1e-8, Linf_tol = 1e-3, max_its = 10, LOUD=0,order=4):
    #print(b.size,x.size)
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    iteration = 0
    converged = 0
    total_its = 0
    Atil = []
    Yplus = []
    Yminus = []
    while ((not(converged)) and (iteration < max_its)):
        for r in range(Rits):
            x_new = matvec(x) + b
            #check convergence
            L2err = np.sum(((x  - x_new)/(x_new+1e-14))**2/math.sqrt(N))
            Linferr = np.max(np.abs(x-x_new)/np.max(np.abs(x_new)+1e-14))
            if (L2err < L2_tol) and (Linferr < Linf_tol):
                converged = 1
                #print("Converged")
                
            if (iteration == 0) and (r==0):
                change = np.zeros(1) + L2err
                change_linf = np.zeros(1) + Linferr
            else:
                change = np.append(change,L2err)
                change_linf = np.append(change_linf,Linferr)
            x = x_new.copy()
            if LOUD:
                print("Iteration:", iteration+1, " Rich:", r, "Resid=", L2err, Linferr)
                if (LOUD==1):
                    print("x =",x)
            total_its += 1
            if converged:
                break
        if not(converged):
            x[0:N],change_dmd,change_dmd_linf, Atilde, Yplus_tmp, Yminus_tmp = DMD_prec(matvec, b, K, steady, x = x, step_size=step_size, res=L2err)
            Atil.append(Atilde)
            Yplus.append(Yplus_tmp)
            Yminus.append(Yminus_tmp)
            if LOUD:
                print("Iteration:", iteration+1, "DMD completed." )
                if (LOUD==1):
                    print("Post DMD x =",x)
            for kit in range(K):
                change = np.append(change,change_dmd[kit])
                change_linf = np.append(change_linf,change_dmd_linf[kit])
            total_its += K
        iteration += 1
    if LOUD:
        print("Total iterations is", total_its)
    return x, total_its,change,change_linf, Atil, Yplus,Yminus





# In[11]:


def solver_with_dmd_inc(matvec, b, K = 10, Rits = 2, steady = 1, x = np.zeros(1), step_size = 10, 
                    L2_tol = 1e-8, Linf_tol = 1e-3, max_its = 10, LOUD=0,order=4):
    #print(b.size,x.size)
    N = b.size
    if x.size != b.size:
        x = b.copy()
    assert len(b.shape) == 1
    iteration = 0
    converged = 0
    total_its = 0
    Atil = []
    Yplus = []
    Yminus = []
    
    while ((not(converged)) and (iteration < max_its)):
        for r in range(Rits):
            x_new = matvec(x) + b
            #check convergence
            L2err = np.sum(((x  - x_new)/(x_new+1e-14))**2/math.sqrt(N))
            Linferr = np.max(np.abs(x-x_new)/np.max(np.abs(x_new)+1e-14))
            if (L2err < L2_tol) and (Linferr < Linf_tol):
                converged = 1
                #print("Converged")
                
            if (iteration == 0) and (r==0):
                change = np.zeros(1) + L2err
                change_linf = np.zeros(1) + Linferr
            else:
                change = np.append(change,L2err)
                change_linf = np.append(change_linf,Linferr)
            x = x_new.copy()
            if LOUD:
                print("Iteration:", iteration+1, " Rich:", r, "Resid=", L2err, Linferr)
                if (LOUD==1):
                    print("x =",x)
            total_its += 1
            if converged:
                break
        if not(converged):
            x[0:N],change_dmd,change_dmd_linf, Atilde, Yplus_tmp, Yminus_tmp = DMD_prec_inc(matvec, b, K, 
                                                                                            steady, x = x, 
                                                                                            step_size=step_size, 
                                                                                            res=L2err)
            its_out = change_dmd.size
            if (LOUD != 0):
                print("DMD Iterations:",its_out)
            Atil.append(Atilde)
            Yplus.append(Yplus_tmp)
            Yminus.append(Yminus_tmp)
            if LOUD:
                print("Iteration:", iteration+1, "DMD completed." )
                if (LOUD==1):
                    print("Post DMD x =",x)
            for kit in range(its_out):
                change = np.append(change,change_dmd[kit])
                change_linf = np.append(change_linf,change_dmd_linf[kit])
            total_its += its_out
        iteration += 1
    if LOUD:
        print("Total iterations is", total_its)
    return x, total_its,change,change_linf, Atil, Yplus,Yminus






# In[12]:


def temp_solve_dmd(I,hx,q,sigma_func,N,BCs, Cv, phi,psi, T,
               dt_val = 0.001, tfinal = 1.0,
                     Linf_tol = 1.0e-5, tolerance = 1.0e-8,maxits = 100, 
                     LOUD=False, plot = False,order=4, fix=0, K= 10, R=3 ):
    t_current = 0.0
    c = 300.0 # speed of light
    a = 0.01372 #radiation constant
    ac = a*c
    phis = []
    Ts = []
    phi_old = phi.copy()
    print(psi.shape)
    psi_old = psi.copy()
    T_old = T.copy()
    phis.append(phi_old)
    Ts.append(T_old)
    print("|", end = '')
    curr_step = 0
    iterations = 0
    while (t_current < tfinal):
        dt = np.min([dt_val,tfinal-t_current])
        t_current += dt
        if (int(10*t_current/tfinal) > curr_step):
            curr_step += 1
            print(curr_step, end = '')
        icdt = 1.0/(c*dt)
        beta = 4*a*T_old**3/Cv
        sigma = sigma_func(T_old)
        f = 1.0/(1+beta*c*dt*sigma)
        sigma_a = f*sigma
        sigma_s = (1-f)*sigma
        source = q + (sigma_a*ac*T_old**4)[:,None,:] + icdt*psi_old 
        sigma_t = sigma+icdt
        """print("f = ",f)
        print("sigma_t = ",sigma_t)
        print("sigma_s = ",sigma_s)
        print("source = ",source)
        """
        #print("sigma_s", sigma_s, "phi.shape",phi.shape,"tog",((sigma_s*phi.reshape((I,order+1)))[:,None,:] + source*0).shape)
        mv = lambda phi: single_source_iteration(I,hx,((sigma_s*phi.reshape((I,order+1)))[:,None,:] + source*0),sigma_t,N,BCs(t_current)*0,
                                                 order=order, fix=1).reshape((order+1)*I)
        b = single_source_iteration(I,hx,source,sigma_t,N,BCs(t_current),order=order, fix=1).reshape((order+1)*I)
        phi, total_its,change,change_linf,Atil,Yplus,Yminus = solver_with_dmd(matvec=mv, b=b, 
                                                                              K=K,max_its=maxits, steady=1, 
                                                                      x = phi.flatten(), 
                                                    Rits=R, LOUD=LOUD, order=order, L2_tol=tolerance, Linf_tol=Linf_tol)
        psi = single_source_iteration_psi(I,hx,((sigma_s*phi.reshape((I,order+1)))[:,None,:] + source),sigma_t,N,BCs(t_current),
                                                 order=order, fix=1)
        iterations += total_its
        phi = phi.reshape((I,order+1))
        T = T_old + sigma_a*dt*(phi-ac*T_old**4)/Cv
        #print(T)
        T_old = T.copy()
        phi_old = phi.copy()
        psi_old = psi.copy()
        phis.append(phi_old)
        Ts.append(T_old)
    return x, phis, Ts, iterations


# In[13]:


def temp_solve_dmd_inc(I,hx,q,sigma_func,scat_func,N,BCs, EOS, invEOS, phi,psi, T,
                       dt_min = 1e-5, dt_max = 0.001, tfinal = 1.0,
                       Linf_tol = 1.0e-5, tolerance = 1.0e-8,maxits = 100, 
                       LOUD=False, plot = False,order=4, fix=0, K= 100, R=3, time_outputs=None ):
    t_current = 0.0
    c = 300.0 # speed of light
    a = 0.01372 #radiation constant
    ac = a*c
    phis = []
    Ts = []
    phi_old = phi.copy()
    print(psi.shape)
    psi_old = psi.copy()
    T_old = T.copy()
    T_old2 = T.copy() #temperature from two time steps ago
    e = EOS(T)
    e_old = e.copy()
    phis.append(phi_old)
    Ts.append(T_old)
    print("|", end = '')
    curr_step = 0
    iterations = 0
    h = 1e-12
    hreal = 1e-7
    ts = [t_current]
    delta_step = 1e-3
    step_num = 0
    dt_val = dt_min
    dt_old = dt_min
    dt_old2 = dt_min
    deriv_val = 0
    dt = dt_min
    t_output_index = 0
    while (t_current < tfinal):
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1
        if step_num>2:
            dt_prop = np.sqrt(delta_step*deriv_val)
            print(dt_prop)
            if (dt_prop > dt_max):
                dt_prop = dt_max
            if (dt_prop < dt_min):
                dt_prop = dt_min
            if (dt_prop > 2*dt):
                dt_prop = dt*1.5 #make maximum increase 50%
            dt = dt_prop
        else:
            dt = dt_min
        #don't step past the endpoint
        if (tfinal-t_current) < dt:
            dt = tfinal-t_current
        #don't step past the next output time
        try:
            if (time_outputs is not None) and (t_current + dt > time_outputs[t_output_index]) and (t_output_index < time_outputs.size):
                dt = time_outputs[t_output_index] - t_current
                t_output_index += 1
        except:
            print("Finished the last time")
        if (math.isnan(dt)):
            dt = dt_min
        print("t = %0.4e, Current dt = %0.4e, old dt = %0.4e" %(t_current,dt,dt_old))
        t_current += dt
        ts.append(t_current)
        if (int(10*t_current/tfinal) > curr_step):
            curr_step += 1
            print(curr_step, end = '')
        icdt = 1.0/(c*dt)
        Cv = (EOS(T+hreal) - EOS(T-hreal))/(2*hreal)
        beta = 4*a*T_old**3/Cv
        sigma = sigma_func(T_old)
        f = 1.0/(1+beta*c*dt*sigma)

        sigma_a = f*sigma
        sigma_s = (1-f)*sigma + scat_func(T_old)
        source = q + (sigma_a*ac*T_old**4)[:,None,:] + icdt*psi_old 
        sigma_t = sigma+icdt + scat_func(T_old)
        
        mv = lambda phi: single_source_iteration(I,hx,((sigma_s*phi.reshape((I,order+1)))[:,None,:] + source*0),sigma_t,N,BCs(t_current)*0,
                                                 order=order, fix=1).reshape((order+1)*I)
        b = single_source_iteration(I,hx,source,sigma_t,N,BCs(t_current-dt/2),order=order, fix=1).reshape((order+1)*I)
        phi, total_its,change,change_linf,Atil,Yplus,Yminus = solver_with_dmd_inc(matvec=mv, b=b, 
                                                                              K=K,max_its=maxits, steady=1, 
                                                                      x = phi.flatten(), 
                                                    Rits=R, LOUD=LOUD, order=order, L2_tol=tolerance, Linf_tol=Linf_tol)
        psi = single_source_iteration_psi(I,hx,((sigma_s*phi.reshape((I,order+1)))[:,None,:] + source),sigma_t,N,BCs(t_current),
                                                 order=order, fix=1)
        iterations += total_its
        phi = phi.reshape((I,order+1))
        e = e_old + sigma_a*dt*(phi-ac*T_old**4)
        T = invEOS(e)
        #print(T)
        #compute second-derivative of T in time
        if step_num>=2:
            deriv_val = np.mean(T)/np.mean((np.abs(T/(dt**2)-(dt+dt_old)/(dt**2*dt_old)*T_old + T_old2/(dt_old*dt))))
            
        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        phi_old = phi.copy()
        psi_old = psi.copy()
        phis.append(phi_old)
        Ts.append(T_old)
    return phis, Ts, iterations,np.array(ts)