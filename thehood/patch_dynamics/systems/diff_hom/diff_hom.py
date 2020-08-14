"""
This module defines a microscopic model for homogenization
diffusion, along with its coarse timestepper.
"""

import scipy
from scipy.integrate import simps, solve_ivp
from scipy.integrate import odeint
import numpy

import interfaces
import macro.fd.fd_estimator

class DiffHom(interfaces.MicroSolver):
    """
    This class defines everything that concerns the microscopic 
    simulations.
    epsilon : value of the small scale parameter

    Constructor
       DiffHom(epsilon,delta_x):
           - epsilon is the small scale parameter
           - delta_x is the spatial resolution

    Methods
       - initialize(state,t):
           - state.x is coordinate vector
           - state.y is state vector over x
           - t is time
           
       - step(), getTime(), getState(): override methods of MicroSolver

       - getX()
       - getDiffCoeff():  gets the diffusion coefficient
       - rhs(self,y,t):   definition of the microscopic 
         jac(self,y,t):      equations
    """
    def __init__(self,epsilon,delta_x):
        """ Constructs a microscopic solver.

        Default values are given here so that one can create a
        microsolver object without initializing it with proper initial
        conditions.
        
        """
        self.epsilon=epsilon
        self.delta_x=delta_x
        
    def initialize(self,state,t):
        """
        This method initializes the microscopic solver.
        It takes a state as input, where state contains
           - state.x: the spatial coordinate vector
           - state.y: the concentration vector
                      if multiple concentrations, this will enter as a matrix of size (neq*nx)
        After the method has been called, step(delta_t) can be executed.

        """
        # print('initializing DiffHom')
        self.x=state.x
        self.aeps=self.diffCoeff(self.x)
        self.y=state.y[:]  # we add this to avoid flatiter issue
        self.t=t
        # print('initizales y defalt as '+str(self.y))
                
    def step(self,delta_t):
        """
        Step() can only run if the model has been initialized properly
        Error if initialize() did not execute before step()
        """

        self.y=numpy.squeeze(self.y)

        y,infodict=odeint(self.rhs,self.y,scipy.array([self.t,self.t+delta_t/self.delta_x**2]),ml=1,mu=1,full_output=1)

        self.t=self.t+delta_t
    
    def diffCoeff(self,x):
        xa=x[0:-1]+self.delta_x
        return 1.1+scipy.sin(2*scipy.pi*xa/self.epsilon)
        # return numpy.ones_like(xa)
                        
    def getState(self):
        state=interfaces.State()
        state.x=self.x
        state.y=self.y
        state.t=self.t
        return state

    def rhs(self,y,t):

        ny=scipy.size(y)
        ydot=scipy.zeros(ny,scipy.float64)
        ydot=scipy.zeros(ny,scipy.float64)
        ydot[1:-1]=(self.aeps[1:]*(y[2:]-y[1:-1])-self.aeps[:-1]*(y[1:-1]-y[:-2]))
        return ydot

class DiffHomBoxTimestepper(interfaces.BoxTimestepper):
    def __init__(self,microsolver,h,H):
        neq = 1
        N = 1
        delta_x=microsolver.delta_x
        interfaces.BoxTimestepper.__init__(self,microsolver,neq,N,h,H,delta_x)
        
    def lift(self):
        """
        Lift initializes the microscopic code based on the current macro-state
        """    
        state=interfaces.State()
        state.x=self.mesh
        y=scipy.zeros((self.neq,len(state.x)),scipy.float64)
        for i in scipy.arange(0,self.neq,1):
                y[i,:]=self.profile[i,:]
        y=y.flat
        state.y=y
        self.microsolver.initialize(state,self.t)

    def relevant(self,state):
        """
        returns the non-buffer (thus *relevant* part) of the concentration vector of the
        state
        """
        return state.y[scipy.argmin(abs(state.x-self.bounds[0])):\
                       scipy.argmin(abs(state.x-self.bounds[1]))+1]
        
    def restrict(self):
        """
        restrict returns the macroscopic value that is associated with the current
        microscopic state of its microscopic solver, together with the time variable
        """
        state=self.microsolver.getState()
        t = state.t
        y=self.relevant(state)        
        # the coarse timestepper has to know what the microscopic state is to lift, anyway
        # so we are allowed to use this knowledge here
        # here the micro-state is an the function evaluated on an equidistant mesh
        u=simps(y,dx=self.delta_x)/self.h
        return u

class ReferenceEstimator(macro.fd.fd_estimator.FDEstimator):
    """
	Reference macroscopic solution for this problem
	"""
	
    def __init__(self,diff):
        # self.grad is True if the effective coeff
        # has macroscale variations.
        # this means that a term in the finite difference
        # equations will need to be added.
        self.a = 0.45825686
        maxDeriv = 2
        macro.fd.fd_estimator.FDEstimator.__init__(self,diff,maxDeriv)
	
    def estimate(self,u,t,x,bc):
        neq=1
        h=2e-3 # this value is irrelevant, because not used
        taylorCoeff=self.getTaylor(x,u,t,h,bc)
        d2 = taylorCoeff[:,0,:]
        deriv = scipy.zeros(scipy.shape(u),scipy.float64)
        # only implemented for Dirichlet BC
        deriv[:,1:-1] = self.a*2*d2[:,1:-1]
        return deriv
    
    
    

    

        
    

        
        
    
        


