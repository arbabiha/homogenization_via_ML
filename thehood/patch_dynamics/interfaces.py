# !/usr/bin/python
import scipy

class State(object):
    " This class can be used to store data members as a struct in matlab"
    pass

class MicroSolver(object):
    """ Class that is an interface to any microscopic simulator.
    
    Subclasses need to implement
         - step(self,delta_t)     which advances the model over delta_t
         - getState(self)         which returns the state
         - getTime(self)          which returns current time
    """

    def step(self,delta_t):
        """
        Performs one timestep of the microscopic model
        Effects:
             *  getTime()=old.getTime()+delta_t
             *  getState() has changed to state after integration of time
                 delta_t
        Method needs to be overridden by subclasses
        """
        assert 1, "method needs to be overridden by subclass"

    def initialize(state,t):
        assert 1, "method has to be overridden by subclass"

    def getState(self):
        assert 1, "method needs to be overridden since state can" + \
               "differ from micro-model to micro-model"

    def getTime(self):
        assert 1, "has to be implemented in concrete micromodels"
        
class CoarseTimestepper(object):
    """
    This class contains the interface for any coarse timestepper.

    Constructed as
        CoarseTimestepper(microsolver):
             Constructor does type-checking; checks in microsolver is instance of
             MicroSolver class.

             Actual construction is left to subclasses !!
             (In order to reduce levels of indirection (may be stupid of me))
             (Second reason, a coarse timestepper is built for one particular
             microscopic model, so maybe the checks there need to be more severe.)
    
    Contains the method
         - step(delta_t,N): step over delta_t, using N realizations

    Subclasses need to implement
         - constructor
         - lift(self)    which returns a microscopic initial condition, 
        based on the current state
         - run(self, delta_t)  which advances the micromodel over a time delta_t
               NOTE:  this method does NOT have to be equal to the step method 
                in Microsolver
         - restrict(self) which return the macrostate corresponding to 
                the micro_state
    """
    def __init__(self, microsolver,neq,N):
        if not isinstance(microsolver,MicroSolver):
            assert 1, "argument not an instance of MicroSolver"
        self.microsolver=microsolver
        self.neq=neq
        self.N=N
           
    def initialize(macro_state,t):
        assert 1, "not implemented in abstract CoarseTimestepper class"
        
    def lift(self):
        assert 1, "not implemented in abstract CoarseTimestepper class"

    def getMicrosolver(self):
        return self.microsolver

    def restrict(self):
        assert 1, "not implemented in abstract CoarseTimeStepper class"

    def getNbRealizations(self):
        return self.N

    def getState(self):
        assert 1, "getState() needs to be implemented in concrete " +\
            "Coarse timestepper class"

    def step(self,delta_t):
        """
        takes a coarse time-step (consisting of a lift-run-restrict procedure), 
        computes the average of doing this N times.  
        Default N (number of realizations) =1,
        and returns the value after this coarse time-step.
        """
        # print('--inside a step of micro-solver')
        
        total=scipy.zeros(self.neq) 
        for i in scipy.arange(0,self.N,1):
            self.lift()
            # print('-- -lift done')
            self.run(delta_t)
            # print('-- -run done')
            total=total+self.restrict()
        self.setRestriction(total/self.N)
        
    
    def run(self,delta_t):
        self.microsolver.step(delta_t)
    
    def setRestriction(self,something):
        self.hasRestriction=True
        self.restriction=something
    
    
    def getRestriction(self):
        if not self.hasRestriction:
            print('NO Restriction AVAILABLE !')
            return 0
        else:
            return self.restriction

class DerivativeEstimator(object):
    """
    This class contains the interface for a derivative estimator.

    Constructed as
         DerivativeEstimator(ct), where ct has to be a CoarseTimestepper
          (NOTE: this is checked here, but actual initialization is left to
          concrete estimators.)
    """
    def estimate(self,x,u,t):
        assert 1, "needs to be overridden by concrete estimator"

    def getTaylor(self,x,u,t,h,bc):
        assert 1, "needs to be overridden by concrete estimator"

# definition of boundary conditions
DIRICHLET=0
PERIODIC=1
NOFLUX=2
INTERNAL=3
REFLECTING=4

class MacroSolver(object):

    def integrate(x,u,t,bc):
        assert 1, "needs to be overridden" 
    

class BoxTimestepper(CoarseTimestepper):

    def __init__(self,microsolver,neq,N,h,H,delta_x,flux_form=False):
        CoarseTimestepper.__init__(self,microsolver,neq,N)
        self.h=h
        self.H=H
        self.delta_x=delta_x
        self.flux_form=flux_form
    
    def initialize(self,macro_state):
        """
        Initialize assigns the current macro_state to the coarse time-stepper
        For this coarse timestepper, the macro_state consists of
        Attributes:
          - macro_state.profile: an neq*(len(x)+1) array of values
          - macro_state.time : the current time
          - macro_state.x: abscissa for the "profile" values
        """
        self.x=macro_state.x
        self.mesh=macro_state.mesh
        self.profile=macro_state.profile
        self.buffer=scipy.array([self.x-self.H/2,self.x+self.H/2])
        self.bounds=scipy.array([self.x-self.h/2,self.x+self.h/2])
        self.t=macro_state.time

    def getState(self):
        state=State()
        state.profile=self.profile
        state.mesh=self.mesh
        state.x=self.x
        state.buffer=self.buffer
        state.bounds=self.bounds
        state.time=self.t
        return state
    
    def restrict(self):
        if self.flux_form:
            return self.restrictToFlux()
        else: 
            return self.restrictToDensity()
    
    def restrictToFlux(self):
        print ('needs to be overridden')

    def restrictToDensity(self):
        print ('needs to be overridden')
    
class Derivative(object):
    def derivative(self,u,x,i):
        """
        no implementation yet -- needs to be subclassed
        """
        print ('no actual difference scheme')
