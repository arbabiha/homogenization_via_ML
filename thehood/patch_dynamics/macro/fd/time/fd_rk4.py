import interfaces
import scipy

class MacroTimestepper(interfaces.MacroSolver):

    def __init__(self,estimator, param = None):
        """
        initializes the macrosolver with an estimator for the time
        derivative, and with the maximal spatial derivative that will
        occur
        """
        self.estimator=estimator

    def setGrid(self,grid):
        self.estimator.setGrid(grid)

    def integrate(self,u,x,Dt,t,bc):
        """
        this asks the macrosolver to integrate starting from (x,u0) at t[0],
        and report the results at times in t
        boundary conditions:
           - bc=0    Dirichlet
           - bc=1    periodic
        """
        time=t[0]
        i=1
        result=interfaces.State()
        result.x=scipy.zeros((len(x),len(t)),scipy.float64)
        result.x[:,0]=x
        s=list(scipy.shape(u))
        s.append(len(t))
        result.u=scipy.zeros(s,scipy.float64)
        result.u[:,:,0]=u
        result.time=t
        
        while not scipy.allclose(time,t[-1],rtol=1e-8,atol=1e-10):
            k1=self.estimator.estimate(u,time,x,bc)
	        k2=self.estimator.estimate(u+0.5*Dt*k1,time+0.5*Dt,x,bc)
	        k3=self.estimator.estimate(u+0.5*Dt*k2,time+0.5*Dt,x,bc)
	        k4=self.estimator.estimate(u+Dt*k3,time+Dt,x,bc)
	        u=u+Dt*(1./6.*k1+1./3.*k2+1./3.*k3+1./6.*k4)
            time=time+Dt
	    if scipy.allclose(time,t[i],rtol=1e-8,atol=1e-10):
               	result.x[:,i]=x
                result.u[:,:,i]=u
               	i+=1
        return result

    
            
                
            
            
        
        
        
        
        
