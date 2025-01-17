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
        """
        result=interfaces.State()
        result.x=scipy.zeros((len(x),len(t)),scipy.float64)
        result.x[:,0]=x
        s=list(scipy.shape(u))
        s.append(len(t))
        result.u=scipy.zeros(s,scipy.float64)
        result.u[:,:,0]=u
        result.time=t

        time=t[0]
        i=1
        
        i_h = 0
        # print('inside the solver_ref integartion')

        while not scipy.allclose(time,t[-1],rtol=1e-8,atol=1e-10):
            dudt=self.estimator.estimate(u,time,x,bc)
            # print(i_h)
            u=u+Dt*dudt
            time=time+Dt
            i_h=i_h+1

            if scipy.allclose(time,t[i],rtol=1e-8,atol=1e-10):
                result.x[:,i]=x
                result.u[:,:,i]=u
                i+=1
        return result
