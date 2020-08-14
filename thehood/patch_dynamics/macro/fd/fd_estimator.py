import interfaces
import numpy
from scipy.special import factorial as fac

class FDEstimator(interfaces.DerivativeEstimator):    
    def __init__(self, diff, maxDeriv=2):
        """
        initializes the macrosolver with an estimator for the time
        derivative, and with the maximal spatial derivative that will
        occur
        """
        self.diff=diff
        self.maxDeriv=maxDeriv
        self.grid = None

    def setGrid(self,grid):
        self.grid = grid

    def getTaylor(self,x,u,t,h,bc):
        if self.maxDeriv>2:
            assert 1, "higher derivatives are not implemented (yet)"
        u_ghost,x_ghost = self.add_boundaries(u, x, bc)
                
        result=numpy.zeros((numpy.shape(u)[0],\
                        self.maxDeriv+1,len(x)),numpy.float64)
        for i in range(self.maxDeriv):
                result[:,i,:]=1./fac(self.maxDeriv-i)*\
                        self.diff.derivative(u_ghost,x_ghost,\
                                                self.maxDeriv-i)
        if self.maxDeriv == 2:
            result[:,self.maxDeriv,:]=u-h**2/12*result[:,self.maxDeriv-2,:]
        else:
            result[:,self.maxDeriv,:]=numpy.array(u)
        return result

    def add_boundaries(self,u,x,bc):
        if self.grid == None:
            nghost = self.diff.nghost
            neq = numpy.shape(u)[0]
            ghost_left=numpy.zeros((neq,nghost),numpy.float64)
            ghost_right=numpy.zeros((neq,nghost),numpy.float64)
            # for the moment, we are assuming that we have the same bc
            # at both ends
            if bc[0] == interfaces.PERIODIC:
                ghost_left = u[:,-nghost:]
                ghost_right = u[:,:nghost]
            if bc[0] == interfaces.NOFLUX:
                for i in range(nghost):
                    ghost_left[:,i]=u[:,0]
                    ghost_right[:,i]=u[:,-1]
            if bc[0] == interfaces.DIRICHLET:
                for i in range(nghost):
                    ghost_left[:,i]=0
                    ghost_right[:,i]=0
            u_ghost = numpy.c_[ghost_left,u,ghost_right]
            delta_x = x[nghost:0:-1]-x[0]
            x_ghost = numpy.r_[x[0]-delta_x,x,x[-1]+delta_x[::-1]]
            return u_ghost,x_ghost
        else:
            return self.grid.add_boundaries(bc)
