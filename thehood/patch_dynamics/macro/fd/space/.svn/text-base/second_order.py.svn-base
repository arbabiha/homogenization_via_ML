import scipy
import interfaces

class SecondOrderFD(interfaces.Derivative):
    
    def __init__(self):
        self.nghost=1
        
    def derivative(self,u_ghost,x_ghost,i):
        d=scipy.zeros((scipy.shape(u_ghost)[0],\
                scipy.shape(u_ghost)[1]-2),scipy.float64)
        dx=x_ghost[1:]-x_ghost[:-1]
        if i==1:
            d=dx[:-1]/(dx[1:]*(dx[1:]+dx[:-1]))*u_ghost[:,2:]+\
                (dx[1:]-dx[:-1])/(dx[:-1]*dx[1:])*u_ghost[:,1:-1]-\
                dx[1:]/(dx[:-1]*(dx[1:]+dx[:-1]))*u_ghost[:,:-2]
        if i==2:
            d=2./(dx[1:]*(dx[1:]+dx[:-1]))*u_ghost[:,2:]-\
                2./(dx[1:]*dx[:-1])*u_ghost[:,1:-1]+\
                2./(dx[:-1]*(dx[1:]+dx[:-1]))*u_ghost[:,:-2]
        return d
                                                                          
        
