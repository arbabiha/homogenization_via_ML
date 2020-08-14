import interfaces
import macro.fd.fd_estimator
import scipy
import mpi4py.MPI as MPI
import numpy

comm=MPI.COMM_WORLD.Dup()

my_rank=comm.Get_rank()
nprocs=comm.Get_size()

class FDGaptoothEstimator(macro.fd.fd_estimator.FDEstimator):
    """
    This class defines everything that is needed to
    obtain an estimate of the time derivative of
    the unavailable macroscopic equations.
    """

    def __init__(self,diff,boxtimestepper,gaptoothStep,maxDeriv):
        """
        The constructor takes a coarse timestepper
        and an order of approximation
        """
        self.boxts=boxtimestepper
        self.delta_t=gaptoothStep
        macro.fd.fd_estimator.FDEstimator.__init__(self,diff,maxDeriv)

    def getGaptoothStep(self):
        return self.delta_t

    def getBufferSize(self):
        return self.boxts.H

    def getBoxSize(self):
        return self.boxts.h
        
    def estimate(self,u,t,x,bc):
        neq=self.boxts.neq
        h=self.getBoxSize()

        taylorCoeff=self.getTaylor(x,u,t,h,bc)
        if bc[0]==interfaces.DIRICHLET:
            dir=1
        else:
            dir=0
        # note that we are assuming Dirichlet BC on both ends 
        ntotal = len(x)-2*dir
        nloc = int(ntotal/nprocs) # integer division HA: added int
        remainder = ntotal % nprocs
        
        if my_rank < remainder:
            nlocal=nloc+1
        else:
            nlocal=nloc

        ownership = scipy.zeros((nprocs,),scipy.integer)
        for i in range(nprocs):
            ownership[i]=i*nloc+min(remainder,i)
        
        my_start = ownership[my_rank] 
        if dir:
            my_start=my_start + 1
        elif my_rank == nprocs - 1 and dir:
            nlocal = nlocal - 1
        
        # added ints here
        u_delta_local=scipy.zeros((neq,nloc+1),\
            dtype = scipy.float64, order = 'F')   
        H=self.boxts.H
        delta_x=self.boxts.delta_x


        i_h=0
        for i in scipy.arange(my_start,my_start+nlocal,1):
            i_h=i_h+1
            # create macro_state
            macro_state=interfaces.State()
            i=int(i)
            macro_state.x=x[i]
            macro_state.mesh=scipy.arange(\
                x[i]-H/2,x[i]+H/2+delta_x/2,delta_x)
            macro_state.profile=scipy.zeros(\
                (neq,len(macro_state.mesh)),scipy.float64)
            # print('pareallel loop: formed the state')
            for j in range(scipy.shape(taylorCoeff)[0]):
                polynom=scipy.poly1d(taylorCoeff[j,:,i])
                macro_state.profile[j,:]=polynom(macro_state.mesh-x[i])
            macro_state.time=t
            # initialize

            self.boxts.initialize(macro_state)

            # step
            self.boxts.step(self.delta_t)
            # print('pareallel loop: took the '+str(i_h)+'-step')
            restriction = self.boxts.getRestriction()
            u_delta_local[:,i-my_start]=restriction
            
        # hier moet iedereen zijn u_delta_local naar iedereen sturen
        # print('finished the pareallel loop')

        # fortran ordering is necessary to be able to pass columns
        u_delta_global = scipy.zeros((int(neq),int((nloc+1)*nprocs)),\
            dtype=scipy.float64,order='F')
        comm.Allgather([u_delta_local,(nloc+1)*neq,MPI.DOUBLE], # send
            [u_delta_global,(nloc+1)*neq, MPI.DOUBLE])
    
        u_delta=scipy.zeros(scipy.shape(u),scipy.float64)

        if bc[0]==interfaces.DIRICHLET:
            u_delta[:,0]=u[:,0]
        for i in range(nprocs):
            if i < nprocs-1:
                length = ownership[i+1]-ownership[i]
            else:
                length = ntotal - ownership[nprocs-1]
            u_delta[:,i*(nloc+1)+dir:i*(nloc+1)+dir+length]=\
                        u_delta_global[:,i*(nloc+1):i*(nloc+1)+length]
        
        if bc[1]==interfaces.DIRICHLET:
            u_delta[:,-1]=u[:,-1]
        
        deriv=u_delta-u   
    
        return deriv/self.delta_t
