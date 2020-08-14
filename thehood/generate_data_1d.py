"""Generating data from patch dynamics simulation of 1d heterogeneous diffusion.

The simulation code is written by Giovanni Samaey at KU Leuven (University of Leuven)
and adapted for Python3 by Hassan Arbabi, June 2020.
"""




from sys import path
path.append('./patch_dynamics/')
path.append('./patch_dynamics/systems/diff_hom/')
import numpy as np

import systems.diff_hom.diff_hom as diffusion

import scipy
import tables

import interfaces
import algorithms.fd_gaptooth as gaptooth
import macro.fd.time.fd_forweuler as macro_scheme
import macro.fd.space.second_order as central

from mpi4py import MPI as MPI
import timeit



def diffusion_solver(initial_profile : callable, tmax : float, eps : float=1e-5):
    """Solving the homogenized diffusion.

    Args:
        initial_condition: the the initial profile of temperature on [0,1]
        tmax: length of trajectory
        eps: the length scale of heterogenity

    Returns:
        U: the solution of patch dynamics
        U_fd: solution of finite difference
        U_ref: the reference detailed solution
        T: time stamps
        x, x_fd, x_ref: spatial grids for U, U_fd and U_ref

    """
    ttin = timeit.default_timer()

    dx = 1e-7

    # create microsolver
    sys=diffusion.DiffHom(eps,dx)

    # create box solver
    h = 1e-4
    H = 8e-3
    boxts=diffusion.DiffHomBoxTimestepper(sys,h,H)

    # create gaptooth and reference estimator
    diff = central.SecondOrderFD()
    delta_t = 1e-6
    estimator=gaptooth.FDGaptoothEstimator(diff,boxts,gaptoothStep=delta_t,maxDeriv=2)
    ref_estimator=diffusion.ReferenceEstimator(diff)

    # finite difference parameters
    Delta_x=1e-1
    Dt=1e-3

    # tend = 9.+Dt/2
    tend = int(tmax/Dt)*Dt
    x0=0.
    x1=1.
    # reference solution parameters
    Dt_ref = 1e-6
    Delta_x_ref = 5e-3
    ratio = int(round(Delta_x/Delta_x_ref))

    # create macro_scheme solver
    solver=macro_scheme.MacroTimestepper(estimator)
    solver_fd=macro_scheme.MacroTimestepper(ref_estimator)
    solver_ref=macro_scheme.MacroTimestepper(ref_estimator)

    x=scipy.arange(x0,x1+Delta_x/2.,Delta_x)
    x_fd=scipy.arange(x0,x1+Delta_x/2,Delta_x)
    x_ref=scipy.arange(x0,x1+Delta_x_ref/2,Delta_x_ref)

    sx = len(x)
    sxref = len(x_ref)


    Delta_t = Dt
    t=scipy.arange(0,tend+Delta_t/2,Delta_t)


    u=scipy.zeros((1,len(x)),scipy.float64)
    u[0,:]= initial_profile(x) # here is where the initial condition is

    u_fd=scipy.zeros((1,len(x_fd)),scipy.float64)
    u_fd[0,:]=initial_profile(x_fd)

    u_ref=scipy.zeros((1,len(x_ref)),scipy.float64)
    u_ref[0,:]=initial_profile(x_ref)
    
    U_ref=[u_ref]
    U_fd =[u_fd]
    U = [u]
    T = [t[0]]


    # run the solver
        
    bc = scipy.array([interfaces.DIRICHLET,interfaces.DIRICHLET])

    for i in scipy.arange(len(t)-2):
        result=solver.integrate(u,x,Dt,t[i:i+2],bc)
        result_fd=solver_fd.integrate(u_fd,x_fd,Dt,t[i:i+2],bc)
        result_ref=solver_ref.integrate(u_ref,x_ref,Dt_ref,t[i:i+2],bc)


        u=result.u[:,:,-1]
        u_fd=result_fd.u[:,:,-1]
        u_ref=result_ref.u[:,:,-1]

        U.append(u)
        U_fd.append(u_fd)
        U_ref.append(u_ref)
        T.append(t[i+1])

    U = np.stack(U).squeeze()
    U_fd = np.stack(U_fd).squeeze()
    U_ref = np.stack(U_ref).squeeze()
    T = np.stack(T).squeeze()
    print('This trajectory took {} seconds'.format(timeit.default_timer() - ttin))

    return U, U_fd, U_ref, T, x, x_fd, x_ref

def get_random_IC():
    """Generates a random profile on [0,1].

    Returns:
        a callable  that takes x and spits IC(x).
    """

    N= 20
    A = np.random.rand(N)-.5
    phi=np.random.rand(N)*2*np.pi
    l = np.random.rand(N) * 4

    def rho(x):
        y = 0
        for k in range(N):
            y = y + A[k]*np.sin(l[k]*x* 2*np.pi + phi[k])
        return y
            
    # make it positive
    x = np.linspace(0,2*np.pi,num=128)
    r = rho(x)


    rmin = np.min(r)
    if rmin<0.05:
        rho2= lambda x: rho(x) + np.abs(rmin) + .1
    else:
        rho2 = rho

    return rho2

def GenerateData(n_traj: int, length_traj: float, filename: str):
    """ Gnerating data from homogenized diffusion problem.
    
    Args:
        no_traj: number of trajectories
        length_traj: length of time interval
        filename: name of the saved file
        
    Returns:
        U: the solution of patch dynamics
        U_fd: solution of finite difference
        U_ref: the reference detailed solution
        T: time stamps
        x, x_fd, x_ref: spatial grids for U, U_fd and U_ref
    """

    U,U_fd,U_ref = [],[],[]

    # epss=np.logspace(-6,-3,21)
    eps = 1e-5

    for _ in range(n_traj):
        ttin = timeit.default_timer()
        print('traj #'+str(_))
        initial_profile = get_random_IC(same_BC=True)
        u, u_fd, u_ref, T, x, x_fd, x_ref = diffusion_solver(initial_profile,length_traj,eps=eps)

        U.append(u)
        U_fd.append(u_fd)
        U_ref.append(u_ref)
        print('This trajectory took {} seconds'.format(timeit.default_timer() - ttin))

    
    U = np.stack(U).squeeze()
    U_fd = np.stack(U_fd).squeeze()
    U_ref = np.stack(U_ref).squeeze()

    np.savez(filename, U=U, U_fd=U_fd, U_ref=U_ref, T=T, x=x, x_fd=x_fd, x_ref=x_ref,eps=eps)






if __name__ == "__main__":
    ttin = timeit.default_timer()
    np.random.seed(42)
    no_traj = 10
    tmax=1
    filename= './data/hdiff_variedBC.npz'
    GenerateData(no_traj, tmax,filename)
    print('Whole computation took {} seconds'.format(timeit.default_timer() - ttin))



