"""Data-driven discovery of the homogenized PDE for 2d heterogeneous diffusion.

This is example 2 in "Linking Machine Learning with Multiscale Numerics: 
Data-Driven Discovery of Homogenized Equations", by H. Arbabi et al., 2020

Send comments and questions to arbabiha - AT - gmail.com.
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.io as sio
import os
import tensorflow as tf
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from scipy.fftpack import diff as fft_diff

from sys import path
path.append('./thehood/')
import model_library as ML 



plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=10)
bdazz_blue='#28587B'
dudt_colormap = 'RdGy'
SavePath = './figs/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

def prepare_data():
    """Loading the data from 1d diffusion and preprocessing.
    
    Returns:
        xgrid: x grid points of gap-tooth solution
        ygrid: x grid points of gap-tooth solution
        v: gap-tooth solution
        dvdt: time derivatives of gap-tooth solution
        T: time stamps of gap-tooth solution
        v_hom: homogenized PDE solution subsampled on the same grid as gap-tooth
        dvdt_hom: homogenized PDE solution time derivative
    """

    Data=sio.loadmat('./thehood/hdiff_2d_16x3_T1.mat')
    U_hom,Ut_hom = Data['U_hom'],Data['Ut_hom']
    U_gt,Ut_gt,x_gt,y_gt=Data['U_pd'],Data['Ut_pd'],Data['x_pd'],Data['y_pd']
    T = Data['time'].squeeze()

    nm = 3 # the priodicty of heterogeneity
    nc =nm//2
    xgrid= x_gt[nc::nm,nc::nm]
    ygrid= y_gt[nc::nm,nc::nm]
    dxy = [xgrid[1,2]-xgrid[1,1],ygrid[2,1]-ygrid[1,1]]


    v=np.concatenate(U_gt,axis=0)
    dvdt = np.concatenate(Ut_gt,axis=0)
    v_hom=np.concatenate(U_hom,axis=0)
    dvdt_hom = np.concatenate(Ut_hom,axis=0)

    return xgrid,ygrid,v,dvdt,T,v_hom,dvdt_hom


def learn_functional_PDE():
    """Learning the homogenized PDE in the functional form.
        
    Returns:
        keras.model(u,dudt) trained by the gap-tooth data
    """
    _,_,v,dvdt,_,_,_=prepare_data()

    ddx = lambda u: np.apply_along_axis(fft_diff,-1,u)
    ddy = lambda u: np.apply_along_axis(fft_diff,-2,u)
    vx,vy = ddx(v),ddy(v)
    vxx,vxy,vyy= ddx(vx),ddy(vx),ddy(vy)
    v_input = np.stack((v,vx,vy,vxx,vyy,vxy),axis=-1)


    x_train,x_test,y_train,y_test=train_test_split(v_input,dvdt,test_size=0.15,shuffle=False)

    nn_model = ML.Functional_PDE2D_net_wfft(v_input.shape[-3:],n_conv=3)
    print(nn_model.summary())

    adamopt=tf.keras.optimizers.Adam(learning_rate=.001)
    nn_model.compile(optimizer=adamopt,loss='mse')

    PDEfit_history=nn_model.fit(x_train,y_train,
        batch_size=64,epochs=32,
        verbose=1,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend()
    np.array(PDEfit_history.history['val_loss'])
    plt.savefig(SavePath+'2d_funcPDE_fit.png',dpi=450)

    eval_loss = np.mean((nn_model.predict(x_test).squeeze()-y_test)**2)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )

    return nn_model



def learn_discretized_PDE(n_stencil=5):
    """Learning the homogenized PDE in the discretized form.
    
    Args:
        n_stencil: the stencil size of discretization (i.e. kernel size of first convolution layer)
        
    Returns:
        keras.model(u,dudt) trained by the gap-tooth data
    """
    _,_,v,dvdt,_,_,_=prepare_data()

    x_train,x_test,y_train,y_test=train_test_split(v,dvdt,test_size=0.15,shuffle=False)

    n_stencil= 5
    n_filter = 32

    nn_model = ML.Discretized_PDE2D_net(v.shape[-2:],n_stencil,n_conv=3,n_filter1=n_filter)
    print(nn_model.summary())

    adamopt=tf.keras.optimizers.Adam(learning_rate=.001)
    nn_model.compile(optimizer=adamopt,loss='mse')

    PDEfit_history=nn_model.fit(x_train,y_train,
        batch_size=64,epochs=128,
        verbose=1,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend()
    np.array(PDEfit_history.history['val_loss'])
    plt.savefig(SavePath+'2d_discPDE_fit.png',dpi=450)

    eval_loss = np.mean((nn_model.predict(x_test).squeeze()-y_test)**2)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )

    return nn_model



def test_model(nn_model,tag,requires_FFTderivative=True):
    """Test the learned model in RHS and integration.

    Args:
        nn_model: the trained keras.model(u,dudt)
        requires_FFTderivative: does it take the precomputed derivatives from fft?
        tag: postfix for saved figures

    Returns:
        saves comparsion figures.
    """

    if requires_FFTderivative:
        ddx = lambda u: np.apply_along_axis(fft_diff,-1,u)
        ddy = lambda u: np.apply_along_axis(fft_diff,-2,u)
        def nn_predict(u):
            """Map from u to dudt."""
            ux,uy=ddx(u),ddy(u)
            uxx,uyy,uxy = ddx(ux),ddy(uy),ddy(ux)
            u_input = np.stack((u,ux,uy,uxx,uyy,uxy),axis=-1)
            dudt = nn_model.predict(u_input[np.newaxis,...])
            return dudt.squeeze()
    else:
        def nn_predict(u):
            """Map from u to dudt."""
            dudt= nn_model.predict(u[np.newaxis,...])
            return dudt.squeeze()

    xgrid,ygrid,v,dvdt,T,v_hom,dvdt_hom=prepare_data()
    ngrid=[v.shape[-2],v.shape[-1]]

    _,x_test,_,dvdt_truth=train_test_split(v,dvdt_hom,test_size=0.15,shuffle=False)

    # comparison of dudt
    tindex =[1319,293]
    plt.figure(figsize=[6.5,4])
    irow=0
    for index in tindex:
        plt.subplot(2,2,irow*2+1)
        plt.contourf(xgrid,ygrid,dvdt_truth[index],20,cmap=dudt_colormap)
        plt.colorbar()
        plt.title(r'$\partial_t u$ hom.')

        plt.subplot(2,2,irow*2+2)
        dvdt_nn = nn_predict(x_test[index])
        plt.contourf(xgrid,ygrid,dvdt_nn-dvdt_truth[index],20,cmap=dudt_colormap)
        plt.colorbar()
        plt.title(r'nn error in $\partial_t u$')

        irow=irow+1
    plt.tight_layout()
    plt.savefig(SavePath+'2d_dudt_'+tag+'.png',dpi=450)   

    # comparison in trajectory integration
    nt = T.shape[-1]
    v0= v[-nt].reshape(-1)

    def nn_dudt(t,u):
        """Another wrapper for passing to ODE solver."""
        u = u.reshape(ngrid)
        dudt = nn_predict(u)
        dudt=dudt.reshape(ngrid[0]*ngrid[1])
        return dudt.squeeze()

    Sol = solve_ivp(nn_dudt,[T[0],T[-1]],v0,method='BDF',t_eval=T,max_step=0.01)
    v_nn=Sol.y.T.reshape(T.shape[0],v.shape[-2],v.shape[-1])

    tindex =[10,20,50,100]
    v_truth = v_hom[-nt:]


    plt.figure(figsize=[3.25,6])
    irow=0
    for it in tindex:
        plt.subplot(4,2,2*irow+1)
        plt.contourf(xgrid,ygrid,v_truth[it],20,cmap=dudt_colormap)
        plt.title('$u$ hom. \n $t=${:.2f}'.format(T[it]) )
        plt.colorbar()
        plt.xlabel('$x$'),plt.ylabel('$y$')

        plt.subplot(4,2,2*irow+2)
        plt.contourf(xgrid,ygrid,v_nn[it]-v_truth[it],20,cmap=dudt_colormap)
        plt.title(r'nn error')
        plt.colorbar()
        plt.xlabel(r'$x$'),plt.ylabel(r'$y$')

        irow=irow+1

    plt.tight_layout()
    plt.savefig(SavePath+'2d_traj_'+tag+'.png',dpi=450)

if __name__ == "__main__":
    tt = timeit.default_timer()
    disc_model=learn_discretized_PDE()
    test_model(disc_model,'discPDE',requires_FFTderivative=False)

    func_model=learn_functional_PDE()
    test_model(func_model,'funcPDE',requires_FFTderivative=True)
    print('... took {} seconds'.format(timeit.default_timer() - tt))


