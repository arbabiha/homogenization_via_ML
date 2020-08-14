"""Data-driven discovery of the homogenized PDE for 1d heterogeneous diffusion.

This is example 1 in "Linking Machine Learning with Multiscale Numerics: 
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

from sys import path
path.append('./thehood/')
import model_library as ML 


plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=10)
bdazz_blue='#28587B'

SavePath = './figs/'
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

def prepare_data():
    """Loading the data from 1d diffusion and preprocessing.
    
    Returns:
        x: the grid of patch dynamics solutions
        t: time stamps of patch dynamics solution
        v: patch dynamics solution
        dvdt: time derivatives of patch dynamics solution
        x_ref: grid of the reference solution 
        v_ref: refernce solution for comparison
        l_traj: length of a trajectory
    """

    Data=np.load('./thehood/hdiff_1d_variedBC.npz')
    U_ref,x_ref=Data['U_ref'],Data['x_ref']
    U,x,t=Data['U'],Data['x'].squeeze(),Data['T'].squeeze()

    dt = np.diff(t)[0]
    Ut = (U[:,1:,:]-U[:,:-1,:])/dt
    Ut_ref= (U_ref[:,1:,:]-U_ref[:,:-1,:])/dt
    
    tstart,tend=0,500 

    v = U[:,tstart:tend,:]
    v_ref = U_ref[:,tstart:tend,:]
    dvdt_ref=Ut_ref[:,tstart:tend,:]
    dvdt= Ut[:,tstart:tend,:]
    t = t[tstart:tend]

    v=np.concatenate(v,axis=0)
    dvdt = np.concatenate(dvdt,axis=0)
    v_ref = np.concatenate(v_ref,axis=0)
    dvdt_ref = np.concatenate(dvdt_ref,axis=0)

    l_traj=tend-tstart

    return x,t,v,dvdt,x_ref,v_ref,dvdt_ref,l_traj

def learn_functional_PDE():
    """Learning the homogenized PDE in the functional form.
    
    Returns:
        keras.model(u,dudt) trained from patch dynamics
    """

    x,_,v,dvdt,_,_,_,_=prepare_data()
    x_train,x_test,y_train,y_test=train_test_split(v,dvdt,test_size=0.15,shuffle=False)

    nn_model = ML.Discretized_PDE1D_net(x.shape[0],3,n_conv=3)

    print(nn_model.summary())

    adamsky=tf.keras.optimizers.Adam(learning_rate=5e-4)
    nn_model.compile(optimizer=adamsky,loss='mse')
    PDEfit_history=nn_model.fit(x_train,y_train,
        batch_size=64,epochs=512,
        verbose=1,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SavePath+'1d_funcPDE_fit.png',dpi=450)

    eval_loss = np.mean((nn_model.predict(x_test).squeeze()-y_test)**2)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )
    
    return nn_model

def learn_discretized_PDE():
    """Learning the homogenized PDE in the discretized form.
    
    Returns:
        keras.model(u,dudt) trained from patch dynamics
    """

    x,_,v,dvdt,_,_,_,_=prepare_data()
    x_train,x_test,y_train,y_test=train_test_split(v,dvdt,test_size=0.15,shuffle=False)

    nn_model = ML.Discretized_PDE1D_net(x.shape[0],3,n_conv=3)

    print(nn_model.summary())

    adamsky=tf.keras.optimizers.Adam(learning_rate=5e-4)
    nn_model.compile(optimizer=adamsky,loss='mse')
    PDEfit_history=nn_model.fit(x_train,y_train,
        batch_size=64,epochs=512,
        verbose=1,validation_split=.1)

    plt.figure(figsize=[3,2.5])
    plt.plot(PDEfit_history.history['loss']/np.var(y_test),label='training loss')
    plt.plot(PDEfit_history.history['val_loss']/np.var(y_test),label='validation loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SavePath+'1d_discPDE_fit.png',dpi=450)

    eval_loss = np.mean((nn_model.predict(x_test).squeeze()-y_test)**2)
    eval_lossp=100*eval_loss/np.var(y_test)
    print('test loss %',eval_lossp )

    return nn_model


def test_model(nn_model,tag):
    """Test the learned model in RHS and integration.

    Args:
        nn_model: the trained keras.model(u,dudt)
        tag: postfix for saved figures

    Returns:
        saves comparsion figures.
    """
    x,t,v,_,x_ref,v_ref,dvdt_ref,l_traj=prepare_data()

    _,x_test,_,dvdt_truth=train_test_split(v,dvdt_ref,test_size=0.15,shuffle=False)

    # compare the snapshots of du/dt
    dvdt_nn = nn_model.predict(x_test).squeeze()

    tindex=[358,264,306]
    plt.figure(figsize=[2.15,2.5])
    for j in tindex:
        plt.plot(x_ref,dvdt_truth[j],'k--',label='truth')
        plt.plot(x,dvdt_nn[j],'x',color=bdazz_blue,label='nn')
        if j == tindex[0]:
            plt.legend()
    err = dvdt_nn-dvdt_truth[:,::20]
    rMSE= np.mean(err**2)/np.var(dvdt_truth[:,::20])
    plt.title('rMSE={:.3e}'.format(rMSE))
    plt.savefig(SavePath+'1d_dudt_'+tag+'.png',dpi=450)


    # compare trajectory integration
    v_traj = v[-l_traj:]
    v_traj_ref=v_ref[-l_traj:]

    def Net_Model(t,u):
        dudt = nn_model.predict(u[np.newaxis,:]).squeeze()
        return dudt

    u0 = v_traj[0]
    t_eval=t

    Sol = solve_ivp(Net_Model,[t_eval[0],t_eval[-1]],u0,method='BDF',t_eval=t_eval,max_step=0.01)
    v_nn=Sol.y.T

    plt.figure(figsize=[6.5,2])
    plt.subplot(1,3,1)
    plt.contourf(x_ref,t_eval,v_traj_ref,30,cmap='jet')
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')
    plt.title('truth')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.contourf(x,t_eval,v_nn,30,cmap='jet')
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')
    plt.title('nn')
    plt.colorbar()    

    err = np.abs(v_nn-v_traj_ref[:,::20])
    rMSE= np.mean(err**2)/np.var(v_traj_ref[:,::20])
    plt.subplot(1,3,3)
    plt.contourf(x,t_eval,err,30,cmap='jet')
    plt.xlabel(r'$x$'),plt.ylabel(r'$t$')
    plt.title(('error \n rMSE={:.2e}'.format(rMSE)))
    plt.colorbar()   

    plt.tight_layout()
    plt.savefig(SavePath+'1d_traj_'+tag+'.png',dpi=450)

if __name__ == "__main__":
    nn_model=learn_discretized_PDE()
    test_model(nn_model,tag='discPDE')

    nn_model=learn_functional_PDE()
    test_model(nn_model,tag='funcPDE')

    