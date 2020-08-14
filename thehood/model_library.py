"""
Library of neural nets for learning PDEs.

H. Arbabi, June 2020, arbabiha--AT--gmail.com.
"""



import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Concatenate, Conv1D, Conv2D


import BarLegacy as BL

tf.keras.backend.set_floatx('float64')

def Functional_PDE1D_net(n_grid: int, dx: float, n_stencil: int, n_conv=3):
    """A neural net model for learning 1d PDEs with Fixed BC.

    The model computes u_x,u_xx via finite difference and then
    at each x models u_t=f(u,u_x,u_xx) with trainable neural net.

    Args:
        n_grid: size of the (periodically-padded) input grid
        dx: uniform grid spacing
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """


    u=tf.keras.Input(shape=(n_grid,),name="input_field")

    # fixed layer for u_xx
    laplacian_layer = finite_diff_layer(dx,2,n_stencil)
    u_xx= laplacian_layer(u)

    # fixed layer for u_x
    laplacian_layer = finite_diff_layer(dx,1,n_stencil)
    u_x= laplacian_layer(u)

    if n_stencil>3:
        raise NotImplementedError('stencil size larger than 3 requires special treatment of BC')

    # putting u,u_x and u_xx together
    us = tf.stack((u,u_x,u_xx),axis=-1,name='stack_u_ux_uxx')

    # 1st convolution+ activation layers
    clayer1= Conv1D(32,1,padding='valid',name='convolution_1')
    u_out = clayer1(us)
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv1D(32,1,padding='valid',name='convolution_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv1D(1,1,padding='valid',name='convolution_'+str(n_conv))
    u_out = clayer(u_out)

     # pad the two sides with zero (du/dt at boundary points is zero)
    padding_size = tf.constant([[0,0],[1, 1,], [0, 0]])
    u_out=tf.pad(u_out, padding_size, mode='CONSTANT', constant_values=0, name='pad_zero_BC')


    return tf.keras.Model(u,u_out)



def stencil_embedding(inputs:tf.Tensor,stencil_width:int,fixed_Dirichlet_BC=False)-> tf.Tensor:
    """Embedding the input data with the size of stencil.

    Args:
        inputs: values of the field on a periodic 1d grid, shape=(...,x)
        stencil_width: width of the stencil
        fixed_Dirichlet_BC: what is the BC?

    Returns:
        tensor of shape (...,x,stencil_width) the values of stencil nodes
    """
    if stencil_width % 2 ==1:
        npad = stencil_width//2
    else:
        raise NotImplementedError


    padded_inputs=tf.concat([inputs[:,-npad:],inputs,inputs[:,:npad]],axis=-1)

    # we add (y,depth) dimension to fake an image
    embedded=tf.image.extract_patches(padded_inputs[...,tf.newaxis,tf.newaxis],
                                      sizes=[1,stencil_width,1,1],
                                      strides=[1, 1, 1, 1],
                                      rates=[1, 1, 1, 1],
                                      padding='VALID',name='stencil_embeded')


    return tf.squeeze(embedded,axis=-2,name='squeeeze_me')  # remove (y,) dimension

def Discretized_PDE1D_net(n_grid: int, n_stencil: int, n_conv=3):
    """A CNN model for learning FIXED-BC homogenized diffusion.

    E.g. the model with n_stencil=3 learns du_j/dt=f(u_j,u_{j-1},u_{j+1}).

    Args:
        n_grid: size of the (periodically-padded) input grid
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """
    u=tf.keras.Input(shape=(n_grid,),name="input_field")


    # 1st convolution+ activation layers
    clayer1= Conv1D(32,n_stencil,padding='valid',name='convolution_1')
    u_out = clayer1(u[...,tf.newaxis])
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv1D(32,1,padding='valid',name='convolution_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv1D(1,1,padding='valid',name='convolution_'+str(n_conv))
    u_out = clayer(u_out)

     # pad the two sides with zero (du/dt at boundary points is zero)
    padding_size = tf.constant([[0,0],[1, 1,], [0, 0]])
    u_out=tf.pad(u_out, padding_size, mode='CONSTANT', constant_values=0, name='pad_zero_BC')

    return tf.keras.Model(u,u_out)



def Discretized_PDE2D_net(n_grid: list, n_stencil: int, n_conv=3, n_filter1 = 16):
    """A CNN model for learning Burgers PDE.

    First layer is a 2D convolution, then a couple of dense layers + ReLu

    Args:
        n_grid: size of the (periodically-padded) input grid
        n_stencil: the kernel size of the first convolutional layer, 
            AND the stencil size of the finite difference
        n_conv: total number of convolutional layers
        n_filter1: number of filters for the first convolutional layer  

    Returns:
        tf.keras.model that maps the field u (input) to u_t (output). 
    """
    u=tf.keras.Input(shape=(n_grid[0],n_grid[1],),name="input_field")

    # periodify the field
    # u_padded = periodify_2D(u,n_stencil)
    player = periodify2D(nkernel=n_stencil)
    u_out = player(u)

    # add a dummy channel at the end
    channel_layer = add_dummy_channel()
    u_out = channel_layer(u_out)

    # first convolution
    clayer1= Conv2D(n_filter1,n_stencil,padding='valid',name='convolution_1') 
    u_out = clayer1(u_out)
    u_out = tf.keras.activations.relu(u_out)



    for layer_ind in range(n_conv-2):
        clayer= Conv2D(32,1,padding='valid',name='pointwise_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv2D(1,1,padding='valid',name='pointwise_'+str(n_conv))
    u_out = clayer(u_out)

    return tf.keras.Model(u,u_out)


def Functional_PDE2D_net_wfft(input_shape: list,  n_conv=3, n_filter1 = 32):
    """Functional model of learning 2D PDE with periodic BC.

    We have compute u_x,u_xy,... and here feed them to the network.

    Args:
        input_shape: [xgrid_size, ygrid_size, number of derivatives]
        n_conv: total number of convolutional layers
        n_filter1: number of filters for the first convolutional layer  

    Returns:
        tf.keras.model that maps the field (u,u_x,...) (input) to u_t (output). 
    """
    u=tf.keras.Input(shape=input_shape,name="input_field")

    # first convolution (really a dense layer)
    clayer1= Conv2D(n_filter1,1,padding='valid',name='convolution_1') 
    u_out = clayer1(u)
    u_out = tf.keras.activations.relu(u_out)

    for layer_ind in range(n_conv-2):
        clayer= Conv2D(16,1,padding='valid',name='pointwise_'+str(layer_ind+2))
        u_out = clayer(u_out)
        u_out = tf.keras.activations.relu(u_out)
    
    clayer= Conv2D(1,1,padding='valid',name='pointwise_'+str(n_conv))
    u_out = clayer(u_out)

    return tf.keras.Model(u,u_out)



class finite_diff_layer(tf.keras.layers.Layer):
    """A layer of frozen finite difference on uniform periodic grid."""

    def __init__(self, dx: float, derivative_order: int, stencil_size: int):
        """Constructor.

        Args:
            dx: spacing between grid points
            derivative_order: larger than 0
            stencil_size: at this point we only accept odd numbers 
        """
        super(finite_diff_layer, self).__init__()
        assert stencil_size % 2 ==1, "I accept only odd stencil size"
        self.stencil_size=stencil_size

        int_grid= np.arange(-stencil_size//2+1,stencil_size//2+1,1)
        local_grid = int_grid* dx   # local position of points

        damethod=BL.Method.FINITE_DIFFERENCES
        standard_coeffs= BL.coefficients(local_grid,damethod,derivative_order)
        
        self.coeffs=tf.constant(standard_coeffs,dtype=tf.float64,name='df_coeffs_O'+str(derivative_order))

    def build(self, input_shape):
        pass

    def call(self,u):
        u_embeded=stencil_embedding(u, self.stencil_size)
        return tf.einsum('s,bxs->bx', self.coeffs, u_embeded)



class periodify2D(tf.keras.layers.Layer):
    """Pad the input (last axis) so that it is effectively periodic."""

    def __init__(self,nkernel:int=3):
        """ Constructor for the periodify layer.
        
        Args:
            nkernel: size of the kernel to be applied to data. 
                The length of appended length is (int(nkernel/2)).
        """
        super(periodify2D, self).__init__()
        assert nkernel % 2 == 1, "this baby only accepts odd nkernels."
        self.npad=nkernel//2

    def build(self, input_shape):
        # print('what is goin on in build?')
        super(periodify2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # print('what is goin on in compute_output_shape?')
        return (input_shape[-3], input_shape[-2] + self.npad + self.npad,input_shape[-1] + self.npad + self.npad)

    def call(self,u):
        padded_x=tf.concat([u[...,-self.npad:],u,u[...,:self.npad]],axis=-1)
        padded_xy = tf.concat([padded_x[...,-self.npad:,:],padded_x,padded_x[...,:self.npad,:]],axis=-2)
        return padded_xy


class add_dummy_channel(tf.keras.layers.Layer):
    """Adding a channel to the end."""

    def __init__(self):
        """ Constructor for the add channel layer.
        
        """
        super(add_dummy_channel, self).__init__()

    def build(self, input_shape):
        # print('what is goin on in build?')
        super(add_dummy_channel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # print('what is goin on in compute_output_shape?')
        return (input_shape[-3], input_shape[-2],input_shape[-1],1)

    def call(self,u):
        return u[...,tf.newaxis]


if __name__=='__main__':
    pass



