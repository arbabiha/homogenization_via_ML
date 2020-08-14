import scipy
import interfaces

class UpwindB(interfaces.Derivative):
	def __init__(self):
		self.nghost=2
	def derivative(self,u_ghost,x_ghost,i):
		# only implemented for equidistant meshes
		d=scipy.zeros((scipy.shape(u_ghost)[0],\
				scipy.shape(u_ghost)[1]-4),scipy.float64)
		dx=x_ghost[1]-x_ghost[0]
		if i==1:
			d=(1/6.*u_ghost[:,:-4]-u_ghost[:,1:-3]+\
				1./2.*u_ghost[:,2:-2]+1./3.*u_ghost[:,3:-1])/dx
		return d
																	      
		
