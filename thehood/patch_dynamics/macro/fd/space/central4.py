import scipy
import interfaces

class Central4(interfaces.Derivative):
	def __init__(self):
		self.nghost=2
	def derivative(self,u_ghost,x_ghost,i):
		# only implemented for equidistant meshes
		d=scipy.zeros((scipy.shape(u_ghost)[0],\
				scipy.shape(u_ghost)[1]-4),scipy.float64)
		dx=x_ghost[1]-x_ghost[0]
		if i==1:
			d=(1/12.*u_ghost[:,:-4]-2./3.*u_ghost[:,1:-3]+\
				2./3.*u_ghost[:,3:-1]-1./12.*u_ghost[:,4:])/dx
		return d
																	      
		
