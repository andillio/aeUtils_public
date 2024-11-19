# pylint: disable=C,W
import numpy as np 
import warnings as warn
CUPY_IMPORTED = True
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False  

if not(CUPY_IMPORTED):
	warn.warn("cupy was not loaded")	

# TODO:
# 	- implement staggered imports
class State(object):

	def __init__(self):
		self.gpu_all = False
STATE = State()
# func(b)
def GPU_ALL(state = True):
	"""
	sets default for actions to take place on the gpu

	:state: bool, whether default is for actions to take place on the gpu
	"""
	if not(CUPY_IMPORTED) and state:
		warn.warn(
			"cannot set GPU_ALL = True because cupy was not successfully imported."
			)
		return
	STATE.gpu_all = state

# notes: in examples
# 	- x represents an arbitrary array
# 	- X represents an arbitrary array
# 	- I represents an iterable
#  	- i represents an arbitrary int
# 	- f represents an arbitrary float


def logSpace(min_, max_, N, gpu = False, dtype = None):
	"""
	returns a logspace between min and max
	"""
	np_ = np
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp

	dtype_ = np_.double

	if dtype != None:
		dtype_ = dtype

	return np_.logspace(np_.log10(min_), np_.log10(max_), N, dtype=dtype_)


# ex call grid( (i,i), f, b)
# warning: grid needs same N in each dimension
def grid( N, L = 1., centered = True, gpu = False, dtype = None):
	""" 
	returns a numpy grid giving the coordinate values at the
	center of each grid cell

	:N: array-like or integer, gives the resolution of the grid
	:L: float, gives the box length of the grid
	:centered: bool, should the grid be centered at 0 (as opposed
 			   to at L/2.), dafault True
	:gpu: bool, should run on gpu
	:dtype: data-type, default double

	:return: arrays representing coordinate values 
			defined at center of grid cells
	""" 
	np_ = np
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp
	N_ = N 

	if type(N) != int:
		N_ = N[0]

	dtype_ = np_.double

	if dtype != None:
		dtype_ = dtype

	dx = L * 1. / N_
	x = np_.arange(N_, dtype=dtype_)*dx + dx / 2.

	if centered:
		x = x - L/2.

	if type(N) == int or len(N) == 1:
		# 1d grid
		return x 
		


	ones = np_.ones(N_, dtype=dtype_)

	if len(N) == 2:
		# 2d grid
		X = np_.einsum("i,j->ij", x, ones)
		Y = np_.einsum("i,j->ij", ones, x)

		return X, Y 
	if len(N) == 3: 
		# 3d grid
		X = np_.einsum("i,j,k->ijk", x, ones, ones)
		Y = np_.einsum("i,j,k->ijk", ones, x, ones)
		Z = np_.einsum("i,j,k->ijk", ones, ones, x)

		return X, Y, Z
	raise ValueError(
		"trying to implement a grid dimensionality that is not implemented\n" +\
		"reccomendation: N should be an int or 1D array of length <= 3"
		)

# ex cpu2gpu(I,I)
# tries to convert objects saved on the cpu to the gpu
def cpu2gpu(*args):
	"""
	moves objects from the cpu to gpu

	:*args: iterable of array like objects 
	:return: gpu pointers to objects in *args
	""" 
	rvals = []
	for arg in args:
		rvals.append(gpuThis(arg))

	return tuple(rvals)

# ex func(I,I)
# tries to convert objects saved on the gpu to the cpu
def gpu2cpu(*args):
	"""
	moves objects from the gpu to cpu

	:*args: iterable of array like objects 
	:return: cpu pointers to objects in *args
	""" 
	rvals = []
	for arg in args:
		rvals.append(cpuThis(arg))
	return tuple(rvals)

# ex func(I)
# tries to convert x saved on the cpu to the gpu
def gpuThis(x):
	"""
	moves objects from the cpu to gpu

	:x: array-like object 
	:return: gpu pointers to objects in *args
	""" 
	if CUPY_IMPORTED:
		return cp.asarray(x)
	raise ModuleNotFoundError(
		"object could not be converted because cupy was not loaded sucessfully\n" +\
		"recommendation: run on cpu (try checking configuration paramters) or install cupy"
		)

# ex func(I)
# tries to convert x saved on the gpu to the cpu
def cpuThis(x, supress_warning = False):
	"""
	moves objects from the cpu to gpu

	:x: array-like object
	:supress_warning: should I ignore gpu not configured warning 
	:return: gpu pointers to objects in *args
	""" 
	if CUPY_IMPORTED:
		return cp.asnumpy(x)
	if not(supress_warning):
		warn.warn(
			"possible erroneous call to cpuThis. cupy is not loaded and so" +\
			" it is likely that the object is already in cpu memory"
			)
	return x 

# ex: func(X,X,X,b)
# converts 3d cartesian coordinates to 3d spherical coordinates 
def cart2sphr(X,Y,Z, gpu = False):
	"""
	converts coordinate arrays X,Y,Z to spherical coordinates R,Theta,Phi

	:X,Y,Z: array-like objects 
	:gpu: bool, try to perform operation using gpu
	:return: pointers to coordinate arrays R, Theta (azimuthal), Phi (polar)
	""" 
	np_ = np
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp
	s2 = X**2 + Y**2
	R = np_.sqrt(s2 + Z**2) # radius
	Phi = np_.arctan2(np_.sqrt(s2), Z) # polar angle
	Theta = np_.arctan2(Y, X) # azimuthal angle
	return R, Theta, Phi


def sphrGrid(N, L, gpu = False, dtype = None):
	""" 
	returns a numpy grid giving the coordinate values at the
	center of each grid cell

	:N: integer, gives the resolution of the grid
	:L: float, gives the box length of the grid
	:gpu: bool, try to perform operation using gpu
	:dtype: data-type, default double

	:return: array-like, (R,Theta,Phi), coordinate values 
			defined at center of grid cells \n
			R - radial coordinate, \n
			Theta - azimuthal angle, \n
			Phi - polar angle
	""" 
	X, Y, Z = grid((N, N, N), L = L, gpu = gpu, dtype=dtype)
	return cart2sphr(X,Y,Z, gpu = gpu)

# converts 2d cartesian coordinates to 2d circular coordinates
def cart2cir(X,Y, gpu = False):
	"""
	converts coordinate arrays X,Y to circular coordinates R,Phi

	:X,Y: array-like, objects 
	:gpu: bool, try to perform operation using gpu
	:return: pointers to coordinate arrays R, Theta (azimuthal)
	""" 
	np_ = np
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp
	s2 = X**2 + Y**2
	Theta = np_.arctan2(Y, X) # azimuthal angle
	return np.sqrt(s2), Theta

def cirGrid(N, L, gpu = False):
	""" 
	returns a numpy grid giving the coordinate values at the
	center of each grid cell

	:N: integer, gives the resolution of the grid
	:L: float, gives the box length of the grid
	:gpu: try to perform operation using gpu 

	:return: array-like, (R,Phi), coordinate values 
			defined at center of grid cells \n
			R - radial coordinate, \n
			Theta - azimuthal angle
	""" 
	X, Y = grid((N, N), L = L, gpu = gpu)
	return cart2cir(X,Y, gpu = gpu)