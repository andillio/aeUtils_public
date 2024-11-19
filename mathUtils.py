# pylint: disable=C,W
# basic math function utility file
import gridUtils as gu
from matplotlib.pyplot import axis
CUPY_IMPORTED = True 
try:
	import cupy as cp 
	import cupyx as cpx
except ImportError:
	CUPY_IMPORTED = False
import scipy.special as spp
import scipy.integrate as integrate
from scipy import linalg as LA
import scipy.stats as stats
from scipy import signal
import warnings as warn
import numpy as np
PYSHTOOLS_IMPORTED = True 
try:
	import pyshtools as pysh 
except ImportError:
	PYSHTOOLS_IMPORTED = False

# type notes:
# 	- x represents an array
# 	- X represents an array
# 	- I represents an iterable
#  	- i represents an int
# 	- f represents a float
# 	- o represents an object

# TODO:
# - add stagered import functionality
class State(object):

	def __init__(self):
		self.gpu_all = False
STATE = State()

# func(X,X,X,b)
# converts 3d cartesian coordinates to 3d spherical coordinates 
def cart2sphr(X,Y,Z, gpu = False):
	"""
	converts coordinate arrays X,Y,Z to spherical coordinates R,Theta,Phi

	:X,Y,Z: array-like spatial coordinates
	:gpu: try to perform operation using gpu
	:return: pointers to coordinate arrays R,Theta,Phi
	""" 
	return gu.cart2sphr(X,Y,Z, gpu = gpu or STATE.gpu_all)

# func(i,i,X,X,b)
# spherical harmonic
def sph_harm(m, l, theta, phi, gpu = False):
	"""
	spherical harmonic

	:m: integer, denoting energy quantum number
	:l: integer, denoting angular momentum quantum number
	:theta: array-like, azimuthal angle
	:phi: array-like, polar angle
	:gpu: bool, try to run on gpu, default False
	:return: array-like, values at theta, phi
	""" 
	# check if this is l > 86
	# if it is then try to use pysh 
	# if its not then check if it needs to run on the gpu
	# if it does use cpx 
	# if it does not then try to use pysh

	if l >= 86:
		return sph_harm_pysh(m, l, theta, phi, gpu = gpu)

	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		return cpx.scipy.special.sph_harm(m,l,theta,phi)

	if gpu:
		warn.warn(
			"gpu option set to True, but cupy was not loaded successfully"
			)

	if PYSHTOOLS_IMPORTED:
		return sph_harm_pysh(m, l, theta, phi, gpu = gpu)

	return spp.sph_harm(m,l,theta,phi)


def sph_harm_pysh(m, l, theta, phi, gpu = False):
	"""
	spherical harmonic

	:m: integer, denoting energy quantum number
	:l: integer, denoting angular momentum quantum number
	:theta: array-like, azimuthal angle
	:phi: array-like, polar angle
	:gpu: bool, try to run on gpu, default False

	:return: array-like, values at theta, phi
	""" 

	if gpu:
		warn.warn(
			"gpu option set to True, but gpu not implemented for pyshtools."
			)

	sign = 1. 
	if m < 0:
		sign = 1 if m%2==0 else -1
	x = np.cos(phi)
	phase = np.exp(1j*m*theta)
	rval = pysh.legendre.legendre_lm(l,np.abs(m),x,
	    csphase = -1, normalization = 'ortho')/np.sqrt(2)
	rval = rval*phase

	if m<0:
		rval = np.conjugate(rval)

	return rval*sign

# func(x,x,f,f)
# integrates given array
def cumInt(y, x = [], dx = 1., initial = 0, gpu = False):
	"""
	calculates the cumulative integral of y using trapezoid rule
	only 1D implemented 

	:y: array-like, the data points to be integrated
	:x: array-like, data x values, default is None
	:dx: float, differential, default is 1
	:initial: float, initial value, default is 0

	:return: array-like, integral of data 
	"""
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:

		dX = cp.ones(y.shape)*dx	
		if len(x) > 0:
			dX = x - cp.roll(x,1)

		rval = .5*(y + cp.roll(y,1))*dX 
		rval[0] = initial 
		return cp.cumsum(rval)

	if len(x) > 0:
		return integrate.cumulative_trapezoid(y, x, initial=initial)

	return integrate.cumulative_trapezoid(y, dx = dx, initial= initial)

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


# gives differentiation stencils
# func(i,i,i,f,b,b)
def stencil(N, pts = 3, order = 1, dx = 1., periodic = True, gpu = False):
	"""
	finite central difference stencils

	:N: int, grid resolution
	:pts: int, number of points in stencil, default 3
	:order: int, order of the derivative, default 1
	:dx: float, differential element, default 1
	:periodic: bool, should we have periodic boundaries
	:gpu: bool, try to run on gpu, default False

	:return: array-like, N by N finite difference matrix 	
	"""
	if pts > N:
		raise Exception(
			"N cannot be less than pts.\n" +\
			"reccomendation: increase N or lower pts"
			)

	if pts > 9:
		warn.warn("stencil is only implmented to pts <= 8")
	if order > 2:
		warn.warn("stencil is only implement to order <= 2")

	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp 

	I = np_.eye(N)
	M = np_.zeros(I.shape)

	if order == 1:
		
		I *= 1./dx

		if pts == 3:
			M = -5.*np_.roll(I,1, axis = 0) + .5*np_.roll(I,-1, axis = 0)
		if pts == 5:
			M = (1./12)*np_.roll(I,2, axis = 0) + (-2/3.)*np_.roll(I,1, axis = 0) +\
				(2/3.)*np_.roll(I,-1, axis = 0) + (-1./12)*np_.roll(I,-2, axis = 0)
		if pts == 7:
			M = (-1./60)*np_.roll(I,3, axis = 0) + (3/20.)*np_.roll(I,2, axis = 0) +\
				(-3./4)*np_.roll(I,1, axis = 0) + (3/4.)*np_.roll(I,-1, axis = 0) +\
				(-3/20.)*np_.roll(I,-2, axis = 0) + (1./60)*np_.roll(I,-3, axis = 0)
		if pts == 9:
			M = (1./280)*np_.roll(I,4, axis = 0) + (-1/280.)*np_.roll(I,-4, axis = 0) +\
				(-4./105)*np_.roll(I,3, axis = 0) + (4/105.)*np_.roll(I,-3, axis = 0) +\
				(1/5.)*np_.roll(I,2, axis = 0) + (-1./5)*np_.roll(I,-2, axis = 0) +\
				(-4/5.)*np_.roll(I,1, axis = 0) + (4./5)*np_.roll(I,-1, axis = 0)

	if order == 2:

		I *= 1./dx**2

		if pts == 3:
			M = -2*I + np_.roll(I,1, axis = 0) + np_.roll(I,-1, axis = 0)
		if pts == 5:
			M = -(5/2.)*I + (4/3.)*np_.roll(I,1, axis = 0) + (4/3.)*np_.roll(I,-1, axis = 0) +\
				(-1/12.)*np_.roll(I,2, axis = 0) + (-1/12.)*np_.roll(I,-2, axis = 0)
		if pts == 7:
			M = -(49/18.)*I + (3/2.)*np_.roll(I,1, axis = 0) + (3/2.)*np_.roll(I,-1, axis = 0) +\
				(-3/20.)*np_.roll(I,2, axis = 0) + (-3/20.)*np_.roll(I,-2, axis = 0) +\
				(1/90.)*np_.roll(I,3, axis = 0) + (1/90.)*np_.roll(I,-3, axis = 0)
		if pts == 9:
			M = -(205/72.)*I + (8/5.)*np_.roll(I,1, axis = 0) + (8/5.)*np_.roll(I,-1, axis = 0) +\
				(-1/5.)*np_.roll(I,2, axis = 0) + (-1/5.)*np_.roll(I,-2, axis = 0) +\
				(8/315.)*np_.roll(I,3, axis = 0) + (8/315.)*np_.roll(I,-3, axis = 0) +\
				(-1/560.)*np_.roll(I,4, axis = 0) + (-1/560.)*np_.roll(I,-4, axis = 0)

	if not(periodic):
		for j in range(pts):
			for i in range(pts-j):
				M[j,N-1-i] = 0
				M[N-1-j,i] = 0

	return M


# solves eigenvalue problem for matrix H
def Eig(H, gpu = False):
	"""
	gives the eigenvalues and corresponding vectors sorted 
	by the real part of the eigenvalues

	:H: array-like, square matrix to be solved
	:gpu: bool, try to run on gpu, default False

	:return: (array-like, array-like), (eigenvalues, eigenvectors),\n
		eigenvalues - [N] array of eigenvalues in ascending order,\n
		eigenvectors - [N,N] array of eigenvectors ordered by eigenvalue
	"""
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		E, psi = cp.linalg.eigh(H)
	else:
		E, psi = np.linalg.eig(H)
	psi = sortVects(E.real,psi)
	E = sortE(E.real,E)
	return E,psi

# func(x,x,b)
# solves eigenvalue problem for matrix H
def Eig_tridiag(diag, offdiag, gpu = False):
	"""
	gives the eigenvalues and corresponding vectors sorted 
	by the real part of the eigenvalues
	(warning: gpu functionality not implemented)

	:diag: array-like, [N] diagonal elements of Hamiltonian
	:offdiag: array-like, [N-1] off diagonal elements of Hamiltonian
	:gpu: bool, try to run on gpu, default: False

	:return: (array-like, array-like), (eigenvalues, eigenvectors),\n
		eigenvalues - [N] array of eigenvalues in ascending order,\n
		eigenvectors - [N,N] array of eigenvectors ordered by eigenvalue 
	"""
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		warn.warn("GPU functionality not added to this function." +\
			"Using standard Eig function instead.")
		N = len(diag)
		H = cp.diag(diag)
		H_off = cp.diag(offdiag)
		H[1:,1:] += H_off
		H[:N-1,:N-1] += H_off
		return Eig(H, gpu = True)
	E, psi = LA.eigh_tridiagonal(diag, offdiag)
	psi = sortVects(E.real, psi)
	E = sortE(E.real,E)
	return E,psi

# sorts the eigenvectors A
# using key
def sortVects(key, A):
	"""
	sorts A using key

	:key: array-like, key[N], keys with which to sort A
	:A: array-like, A[N,M], list to be sorted

	:return: array-like, sorted(A)
	"""
	inds = key.argsort()
	A = A[:,inds]
	return A

# sort eigenvalues A
# using key
def sortE(key, A):
	"""
	sorts A using key

	:key: array-like, key[N], keys with which to sort A
	:A: array-like, A[N], list to be sorted

	:return: array-like, sorted(A)
	"""
	inds = key.argsort()
	A = A[inds]
	return A

# noramlizes the passed in vector
# func(x, f, x, b, b)
def vNormalize(v, dx = 1., r = None, radial = False, gpu = False):
	"""
	normalizes v using its L2 norm

	:v: array-like, the vector to normalize
	:dx: float, differential element, default 1
	:r: array-like, radius values, default None, needed if radial is true
	:radial: bool, should sum using a radial integral
	:gpu: bool, try to run on gpu, default False
	
	:return: None, this function directly normalizes the passed in vector
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp 

	if radial:
		v /= np.sqrt(np_.sum(r**2 * np_.pi * 4 * np_.abs(v)**2)*dx)
	else:
		dim = len(v.shape)

		v /= np_.sqrt(np_.sum(np_.abs(v)**2)*dx**dim)


def Normalize(y, dx = 1., r = None, radial = False, gpu = False):
	"""
	normalizes y using its L1 norm

	:y: array-like, the vector to normalize
	:dx: float, differential element, default 1
	:r: array-like, radius values, default None, needed if radial is true
	:radial: bool, should sum using a radial integral
	:gpu: bool, try to run on gpu, default False
	
	:return: None, this function directly normalizes the passed in vector
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp 

	if radial:
		y /= np_.sum(r**2 * np_.pi * 4 * np_.abs(y))*dx
	else:
		dim = len(y.shape)

		y /= np_.sum(np_.abs(y)*dx**dim)


def Norm_L2(v, dx = 1., D = 1, axes = None, gpu = False):
	"""
	gives the L2 norm of the given vector
	
	:v: array-like, vector to be normalized
	:dx: float, integration metrix
	:D: int, number of spatial dimensions integrated over

	:return: norm of v
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp 
	return np_.sqrt(np_.sum(np_.abs(v)**2, axis = axes)*dx**D)


def Norm_L1(v, dx = 1., D = 1, axes = None, gpu = False):
	"""
	gives the L1 norm of the given vector
	
	:v: array-like, vector to be normalized
	:dx: float, integration metrix
	:D: int, number of spatial dimensions integrated over

	:return: norm of v
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp 
	return np_.sum(np_.abs(v), axis = axes)*dx**D



# TODO:
# - rewrite this with scipy signal
# - implement either using L2 or L1 norm for kernel
def fftConvolve(X1,X2, D = None, gpu = False):
	"""
	convolves two functions using an fft

	:X1: array-like, first function
	:X2: array-like, convolution kernel
	:D: integer, dimensionality of X1
	:gpu: bool, should run on gpu

	:return: convolution of X1 and X2
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp

	N = X1.shape[0]

	if D == None:
		D = len(np_.shape(X1))

	if D == 1:
		K1 = np_.fft.fft(X1)
		K2 = np_.fft.fft(X2)
		return np_.fft.fftshift(np_.fft.ifft(K1*K2))
	elif D == 2:
		K1 = np_.fft.fft2(X1)
		K2 = np_.fft.fft2(X2)
		return np_.fft.fftshift(np_.fft.ifft2(K1*K2), axes = (0,1))
	elif D == 3:
		K1 = np_.fft.fftn(X1)
		K2 = np_.fft.fftn(X2)
		return np_.fft.fftshift(np_.fft.ifftn(K1*K2), axes = (0,1,2))
	else:
		raise Exception("Dimension of array not supported.\n"+\
			"reccomendation: check shape of array")


def powerspectrum(rho, dx = 1., dV_norm = False):
	"""
	finds the power spectrum of the given matrix

	:rho: array-like
	:dx: float, pixel size, dafault 1

	:returns: (array-like, array-like), (kvals, Abins) \n
	kvals - k values power spectrum is evaluated at \n
	Abins - power spectrum values
	"""
	N = rho.shape[0]
	D = len(rho.shape)

	rho_k_amps = np.abs( np.fft.fftn(rho) / np.sqrt(N)**D )**2
	
	kx = 2*np.pi*np.fft.fftfreq(N,d = dx)
	ones = np.ones(N)
	knorm = kx

	if D == 2:
		kX = np.einsum("i,j->ij", kx, ones)
		kY = np.einsum("i,j->ij", ones, kx)
		knorm = np.sqrt(kX**2 + kY**2)
	elif D == 3:
		kX = np.einsum("i,j,k->ijk", kx, ones, ones)
		kY = np.einsum("i,j,k->ijk", ones, kx, ones)
		kZ = np.einsum("i,j,k->ijk", ones, ones, kx)
		knorm = np.sqrt(kX**2 + kY**2 + kZ**2)

	knorm = knorm.flatten()
	rho_k_amps = rho_k_amps.flatten()

	kbins = np.arange(0.5, N//2+1, 1.)*(2.*np.pi/dx/N)

	dV = kbins[1:] - kbins[:-1]
	if D == 2:
		dV = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
	elif D == 3:
		dV = 4*np.pi/3 * (kbins[1:]**3 - kbins[:-1]**3)

	kvals = 0.5 * (kbins[1:] + kbins[:-1])
	Abins, _, _ = stats.binned_statistic(knorm, rho_k_amps,
	    statistic = "mean", bins = kbins)
	if dV_norm:
		Abins *= dV

	return kvals, Abins


def gradient_1D(phi, dx = 1., axis_ = 0, gpu = False, padded = False):
	"""
	gives the gradient of phi in one direction using 5-pt stencil

	:phi: array-like
	:dx: float, grid resolution
	:axis_: int, axis to take gradient along
	:gpu: bool, run on the gpu
	:padded: bool, edges are padded
	"""
	np_ = np 
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp

	dphi = 8*np_.roll(phi,-1,axis = axis_) \
	    - 8.*np_.roll(phi,1,axis = axis_) \
	    - np_.roll(phi,-2,axis = axis_) \
	    + np_.roll(phi,2,axis = axis_)
	
	dphi /= dx*12.

	N = phi.shape[axis_]
	if padded:
		if axis_ == 0:
			dphi[1] = .5*(phi[2] - phi[0])/dx
			dphi[N-2] = .5*(phi[N-1] - phi[N-3])/dx
			dphi[N-1] = (phi[N-1] - phi[N-2])/dx
			dphi[0] = (phi[1] - phi[0]) / dx
		if axis_ == 1:
			dphi[:,1] = .5*(phi[:,2] - phi[:,0])/dx
			dphi[:,N-2] = .5*(phi[:,N-1] - phi[:,N-3])/dx
			dphi[:,N-1] = (phi[:,N-1] - phi[:,N-2])/dx
			dphi[:,0] = (phi[:,1] - phi[:,0]) / dx
		if axis_ == 2:
			dphi[:,:,1] = .5*(phi[:,:,2] - phi[:,:,0])/dx
			dphi[:,:,N-2] = .5*(phi[:,:,N-1] - phi[:,:,N-3])/dx
			dphi[:,:,N-1] = (phi[:,:,N-1] - phi[:,:,N-2])/dx
			dphi[:,:,0] = (phi[:,:,1] - phi[:,:,0]) / dx

	return dphi 


def GetFFt(psi, Forward = True, gpu = False):
	"""
	calculate the fft of psi

	:psi: array-like, [N^D], the fields
	:Forward: bool, forward or backward fft, default: True
	:gpu: bool, run on the gpu

	:return: array-like, fft of psi
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp
	D = len(psi.shape)
	N = len(psi)

	if Forward:
		if D == 1:
			return np_.fft.fft(psi, axis = 0)
		if D == 2:
			return np_.fft.fft2(psi, axes = (0,1))
		if D == 3:
			return np_.fft.fftn(psi, axes = (0,1,2))
	else:
		if D == 1:
			return np_.fft.ifft(psi, axis = 0)
		if D == 2:
			return np_.fft.ifft2(psi, axes = (0,1))
		if D == 3:
			return np_.fft.ifftn(psi, axes = (0,1,2))


def interpAtPos3D(r, dx, yf, leftEdge = 0, gpu = False):
	"""
	linear interps the values of a function at positions x in 3D

	:r: array-like (n_points, 3), x values to perform the interpolation at
	:dx: float, grid resolution
	:yf: array-like (N, N, N), function values
	:leftEdge: float, left edge of x grid

	:return: array-like (n_points), function values interped at x
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp
	N = len(yf)
	n_points = len(r)

	rval = np_.zeros(n_points)
	
	ijk = np_.floor((r - leftEdge) / dx).astype(int) # [n_particles, [i,j,k]]\
	ijk %= N

	x = dx*(.5+np_.arange(-1*N//2, N//2))

	# i,j,k
	f1 = 1 - np_.abs(x[ijk[:,0]] - r[:,0])/dx
	f2 = 1 - np_.abs(x[ijk[:,1]] - r[:,1])/dx
	f3 = 1 - np_.abs(x[ijk[:,2]] - r[:,2])/dx
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i+1,j,k
	f1 = 1. - f1 # dx
	ijk[:,0] += 1
	ijk[:,0] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i+1,j+1,k
	f2 = 1. - f2 # tx
	ijk[:,1] += 1
	ijk[:,1] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i,j+1,k
	f1 = 1. - f1
	ijk[:,0] -= 1
	ijk[:,0] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i,j+1,k+1
	f3 = 1. - f3
	ijk[:,2] += 1
	ijk[:,2] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i,j,k+1
	f2 = 1. - f2
	ijk[:,1] -= 1
	ijk[:,1] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i+1,j,k+1
	f1 = 1. - f1
	ijk[:,0] += 1
	ijk[:,0] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	# i+1,j+1,k+1
	f2 = 1. - f2
	ijk[:,1] += 1
	ijk[:,1] %= N
	rval[:] += yf[ijk[:,0],ijk[:,1],ijk[:,2]]*f1*f2*f3

	return rval


def AngleBetweenVectors(vector_1, vector_2):
	"""
	calculates the angle between two vectors

	:vector_1: array-like
	:vector_2: array-like

	:return: float, the angle between the vectors (0, pi)
	"""
	unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
	unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
	if np.linalg.norm(vector_1) == 0 or np.linalg.norm(vector_2) == 0:
		return np.arccos(0)
	if np.allclose(vector_1, vector_2):
		return 0
	return np.arccos(np.dot(unit_vector_1, unit_vector_2))


def nCx(n, x, int_return = False):
	"""
	calculate n choose x

	:n: int, number of things
	:x: int, number of elements taken
	:int_return: bool, return an integer, default False

	:returns: int,float,
	"""
	return spp.comb(n,x, exact = int_return)

def RemoveNans(M, gpu = False):
	"""
	removes the infs and nans from the given matrix
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	M[np_.isnan(M)] = 0
	M[np_.isinf(M)] = 0


def rms(a, gpu = False):
	"""
	removes the root mean square of a matrix
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	return np_.sqrt(np_.mean(np_.abs(a)**2))

def maximum(*args, gpu = False):
	"""
	finds the maximum value for each element in a series of arrays
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	rval = args[0]
	for i in range(1,len(args)):
		rval = np_.maximum(rval, args[i])
	return rval

def minimum(*args, gpu = False):
	"""
	finds the minimum value for each element in a series of arrays
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	rval = args[0]
	for i in range(1,len(args)):
		rval = np_.minimum(rval, args[i])
	return rval


def array_to_pdf(y, res = 256):
	"""
	returns a normalized pdf for the entries in the array y

	:y: array-like, values used to construct pdf
	:res: int, resolution of returned array, default 256

	:returns: (array-like, array-like), (x, p(x)) \n
	x - locations pdf defined at \n
	pdf - pdf values
	"""
	xMin = np.min(y)
	xMax = np.max(y)
	dx = (xMax - xMin) / res

	x = (np.arange(res) + 0.5) * dx + xMin

	cdf = np.zeros(res + 4)
	cdf[res+3] = 1
	cdf[res+2] = 1

	n = len(y)

	for i in range(2, res + 2):
		x_val = x[i-2]
		cdf[i] = np.sum(y<x_val) / (1.0*n)

	pdf = -1*np.roll(cdf, -2) + 8*np.roll(cdf, -1) - 8*np.roll(cdf, 1) + 1*np.roll(cdf, 2)
	pdf /= 12*dx
	Normalize(pdf, dx)
	return x, pdf[2:res + 2]