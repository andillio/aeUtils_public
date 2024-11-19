import numpy as np 
import scipy.fftpack as sp 

# x has to be defined from -L/2, L/2 or this won't work
def K_H(x, x_, u, hbar_, sig_x, L, full = False):
	X, X_, U = np.meshgrid(x, x_, u) # x is on axis 1, x_ is on axis 0, u is on axis 2

	X_[X_ - X > L/2] -= L # shift x_ so it is centered at a given x value
	X_[X_ - X < -L/2] += L

	arg = (-(X - X_)**2)/(4.*sig_x**2) + 0j
	arg -= U*X_*(1.j/hbar_)
	denom = np.sqrt(2.*np.pi*hbar_)
	denom *= (2.*np.pi*sig_x**2)**(.25)
	if full:
		return np.exp(arg)/denom, X, X_, U
	return np.exp(arg)/denom


def f_H(psi, K, dx):
	return (np.abs(psi_H(psi, K, dx))**2).transpose()


def psi_H(psi, K, dx):
	integrand = psi[:, None, None]*K
	return dx*integrand.sum(axis = 0)
#	return dx*sp.simps(integrand, axis = 0)


def HusimiFunc(psi, hbar_ = 1., L = 1., K = None):
	"""
	returns the Husimi function of psi

	:psi: array-like, complex 1-d field
	:hbar_: float, hbar / field_mass, default 1
	:L: float, length of box, default 1
	:K: array-like, husimi kernel, default None
	"""
	N = len(psi)
	dx = L/N
	if K is None:
		x = dx*(.5+np.arange(-N//2, N//2))
		x_ = x.copy()
		kx = 2*np.pi*sp.fftfreq(N, d = dx)
		u = hbar_*sp.fftshift(kx)
		K = K_H(x, x_, u, hbar_, 5*dx, L)
	return f_H(psi, K, dx)
