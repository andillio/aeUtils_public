# pylint: disable=C,W
import mathUtils as mu
import scipy.stats as stats
import statUtils as stu
import sysUtils as su
import time
import numpy as np 
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False

class State(object):

	def __init__(self):
		self.gpu_all = False
STATE = State()

G = 4.49e-12 # Newton's constant [kpc^3 M_solar^-1 Myr^-2]
kms2kpcMyr = 0.001022 # conversion between km/s to kpc/Myr
speed_of_light = 3e5 * kms2kpcMyr # speed of light [kpc Myr^-1]
local_dm_density = 1e7 # M_solar / kpc^3
au2kpc = 4.84814e-9 # conversion between an au and kpc
hbar = 1.757e-90 # Planck's constant [kpc^2 M_solar Myr^-1]
eV2SolarMass = 8.97e-67 

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

# func(X, f, f)
# Navarro-Frank-White profile
# https://arxiv.org/abs/astro-ph/9508025
def NFW(r, Rs, rho0 = 1.):
	"""
	Navarro-Frank-White galatic halo profile

	:r: array-like, radius coordinates
	:Rs: float, scale radius
	:rho0: float, characteristic density

	:return: array-like, the profile density at r (units of rho0)
	"""
	rval = rho0 / (r/Rs * (1 + r/Rs)**2)
	rval[r == 0] = 0
	return rval

def M_NFW(r, Rs, rho0 = 1., gpu = False):
	"""
	returns the mass enclosed for an NFW profile

	:r: array-like, radius coordinates
	:Rs: float, scale radius
	:rho0: float, characteristic density
	:gpu: bool, should run on gpu, default false

	:return: array-like, the enclosed mass at r
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all

	np_ = np
	if gpu_:
		np_ = cp

	rval = 4*np_.pi*rho0*Rs**3 * (np_.log((Rs + r)/Rs) - r/(Rs + r))
	return rval

def V_NFW(r, Rs, rho0 = 1., G = G):
	"""
	returns the potential for an NFW profile

	:r: array-like, radius coordinates
	:Rs: float, scale radius
	:rho0: float, characteristic density

	:return: array-like, the potential at r
	"""
	rval = -4*np.pi*G*rho0*Rs**3 / r * np.log(1. + (r/Rs))
	rval[r == 0] = -4*np.pi*G*rho0*Rs**2
	return rval


# func(X, f, f)
# https://arxiv.org/abs/1407.7762
# ^eqn 3
def Core(r, m22, rc):
	"""
	approximation to inner core of density profile in ULDM 

	:r: array-like, radius coordinates
	:m22: float, mass of uldm particle m/1e-22 eV
	:rc: float, core radius [kpc] 

	:return: array-like, the profile density at r [M_solar / kpc^3]
	"""
	factor = 0.019*m22**(-2) * (rc)**(-4) * 1e9
	rval = factor / (1 + 0.091*(r/rc)**2)**8
	return rval


# func(x,f,f,f,f,f,f,f)
def CoredNFW(r, Rs, rho0 = 1., m22 = 1., rc = None, Mvir = None, a = 1, zeta = 350.):
	"""
	cored radial density profile for halo

	:r: array-like, radius coordinate
	:Rs: float, scale radius
	:rho0: float, characteristic density, default 1
	:m22: float, mass of uldm particle m/1e-22 eV, default 1
	:rc: float, core radius [kpc], default None 
	:Mvir: float, halo mass [solar masses], default None
	:a: float, scale factor, default 1
	:zeta: float, zeta(z)/zeta(0) factor, ~350(180) for z = 0(z > 1), default 350

	:return: array-like, radial density profile
	"""
	if Mvir == None and rc == None:
		raise Exception(
			"Neither the virial mass or core radius have been specified.\n" +\
			"recommendation: specify a value for rc or Mvir."
			)

	if rc == None:
		rc = CoreRadius(m22, Mvir= Mvir, a = a, zeta = zeta)

	rho_NFW = NFW(r, Rs, rho0)
	rho_core = Core(r, m22, rc)
	if len(r[rho_core > rho_NFW]) == 0:
		return rho_NFW
	r_a = np.max(r[rho_core > rho_NFW])

	rho_total = rho_core
	rho_total[r > r_a] = rho_NFW[r > r_a]

	return rho_total



# func(f,f,f,f)
# https://arxiv.org/abs/1407.7762
# ^eqn 7
def CoreRadius(m22 = 1.0, Mvir = 1e9, a = 1, zeta = 1.):
	"""
	returns estimation of core radius given core-mass relation
	worked out by schive et al 2014

	:m22: float, mass of uldm particle m/1e-22 eV, default 1
	:Mvir: float, halo mass [solar masses], default 1e9
	:a: float, scale factor, default 1
	:zeta: float, zeta(z)/zeta(0) factor ~350(180) for z = 0(z > 1), default 1
	:return: float, core radius [kpc]
	"""
	return 1.6 * m22**-1 * a**.5 * zeta**(1./6) * (Mvir / 1e9)**(-1./3)

#func(f)
# https://arxiv.org/abs/1407.7762
# ^below eqn 4
def zeta(Omega_m):
	"""
	a factor on which things like soliton mass depend weakly

	:Omega_m: float, the matter energy density fraction
	:return: float, the value of zeta
	"""
	return (18*np.pi**2 + 82*(Omega_m-1) - 39*(Omega_m-1)**2)/Omega_m

# computes the radial potential given a density and radii
# func(x, x, f, b)
def radialPotential(r, rho, G = 4.49e-12, initial = 0, gpu = False):
	"""
	calculates the radial potential given a density and radius.
	for spherically symmetric potential

	:r: array-like, the radial positions
	:rho: array-like, the density values
	:G: float, Newton's constant, default 4.49e-12 [kpc^3 M_solar^-1 Myr^-2]
	:initial: float, the integral from inf to max(r), default 0
	:gpu: bool, try to run on gpu, default False

	:return: array-like, radial potential at r
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all

	np_ = np
	if gpu_:
		np_ = cp
	N = len(r)
	Mass_encl = mu.cumInt(rho*4*np_.pi*r**2, r, gpu = gpu_)
	integrand = np_.flip(G*Mass_encl/r**2)
	Phi = mu.cumInt(integrand,np_.flip(r), gpu = gpu_)

	return np.flip(Phi) - initial

# func(f)
# computes hbar_
def h_tilde(m22 = 1.):
	"""
	computes hbar_tilde for given mass in astrophysical units

	:m22: float, uldm mass m / 1e-22 eV/c^2, default 1

	:return: float, hbar_tilde [kpc^2 / Myr]
	"""
	return .01959 / m22

# func(i, f, i, f, b)
# returns the kinetic operator matrix
def kineticOperator(N, hbar_ = 1., pts = 5, dx = 1., periodic = True, gpu = False):
	"""
	returns the kinetic operator T / hbar_ = -hbar_ * nabla^2 / 2

	:N: int, grid resolution
	:hbar_: float, plancks reduced constant over the particle mass
	:pts: int, number of points in stencil, default 3
	:dx: float, differential element
	:periodic: bool, boundary conditions, default True
	:gpu: bool, try to run on gpu, default False

	:return: array-like, N by N kinetic operator matrix [time^-1] 
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	return -.5*mu.stencil(N, pts = pts, order = 2, dx = dx, periodic=periodic, gpu = gpu_)*hbar_

# func(x, x, i, f, i, f, b)
def Hamiltonian_radial(r, Phi_r, l = 0, hbar_ = 1., pts = 3, dx = None, gpu = False):
	"""
	gives the radial Hamiltonian for the given radial potential

	:r: array like, the radial coordinates, should be (0,r_max) format
	:Phi_r: array like, the radial potential at r
	:l: int, the angular momentum quantum number, default 0
	:hbar_: float, hbar/m, used for kinetic term, default 1
	:pts: int, number of pts in differentiation stencil, default 5
	:dx: float, differential element, default use r to get dx
	:gpu: bool, try to run on gpu, default False

	:return: array-like, N by N matrix, radial Hamiltonian
	"""
	np_ = np 
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	if gpu_:
		np_ = cp 

	N = len(r)

	if dx == None:
		dx = np.abs(r[1] - r[0])

	T_ = kineticOperator(N, hbar_, pts = pts, dx = dx, periodic=False, gpu = gpu_)
	V_ = Phi_r/hbar_ + .5*hbar_*l*(l+1)/r**2
	V_ = np_.diag(V_)
	return T_ + V_ 


# func(x, x, i, f, i, f, b)
def Hamiltonian_radial_tridiag(r, Phi_r, l = 0, hbar_ = 1., dx = None, gpu = False):
	"""
	gives the radial Hamiltonian / hbar_ for a given radial potential
	calculated using a three point stencil for the kinetic term
	returns the diagonal and off diagonal portions of the Hamiltonian

	:r: array like, the radial coordinates, should be (0,r_max) format
	:Phi_r: array like, the radial potential at r
	:l: int, the angular momentum quantum number, default 0
	:hbar_: float, hbar/m, used for kinetic term, default 1
	:dx: float, differential element, default use r to get dx
	:gpu: bool, try to run on gpu, default False

	:return: (array-like, array-like), (diagonal, offdiagonal),
		diagonal - [N] array, representing diagonal of Hamiltonian,
		offdiagonal - [N-1] array, representing off diagonal values 
			in Hamiltonian
	"""
	np_ = np 
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	if gpu_:
		np_ = cp 

	N = len(r)

	if dx == None:
		dx = np_.abs(r[1] - r[0])

	diag = Phi_r/hbar_ + .5*hbar_*l*(l+1)/r**2 + -.5*hbar_*-2 / dx**2
	offdiag = np_.ones(N-1)*-.5*hbar_*1./dx**2
	return diag, offdiag


def powerspectrum(rho, dx=1., dV_norm = False):
	"""
	finds the power spectrum of the given density

	:rho: array-like
	:dx: float, pixel size, dafault 1

	:returns: (array-like, array-like), (kvals, PS) \n
	kvals - k values power spectrum is evaluated at \n
	PS - power spectrum values
	"""
	return mu.powerspectrum(rho, dx, dV_norm)


def radialProfile(R, rho):
	"""
	finds the radial profile given the density

	:R: array-like, [N^D] positions, centered at 0
	:rho: array-like, [N^D] density matrix

	:returns: (array-like, array-like), (r, rho_profile) \n 
	r - [N/2] position \n
	rho_profile - [N/2] radial density profile
	"""
	N = rho.shape[0]
	D = len(rho.shape)

	dr = np.max(R) / (N-1) / np.sqrt(D)
	rbins = np.arange(0, N//2+1, 1.)*dr * 2

	rvals = 0.5 * (rbins[1:] + rbins[:-1])
	rho_profile, _, _ = stats.binned_statistic(R.flatten(), rho.flatten(),
	    statistic = "mean", bins = rbins)

	return rvals, rho_profile


# generate random positions and velocities in plummer sphere
def PlummerSphere(M, R, n, G = G, gpu = False, L = 0):
	"""
	generate positions and velcoities in a plummer sphere

	:M: float, total mass
	:R: float, scale radius
	:n: int, number of particles
	:G: float, graviational constant, default: [kpc^3 M_solar^-1 Myr^-2]
	:gpu: bool, run on gpu, default: false
	:L: float, box size, default: infinite

	:returns: (array-like, array-like), (r, v), ([n,3], [n,3]) \n
		r - x, y, z positions of particles \n 
		v - vx, vy, vz velocities of particles 
	"""
	np_ = np 
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	if gpu_:
		np_ = cp 

	r = np_.zeros(n)
	r_tilde = np_.zeros((n,3))
	if L == 0:
		X1 = np_.random.uniform(0,1,n)
		r = 1. / np_.sqrt(X1**(-2./3.) - 1.)
		r_tilde = stu.randomOnSphere(n, gpu = gpu_)*r[:,np_.newaxis]
	else:
		found = 0
		time0 = time.time()
		while(found < n):
			X1 = np_.random.uniform(0,1)
			r_ = 1. / np_.sqrt(X1**(-2./3.) - 1.)
			if (r_ < L/2.):
				r[found] = r_
				r_tilde[found] = stu.randomOnSphere(1, gpu = gpu_)*r_
				found += 1
				su.PrintTimeUpdate(found, n, time0)



	Ve = np_.sqrt(2)*(1+r**2)**(-1./4)

	q = np_.zeros(n)
	found = 0
	time0 = time.time()
	while(found < n):
		X1 = np_.random.uniform(0,1)
		X2 = np_.random.uniform(0,1)
		q_ = X1 
		accept = .1*X2 < q_**2 * (1 - q_**2)**(7./2.)

		if accept:
			q[found] = q_
			found += 1
			su.PrintTimeUpdate(found, n, time0)


	v_tilde = stu.randomOnSphere(n, gpu = gpu_)*q[:,np_.newaxis]*Ve[:,np_.newaxis]

	return r_tilde*R, v_tilde*np_.sqrt(M*G/R)


# generate random positions and velocities in plummer sphere
def PlummerSphereExt(M, R, n, V_ext, G = G, gpu = False):
	"""
	generate positions and velcoities in a plummer sphere

	:M: float, total mass
	:R: float, scale radius
	:n: int, number of particles
	:V_ext: function, function giving potential enclosed mass at given radius
	:G: float, graviational constant, default: [kpc^3 M_solar^-1 Myr^-2]
	:gpu: bool, run on gpu, default: false

	:returns: (array-like, array-like), (r, v), ([n,3], [n,3]) \n
		r - x, y, z positions of particles \n 
		v - vx, vy, vz velocities of particles 
	"""
	np_ = np 
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	if gpu_:
		np_ = cp 

	X1 = np_.random.uniform(0,1,n)
	r = R / np_.sqrt(X1**(-2./3.) - 1.)
	r_tilde = stu.randomOnSphere(n, gpu = gpu_)*r[:,np_.newaxis]

	U_internal = -G*M/R / np_.sqrt(1 + (r/R)**2) 
	Ve = np_.sqrt(-2* (U_internal + V_ext(r)) )

	q = np_.zeros(n)
	found = 0
	time0 = time.time()
	while(found < n):
		X1 = np_.random.uniform(0,1)
		X2 = np_.random.uniform(0,1)
		q_ = X1 
		accept = .1*X2 < q_**2 * (1 - q_**2)**(7./2.)

		if accept:
			q[found] = q_
			found += 1
			su.PrintTimeUpdate(found, n, time0)


	v_tilde = stu.randomOnSphere(n, gpu = gpu_)*q[:,np_.newaxis]*Ve[:,np_.newaxis]

	return r_tilde, v_tilde


def compute_phi(rho, L, C = 4*np.pi*G, padded = False, gpu = False):
	"""
	calculates the potential for a given density

	:rho: array-like, density fields 
	:L: float, box length
	:C: float, Poisson's constant, default gravity in astro units
	:padded: bool, pad the density with 0s, default false
	:gpu: bool, run on the gpu, default false

	:return: array-like, potential
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	N = len(rho)
	D = len(rho.shape)
	rval = rho

	if padded:
		rval = pad_density(rval, gpu = gpu_)

	rval = mu.GetFFt(rval, gpu = gpu_)

	K = 0
	if padded:
		K = get_K(N*2, L*2, D, gpu = gpu_)
	else:
		K = get_K(N, L, D, gpu = gpu_) 
	rval = -1*C*rval / K
	if D == 3:
		rval[0,0,0] = 0.0
	elif D ==2:
		rval[0,0] = 0.
	elif D == 1:
		rval[0] = 0

	rval = mu.GetFFt(rval, Forward = False, gpu = gpu_)
	rval = rval.real

	if padded:
		if D == 3:
			rval = rval[:N,:N,:N]
		elif D == 2:
			rval = rval[:N,:N]
		elif D == 1:
			rval = rval[:N]

	return rval.real


def pad_density(rho, gpu = False):
	"""
	puts rho in the bottom corner of zeros
	
	:rho: array-like, density
	:gpu: bool, run on the gpu

	:returns: array-like, zeros with bottom corner equal to rho
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp
	D = len(rho.shape)
	N = len(rho)

	zeros = 0.
	if D == 1:
		zeros = np_.zeros(N*2)
		zeros[:N] = rho
	if D == 2:
		zeros = np_.zeros((N*2, N*2))
		zeros[:N, :N] = rho
	if D == 3:
		zeros = np_.zeros((N*2, N*2, N*2))
		zeros[0:N, 0:N, 0:N] = rho

	return zeros


def get_K(N, L, D, gpu = False):
	"""
	calculate the spectral grid

	:N: int, the grid resolution
	:L: float, box length
	:D: int, number of spatial dimensions
	:gpu: bool, run on the gpu, default False

	:return: array-like, [N^D], spectral grid
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp

	dx = L/N
	kx = 2*np_.pi*np_.fft.fftfreq(N, d = dx)
	ones = np_.ones(N)

	if D == 1:
		return kx**2
	if D == 2:
		K = np_.einsum("i,j->ij", kx**2, ones)
		K += np_.einsum("i,j->ij", ones, kx**2)
		return K
	if D == 3:
		K = np_.einsum("i,j,k->ijk", kx**2, ones, ones)
		K += np_.einsum("i,j,k->ijk", ones, kx**2, ones)
		K += np_.einsum("i,j,k->ijk", ones, ones, kx**2)
		return K


def PlummerSphereDensity(r, R, M, gpu = False):
	"""
	gives the density of the plummer sphere evaluated at r

	:r: array-like, position at which to evaluate density
	:R: float, scale radius of the Plummer sphere
	:M: float, Plummer sphere mass
	"""
	gpu_ = (gpu and CUPY_IMPORTED) or STATE.gpu_all
	np_ = np
	if gpu_:
		np_ = cp
	scale = 3*M/4./np_.pi/R**3
	pos = (1. + (r/R)**2 )**(-5./2)
	return scale*pos


def calcKappa(delta_rho_along_path, affine):
	"""
	gives the value of microlensing integrated along the line of sight

	:delta_rho_along_path: array-like, 1-D, density along long of sight, [M_solar / kpc^-3]
	:affine: array-like, 1-D, affine parameter between source and observer,
		 max(affine) = pathlength, [kpc]

	:return: float, microlensing value
	"""
	pathlength = np.max(affine)
	dl = pathlength / len(affine)
	factor = 4*np.pi*G / speed_of_light**2

	kappa = np.sum(delta_rho_along_path*affine*(pathlength - affine)*dl/pathlength)
	return kappa * factor
