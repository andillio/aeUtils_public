# pylint: disable=C,W
# basic statistics and random function utility file
import numpy as np
CUPY_IMPORTED = True 
try:
	import cupy as cp 
	import cupyx as cpx
except ImportError:
	CUPY_IMPORTED = False
import gridUtils as gu
import warnings as warn
from scipy.optimize import nnls

# TODO: 
# - implement staggered imports
class State(object):

	def __init__(self):
		self.gpu_all = False
STATE = State()

# returns best fit parameters
# func(X,x,b)
def linearFit(X, y, gpu = False):
	"""
	gives the least squared best fit for y = Xb

	:X: array-like, the predictor data matrix, [params, data]
	:y: array-like, the response data matrix, [data] 
	:gpu: bool, try to run on gpu, default False

	:return: array-like, best fit values for b
	"""

	np_ = np 	
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp

	if X.shape[1] == len(y):
		XtX = np_.einsum("ij,kj->ik", X, X)
		cov = np_.linalg.inv(XtX)
		return np_.einsum("ij,jk,k->i",cov, X, y)

	XtX = np_.einsum("ji,jk->ik", X, X)
	cov = np_.linalg.inv(XtX)
	return np_.einsum("ij,kj,k->i",cov, X, y)


def fitLine(x,y, include_error = False):
	"""
	returns the best fit line y = mx + b

	:x: array-like, 1D predictor data matrix
	:y: array-like, 1D response data matrix

	:return: (xhat,yhat, mhat, bhat), range of predictor points,
		predicted data values, best fit slope, best fit intercept
	"""
	X_bar = np.mean(x)
	Y_bar = np.mean(y)

	x_hat = np.linspace(np.min(x), np.max(x), 100)

	mhat = np.sum( (x - X_bar) * (y - Y_bar) ) / np.sum( (x - X_bar)**2 )
	bhat = Y_bar - mhat*X_bar

	y_hat = mhat*x_hat + bhat


	if include_error:
		n = len(x)
		Y_hat = mhat*x + bhat
		m_se = np.sqrt(np.sum( (y-Y_hat)**2 ) / (n - 2.)) / np.sqrt(np.sum( (x - X_bar)**2 ))
		X2_BAR = np.mean(x**2)
		b_se = np.sqrt(np.sum( (y-Y_hat)**2 ) / (n - 2.)) * np.sqrt(1. / n + X2_BAR / np.sum( (x - X_bar)**2 ) )
		return x_hat, y_hat, mhat, bhat, m_se, b_se


	return x_hat, y_hat, mhat, bhat



# returns best fit parameters
# subject to constraint that all predictors are positive
# func(X,x,b)
def linearFit_beta_pos(X, y, gpu = False):
	"""
	gives the least squared best fit for y = Xb, 
	subject to constraint that all b > 0.
	Warning: gpu functionality not implemented

	:X: array-like, the predictor data matrix, [params, data]
	:y: array-like, the response data matrix, [data] 
	:gpu: bool, try to run on gpu, default False

	:return: array-like, best fit values for b
	"""
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		warn.warn("GPU functionality not implemented."+\
			" Moving data to cpu and back.")
		X, y = gu.gpu2cpu(X,y)

	beta, res = nnls(X.T, y)

	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		beta, X, y = gu.cpu2gpu(beta, X, y)

	return beta

# returns a random variable on a sphere
def randomOnSphere(n, gpu = False):
	"""
	returns random variables on a sphere of radius 1

	:n: int, number of random variables
	:gpu: boolean, run on gpu, default: false

	:returns: array-like, x,y,z coordinates [n,3]
	"""
	np_ = np 	
	if (gpu and CUPY_IMPORTED) or STATE.gpu_all:
		np_ = cp

	X1 = np_.random.uniform(0,1, size = n)
	X2 = np_.random.uniform(0,1, size = n)

	rval = np_.zeros((n,3))
	rval[:,0] = 1 - 2*X1
	rval[:,1] = np_.sqrt(1-rval[:,0]**2)*np_.cos(2*np_.pi*X2)
	rval[:,2] = np_.sqrt(1-rval[:,0]**2)*np_.sin(2*np_.pi*X2)

	return rval

