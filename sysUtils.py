# pylint: disable=C,W
# basic system function utility file 
import time
import sys
import toml 
import os
import warnings as warn
CUPY_IMPORTED = True
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False  

def remaining(done, total, start):
	"""
	returns the estimated remaining time in the simulation

	:done: int, the number of steps completed
	:total: int, the total number of steps
	:start: float, the start time

	:return: float, the linear estimation of the remaining time in seconds
	"""
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	"""
	given a time in seconds return a tuple with the (hrs, mins, s)

	:T: float, time in seconds

	:return: (int, int, int), time in (hrs, min, s)
	"""
	r = T
	hrs = int(r)//(60*60)
	mins = int(r%(60*60))//(60)
	s = int(r%60)
	return (hrs, mins, s)

def repeat_print(string):
	"""
	repeat prints to the command line

	:string: string, the string to print
	"""
	sys.stdout.write('\r' +string)
	sys.stdout.flush()

def PrintTimeUpdate(done, total, time0):
	"""
	repeat prints the estimated remaining time in the simulation

	:done: int, the number of steps completed
	:total: int, the total number of steps
	:start: float, the start time
	"""
	repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(done, total, time0)))

def PrintCompletedTime(time0, task = ""):
	"""
	prints the time since time0

	:time0: float, the start time
	:task: string, the task run since the start time
	"""
	str_ = 'completed in %i hrs, %i mins, %i s\n' %hms(time.time()-time0)
	if len(task) > 0:
		print('\n' + task + " " + str_)
	else:
		print('\n' + str_)


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


def makeDataDir(simName, overwrite = False):
	"""
	creates a data directory for the given sim if necessary
	
	:simName: string, name of the data directory to create
	:overwrite: bool, overwrite existing directories if found
	"""

	### check for data directory
	if not(os.path.isdir("../Data")):
		os.mkdir("../Data")

	### check if I need to delete data directory 
	if os.path.isdir("../Data/" + simName) and overwrite:
		os.rmdir("../Data/" + simName)

	### check if sim data directory is made
	if not(os.path.isdir("../Data/" + simName)):
		os.mkdir("../Data/" + simName)


def getDataDir(simName):
	"""
	check if a data directory exists for this sim, create it if not,
	and return the directory string

	:simName: string, name of simulation data dir

	:returns: string, name of data directory to save simulation data
	"""
	makeDataDir(simName)

	return "../Data/" + simName + "/"


def ConvertStringDict(keys, types = [], **kwargs):
	'''
	converts a string dictionary to a dictionary of specified values

	:keys: array-like, keys of kwargs
	:types: array-like, types to convert the dictionary values to
	:kwargs: dictionary, the dictionary containing the strings

	:return: dictionary, kwargs with values converted to types 
	'''
	rval = {}

	for i in range(len(kwargs)):
		if len(keys) > i:
			key = keys[i]

			type_ = float
			if len(types) > i:
				type_ = types[i]

			if key in kwargs:
				if type_ == bool:
					rval[key] = kwargs[key] == 'True'
				else:
					rval[key] = type_(kwargs[key])

	return rval

def AddLines2Toml(dict2add, tomlName):
	'''
	appends a dictionary to a toml

	:dict2add: dictionary, data containing dictionary to add to toml
	:tomlName: string, file name
	'''
	data = toml.load(tomlName)
	data.update(dict2add)
	with open(tomlName, "w") as toml_file:
		toml.dump(data, toml_file)


def Dict2Toml(dict_, tomlName):
	'''
	writes a dictionary as a toml

	:dict_: dict, data
	:tomlName: string, name of toml to write
	'''
	with open(tomlName, "w") as toml_file:
		toml.dump(dict_, toml_file)