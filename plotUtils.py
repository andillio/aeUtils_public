# pylint: disable=C,W
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patches as patches
import qUtils as QU
# TODO: 
# - add look_at_this_func
# - add configure_kwargs private function
class FigObj(object):

	# func(i,i,f,f)
	def __init__(self, ncols = 1, nrows = 1, width = None, height = None):
		"""
		constructor for fig object
		this object handles simple plotting

		:ncols: integer, number of columns in figure, default 1
		:nrows: integer, number of rows in figure, default 1
		:width: float, width of figure, default 6*ncols
		:height: float, height of figure, default 6*nrows
		
		:return: FigObj, reference to constructed object
		""" 
		self.orientPlot()

		if width == None or height == None:
			self.fig, self.axs = plt.subplots(nrows, ncols, figsize = (6*ncols,6*nrows))
		else:
			self.fig, self.axs = plt.subplots(nrows, ncols, figsize = (width,height))	

		# dimensionality of axs
		self.dim = 0 # dimensionality of axes object
		if (ncols == 1 and nrows > 1) or (nrows == 1 and ncols > 1):
			self.dim = 1 
		elif ncols > 1 and nrows > 1:
			self.dim = 2

		self.axs_fl = np.array([self.axs]) 
		if self.dim >= 1:
			self.axs_fl = np.ndarray.flatten(self.axs)

		# a dictionary to track all the image objects in the figure
		self.ims = {}

		# number of axes that have been used so far
		self.axsUsed = 0
		self.axsTotal = ncols*nrows

		self.name_ = None 
		self.ls = '-'
		self.mk = ''
		self.alpha = 1.
		self.color = 'k'
		self.label = ''


	def configure_kwargs(self, **kwargs):
		"""
		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label
		:color: string, line color
		"""
		if "name" in kwargs:
			name_ = kwargs["name"]
		
		ls = '--'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']


	def AddPlot(self,*args,**kwargs):
		"""
		adds a plot to the current FigObj if there is an available plot

		args
		:func(X): will plot imshow(X)
		:func(x,x): will plot(x,x)

		kwargs
		:name: string, name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:label: string, legend label

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		# handle no remaining axes to use
		if self.axsUsed >= self.axsTotal:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"recommendation: initialize the FigObj with more subplots"
				)
		
		# handle given a single argument (eg y)
		if len(args) == 1:
			# handle imshow and plot just given y
			if len(np.shape(args[0])) == 2:
				return self.AddImshow(*args,**kwargs)
		# handle given 2 arguments (eg x,y)
		elif len(args) == 2:
			# handle standard plot
			if len(np.shape(args[1])) == 1:
				return self.AddStdPlot(*args,**kwargs)


	def AddImshow(self,*args,**kwargs):
		"""
		adds an imshow to the current FigObj if there is an available plot

		args
		:func(X): will plot imshow(X)

		kwargs
		:name: name of added axesImage in figObj ims attribute

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		if self.axsUsed >= self.axsTotal:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		Y = None
		if len(args) == 1:
			Y = args[0]
		else:
			Y = args[1]
		index_ = self.axsUsed

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		ax = self.axs_fl[index_]

		im = ax.imshow(Y, interpolation = "none", alpha = alpha,
			origin = "lower")

		self.ims[name_] = im 

		self.axsUsed += 1
		return ax, im


	def AddStdPlot(self, *args,**kwargs):
		"""
		adds a plot to the current FigObj if there is an available plot

		args
		:func(x, x): will plot(x,x)

		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		if self.axsUsed >= self.axsTotal:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"recommendation: initialize the FigObj with more subplots"
				)
		x = args[0]

		y = None
		if len(args) == 1:
			y = args[0]
		else:
			y = args[1]
		index_ = self.axsUsed

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		ax = self.axs_fl[index_]

		ls = '-'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1.
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		if "color" in kwargs:
			color = kwargs['color']
			im, = ax.plot(x,y, ls = ls, marker = mk, alpha= alpha, color = color)
		else:
			im, = ax.plot(x,y, ls = ls, marker = mk, alpha= alpha)


		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		set_lims = True
		if "set_lims" in kwargs:
			set_lims = kwargs["set_lims"]

		if set_lims:
			if np.min(y) > 0:
				ax.set_ylim(0, np.max(y))
			else:
				ax.set_ylim(np.min(y), np.max(y)*1.05)
			if np.min(x) > 0:
				ax.set_xlim(0, np.max(x))
			else:
				ax.set_xlim(np.min(x), np.max(x))

		self.ims[name_] = im 

		self.axsUsed += 1
		return ax, im


	def show(self):
		"""
		runs plt.show()
		"""
		plt.show()

	def legend(self, ax = None, loc = ""):
		"""
		runs ax.legend() 

		:ax: axesObj, ax to add legend to, most recently added plot by default
		"""
		ax = self.private_GetAxis(ax)
		if len(loc) > 0:
			ax.legend(loc = loc)
		else:
			ax.legend()

	# func(i)
	def orientPlot(self, fontSize = 22):
		"""
		sets figure parameters to be publication ready

		:fontsize: int, the fontsize used in the figures
		"""
		plt.rc("font", size=fontSize)
		plt.rc("text", usetex=True)
		# plt.figure(figsize=(6,6))
		# fig,ax = plt.subplots(figsize=(6,6))
		plt.rcParams["axes.linewidth"]  = 1.5
		plt.rcParams["xtick.major.size"]  = 8
		plt.rcParams["xtick.minor.size"]  = 3
		plt.rcParams["ytick.major.size"]  = 8
		plt.rcParams["ytick.minor.size"]  = 3
		plt.rcParams["xtick.direction"]  = "in"
		plt.rcParams["ytick.direction"]  = "in"
		plt.rcParams["legend.frameon"] = 'False'


	# funx(x, x, axsObj)
	def SetLogLog(self, x = [], y = [], ax = None):
		"""
		set axes to be log-log scale

		:x: array-like, the values used for the x axis
		:y: array-like, the values used for the y axis
		:ax: axsObj, the axis who axes are being changed
		"""
		ax = self.private_GetAxis(ax)

		xlow, xhigh = ax.get_xlim()
		ylow, yhigh = ax.get_ylim()

		# make sure we can set loglog scale
		if xlow <= 0 and (np.size(x) == 0 or np.min(x) < 0):
			raise Exception(
				"x lower limit is current <= 0.\n" +\
				"recommendation: try specifying x range with SetLogLog(x = x),\n" +\
				"or try SetLogY method instead"
				)
		if ylow <= 0 and (np.size(y) == 0 or np.min(y) < 0):
			raise Exception(
				"y lower limit is current <= 0.\n" +\
				"recommendation: try specifying y range with SetLogLog(y = y),\n" +\
				"or try SetLogX method instead"
				)

		if np.size(x) > 1:
			xlow = np.min(x)

			if xlow <= 0:
				xlow = x[1]

			xhigh = np.max(x)
		if np.size(y) > 1:
			ylow = np.min(y)

			if ylow <= 0:
				ylow = x[1]

			yhigh = np.max(y)

		ax.set_xlim(xlow, xhigh)
		ax.set_ylim(ylow, yhigh)

		ax.set_yscale("log")
		ax.set_xscale("log")


	def SetXLog(self, x = [], ax = None):
		"""
		set x-axis to be log-log scale

		:x: array-like, the values used for the x axis
		:ax: axsObj, the axis who axes are being changed
		"""
		ax = self.private_GetAxis(ax)

		xlow, xhigh = ax.get_xlim()

		# make sure we can set loglog scale
		if xlow <= 0 and (np.size(x) == 0 or np.min(x) < 0):
			raise Exception(
				"x lower limit is current <= 0.\n" +\
				"recommendation: try specifying x range with SetLogLog(x = x),\n" +\
				"or try SetLogY method instead"
				)

		if np.size(x) > 1:
			xlow = np.min(x)

			if xlow <= 0:
				xlow = x[1]

			xhigh = np.max(x)

		ax.set_xlim(xlow, xhigh)
		ax.set_xscale("log")

	def SetYLog(self, y = [], ax = None):
		"""
		set y-axis to be log-log scale

		:y: array-like, the values used for the y axis
		:ax: axsObj, the axis who axes are being changed
		"""
		ax = self.private_GetAxis(ax)

		xlow, xhigh = ax.get_ylim()

		# make sure we can set loglog scale
		if xlow <= 0 and (np.size(y) == 0 or np.min(y) < 0):
			raise Exception(
				"y lower limit is current <= 0.\n" +\
				"recommendation: try specifying y range with SetLogLog(y = y),\n" +\
				"or try SetLogY method instead"
				)

		if np.size(y) > 1:
			xlow = np.min(y)

			if xlow <= 0:
				xlow = y[1]

			xhigh = np.max(y)

		ax.set_ylim(xlow, xhigh)
		ax.set_yscale("log")

	# func(x,x)
	def AddLine(self, x, y, ax = None, **kwargs):
		"""
		adds data to existing plot

		args
		:x: array-like, data x values
		:y: array-like, data y values
		:ax: axesObject, axes to add the line to, default most recently used axes

		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		if self.axsUsed == 0:
			return self.AddStdPlot(x,y, **kwargs)
		if ax == None:
			ax = self.axs_fl[self.axsUsed - 1]

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]
		
		ls = '-'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		color = 'b'
		if 'color' in kwargs:
			color = kwargs['color']

		alpha = 1.
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		im, = ax.plot(x,y, ls = ls, marker = mk, alpha=alpha, color = color)

		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		set_lims = False
		if "set_lims" in kwargs:
			set_lims = kwargs["set_lims"]

		if set_lims:
			if np.min(y) > 0:
				ax.set_ylim(0, np.max(y))
			else:
				ax.set_ylim(np.min(y), np.max(y)*1.05)
			if np.min(x) > 0:
				ax.set_xlim(0, np.max(x))
			else:
				ax.set_xlim(np.min(x), np.max(x))

		self.ims[name_] = im 

		return ax, im

	# func(f,x)
	def AddVertLine(self, x, ylims = [], ax = None, **kwargs):
		"""
		adds a vertical line to an existing plot

		args
		:x: float, x value for the line
		:ax: axesObject, axes to add the line to, default most recently used axes

		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label
		:color: string, line color

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""

		if self.axsUsed == 0:
			if len(ylims) > 0:
				return self.AddStdPlot(np.array([x,x]),ylims,**kwargs)
			raise Exception(
				"No plots have been added to this figure.\n" +\
				"recommendation: add a plot before calling this function"
				)
		if ax == None:
			ax = self.axs_fl[self.axsUsed - 1]

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]
		
		ls = '--'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1.
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']

		y = np.array(ax.get_ylim())
		if len(ylims) == 2:
			y = ylims

		im, = ax.plot([x,x],y, ls = ls, marker = mk, alpha = alpha, color = color)

		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		self.ims[name_] = im 

		return ax, im

	# func(f,x)
	def AddHorLine(self, y, xlims = [], ax = None, **kwargs):
		"""
		adds a horizontal line to an existing plot

		args
		:y: float, y value for the line
		:ax: axesObject, axes to add the line to, default most recently used axes

		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label
		:color: string, line color

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""

		if self.axsUsed == 0:
			if len(xlims) > 0:
				return self.AddStdPlot(xlims, np.array([y,y]),**kwargs)
			raise Exception(
				"No plots have been added to this figure.\n" +\
				"recommendation: add a plot before calling this function"
				)
		if ax == None:
			ax = self.axs_fl[self.axsUsed - 1]

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]
		
		ls = '--'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']

		x = np.array(ax.get_xlim())
		if len(xlims) == 2:
			x = xlims

		im, = ax.plot(x,[y,y], ls = ls, marker = mk, alpha = alpha, color = color)

		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		self.ims[name_] = im 

		return ax, im


	def AddCircle(self, R, res = 100, ax = None, **kwargs):
		"""
		draws a circle

		:R: float, radius of circle
		:res: int, resolution of line
		:ax: axesObject, axes to add the line to, default most recently used axes

		kwargs
		:name: name of added axesImage in figObj ims attribute
		:set_lims: bool, should the axes limits be changed
		:ls: string, linestyle (eg '-' solid, '--' dashed, '-.' dashdot, ':' dotted, etc.)
		:mk: string, marker style (eg 'x', 'v' triangle down, 'o' circle, '2' tri left, etc.)
		:label: string, legend label
		:color: string, line color

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]
		
		ls = '--'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']

		dx = 2*np.pi/res 
		theta = np.arange(res)*dx
		R = np.ones(res)*R
		x = np.cos(theta)*R
		y = np.sin(theta)*R

		if self.axsUsed == 0:
			return self.AddStdPlot(x, y,**kwargs)
		if ax == None:
			ax = self.axs_fl[self.axsUsed - 1]

		im, = ax.plot(x,y, ls = ls, marker = mk, alpha = alpha, color = color)

		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		self.ims[name_] = im 

		return ax, im




	def AddDens2d(self, x, rho, y = [], **kwargs):
		"""
		adds a 2d plot of the density rho

		:x: array-like, bounds of the density plot
		:rho: array-like, density to be plotted (should be two dimensional)
		:y: array-like, y-axis bounds if different from x-axis

		kwargs
		:log: bool, take log of rho
		:units: string, axes units (eg 'astro', 'natural')

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		if self.axsUsed >= self.axsTotal:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		rho_ = rho.copy()
		log = False

		if "log" in kwargs:
			log = kwargs["log"]
		if log:
			rho_ = np.log10(rho_)

		ax, im = self.AddImshow(rho_)

		units_str = "astro"
		if "units" in kwargs:
			units_str = kwargs["units"]

		if units_str == "astro":
			self.SetLabels(r"$x \, [\mathrm{kpc}]$", r"$y \, [\mathrm{kpc}]$")
		elif units_str == "natural":
			self.SetLabels(r"$x \, [L]$", r"$y \, [L]$")

		xlow = np.min(x)
		xhigh = np.max(x)
		ylow = xlow
		yhigh = xhigh

		if len(y) > 0:
			ylow = np.min(y)
			yhigh = np.max(y)

		self.SetExtent((xlow, xhigh, ylow, yhigh), im)

		self.ims[name_] = im 

		return ax, im


	def AddHist(self, x, nBins = 100, bins = [], density = True,
	 weights = [], ax = None, **kwargs):
		"""
		adds a new histogram plot

		:x: array-like, array of values to be binned
		:nBins: int, number of bins, default 100
		:bins: array-like, explicit bin edges, default use auto
		:density: bool, normalize histogram, default True
		:weights: array-like, weights of each point

		kwargs
		:log: bool, take log of counts

		"""
		if self.axsUsed >= self.axsTotal and ax == None:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		weights_ = np.ones(np.shape(x))
		if len(weights) > 0:
			weights_ = weights

		n, edges = np.zeros(nBins), np.zeros(nBins + 1)
		if len(bins) > 0:
			n, edges = \
				np.histogram(x, bins = bins, density = density, weights=weights_)
		else:
			n, edges = \
				np.histogram(x, bins = nBins, density = density, weights=weights_)
		
		log = False
		if "log" in kwargs:
			log = kwargs["log"]
		
		index_ = self.axsUsed

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		if ax == None:
			self.axsUsed += 1
			ax = self.axs_fl[index_]

		ls = '-'
		if 'ls' in kwargs:
			ls = kwargs['ls']

		mk = ''
		if 'mk' in kwargs:
			mk = kwargs['mk']

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']


		x_ = .5*(edges[0:len(edges)-1] + edges[1:])

		im, = ax.plot(x_, n, color = color, alpha = alpha, ls = ls, marker = mk)

		if 'label' in kwargs:
			im.set_label(kwargs['label'])

		self.ims[name_] = im 

		return ax, im


	def AddHist2d(self, x,y, nBins=100, bins = [], density = True, 
		weights = [], ax = None, **kwargs):
		"""
		adds a new histogram plot

		:x: array-like, array of x values to be binned
		:y: array-like, array of y values to be binned
		:nBins: int, number of bins, default 100
		:bins: array-like, explicit bin edges, default use auto
		:density: bool, normalize histogram, default True
		:weights: array-like, weights of each point

		kwargs
		:log: bool, take log of counts

		"""
		if self.axsUsed >= self.axsTotal and ax == None:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		weights_ = np.ones(np.shape(x))
		if len(weights) > 0:
			weights_ = weights

		n, xedges, yedges = np.zeros((nBins, nBins)), np.zeros(nBins + 1), np.zeros(nBins + 1)

		if len(bins) > 0:
			n, xedges, yedges = \
				np.histogram2d(x,y, bins = bins, density = density, weights=weights_)
		else:
			n, xedges, yedges = \
				np.histogram2d(x,y, bins = nBins, density = density, weights=weights_)

		log = False
		if "log" in kwargs:
			log = kwargs["log"]
		
		index_ = self.axsUsed

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		if ax == None:
			self.axsUsed += 1
			ax = self.axs_fl[index_]

		alpha = 1
		if "alpha" in kwargs:
			alpha = kwargs['alpha']

		if log:
			n = np.log(n)

		im = ax.imshow(n.T, interpolation = "none", alpha = alpha, origin = "lower", aspect = "equal",
			extent = [np.min(xedges),np.max(xedges),np.min(yedges),np.max(yedges)])

		self.ims[name_] = im 
		# ax.set_aspect(1.*ax.get_data_ratio())
		# ax.set_aspect(1.)

		return ax, im


	# func(s,s,o)
	# set plot labels
	def SetLabels(self, xlabel, ylabel, ax = None):
		"""
		sets the x and y labels on the given axis

		:xlabel: string, x-axis label
		:ylabel: string, y-axis label
		:ax: axesObj, the axis to be set, default last axis used
		"""
		ax = self.private_GetAxis(ax)

		self.SetXLabel(xlabel, ax)
		self.SetYLabel(ylabel, ax)


	def RemoveYLabels(self, ax = None):
		'''
		removes the y labels from a plot
		'''
		ax = self.private_GetAxis(ax)
		ax.set_yticklabels([])
		self.SetYLabel("")

	def RemoveXLabels(self, ax = None):
		'''
		removes the x labels from a plot
		'''
		ax = self.private_GetAxis(ax)
		ax.set_xticklabels([])
		self.SetXLabel("")


	def SetXLabel(self, xlabel, ax = None, units = ""):
		"""
		sets the x label on the given axis

		:xlabel: string, x-axis label
		:ax: axesObj, the axis to be set, default: last axis used
		:units: string, units axis values, default:empty
		"""
		ax = self.private_GetAxis(ax)

		if len(units) == 0:
			ax.set_xlabel(xlabel)
		else: 
			ax.set_xlabel(xlabel + r"$\, [\mathrm{%s}]$"%(units))

	def SetYLabel(self, ylabel, ax = None, units = ""):
		"""
		sets the y label on the given axis

		:ylabel: string, y-axis label
		:ax: axesObj, the axis to be set, default last axis used
		:units: string, units axis values, default:empty
		"""
		ax = self.private_GetAxis(ax)

		if len(units) == 0:
			ax.set_ylabel(ylabel)
		else: 
			ax.set_ylabel(ylabel + r"$\, [\mathrm{%s}]$"%(units))

	def IncreaseYLim(self, upper_factor = 1.05, lower_factor = 1., ax = None):
		"""
		changes the y lims by the factors given, 

		:upper_factor: float, new_upper = old*upper_factor, default 1.05
		:lower_factor: float, new_lower = old*lower_factor, default 1
		:ax: axesObj, the axis to be set, default last axis used
		"""
		ax = self.private_GetAxis(ax)

		ylim = ax.get_ylim()
		ax.set_ylim(( ylim[0]*lower_factor, ylim[1]*upper_factor ))


	def SetTitle(self, title, ax = None):
		"""
		sets plot title

		:title: string, the title of the plot
		"""
		ax = self.private_GetAxis(ax)

		ax.set_title(title)


	def private_GetAxis(self, ax):
		"""
		returns the axis to be acted upon, if ax is None then returns most 
		recent axis created

		:ax: axis object

		:returns: axis object
		"""
		if self.axsUsed == 0:
			raise Exception(
				"No plots have been added to this figure.\n" +\
				"recommendation: add a plot before calling this function"
				)
		if ax == None:
			ax = self.axs_fl[self.axsUsed - 1]

		return ax 	

	# set_xlim
	# func(f,f,o)
	def SetXLim(self, xLow = None, xHigh = None, ax = None):
		"""
		sets x limits of axes object

		:xLow: float, lower limit, default: leave limit unchanged
		:xHigh: float, upper limit, default: leave limit unchanged
		:ax: axes object, the axes object to be set, default: last one used
		"""
		ax = self.private_GetAxis(ax)

		xlow, xhigh = ax.get_xlim()
		if xLow == None:
			xLow = xlow
		if xHigh == None:
			xHigh = xhigh

		ax.set_xlim(xLow, xHigh)

	# set_ylim
	# func(f,f,o)
	def SetYLim(self, yLow = None, yHigh = None, ax = None):
		"""
		sets y limits of axes object

		:yLow: float, lower limit, default: leave limit unchanged
		:yHigh: float, upper limit, default: leave limit unchanged
		:ax: axes object, the axes object to be set, default: last one used
		"""
		ax = self.private_GetAxis(ax)

		ylow, yhigh = ax.get_ylim()
		if yLow == None:
			yLow = ylow
		if yHigh == None:
			yHigh = yhigh

		ax.set_ylim(yLow, yHigh)


	def AddText(self, text, loc = 'upper right', pos = [], ax = None,
		bbox_off = False):
		'''
		adds text to a plot

		:text: string, text to add to plot
		:loc: string, location text, default 'upper right'
		:pos: array-like, (xPos, yPos)
		:ax: ax-object

		return imObject
		'''
		ax = self.private_GetAxis(ax)

		xPos = .5
		if "left" in loc:
			xPos = .3
		if "right" in loc:
			xPos = .8
		if len(pos) > 0:
			xPos = pos[0]

		yPos = .5
		if "upper" in loc:
			yPos = .9
		if "lower" in loc:
			yPos = .3
		if len(pos) > 0:
			yPos = pos[1]


		if not(bbox_off):
			imText = ax.text(xPos,yPos, text,
			  ha='center', va='center', transform= ax.transAxes,
			  bbox = {'facecolor': 'white', 'pad': 5})
		else:
			imText = ax.text(xPos,yPos, text,
			  ha='center', va='center', transform= ax.transAxes)

		return imText

	def AddTextRightLabel(self, text, ax = None):
		'''
		adds text label to the right of a plot

		:text: string, text to add to plot
		:ax: ax-object

		return imObject
		'''
		ax = self.private_GetAxis(ax)

		imText = ax.text(1.07,.5, text,
		  ha='center', va='center', transform= ax.transAxes,
		  bbox = {'facecolor': 'white', 'pad': 5}, rotation = -90)

		return imText

	def AddColorbar(self, im):
		'''
		add colorbar to axes
		'''		
		# from mpl_toolkits.axes_grid1 import make_axes_locatable
		# divider = make_axes_locatable(ax)
		# cax = divider.append_axes('right', size='5%', pad=0.05)
		import matplotlib
		cax = self.fig.add_axes([.9, 0.1, 0.01, 0.4])
		self.fig.colorbar(im, cax=cax, orientation='vertical',  norm=matplotlib.colors.LogNorm())


	def SetWhiteSpace(self, width = None, height = None):
		"""
		set the white space around each plot

		:width: float, the horizontal white space
		:height: float, the vertical white space
		"""
		plt.subplots_adjust(wspace=width, hspace=height)

	def RemoveWhiteSpace(self):
		"""
		sets the whitespace around each plot to be 0
		"""
		self.SetWhiteSpace(0,0)

	# TODO: 
	# - comment this
	# - make this get the most recent im object
	def SetExtent(self, extent, im):
		
		im.set_extent(extent)

	# saves the fig object
	# func(s)
	def Save(self, name = ""):
		"""
		saves the current fig object as a pdf

		:name: string, figure file name
		"""
		import os 

		dir_ = ""
		if os.path.isdir("../Figs"):
			dir_ = "../Figs/"

		if len(name) == 0:
			import time
			name = "fig" + str(time.time())

		self.fig.savefig(dir_ + name + ".pdf", bbox_inches='tight')

	# function underload
	def save(self, name = ""):
		"""
		saves the current fig object as a pdf

		:name: string, figure file name
		"""
		self.Save(name)


	def SavePng(self, name = ''):
		"""
		saves the current fig object as a png

		:name: string, figure file name
		"""
		import os 

		dir_ = ""
		if os.path.isdir("../Figs"):
			dir_ = "../Figs/"

		if len(name) == 0:
			import time
			name = "fig" + str(time.time())

		self.fig.savefig(dir_ + name + ".png", bbox_inches='tight')


	def SaveInDataDir(self, simName, name):
		'''
		save the current fig object as a pdf in the data directory
		'''
		import os 

		if os.path.isdir("../Data/" + simName + "/"):
			dir_ = f"../Data/{simName}/"
			self.fig.savefig(dir_ + name + ".pdf", bbox_inches='tight')
		else:
			self.Save(name)

	def AddRectangle(self, leftCorner, width, height, fill = False, ax = None, **kwargs):
		"""
		adds a rectangle to the given axis

		:leftCorner: tuple, (xPos, yPos)
		:width: float, width of rectangle
		:height: float, heigth of rectangle
		:ax: axesObj, the axis to be set, default last axis used
		"""
		if self.axsUsed >= self.axsTotal and ax == None:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		index_ = self.axsUsed
		if ax == None:
			self.axsUsed += 1
			ax = self.axs_fl[index_]

		color = 'k'
		if "color" in kwargs:
			color = kwargs['color']

		linewidth = 1
		if 'linewidth' in kwargs:
			linewidth = kwargs['linewidth']

		alpha = 1.
		if 'alpha' in kwargs:
			alpha = kwargs['alpha']

		ax.add_patch(
		    patches.Rectangle(
		        xy=leftCorner,  # point of origin.
		        width=width, height=height, linewidth=linewidth,
		        color=color, fill=fill, alpha = alpha))

		return ax

	def AddHusimi(self, psi, L = 1., hbar_ = 1., **kwargs):
		"""
		adds a plot of the husimi function of the field psi

		:psi: array-like, complex field
		:L: float, length of box
		:hbar_: float, hbar / field_mass

		:return: axesObj, axesImageObj; reference to axes and image of new plot
		"""
		if self.axsUsed >= self.axsTotal:
			raise Exception(
				"there are no free axs to add a plot to.\n" +\
				"reccomendation: initialize the FigObj with more subplots"
				)

		name_ = str(len(self.ims))

		if "name" in kwargs:
			name_ = kwargs["name"]

		H = QU.HusimiFunc(psi, hbar_ = hbar_, L = L)

		ax, im = self.AddImshow(H)

		N = len(psi)
		vmax = np.pi * N * hbar_ / L
		self.SetExtent((-L/2., L/2., -1*vmax, vmax), im)

		self.SetXLabel(r'$x$')
		self.SetYLabel(r'$v$')

		self.ims[name_] = im 
		ax.set_aspect(1./ax.get_data_ratio())
		return ax, im




def UpdateHist(x, im, nBins = 100, bins = [], density = True,
 weights = [], range = [], **kwargs):
	"""
	adds a new histogram plot

	:x: array-like, array of values to be binned
	:im: image object, the matplotlib image object to update
	:nBins: int, number of bins, default 100
	:bins: array-like, explicit bin edges, default use auto
	:density: bool, normalize histogram, default True
	:weights: array-like, weights of each point

	kwargs
	:log: bool, take log of counts

	"""
	weights_ = np.ones(np.shape(x))
	if len(weights) > 0:
		weights_ = weights

	n, edges = np.zeros(nBins), np.zeros(nBins + 1)
	if len(bins) > 0:
		n, edges = \
			np.histogram(x, bins = bins, density = density, weights=weights_)
	else:
		n, edges = \
			np.histogram(x, bins = nBins, density = density, weights=weights_)
	
	log = False
	if "log" in kwargs:
		log = kwargs["log"]

	x_ = .5*(edges[0:len(edges)-1] + edges[1:])

	im.set_data(x_, n)



