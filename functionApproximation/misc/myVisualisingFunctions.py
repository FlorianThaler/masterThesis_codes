"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements functions to 
					visualise the results of the function approximation procedure.
	Year:			2019
"""


###########################################################
# importing stuff
###########################################################

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import re

from cpd.myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc

from misc.myLoggingFunctions import computeEOC
###########################################################
# definition of functions
###########################################################

def createSurfPlot(modelFunc, exactFunc, path2FigDir):
	"""
		@param[in] ### modelFunc ###
		@param[in] ### exactFunc ###
		@param[in] ### path2FigDir ###
		
		this function produces a surface plot of the exact function and its approximation - provided the 
		dimension of the approximation domain is less than 3.
	"""
	# matplotlib.rc('font', size = 6)
	
	dim = modelFunc.getDim()
	
	if dim == 1:
		# case d = 1
		print('not implemented so far ...')
	elif dim == 2:
		# case d = 2
		
		# provide mesh
		M = 150
		x1 = np.linspace(modelFunc.getBoundingBox()[0][0], modelFunc.getBoundingBox()[0][1], M)
		x2 = np.linspace(modelFunc.getBoundingBox()[1][0], modelFunc.getBoundingBox()[1][1], M)		
		[xx1, xx2] = np.meshgrid(x1, x2)
		
		# compute function values
		zz = np.zeros((M, M))
		
		argsList = []
		argsList.append(xx1)
		argsList.append(xx2)
		zz1 = modelFunc.evaluate2d(argsList, M)
		zz2 = exactFunc(xx1, xx2)
			
		# produce surface plot of approximation
		fig1 = plt.figure()
		
		# plt.rc('axes', titlesize = 8)
		# plt.rc('axes', labelsize = 6)
		# plt.rc('xtick', labelsize = 4)
		# plt.rc('ytick', labelsize = 4)
		
		ax = fig1.gca(projection = '3d')
		surf = ax.plot_surface(xx1, xx2, zz1, antialiased = False, rstride = 1, cstride = 1, cmap = cm.coolwarm)
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')

		# write figure 2 file
		plt.savefig(path2FigDir + '/surfPlotApprox.png', dpi = 300)
		plt.close(fig1)
		
		# produce surface plot of exact function
		fig2 = plt.figure()
		ax = fig2.gca(projection = '3d')
		surf = ax.plot_surface(xx1, xx2, zz2, antialiased = False, rstride = 1, cstride = 1, cmap = cm.coolwarm)
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		
		# write figure 2 file
		plt.savefig(path2FigDir + '/surfPlotExact.png', dpi = 300)
		plt.close(fig2)
		
def createComparativeSurfPlot(dim, path2SignDir, subSignList, modFileNameList):
	"""
		@param[in] ### dim ### integer corresponding to the dimension of the approximation domain
		@param[in] ### path2SignDir ### string giving the path to the directory of the training session which should
			be investigated
		@param[in] ### subSignList ### list of strings giving the subsignatures of those directories which 
			contain the data which should be depicted
		@param[in] ### modFileNameList ### a list containing strings corresponding to the name of the files containing
			the model data, i.e. parameters, ...
			
		create surface plots depicting the exact function and two approximations.		
		NOTE:
			> several things have to be adjusted by 'hand', this is the approximation domain and the representation
				of the exact function
			> the elements in modFileNameList do not contain any extension!
	"""

	modFileDirName = 'model'
	setupFileName = 'setup.txt'

	N = len(subSignList)
	
	M = 150
	if N <= 2:

		dim = 2
		
		# default parameters 
		rank = 1
		degrs = [2, 2]

		boundingBox = []
		boundingBox.append((-0.5, 0.5))
		boundingBox.append((-0.5, 0.5))
		
			# boundingBox.append((0.25, 0.75))
			# boundingBox.append((0.25, 0.75))
			
			# boundingBox.append((0, 1.0))
			# boundingBox.append((0, 1.0))

		x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
		x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)
		
		[xx1, xx2] = np.meshgrid(x1, x2)
		
		zz = np.zeros((M, M))
		# zz = xx1 ** 2 + xx2 ** 2
		zz = np.sin(50 * xx1 * xx2)
		
		# list containing the surface data
		zList = []
		
		# list containing parameter details used for the title
		rankList = []
		degrList = []
		numDataPtsList = []
		
		for i in range(0, N):
			
			# ### read parameter settings from setup file
			
			# first read stuff from setup file ...
		
			path2SubDir = path2SignDir + '/' + subSignList[i]
			path2SetupFile = path2SubDir + '/' + setupFileName		
				
			stpFile = open(path2SetupFile, 'r')
			for line in stpFile:
				if 'numDataPoints' in line:
					numDataPtsList.append(np.int(re.findall('\d+$', line)[0]))
				elif 'rank' in line:
					rankList.append(np.int(re.findall('\d+$', line)[0]))
				elif 'degrCrdnt1' in line:
					degrList.append(np.int(re.findall('\d+$', line)[0]))
			
			stpFile.close()	
			
			# ### read model from file
			
			path2ModDir = path2SubDir + '/' + modFileDirName
		
			modelFunc = MyCPDRadialAnsatzFunc(dim, boundingBox, rank, degrs)
			modelFunc.readParamsFromFile(path2ModDir, modFileNameList[i])
		
			xList = []
			xList.append(xx1)
			xList.append(xx2)
			zList.append(modelFunc.evaluate2d(xList, M))
			
			del modelFunc

		fig = plt.figure(figsize = (15, 4.8))
			
		plt.rc('axes', titlesize = 8)
		plt.rc('axes', labelsize = 8)
		plt.rc('xtick', labelsize = 6)
		plt.rc('ytick', labelsize = 6)
		
		ax = fig.add_subplot(1, N + 1, 1, projection = '3d')
		# surf = ax.plot_surface(xx1, xx2, zz, antialiased = False, rstride = 1, cstride = 1, cmap = cm.coolwarm)
		surf = ax.plot_surface(xx1, xx2, zz, cmap = cm.coolwarm)
		ax.set_zlim(-1.2, 1.2)
		ax.view_init(elev = 30, azim = 255)
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_title('Graph of f')
		
		
		for i in range(0, N):
			ax = fig.add_subplot(1, N + 1, i + 2, projection = '3d')
			# surf = ax.plot_surface(xx1, xx2, zList[i], antialiased = False, rstride = 1, cstride = 1, cmap = cm.coolwarm)
			surf = ax.plot_surface(xx1, xx2, zList[i], cmap = cm.coolwarm)
			ax.set_zlim(-1.2, 1.2)
			ax.view_init(elev = 30, azim = 255)
			ax.set_xlabel('x1')
			ax.set_ylabel('x2')
			ax.set_title('Approximation to (r, N, m) = (' + str(rankList[i]) + ', ' + str(degrList[i]) + ', ' + str(np.int(numDataPtsList[i] / (2 * rankList[i] * degrList[i]))) + ')')
		
		# plt.subplots_adjust(wspace = 1.1)
		
		plt.show()
		

# def createPerfPlots(costFuncValList, l2ApprErrList, lInfApprErrList, cgPerformanceList, path2FigDir):
def createPerfPlots(costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList, path2FigDir):
	"""
		@param[in] ### costFuncValList ###
		@param[in] ### l2ApprErrList ###
		@param[in] ### lInfApprErrList ###
		@param[in] ### cgPerformanceList ###
		@param[in] ### path2FigDir ###
	
	"""
	
	numALSIter = len(costFuncValList)
	
	# ------------------------------------------
	
	fig1 = plt.figure()
	
	# plt.rc('axes', titlesize = 8)
	# plt.rc('axes', labelsize = 6)
	# plt.rc('xtick', labelsize = 4)
	# plt.rc('ytick', labelsize = 4)
	
	plt.semilogy(np.arange(1, numALSIter + 1), costFuncValList)
	plt.xlabel('ALS iterations')
	plt.title('Values of the cost functional')
	plt.xticks(np.arange(1, numALSIter + 1))
	plt.savefig(path2FigDir + '/costFuncVals.png', dpi = 300)
	plt.close(fig1)
	
	# ------------------------------------------
	
	fig2 = plt.figure()

	# plt.rc('axes', titlesize = 8)
	# plt.rc('axes', labelsize = 6)
	# plt.rc('xtick', labelsize = 4)
	# plt.rc('ytick', labelsize = 4)

	# plt.semilogy(np.arange(1, numALSIter + 1), l2ApprErrList)
	plt.semilogy(np.arange(1, numALSIter + 1), mseApprErrList)
	# plt.title(r'$l^{2}$' + ' approximation error')
	plt.title('Mean squared error')
	plt.xlabel('ALS iterations')
	plt.xticks(np.arange(1, numALSIter + 1))
	# plt.savefig(path2FigDir + '/l2ApprErrs.png', dpi = 300)
	plt.savefig(path2FigDir + '/mse.png', dpi = 300)
	plt.close(fig2)
	
	
	# ------------------------------------------
	
	fig3 = plt.figure()

	# plt.rc('axes', titlesize = 8)
	# plt.rc('axes', labelsize = 6)
	# plt.rc('xtick', labelsize = 5)
	# plt.rc('ytick', labelsize = 5)	

	plt.semilogy(np.arange(1, numALSIter + 1), lInfApprErrList)
	plt.title(r'$l^{\infty}$' + ' approximation error')
	plt.xlabel('ALS iterations')
	plt.xticks(np.arange(1, numALSIter + 1))
	plt.savefig(path2FigDir + '/lInfApprErrs.png', dpi = 300)
	plt.close(fig3)
	
	# ------------------------------------------
	
	fig4, (ax1, ax2) = plt.subplots(1, 2)	
	
	# plt.rc('axes', titlesize = 8)
	# plt.rc('axes', labelsize = 6)
	# plt.rc('xtick', labelsize = 4)
	# plt.rc('ytick', labelsize = 4)

	ax1.set_title('Euclidian norm of gradient')
	ax2.set_title('Number of cg iterations')
	ax1.set_xlabel('ALS iterations')
	ax2.set_xlabel('ALS iterations')
	
	dim = len(cgPerformanceList)
	for d in range(0, dim):	
		tmpList1 = [elem[1] for elem in cgPerformanceList[d]]
		tmpList2 = [elem[0] for elem in cgPerformanceList[d]]
		
		ax1.plot(np.arange(1, numALSIter + 1), tmpList1, label = 'x' + str(d + 1))
		ax2.plot(np.arange(1, numALSIter + 1), tmpList2, label = 'x' + str(d + 1))
		
	ax1.legend()
	ax2.legend()
	plt.tight_layout()
	plt.savefig(path2FigDir + '/cgPerformance.png', dpi = 300)
	plt.close(fig4)
	


def createComparativePerfPlot(dim, path2SignDir, subSignList):
	"""
		@param[in] ### dim ### integer corresponding to the dimension of the approximation domain
		@param[in] ### path2SignDir ### string giving the path to the directory of the training session which should
			be investigated
		@param[in] ### subSignList ### list of strings giving the subsignatures of those directories which 
			contain the data which should be depicted
	
		put the performance data stored in the directories whos names are given in subSignList in one plot and store
		the plots to file.
		NOTE:
			> make plots of the history of l^inf approximation error and the value of the cost functional.
			> make plots also of the history of eocs - in a different figure!
			> plots will be built only if N <= 5 !
	"""
	
	N = len(subSignList)
	
	colorList = ['b', 'g', 'r', 'c', 'm']
	
	if N <= 5:
			
		perfDataDirName = 'perfData'
		modDataDirName = 'model'
		
		perfDataFileName = 'performance.txt'
		setupFileName = 'setup.txt'


		# list of numpy arrays containing the sought data
		costFuncValList = []
		lInfApprErrList = []
		mseList = []
		eocList = []
		
		# list of lables containing the rank, the degree, and the number of data points used - these strings are
		# used as labels for the plots to be produced down below.
		labelList = []
		
		for elem in subSignList:

			#################################################################################
			# ### first gather data and store this data in list of arrays
			
			
			# care on performence data first
			path2PerfDataFile = path2SignDir + '/' + elem + '/' + perfDataDirName + '/' + perfDataFileName
			
			perfDataFile = open(path2PerfDataFile, 'r')
			
			linesList = perfDataFile.readlines()
			numALSIter = len(linesList) - 1
			
			costFuncArr = np.zeros(numALSIter)					# store values of cost functional in this array
			# l2ApprErrArr = np.zeros(numALSIter)					# store values of approximation error in these arrays ...
			mseArr = np.zeros(numALSIter)					# store values of approximation error in these arrays ...
			lInfApprErrArr = np.zeros(numALSIter)
						
			for i in range(0, numALSIter):
			
				fields = linesList[i + 1].split(',')
				costFuncArr[i] = np.float(fields[0])
				# l2ApprErrArr[i] = np.float(fields[1])
				mseArr[i] = np.float(fields[1])
				lInfApprErrArr[i] = np.float(fields[2])
			
			perfDataFile.close()
			
			costFuncValList.append(costFuncArr)
			lInfApprErrList.append(lInfApprErrArr)
			mseList.append(mseArr)

			# now deal with model data - needed for the eoc
			path2ModDir = path2SignDir + '/' + elem + '/' + modDataDirName
			_, tmpList = computeEOC(path2ModDir, 'modelData', '', storeFig = False)
			
			eocList.append(np.asarray(tmpList))

			#################################################################################
			# ### now read rank, degr and number of data points used from setup file - used for the legend in the
			# 		plots down below.
			degrList = []
			degrCrdntStrgList = []
			for d in range(0, dim):
				degrCrdntStrgList.append('degrCrdnt' + str(d + 1))

			path2SetupFile = path2SignDir + '/' + elem + '/' + setupFileName
		
			stpFile = open(path2SetupFile, 'r')
			for line in stpFile:
				if 'numDataPoints' in line:
					numDataPoints = np.int(re.findall('\d+$', line)[0])
				elif 'rank' in line:
					rank = np.int(re.findall('\d+$', line)[0])
				elif any(strg in line for strg in degrCrdntStrgList):
					degrList.append(np.int(re.findall('\d+$', line)[0]))
			
			stpFile.close()

			# tmp = 1
			# for i in range(0, len(degrList)):
				# tmp *= degrList[i]
			labelList.append('(' + str(rank) + ', ' + str(degrList[0]) + ', ' + str(np.int(numDataPoints / (dim * degrList[0] * rank))) + ')')


		fig1 = plt.figure(figsize = (12, 4.8))

		plt.subplot(1, 3, 1)
		for i in range(0, N):
			plt.plot(np.arange(1, len(costFuncValList[i]) + 1), costFuncValList[i], color = colorList[i], label = labelList[i])
		ax1 = plt.gca()
		ax1.set_title('Values of cost functional')
		ax1.set_xlabel('ALS iterations')
		ax1.legend()
		
		plt.subplot(1, 3, 2)
		for i in range(0, N):
			plt.plot(np.arange(1, len(mseList[i]) + 1), mseList[i], color = colorList[i], label = labelList[i])
		ax2 = plt.gca()
		ax2.set_title('Mean squared error')
		ax2.set_xlabel('ALS iterations')
		ax2.legend()
		
		plt.subplot(1, 3, 3)
		for i in range(0, N):
			plt.plot(np.arange(1, len(apprErrList[i]) + 1), apprErrList[i], color = colorList[i], label = labelList[i])
		ax3 = plt.gca()
		ax3.set_title(r'$l^{\infty}$' + ' approximation errors')
		ax3.set_xlabel('ALS iterations')
		ax3.legend()
		
		# plt.subplots_adjust(wspace = 1.3)

		fig2 = plt.figure()

		for i in range(0, N):
			plt.plot(np.arange(1, len(eocList[i]) + 1), eocList[i], color = colorList[i], label = labelList[i])
		ax3 = plt.gca()
		ax3.set_title('Experimental order of convergence')
		ax3.set_xlabel('Level')
		ax3.legend()
		
		plt.show()
