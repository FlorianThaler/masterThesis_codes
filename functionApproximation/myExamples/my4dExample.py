"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements the optimiser 
					used for the update process in the context of function approximation.
	Year:			2019
"""

########################################################################################################################
# importing stuff
########################################################################################################################	

import numpy as np
import random

import logging

from cpd.myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc
from cpd.myALSOptimiser import *

from misc.myLoggingFunctions import createSetupFile
from misc.myLoggingFunctions import writePerformanceData2File
from misc.myLoggingFunctions import computeEOC
from misc.myVisualisingFunctions import createSurfPlot
from misc.myVisualisingFunctions import createPerfPlots

import time

########################################################################################################################	
# definition of functions
########################################################################################################################	

def myDataFunc4d(x1, x2, x3, x4):
	"""
		@param[in] ### x1 ### numpy array of points along the x1 axes
		@param[in] ### x2 ### numpy array of points along the x2 axes
		@param[in] ### x3 ### numpy array of points along the x3 axes
		@param[in] ### x4 ### numpy array of points along the x4 axes
		@return ### retVal ### numpy array corresponding to the function values of the function to be
			approximated at (x1, x2, x3, x4)
	
		this function generates function values of a known function, given points in the approximation domain
	"""
	
	# Park (1991) function 1
	return 0.5 * x1 * (np.sqrt(1 + (x2 + x3 ** 2) * (x4 / (x1 ** 2))) - 1) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
	
########################################################################################################################	

def my4dExample(L, rank, degrs, signature, subSignature, path2SubSignDir, path2FigDir,\
						path2ModDir, path2PerfDataDir):
	
	"""
		@param[in] ### L ### number of data points to be used
		@param[in] ### rank ### rank of cpd decomposition which will be used
		@param[in] ### degrs ### list of numbers corresponding to the degrees for the cpd decomposition along each of
					the coordinate axis.
		@param[in] ### signature ### string corresponding to the signature of the training session
		@param[in] ### subsignature ### string corresponding to the subsignature of the training session
		@param[in] ### path2SubSignDir ### string corresponding to the path of the directory where information/data
					of the corresponding subsession should be stored.
		@param[in] ### path2FigDir ### string corresponding to the path of the directory where figures
					of the corresponding subsession should be stored.
		@param[in] ### path2ModDir ### string corresponding to the path of the directory where the model
					of the corresponding subsession should be stored.
		@param[in] ### path2PerfDataDir ### string corresponding to the path of the directory where performance data
					of the corresponding subsession should be stored.
		
		this function corresponds to the 2d function approximation example, i.e. data points in a approximation 
		domain will be generated, then approximation algorithm will be called, then evaluation/post processing will be
		done.
	"""
	
	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### provide data: approximation domain, data points, ...
	#################################################################

	dim = 4

	boundingBox = []
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))

	x1Data = np.random.uniform(boundingBox[0][0], boundingBox[0][1], L)
	x2Data = np.random.uniform(boundingBox[1][0], boundingBox[1][1], L)
	x3Data = np.random.uniform(boundingBox[2][0], boundingBox[2][1], L)
	x4Data = np.random.uniform(boundingBox[3][0], boundingBox[3][1], L)
	
	xData = np.zeros((dim, L))
	xData[0, :] = x1Data
	xData[1, :] = x2Data
	xData[2, :] = x3Data
	xData[3, :] = x4Data
	
	yData = myDataFunc4d(x1Data, x2Data, x3Data, x4Data)

	#################################################################
	# ### fix ALS and CG parameters
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 200
	eta = 1e-4
	epsALS = 1e-5
	
	# fix CG parameters (there are default parameters ... )
	maxNumCGIter = 50 
	epsCG = 1e-5
	resNormFrac = 1e-1
	
	#################################################################
	# ### write parameters into to dict - for logging purposes
	#################################################################
	
	# cpd parameters
	cpdParamsDict = {}
	cpdParamsDict['rank'] = rank
	for d in range(0, dim):
		cpdParamsDict['degrCrdnt' + str(d + 1)] = degrs[d]
	
	# als parameters
	alsParamsDict = {}
	alsParamsDict['boundingBox_x1_left'] = boundingBox[0][0]
	alsParamsDict['boundingBox_x1_right'] = boundingBox[0][1]
	alsParamsDict['boundingBox_x2_left'] = boundingBox[1][0]
	alsParamsDict['boundingBox_x2_right'] = boundingBox[1][1]
	alsParamsDict['boundingBox_x3_left'] = boundingBox[2][0]
	alsParamsDict['boundingBox_x3_right'] = boundingBox[2][1]
	alsParamsDict['boundingBox_x4_left'] = boundingBox[3][0]
	alsParamsDict['boundingBox_x4_right'] = boundingBox[3][1]
	alsParamsDict['eta'] = eta
	alsParamsDict['maxNumALSIter'] = maxNumALSIter
	cpdParamsDict['numDataPoints'] = L
	alsParamsDict['descentBound'] = epsALS
	
	# cg parameters
	cgParamsDict = {}
	cgParamsDict['maxNumCGIter'] = maxNumCGIter
	cgParamsDict['residualBound'] = epsCG
	cgParamsDict['residualFraction'] = resNormFrac
	
	#################################################################
	# ### initialise ansatz function
	#################################################################
	
	modelFunc = MyCPDRadialAnsatzFunc(dim, boundingBox, rank, degrs)
	modelFunc.initialise()
	
	#################################################################
	# ### initialise ansatz function
	#################################################################

	
	logging.info('------------------------------------------------------------------------------------------')
	logging.info('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')
	
	# initialise object of optimiser class
	optimiser = MyALSRbfOptimiser(eta = eta, maxNumALSIter = maxNumALSIter, epsALS = epsALS)
		
	print('> start optimisation procedure')
		
	t0 = time.time()
	# costFuncValList, l2ApprErrList, lInfApprErrList, cgPerformanceList = \
	costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList = \
			optimiser.myRbfCgOptimiser(L, xData, yData, modelFunc, path2ModDir, 'modelData', \
			maxNumCGIter = maxNumCGIter, epsCG = epsCG, \
			resNormFrac = resNormFrac, warmUp = False, verbose = False)
	
	t1 = time.time()
	
	logging.info('> optimisation procedure corresponding to sub signature # ' + subSignature + ' # finished after # ' + str(t1 - t0) + ' # seconds')
	
	print('> optimisation procedure finished after # ' + str(t1 - t0) + ' # seconds')
	
	####################################################################################################################
	# ### post processing
	####################################################################################################################
	
	# first create setup file
	logging.info('> create setup file')
	createSetupFile(alsParamsDict, cgParamsDict, cpdParamsDict, path2SubSignDir)
	
	# write performance data 2 file
	logging.info('> write performance data to file')
	# writePerformanceData2File(costFuncValList, l2ApprErrList, lInfApprErrList, cgPerformanceList, path2PerfDataDir)
	writePerformanceData2File(costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList, path2PerfDataDir)
	
	################################################################################################
	# ### parameter will be stored inside the optimiser method! simply because 
	#		the parameter after each iteration should be stored! the purpose: experimental order
	#		of convergence
	#
	# store model parameters 2 file
	# logging.info('> write model parameters to file')
	# modelFunc.writeParams2File(path2ModDir, 'modelData')
	#
	# ###
	################################################################################################

	# NOTE:
	#	> all the plots will be stored to file!

	# produce plots of exact function, approximation and further the difference of both - as surface plot!
		# createSurfPlot(modelFunc, myDataFunc2d, path2FigDir)
	
	# produce performance plots
		# createPerfPlots(costFuncValList, l2ApprErrList, lInfApprErrList, cgPerformanceList, path2FigDir)
	createPerfPlots(costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList, path2FigDir)
	
	# produce plot showing the experimental order of convergence
	computeEOC(path2ModDir, 'modelData', path2FigDir)
