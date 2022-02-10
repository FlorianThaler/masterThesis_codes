"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements a 2d 
					optimal control problem example.
	Year:			2019/2020
"""

########################################################################################################################
# importing stuff
########################################################################################################################	

import numpy as np
import random

import logging

import re


from matplotlib import pyplot as plt

from collections import deque

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import time

from valueFunctionApprox.myValueFunctionApproximation import MyValueFunctionApproximation
from misc.myLoggingFunctions import createSetupFile
from misc.myLoggingFunctions import depositeIntermediateTrainingResults

import sys
sys.path.append('../functionApproximation/cpd')
from myALSOptimiser import MyALSRbfOptimiser

########################################################################################################################	
# definition of functions
########################################################################################################################	

def terminalCostFunc(xList, domainBox):
	"""
		@param[in] ### xList ### list of length two containing data points, where 
			xList[0] corresponds to the x1 data, and xList[1] corresponds to the x2 data
			NOTE:
				> data can be given in scalar or in matrix form!
		@param[in] ### domainBox ### list of two tuples representing the boundary of the domain the vehicle should 
			not leave
		@return ### retVal ### a scalar or a matrix modeling the cost of achieving the states
			given in xList at the final time
	"""
	
	# take care of the following:
	#	> there are logical operators in python for boolean arrays and for boolean scalars - there is no approach to 
	#		cover both cases ...
	#	> hence distinguish two cases: is xList a list of two scalars or is it a list of arrays ....
	
	tmp = 100.0
	
	if np.isscalar(xList[0]) == True:
		if (domainBox[0][0] <= xList[0] <= domainBox[0][1]) and (domainBox[1][0] <= xList[1] <= domainBox[1][1]):
			return 0.0
		else:
			return tmp
			
	else:	
		retVal = tmp * np.ones(xList[0].shape)
		
		tmp1 = np.logical_and((domainBox[0][0] <= xList[0]), (xList[0] <= domainBox[0][1]))
		tmp2 = np.logical_and((domainBox[1][0] <= xList[1]), (xList[1] <= domainBox[1][1]))
		
		retVal[np.logical_and(tmp1, tmp2)] = 0
	
		return retVal

def stageCostFunc(xList, u, domainBox):
	"""
		@param[in] ### xList ### as in terminalCostFunc
		@param[in] ### u ### a single control value (a scalar, not a vector or a matrix)
		@param[in] ### domainBox ### as in terminalCostFunc
		@return ### retVal ### a scalar or a matrix modeling the cost of performing the control u given the states xList
	"""

	return xList[0] ** 2 + xList[1] ** 2 + u ** 2 + terminalCostFunc(xList, domainBox)
	# return 1000 * xList[0] ** 2 + xList[1] ** 2 + 0.000001 * u ** 2 + terminalCostFunc(xList, domainBox)


def mySdeCoeffsFunc(xList, u, sig1, sig2):
	"""
		@param[in] ### xList ### as above
		@param[in] ### u ### as above
		@param[in] ### sig1 ### volatility parameter of the sde
		@param[in] ### sig2 ### volatility parameter of the sde
		
		@return ### retVal1, retVal2 ### vectors or matrices containing the coefficients of the
			sde given the input parameters
	"""

	if np.isscalar(xList[0]) == True:
		retVal1 = np.zeros(2)
		retVal2 = np.zeros(2)
		
		retVal1[0] = xList[1]
		retVal1[1] = u
		
		retVal2[0] = sig1
		retVal2[1] = sig2
		
		return retVal1, retVal2
	else:
		retVal1 = np.zeros((2, xList[0].shape[0]))
		retVal2 = np.zeros((2, xList[1].shape[0]))
		
		retVal1[0, :] = xList[1]
		retVal1[1, :] = u
		
		retVal2[0, :] = sig1
		retVal2[1, :] = sig2
		
		return retVal1, retVal2
	
def my2dExample_a(L, rank, degrs, signature, subSignature, path2SubSignDir, path2FigDir,\
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
		
		NOTE:
			> at every time point the same ansatz function is used for the approximation of the value function!
	"""

	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### initialisations
	#################################################################

	####################### TEMPORAL AND SPATIAL INITIALISATIONS ########################

	dim = 2
	boundingBox = []
	boundingBox.append((-4.0, 4.0))
	boundingBox.append((-4.0, 4.0))
	
	sampleBox = []
	sampleBox.append((-4.0, 4.0))
	sampleBox.append((-4.0, 4.0))
	
	numControls = 21
	controlVec = np.linspace(-1, 1, numControls)
	
	t0 = 0
	T = 3.0
	numTimeIntervals = 30
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	


	####################### USER CHECK ########################
	
		# print('### check parameters ###')
		# print('	> t0 = ' + str(t0))
		# print('	> T = ' + str(T))
		# print('	> numTimeIntervals = ' + str(numTimeIntervals))
		# print('	> numTimePts = ' + str(numTimePts))
		# print('	> dt = ' + str(dt))
		# print('	> sqrt(dt) = ' + str(sqrtDt))
		# print('	> numControls = ' + str(numControls))
		# print('########################')
		
		# input(' ... hit ENTER to continue ...')
	
	####################### MODEL PARAMETERS ########################

	prob = 1.0 / (2 * dim)

	sigma1 = 5e-2
	sigma2 = 5e-2
	
	domainBox = []
	domainBox.append((-2, 2))
	domainBox.append((-2, 2))

	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 200
	eta = 1e-4
	epsALS = 1e-5
	
	# fix CG parameters (there are default parameters ... )
	maxNumCGIter = 50
	epsCG = 1e-5
	resNormFrac = 1e-1
	
	# fix parameters regarding the value iteration
	maxNumValIter = 1
	
	#################################################################
	# ### write parameters into to dict - for logging purposes
	#################################################################

	# sde parameters
	sdeParamsDict = {}
	sdeParamsDict['sigma1'] = sigma1
	sdeParamsDict['sigma2'] = sigma2
	sdeParamsDict['domainBox_x1_left'] = domainBox[0][0]
	sdeParamsDict['domainBox_x1_right'] = domainBox[0][1]
	sdeParamsDict['domainBox_x2_left'] = domainBox[1][0]
	sdeParamsDict['domainBox_x2_right'] = domainBox[1][1]

	# parameters of the control problem
	ctrlProbDict = {}
	ctrlProbDict['numTimePts'] = numTimePts
	ctrlProbDict['startTime'] = t0
	ctrlProbDict['endTime'] = T
	ctrlProbDict['numControls'] = numControls
	ctrlProbDict['controlLwrBnd'] = controlVec[0]
	ctrlProbDict['controlUprBnd'] = controlVec[-1]
	
	# value iteration parameters
	valIterParamsDict = {}
	valIterParamsDict['sampleBox_x1_left'] = sampleBox[0][0]
	valIterParamsDict['sampleBox_x1_right'] = sampleBox[0][1]
	valIterParamsDict['sampleBox_x2_left'] = sampleBox[1][0]
	valIterParamsDict['sampleBox_x2_right'] = sampleBox[1][1]
	valIterParamsDict['maxNumValIter'] = maxNumValIter
		
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
	alsParamsDict['eta'] = eta
	alsParamsDict['maxNumALSIter'] = maxNumALSIter
	alsParamsDict['numDataPoints'] = L
	alsParamsDict['descentBound'] = epsALS
	
	cgParamsDict = {}
	cgParamsDict['maxNumCGIter'] = maxNumCGIter
	cgParamsDict['residualBound'] = epsCG
	cgParamsDict['residualFraction'] = resNormFrac
	
	#################################################################
	# ### initialise model function
	#################################################################
	
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, domainBox))
	apprParams = []
	for i in range(0, numTimePts - 1):
		apprParams.append((rank, degrs))		

	modelFunc.initialise(apprParams)
	
	totNumParams = 0
	for d in range(0, dim):
		totNumParams += rank * degrs[d]
	
	#################################################################
	# ### start value function approximation procedure
	#################################################################
	
	logging.info('------------------------------------------------------------------------------------------')
	logging.info('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')
	
	print('------------------------------------------------------------------------------------------')
	print('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')

	xDataList = []
	for i in range(0, numTimePts - 1):
		tmpXData = np.zeros((dim, L))
		tmpXData[0, :] = np.random.uniform(sampleBox[0][0], sampleBox[0][1], L)
		tmpXData[1, :] = np.random.uniform(sampleBox[1][0], sampleBox[1][1], L)
		xDataList.append(tmpXData)

	# initialise object of optimiser class
	optimiser = MyALSRbfOptimiser(eta = eta, maxNumALSIter = maxNumALSIter, epsALS = epsALS)
		
	# provide a data structure where certain results of the optimisation procedure will be stored ...
	optimData = np.zeros((maxNumValIter, numTimePts - 1, 3))
	stateData = np.zeros((numTimePts, dim, L))
	controlData = np.zeros((numTimePts - 1, L))

	##########################################################################
	
	depositeIntermediateTrainingResults(modelFunc, path2ModDir, 0)
	
	for k in range(0, maxNumValIter):
		
		logging.info('start policy iteration number # ' + str(k) + ' #')
		print('start policy iteration number # ' + str(k) + ' #')
						
		#################################################################
		# ### update phase
		#################################################################

		# RECALL:
		#	> there are numTimePts - 1 ansatz functions which have to be optimised!
		#	> the value function at the final time point equals the terminal cost function
		
		for i in range(numTimePts - 2, -1, -1):
	
			################################################################################
			
			print('# update - time point t' + str(i))
			logging.info('# update - time point t' + str(i))
			
			################################################################################
						
			# reinitialise data
			xData = np.zeros((dim, L))
			xData = xDataList[i].copy()
			yData = np.zeros(L)
			stgCost = np.zeros(L)
			
			################################################################################
			
			f0 = np.zeros((numControls, dim, L))
			f1 = np.zeros((numControls, dim, L))
			f2 = np.zeros((numControls, dim, L))
			f3 = np.zeros((numControls, dim, L))
			
			controlData = np.zeros(L)
			
			sampleIdx = np.zeros(L)
			sampleIdx = np.random.choice([0, 1, 2, 3], L, replace = True)

			sampleData = np.zeros((2, L))
			sampleData = sampleSpace[sampleIdx].transpose()

			tmpVec = np.zeros((numControls, L))
			
			for j in range(0, numControls):
									
				a1 = np.zeros((dim, L))
				b1 = np.zeros((dim, L))
				
				stgCost = np.zeros(L)
				
				a1, b1 = mySdeCoeffsFunc([xData[0, :], xData[1, :]], controlVec[j], sigma1, sigma2)

				f0[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([1, 0]).reshape(2, 1)
				f1[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 1]).reshape(2, 1)
				f2[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, -1]).reshape(2, 1)
				f3[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([-1, 0]).reshape(2, 1)
				
				stgCost = stageCostFunc([xData[0, :], xData[1, :]], controlVec[j], domainBox)
				
				tmpVec[j, :] = stgCost * dt \
					+ prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]], L) \
								+ modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]], L) \
								+ modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]], L) \
								+ modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]], L))
	
	
			################################################################################
			
			# compute target values
			yData = tmpVec.min(axis = 0)

			# start optimisation process
					
			costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList = \
				optimiser.myRbfCgOptimiser(L, xData, yData, modelFunc.getPartialModelFunc(i), \
					path2ModDir, 'modelFunc_t' + str(i), \
					maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
					warmUp = False, verbose = False, write2File = False)

		depositeIntermediateTrainingResults(modelFunc, path2ModDir, k + 1)
	
	####################################################################################################################
	# ### post processing
	####################################################################################################################

	# create setup file
	logging.info('> create setup file')
	createSetupFile(sdeParamsDict, ctrlProbDict, valIterParamsDict, alsParamsDict, cgParamsDict, cpdParamsDict, path2SubSignDir)
	
	# write performance data 2 file
	# writePerformanceData2File(numTimePts, optimData, pathNames[0])	

def simulateFromFile_2d_a(sessId, subSessId, iterId):
	"""
		this function reads a model from file (more exact: its parameter) and simulates the dynamics of the system
		following the policy given by the approximation to the value function
		
		@param[in] ### sessId ### integer corresponding to the of the training session whos model files should be load
		@param[in] ### subsSessId ### integer corresponding to the of the training sub session
			whos model files should be load
		@param[in] ### iterId ### integer corresponding to the of the iteration level whos model files should be load
		
	"""
	
	sign = str(sessId)
	subSign = str(subSessId)

	path2Dir = '../../results/valueIteration/2d_a_' + sign + '/2d_a_' + sign + '_' + subSign
	
	#################################################################
	# ### initialisations
	#################################################################

	####################### TEMPORAL AND SPATIAL INITIALISATIONS ########################

	dim = 2
	boundingBox = []
	boundingBox.append((-4.0, 4.0))
	boundingBox.append((-4.0, 4.0))
	
	# sampleBox = []
	# sampleBox.append((-3.0, 3.0))
	# sampleBox.append((-3.0, 3.0))
	
	numControls = 21
	controlVec = np.linspace(-1, 1, numControls)
	
	t0 = 0
	T = 3.0
	numTimeIntervals = 30
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	
	
	####################### MODEL PARAMETERS ########################

	prob = 1.0 / (2 * dim)

	sigma1 = 5e-2
	sigma2 = 5e-2
	
	domainBox = []
	domainBox.append((-2, 2))
	domainBox.append((-2, 2))
	
	####################### USER CHECK ########################
	
		# print('### check parameters ###')
		# print('	> t0 = ' + str(t0))
		# print('	> T = ' + str(T))
		# print('	> numTimeIntervals = ' + str(numTimeIntervals))
		# print('	> numTimePts = ' + str(numTimePts))
		# print('	> dt = ' + str(dt))
		# print('	> sqrt(dt) = ' + str(sqrtDt))
		# print('	> numControls = ' + str(numControls))
		# print('########################')
		
		# proceed by pressing enter
		# input(' ... hit ENTER to continue ...')

	###########################################################
		
	#################################################################
	# ### initialise model function
	#################################################################	
	
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, \
		boundingBox, lambda x : terminalCostFunc(x, domainBox))
	modelFunc.readParamsFromFile(path2Dir + '/model' + '/iter_' + str(iterId), 'modelData_t')

	for i in range(0, numTimePts - 1):
		modelFunc.partialModelFuncList[i].setBoundingBox(boundingBox)
	
	#################################################################
	# ### provide data structures needed in the simulation process
	#################################################################	
	
	numEvalPaths = 5
	f0 = np.zeros((numControls, dim, numEvalPaths))
	f1 = np.zeros((numControls, dim, numEvalPaths))
	f2 = np.zeros((numControls, dim, numEvalPaths))
	f3 = np.zeros((numControls, dim, numEvalPaths))

	sampleIdx = np.zeros((numTimePts, numEvalPaths))
	sampleIdx = np.random.choice([0, 1, 2, 3], (numTimePts, numEvalPaths), replace = True)
		
	sampleData = np.zeros((dim, numEvalPaths))
	
	X = np.zeros((numTimePts, dim, numEvalPaths))
	U = np.zeros((numTimePts - 1, numEvalPaths))
	
	# choose initial value
	X[0, 0, :] = 1.2
	X[0, 1, :] = 0.5

	for i in range(0, numTimePts - 1):
					
		####################################################
		#
		# simulate using approximative value function
		#
		####################################################

		# the same as above using the approximation to the value function
		
		tmpVec = np.zeros((numControls, numEvalPaths))
		sampleData = sampleSpace[sampleIdx[i, :]].transpose()

		for j in range(0, numControls):
			
			a = np.zeros((dim, numEvalPaths))
			b = np.zeros((dim, numEvalPaths))
			
			a, b = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :]], controlVec[j], sigma1, sigma2)
				
			f0[j, :, :] = X[i, :, :] + a * dt + sqrtDt * b * np.asarray([1, 0]).reshape(2, 1)
			f1[j, :, :] = X[i, :, :] + a * dt + sqrtDt * b * np.asarray([0, 1]).reshape(2, 1)
			f2[j, :, :] = X[i, :, :] + a * dt + sqrtDt * b * np.asarray([0, -1]).reshape(2, 1)
			f3[j, :, :] = X[i, :, :] + a * dt + sqrtDt * b * np.asarray([-1, 0]).reshape(2, 1)
												
			stgCost = stageCostFunc([X[i, 0, :], X[i, 1, :]], controlVec[j], domainBox)
				
			tmpVec[j, :] = 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]], numEvalPaths)) \
				+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]], numEvalPaths)) \
				+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]], numEvalPaths)) \
				+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]], numEvalPaths))
	
		# determine optimal control
		
		U[i, :] = controlVec[tmpVec.argmin(axis = 0)]
		
		a = np.zeros((2, numEvalPaths))
		b = np.zeros((2, numEvalPaths))
		
		# ... and finally the subsequent state
		a, b = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :]], U[i, :], sigma1, sigma2)
		
		tmpMtrx = np.zeros(b.shape)

		for row in range(0, dim):
			tmpMtrx[row, :] = sampleData[row, :] * b[row, :]

		X[i + 1, :, :] = X[i, :, :] + a * dt + sqrtDt * tmpMtrx

	##########################################################################################################

	# ### produce plot showing a variety of paths
	
	fig1 = plt.figure(1)
	for l in range(0, numEvalPaths):
		plt.plot(X[:, 0, l], X[:, 1, l])
	
	plt.title('Solution paths over time')
	plt.xlabel('Position')
	plt.ylabel('Velocity')
	plt.xlim(0, 2)
	plt.ylim(-1, 1)
	# plt.xlim(boundingBox[0][0], boundingBox[0][1])
	# plt.ylim(boundingBox[1][0], boundingBox[1][1])
	plt.savefig(path2Dir + '/figures/' + 'solutionPaths.png')
	plt.close(fig1) 	
	
	
	
	# ### produce a plot showing the control paths over time
	
	fig2 = plt.figure(2)
	for l in range(0, numEvalPaths):
		plt.plot(np.arange(0, numTimePts - 1), U[:, l])
	
	plt.title('Control paths over time')
	plt.xlabel('Time')
	plt.ylabel('Acceleration')
	# plt.xlim(domainBox[0][0], domainBox[0][1])
	plt.ylim(-1.1, 1.1)
	plt.savefig(path2Dir + '/figures/' + 'controlPaths.png')
	plt.close(fig2)
	
	
	# plt.show()
	
	# ### produce plots showing/depicting the value function at every time point
	
	M = 100

	
	for i in range(0, numTimePts):
		x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
		x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)
	
		[xx1, xx2] = np.meshgrid(x1, x2)	
		vv = modelFunc.evaluate2d(i, [xx1, xx2], M)
		
		fig = plt.figure()
				# ax = fig.gca(projection = '3d')
				# ax.plot_surface(xx1, xx2, vv)
		
		plt.contourf(vv)
		plt.colorbar()
		plt.title('Contour plot of value function at time t = t_' + str(i))
		plt.xticks([0, 25, 50, 75, 99], [-4, -2, 0, 2, 4])
		plt.yticks([0, 25, 50, 75, 99], [-4, -2, 0, 2, 4])
		plt.xlabel('Position')
		plt.ylabel('Velocity')
		
		plt.savefig(path2Dir + '/figures/' + 'valFuncAtTime_t' + str(i) + '.png')
		# plt.show()
		plt.close(fig)
