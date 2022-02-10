"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements a 1d 
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

def exactValFunc(xList, t, r, mu, sigma, alpha, T):
	tmp = alpha * (r + 0.5 * (((mu - r) ** 2) / ((1 - alpha) * (sigma ** 2))))
	return (xList[0] ** alpha) * np.exp(tmp * (T - t))


def terminalCostFunc(xList, alpha):
	"""
		@param[in] ### xList ### list of length one containing data points in the approximation domain
		@param[in] ### alpha ### parameter in the interval (0, 1)
		@return ### retVal ### terminal costs of the states given in x1
	"""
	
	retVal = np.zeros(xList[0].shape)
	retVal = xList[0] ** alpha	
	return retVal	

def mySdeCoeffsFunc(xList, u, r, mu, sig):
	"""
		@param[in] ### xList ### list of length one containing state values
		@param[in] ### u ### control value in the interval [0, 1]
		@param[in] ### r ### interest rate of the riskless asset
		@param[in] ### mu ### interest rate of the risky asset
		@param[in] ### sig ### volatility of the dynamics
		
		@return ### retVal1, retVal2 ### coefficients of the sde given the input parameters
	"""
	retVal1 = (r + u * (mu - r)) * xList[0]
	retVal2 = sig * u * xList[0]
	return retVal1, retVal2


def my1dExample_a(L, rank, degrs, signature, subSignature, path2SubSignDir, path2FigDir,\
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

	dim = 1
	boundingBox = []
	boundingBox.append((0.0, 10.0))
	
	sampleBox = []
	sampleBox.append((0.0, 10.0))
	
	numControls = 21
	controlVec = np.linspace(0, 1, numControls)
	
	t0 = 0
	T = 5.0
	numTimeIntervals = 50
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	# sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	
	sampleSpace = np.array([-1, 1])


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

	r = 0.03
	mu = 0.035
	alpha = 0.1
	sigma = 1e-1

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
	sdeParamsDict['r'] = r
	sdeParamsDict['mu'] = mu
	sdeParamsDict['alpha'] = alpha
	sdeParamsDict['sigma'] = sigma
	
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

	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, alpha))
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
			
			controlData = np.zeros(L)
			
			sampleIdx = np.zeros(L)
			sampleIdx = np.random.choice([0, 1], L, replace = True)

			sampleData = np.zeros((2, L))
			sampleData = sampleSpace[sampleIdx].transpose()

			tmpVec = np.zeros((numControls, L))
			
			for j in range(0, numControls):
									
				a1 = np.zeros((dim, L))
				b1 = np.zeros((dim, L))
				
				stgCost = np.zeros(L)

				a1, b1 = mySdeCoeffsFunc([xData[0, :]], controlVec[j], r, mu, sigma)

				f0[j, :, :] = xData + a1 * dt + sqrtDt * b1 * 1
				f1[j, :, :] = xData + a1 * dt + sqrtDt * b1 * (-1)
				
				# ... there is no stage cost in this problem!
				tmpVec[j, :] = prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :]], L) \
								+ modelFunc.evaluate(i + 1, [f1[j, 0, :]], L))
	
	
			################################################################################
			
			# compute target values
			# yData = tmpVec.min(axis = 0)
			yData = tmpVec.max(axis = 0)

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

def simulateFromFile_1d_a(sessId, subSessId, iterId):
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

	path2Dir = '../../results/valueIteration/1d_a_' + sign + '/1d_a_' + sign + '_' + subSign
	
	#################################################################
	# ### initialisations
	#################################################################

	####################### TEMPORAL AND SPATIAL INITIALISATIONS ########################

	dim = 1
	boundingBox = []
	boundingBox.append((0.0, 10.0))
	
	sampleBox = []
	sampleBox.append((0.0, 10.0))
	
	numControls = 21
	controlVec = np.linspace(0, 1, numControls)
	
	t0 = 0
	T = 5.0
	numTimeIntervals = 50
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	sampleSpace = np.array([-1, 1])	
	
	####################### MODEL PARAMETERS ########################

	prob = 1.0 / (2 * dim)

	r = 0.03
	mu = 0.035
	alpha = 0.1
	sigma = 1e-1
	
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
		boundingBox, lambda x : terminalCostFunc(x, alpha))
	modelFunc.readParamsFromFile(path2Dir + '/model' + '/iter_' + str(iterId), 'modelData_t')

	for i in range(0, numTimePts - 1):
		modelFunc.partialModelFuncList[i].setBoundingBox(boundingBox)
	
	#################################################################
	# ### provide data structures needed in the simulation process
	#################################################################	
	
	numEvalPaths = 5
	colors = ['b', 'g', 'c', 'm', 'k']	
	
	f0 = np.zeros((numControls, dim, numEvalPaths))
	f1 = np.zeros((numControls, dim, numEvalPaths))

	sampleIdx = np.zeros((numTimePts, numEvalPaths))
	sampleIdx = np.random.choice([0, 1], (numTimePts, numEvalPaths), replace = True)
		
	X = np.zeros((numTimePts, dim, numEvalPaths))
	U = np.zeros((numTimePts - 1, numEvalPaths))
	
	# choose initial value
	# X[0, 0, :] = 2.7
	X[0, 0, :] = 0.25
	
	exctX = np.zeros((numTimePts, dim, numEvalPaths))
	# exctX[0, 0, :] = 2.7
	exctX[0, 0, :] = 0.25

	for i in range(0, numTimePts - 1):
					
		####################################################
		#
		# simulate using approximative value function
		#
		####################################################

		# the same as above using the approximation to the value function
		
		tmpVec = np.zeros((numControls, numEvalPaths))
		
		sampleData = np.zeros(numEvalPaths)
		sampleData = sampleSpace[sampleIdx[i, :]]
		# print(sampleData)
		
		
		for j in range(0, numControls):
			
			a = np.zeros(numEvalPaths)
			b = np.zeros(numEvalPaths)
			a, b = mySdeCoeffsFunc([X[i, 0, :]], controlVec[j], r, mu, sigma)
			
			f0[j, 0, :] = X[i, 0, :] + a * dt + sqrtDt * b * 1
			f1[j, 0, :] = X[i, 0, :] + a * dt + sqrtDt * b * (-1)
				
			tmpVec[j, :] = prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :]], numEvalPaths)) \
				+ prob * (modelFunc.evaluate(i + 1, [f1[j, 0, :]], numEvalPaths))
					
		# determine optimal control
		
		
		U[i, :] = controlVec[tmpVec.argmax(axis = 0)]
		
		for l in range(0, numEvalPaths):
			
			a, b = mySdeCoeffsFunc([X[i, 0, l]], U[i, l], r, mu, sigma)
			X[i + 1, 0, l] = X[i, 0, l] + a * dt + sqrtDt * sampleData[l] * b

			a, b = mySdeCoeffsFunc([exctX[i, 0, l]], (mu - r) / ((sigma ** 2) * (1 - alpha)), r, mu, sigma)
			exctX[i + 1, 0, l] = exctX[i, 0, l] + a * dt + sqrtDt * sampleData[l] * b

	##########################################################################################################

	
	# ### produce plot showing a variety of paths
	
	timeVec = dt * np.arange(0, numTimePts)
	
	fig1 = plt.figure(1)
	plt.subplot(211)
	for l in range(0, numEvalPaths):
		# plt.plot(timeVec, exctX[:, 0, l], color = colors[l])
		plt.plot(timeVec, X[:, 0, l], color = colors[l])
	plt.title('Solution paths over time (using approximate controls)')
	plt.xlabel('Time')
	plt.ylabel('Wealth')
	plt.xlim(0, T)
	# plt.ylim(2, 4.0)
	plt.ylim(0, 0.5)
	plt.grid(True)
	
	plt.subplot(212)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec, exctX[:, 0, l], color = colors[l])

	plt.title('Solution paths over time (using exact controls)')
	plt.xlabel('Time')
	plt.ylabel('Wealth')
	plt.xlim(0, T)
	# plt.ylim(2, 4.0)
	plt.ylim(0, 0.5)
	plt.grid(True)

	plt.subplots_adjust(hspace = 0.8)

	
	plt.savefig(path2Dir + '/figures/' + 'solutionPaths.png')
	plt.close(fig1)
	

	
	# ### produce a plot showing the control paths over time
	
	fig2 = plt.figure(2)
	exctCtrl = (mu - r) / ((sigma ** 2) * (1 - alpha))
	for l in range(0, numEvalPaths):
		plt.plot(dt * np.arange(0, numTimePts - 1), U[:, l], color = colors[l])
	plt.plot(dt * np.arange(0, numTimePts - 1), exctCtrl * np.ones(numTimePts - 1), '--r', linewidth = 0.6, label = 'Exact control path')
	plt.title('Control paths over time')
	plt.xlabel('Time')
	# plt.ylabel('Investment')
	plt.ylim(controlVec[0], 1.1)
	plt.legend(loc = 'lower right')
	plt.savefig(path2Dir + '/figures/' + 'controlPaths.png')
	
	plt.close(fig2)

		
	# ### produce plots showing/depicting the value function at every time point
	
	
	M = 100

	lInfErrors = np.zeros(numTimePts)
	for i in range(0, numTimePts):
		x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
		x2 = np.linspace(2, 6, M)
			
		v1 = modelFunc.evaluate(i, [x1], M)
		v2 = modelFunc.evaluate(i, [x2], M)
		
		vExct1 = exactValFunc([x1], timeVec[i], r, mu, sigma, alpha, T)
		vExct2 = exactValFunc([x2], timeVec[i], r, mu, sigma, alpha, T)
		
		# measure approximation on quality on subdomain by means of l^inf error
		lInfErrors[i] = np.linalg.norm(v2 - vExct2, np.inf)
		
		fig = plt.figure()
			# ax = fig.gca(projection = '3d')
			# ax.plot_surface(xx1, xx2, v1v1)
		
		plt.plot(x1, v1, label = 'Approximate value function')
		plt.plot(x1, vExct1, '--r', linewidth = 0.6, label = 'Exact value function')
			# plt.ylim([0, 1.5])

		plt.legend(loc = 'lower right')
		plt.title('Plot of value function at time t = t_' + str(i))
				
		plt.savefig(path2Dir + '/figures/' + 'valFuncAtTime_t' + str(i) + '.png')
		
		plt.close(fig)

	# fig3 = plt.figure(3)
	# plt.plot(timeVec, lInfErrors)
	# plt.title('Evolution of '  + r'$l^{\infty}$' + ' approximation error')
	# plt.xlabel('Time')
	# plt.show()
	
	
