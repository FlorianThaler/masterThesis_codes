"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements a 3d 
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
# initialisations
########################################################################################################################	

boundingBox = []
# boundingBox.append((-1.0, 2.0))
# boundingBox.append((-1.0, 2.0))
# boundingBox.append((-1.0, 2.0))	
boundingBox.append((0.0, 1.0))
boundingBox.append((0.0, 1.0))
boundingBox.append((0.0, 1.0))	

sampleBox = []
sampleBox.append((-1.0, 2.0))
sampleBox.append((-1.0, 2.0))
sampleBox.append((-1.0, 2.0))

########################################################################################################################	
# definition of functions
########################################################################################################################	


def terminalCostFunc(xList):
	"""
		@param[in] ### xList ### list containing state data points
		
		@return ### retVal ### cost arising at final time given the states in xList
	"""
	
	retVal = np.zeros(xList[0].shape)
	retVal += xList[2]
	
	return retVal
	
def stageCostFunc(xList):
	"""
		@param[in] ### xList ### as above
		
		@return ### retVal ### stage cost arising given the states in xList
	"""

	retVal = np.zeros(xList[0].shape)						
		
	return retVal
	
def mySdeCoeffsFunc(xList, u, k1, k2, k3, sig1, sig2, sig3):
	"""
		@param[in] ### xList ### as above
		@param[in] ### u ### control or control vector in the interval [0, 1]
		@param[in] ### k1 ### velocity constant 
		@param[in] ### k2 ### velocity constant 
		@param[in] ### k3 ### velocity constant 
		@param[in] ### sig1 ### diffusion parameter
		@param[in] ### sig2 ### diffusion parameter
		@param[in] ### sig3 ### diffusion parameter
		
		@return ### retVal1, retVal2 ### coefficients of the sde given the input parameters
		
	"""	
	
	
	retVal1 = np.zeros((3, xList[0].shape[0]))
	retVal2 = np.zeros((3, xList[0].shape[0]))
	
	retVal1[0, :] = -u * (k1 * xList[0] - k2 * xList[1])
	retVal1[1, :] = u * (k1 * xList[0] - k2 * xList[1]) - (1 - u) * k3 * xList[1]
	retVal1[2, :] = (1 - u) * k3 * xList[1]
	
	retVal2[0, :] = sig1
	retVal2[1, :] = sig2
	retVal2[2, :] = sig3
	
	return retVal1, retVal2

########################################################################################################################	
# parameter settings
########################################################################################################################	

numControls = 21
controlVec = np.linspace(0, 1, numControls)
	
t0 = 0
T = 4.0
numTimeIntervals = 40

########################################################################################################################	
# problem solving function
########################################################################################################################	


def my3dExample_c(L, rank, degrs, signature, subSignature, path2SubSignDir, path2FigDir,\
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

	dim = 3
	
	# numTimeIntervals is a global variable defined before ...
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	sampleSpace = np.array([np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), \
				np.asarray([0, 0, 1]), np.asarray([0, 0, -1])])

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

	k1 = 1
	k2 = 10
	k3 = 1
	
	sigma1 = 3 * 1e-2
	sigma2 = 3 * 1e-2
	sigma3 = 3 * 1e-2

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
	sdeParamsDict['k1'] = k1
	sdeParamsDict['k2'] = k2
	sdeParamsDict['k3'] = k3
	sdeParamsDict['sigma1'] = sigma1
	sdeParamsDict['sigma2'] = sigma2
	sdeParamsDict['sigma3'] = sigma3
	
	# parameters of the control problem
	ctrlProbDict = {}
	ctrlProbDict['numTimePts'] = numTimePts
	ctrlProbDict['startTime'] = t0
	ctrlProbDict['endTime'] = T
	ctrlProbDict['numControls'] = numControls
		# ctrlProbDict['controlLwrBnd'] = controlVec[0]
		# ctrlProbDict['controlUprBnd'] = controlVec[-1]
	
	# value iteration parameters
	valIterParamsDict = {}
	valIterParamsDict['sampleBox_x1_left'] = sampleBox[0][0]
	valIterParamsDict['sampleBox_x1_right'] = sampleBox[0][1]
	valIterParamsDict['sampleBox_x2_left'] = sampleBox[1][0]
	valIterParamsDict['sampleBox_x2_right'] = sampleBox[1][1]
	valIterParamsDict['sampleBox_x3_left'] = sampleBox[2][0]
	valIterParamsDict['sampleBox_x3_right'] = sampleBox[2][1]
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
	alsParamsDict['boundingBox_x3_left'] = boundingBox[2][0]
	alsParamsDict['boundingBox_x3_right'] = boundingBox[2][1]
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

	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, \
		lambda x : terminalCostFunc(x))
			
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
		tmpXData[2, :] = np.random.uniform(sampleBox[2][0], sampleBox[2][1], L)
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
			
			################################################################################
			
			f0 = np.zeros((numControls, dim, L))
			f1 = np.zeros((numControls, dim, L))
			f2 = np.zeros((numControls, dim, L))
			f3 = np.zeros((numControls, dim, L))
			f4 = np.zeros((numControls, dim, L))
			f5 = np.zeros((numControls, dim, L))
					
			sampleIdx = np.zeros(L)
			sampleIdx = np.random.choice([0, 1, 2, 3, 4, 5], L, replace = True)

			sampleData = np.zeros((2, L))
			sampleData = sampleSpace[sampleIdx].transpose()

			tmpVec = np.zeros((numControls, L))
			
			for j in range(0, numControls):
									
				a1 = np.zeros((dim, L))
				b1 = np.zeros((dim, L))
						
				a1, b1 = mySdeCoeffsFunc([xData[0, :], xData[1, :], xData[2, :]], controlVec[j], k1, k2, k3, sigma1, sigma2, sigma3)

				f0[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([1, 0, 0]).reshape(3, 1)
				f1[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([-1, 0, 0]).reshape(3, 1)
				f2[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 1, 0]).reshape(3, 1)
				f3[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, -1, 0]).reshape(3, 1)
				f4[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, 1]).reshape(3, 1)
				f5[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, -1]).reshape(3, 1)
				
				tmpVec[j, :] = prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :], f0[j, 2, :]], L) \
								+ modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :], f1[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :], f2[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :], f3[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f4[j, 0, :], f4[j, 1, :], f4[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f5[j, 0, :], f5[j, 1, :], f5[j, 2, :]], L))
	
			################################################################################
			
			# compute target values
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

def simulateFromFile_3d_c(sessId, subSessId, iterId):
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

	path2Dir = '../../results/valueIteration/3d_c_' + sign + '/3d_c_' + sign + '_' + subSign
	
	#################################################################
	# ### initialisations
	#################################################################

	####################### TEMPORAL AND SPATIAL INITIALISATIONS ########################
	dim = 3
		
	numTimePts = numTimeIntervals + 1
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)
	
	sampleSpace = np.array([np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), \
				np.asarray([0, 0, 1]), np.asarray([0, 0, -1])])
	####################### MODEL PARAMETERS ########################

	prob = 1.0 / (2 * dim)

	k1 = 1
	k2 = 10
	k3 = 1

	sigma1 = 3 * 1e-2
	sigma2 = 3 * 1e-2
	sigma3 = 3 * 1e-2
	
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
	
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, \
		lambda x : terminalCostFunc(x))

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
	f4 = np.zeros((numControls, dim, numEvalPaths))
	f5 = np.zeros((numControls, dim, numEvalPaths))

	sampleIdx = np.zeros((numTimePts, numEvalPaths))
	sampleIdx = np.random.choice([0, 1, 2, 3, 4, 5], (numTimePts, numEvalPaths), replace = True)
		
	sampleData = np.zeros((dim, numEvalPaths))
	
	X = np.zeros((numTimePts, dim, numEvalPaths))
	U = np.zeros((numTimePts - 1, numEvalPaths))
	
	# choose initial value
	X[0, 0, :] = 1.0
	X[0, 1, :] = 0.0
	X[0, 2, :] = 0.0

	for i in range(0, numTimePts - 1):
					
		####################################################
		#
		# simulate using approximative value function
		#
		####################################################
	
		tmpVec = np.zeros((numControls, numEvalPaths))
		sampleData = sampleSpace[sampleIdx[i, :]].transpose()

		for j in range(0, numControls):

		
			a1, b1 = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :], X[i, 2, :]], controlVec[j], k1, k2, k3, sigma1, sigma2, sigma3)

			f0[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([1, 0, 0]).reshape(3, 1)
			f1[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([-1, 0, 0]).reshape(3, 1)
			f2[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 1, 0]).reshape(3, 1)
			f3[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, -1, 0]).reshape(3, 1)
			f4[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, 1]).reshape(3, 1)
			f5[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, -1]).reshape(3, 1)
					
			tmpVec[j, :] = prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :], f0[j, 2, :]], numEvalPaths) \
							+ modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :], f1[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :], f2[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :], f3[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f4[j, 0, :], f4[j, 1, :], f4[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f5[j, 0, :], f5[j, 1, :], f5[j, 2, :]], numEvalPaths))

		# determine optimal control
		U[i, :] = controlVec[tmpVec.argmax(axis = 0)]

		a = np.zeros((dim, numEvalPaths))
		b = np.zeros((dim, numEvalPaths))
		
		# ... and finally the subsequent state
		a, b = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :], X[i, 2, :]], U[i, :], k1, k2, k3, sigma1, sigma2, sigma3)
				
		tmpMtrx = np.zeros(b.shape)

		for row in range(0, dim):
			tmpMtrx[row, :] = sampleData[row, :] * b[row, :]

		X[i + 1, :, :] = X[i, :, :] + a * dt + sqrtDt * tmpMtrx

	##########################################################################################################

	# ### produce plot showing a variety of paths
	
	timeVec = dt * np.arange(0, numTimePts)
	
	fig1 = plt.figure(1)
	plt.subplot(311)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec, X[:, 0, l])
	plt.title('Mole fraction of substance A')
	plt.xlabel('Time')
	plt.xlim(0, T)
	plt.ylim(0.6, 1.025)
	plt.yticks(np.array([0.6, 0.8, 1.0]))
	plt.grid(True)
	
	plt.subplot(312)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec, X[:, 1, l])
	plt.title('Mole fraction of substance B')
	plt.xlabel('Time')
	plt.xlim(0, T)
	plt.ylim(-0.025, 0.3)
	plt.yticks(np.array([0.0, 0.1, 0.2]))
	plt.grid(True)
	
	plt.subplot(313)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec, X[:, 2, l])
	plt.title('Mole fraction of substance C')
	plt.xlabel('Time')
	plt.xlim(0, T)
	plt.ylim(-0.025, 0.3)
	plt.yticks(np.array([0.0, 0.1, 0.2]))
	plt.grid(True)
	
	plt.subplots_adjust(hspace = 0.9)
	
	plt.savefig(path2Dir + '/figures/' + 'solutionPaths.png')
	plt.close(fig1)

	
	fig2 = plt.figure(2)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec[0 : numTimePts - 1], U[:, l])
	
	plt.ylim([-0.1, 1.1])
	plt.title('Control paths over time')
	plt.xlabel('Time')

	plt.savefig(path2Dir + '/figures/' + 'controlPaths.png')
	plt.close(fig2)
	
	###########################################################################################################
	# ### visualisation of the value function for fixed mole fraction of the first substance ###
	
	M = 100

	x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)
	x3 = np.linspace(boundingBox[2][0], boundingBox[2][1], M)
	
	for i in range(0, numTimePts):
	
		vv = np.zeros((M, M))
		
		[xx2, xx3] = np.meshgrid(x2, x3)	
		vv = modelFunc.evaluate(i, [np.ones(M ** 2), np.ravel(xx2), np.ravel(xx3)], M ** 2).reshape(xx2.shape)
		
		fig = plt.figure()
		
		plt.contourf(vv)
			# plt.plot([33, 66], [33, 33], 'r')
			# plt.plot([66, 66], [33, 66], 'r')
			# plt.plot([66, 33], [66, 66], 'r')
			# plt.plot([33, 33], [66, 33], 'r')
		
		plt.colorbar()
		plt.title('Contour plot of value function \n for fixed heading at time t = t_' + str(i))
		plt.xticks([0, 99], [boundingBox[1][0], boundingBox[1][1]])
		plt.yticks([0, 99], [boundingBox[2][0], boundingBox[2][1]])
		plt.xlabel('x2')
		plt.ylabel('x3')
		
		plt.savefig(path2Dir + '/figures/' + 'valFuncAtTime_t' + str(i) + '.png')			
		# plt.show()
		
		plt.close(fig)

