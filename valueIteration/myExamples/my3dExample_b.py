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
# parameter settings
########################################################################################################################	

maxLinSpeed = 10.0
maxAngSpeed = 9.0

numLinSpeedCtrls = 11
numAngSpeedCtrls = 21
numControls = numLinSpeedCtrls * numAngSpeedCtrls

linSpeedCtrlArr = np.linspace(0, 1, numLinSpeedCtrls)
angSpeedCtrlArr = np.linspace(-1, 1, numAngSpeedCtrls)

controlSpace = []
for j in range(0, numAngSpeedCtrls):
	for i in range(0, numLinSpeedCtrls):
		controlSpace.append(np.array([angSpeedCtrlArr[j], linSpeedCtrlArr[i]]))
controlSpace = np.asarray(controlSpace).transpose()

t0 = 0
T = 1.5
numTimeIntervals = 30

nrmExpnt = 2
costConst = 1.0

sigma1 = 50 * 1e-2	
sigma2 = 50 * 1e-2	
sigma3 = 50 * 1e-2	

boundingBox = []
boundingBox.append((-1.0, 10.0))			# sampleBox.append((-1.0, 7.5))
boundingBox.append((-4.0, 4.0))			# sampleBox.append((-2.0, 2.0))
boundingBox.append((-np.pi - 1, np.pi + 1))

sampleBox = []
sampleBox.append((-1.0, 10.0))			# sampleBox.append((-1.0, 7.5))
sampleBox.append((-4.0, 4.0))			# sampleBox.append((-2.0, 2.0))
sampleBox.append((-np.pi - 1, np.pi + 1))

# ### chance contraints

delta = 0.25# 1e-3		# radius of acceptance region
eps = 1e-2			# 1 - eps gives the probability with which the terminal position should lie inside the ball 
					# with radius delta

########################################################################################################################	
# definition of functions
########################################################################################################################	

def myPath(numTimePts):
	"""
		@param[in] ### numTimePts ### number of time points in the discretisation of the time interval
		
		@return ### retVal ### matrix of containing columnwise the position of the way points of the path
	"""	
	retVal = np.zeros((2, numTimePts))
	
	#################################			
	# ### straight line ###
	#################################
	
	# retVal[:, 0] = np.zeros(2)
	# retVal[:, -1] = np.array([6, 0])
		
	# for i in range(1, numTimePts - 1):
		# retVal[:, i] = retVal[:, 0] + (i / (numTimePts - 1)) * (retVal[:, -1] - retVal[:, 0])
	
	#################################
	# ### sinus curve 1 ###
	#################################
	
	# retVal[0, :] = (2 * np.pi / (numTimePts - 1)) * np.arange(0, numTimePts)
	
	# for i in range(1, numTimePts):
		# retVal[1, i] = 2 * np.sin((np.pi) * (i / (numTimePts - 1)))
		
	#################################
	# ### sinus curve 2 ###
	#################################
		
	retVal[0, :] = (2 * np.pi / (numTimePts - 1)) * np.arange(0, numTimePts)

	for i in range(1, numTimePts):
		retVal[1, i] = np.sin((2 * np.pi) * (i / (numTimePts - 1)))
		
	# shorten the path ...
	for i in range(24, numTimePts):
		retVal[0, i] = retVal[0, 24]
		retVal[1, i] = retVal[1, 24]
		
	return retVal

def terminalCostFunc(xList, currWayPt, lam = 0.0):
	"""
		@param[in] ### xList ### list containing state data points
		@param[in] ### currWayPt[0] ### current way point
		
		@return ### retVal ### cost arising at final time given the states in xList
	"""
	
	retVal = np.zeros(xList[0].shape)
	
	retVal += costConst * np.minimum(np.sqrt((xList[0] - currWayPt[0]) ** 2 + (xList[1] - currWayPt[1]) ** 2), 1.0) ** 2
	
	# ### take into account probability constraints here
	trmnlWayPt = np.asarray([2 * np.pi, 0])
	retVal += lam * (1 - eps - ((xList[0] - trmnlWayPt[0]) ** 2 + (xList[1] - trmnlWayPt[1]) ** 2 < delta ** 2))
		
	retVal -= 3
		
	return retVal
	
def stageCostFunc(xList, currWayPt, currTime, u = np.zeros(2)):
	"""
		@param[in] ### xList ### as above
		@param[in] ### currWayPt ### as above
		
		@return ### retVal ### stage cost arising given the states in xList
	"""

	retVal = np.zeros(xList[0].shape)
	retVal += costConst * np.minimum(np.sqrt((xList[0] - currWayPt[0]) ** 2 + (xList[1] - currWayPt[1]) ** 2), 1.0) ** 2
	
	# retVal += 1 * 1e-1 * (u[0] ** 2 + u[1] ** 2)
	
	return retVal
	
def mySdeCoeffsFunc(xList, u, sig1, sig2, sig3):
	"""
		@param[in] ### xList ### as above
		@param[in] ### u ### control vector or control vectors ... where: 
			u[1, 0] = speed control
			u[0, 0] = steering control
		@param[in] ### sig1 ### diffusion parameter
		@param[in] ### sig2 ### diffusion parameter
		@param[in] ### sig3 ### diffusion parameter
		@param[in] ### maxSpeed ### maximal speed the vehicle can reach
		
		@return ### retVal1, retVal2 ### coefficients of the sde given the input parameters
	"""	
	
	retVal1 = np.zeros((3, xList[0].shape[0]))
	retVal2 = np.zeros((3, xList[0].shape[0]))
	
	retVal1[0, :] = u[1, :] * maxLinSpeed * np.cos(xList[2])
	retVal1[1, :] = u[1, :] * maxLinSpeed * np.sin(xList[2])
	retVal1[2, :] = u[0, :] * maxAngSpeed
	
	retVal2[0, :] = sig1
	retVal2[1, :] = sig2
	retVal2[2, :] = sig3
	
	return retVal1, retVal2

########################################################################################################################	
# problem solving function
########################################################################################################################	

def my3dExample_b(L, rank, degrs, signature, subSignature, path2SubSignDir, path2FigDir,\
						path2ModDir, path2PerfDataDir, lam = 0.0):
	
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
		@param[in] ### lam ### value of the dual variable in the probability constraint approach
		
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

	path = np.zeros((2, numTimePts))
	path = myPath(numTimePts)

			# fig = plt.figure()
			# plt.plot(path[0, :], path[1, :])
			# plt.show()	
			# asd
	

	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 20
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
	sdeParamsDict['maxLinSpeed'] = maxLinSpeed
	sdeParamsDict['maxAngSpeed'] = maxAngSpeed
	sdeParamsDict['sigma1'] = sigma1
	sdeParamsDict['sigma2'] = sigma2
	sdeParamsDict['sigma3'] = sigma3
	
	# parameters of the control problem
	ctrlProbDict = {}
	ctrlProbDict['numTimePts'] = numTimePts
	ctrlProbDict['startTime'] = t0
	ctrlProbDict['endTime'] = T
	ctrlProbDict['numLinSpeedCtrls'] = numLinSpeedCtrls
	ctrlProbDict['numAngSpeedCtrls'] = numAngSpeedCtrls
	ctrlProbDict['costConst'] = costConst
	ctrlProbDict['nrmExpnt'] = nrmExpnt
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
	valIterParamsDict['lambda'] = lam
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
		lambda x : terminalCostFunc(x, path[:, -1], lam))
			
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
			
			controlData = np.zeros(L)
			
			sampleIdx = np.zeros(L)
			sampleIdx = np.random.choice([0, 1, 2, 3, 4, 5], L, replace = True)

			sampleData = np.zeros((2, L))
			sampleData = sampleSpace[sampleIdx].transpose()

			tmpVec = np.zeros((numControls, L))
			
			for j in range(0, numControls):
									
				a1 = np.zeros((dim, L))
				b1 = np.zeros((dim, L))
				
				stgCost = np.zeros(L)
			
				a1, b1 = mySdeCoeffsFunc([xData[0, :], xData[1, :], xData[2, :]], controlSpace[:, j].reshape(2, -1), sigma1, sigma2, sigma3)

				f0[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([1, 0, 0]).reshape(3, 1)
				f1[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([-1, 0, 0]).reshape(3, 1)
				f2[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 1, 0]).reshape(3, 1)
				f3[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, -1, 0]).reshape(3, 1)
				f4[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, 1]).reshape(3, 1)
				f5[j, :, :] = xData + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, -1]).reshape(3, 1)
				
				
				# take care of periodic boundary conditions for the steering angle right here
				f0[j, 2, :] = np.mod(f0[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				f1[j, 2, :] = np.mod(f1[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				f2[j, 2, :] = np.mod(f2[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				f3[j, 2, :] = np.mod(f3[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				f4[j, 2, :] = np.mod(f4[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				f5[j, 2, :] = np.mod(f5[j, 2, :] + np.pi, 2 * np.pi) - np.pi
				
				stgCost = stageCostFunc([xData[0, :], xData[1, :], xData[2, :]], path[:, i], i * dt, controlSpace[:, j])
								
				tmpVec[j, :] = stgCost * dt \
					+ prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :], f0[j, 2, :]], L) \
								+ modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :], f1[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :], f2[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :], f3[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f4[j, 0, :], f4[j, 1, :], f4[j, 2, :]], L) + \
								+ modelFunc.evaluate(i + 1, [f5[j, 0, :], f5[j, 1, :], f5[j, 2, :]], L))
	
			################################################################################
			
			# compute target values
			yData = tmpVec.min(axis = 0)

					# print(min(yData))
					# print(max(yData))
					# print(np.mean(yData))
					# input(' ... ')
			
			# start optimisation process
					
			costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList = \
				optimiser.myRbfCgOptimiser(L, xData, yData, modelFunc.getPartialModelFunc(i), \
					path2ModDir, 'modelFunc_t' + str(i), \
					maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
					warmUp = False, verbose = False, write2File = False)

					# fig = plt.figure()	
					# ax = fig.add_subplot(111, projection = '3d')
			

					# M = 100

					# x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
					# x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)
			
					# vv = np.zeros((M, M))
					# [xx1, xx2] = np.meshgrid(x1, x2)	
					# vv = modelFunc.evaluate3d_3(i, [np.ones(M)], [xx1, xx2], M)
				
					# ax.plot_surface(xx1, xx2, vv)
				
					# plt.show()
					
		depositeIntermediateTrainingResults(modelFunc, path2ModDir, k + 1)

		
	####################################################################################################################
	# ### post processing
	####################################################################################################################

	# create setup file
	logging.info('> create setup file')
	createSetupFile(sdeParamsDict, ctrlProbDict, valIterParamsDict, alsParamsDict, cgParamsDict, cpdParamsDict, path2SubSignDir)

def simulateFromFile_3d_b(sessId, subSessId, iterId, pC = False, lam = 0.0):
	
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

	if pC:
		path2Dir = '../../results/valueIteration/3d_b_pC_' + sign + '/3d_b_pC_' + sign + '_' + subSign
	else:
		path2Dir = '../../results/valueIteration/3d_b_' + sign + '/3d_b_' + sign + '_' + subSign
	
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

	path = np.zeros((2, numTimePts))
	path = myPath(numTimePts)
	
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
		lambda x : terminalCostFunc(x, path[:, -1], lam))

	modelFunc.readParamsFromFile(path2Dir + '/model' + '/iter_' + str(iterId), 'modelData_t')

	for i in range(0, numTimePts - 1):
		modelFunc.partialModelFuncList[i].setBoundingBox(boundingBox)


			
	#################################################################
	# ### provide data structures needed in the simulation process
	#################################################################	
	
	numEvalPaths = 100
	
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
	U = np.zeros((numTimePts - 1, 2, numEvalPaths))
	
	# choose initial value
	X[0, 0, :] = 0.0
	X[0, 1, :] = 0.0
	X[0, 2, :] = 1.0
	# X[0, 2, :] = 0.0

	for i in range(0, numTimePts - 1):
					
		####################################################
		#
		# simulate using approximative value function
		#
		####################################################
	
		tmpVec = np.zeros((numControls, numEvalPaths))
		sampleData = sampleSpace[sampleIdx[i, :]].transpose()

		for j in range(0, numControls):

			stgCost = np.zeros(numEvalPaths)
		
			a1, b1 = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :], X[i, 2, :]], controlSpace[:, j].reshape(2, -1), sigma1, sigma2, sigma3)

			f0[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([1, 0, 0]).reshape(3, 1)
			f1[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([-1, 0, 0]).reshape(3, 1)
			f2[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 1, 0]).reshape(3, 1)
			f3[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, -1, 0]).reshape(3, 1)
			f4[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, 1]).reshape(3, 1)
			f5[j, :, :] = X[i, :, :] + a1 * dt + sqrtDt * b1 * np.asarray([0, 0, -1]).reshape(3, 1)
			
			# take care of periodic boundary conditions for the steering angle right here
			f0[j, 2, :] = np.mod(f0[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			f1[j, 2, :] = np.mod(f1[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			f2[j, 2, :] = np.mod(f2[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			f3[j, 2, :] = np.mod(f3[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			f4[j, 2, :] = np.mod(f4[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			f5[j, 2, :] = np.mod(f5[j, 2, :] + np.pi, 2 * np.pi) - np.pi
			
			stgCost = stageCostFunc([X[i, 0, :], X[i, 1, :], X[i, 2, :]], path[:, i], i * dt, controlSpace[:, j])

			tmpVec[j, :] = stgCost * dt \
				+ prob * (modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :], f0[j, 2, :]], numEvalPaths) \
							+ modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :], f1[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :], f2[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :], f3[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f4[j, 0, :], f4[j, 1, :], f4[j, 2, :]], numEvalPaths) + \
							+ modelFunc.evaluate(i + 1, [f5[j, 0, :], f5[j, 1, :], f5[j, 2, :]], numEvalPaths))

		# determine optimal control
		U[i, :, :] = controlSpace[:, tmpVec.argmin(axis = 0)]

		a = np.zeros((dim, numEvalPaths))
		b = np.zeros((dim, numEvalPaths))
		
		# ... and finally the subsequent state
		a, b = mySdeCoeffsFunc([X[i, 0, :], X[i, 1, :], X[i, 2, :]], U[i, :, :], sigma1, sigma2, sigma3)
				
		tmpMtrx = np.zeros(b.shape)

		for row in range(0, dim):
			tmpMtrx[row, :] = sampleData[row, :] * b[row, :]

		X[i + 1, :, :] = X[i, :, :] + a * dt + sqrtDt * tmpMtrx
		X[i + 1, 2, :] = np.mod(X[i + 1, 2, :] + np.pi, 2 * np.pi) - np.pi

	##########################################################################################################


	# ### produce plot showing a variety of paths
	
	fig1 = plt.figure(1)
	for l in range(0, numEvalPaths):
		plt.plot(X[:, 0, l], X[:, 1, l])
	
	plt.plot(path[0, :], path[1, :], 'ro', linewidth = 0.25)
	plt.title('Position of the vehicle')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.xlim(boundingBox[0][0], boundingBox[0][1])
	plt.ylim(boundingBox[1][0], boundingBox[1][1])
	plt.grid(True)
	
	plt.savefig(path2Dir + '/figures/' + 'positionPaths.png')
	plt.close(fig1)

	timeVec = dt * np.arange(0, numTimePts)

	fig2 = plt.figure(2)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec, X[:, 2, l])
	
	plt.ylim([-np.pi, np.pi])
	plt.title('Orientation of the vehicle')
	plt.xlabel('Time')
	plt.ylabel('Angle [rad]')

	plt.savefig(path2Dir + '/figures/' + 'orientationPaths.png')
	plt.close(fig2)
	
	fig3 = plt.figure(3)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec[0 : numTimePts - 1], U[:, 0, l])

	plt.ylim([-1.1, 1.1])
	plt.title('Angular velocity control')
	plt.xlabel('Time')
	
	plt.savefig(path2Dir + '/figures/' + 'angVelControlPaths.png')
	plt.close(fig3)
	
	fig4 = plt.figure(4)
	for l in range(0, numEvalPaths):
		plt.plot(timeVec[0 : numTimePts - 1], U[:, 1, l])
	
	plt.ylim([-0.1, 1.1])
	plt.title('Linear velocity control')
	plt.xlabel('Time')

	plt.savefig(path2Dir + '/figures/' + 'linVelControlPaths.png')
	plt.close(fig4)

	fig5 = plt.figure(5)
	dists = np.zeros((numTimePts, numEvalPaths))
	for l in range(0, numEvalPaths):
		for i in range(0, numTimePts):
			dists[i, l] = np.linalg.norm(X[i, 0 : 2, l].transpose() - path[:, i], 2) ** 2
		
	
		plt.plot(timeVec, dists[:, l])
		
	plt.title('Distance to the reference path')
	plt.xlabel('Time')
	
	plt.savefig(path2Dir + '/figures/' + 'dist2RefPath.png')
	plt.close(fig5)
	

	#########################################################################################
	# ### visualisation of the value function for fixed heading ###
	
	"""
	if pC:
	
		M = 100

		x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
		x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)
		

		for i in range(0, numTimePts):
		
			vv = np.zeros((M, M))
			
			[xx1, xx2] = np.meshgrid(x1, x2)
			vv = modelFunc.evaluate(i, [np.ravel(xx1), np.ravel(xx2), np.ones(M ** 2)], M ** 2).reshape(xx1.shape)
			
			fig = plt.figure()
			
			plt.contourf(vv)
			plt.colorbar()
			plt.title('Contour plot of value function \n for fixed heading at time t = t_' + str(i))
			plt.xticks([0, 99], [-1, 7.5])
			plt.yticks([0, 99], [-2, 2])
			plt.xlabel('x1')
			plt.ylabel('x2')
			
			plt.savefig(path2Dir + '/figures/' + 'valFuncAtTime_t' + str(i) + '.png')			
			# plt.show()
			
			plt.close(fig)
	"""
	trmnlWayPt = np.asarray([2 * np.pi, 0])

	prob = np.sum((X[numTimePts - 1, 0, :] - trmnlWayPt[0]) ** 2 + (X[numTimePts - 1, 1, :] - trmnlWayPt[1]) ** 2 < delta ** 2) / numEvalPaths
	print(prob)

	if pC:
		"""
			compute estimator of probability/expected value here
		"""
		
		cnstrnt = np.sum(1 - eps - ((X[numTimePts - 1, 0, :] - trmnlWayPt[0]) ** 2 + (X[numTimePts - 1, 1, :] - trmnlWayPt[1]) ** 2 < delta ** 2)) / numEvalPaths
		
		print(cnstrnt)
			
		return cnstrnt
