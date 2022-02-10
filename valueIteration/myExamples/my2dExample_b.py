########################################################################################################################
# importing stuff
########################################################################################################################	

import numpy as np
import random

import logging

from myValueFunctionApproximation import MyValueFunctionApproximation
from myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc
from myALSOptimiser import *

from matplotlib import pyplot as plt

from misc import *

import time

import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from collections import deque

########################################################################################################################	
# problem description - riccati example or linear quadratic gaussian control problem
########################################################################################################################	



########################################################################################################################	
# definition of function which represents a value function approximation example in 2d state space
########################################################################################################################	

def boxCost(xList, boundingBox, domainBox, cst = 1000):
	
	retVal = np.zeros(xList[0].shape)
	
	tmp1 = np.logical_and((boundingBox[0][0] <= xList[0]), (xList[0] <= boundingBox[0][1]))
	tmp2 = np.logical_and((boundingBox[1][0] <= xList[1]), (xList[1] <= boundingBox[1][1]))
	
	# ... gives the indices of the elements lying inside the bounding box
	mask1 = np.logical_and(tmp1, tmp2)
	
	tmp3 = np.logical_and((domainBox[0][0] <= xList[0]), (xList[0] <= domainBox[0][1]))
	tmp4 = np.logical_and((domainBox[1][0] <= xList[1]), (xList[1] <= domainBox[1][1]))

	# ... gives the indices of the elements lying inside the domain box
	mask2 = np.logical_and(tmp3, tmp4)
	
	retVal[np.logical_and(mask1, np.logical_not(mask2))] = cst
	
	return retVal


def terminalCostFunc(xList, S, boundingBox, domainBox, lam = 0.0):
	"""
		@param[in] ### xList ### list of length two containing data points, where 
			xList[0] corresponds to the x1 data, and xList[1] corresponds to the x2 data
		@param[in] ### S ### 2x2 matrix
	"""
	delta = 0.5
	trg = -0.5
	
	
	
	
	if np.isscalar(xList[0]) == True:
		
		retVal = 0.0
		X = np.zeros((2, 1))
		X[0, 0] = xList[0]
		X[1, 0] = xList[1]
		
		retVal = 0.5 * quadrForm(X, S)
		
		return retVal
		
	else:
	
		retVal = np.zeros(xList[0].shape[0])
	
		X = np.zeros((2, xList[0].shape[0]))
		X[0, :] = xList[0].copy()
		X[1, :] = xList[1].copy()
		
		retVal = 0.5 * quadrForm(X, S)
		
		return retVal



def stageCostFunc(xList, u, Q, R, boundingBox, domainBox):
	
	if np.isscalar(xList[0]) == True:
		retVal = 0.0
		X = np.zeros((2, 1))
		X[0, 0] = xList[0]
		X[1, 0] = xList[1]
		
		retVal = 0.5 * quadrForm(X, Q) + 0.5 * (u ** 2) * R
		
		return retVal
		
	else:
	
		retVal = np.zeros(xList[0].shape[0])
	
		X = np.zeros((2, xList[0].shape[0]))
		X[0, :] = xList[0].copy()
		X[1, :] = xList[1].copy()
		
		retVal = 0.5 * quadrForm(X, Q) + 0.5 * (u ** 2) * R
		
		return retVal

def mySdeCoeffsFunc(xList, u, A, B, sig):
	"""
		@param[in] # A # matrix
		@param[in] # B # vector
		
	"""
	
	pass
	
	retVal1 = np.zeros((2, xList[0].shape[0]))
	retVal2 = np.zeros((2, xList[1].shape[0]))
	
	for i in range(0, xList[0].shape[0]):
		retVal1[:, i] = np.dot(A, np.array([xList[0][i], xList[1][i]])) + B * u
		retVal2[:, i] = sig * np.array([xList[0][i], xList[1][i]])
	
	return retVal1, retVal2

def quadrForm(X, A):
	"""
		assume X, A to be matrices
		
		X contains column per column the state data
		
	"""
	
	return (X.transpose() * np.dot(A, X).transpose()).sum(axis = 1)
	
def my2dValFuncExample_b(L, rank, degrs, signature, subSignature, pathNames):
	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### initialisations
	#################################################################
		
	####################### TEMPORAL AND SPATIAL INITIALISATIONS ########################

	boundingBox = []
	boundingBox.append((-2, 2))
	boundingBox.append((-2, 2))	
	
	domainBox = []
	# domainBox.append((-2, 2))
	# domainBox.append((-2, 2))
	
	numControls = 20
	controlVec = np.linspace(-1, 1, numControls + 1)
	
	# NOTE: it should hold true that  t0 + numTimePts * dt = T
	# -------------
	t0 = 0
	T = 3
	numTimeIntervals = 30
	numTimePts = numTimeIntervals + 1
	# -------------
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)


	sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	

	####################### USER CHECK ########################
	
	print('### check parameters ###')
	print('	> t0 = ' + str(t0))
	print('	> T = ' + str(T))
	print('	> numTimeIntervals = ' + str(numTimeIntervals))
	print('	> numTimePts = ' + str(numTimePts))
	print('	> dt = ' + str(dt))
	print('	> sqrt(dt) = ' + str(sqrtDt))
	print('########################')
	
	# proceed by pressing enter
	input(' ... hit ENTER to continue ...')
	
	####################### MODEL PARAMETERS ########################
	
	R = 10
	Q = np.zeros((2, 2))
	Q[0, 0] = 1
	
	S = np.zeros((2, 2))
	S[0, 0] = 1
	S[1, 1] = 1
	
	# model parameters
	sigma = 0.01
	
	A = np.ones((2, 2))
	A[1, 0] = 0

	B = np.zeros(2)
	B[0] = 0.5
	B[1] = 1
	
		
	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 5
	eta = 1e-5
	epsALS = 1e-2

	
	# fix CG parameters (there are default parameters ... )
	maxNumCGIter = 50
	epsCG = 1e-4
	resNormFrac = 1e-2
	
	
	# fix parameters regarding the value iteration
	maxNumPlcyIter = 5
	numMCIter = L
	# numMCIter = 30
	
	numAddtnlDataPts = -1
	
	# the number of additional data points - generated due to randomly chosen controls - will be determined as
	# function of the iteration number and the minimal exploration rate, which gives the minimal fraction of 
	# numMCIter paths/data points which will be generated additionally.
	minExplRate = 0.1

	# log the exploration factors in a list
	explRateList = []

	# for logging purposes write parameters into a dict ...
	# alsParamsDict = {}
	# alsParamsDict['maxNumALSIter'] = maxNumALSIter
	# alsParamsDict['eta'] = eta
	# alsParamsDict['descentBound'] = epsALS
	
	# cgParamsDict = {}
	# cgParamsDict['maxNumCGIter'] = maxNumCGIter
	# cgParamsDict['residualBound'] = epsCG
	# cgParamsDict['residualFraction'] = resNormFrac
	
	# valIterParamsDict = {}
	# valIterParamsDict['maxNumPlcyIter'] = maxNumPlcyIter
	# valIterParamsDict['numMCIter'] = numMCIter
	# valIterParamsDict['numTimePts'] = numTimePts
	
	# initialise model function
	
	logging.info('------------------------------------------------------------------------------------------')
	logging.info('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')
	
	dim = 2
	# modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, S, boundingBox, domainBox))
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, S, boundingBox, domainBox))
	apprParams = []
	for i in range(0, numTimePts - 1):
		apprParams.append((rank, degrs))		
		
	modelFunc.initialise(apprParams)
	
	totNumParams = 0
	for d in range(0, dim):
		totNumParams += rank * degrs[d]
	
	#################################################################
	# ### exploration phase ###
	#	> explore the domain due to randomly chosen starting points and
	#		randomly chosen controls
	#	> it is necessary a buffer for each time point ... =/
	#################################################################
	
	# introduce replay memory/buffer for each time point

	maxBufferSize = 2 ** 15
	minBufferSize = np.int(0.9 * maxBufferSize)
	
	bufferList = []
	for i in range(0, numTimePts - 1):
		tmpBffr = []
		tmpBffr = deque(maxlen = maxBufferSize)
		bufferList.append(tmpBffr)
		
	
	# fill replay buffer partially with tupels (state, stageCost, terminalCost, newState) arising due to randomly
	# chosen controls, such that each of the buffers contains at least minBufferSize datapoints
	
	for i in range(0, numTimePts - 1):
		
		print('filling buffer ' + str(i))
		
		s = np.zeros(dim)
		sNew = np.zeros(dim)
		
		done = False
		while not done:
			
			s = np.zeros(dim)
			sNew = np.zeros(dim)
			s[0] = np.random.uniform(boundingBox[0][0], boundingBox[0][1])
			s[1] = np.random.uniform(boundingBox[1][0], boundingBox[1][1])
			smpl = -1
			ctrl = -np.inf
			
			# draw a sample
			smpl = np.random.choice([0, 1, 2, 3], 1, replace = True)[0]
			ctrl = np.random.choice(controlVec, 1, replace = True)[0]
			
		
			
			sNew = np.dot(A, s) + sigma * sqrtDt * sampleSpace[smpl] + B * ctrl
			
			# check if sNew is inside the bounding Box
			if boundingBox[0][0] <= sNew[0] <= boundingBox[0][1] and boundingBox[1][0] <= sNew[1] <= boundingBox[1][1]:
				bufferList[i].append((s, ctrl, sNew, 'e'))
			
			done = (len(bufferList[i])) >= minBufferSize

	#################################################################
	# ### initialisation phase ###
	#	> do some ALS iterations, such that the value function at
	#		any time points approximates the terminal cost ...
	#################################################################
	
	"""
	numDataPts4Intln = L
	# just for initialisation purpose
	optimiser0 = MyALSRbfOptimiser(eta = 1e-8, maxNumALSIter = 15, epsALS = 1e-4)
		
	logging.info('### INITIALISATION PHASE ### ')
	for i in range(0, numTimePts - 1):
		# generate some random data in the bounding box
		x0Data = np.zeros((dim, numDataPts4Intln))
		y0Data = np.zeros(numDataPts4Intln)
		
		x0Data[0, :] = np.random.uniform(boundingBox[0][0], boundingBox[0][1], numDataPts4Intln).reshape(1, numDataPts4Intln)
		x0Data[1, :] = np.random.uniform(boundingBox[1][0], boundingBox[1][1], numDataPts4Intln).reshape(1, numDataPts4Intln)
		y0Data = terminalCostFunc([x0Data[0, :], x0Data[1, :]], S, boundingBox, domainBox)
		
		costFuncValList, apprErrList, cgPerformanceList = \
				optimiser0.myRbfCgOptimiser(numDataPts4Intln, x0Data, y0Data, modelFunc.getPartialModelFunc(i), \
					maxNumCGIter = 10, epsCG = 1e-3, resNormFrac = 1e-2, \
					warmUp = False, verbose = False, write2File = True)
	"""
	# initialise object of optimiser class
	optimiser = MyALSRbfOptimiser(eta = eta, maxNumALSIter = maxNumALSIter, epsALS = epsALS)
		
	print('> start policy iteration procedure')	
	
	
	#################################################################
	# ### start policy iteration procedure
	#################################################################
							
	# provide a data structure where certain results of the optimisation procedure will be stored ...
	optimData = np.zeros((maxNumPlcyIter, numTimePts - 1, 3))
	
	stateData = np.zeros((numTimePts, dim, numMCIter))
	controlData = np.zeros((numTimePts - 1, numMCIter))
		
	##########################################################################
	# ... DESPERATION ...
	valFuncMtrcs = np.zeros((numTimePts, 2, 2))

	valFuncMtrcs[-1, :, :] = S.copy()

	for i in range(numTimePts - 2, -1, -1):
			
		W = valFuncMtrcs[i + 1, :, :].copy()
		
		M = R + np.dot(B.transpose(), np.dot(W, B))
		
		invM = 1.0 / M
		
		tmp = np.dot(B.transpose(), np.dot(W, A)).reshape(1, -1)
		valFuncMtrcs[i, :, :] = Q + np.dot(A.transpose(), np.dot(W, A)) - invM * np.dot(tmp.transpose(), tmp)
		
	##########################################################################
	
	prob = 1.0 / (2 * dim)
	
	logging.info('### SIMULATION AND UPDATE PHASE ### ')

	# store also model parameters corresponding to the only initialised model to file.
	depositeIntermediateTrainingResults(modelFunc, pathNames[2], 0)

	eps = 1.0
	epsRed = 1e-4

	gamma = 1.0

	for k in range(0, maxNumPlcyIter):
		
		# explRateList.append(np.maximum(minExplRate, (maxNumPlcyIter - k) / maxNumPlcyIter))
		# numAddtnlDataPts = np.int(np.maximum(minExplRate, (maxNumPlcyIter - k) / maxNumPlcyIter) * numMCIter)
		# numAddtnlDataPts = numMCIter

		# reinitialise variables!
		# initVals = np.zeros((dim, numMCIter))
		# initVals[0, 0 : numMCIter] = np.random.uniform(-1.0, 1.0, numMCIter).reshape(1, numMCIter)
		# initVals[1, 0 : numMCIter] = np.random.uniform(-1.0, 1.0, numMCIter).reshape(1, numMCIter)
		
		# stateData = np.zeros((numTimePts, dim, numMCIter))
		# stateData[0, :, :] = initVals.copy()
	
		# controlData = np.zeros((numTimePts - 1, numMCIter))
		
		# f0 = np.zeros((numControls, dim, numMCIter))
		# f1 = np.zeros((numControls, dim, numMCIter))
		# f2 = np.zeros((numControls, dim, numMCIter))
		# f3 = np.zeros((numControls, dim, numMCIter))
		
		# draw new samples ...
		
		# print('start policy iteration number # ' + str(k) + ' #')

		
		# sampleIdx1 = np.zeros((numTimePts, numMCIter))		
		# sampleIdx1 = np.random.choice([0, 1, 2, 3], (numTimePts, numMCIter), replace = True)
		

		#################################################################
		# ### simulation phase ### 
		#	> simulate data through the dynamics 
		#################################################################
						
			# epsVec = eps * np.ones(numMCIter) - epsRed * np.arange(0, numMCIter)
			# randVec = np.random.uniform(0, 1, numMCIter)
			
			# randControlVec = np.random.randint(0, numControls, numMCIter)
			
			# controlData = (randVec <= epsVec) * randControlVec + not(randVec <= epsVec) * 'erfahrungskontrolle'
		"""
		for l in range(0, numMCIter):
				
			if l % 1000 == 0:
				print(l)
				
			s = np.zeros(dim)
			sNew = np.zeros(dim)
					
			s[0] = np.random.uniform(-1.0, 1.0)
			s[1] = np.random.uniform(-1.0, 1.0)
			
			ctrl = np.inf	
			done = False
			i = 0
			
			while (done == False) and (i < numTimePts - 1):
				
				if np.random.uniform(0, 1) < eps:
					# ctrl = np.random.randint(0, numControls)
					ctrl = np.random.choice(controlVec, 1, replace = True)[0]
					
				else:
			
					tmpVec = np.zeros(numControls)
					
					f0 = np.zeros((numControls, dim))
					f1 = np.zeros((numControls, dim))
					f2 = np.zeros((numControls, dim))
					f3 = np.zeros((numControls, dim))
					
					for j in range(0, numControls):
						a0 = np.zeros(2)
						b0 = np.zeros(2)
						
						a0 = np.dot(A, s)
						b0 = B * controlVec[j]
						
						stgCost = -np.inf
						stgCost = stageCostFunc([s[0], s[1]], controlVec[j], Q, R, boundingBox, domainBox)
						
						f0[j, :] = a0 + b0 + sigma * np.array([1, 0]) * sqrtDt
						f1[j, :] = a0 + b0 + sigma * np.array([-1, 0]) * sqrtDt
						f2[j, :] = a0 + b0 + sigma * np.array([0, 1]) * sqrtDt
						f3[j, :] = a0 + b0 + sigma * np.array([0, -1]) * sqrtDt
						
						tmpVec[j] = prob * (stgCost * dt + gamma * modelFunc.evaluate(i + 1, [f0[j, 0], f0[j, 1]])) \
							+ prob * (stgCost * dt + gamma * modelFunc.evaluate(i + 1, [f1[j, 0], f1[j, 1]])) \
							+ prob * (stgCost * dt + gamma * modelFunc.evaluate(i + 1, [f2[j, 0], f2[j, 1]])) \
							+ prob * (stgCost * dt + gamma * modelFunc.evaluate(i + 1, [f3[j, 0], f3[j, 1]]))
					
					ctrl = controlVec[tmpVec.argmin(axis = 0)]
					
				a = np.zeros(2)
				b = np.zeros(2)
				
				a = np.dot(A, s)
				b = B * ctrl
		
				smpl = np.random.choice([0, 1, 2, 3], 1, replace = True)[0]
				sNew = a + b + sigma * sampleSpace[smpl] * sqrtDt
			
				done = not((boundingBox[0][0] <= sNew[0] <= boundingBox[0][1]) and (boundingBox[1][0] <= sNew[1] <= boundingBox[1][1]))
				bufferList[i].append((s, ctrl, sNew, 'p'))
				
				# reinitialise variables
				s = np.zeros(2)
				s[0] = sNew[0]
				s[1] = sNew[1]
				
				sNew = np.zeros(2)
				
				# reduce exploration rate
				eps = np.maximum(eps - epsRed, 0.1)	
				i += 1
		"""
		"""			
		for i in range(0, numTimePts - 1):
						
			tmpVec = np.zeros((numControls, numMCIter))			
			sampleData1 = np.zeros((dim, numMCIter))
			sampleData1 = sampleSpace[sampleIdx1[i, :]].transpose()


			for j in range(0, numControls):				
								
				f0[j, :, :] = np.dot(A, stateData[i, :, :]) + sigma * np.array([1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f1[j, :, :] = np.dot(A, stateData[i, :, :]) + sigma * np.array([-1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f2[j, :, :] = np.dot(A, stateData[i, :, :]) + sigma * np.array([0, 1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f3[j, :, :] = np.dot(A, stateData[i, :, :]) + sigma * np.array([0, -1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]						
				
				stgCost = np.zeros((dim, numMCIter))
			
				stgCost = stageCostFunc([stateData[i, 0, :], stateData[i, 1, :]], controlVec[j], Q, R, boundingBox, domainBox)
									
				tmpVec[j, :] = prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]]))
							
				# tmpVec[j, :] = prob * (stgCost * dt + 0.5 * quadrForm(f0[j, :, :], valFuncMtrcs[i + 1, :, :])) \
						# + prob * (stgCost * dt + 0.5 * quadrForm(f1[j, :, :], valFuncMtrcs[i + 1, :, :])) \
						# + prob * (stgCost * dt + 0.5 * quadrForm(f2[j, :, :], valFuncMtrcs[i + 1, :, :])) \
						# + prob * (stgCost * dt + 0.5 * quadrForm(f3[j, :, :], valFuncMtrcs[i + 1, :, :]))
							
						
			controlData[i, :] = controlVec[tmpVec.argmin(axis = 0)]
			
			stateData[i + 1, :, :] = np.dot(A, stateData[i, :, :]) \
					+ sigma * sqrtDt * sampleData1 + B.reshape(2, -1) * controlData[i, :]								
	
			#################################################################################
	
			# now write data into the buffer, if proper ...
			
	
			tmp1 = np.zeros(numMCIter, dtype = bool)
			tmp2 = np.zeros(numMCIter, dtype = bool)
			tmp3 = np.zeros(numMCIter, dtype = bool)
			tmp4 = np.zeros(numMCIter, dtype = bool)
						
			tmp1 = np.logical_and(boundingBox[0][0] <= stateData[i + 1, 0, :], stateData[i + 1, 0, :] <= boundingBox[0][1])
			tmp2 = np.logical_and(boundingBox[1][0] <= stateData[i + 1, 1, :], stateData[i + 1, 1, :] <= boundingBox[1][1])	
			tmp3 = np.logical_and(boundingBox[0][0] <= stateData[i, 0, :], stateData[i, 0, :] <= boundingBox[0][1])
			tmp4 = np.logical_and(boundingBox[1][0] <= stateData[i, 1, :], stateData[i, 1, :] <= boundingBox[1][1])
			
			mask1 = np.where(np.logical_and(np.logical_and(tmp1, tmp2), np.logical_and(tmp3, tmp4)))
		
			for index in mask1[0]:
				bufferList[i].append((stateData[i, :, index], controlData[i, index],  stateData[i + 1, :, index], 'p'))
			
			# for index in range(0, numMCIter):			
				# bufferList[i].append((stateData[i, :, index], controlData[i, index],  stateData[i + 1, :, index], 'p'))

			#################################################################################

		#################################################################
		# ### augmentation phase ### 
		#	> create data due to random actions - leads to an quite
		#		good approximation on a larger area
		#	> data points will be immediately written into the buffer
		#		if they lay inside the approximation domain given 
		#		by the bounding box
		#################################################################

		for i in range(0, numTimePts - 1):
	
			s = np.zeros(dim)
			sNew = np.zeros(dim)
			
			explCounter = 0
			while explCounter < numAddtnlDataPts:
				
				s = np.zeros(dim)
				sNew = np.zeros(dim)
				s[0] = np.random.uniform(boundingBox[0][0], boundingBox[0][1])
				s[1] = np.random.uniform(boundingBox[1][0], boundingBox[1][1])
				
				# draw a sample
				smpl = np.random.choice([0, 1, 2, 3], 1, replace = True)[0]
				ctrl = np.random.choice(controlVec, 1, replace = True)[0]
									
				sNew = np.dot(A, s) + sigma * sqrtDt * sampleSpace[smpl] + B * ctrl
				
				# check if sNew is inside the bounding Box
				if boundingBox[0][0] <= sNew[0] <= boundingBox[0][1] and boundingBox[1][0] <= sNew[1] <= boundingBox[1][1]:
					bufferList[i].append((s, ctrl, sNew, 'e'))
					explCounter += 1

			#################################################################################
		"""
		
		# print('curr eps = ' + str(eps))
		
		#################################################################
		# ### update phase ### 
		#################################################################

		for i in range(numTimePts - 2, -1, -1):
		
			#########################################
			
			print('# update - time point t = t_' + str(i))
						
			############################################
			
			
			# numEpochs = 1
			sampleSize = np.minimum(totNumParams * 10, len(bufferList[i]))
			
			
			# for epoch in range(0, numEpochs):
			batch = []
			batch = random.sample(bufferList[i], sampleSize)
			# extract first components and second components of the tupels manually
			
			tmpList1 = []
			tmpList2 = []
			tmpList3 = []
			tmpList4 = []
			
			for elem in batch:
				tmpList1.append(elem[0])
				tmpList2.append(elem[1])
				tmpList3.append(elem[2])
				tmpList4.append(elem[3])
			
			# reinitialise data		
			xData = np.zeros((dim, sampleSize))
			xxData = np.zeros((dim, sampleSize))
			yData = np.zeros(sampleSize)
			uData = np.zeros(sampleSize)
			stgCost = np.zeros(sampleSize)
			
			# write data from buffer into data structures
			xData = np.asarray(tmpList1).transpose()
			# xxData = np.asarray(tmpList3).transpose()
			# uData = np.asarray(tmpList2).transpose()


			################################################################################
			
			f0 = np.zeros((numControls, dim, sampleSize))
			f1 = np.zeros((numControls, dim, sampleSize))
			f2 = np.zeros((numControls, dim, sampleSize))
			f3 = np.zeros((numControls, dim, sampleSize))
			
			controlData = np.zeros((numTimePts - 1, sampleSize))
			
			sampleIdx = np.zeros(sampleSize)
			sampleIdx = np.random.choice([0, 1, 2, 3], sampleSize, replace = True)

			sampleData = np.zeros((2, sampleSize))
			sampleData = sampleSpace[sampleIdx].transpose()

			tmpVec = np.zeros((numControls, sampleSize))
			
			for j in range(0, numControls):
									
				a1 = np.zeros((dim, sampleSize))
				b1 = np.zeros((dim, sampleSize))
				
				stgCost = np.zeros(sampleSize)
				

				a1 = np.dot(A, xData)
				b1 = B.reshape(2, -1) * controlVec[j]

				f0[j, :, :] = a1 + sqrtDt * sigma * np.asarray([1, 0]).reshape(2, 1) + b1
				f1[j, :, :] = a1 + sqrtDt * sigma * np.asarray([-1, 0]).reshape(2, 1) + b1
				f2[j, :, :] = a1 + sqrtDt * sigma * np.asarray([0, 1]).reshape(2, 1) + b1
				f3[j, :, :] = a1 + sqrtDt * sigma * np.asarray([0, -1]).reshape(2, 1) + b1
				
				stgCost = stageCostFunc([xData[0, :], xData[1, :]], controlVec[j], Q, R, boundingBox, domainBox)
								
				tmpVec[j, :] = prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]])) \
						+ prob * (stgCost * dt + modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]]))
				
			# determine optimal control
			controlData = controlVec[tmpVec.argmin(axis = 0)]
						
			xxData = np.dot(A, xData) + sqrtDt * sigma * sampleSpace[sampleData].transpose() + B.reshape(2, -1) * controlData
	
			################################################################################

			# compute target values
		
			stgCost = stageCostFunc([xData[0, :], xData[1, :]], controlData, Q, R, boundingBox, domainBox)
			yData = stgCost * dt + modelFunc.evaluate(i + 1, [xxData[0, :], xxData[1, :]])
		
			# fig0 = plt.figure(0)
			# plt.scatter(xData[0, :], xData[1, :])
			
			
			# fig1 = plt.figure(1)
			# ax = fig1.add_subplot(111, projection = '3d')
			# ax.scatter(xxData[0, :], xxData[1, :], yData)
		
			# plt.show()
		
			costFuncValList, apprErrList, cgPerformanceList = \
				optimiser.myRbfCgOptimiser(sampleSize, xData, yData, modelFunc.getPartialModelFunc(i), \
					maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
					warmUp = False, verbose = False, write2File = True)
			
		depositeIntermediateTrainingResults(modelFunc, pathNames[2], k + 1)
		
	# fig0 = plt.figure()
	# fig0.plot(range(1, maxNumPlcyIter + 1), explRateList)
	# plt.xlabel('iteration number')
	# plt.ylabel('exploration rate')
	# plt.show()
	
	# write performance data 2 file
	# writePerformanceData2File(numTimePts, optimData, pathNames[0])	

def policyEvaluation():
	pass

def simulateFromFile(sId, iterId):
	sign = str(sId)

	random.seed(0)
	np.random.seed(0)

	path2Dir = '../results/policyIteration/2d_b_' + sign + '/2d_b_' + sign + '_0'
	stpFileName = 'setup'


	dim = 2
	boundingBox = []
	boundingBox.append((-2, 2))
	boundingBox.append((-2, 2))	
	
	domainBox = []
	# domainBox.append((-2, 2))
	# domainBox.append((-2, 2))
	
	sigma = 0.01

	# NOTE: it should hold true that  t0 + numTimePts * dt = T
	# -------------
	t0 = 0
	T = 3.0
	numTimeIntervals = 30
	numTimePts = numTimeIntervals + 1
	# -------------
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)

	####################### USER CHECK ########################
	
	print('### check parameters ###')
	print('	> t0 = ' + str(t0))
	print('	> T = ' + str(T))
	print('	> numTimeIntervals = ' + str(numTimeIntervals))
	print('	> numTimePts = ' + str(numTimePts))
	print('	> dt = ' + str(dt))
	print('	> sqrt(dt) = ' + str(sqrtDt))
	print('########################')
	
	# proceed by pressing enter
	input(' ... hit ENTER to continue ...')


	tVec = np.linspace(t0, T, numTimePts)
	
	ctrlLwrBnd = -1.0
	ctrlUprBnd = 1.0	
	
	numControls = 20
	
	controlVec = np.linspace(ctrlLwrBnd, ctrlUprBnd, numControls)
	
	
	
	# gather last parameters and initialise data structures
	
	R = 10
	Q = np.zeros((2, 2))
	Q[0, 0] = 1
	
	
	S = np.zeros((2, 2))
	S[0, 0] = 1
	S[1, 1] = 1
	
	# model parameters
	sigma = 0.01
	
	A = np.ones((2, 2))
	A[1, 0] = 0

	B = np.zeros(2)
	B[0] = 0.5
	B[1] = 1
	
	# initialise model function
	# modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, S, boundingBox, domainBox))
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, S, boundingBox, domainBox))
	modelFunc.readParamsFromFile(path2Dir + '/model' + '/iter_' + str(iterId), 'modelData_t')
	
	for i in range(0, numTimePts - 1):
		modelFunc.partialModelFuncList[i].setBoundingBox(boundingBox)
	
	numEvalPaths = 20
	sampleData = np.random.choice([0, 1, 2, 3], (numTimePts, numEvalPaths), replace = True)
	sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	
	
	X = np.zeros((numTimePts, dim, numEvalPaths))
	U = np.zeros((numTimePts - 1, numEvalPaths))
	
	# gather sampling data needed to simulate the paths ...
	

	# data structures used to store the paths ... 

	
	X[0, 0, :] = np.random.uniform(-1.0, 1.0, numEvalPaths).reshape(1, numEvalPaths)
	X[0, 1, :] = np.random.uniform(-1.0, 1.0, numEvalPaths).reshape(1, numEvalPaths)
	
	# data structures used to store the controls chosen in the updating procedure
	U = np.zeros((numTimePts - 1, numEvalPaths))


	for i in range(0, numTimePts - 1):
			
		f0 = np.zeros((numControls, dim, numEvalPaths))
		f1 = np.zeros((numControls, dim, numEvalPaths))
		f2 = np.zeros((numControls, dim, numEvalPaths))
		f3 = np.zeros((numControls, dim, numEvalPaths))

		# the same as above using the approximation to the value function
		
		tmpVec = np.zeros((numControls, numEvalPaths))

		# in here data will be produced using an approximation to the value function 		
		for j in range(0, numControls):
								
			f0[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
			f1[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([-1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
			f2[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([0, 1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
			f3[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([0, -1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
			
			# stgCost = 0.5 * (quadrForm(X[i, :, :], Q) + R * (controlVec[j] ** 2))
			
			stgCost = np.zeros((dim, numEvalPaths))
			
			stgCost = stageCostFunc([X[i, 0, :], X[i, 1, :]], controlVec[j], Q, R, boundingBox, domainBox)

											
			tmpVec[j, :] = 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]])) \
					+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]])) \
					+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]])) \
					+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]]))
					
		# determine optimal control
		U[i, :] = controlVec[tmpVec.argmin(axis = 0)]
			
		# ... and finally the subsequent state
			
		X[i + 1, :, :] = np.dot(A, X[i, :, :]) \
			+ sigma * sqrtDt * sampleSpace[sampleData[i, :]].transpose() + B.reshape(2, -1) * U[i, :]
	
	# do some policy evaluation here
	
	tmp = 0
		
	tmp = np.zeros(numEvalPaths)
	s = 0
	
	for i in range(0, numTimePts - 1):
		tmp += stageCostFunc([X[i, 0, :], X[i, 1, :]], U[i, :], Q, R, boundingBox, domainBox)
	tmp += terminalCostFunc([X[numTimePts - 1, 0, :], X[numTimePts - 1, 1, :]], S, boundingBox, domainBox)
	s = sum(tmp)
	s /= numEvalPaths
	
	print('policy evaluation value = ' + str(s))
	
	fig = plt.figure()

	ax1 = plt.subplot(1, 2, 1)
	ax2 = plt.subplot(1, 2, 2)
	
	
	for l in range(0, numEvalPaths):
		ax1.plot(X[:, 0, l], X[:, 1, l])
		# ax1.set_xlim([-1, 8])
		# ax1.set_ylim([-3, 3])
			
		ax2.plot(tVec[np.arange(0, numTimePts - 1)], U[:, l])
		ax2.set(aspect = 'equal')
	
	
	#######################################################
	# ### EXACT STUFF 
	
	valFuncMtrcs = np.zeros((numTimePts, 2, 2))

	valFuncMtrcs[-1, :, :] = S.copy()

	for i in range(numTimePts - 2, -1, -1):
			
		W = valFuncMtrcs[i + 1, :, :].copy()
		
		M = R + np.dot(B.transpose(), np.dot(W, B))
		
		invM = 1.0 / M
		
		tmp = np.dot(B.transpose(), np.dot(W, A)).reshape(1, -1)
		valFuncMtrcs[i, :, :] = Q + np.dot(A.transpose(), np.dot(W, A)) - invM * np.dot(tmp.transpose(), tmp)
	
	#######################################################
	
	# idx = 29
	
	M = 100
	
	x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], M)
	x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], M)

	# x3 = np.linspace(-0.5, 0.5, M)
	# x4 = np.linspace(-0.5, 0.5, M)

	[xx1, xx2] = np.meshgrid(x1, x2)
	# [xx3, xx4] = np.meshgrid(x3, x4)
	
	for idx in range(0, 29):
	
		vv0 = np.zeros(xx1.shape)
	
		vv0 = modelFunc.evaluate(idx, [xx1, xx2])
		vv1 = 0.5 * ((xx1 ** 2) * valFuncMtrcs[idx, 0, 0] + (xx2 ** 2) * valFuncMtrcs[idx, 1, 1] + 2 * xx1 * xx2 * valFuncMtrcs[idx, 1, 0])
		
	# vv2 = modelFunc.evaluate(idx, [xx3, xx4])
	# vv3 = 0.5 * ((xx3 ** 2) * valFuncMtrcs[idx, 0, 0] + (xx4 ** 2) * valFuncMtrcs[idx, 1, 1] + 2 * xx3 * xx4 * valFuncMtrcs[idx, 1, 0])
	
		fig1 = plt.figure()
		ax = fig1.gca(projection = '3d')
		surf1 = ax.plot_surface(xx1, xx2, vv0, cmap = cm.winter)
		surf2 = ax.plot_surface(xx1, xx2, vv1, cmap = cm.autumn)
		ax.set_title('approximative (dark) vs. exact value function (bright) at time t = t_' + str(idx))
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
	
		# ax = fig1.gca()
		# levels1 = ax.contourf(np.flip(vv0))
		# ax.set_title('approximate val func')
		# fig1.colorbar(levels1)
	
	# fig2 = plt.figure()
	# ax = fig2.gca(projection = '3d')
	# ax.plot_surface(xx3, xx4, vv2, cmap = cm.winter)
	# ax.plot_surface(xx3, xx4, vv3, cmap = cm.autumn)
	# ax.set_title('approximative (dark) vs. exact value function (bright) at time t = t_' + str(idx))
	# ax.set_xlabel('x1')
	# ax.set_ylabel('x2')
	
	# ax = fig2.gca()
	# levels2 = ax.contourf(np.flip(vv1))
	# ax.set_title('exact val func')
	# fig2.colorbar(levels2)
	
	# fig3 = plt.figure()
	# ax = fig3.gca(projection = '3d')
	# ax.plot_surface(xx1, xx2, np.abs(vv1 - vv0))
	# ax.set_title('absolute error at time t = t_' + str(idx))
	# ax.set_xlabel('x1')
	# ax.set_ylabel('x2')
	
	# fig4 = plt.figure()
	# ax = fig4.gca(projection = '3d')
	# ax.plot_surface(xx3, xx4, np.abs(vv2 - vv3))
	# ax.set_title('absolute error at time t = t_' + str(idx))
	# ax.set_xlabel('x1')
	# ax.set_ylabel('x2')
	
	
	# print('V_00')
	# print(2 * modelFunc.evaluate(idx, [np.ones(1), np.zeros(1)]))
	# print('V_11')
	# print(2 * modelFunc.evaluate(idx, [np.zeros(1), np.ones(1)]))
	
	
	print(U.sum(axis = 0).sum(axis = 0) / ((numTimePts - 1) * numEvalPaths) )
	# print(exctCtrlVec.sum(axis = 0).sum(axis = 0) / ((numTimePts - 1) * numMCIter) )
	
	s = np.zeros(2)
	for l in range(0, numEvalPaths):
		s += X[-1, :, l]
	
	print(s / numEvalPaths)
	
	plt.show()

def evaluatePolicies(sessionId, numIter):
	
	random.seed(0)
	np.random.seed(0)
	
	sign = str(sessionId)
	
	path2Dir = '../results/policyIteration/2d_b_' + sign + '/2d_b_' + sign + '_0'
	stpFileName = 'setup'

	dim = 2
	boundingBox = []
	boundingBox.append((-2, 2))
	boundingBox.append((-2, 2))
	
	domainBox = []
	
	sigma = 0.01

	# NOTE: it should hold true that  t0 + numTimePts * dt = T
	# -------------
	t0 = 0
	T = 2
	numTimeIntervals = 20
	numTimePts = numTimeIntervals + 1
	# -------------
	
	dt = (T - t0) / numTimeIntervals
	sqrtDt = np.sqrt(dt)

	####################### USER CHECK ########################
	
	print('### check parameters ###')
	print('	> t0 = ' + str(t0))
	print('	> T = ' + str(T))
	print('	> numTimeIntervals = ' + str(numTimeIntervals))
	print('	> numTimePts = ' + str(numTimePts))
	print('	> dt = ' + str(dt))
	print('	> sqrt(dt) = ' + str(sqrtDt))
	print('########################')
	
	# proceed by pressing enter
	input(' ... hit ENTER to continue ...')


	tVec = np.linspace(t0, T, numTimePts)
	
	ctrlLwrBnd = -1.0
	ctrlUprBnd = 1.0	
	
	numControls = 10
	
	controlVec = np.linspace(ctrlLwrBnd, ctrlUprBnd, numControls)
	
	
	
	# gather last parameters and initialise data structures
	
	R = 10
	Q = np.zeros((2, 2))
	Q[0, 0] = 1
		
	S = np.zeros((2, 2))
	S[0, 0] = 1
	S[1, 1] = 1
	
	
	sigma = 0.01
	
	A = np.ones((2, 2))
	A[1, 0] = 0

	B = np.zeros(2)
	B[0] = 0.5
	B[1] = 1

	numEvalPaths = 100

	policyScoreList = []

	for iterId in range(0, numIter):
	
		# initialise model function
		modelFunc = MyValueFunctionApproximation(t0, T, numTimePts - 1, dim, boundingBox, lambda x : terminalCostFunc(x, S, boundingBox, domainBox))
		modelFunc.readParamsFromFile(path2Dir + '/model' + '/iter_' + str(iterId), 'modelData_t')
		
		for i in range(0, numTimePts - 1):
			modelFunc.partialModelFuncList[i].setBoundingBox(boundingBox)
		

		sampleData = np.random.choice([0, 1, 2, 3], (numTimePts, numEvalPaths), replace = True)
		sampleSpace = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])	

		# data structures used to store the paths ... 		
		X = np.zeros((numTimePts, dim, numEvalPaths))
		U = np.zeros((numTimePts - 1, numEvalPaths))
		
		X[0, 0, :] = np.random.uniform(-1.0, 1.0, numEvalPaths).reshape(1, numEvalPaths)
		X[0, 1, :] = np.random.uniform(-1.0, 1.0, numEvalPaths).reshape(1, numEvalPaths)
		
		# data structures used to store the controls chosen in the updating procedure
		U = np.zeros((numTimePts - 1, numEvalPaths))


		for i in range(0, numTimePts - 1):
				
			f0 = np.zeros((numControls, dim, numEvalPaths))
			f1 = np.zeros((numControls, dim, numEvalPaths))
			f2 = np.zeros((numControls, dim, numEvalPaths))
			f3 = np.zeros((numControls, dim, numEvalPaths))

			# the same as above using the approximation to the value function
			
			tmpVec = np.zeros((numControls, numEvalPaths))

			# in here data will be produced using an approximation to the value function 		
			for j in range(0, numControls):
									
				f0[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f1[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([-1, 0]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f2[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([0, 1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				f3[j, :, :] = np.dot(A, X[i, :, :]) + sigma * np.array([0, -1]).reshape(2, 1) * sqrtDt + B.reshape(2, -1) * controlVec[j]
				
				stgCost = stageCostFunc([X[i, 0, :], X[i, 1, :]], U[i, :], Q, R, boundingBox, domainBox)
												
				tmpVec[j, :] = 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f0[j, 0, :], f0[j, 1, :]])) \
						+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f1[j, 0, :], f1[j, 1, :]])) \
						+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f2[j, 0, :], f2[j, 1, :]])) \
						+ 0.25 * (stgCost * dt + modelFunc.evaluate(i + 1, [f3[j, 0, :], f3[j, 1, :]]))
						
			# determine optimal control
			U[i, :] = controlVec[tmpVec.argmin(axis = 0)]
				
			# ... and finally the subsequent state
				
			X[i + 1, :, :] = np.dot(A, X[i, :, :]) \
				+ sigma * sqrtDt * sampleSpace[sampleData[i, :]].transpose() + B.reshape(2, -1) * U[i, :]
		
		# do some policy evaluation here
		
		tmp = np.zeros(numEvalPaths)
		s = 0
		
		for i in range(0, numTimePts - 1):
			tmp += stageCostFunc([X[i, 0, :], X[i, 1, :]], U[i, :], Q, R, boundingBox, domainBox)
			# stageCostFunc([X[i, 0, :], X[i, 1, :]], U[i, :,], targetBox, loc, var)
		tmp += terminalCostFunc([X[numTimePts - 1, 0, :], X[numTimePts - 1, 1, :]], S, boundingBox, domainBox)
		s = sum(tmp)
		s /= numEvalPaths
		
		policyScoreList.append(s)
	
		del modelFunc
	
	
	fig = plt.figure()
	
	plt.plot(np.arange(1, numIter + 1), policyScoreList)
	plt.title('policy evaluation: iteration vs. average cost - numEvalPaths = ' + str(numEvalPaths))
	plt.xlabel('iteration')	
	plt.show()
															
if __name__ == '__main__':
	
	sessionId = np.int(sys.argv[1])
	iterId = np.int(sys.argv[2])
	
	# evaluatePolicies(sessionId, 150)
	
	simulateFromFile(sessionId, iterId)
	
	
	
	
