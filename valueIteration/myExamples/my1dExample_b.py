########################################################################################################################
# importing stuff
########################################################################################################################	

import numpy as np
import random

import logging

from myValueFunctionApproximation import MyValueFunctionApproximation
from myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc
from myALSOptimiser import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import pyplot as plt

import time

########################################################################################################################	
# problem description
########################################################################################################################	

# consider again the sde (same as in my1dValFuncExample_a)
#
#	(1)	dX_t = (r + u(t) * (mu - r)) * X_t dt + X_t * sigma * u(t) dW_t,
#
# and the corresponding optimal control problem
#
#	max E((X_T) ^ alpha).
#
# introducing stochastic process Y_t via Y_t = log(X_t) the sde (1) reads
#
#	(2) dY_t = (r + u(t) * (mu - r) - 0.5 * u(t)^2 * sigma^2) * dt + u(t) * sigma dW_t;
#
# and the corresponding optimal control problem is of the form
#
#	max E(exp(alpha * Y_T))
#

########################################################################################################################	
# definition of function which represents a value function approximation example in 1d state space
########################################################################################################################	

def exactValFunc(x1, t, r, mu, sigma, alpha, T):
	tmp = alpha * (r + 0.5 * (((mu - r) ** 2) / ((1 - alpha) * (sigma ** 2))))
	return np.exp(alpha * x1) * np.exp(tmp * (T - t))

def mySdeCoeffsFunc1(x1, u, r, mu, sig):
	tmp = u * sig
	
	retVal1 = (r + u * (mu - r)) - 0.5 * (tmp ** 2)
	retVal2 = tmp
	return retVal1, retVal2

def terminalCostFunc(x1, alpha):
	"""
		@param[in] x1 list of length one containing data points
	"""
	retVal = np.zeros(x1[0].shape)
	retVal = np.exp(x1[0] * alpha)
	return retVal

def my1dValFuncExample_b(L, rank, degrs, write2File, signature, subSignature):
	"""
		@param[in]
		@param[in] 
		@param[in] degrs a list containing the degrees; use the same degrees for all time points!	
	"""
	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### initialisations
	#################################################################

	boundingBox = []
	boundingBox.append((0, 10))
	
	numControls = 10
	controlVec = np.linspace(0, 1, numControls + 1)
	
	# parameters regarding time and corresponding discretisation
	t0 = 0
	T = 5
	dt = 1e-1
	sqrtDt = np.sqrt(dt)
	tVec = np.arange(t0, T + dt, dt)
	numTimePts = tVec.size
	
	# model parameters
	r = 0.03
	mu = 0.035
	alpha = 0.3
	sigma = 0.1
	
	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 15
	eta = 1e-4
	epsALS = 1e-2
	
	# fix CG parameters (there are default parameters ... )
	maxNumCGIter = 15 
	epsCG = 1e-5
	resNormFrac = 1e-1
	
	# fix parameters regarding the value iteration
	maxNumValIter = 20
	# numMCIter = L * numTimePts
	numMCIter = L
	
	# for logging purposes write parameters into a dict ...
	alsParamsDict = {}
	alsParamsDict['maxNumALSIter'] = maxNumALSIter
	alsParamsDict['eta'] = eta
	alsParamsDict['descentBound'] = epsALS
	
	cgParamsDict = {}
	cgParamsDict['maxNumCGIter'] = maxNumCGIter
	cgParamsDict['residualBound'] = epsCG
	cgParamsDict['residualFraction'] = resNormFrac
	
	valIterParamsDict = {}
	valIterParamsDict['maxNumValIter'] = maxNumValIter
	valIterParamsDict['numMCIter'] = numMCIter
	valIterParamsDict['numTimePts'] = numTimePts
	
	# initialise model function
	dim = 1
	modelFunc = MyValueFunctionApproximation(t0, T, numTimePts, dim, boundingBox, lambda x : terminalCostFunc(x, alpha))
	apprParams = []
	for i in range(0, numTimePts - 1):
		apprParams.append((rank, degrs))
	modelFunc.initialise(apprParams)
	
	if write2File == True:
		logging.info('------------------------------------------------------------------------------------------')
		logging.info('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')

	# initialise object of optimiser class
	optimiser = MyALSRbfOptimiser(eta = eta, maxNumALSIter = maxNumALSIter, epsALS = epsALS)
		
	print('> start policy iteration procedure')	
	
	#################################################################
	# ### start policy iteration procedure
	#################################################################
	
	# provide a data structure where certain results of the optimisation procedure will be stored ...
	optimData = np.zeros((maxNumValIter, numTimePts - 1, 3))
	
	stateData = np.zeros((numTimePts, dim, L))
	controlData = np.zeros((numTimePts - 1, L))
	sampleData = np.random.choice([-1, 1], (numTimePts, L), replace = True)
	
	initVals = np.zeros((dim, L))
	initVals[0, :] = np.random.uniform(0, 8, L).reshape(1, L)

	stateData[0, 0, :] = initVals
	
	f0 = np.zeros((numControls, dim, L))
	f1 = np.zeros((numControls, dim, L))
		
	for k in range(0, maxNumValIter):

		print('policy iteration ' + str(k))

		#################################################################
		# ### simulation phase
		#################################################################
		
		for i in range(0, numTimePts - 1):
		
			if i < numTimePts - 2:
				# in here data will be produced using an approximation to the value function 
				
				tmpVec = np.zeros((numControls, L))
				
				for j in range(0, numControls):
					
					a, b = mySdeCoeffsFunc1(stateData[i, 0, :], controlVec[j], r, mu, sigma)
					
					f0[j, 0, :] = stateData[i, 0, :] + a * dt + sqrtDt * b
					f1[j, 0, :] = stateData[i, 0, :] + a * dt - sqrtDt * b
										
					tmpVec[j, :] = 0.5 * (modelFunc.evaluate(i + 1, [f0[j, 0, :]])) \
							+ 0.5 * (modelFunc.evaluate(i + 1, [f1[j, 0, :]]))
					
				# determine optimal control
				optCntrIdxVec = tmpVec.argmax(axis = 0)
				optCntrVec = controlVec[optCntrIdxVec]
				
				controlData[i, :] = optCntrVec
				
				# ... and finally the subsequent state
				a, b = mySdeCoeffsFunc1(stateData[i, 0, :], optCntrVec, r, mu, sigma)
				
				stateData[i + 1, 0, :] = stateData[i, 0, :] + a * dt + sqrtDt * sampleData[i, :] * b
				
			else:
				# in here data can be produced by means of the exact value function arising due to
				# stageCostFunc() and terminalCostFunc() ...
				
				for j in range(0, numControls):
					
					a, b = mySdeCoeffsFunc1(stateData[i, 0, :], controlVec[j], r, mu, sigma)
					
					f0[j, 0, :] = stateData[i, 0, :] + a * dt + sqrtDt * b
					f1[j, 0, :] = stateData[i, 0, :] + a * dt - sqrtDt * b
					
					tmpVec[j, :] = 0.5 * (modelFunc.evaluate(i + 1, [f0[j, 0, :]])) \
							+ 0.5 * (modelFunc.evaluate(i + 1, [f1[j, 0, :]]))

				# determine optimal control
				optCntrIdxVec = tmpVec.argmax(axis = 0)
				optCntrVec = controlVec[optCntrIdxVec]
				
				controlData[i, :] = optCntrVec
				
				# ... and finally the subsequent state
				a, b = mySdeCoeffsFunc1(stateData[i, 0, :], optCntrVec, r, mu, sigma)
				
				stateData[i + 1, :, :] = stateData[i, :, :] + a * dt + sqrtDt * sampleData[i, :] * b
					
		#################################################################
		# ### update phase
		#################################################################
		
		for i in range(numTimePts - 2, -1, -1):
		
			if (i == numTimePts - 2):
				
				xData = np.zeros((dim, numMCIter))
				xData[0, :] = stateData[numTimePts - 1, 0, :].copy()
				yData = modelFunc.evaluate(i + 1, [xData[0, :]])
											
				costFuncValList, avrgdCostFuncValList, apprErrList, cgPerformanceList = \
					optimiser.myRbfCgOptimiser(L, xData, yData, modelFunc.getPartialModelFunc(i), \
						maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
						warmUp = False, verbose = False, write2File = write2File)
				
				# log optimisation results ...
				optimData[k, i, 0] = costFuncValList[-1]
				optimData[k, i, 1] = avrgdCostFuncValList[-1]
				optimData[k, i, 2] = apprErrList[-1]
				
			else:
				
				xData = np.zeros((dim, numMCIter))
				xData[0, :] = stateData[i, 0, :].copy()
				yData = modelFunc.evaluate(i + 1, [xData[0, :]])
				
				costFuncValList, avrgdCostFuncValList, apprErrList, cgPerformanceList = \
					optimiser.myRbfCgOptimiser(L, xData, yData, modelFunc.getPartialModelFunc(i), \
						maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
						warmUp = False, verbose = False, write2File = write2File)

				# log optimisation results ...
				optimData[k, i, 0] = costFuncValList[-1]
				optimData[k, i, 1] = avrgdCostFuncValList[-1]
				optimData[k, i, 2] = apprErrList[-1]



	###########################################################################################################
	# now simulate several paths due to exact value function and due to its approximation
	#	> do not like it that this part of code is placed here .... 
	###########################################################################################################
		
	resultsDirName = 'results/valueIteration'
	figuresDirName = 'figures'
	path2FigDir = '../' + resultsDirName + '/' + signature + '/' + subSignature + '/figures'

	initVal = 4		
	numEvalPaths = 500

	# data structures used to store the paths ... 
	X = np.zeros((numTimePts, numEvalPaths))
	Y = np.zeros((numTimePts, numEvalPaths))
	X[0, :] = initVal * np.ones(numEvalPaths)
	Y[0, :] = initVal * np.ones(numEvalPaths)

	# data structures used to store the controls chosen in the updating procedure
	U = np.zeros((numTimePts - 1, numEvalPaths))
	V = np.zeros((numTimePts - 1, numEvalPaths))
			
	# simulate ...		

	for i in range(0, numTimePts - 1):
	
		for l in range(0, numEvalPaths):

			####################################################
			#
			# simulate using exact value function
			#
			####################################################
			
			# determine new states for each possible control ...

			aVec, bVec = mySdeCoeffsFunc1(X[i, l], controlVec, r, mu, sigma)
																														
			f00 = X[i, l] + aVec * dt + sqrtDt * bVec
			f11 = X[i, l] + aVec * dt - sqrtDt * bVec
															
				# tmpVec1 = np.zeros(numControls)
								
			tmpVec1 = 0.5 * exactValFunc(f00, tVec[i + 1], r, mu, sigma, alpha, T) + \
				0.5 * exactValFunc(f11, tVec[i + 1], r, mu, sigma, alpha, T)
			
			# ... finally determine optimal control and the arising new state
		
			u = controlVec[tmpVec1.argmax()]
			U[i, l] = u
			
			tmp1 = np.random.choice([-1, 1], 1, replace = True)
			
			aSc, bSc = mySdeCoeffsFunc1(X[i, l], u, r, mu, sigma)
			
			X[i + 1, l] = X[i, l] + aSc * dt + sqrtDt * tmp1 * bSc
		
			####################################################
			#
			# simulate using approximative value function
			#
			####################################################
	
			# the same as above using the approximation to the value function
													
			aVec, bVec = mySdeCoeffsFunc1(Y[i, l], controlVec, r, mu, sigma)					
					
			f00 = Y[i, l] + aVec * dt + sqrtDt * bVec
			f11 = Y[i, l] + aVec * dt - sqrtDt * bVec
																			
			tmpVec2 = 0.5 * modelFunc.evaluate(i + 1, [f00]) + 0.5 * modelFunc.evaluate(i + 1, [f11])
			
			u = controlVec[tmpVec2.argmax(axis = 0)]							
			V[i, l] = u
			
			tmp2 = np.random.choice([-1, 1], 1, replace = True)
			
			aSc, bSc = mySdeCoeffsFunc1(Y[i, l], u, r, mu, sigma)
			
			Y[i + 1, l] = Y[i, l] + aSc * dt + sqrtDt * tmp2 * bSc
					
	# now visualize the paths ...
	fig1 = plt.figure()

	ax1 = plt.subplot(2, 2, 1)
	ax2 = plt.subplot(2, 2, 2)
	ax3 = plt.subplot(2, 2, 3)
	ax4 = plt.subplot(2, 2, 4)
	
	ax1.set_xlabel('time')
	ax1.set_ylabel('wealth')
	ax1.set_title('using exact value function')
	
	ax2.set_xlabel('time')
	ax2.set_ylabel('wealth')
	ax2.set_title('using appr. value function')
	
	ax3.set_xlabel('time')
	ax3.set_ylabel('control')
	ax3.set_title('using exact value function')
	
	ax4.set_xlabel('time')
	ax4.set_ylabel('control')
	ax4.set_title('using appr. value function')
	
	for l in range(0, numEvalPaths):
		ax1.plot(tVec, X[:, l])
		ax1.set_ylim([0, 10])
		
		ax2.plot(tVec, Y[:, l])
		ax2.set_ylim([0, 10])
		
		
		ax3.plot(tVec[0 : numTimePts - 1], U[:, l])
		ax3.set_ylim([0, 1])
		
		ax4.plot(tVec[0 : numTimePts - 1], V[:, l])
		ax4.set_ylim([0, 1])
			
	plt.tight_layout()
	
	
	
	# store figures properly if desired ....
	
	if write2File == True:
		plt.savefig(path2FigDir + '/evalFig1.png', dpi = 300)
		plt.close(fig1)		
	
	# ------------------------------------------------------
	
	# ... and last but not least print to console a few results which allow to judge the result - those results
	# will be written to file too, if desired.
	
	# averaged end point costs
	expValue1 = ((X[numTimePts - 1, :] ** alpha).sum()) / (numEvalPaths)
	expValue2 = ((Y[numTimePts - 1, :] ** alpha).sum()) / (numEvalPaths)
	
	# ... and end point wealths
	expValue3 = ((X[numTimePts - 1, :]).sum()) / (numEvalPaths)
	expValue4 = ((Y[numTimePts - 1, :]).sum()) / (numEvalPaths)
	
	# averaged controls
	avrgContr1 = (U.sum(axis = 0).sum(axis = 0)) / ((numTimePts - 1) * numEvalPaths)
	avrgContr2 = (V.sum(axis = 0).sum(axis = 0)) / ((numTimePts - 1) * numEvalPaths)
	
	print('######################################################################\n')
	print('\n')
	print('##### terminal costs #####')
	print('\n')
	print(' > using exact value function: ' + str(expValue1))
	print(' > using approximative value function: ' + str(expValue2))
	print('\n')
	print('######################################################################\n')
	print('\n')
	print('##### terminal wealth #####')
	print('\n')
	print(' > using exact value function: ' + str(expValue3))
	print(' > using approximative value function: ' + str(expValue4))
	print('\n')
	print('######################################################################\n')
	print('\n')
	print('##### average controls #####')
	print('\n')
	print(' > using exact value function: ' + str(avrgContr1))
	print(' > using approximative value function: ' + str(avrgContr2))
	print('\n')


	if write2File == True:
		evalResFile = open(path2FigDir + '/' + 'evalResults.txt', 'w')
		
		evalResFile.write('######################################################################\n')
		evalResFile.write('\n')
		evalResFile.write('##### terminal cost #####')
		evalResFile.write('\n')
		evalResFile.write(' > using exact value function: ' + str(expValue1) + '\n')
		evalResFile.write(' > using approximative value function: ' + str(expValue2) + '\n')
		evalResFile.write('\n')
		evalResFile.write('######################################################################\n')
		evalResFile.write('\n')
		evalResFile.write('##### terminal wealth #####')
		evalResFile.write('\n')
		evalResFile.write(' > using exact value function: ' + str(expValue3) + '\n')
		evalResFile.write(' > using approximative value function: ' + str(expValue4) + '\n')
		evalResFile.write('\n')
		evalResFile.write('######################################################################\n')
		evalResFile.write('\n')
		evalResFile.write('##### average control #####')
		evalResFile.write('\n')
		evalResFile.write(' > using exact value function: ' + str(avrgContr1) + '\n')
		evalResFile.write(' > using approximative value function: ' + str(avrgContr2) + '\n')
		evalResFile.write('\n')
		
		evalResFile.close()	
		
	
	return modelFunc, optimData, alsParamsDict, cgParamsDict, valIterParamsDict,\
		(lambda x1, t: exactValFunc(x1, t, r, mu, sigma, alpha, T))

