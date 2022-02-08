########################################################################################################################
# importing stuff
########################################################################################################################	

import numpy as np
import random

import logging

from myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc
from myALSOptimiser import *

import time

########################################################################################################################	
# definition of function which represents a function approximation example in 5d
########################################################################################################################	

def myDataFunc5d(x1, x2, x3, x4, x5):
	
	return 10 * np.sin(np.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5
	
def my5dExample(L, rank, degrs, write2File, signature, subSignature):
	
	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### approximating explicitely given function
	#################################################################

	boundingBox = []
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))
	boundingBox.append((0, 1))

	x1Data = np.random.uniform(boundingBox[0][0], boundingBox[0][1], L)
	x2Data = np.random.uniform(boundingBox[1][0], boundingBox[1][1], L)
	x3Data = np.random.uniform(boundingBox[2][0], boundingBox[2][1], L)
	x4Data = np.random.uniform(boundingBox[3][0], boundingBox[3][1], L)
	x5Data = np.random.uniform(boundingBox[4][0], boundingBox[4][1], L)
	
	xData = np.zeros((5, L))
	xData[0, :] = x1Data
	xData[1, :] = x2Data
	xData[2, :] = x3Data
	xData[3, :] = x4Data
	xData[4, :] = x5Data
	
	yData = myDataFunc5d(x1Data, x2Data, x3Data, x4Data, x5Data)

	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	# fix ALS parameters (there are default parameters ... )
	maxNumALSIter = 200
	eta = 1e-4
	epsALS = 1e-2
	
	# fix CG parameters (there are default parameters ... )
	maxNumCGIter = 15 
	epsCG = 1e-5
	resNormFrac = 1e-1
	
	# for logging purposes write parameters into a dict ...
	alsParamsDict = {}
	alsParamsDict['maxNumALSIter'] = maxNumALSIter
	alsParamsDict['eta'] = eta
	alsParamsDict['descentBound'] = epsALS
	
	cgParamsDict = {}
	cgParamsDict['maxNumCGIter'] = maxNumCGIter
	cgParamsDict['residualBound'] = epsCG
	cgParamsDict['residualFraction'] = resNormFrac
	
	# initialise ansatz function
	dim = 5
	modelFunc = MyCPDRadialAnsatzFunc(dim, boundingBox, rank, degrs)
	
	if write2File == True:
		logging.info('------------------------------------------------------------------------------------------')
		logging.info('> start optimisation procedure corresponding to the sub signature # ' + subSignature + ' #')
	
	# initialise object of optimiser class
	optimiser = MyALSRbfOptimiser(eta = eta, maxNumALSIter = maxNumALSIter, epsALS = epsALS)
		
	print('> start optimisation procedure')
		
	t0 = time.time()
	costFuncValList, avrgdCostFuncValList, apprErrList, cgPerformanceList = \
			optimiser.myRbfCgOptimiser(L, boundingBox, xData, yData, modelFunc, maxNumCGIter = maxNumCGIter, epsCG = epsCG, resNormFrac = resNormFrac, \
				warmUp = True, verbose = False, write2File = write2File)
	
	t1 = time.time()
	
	if write2File == True:
		logging.info('> optimisation procedure corresponding to sub signature # ' + subSignature + ' # finished after # ' + str(t1 - t0) + ' # seconds')
	
	print('> optimisation procedure finished after # ' + str(t1 - t0) + ' # seconds')
	
	return modelFunc, costFuncValList, avrgdCostFuncValList, apprErrList, cgPerformanceList, alsParamsDict, cgParamsDict
	
