"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements the optimiser 
					used for the update process in the context of function approximation.
	Year:			2019
"""

###########################################################
# importing stuff
###########################################################

import numpy as np
import logging

import time

###########################################################
# definition of functions used in the update process 
# (as non class or memeber functions)
###########################################################

def myMatrixFunc(A, rank, degr, L, xiVecs, psiVecs, eta):
	"""
		this function implements the evaluation of the linear operator showing up in the optimality condition of first
		order.
		
		@param[in] ### A ### a matrix of # rank # rows and # degr # columns, at which the operator will be evaluated
		@param[in] ### rank ###	number of rows of A
		@param[in] ### degr ### number of columns of A
		@param[in] ### L ### number of data points used in the optimisation process - RECALL: the linear operator which
			has to be evaluated here is a sum of L terms.
		@param[in] ### xiVecs ### auxiliary quantity of L rows and # degr # columns
		@param[in] ### psiVecs ### auxiliary quantity of L rows and # rank # columns
		@param[in] ### eta ### penalisation parameter
		@return ### retVal ### matrix of the same shape as A, corresponding to the evaluation of the linear operator
			appearing in the first order necessity condition
	"""
	retVal = np.zeros((rank, degr))
	
	for l in range(0, L):
		tmp = psiVecs[l, :].reshape(-1, 1).dot(xiVecs[l, :].reshape(1, -1))
		retVal += np.sum(A * tmp) * tmp
	
	retVal /= L
	
	# adding penalty term
	retVal += eta * A
	
	return retVal

def myRhsFunc(rank, degr, L, xiVecs, psiVecs, yData):
	"""
		this function implements the evaluation of the rhs of the linear system corresponding to necessary optimality
		condition of first order.
		
		@param[in] ### rank ###	number of rows of return value
		@param[in] ### degr ### number of columns of return value
		@param[in] ### L ### number of data points used in the optimisation process - RECALL: the term which
			has to be computed here is a sum of L terms.
		@param[in] ### xiVecs ### auxiliary quantity of L rows and # degr # columns
		@param[in] ### psiVecs ### auxiliary quantity of L rows and # rank # columns
		@param[in] ### yData ### 
		@return ### retVal ### matrix of # rank # rows, # degr # columns corresponding to the mentioned rhs
	"""
	
	retVal = np.zeros((rank, degr))
	for l in range(0, L):
		retVal += yData[l] * psiVecs[l, :].reshape(-1, 1).dot(xiVecs[l, :].reshape(1, -1))
	
	retVal /= L
		
	return retVal	

def costFunctional(xList, yData, L, modelFunc, eta):
	"""
		this function implements the evaluation of the cost functional.
		
		@param[in] ### xList ### list of data points in the approximation domain where functional should be evaluated
			it consists of a list of numpy arrays, where each of this arrays contains the respective coordinates
		@param[in] ### yList ### numpy array of exact function values.
		@param[in] ### L ### number of data points
		@param[in] ### modelFunc ### instance of model function which is used to approximate the function corresponding
			to the function values yData
		@param[in] ### eta ### penalisation parameter
		@return ### retVal ### scalar corresponding to the function value of the cost
	"""
		
	retVal = 0
	

	retVal += np.linalg.norm(modelFunc.evaluate(xList, L) - yData, 2) ** 2
	retVal /= L
	
	# add penalty term
	for d in range(0, modelFunc.getDim()):
		retVal += eta * np.linalg.norm(modelFunc.getCoeffs(d), 'fro') ** 2

	retVal *= 0.5
	
	return retVal

def approximationError(xList, yData, L, modelFunc, norm = 'l2'):
	"""
		this function computes the approximation error, i.e. the error between approximated function
		values and real function values. 
		
		@param[in] ### xList ### as usual: a list of numpy arrays corresponding to the coordinates of the 
			data points
		@param[in] ### yData ### numpy array representing the real function values
		@param[in] ### modelFunc ### instance of class MyCPDRadialAnsatzFunc
		@param[in] ### norm ### string indicating which norm should be computed. by default it is the l2 norm; 
			one can choose the l^{\infty} norm as well
		@return ### retVal ### scalar corresponding to the approximation error
	"""
	
	retVal = -1
	
	if norm == 'l2':
		retVal = np.linalg.norm(modelFunc.evaluate(xList, L) - yData, 2)
	elif norm == 'lInf':
		retVal = max(np.abs(modelFunc.evaluate(xList, L) - yData))
	elif norm == 'mse':
		retVal = (1 / (2 * L)) * np.linalg.norm(modelFunc.evaluate(xList, L) - yData, 2) ** 2
	return retVal

###########################################################
# definition of several classes
###########################################################

class MyALSOptimiser:
	"""
		this class is just the basic class of each of the ALS optimiser used (or tried out respectively ...)
	"""
	
	def __init__(self, eta, maxNumALSIter, epsALS):
		"""
			the usual initialiser method ...
			
			@param[in] ### eta ### penalisation parameter
			@param[in] ### maxNumALSIter ### maximal number of ALS iterations
			@param[in] ### epsALS ### used as stopping criterion for the ALS method. if current and previous value
				of cost function differ by less than epsALS, than method will be stopped.
		"""

		self.eta = eta
		self.maxNumALSIter = maxNumALSIter
		self.epsALS = epsALS
		
class MyALSRbfOptimiser(MyALSOptimiser):
	"""
		this class - a derivation of MyALSOptimiser - implements the ALS optimisation procedure in the context
		of function approximation by means of functions in CPD format on the basis of Gaussian ansatz functions.
		
		NOTE:
			> the class is not applicable for non Gaussian ansatz functions ...
	"""
	
	def __init__(self, eta = 1e-4, maxNumALSIter = 200, epsALS = 1e-1):
		"""
			call initialiser method of super class and set default values
			
			@param[in] ### eta ### penalisation parameter
			@param[in] ### maxNumALSIter ### maximal number of ALS iterations
			@param[in] ### epsALS ### (additional) stopping criterion for the ALS iteration
			
		"""
		super().__init__(eta, maxNumALSIter, epsALS)
		
	def myRbfCgOptimiser(self, L, xData, yData, modelFunc, path2ModDir, modFileName, \
			maxNumCGIter = 8, epsCG = 1e-2, resNormFrac = 1e-1, warmUp = True, verbose = True, write2File = True):
				
		"""
			this function implements the ALS update procedure by means of a CG approach.
			
			@param[in] ### L ### number of data points
			@param[in] ### xData ### data set in matrix format, where each row contains the coordinates of the 
				corresponding data points.
			@param[in] ### yData ### real function values
			@param[in] ### modelFunc ### instance of class MyCPDRadialAnsatzFunc
			@param[in] ### path2ModDir ### string corresponding to the path to the directory where model data should
				be stored.
			@param[in] ### modFileName ### string corresponding to the basic model name - without extension
			@param[in] ### maxNumCGIter ### maximal number of CG iterations which will performed per axis and per
				ALS iteration to solve the linear system corresponding to the necessary optimality condition of first
				order
			@param[in] ### epsCG ### further stopping criterion for cg method - iteration will be stopped, when 
				residual is smaller than epsCG
			@param[in] ### resNormFrac ### further stopping criterion for the cg method - iteration will be stopped, 
				when norm of the residual is smaller than resNormFrac times the initial residual
			@param[in] ### warmUp ### boolean variable used to decide if some kind of warm up training will be 
				performed
			@param[in] ### verbose ### boolean variable indicating if during training messages should be printed to
				console
			@param[in] ### write2File ### boolean variable indicating if logging data should be written to file.
			
			NOTE:
				> this function modifies member variables of the instance modelFunc !!!
		"""

		# introduce some variable representing how often data was already written to file - needed to determine
		# the proper file name in the context of storing parameter to file.
		writeCounter = 0
		
		if write2File:
			# write initial parameters to file
			logging.info('> write model parameters to file')
			modelFunc.writeParams2File(path2ModDir, modFileName + str(writeCounter))
			writeCounter += 1
			
		
		# gather some data which will be needed quite often ...
		
		currDim = modelFunc.getDim()						# current dimension
		currDegrs = modelFunc.getDegrs()					# current degree of cpd function
		currRank = modelFunc.getRank()						# current rank of cpd function
		
		# introduce lists to store the value of the cost functional after each optimisation step and the norms of the
		# gradients of the cost functional w.r.t. to the variable which is the current optimisation variable also after
		# a full cg iteration step

		costFuncValList = []
		cgPerformanceList = []
		lInfApprErrList = []
		# l2ApprErrList = []
		mseList = []
		

		# put input data into list
		# 	> ### IS THIS REALLY NECESSARY? ###
		xList = []		
		for d in range(0, currDim):
			xList.append(xData[d, :])
			
			# initialise cg performance list as list of lists ...
			cgPerformanceList.append([])
		
		####################################################################################################################
		# --- do some warm up optimisation --- 
		####################################################################################################################
		
		if warmUp == True:
			
			# make only a few ALS iterations, but using a very high number of (possible) CG iterations
			
			numWarmUpALSIter = 1
			maxNumCGWarmUpIter = 2 * max([currDegrs[d] * currRank for d in range(0, currDim)])
		
			if verbose == True:				
				print('	+ start warm up procedure')
				print('		* perform ' + str(numWarmUpALSIter) + ' ALS iterations')
				print('		* CG parameters:')
				print('			- maximal number of iterations = ' +  str(maxNumCGWarmUpIter))
				print('			- residual accuracy = ' +  str(epsCG))
				

			logging.info('	+ start warm up procedure')
			logging.info('		* perform ' + str(numWarmUpALSIter) + ' ALS iterations')
			logging.info('		* CG parameters:')
			logging.info('			- maximal number of iterations = ' +  str(maxNumCGWarmUpIter))
			logging.info('			- residual accuracy = ' +  str(epsCG))
				

			for i in range(0, numWarmUpALSIter):
								
				for d in range(0, currDim):
		
					# now start cg method ...
					
					psiVecs = np.zeros((L, currRank))
					xiVecs = np.zeros((L, currDegrs[d]))
					
					#######################################
					# compute auxiliary quantities ...

					phiVecTr = np.zeros((currDegrs[d], L))
					psiVecTr = np.ones((currRank, L))

					t0 = time.time()

					for m in range(0, currDim):
						if m != d:		
							for k in range(0, currRank):
								psiVecTr[k, :] *= modelFunc.evalLinCombRadBasFunc1d(xList[m], m, k)
					
					for nu in range(0, currDegrs[d]):
						phiVecTr[nu, :] = modelFunc.evalRadBasFunc1d(xList[d], d, nu)

					xiVecs = phiVecTr.transpose()
					psiVecs = psiVecTr.transpose()			
											
					x = modelFunc.getCoeffs(d)									
					b = myRhsFunc(currRank, currDegrs[d], L, xiVecs, psiVecs, yData)
					r = b - myMatrixFunc(x, currRank, currDegrs[d], L, xiVecs, psiVecs, self.eta)
					
					resNorm = np.sum(r * r)
					resNorm0 = np.sqrt(resNorm)
					
					searchDir = r.copy()
			
					k = 0
					
					while k < maxNumCGWarmUpIter and np.sqrt(resNorm) > epsCG:
						z = myMatrixFunc(searchDir, currRank, currDegrs[d], L, xiVecs, psiVecs, self.eta)
													
						tmp = np.sum(r * r)
						
						alpha = (tmp) / (np.sum(searchDir * z))
						
						# determine new iterate
						x += alpha * searchDir
						
						# adjust residuals
						rNew = (r - alpha * z).copy()
						
						resNorm = np.sum(rNew * rNew)
						
						beta = (resNorm) / (tmp)

						r = rNew.copy()
						searchDir = (r + beta * searchDir).copy()
						
						k += 1
					
					modelFunc.setCoeffs(d, x)
		
				if write2File:
					# write parameters to file
					logging.info('> write model parameters to file')
					modelFunc.writeParams2File(path2ModDir, modFileName + str(writeCounter))
					writeCounter += 1
			
		####################################################################################################################
		# --- start serious optimsation here --- 
		####################################################################################################################
		
		# determine the number of parameters which have to approximated
		totNumParams = 0
		for d in range(0, currDim):
			totNumParams += currDegrs[d] * modelFunc.getRank()
			
		if verbose == True:
		
			print('	----------------------------------------------------------------------')
			print('	+ start ALS optimisation procedure')
			print('		* number of parameters to estimate in total = ' + str(totNumParams))
			print('		* number of data points = ' + str(L))
			print('		* maximal number of ALS iterations = ' + str(self.maxNumALSIter))
			print('		* descent bound = ' + str(self.epsALS))
			print('		* CG parameters:')
			print('			- maximal number of iterations = ' +  str(maxNumCGIter))
			print('			- residual accuracy = ' +  str(epsCG))
			print('			- residual fraction = ' +  str(resNormFrac))
			print('	  --------------------------------------------------------------------')

		logging.info('	----------------------------------------------------------------------')
		logging.info('	+ start ALS optimisation procedure')
		logging.info('		* number of parameters to estimate in total = ' + str(totNumParams))
		logging.info('		* number of data points = ' + str(L))
		logging.info('		* maximal number of ALS iterations = ' + str(self.maxNumALSIter))
		logging.info('		* descent bound = ' + str(self.epsALS))
		logging.info('		* CG parameters:')
		logging.info('			- maximal number of iterations = ' +  str(maxNumCGIter))
		logging.info('			- residual accuracy = ' +  str(epsCG))
		logging.info('			- residual fraction = ' +  str(resNormFrac))
		logging.info('	  --------------------------------------------------------------------')
			
		# introduce iteration counter
		i = 0
			
		costFuncValNew = costFunctional(xList, yData, L, modelFunc, self.eta)
		costFuncValOld = costFuncValNew - 1	

		# ### START ALS ITERATIONS HERE ###
		while (i < self.maxNumALSIter) and (np.abs(costFuncValOld - costFuncValNew) > self.epsALS):

			print('	  start ALS iteration number # ' + str(i + 1) + ' # ')			
			logging.info('	  start ALS iteration number # ' + str(i + 1) + ' # ')
			
			for d in range(0, currDim):
			
				# now start cg method ...
								
				#######################################
				# compute auxiliary quantities ...
				#######################################
				
				psiVecs = np.zeros((L, currRank))
				xiVecs = np.zeros((L, currDegrs[d]))
				
				xiVecTr = np.zeros((currDegrs[d], L))
				psiVecTr = np.ones((currRank, L))

				t0 = time.time()

				for m in range(0, currDim):
					if m != d:		
						for k in range(0, currRank):
							psiVecTr[k, :] *= modelFunc.evalLinCombRadBasFunc1d(xList[m], m, k)
				
				for nu in range(0, currDegrs[d]):
					xiVecTr[nu, :] = modelFunc.evalRadBasFunc1d(xList[d], d, nu)

				t1 = time.time()

				xiVecs = xiVecTr.transpose()
				psiVecs = psiVecTr.transpose()			
				
				if verbose == True:
					print('		-----------------------------------------------------------')
					print('		* start cg iteration to optimise w.r.t. x' + str(d + 1) + '- coordinates')
				
				logging.info('		-----------------------------------------------------------')
				logging.info('		* start cg iteration to optimise w.r.t. x' + str(d + 1) + '- coordinates')
				
				#######################################
				# start with CG ...
				#######################################
				
				x = modelFunc.getCoeffs(d)						
				b = myRhsFunc(currRank, currDegrs[d], L, xiVecs, psiVecs, yData)
				r = b - myMatrixFunc(x, currRank, currDegrs[d], L, xiVecs, psiVecs, self.eta)
				
				resNorm = np.sum(r * r)
				resNorm0 = np.sqrt(resNorm)
				
				searchDir = r.copy()
		
				k = 0
				stop = False
				reason = ''
				
				while stop == False:
					z = myMatrixFunc(searchDir, currRank, currDegrs[d], L, xiVecs, psiVecs, self.eta)
					
					tmp = np.sum(r * r)
					
					alpha = (tmp) / (np.sum(searchDir * z))

					# determine new iterate
					x += alpha * searchDir
					
					# adjust residuals
					rNew = (r - alpha * z).copy()
					
					resNorm = np.sum(rNew * rNew)
					
					beta = (resNorm) / (tmp)

					r = rNew.copy()
					searchDir = (r + beta * searchDir).copy()
					
					k += 1
					
					stop = (k >= maxNumCGIter) or np.sqrt(resNorm) <= np.max(np.asarray([epsCG, resNormFrac * resNorm0]))
					reason = (k >= maxNumCGIter) * ' ### maximal number of CG iterations reached ### ' \
						+ (np.sqrt(resNorm) <= np.max(np.asarray([epsCG, resNormFrac * resNorm0]))) * ' ### residual is sufficiently small ### '
				
				t2 = time.time()
				
				# log results
				cgPerformanceList[d].append((k, np.sqrt(resNorm), reason))						
						
				if verbose == True:
					print('		  ---------------------------------------------------------')
					print('		* cg iteration stopped since: ### ' + reason + ' ###')
					print('			- iteration stopped after # ' + str(k) + ' # iterations')
					print('			- residual accuracy = # ' + str(resNorm) + ' #')
					print('		  ---------------------------------------------------------')
					print('		* it took ' + str(t2 - t0) + ' s to perform the whole cg step')
					print('		* it took ' + str(t1 - t0) + ' s to compute the vectors psiVec, phiVec')
					print('		* norm of the gradient: ' + str(np.sqrt(resNorm)))
				
				logging.info('		  ---------------------------------------------------------')
				logging.info('		* cg iteration stopped since: ### ' + reason + ' ###')
				logging.info('			- iteration stopped after # ' + str(k) + ' # iterations')
				logging.info('			- residual accuracy = # ' + str(resNorm) + ' #')
				logging.info('		  ---------------------------------------------------------')
				logging.info('		* it took ' + str(t2 - t0) + ' s to perform the whole cg step')
				logging.info('		* it took ' + str(t1 - t0) + ' s to compute the vectors psiVec, phiVec')
				logging.info('		* norm of the gradient: ' + str(np.sqrt(resNorm)))
			
				modelFunc.setCoeffs(d, x)
			
			currCostFuncVal = costFunctional(xList, yData, L, modelFunc, self.eta)
			# l2ApprErr = approximationError(xList, yData, L, modelFunc, norm = 'l2')
			mse = approximationError(xList, yData, L, modelFunc, norm = 'mse')
			lInfApprErr = approximationError(xList, yData, L, modelFunc, norm = 'lInf')
			
			# log results
			costFuncValList.append(currCostFuncVal)
			# l2ApprErrList.append(l2ApprErr)
			mseList.append(mse)
			lInfApprErrList.append(lInfApprErr)

			if verbose == True:
				print('		-----------------------------------------------------------')
				print('	  ALS iteration finished')
				print('		* value of the cost functional = ' + str(currCostFuncVal))
				# print('		* l2 approximation error = ' + str(l2ApprErr))
				print('		* mse = ' + str(mse))
				print('		* lInf approximation error = ' + str(lInfApprErr))
				print('	  --------------------------------------------------------------------')
			
			logging.info('		-----------------------------------------------------------')
			logging.info('	  ALS iteration finished')
			logging.info('		* value of the cost functional = ' + str(currCostFuncVal))
			# logging.info('		* l2 approximation error = ' + str(l2ApprErr))
			logging.info('		* mse = ' + str(mse))
			logging.info('		* lInf approximation error = ' + str(lInfApprErr))
			logging.info('	  --------------------------------------------------------------------')
			
			i += 1
			
			costFuncValOld = costFuncValNew
			costFuncValNew = costFunctional(xList, yData, L, modelFunc, self.eta)
			
			if write2File:
				# write parameters to file
				logging.info('> write model parameters to file')
				modelFunc.writeParams2File(path2ModDir, modFileName + str(writeCounter))
				writeCounter += 1

		# return costFuncValList, l2ApprErrList, lInfApprErrList, cgPerformanceList
		return costFuncValList, mseList, lInfApprErrList, cgPerformanceList


	####################################################################################################################
	####################################################################################################################
	####################################################################################################################

	

########################################################################################################################
				
				
		
	
	
	
