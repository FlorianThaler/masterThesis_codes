"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements provides the data
					structure which is used to approximate the value function of a given control problem over the
					whole discrete time range by means of Gaussian CPD functions.
	Year:			2019/2020
"""

import numpy as np

import os
import tarfile

import sys
sys.path.append('../functionApproximation/cpd/')

from myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc

class MyValueFunctionApproximation:
	
	"""
		this class is the data structure used to approcimate the value function of control problem
		over the whole discrete time range
	"""
	
	def __init__(self, t0, T, numTimePts, dim, box, endPointCost):
		"""
			@param[in] ### t0 ### initial time
			@param[in] ### T ### final time
			@param[in] ### numTimePTs ### integer representing the number of time points without taking into account
				the final time point
			@param[in] ### dim ### integer corresponding to the dimension of the state space
			@param[in] ### box ### list of the tupels representing the approximation domain which is a cuboid in R^{dim}
			@param[in] ### endPointCost ### function corresponding to the terminal cost function which is known!
		
			NOTE:
				> numTimePts denotes here the number of time points ### without ### taking into account the last
					time point
		"""
	
		self.startTime = t0
		self.endTime = T
		self.numTimePts = numTimePts
	
		self.stateSpaceDim = dim
		self.boundingBox = box.copy()
		
		self.endPointCostFunc = endPointCost
		# this list is of length self.numTimePts
		self.partialModelFuncList = []
	
	# ### getter and setter ### 
	
	# --------------------------------------------------
	
	def getStartTime(self):
		return self.startTime
	
	# --------------------------------------------------
		
	def getEndTime(self):
		return self.endTime

	# --------------------------------------------------
	
	def getNumTimePoints(self):
		return self.numTimePts

	# --------------------------------------------------
	
	def getStateSpaceDim(self):
		return self.stateSpaceDim

	# --------------------------------------------------
	
	def getBoundingBox(self, timeIdx):
		return self.partialModelFuncList[timeIdx].getBoundingBox()
		
	def setBoundingBox(self, timeIdx, box):
		self.partialModelFuncList[timeIdx].setBoundingBox(box)

	# --------------------------------------------------
	
	def getPartialModelFunc(self, timeIdx):
		return self.partialModelFuncList[timeIdx]

	# --------------------------------------------------
	
	# some more class methods	
		
	def initialise(self, apprParams, func = 'rbf'):
		"""
			@param[in] apprParams list which contains for any time point except the final one a tuple consisting of
				the rank of the cpd approximation and a list containing the degrees of the approximation for each 
				coordinate, i.e.
					apprParams[i] = (rnk, [degr_0, ..., degr_(dim - 1)])
			@param[in] func string indicating which kind of ansatz functions shall be used. by default for any 
				time point rbf functions will be chosen; further at any time point the same (!!!) kind of functions will
				be used!
		"""
		
		if func == 'rbf':
		
			for i in range(0, self.numTimePts):
				
			
				currRank = apprParams[i][0]
				currDegrs = apprParams[i][1]
				
				# tmpAnsatzFunc = MyCPDRadialAnsatzFunc(self.stateSpaceDim, self.boundingBox, currRank, currDegrs, offSet)
				tmpAnsatzFunc = MyCPDRadialAnsatzFunc(self.stateSpaceDim, self.boundingBox, currRank, currDegrs)
				tmpAnsatzFunc.initialise()
											
				self.partialModelFuncList.append(tmpAnsatzFunc)
		else:
			print('not implemented yet ...')
	
	def evaluate(self, timeIdx, xList, L):
		"""
			@param[in] ### timeIdx ### index of time point at which the approximation should be evaluated
			@param[in] ### xList ### list of data points per coordinate direction - see evaluate() in myCPDAnsatzFunctions.py
			@param[in] ### L ### number of data points
			
			NOTE: 
				> no exception handling is done, i.e. index could be out of range!
			
		"""
		
		retVal = np.zeros(xList[0].shape)
		
		if (timeIdx == self.numTimePts):
			retVal = self.endPointCostFunc(xList)
		else:
			retVal = self.partialModelFuncList[timeIdx].evaluate(xList, L)
			
		return retVal
		
	def evaluate2d(self, timeIdx, xList, L):
		"""
			this function has the same functionality as the above function. the only difference is that xList contains
			a list of matrices and that the corresponding evaluate2d function of the class MyCPDAnsatzFunctions is
			called.
		"""
		
		retVal = np.zeros(xList[0].shape)
		
		if (timeIdx == self.numTimePts):
			retVal = self.endPointCostFunc(xList)
		else:
			retVal = self.partialModelFuncList[timeIdx].evaluate2d(xList, L)
			
		return retVal

	def writeParams2File(self, path2ModDir, fileName):
		
		"""
			@param[in] ### path2ModDir ### string corresponding to the directory where model files should be stored
			@param[in] ### fileName ### string without file extension (and time specification)
		
		
			write for each time point except the final one the parameters calling the corresponding method
			from objects of class 'MyCPDRadialAnsatzFunc', and compress them in a tar.gz file

		"""
		
		
		if os.path.exists(path2ModDir) == True:
			
			# os.chdir(path2ModDir)
			
			# store files containing current parameters (for each time point) to model directory (~path2ModDir),
			# then put the files in a single tar.gz file and delete the files created before ...
			# ... creating tar file just makes problems! =/ Hence: omit it!!
			
			for i in range(0, self.numTimePts):	
				self.partialModelFuncList[i].writeParams2File(path2ModDir, fileName + '_t' + str(i))
	
	def readParamsFromFile(self, path2ModDir, fileName):
		"""
			@param[in] ### path2ModDir ### string corresponding to the directory where model files are stored
			@param[in] ### fileName ### string assumed to be of the form 'modelData_t'; 
			
			read parameter from file and set model functions corresponding to discrete time points from 0 to 
			numTimePts - 1; the model function at final time is given due to endPointCostFunc, which has to be set 
			manually!
		"""
		
		successful = False
		
		if os.path.exists(path2ModDir) == True:
			
			# delete list of model functions!
			
			self.partialModelFuncList = []
			
			# provide parameters for dummy initialisation
			dim = 0
			bndBox = []
			rk = 0
			
			for i in range(0, self.numTimePts):
				tmpAnsatzFunc = MyCPDRadialAnsatzFunc(dim, bndBox, rk)
				tmpAnsatzFunc.readParamsFromFile(path2ModDir, fileName + str(i))
				
				self.partialModelFuncList.append(tmpAnsatzFunc)
			
			successful = True
		
		return successful


def myTestFunc(x, theta):
	return theta ** x[0]
	
def test():
	
	tVec = np.linspace(0, 1, 5)
	dim = 1
	
	box = []
	box.append((0, 1))
	
	theta = 2
	
	stageCost = np.cos
	endPointCost = lambda x : myTestFunc(x, theta)
	
	N = 5
	
	modelFunc = MyValueFunctionApproximation(N, tVec, dim, box, endPointCost)
	
	
	paramList = []
	
	for i in range(0, N - 1):
		paramList.append((2, [3, 3]))
	
	modelFunc.initialise(paramList)
	
	# xVec = np.linspace(0, 2 * np.pi, 5)
	xVec = np.linspace(0, 2, 5)
	xData = []

	print(modelFunc.evaluate(4, [xVec]))
	
if __name__ == '__main__':
	test()
	
