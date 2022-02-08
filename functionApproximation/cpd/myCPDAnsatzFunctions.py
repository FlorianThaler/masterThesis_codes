"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements the class of 
					functions in CPD format based on Gaussian basis functions.
	Year:			2019
"""

###########################################################
# importing stuff
###########################################################

import numpy as np
import random
import os

###########################################################
# definition of several classes
###########################################################

class MyBasicCPDAnsatzFunction:
	"""
		this is the basis class of the cpd function approximation classes down below
	"""
	
	def __init__(self, dim, box, rank, degrs, timeIdx = -1):
		"""
			@param[in] ### dim ### dimension of the space where the domain of the function belongs to
			@param[in] ### box ### list of tupels representing the boundary points of 1d intervals which
						determine the approximation domain as a cuboid in R^{dim}
			@param[in] ### rank ### rank of the CPD decomposition
			@param[in] ### degrs ### a list containing the degrees of the ansatz functions per axis
		"""
		
		self.dim = dim
		self.boundingBox = box.copy()
		self.rank = rank
		self.degrs = degrs.copy()
			
		# introduce time index - just for testing purposes: beeing able to check if the right model function
		#	will be accessed!
		self.time = timeIdx
	
	# ### getter and setter methods ###	
	
	# --------------------------------------------------
	
	def getDim(self):
		return self.dim
	
	def setDim(self):
		pass
	
	# --------------------------------------------------
	
	def getBoundingBox(self):
		return self.boundingBox

	def setBoundingBox(self, box):
		self.boundingBox = []
		self.boundingBox = box.copy()
		
	# --------------------------------------------------	
	
	def getRank(self):
		return self.rank
	
	def setRank(self):
		print('not implemented')
		
	# --------------------------------------------------	
	
	def getDegrs(self):
		return self.degrs	
		
	def setDegrs(self):
		print('not implemented')
	
class MyCPDRadialAnsatzFunc(MyBasicCPDAnsatzFunction):
	"""
		This class represents a CPD type function approximation based on Gaussian ansatz functions; it is a derivation
		from MyBasicCPDAnsatzFunction.
	"""
	
	
	def __init__(self, dim, box, rank, degrs = []):
		"""
			@param[in] ### dim ### dimension of the space where the domain of the function belongs to
			@param[in] ### box ### list of tupels representing the boundary points of 1d intervals which
						determine the approximation domain as a cuboid in R^{dim}
			@param[in] ### rank ### rank of the CPD decomposition
			@param[in] ### degrs ### a list containing the degrees of the ansatz function per axis
		"""
		
		# --- initialisation of data structures
		
		tmpDegrs = degrs.copy()
		if len(tmpDegrs) == 0:
			for d in range(0, dim):
				tmpDegrs.append(1)
		
		super().__init__(dim, box, rank, tmpDegrs)
	
		self.coeffs = []						# coefficients we are searching for due to an optimisation process
		self.locParams = []						# list of the location parameters of the Gaussian functions
		self.varParams = []						# list of variation parameters corresponding to the Gaussian functions
				
		for d in range(0, self.dim):
			self.coeffs.append(2 * (np.random.rand(self.rank, self.degrs[d]) - 0.5))
			self.locParams.append(np.inf * np.ones(self.degrs[d]))
			self.varParams.append(-np.inf * np.ones(self.degrs[d]))
	
			
	#########################################################################
	# ### getter and setter methods ###
	#########################################################################
	
	
	# --------------------------------------------------
	
	def getCoeffs(self, d):
		return self.coeffs[d].copy()
	
	def setCoeffs(self, d, coeffs):
		self.coeffs[d] = coeffs.copy()
		
	# --------------------------------------------------	
	
	def getLocParams(self, d):
		return self.locParams[d].copy()
		
	def setLocParams(self, d, locParams):
		self.locParams[d] = locParams.copy()
	
	# --------------------------------------------------
	
	def getVarParams(self, d):
		return self.varParams[d].copy()
		
	def setVarParams(self, d, varParams):
		self.varParams[d] = varParams.copy()

	#########################################################################
	# ### initialiser methods ###
	#########################################################################
	
	def initialise(self):
		"""
			this function is used to set the location and variation parameter of the Gaussian ansatz functions
			used in this approach. therefore an equally spaced partition of the intervals corresponding to the
			approximation domain will be used.
		"""
							
		for d in range(0, self.dim):
				
			# REMINDER:
			#	> np.linspace(a, b, n >= 2) creates an array of exactly n equally spaced points in the interval [a, b],
			#	  where the first point in this array corresponds to a and the last one corresponds to b!
			#	> np.linspace(a, b, 1)  leads to an array containing the element a
				
			locParams = np.linspace(self.boundingBox[d][0], self.boundingBox[d][1], self.degrs[d])
			self.setLocParams(d, locParams)

			varParams = ((locParams[1] - locParams[0])) * np.ones(self.degrs[d])
			self.setVarParams(d, varParams)	
			
	#########################################################################
	# ### i/o methods ###
	#########################################################################
	
	def writeParams2File(self, path2ModDir, fileName):
		"""
			this method is used to store all the parameters of an instance of MyCPDRadialAnsatzFunc to file. it should
			be possible then by loading the data from file to reproduce the CPD function.

			write parameters to file in the following way (using ',' as separator)
			
				---------------------------------------------
				> line 1: dim, rank, degrs[0], ... , degrs[d]
				---------------------------------------------
				> line 2: locParams[0]
				---------------------------------------------
				> line 3: varParams[0]
				---------------------------------------------
				> line 4 until line 4 + rank: coeffs[0]
				---------------------------------------------
				...
				---------------------------------------------
			
			@param[in] ### path2ModDir ### string representing the path to the model directory, i.e. the directory
				where the parameters will be stored
			@param[in] ### fileName ### string without file extension
			@return ### successful ### boolean variable indicating if data could have been written to file
			
		"""
		
		currDim = self.getDim()
		currRank = self.getRank()
		currDegrs = self.getDegrs()
		
		successful = False
		
		#########################################################################
		# assume that: 
		#	> current working directory is 'source'
		#########################################################################
	
		if os.path.exists(path2ModDir) == True:
					
			modFile = open(path2ModDir + '/' + fileName + '.txt', 'w')
			
			# first write basic informations to file, like dim, rank, ...
			degrsStr = ''
			for d in range(0, currDim):
				degrsStr += str(currDegrs[d]) + ','
				
			modFile.write(str(currDim) + ',' + str(currRank) + ',' + degrsStr + '\n')
			
			# now write for each dimension the location parameters, variation parameters and finally the
			# coefficients to file ...
			for d in range(0, currDim):
				
				currLocParams = self.getLocParams(d)
				currVarParams = self.getVarParams(d)
				currCoeffs = self.getCoeffs(d)
				
				
				# care about location parameters ...
				for nu in range(0, currDegrs[d] - 1):
					modFile.write(str(currLocParams[nu]) + ',')
				modFile.write(str(currLocParams[currDegrs[d] - 1]) + '\n')
					
				# care about variation parameters ...
				for nu in range(0, currDegrs[d] - 1):
					modFile.write(str(currVarParams[nu]) + ',')
				modFile.write(str(currVarParams[currDegrs[d] - 1]) + '\n')
			
				# ... and finally care about the coefficients ...
				for k in range(0, currRank):
					for nu in range(0, currDegrs[d] - 1):
						modFile.write(str(currCoeffs[k, nu]) + ',')
					modFile.write(str(currCoeffs[k, currDegrs[d] - 1]) + '\n')

			modFile.close()
						
			successful = True
				
		return successful
		
	def readParamsFromFile(self, path2ModDir, fileName):
		"""
			this function is used to read parameter data from file and to set the corresponding member variables.
			
			@param[in] ### path2ModDir ### path to the directory where data is stored
			@param[in] ### fileName ### name of the text file containing the data
			@return ### successful ### boolean variable indicating if reading from file was successful 
		"""
		
		successful = False
		
		if os.path.exists(path2ModDir) == True:
					
			modFile = open(path2ModDir + '/' + fileName + '.txt', 'r')
			
			# read all lines from file
			linesList = modFile.readlines()
			
			# first of all process information contained in the first line: dim, rank, degrees used in each dimension
			fields = linesList[0].split(',')
			
			dim = np.int(fields[0])
			rank = np.int(fields[1])
			tmpDegrs = []
			for d in range(0, dim):
				tmpDegrs.append(np.int(fields[2 + d]))
		
			self.dim = dim		
			self.rank = rank
			
			self.degrs = []
			self.degrs = tmpDegrs.copy()
		
			self.locParams = []
			self.varParams = []
			self.coeffs = []
			
		
			# now read dimension per dimension the location parameters, variation parameters and coefficients from
			# the upfollowing lines
			
			for d in range(0, dim):
				
				fields1 = linesList[1 + d * (2 + rank)].split(',')
				fields2 = linesList[1 + d * (2 + rank) + 1].split(',')
				
				
				tmpLocParams = np.zeros(tmpDegrs[d])
				tmpVarParams = np.zeros(tmpDegrs[d])
				tmpCoeffs = np.zeros((rank, tmpDegrs[d]))
				
				for nu in range(0, tmpDegrs[d]):
					tmpLocParams[nu] = np.float(fields1[nu])
					tmpVarParams[nu] = np.float(fields2[nu])
					
					for k in range(0, rank):					
						fields3 = linesList[1 + d * (2 + rank) + 2 + k].split(',')
						tmpCoeffs[k, nu] = np.float(fields3[nu])
				
								
				self.locParams.append(tmpLocParams)
				self.varParams.append(tmpVarParams)
				self.coeffs.append(tmpCoeffs)
		
			modFile.close()
			successful = True
		
		return successful
	
	#########################################################################
	# ### evaluation methods ###
	#########################################################################
	
	def evalRadBasFunc1d(self, x, d, nu):
		"""
			this function implements the evaluation of one single exponential term in a
			ansatz function.
		
			@param[in] ### x ### scalar or numpy array where function should be evaluated
			@param[in] ### d ### index of the axis, where the exponential term belongs to
			@param[in] ### nu ### index indicating which Gaussian function (corresponding to the desired
					location and variation parameter) should be evaluated
			@return ### retVal ### function value
		"""
		return np.exp(-0.5 * ((x - self.locParams[d][nu]) / self.varParams[d][nu]) ** 2)
		
	def evalLinCombRadBasFunc1d(self, x, d, k):
		"""
			this function implements the evaluation of one ansatz function (which is a weighted sum of exponentials)
			corresponding to one certain axis.
		
			@param[in] ### x ### scalar argument where ansatz function has to be evaluated
			@param[in] ### d ### index of the axis where the ansatz function belongs to
			@param[in] ### k ### rank index, which indicates which ansatz function is meant ...
			
		"""
		
		retVal = np.zeros(x.shape)
		for nu in range(0, self.degrs[d]):	
					
			retVal += self.coeffs[d][k, nu] * self.evalRadBasFunc1d(x, d, nu)

		return retVal
	
	#################################################################################################################
	# the implementation of the functions down below could be done better ... but ok!
	
	def evaluate(self, xList, L):
		"""
			this function implements the evaluation of an ansatz function at a list of points.
		
			@param[in] ### xList ### list containing numpy arrays of data points, where 
				xList[0] corresponds to the coordinates of the points on the x1 axis 
				xList[1] corresponds to the coordinates of the points on the x2 axis
				...
			@param[in] ### L ### number of data points
			@return ### retVal ### numpy array containing the sought function values
			
		"""
		retVal = np.zeros(L)
		
		for k in range(0, self.rank):
			tmp = np.ones(L)
			for d in range(0, self.dim):
				tmp *= self.evalLinCombRadBasFunc1d(xList[d], d, k)
			retVal += tmp
		
		return retVal

	def evaluate2d(self, xList, L):
		"""
			this function has actually the same functionality as evaluate(...). the only difference is in the
			shape of the return value - it is a matrix such that a plot_surface(...) can be called.
			
			@return ### retVal ### 2d numpy array containing the function values in a matrix shape
		"""
		
		retVal = np.zeros((L, L))
		
		for k in range(0, self.rank):
			tmp = np.ones((L, L))
			for d in range(0, self.dim):
				tmp *= self.evalLinCombRadBasFunc1d(xList[d], d, k)
			retVal += tmp
		
		return retVal


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class MyCPDPolyAnsatzFunc(MyBasicCPDAnsatzFunction):
	"""
		This class represents a CPD type of function approximation based on 
		1d polynomials.
		
		NOTE: 
			> ### DEPRECATED ###
	"""
	
	
	def __init__(self, dim, box, rank, degrs = []):
	
		# the list degrs contains the degrees of polynomials and corresponds hence not to the number of parameters !!!
	
		# dummy initialisation of self.degrs if an empty list is provided
		tmpDegrs = degrs.copy()
		if len(tmpDegrs) == 0:
			for d in range(0, dim):
				tmpDegrs.append(1)
			
		super().__init__(dim, box, rank, tmpDegrs)
	
		# default initialisation of data structures

		self.coeffs = []
		
		##### dummy initialisation #####
		
		for d in range(0, self.dim):
			self.coeffs.append(2 * (np.random.rand(self.rank, self.degrs[d] + 1) - 0.5))
		
	#############################################################
			
	# ### getter and setter methods ###
	
	# --------------------------------------------------
	
	def getCoeffs(self, d):
		return self.coeffs[d].copy()
	
	def setCoeffs(self, d, coeffs):
		self.coeffs[d] = coeffs.copy()
		
	# --------------------------------------------------	

	def evalMonomBasFunc1d(self, x, nu):
		return x ** nu
	
	def evalLinCombMonomBasFunc1d(self, x, d, k):
		"""
			actually this function evaluates a polynomial ... but to keep the names consistent ...
		"""
		retVal = np.zeros(x.shape)
		retVal += self.polyCoeffs[d][k, 0]
		for nu in range(1, self.polyDegrs[d] + 1):
	
			retVal += self.polyCoeffs[d][k, nu] * (x ** nu)

		return retVal
	
	def evaluatePolyAugmented(self, xList):
		tmp1 = np.zeros(xList[0].shape)
		tmp2 = np.zeros(xList[0].shape)
		
		# evaluate the rbf part of the ansatz function ...
		tmp1 = self.evaluate(xList)
		
		# evaluate the pbf part of the ansatz function
		for k in range(0, self.polyRank):
			tmp3 = np.ones(xList[0].shape)			
			for d in range(0, self.dim):
				tmp3 *= self.evalLinCombMonomBasFunc1d(xList[d], d, k)
				
			tmp2 += tmp3	
		
		return tmp1 + tmp2
	
	def readRbfAndPolyParamsFromFile(self, path2ModDir, fileName):
		"""
			first call method from super class, and then read the as in the method of the super class ..
		"""
		
		successful = self.readParamsFromFile(path2ModDir, fileName)
		
		if successful:
			
			modFile = open(path2ModDir + '/' + fileName + '.txt', 'r')
			
			# read all lines from file
			linesList = modFile.readlines()
						
			# first of all process information contained in the first line regarding the polyCPD: (dim,) rank, degrees used in each dimension
			fields = linesList[1 + self.dim * (self.rank + 2)].split(',')
			
			# dim = np.int(fields[0])
			pbfRank = np.int(fields[1])
			tmpPolyDegrs = []
			
			for d in range(0, self.dim):
				tmpPolyDegrs.append(np.int(fields[2 + d]))
		
			self.polyRank = pbfRank
			
			self.polyDegrs = []
			self.polyDegrs = tmpPolyDegrs.copy()
		
			self.polyCoeffs = []
			# now step through all the remaining lines and read out the coefficients!
			
			for d in range(0, self.dim):
				tmpPolyCoeffs = np.zeros((self.polyRank, tmpPolyDegrs[d] + 1))
				
				for nu in range(0, tmpPolyDegrs[d] + 1):
					
					for k in range(0, self.polyRank):					
						fields3 = linesList[2 + self.dim * (self.rank + 2) + d * self.polyRank + k].split(',')
						tmpPolyCoeffs[k, nu] = np.float(fields3[nu])
				
				self.polyCoeffs.append(tmpPolyCoeffs)
		
			modFile.close()
		
		return successful
		
	def writeRbfAndPolyParams2File(self, path2ModDir, fileName):
		"""
			first write rbf parameters 2 file by calling the corresponding method of the basis class 
			'MyCPDRadialAnsatzFunc', then append the data corresponding to the polynomial functions to this file.
		"""
		
		successful = False
		
		# write rbf params 2 file 
		successful = successful or self.writeParams2File(path2ModDir, fileName)
		
		# reopen the file and append the data corresponding to the polynomial basis functions
		
		if successful:

			modFile = open(path2ModDir + '/' + fileName + '.txt', 'a+')
			
			currDim = self.getDim()
			currDegrs = self.getDegrs()
			
			# first write basic informations to file, like dim, rank, ...
			degrsStr = ''
			for d in range(0, currDim):
				degrsStr += str(self.polyDegrs[d]) + ','
							
			modFile.write(str(currDim) + ',' + str(self.polyRank) + ',' + degrsStr + '\n')
			
			# now write for each dimension the location parameters, variation parameters and finally the
			# coefficients to file ...
			for d in range(0, currDim):
							
				currCoeffs = self.getPolyCoeffs(d)
												
				# ... and finally care about the coefficients ...
				for k in range(0, self.polyRank):
					for nu in range(0, self.polyDegrs[d]):
						modFile.write(str(currCoeffs[k, nu]) + ',')
					
					modFile.write(str(currCoeffs[k, self.polyDegrs[d]]) + '\n')
				
			modFile.close()
					
			successful = True
			
		
		return successful



class MyPolyAugmentedCPDRadialAnsatzFunc(MyCPDRadialAnsatzFunc):
	"""
		This class represents a CPD type approximation of functions based on polynomial and
		Gaussian ansatz functions.
		
		NOTE:
			> ### DEPRECATED ###
	"""
	
	def __init__(self, dim, box, rbfRank, rbfDegrs, pbfRank, pbfDegrs):
		
		"""
			NOTE:
				> the derived member variables rank, degrs, coeffs, locParams, varParams correspond
					the parameters of the radial basis functions!!!!
					
		"""

		
		super().__init__(dim, box, rbfRank, rbfDegrs)
		
		self.polyRank = pbfRank
		self.polyDegrs = pbfDegrs.copy()
		self.polyCoeffs = []
		
		# NOTE:
		#	> the quantitiy polyDegrs[d] corresponds to the 'real' degree of a polynomial, hence there have to be
		#		polyDegrs[d] + 1 coefficients to characterise a polynomial.
		#	> the coefficients are stored in increasing power order, i.e. the coefficients the polynomial
		#			p(x) = a_n x^n + ... a_1 x + a_0
		#		would be stored as [a_0, a_1, ..., a_n]
		
		for d in range(0, self.dim):
			self.polyCoeffs.append(2 * (np.random.rand(self.polyRank, self.polyDegrs[d] + 1) - 0.5))

		# print(self.polyCoeffs[0])
		# print(self.polyCoeffs[1])
	
	# ### getter and setter methods ###
	
	# --------------------------------------------------
	
	def getPolyCoeffs(self, d):
		return self.polyCoeffs[d].copy()
	
	def setPolyCoeffs(self, d, coeffs):
		self.polyCoeffs[d] = coeffs.copy()

	def getPolyRank(self):
		return self.polyRank
		
	def getPolyDegrs(self):
		return self.polyDegrs
	
	# --------------------------------------------------
	
	def initialiseRbfAndPbfAnsatzFunc(self):
		self.initialise()
	
	def evalMonomBasFunc1d(self, x, nu):
		return x ** nu
	
	def evalLinCombMonomBasFunc1d(self, x, d, k):
		"""
			actually this function evaluates a polynomial ... but to keep the names consistent ...
		"""
		retVal = np.zeros(x.shape)
		retVal += self.polyCoeffs[d][k, 0]
		for nu in range(1, self.polyDegrs[d] + 1):
	
			retVal += self.polyCoeffs[d][k, nu] * (x ** nu)

		return retVal
	
	def evaluatePolyAugmented(self, xList):
		tmp1 = np.zeros(xList[0].shape)
		tmp2 = np.zeros(xList[0].shape)
		
		# evaluate the rbf part of the ansatz function ...
		tmp1 = self.evaluate(xList)
		
		# evaluate the pbf part of the ansatz function
		for k in range(0, self.polyRank):
			tmp3 = np.ones(xList[0].shape)			
			for d in range(0, self.dim):
				tmp3 *= self.evalLinCombMonomBasFunc1d(xList[d], d, k)
				
			tmp2 += tmp3	
		
		return tmp1 + tmp2
	
	def readRbfAndPolyParamsFromFile(self, path2ModDir, fileName):
		"""
			first call method from super class, and then read the as in the method of the super class ..
		"""
		
		successful = self.readParamsFromFile(path2ModDir, fileName)
		
		if successful:
			
			modFile = open(path2ModDir + '/' + fileName + '.txt', 'r')
			
			# read all lines from file
			linesList = modFile.readlines()
						
			# first of all process information contained in the first line regarding the polyCPD: (dim,) rank, degrees used in each dimension
			fields = linesList[1 + self.dim * (self.rank + 2)].split(',')
			
			# dim = np.int(fields[0])
			pbfRank = np.int(fields[1])
			tmpPolyDegrs = []
			
			for d in range(0, self.dim):
				tmpPolyDegrs.append(np.int(fields[2 + d]))
		
			self.polyRank = pbfRank
			
			self.polyDegrs = []
			self.polyDegrs = tmpPolyDegrs.copy()
		
			self.polyCoeffs = []
			# now step through all the remaining lines and read out the coefficients!
			
			for d in range(0, self.dim):
				tmpPolyCoeffs = np.zeros((self.polyRank, tmpPolyDegrs[d] + 1))
				
				for nu in range(0, tmpPolyDegrs[d] + 1):
					
					for k in range(0, self.polyRank):					
						fields3 = linesList[2 + self.dim * (self.rank + 2) + d * self.polyRank + k].split(',')
						tmpPolyCoeffs[k, nu] = np.float(fields3[nu])
				
				self.polyCoeffs.append(tmpPolyCoeffs)
		
			modFile.close()
		
		return successful
		
	def writeRbfAndPolyParams2File(self, path2ModDir, fileName):
		"""
			first write rbf parameters 2 file by calling the corresponding method of the basis class 
			'MyCPDRadialAnsatzFunc', then append the data corresponding to the polynomial functions to this file.
		"""
		
		successful = False
		
		# write rbf params 2 file 
		successful = successful or self.writeParams2File(path2ModDir, fileName)
		
		# reopen the file and append the data corresponding to the polynomial basis functions
		
		if successful:

			modFile = open(path2ModDir + '/' + fileName + '.txt', 'a+')
			
			currDim = self.getDim()
			currDegrs = self.getDegrs()
			
			# first write basic informations to file, like dim, rank, ...
			degrsStr = ''
			for d in range(0, currDim):
				degrsStr += str(self.polyDegrs[d]) + ','
							
			modFile.write(str(currDim) + ',' + str(self.polyRank) + ',' + degrsStr + '\n')
			
			# now write for each dimension the location parameters, variation parameters and finally the
			# coefficients to file ...
			for d in range(0, currDim):
							
				currCoeffs = self.getPolyCoeffs(d)
												
				# ... and finally care about the coefficients ...
				for k in range(0, self.polyRank):
					for nu in range(0, self.polyDegrs[d]):
						modFile.write(str(currCoeffs[k, nu]) + ',')
					
					modFile.write(str(currCoeffs[k, self.polyDegrs[d]]) + '\n')
				
			modFile.close()
					
			successful = True
			
		
		return successful
		
def test():
	
	
	# dim = 2
	# box = []
	# box.append((-3, 3))
	# box.append((-3, 3))
	
	# rbfRank = 2
	# rbfDegrs = []
	# rbfDegrs.append(2)
	# rbfDegrs.append(2)
	
	
	# pbfRank = 1
	# pbfDegrs = []
	# pbfDegrs.append(0)
	# pbfDegrs.append(0)
	
	
	# provide parameters for dummy initialisation
	dim = 0
	bndBox = []
	rbfRnk = 0
	pbfRnk = 0
	
	func = MyPolyAugmentedCPDRadialAnsatzFunc(dim, bndBox, rbfRnk, [], pbfRnk, [])
	func.readRbfAndPolyParamsFromFile('.', 'test')
	
	print('rbf infos:')
	print(func.getDim())
	print(func.getRank())
	print(func.getDegrs())
	print(func.getCoeffs(0))
	print(func.getCoeffs(1))
	
	
	print('pbf infos:')
	print(func.getPolyRank())
	print(func.getPolyCoeffs(0))
	print(func.getPolyCoeffs(1))
	
	# func = MyPolyAugmentedCPDRadialAnsatzFunc(dim, box, rbfRank, rbfDegrs, pbfRank, pbfDegrs)
	# func.initialiseRbfAndPbfAnsatzFunc()
	
	
	box = []
	box.append((-3, 3))
	box.append((-3, 3))
	
	M = 500
	x1 = np.random.uniform(box[0][0], box[0][1], M)
	x2 = np.random.uniform(box[1][0], box[1][1], M)
	
	z = func.evaluatePolyAugmented([x1, x2])
	
	from matplotlib import pyplot as plt	
	from mpl_toolkits.mplot3d import Axes3D

	f0 = plt.figure()
	ax = f0.gca(projection = '3d')
	ax.scatter(x1, x2, z, s = 2, c = 'r')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	

	plt.show()
	
	# func.writeRbfAndPolyParams2File('~/Documents/ProgrammingInPython/MasterThesis/source', 'test_')
	# func.writeRbfAndPolyParams2File('.', 'test')
	
	
	
if __name__ == '__main__':
	test()
