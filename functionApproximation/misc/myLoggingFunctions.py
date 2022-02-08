"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements functions for the
					purpose of logging setup, results, ...
	Year:			2019
"""

###########################################################
# importing stuff
###########################################################

import datetime
import platform
import os

import numpy as np
from matplotlib import pyplot as plt

###########################################################
# definition of functions
###########################################################

def createSetupFile(alsParamsDict, cgParamsDict, cpdParamsDict, path2StpUpDataDir):
	"""
		@param[in] ### alsParamsDict ### dictionary containing als parameters
		@param[in] ### cgParamsDict ### dictionary containing cg parameters
		@param[in] ### cpdParamsDict ### dictionary containing cpd parameters
		@param[in] ### path2StpUpDataDir ### path to directory where setup file should be stored
		
		this function simply writes general data about the optimisation process (als parameters, ...) to file
	"""

	# open file
	stpFile = open(path2StpUpDataDir + '/' + 'setup.txt', 'w')
	
	# ... write to file ...
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('date: ' + str(datetime.datetime.now()) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('##### system details #####\n')
	stpFile.write('\n')                       
	stpFile.write(' + platform = ' + platform.system() + '\n')
	stpFile.write(' + number of available CPU cores (physical) = ' + str(os.cpu_count()) + '\n')
	stpFile.write('\n')                 
	stpFile.write('######################################################################\n')                       
	stpFile.write('\n')
	stpFile.write('##### ALS details #####\n')
	stpFile.write('\n')
	for key in alsParamsDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(alsParamsDict[key]) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('##### CG details #####\n')
	stpFile.write('\n')
	for key in cgParamsDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(cgParamsDict[key]) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('##### CPD details #####\n')
	stpFile.write('\n')
	for key in cpdParamsDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(cpdParamsDict[key]) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	
	# ... and finally close the file
	stpFile.close()    

def	writePerformanceData2File(costFuncValList, mseApprErrList, lInfApprErrList, cgPerformanceList, path2PerfDataDir):
	"""
		@param[in] ### costFuncValList ### list containing the values of the cost functional over the optimisation
			process
		@param[in] ### apprErrList ### list containing the approximation errors
		@param[in] ### cgPerformanceList ### list containing information on the performance of the cg method: number
			of iterations, reason why the iteration stopped, ...
		@param[in] ### path2PerfDataDir ### string giving the path to the directory where data should be stored.
		
		this function is used to write the data obtained due to the approximation process to file. this are basically
		values of the cost functional, approximation errors.
	"""
	
	##################################################################
	# first of all write approximation data 2 file ...
	##################################################################
	
	apprDataFile = open(path2PerfDataDir + '/performance.txt', 'w')
	
	apprDataFile.write(' # costFuncVal # |')
	# apprDataFile.write(' # l2ApprErr # |')
	apprDataFile.write(' # mseApprErr # |')
	apprDataFile.write(' # lInfApprErr #\n')
	
	N = len(costFuncValList)
	
	for i in range(0, N - 1):
		# apprDataFile.write(str(costFuncValList[i]) + ',' + str(l2ApprErrList[i]) + ',' + str(lInfApprErrList[i]) + '\n')
		apprDataFile.write(str(costFuncValList[i]) + ',' + str(mseApprErrList[i]) + ',' + str(lInfApprErrList[i]) + '\n')
	# apprDataFile.write(str(costFuncValList[N - 1]) + ',' + str(l2ApprErrList[N - 1]) + ',' + str(lInfApprErrList[N - 1]))
	apprDataFile.write(str(costFuncValList[N - 1]) + ',' + str(mseApprErrList[N - 1]) + ',' + str(lInfApprErrList[N - 1]))
	
	apprDataFile.close()
	
	##################################################################
	# now deal with the cg performance data ...
	##################################################################
	
	cgDataFile = open(path2PerfDataDir + '/cgData.txt', 'w')
	
	cgDataFile.write(' # optimisation coordinate # |')
	cgDataFile.write(' # number of cg iterations # |')
	cgDataFile.write(' # resNorm # |')
	cgDataFile.write(' # reason cg iteration stopped #\n')
	
	dim = len(cgPerformanceList)
	N = len(cgPerformanceList[0])
	
	for d in range(0, dim):
		for i in range(0, N - 1):
			cgDataFile.write(str(d + 1) + ',' + str(cgPerformanceList[d][i][0]) + ',' + \
				str(cgPerformanceList[d][i][1]) + ',' + cgPerformanceList[d][i][2] + '\n')
		cgDataFile.write(str(d + 1) + ',' + str(cgPerformanceList[d][N - 1][0]) + ',' + \
				str(cgPerformanceList[d][N - 1][1]) + ',' + cgPerformanceList[d][N - 1][2] + '\n')
	
	cgDataFile.close()

def computeEOC(path2ModDir, fileName, path2FigDir, storeFig = True):
	"""
		@param[in] ### path2ModDir ### string corresponding to the path to the directory where model files are stored
		@param[in] ### fileName ### string corresponding to the base of the name of the files containing
			information of the parameters
		@param[in] ### path2FigDir ### string corresponding to the path to the directory where figures which are
			going to be produced will be stored
			
		this function reads the model data, i.e. the parameters, from file and computes then the experimental order
		of convergence. a figure showing the evolution of the eocs will be produced and written to file as well.
	"""
	
	numModFiles = len(os.listdir(path2ModDir))
	# there have to be enough different parameters in order to compute the eoc ...
	successful = (numModFiles >= 3)
	
	if successful:
		
		coeffsList = []
		eocList = []
		
		# read parameter of all of the model files, i.e. the model files corresponding to each of the training episodes.
		for i in range(0, numModFiles):
			
			path2ModFile = path2ModDir + '/' + fileName + str(i) + '.txt'
			
			tmpList = []		
			modFile = open(path2ModFile, 'r')
			
			# read all lines from file
			linesList = modFile.readlines()
			
			# first of all process information contained in the first line: dim, rank, degrees used in each dimension
			fields = linesList[0].split(',')
			
			dim = np.int(fields[0])
			rank = np.int(fields[1])
			tmpDegrs = []
			for d in range(0, dim):
				tmpDegrs.append(np.int(fields[2 + d]))
		
		
			# now read dimension per dimension the location parameters, variation parameters and coefficients from
			# the upfollowing lines
			
			for d in range(0, dim):
				
				fields1 = linesList[1 + d * (2 + rank)].split(',')
				fields2 = linesList[1 + d * (2 + rank) + 1].split(',')
								
				tmpCoeffs = np.zeros((rank, tmpDegrs[d]))
				
				for nu in range(0, tmpDegrs[d]):
					for k in range(0, rank):					
						fields3 = linesList[1 + d * (2 + rank) + 2 + k].split(',')
						tmpCoeffs[k, nu] = np.float(fields3[nu])
				
				tmpList.append(tmpCoeffs)
		
			modFile.close()			
			coeffsList.append(tmpList)


		# now that all the parameters are accessible we can compute the eoc
		i = 0
		while (i + 2 < numModFiles - 1):
			
			diffNrmList1 = []
			diffNrmList2 = []
			diffNrmList3 = []
			
			for j in range(0, dim):
				diffNrmList1.append(np.linalg.norm(coeffsList[i][j] - coeffsList[-1][j], 'fro'))
				diffNrmList2.append(np.linalg.norm(coeffsList[i + 1][j] - coeffsList[-1][j], 'fro'))
				diffNrmList3.append(np.linalg.norm(coeffsList[i + 2][j] - coeffsList[-1][j], 'fro'))
				# diffNrmList1.append(np.linalg.norm(coeffsList[i][j] - coeffsList[-1][j], np.inf))
				# diffNrmList2.append(np.linalg.norm(coeffsList[i + 1][j] - coeffsList[-1][j], np.inf))
				# diffNrmList3.append(np.linalg.norm(coeffsList[i + 2][j] - coeffsList[-1][j], np.inf))
				# diffNrmList1.append(np.linalg.norm(coeffsList[i][j] - coeffsList[-1][j], 1))
				# diffNrmList2.append(np.linalg.norm(coeffsList[i + 1][j] - coeffsList[-1][j], 1))
				# diffNrmList3.append(np.linalg.norm(coeffsList[i + 2][j] - coeffsList[-1][j], 1))
				
			diffNrm1 = max(diffNrmList1)
			diffNrm2 = max(diffNrmList2)
			diffNrm3 = max(diffNrmList3)
							
			eocList.append(np.log(diffNrm2 / diffNrm3) / np.log(diffNrm1 / diffNrm2))
			
			i += 1
		
		if storeFig:
			# now finally plot the eoc		
			fig = plt.figure()
			plt.plot(np.arange(0, len(eocList)), eocList)
			plt.title('Experimental order of convergence')
			plt.xlabel('Level')
			plt.savefig(path2FigDir + '/eocPlot.png', dpi = 300)
			plt.close(fig)
				
	return successful, eocList
	
