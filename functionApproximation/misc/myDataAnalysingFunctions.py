"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements functions to 
					visualise the results of the function approximation procedure.
	Year:			2019
"""

###########################################################
# importing stuff
###########################################################

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import re
import os

###########################################################
# definition of functions
###########################################################

def analyseCPUTimes(dim, path2SignDir):
	csvData = pd.read_csv(path2SignDir + '/analysis.csv')
	
	print(csvData['m'])
	
def myTrainingDataAnalyser(dim, path2SignDir):
	"""
		@param[in] ### dim ### integer corresponding to the dimensionality of the problem
		@param[in] ### path2SignDir ### string corresponding to the directory of the current session. there all the 
			directories of the 'subsessions' with its model files, performance files, ... are contained.
		
		this function reads all the performance data of the optimisation process from file (after all the optimisation
		processes are finished!!!) and stores them in tabular form to a csv file.
		NOTE:
			> only the last values in the performance files will be used!!!
	"""
	
	setupFileName = 'setup.txt'
	perfDataDirName = 'perfData'
	perfDataFileName = 'performance.txt'

	# initialise some variables
	rank = -1
	numDataPoints = -1
	
	# provide pandas data structure where data will be stored ...
	dataFrame = pd.DataFrame()
			
	# find all subdirectories in the current directory and ...
	subDirList = [elem for elem in os.listdir(path2SignDir) if os.path.isdir(path2SignDir + '/' + elem)]

	# --------------------------------------------
	# --- read data from file 
	# --------------------------------------------
	
	degrCrdntStrgList = []
	for d in range(0, dim):
		degrCrdntStrgList.append('degrCrdnt' + str(d + 1))
	
	for subDir in subDirList:	
		degrList = []
		
		# first read stuff from setup file ...
		
		path2SubDir = path2SignDir + '/' + subDir
		path2SetupFile = path2SubDir + '/' + setupFileName
		path2PerfDataFile = path2SignDir + '/' + subDir + '/' + perfDataDirName + '/' + perfDataFileName
		
	
		stpFile = open(path2SetupFile, 'r')
		for line in stpFile:
			if 'numDataPoints' in line:
				numDataPoints = np.int(re.findall('\d+$', line)[0])
			elif 'rank' in line:
				rank = np.int(re.findall('\d+$', line)[0])
			elif any(strg in line for strg in degrCrdntStrgList):
				degrList.append(np.int(re.findall('\d+$', line)[0]))
		
		stpFile.close()
	
		# now read stuff from performance file
		perfDataFile = open(path2PerfDataFile, 'r')
		

		# count the number of lines in the file:
		#	> number of lines - 1 = number of ALS iterations
		#	> last line contains the sought final approximation data
			
		linesList = perfDataFile.readlines()
		numALSIter = len(linesList) - 1
		
		fields = linesList[numALSIter].split(',')
		
		costFuncVal = np.float(fields[0])
		mse = np.float(fields[1])
		lInfApprErr = np.float(fields[2])

		perfDataFile.close()
	

		# now put data in data frame
		
		tmpDict = {}
		tmpDict['numDataPoints'] = numDataPoints
		tmpDict['rank'] = rank
		for i in range(0, len(degrCrdntStrgList)):
			tmpDict[degrCrdntStrgList[i]] = degrList[i]
		tmpDict['numALSIter'] = numALSIter
		tmpDict['costFuncVal'] = costFuncVal
		tmpDict['mse'] = mse
		tmpDict['lInfApprErr'] = lInfApprErr
		
		
		dnmntr = 0
		for d in range(0, dim):
			dnmntr += rank * degrList[d]
		tmpDict['m'] = numDataPoints / (dnmntr)
		
		tmpDataFrame = pd.DataFrame(tmpDict.items())
		tmpDataFrame = tmpDataFrame.transpose()
		
		tmpDataFrame = tmpDataFrame.rename(index = {0 : '', 1 : subDir})
		
		tmpDataFrame.columns = tmpDataFrame.iloc[0]
		
		tmpDataFrame = tmpDataFrame.drop(tmpDataFrame.index[[0]])
		
		dataFrame = dataFrame.append(tmpDataFrame)
		
	# --------------------------------------------
	# --- gather data which should be depicted:
	#		> first w.r.t. to l^2 norm
	#		> then w.r.t. to l^inf norm
	# --------------------------------------------
	
	# ### l2 norm ### or ### mse ###
	
	# sort by number of data points 
	# dataFrame.sort_values(by = ['numDataPoints'], inplace = True)
	# arr1 = pd.Series(dataFrame['numDataPoints']).to_numpy(dtype = np.int)
	dataFrame.sort_values(by = ['m'], inplace = True)
	arr1 = pd.Series(dataFrame['m']).to_numpy(dtype = np.int)
	arr2 = pd.Series(dataFrame['mse']).to_numpy(dtype = np.float)
	p1 = np.poly1d(np.polyfit(arr1, arr2, 1))
			
	# sort by rank 
	dataFrame.sort_values(by = ['rank'], inplace = True)
	arr3 = pd.Series(dataFrame['rank']).to_numpy(dtype = np.int)
	arr4 = pd.Series(dataFrame['mse']).to_numpy(dtype = np.float)
	p2 = np.poly1d(np.polyfit(arr3, arr4, 1))
	
	# sort by degree 
	dataFrame.sort_values(by = ['degrCrdnt1'], inplace = True)
	arr5 = pd.Series(dataFrame['degrCrdnt1']).to_numpy(dtype = np.int)
	arr6 = pd.Series(dataFrame['mse']).to_numpy(dtype = np.float)
	p3 = np.poly1d(np.polyfit(arr5, arr6, 1))
	
	# ### lInf norm
	
	# sort by number of data points
	# dataFrame.sort_values(by = ['numDataPoints'], inplace = True)
	# arr7 = pd.Series(dataFrame['numDataPoints']).to_numpy(dtype = np.int)
	dataFrame.sort_values(by = ['m'], inplace = True)
	arr7 = pd.Series(dataFrame['m']).to_numpy(dtype = np.int)
	arr8 = pd.Series(dataFrame['lInfApprErr']).to_numpy(dtype = np.float)
	p4 = np.poly1d(np.polyfit(arr7, arr8, 1))
			
	# sort by rank
	dataFrame.sort_values(by = ['rank'], inplace = True)
	arr9 = pd.Series(dataFrame['rank']).to_numpy(dtype = np.int)
	arr10 = pd.Series(dataFrame['lInfApprErr']).to_numpy(dtype = np.float)
	p5 = np.poly1d(np.polyfit(arr9, arr10, 1))
	
	# sort by degree
	dataFrame.sort_values(by = ['degrCrdnt1'], inplace = True)
	arr11 = pd.Series(dataFrame['degrCrdnt1']).to_numpy(dtype = np.int)
	arr12 = pd.Series(dataFrame['lInfApprErr']).to_numpy(dtype = np.float)
	p6 = np.poly1d(np.polyfit(arr11, arr12, 1))
	
	
	# --------------------------------------------
	# --- finally produce plots
	# --------------------------------------------
	
	fig = plt.figure()

	plt.rc('axes', titlesize = 8)
	plt.rc('axes', labelsize = 6)
	plt.rc('xtick', labelsize = 4)
	plt.rc('ytick', labelsize = 4)
	
	# ### row 1
	
	plt.subplot(3, 2, 1)
	plt.plot(arr1, arr2, 'o', markersize = 1)
	plt.plot(arr1, p1(arr1), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax1 = plt.gca()
	# ax1.set_title('number of data points vs. ' + r'$l^{2}$' + ' approximation error')
	ax1.set_title('m vs. mse')
	ax1.set_xlabel('m')

	plt.subplot(3, 2, 2)
	plt.plot(arr7, arr8, 'o', markersize = 1)
	plt.plot(arr7, p4(arr7), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax1 = plt.gca()
	ax1.set_title('m vs. ' + r'$l^{\infty}$' + ' approximation error')
	ax1.set_xlabel('m')

	# ### row 2

	plt.subplot(3, 2, 3)
	plt.plot(arr3, arr4, 'o', markersize = 1)
	plt.plot(arr3, p2(arr3), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax2 = plt.gca()
	ax2.set_title('r vs. mse')
	ax2.set_xlabel('r')
	
	plt.subplot(3, 2, 4)
	plt.plot(arr9, arr10, 'o', markersize = 1)
	plt.plot(arr9, p5(arr9), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax2 = plt.gca()
	ax2.set_title('r vs. ' + r'$l^{\infty}$' + ' approximation error')
	ax2.set_xlabel('r')
	
	# ### row 3
	
	plt.subplot(3, 2, 5)
	plt.plot(arr5, arr6, 'o', markersize = 1)
	plt.plot(arr5, p3(arr5), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax2 = plt.gca()
	ax2.set_title('N vs. mse')
	ax2.set_xlabel('N')
	
	plt.subplot(3, 2, 6)
	plt.plot(arr11, arr12, 'o', markersize = 1)
	plt.plot(arr11, p6(arr11), 'r', linewidth = 0.5)
	plt.yscale('log')
	ax2 = plt.gca()
	ax2.set_title('N vs. ' + r'$l^{\infty}$' + ' approximation error')
	ax2.set_xlabel('N')
	
	plt.subplots_adjust(hspace = 1.3)
	plt.subplots_adjust(wspace = 1.0)

	plt.tight_layout()
	
	plt.savefig(path2SignDir + '/analysis.png', dpi = 300)
	dataFrame.to_csv(path2SignDir + '/analysis.csv')
