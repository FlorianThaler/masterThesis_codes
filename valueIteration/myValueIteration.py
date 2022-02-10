"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements the main function
					of the value iteration package.
	Year:			2019/2020
"""

########################################################################
# importing stuff
########################################################################

import numpy as np

import logging

import sys
import os

import multiprocessing as mp

from matplotlib import pyplot as plt

from myExamples.my1dExample_a import my1dExample_a
from myExamples.my1dExample_a import simulateFromFile_1d_a

from myExamples.my2dExample_a import my2dExample_a
from myExamples.my2dExample_a import simulateFromFile_2d_a

from myExamples.my3dExample_b import my3dExample_b
from myExamples.my3dExample_b import simulateFromFile_3d_b

from myExamples.my3dExample_c import my3dExample_c
from myExamples.my3dExample_c import simulateFromFile_3d_c

from myExamples.my3dExample_d import my3dExample_d
from myExamples.my3dExample_d import simulateFromFile_3d_d

########################################################################
# training runner function
########################################################################

def myTrainingRunnerFunc(dim, paramList, labelList, exampleFunc, lamList, signature, path2SignDir, \
		figuresDirName = 'figures', modelDirName = 'model', perfDataDirName = 'perfData'):
	
	for idx, label in zip(np.arange(0, len(labelList)), labelList):

		subSignature = signature + '_' + str(label)
		
		print('### start training subsession corresponding to the subsignature # ' + subSignature + ' #')
		print('###############################################################################')
		logging.info('### start training subsession corresponding to the subsignature # ' + subSignature + ' #')
		logging.info('###############################################################################')

		####################################################################
		# create subdirectories where performance data, setup data, ... will
		#	will be stored.
		####################################################################

		path2SubSignDir = path2SignDir + '/' + subSignature
		path2FigDir = path2SubSignDir + '/' + figuresDirName
		path2ModDir = path2SubSignDir + '/' + modelDirName
		path2PerfDataDir = path2SubSignDir + '/' + perfDataDirName
		
		os.makedirs(path2SubSignDir)
		os.makedirs(path2FigDir)
		os.makedirs(path2ModDir)
		os.makedirs(path2PerfDataDir)
											
		####################################################################
		# provide data for the optimisation process
		####################################################################

		tmp = paramList[idx]
		currRank = tmp[0]
		currDegrs = []
		for d in range(0, dim):
			currDegrs.append(tmp[d + 1])

		L = tmp[len(tmp) - 1]

		####################################################################
		# start optimisation right here ...
		####################################################################
					
		exampleFunc(L, currRank, currDegrs, signature, subSignature, path2SubSignDir, path2FigDir,\
				path2ModDir, path2PerfDataDir)

########################################################################
# main function
########################################################################

def main():
	
	####################################################################
	# some initialisations ...
	####################################################################
	
	resultsDirName = '../../results/valueIteration'
	figuresDirName = 'figures'
	modelDirName = 'model'
	perfDataDirName = 'perfData'
	
	####################################################################
	#
	# command line arguments:
	#	> in any case:
	#		+ argv[0]: name of the script (... in any case)
	#	> here:
	#		+ argv[1]: --t (for training) or --v (for visualisation from file)
	#		+ argv[2]: one of the strings '1d', '2d', '3d', '5d'
	# 		+ argv[3]: session id, i.e. '1', '2', ...
	# 		+ argv[4]: subSession id, i.e. '1', '2', ...
	# 		+ argv[5]: iteration id, i.e. '1', '2', ...
	#
	#	  NOTE: the latter two parameters are of interest only in the '--v' case
	####################################################################
	
	# process the command line parameters properly ...
	mode = sys.argv[1]
	param1 = sys.argv[2]
	param2 = np.int(sys.argv[3])
	if mode == '--v':
		param3 = np.int(sys.argv[4])
		param4 = np.int(sys.argv[5])
	
	
	signature = param1 + '_' + str(param2)
	
	if mode == '--t':
	
		####################################################################
		# provide directories where results will be stored if desired ...
		# ... and prepare logging stuff
		####################################################################
		
		stop = False
		
		if (os.path.exists('./' + resultsDirName) == False):
			os.makedirs('./' + resultsDirName)

		path2SignDir = resultsDirName + '/' + signature
		
		if (os.path.exists(path2SignDir) == False):
			os.makedirs(path2SignDir)
			
			####################################################################
			# do some logging ...
			####################################################################
			
			logging.basicConfig(level = logging.INFO,
				format = '%(asctime)s %(levelname)-8s %(message)s',
				datefmt = '%a,%d,%b,%Y,%H:%M:%S',
				filename = path2SignDir + '/' + signature + '.log',
				filemode = 'w')
		
		else:
			stop = True
			print('### ERROR ### there is already a training session corresponding to the signature ' + signature)
		
		####################################################################
		#
		# if training session does not already exist, start training!
		#
		####################################################################
	
		# if training session already exists, than abort ...
		if stop == False:
		
			print('### start training session corresponding to the signature # ' + signature + ' #')
			print('###############################################################################')
			logging.info('### start training session corresponding to the signature # ' + signature + ' #')
			logging.info('###############################################################################')
			
			################################################################################################################
			# ### 1d example ###
			################################################################################################################

			if param1 == '1d_a':
			
				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				
				dim = 1
				rankList = [2, 4]
				degrList = [5, 10]
				
				numDataPtsFactorList = [10]
				
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# start optimisation
				####################################################################
				
				multiProc = True
				
				if multiProc == True:
					
					numProcs = 4
					
					N = np.int(len(paramList) / numProcs)
					
					prcsList = []

					# create processes
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N], my1dExample_a,\
							signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], my1dExample_a,\
						signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
													
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					myTrainingRunnerFunc(dim, paramList, labelList, my1dExample_a, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)
						
			################################################################################################################
			
			elif param1 == '1d_b':
				pass
			
			################################################################################################################

			elif param1 == '2d_a':
			
				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				

				# paramList.append((13, 5, 5, 13 * 5 * 5 * 10))		
				
				# paramList.append((13, 3, 3, 500))		# is okay ...
				
				dim = 2
				rankList = [8, 10, 16, 20]
				# degrList = [2, 6, 10, 14, 18]
				degrList = [8]
				numDataPtsFactorList = [10]
				
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# start optimisation 
				####################################################################
						
				multiProc = True
				
				if multiProc == True:
					
					numProcs = 4
					
					N = np.int(len(paramList) / numProcs)
					

					prcsList = []

					# create processes
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N], my2dExample_a,\
							signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], my2dExample_a,\
						signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
								
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					myTrainingRunnerFunc(dim, paramList, labelList, my2dExample_a, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)

			################################################################################################################
				
			elif param1 == '2d_b':
			
				pass

			################################################################################################################

			elif param1 == '3d_a':
						
				pass
			
			################################################################################################################
					
			elif param1 == '3d_b':

				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				
					# paramList.append((13, 3, 3, 3, 500))
					# paramList.append((7, 9, 9, 9, 5000))
					# paramList.append((17, 19, 19, 19, 8000))				
					# paramList.append((23, 9, 9, 9, 7000))	# working good for large angular velocities
				
				dim = 3
					# rankList = [16, 24, 32]
				rankList = [16]
				degrList = [24]
					# degrList = [24, 32]
				numDataPtsFactorList = [10]
		
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
								
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# start optimisation ...
				####################################################################
				
				multiProc = False
				
				if multiProc == True:
					
					numProcs = 6
					
					N = np.int(len(paramList) / numProcs)
					
					prcsList = []

					# create processes ....
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N], my3dExample_b,\
							signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], my3dExample_b,\
						signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
														
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					myTrainingRunnerFunc(dim, paramList, labelList, my3dExample_b, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)
				
			################################################################################################################

			elif param1 == '3d_b_pC':
				
				# probability constraints are considered only in this example

				paramList = []			
				
				dim = 3
				rankList = [16]
				degrList = [24]
				numDataPtsFactorList = [20]
		
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
								
				# NOTE
				# 	> it is iterated over lamList and not over paramList - thus only one entry in paramList is considered here 
				lamList = np.linspace(1.7, 1.9, 11)
				paramIdx = 0
				labelList = np.arange(0, len(lamList)).tolist()

				def dummyFunc_3d_b_pC(partialLamList, partialLabelList):

					for idx, label in zip(np.arange(0, len(partialLabelList)), partialLabelList):

						subSignature = signature + '_' + str(label)
						
						print('### start training subsession corresponding to the subsignature # ' + subSignature + ' #')
						print('###############################################################################')
						logging.info('### start training subsession corresponding to the subsignature # ' + subSignature + ' #')
						logging.info('###############################################################################')

						####################################################################
						# create subdirectories where performance data, setup data, ... will
						#	will be stored.
						####################################################################

						path2SubSignDir = path2SignDir + '/' + subSignature
						path2FigDir = path2SubSignDir + '/' + figuresDirName
						path2ModDir = path2SubSignDir + '/' + modelDirName
						path2PerfDataDir = path2SubSignDir + '/' + perfDataDirName
						
						os.makedirs(path2SubSignDir)
						os.makedirs(path2FigDir)
						os.makedirs(path2ModDir)
						os.makedirs(path2PerfDataDir)
																
						####################################################################
						# provide data for the optimisation process
						####################################################################

						tmp = paramList[paramIdx]
						currRank = tmp[0]
						currDegrs = []
						for d in range(0, dim):
							currDegrs.append(tmp[d + 1])

						L = tmp[len(tmp) - 1]

						####################################################################
						# start optimisation right here ...
						####################################################################
						
						my3dExample_b(L, currRank, currDegrs, signature, subSignature, path2SubSignDir, path2FigDir,\
							path2ModDir, path2PerfDataDir, lam = partialLamList[idx])
				
				multiProc = True			
				if multiProc == True:
					
					numProcs = 10
					N = np.int(len(lamList) / numProcs)
					prcsList = []

					# create processes ....
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = dummyFunc_3d_b_pC, args = (lamList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N])))
					prcsList.append(mp.Process(target = dummyFunc_3d_b_pC, args = (lamList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ])))
								
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					dummyFunc_3d_b_pC(lamList, labelList)

			################################################################################################################

			elif param1 == '3d_c':
				
				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				

					# paramList.append((13, 3, 3, 3, 500))
					# paramList.append((7, 9, 9, 9, 5000))
					# paramList.append((17, 19, 19, 19, 8000))
					# paramList.append((3, 5, 5, 250))
					# paramList.append((7, 17, 200))
					# paramList.append((7, 27, 500))
					
					# paramList.append((3, 5, 5, 5, 700))
				
				dim = 3
				rankList = [3, 5, 7, 9, 11]
				degrList = [5, 10, 15, 20, 25, 30]
				numDataPtsFactorList = [5, 10]
		
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# start optimisation
				####################################################################
				
				multiProc = False
				
				if multiProc == True:
					
					numProcs = 12
					
					N = np.int(len(paramList) / numProcs)
					

					prcsList = []

					# create processes ....
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N], my3dExample_c,\
							signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], my3dExample_c,\
						signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
								
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					myTrainingRunnerFunc(dim, paramList, labelList, my3dExample_c, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)
				
			################################################################################################################


			elif param1 == '3d_d':

				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				
				# paramList.append((23, 9, 9, 9, 7000))	
				
				dim = 3
				# rankList = [5, 10, 15, 20, 25, 30, 35]
				rankList = [5, 10, 15]
				degrList = [10, 15, 20, 25, 30, 40, 50]
				numDataPtsFactorList = [10]
		
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# start optimisation
				####################################################################
				
				multiProc = True
				
				if multiProc == True:
					
					numProcs = 4
					
					N = np.int(len(paramList) / numProcs)
					
					prcsList = []

					# create processes ....
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N], my3dExample_d,\
							signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], my3dExample_d,\
						signature, path2SignDir,figuresDirName, modelDirName, perfDataDirName)))								
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
					
				else:
					myTrainingRunnerFunc(dim, paramList, labelList, my3dExample_d, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)				
				
			################################################################################################################

			else:
				print('### ERROR ### no valid command line parameter')
				
	elif mode == '--v':
		
		path2SignDir = resultsDirName + '/' + signature
		numSubDirs = sum(os.path.isdir(path2SignDir + '/' + elem) for elem in os.listdir(path2SignDir))
		
		if param1 == '1d_a':
			for l in range(0, numSubDirs):
			# for l in range(27, 28):
				simulateFromFile_1d_a(param2, str(l), param4)
		if param1 == '2d_a':
			for l in range(0, numSubDirs):
			# for l in range(17, 18):
				simulateFromFile_2d_a(param2, str(l), param4)
		if param1 == '3d_b':
			# simulateFromFile_3d_b(param2, param3, param4)
			for l in range(0, numSubDirs):
				simulateFromFile_3d_b(param2, str(l), param4)
		if param1 == '3d_b_pC':
			
			# lamList = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, \
								# 6.0, 7.0, 8.0, 9.0, 10.0])
			lamList = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, \
								6.0, 7.0, 8.0, 9.0, 10.0])
			# lamList = np.linspace(1.7, 1.9, 11)
			# lamList = np.linspace(1.6, 1.8, 11)
			resArr = np.zeros(numSubDirs)
			for l in range(0, len(lamList)):
				resArr[l] = simulateFromFile_3d_b(param2, str(l), param4, True, lamList[l])
			# print('lambda:')
			# print(lamList)
			# print('result:')
			# print(resArr)
			
			# fig = plt.figure()
			# plt.plot(np.asarray(lamList), resArr)
			# plt.show()
			
		if param1 == '3d_c':
			simulateFromFile_3d_c(param2, param3, param4)
			# for l in range(0, numSubDirs):
				# simulateFromFile_3d_c(param2, str(l), param4)
		if param1 == '3d_d':
			for l in range(7, 8):
				simulateFromFile_3d_d(param2, str(l), param4)
	else:
		pass
########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
	main()
