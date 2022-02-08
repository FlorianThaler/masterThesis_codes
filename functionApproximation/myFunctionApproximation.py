"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements the main function
					of the function approximation package. 
	Year:			2019/2020
"""

########################################################################
# importing stuff
########################################################################

import numpy as np

import random

import logging

import platform
import datetime

import time

import sys
import os

import multiprocessing as mp

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from misc.myDataAnalysingFunctions import myTrainingDataAnalyser
from misc.myDataAnalysingFunctions import analyseCPUTimes

from misc.myVisualisingFunctions import createComparativePerfPlot
from misc.myVisualisingFunctions import createComparativeSurfPlot
from misc.myLoggingFunctions import computeEOC

from cpd.myCPDAnsatzFunctions import MyCPDRadialAnsatzFunc
from myExamples.my2dExample import my2dExample
from myExamples.my3dExample import my3dExample
from myExamples.my4dExample import my4dExample

########################################################################################################################

########################################################################
# training runner function
########################################################################

def myTrainingRunnerFunc(dim, paramList, labelList, exampleFunc, signature, path2SignDir, \
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
		# start optimisation right here
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
	
	resultsDirName = '../../results/functionApproximation'
	figuresDirName = 'figures'
	modelDirName = 'model'
	perfDataDirName = 'perfData'
	
	####################################################################
	#
	# command line arguments:
	#	> in any case:
	#		+ argv[0]: name of the script
	#	> here:
	#		+ argv[1]: --t (for training) or --a (for analysing)
	#		+ argv[2]: one of the strings '2d', '3d', '5d'
	# 		+ argv[3]: session id, i.e. '1', '2', ...
	#
	####################################################################
	
	# process the command line parameters properly ...
	mode = sys.argv[1]
	param1 = sys.argv[2]
	param2 = np.int(sys.argv[3])
	
	signature = param1 + '_' + str(param2)
	
	####################################################################
	#
	# provide directories where results will be stored ... and prepare
	# logging stuff
	#
	####################################################################
	
	if mode == '--t':
		
		####################################################################
		# training mode
		####################################################################

		stop = False
			
		if (os.path.exists('./' + resultsDirName) == False):
			os.makedirs(resultsDirName)
		
		path2SignDir = resultsDirName + '/' + signature
		
		if (os.path.exists(path2SignDir) == False):
			os.makedirs(path2SignDir)
			
			####################################################################
			# create logging file
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
		
			################################################################################################################
			# ### 2d example ###
			################################################################################################################
		
			if param1 == '2d':

				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				
				dim = 2
				rankList = [16]
				degrList = [24]
				# numDataPtsFactorList = [5, 10, 20, 30]
				numDataPtsFactorList = [10]
				
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()

				####################################################################
				# run training
				####################################################################
				
				myTrainingRunnerFunc(dim, paramList, labelList, my2dExample, signature, path2SignDir, \
					figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)
			
				####################################################################
				# analyse training data
				####################################################################
				myTrainingDataAnalyser(dim, path2SignDir)
				
			################################################################################################################
					
			elif param1 == '3d':
				
				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				labelList = []
				
				dim = 3
				rankList = [6, 8]
				degrList = [20, 30]
				numDataPtsFactorList = [5]
				
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				####################################################################
				# run training
				####################################################################
	
				multiProc = True

				if multiProc:

					# numProcs = numProcs = mp.cpu_count()
					numProcs = 5
					N = np.int(len(paramList) / numProcs)

					prcsList = []
					# create processes
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N],\
							my3dExample, signature, path2SignDir, figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], \
						my3dExample, signature, path2SignDir, figuresDirName, modelDirName, perfDataDirName)))
									
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
			
				else:
					myTrainingRunnerFunc(dim, paramList, my3dExample, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)
				
			
				####################################################################
				# analyse training data
				####################################################################
				myTrainingDataAnalyser(dim, path2SignDir)
				
			################################################################################################################		
			
			elif param1 == '4d':

				####################################################################
				# fix rank, degree of approximative functions ...
				####################################################################
				
				# introduce a list which contains tuples of the form
				#	(rank, degr_1, ..., degr_d, L)
				# for different scenarios which then will be simulated ...
				paramList = []
				labelList = []
				# paramList.append((13, 27, 27, 1000))
				
				dim = 4
				rankList = [6, 8, 10, 12, 14, 20, 30]
				degrList = [20, 30, 40, 50, 60]
				numDataPtsFactorList = [5, 10, 20, 30]
				
				for k in range(0, len(numDataPtsFactorList)):
					for i in range(0, len(rankList)):
						for j in range(0, len(degrList)):
							paramList.append((rankList[i], degrList[j], degrList[j], degrList[j], degrList[j], rankList[i] * degrList[j] * dim * numDataPtsFactorList[k]))
				
				labelList = np.arange(0, len(paramList)).tolist()
				
				multiProc = True

				if multiProc:

					# numProcs = numProcs = mp.cpu_count()
					numProcs = 5
					N = np.int(len(paramList) / numProcs)

					prcsList = []

					# create processes ....
					for i in range(0, numProcs - 1):
						prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[i * N : (i + 1) * N], labelList[i * N : (i + 1) * N],\
							my4dExample, signature, path2SignDir, figuresDirName, modelDirName, perfDataDirName)))
					prcsList.append(mp.Process(target = myTrainingRunnerFunc, args = (dim, paramList[(numProcs - 1) * N : ], labelList[(numProcs - 1) * N : ], \
						my4dExample, signature, path2SignDir, figuresDirName, modelDirName, perfDataDirName)))						
			
					# run processes
					for prcs in prcsList:
						prcs.start()
						
					# join processes
					for prcs in prcsList:
						prcs.join()
			
				else:
					myTrainingRunnerFunc(dim, paramList, my4dExample, signature, path2SignDir, \
						figuresDirName = figuresDirName, modelDirName = modelDirName, perfDataDirName = perfDataDirName)				
			
				####################################################################
				# analyse training data
				####################################################################
				myTrainingDataAnalyser(dim, path2SignDir)
				
			################################################################################################################		
			
			else:
				print('### ERROR ### no valid command line parameter')
	elif mode == '--a':
		path2SignDir = resultsDirName + '/' + signature

		if param1 == '2d':
			myTrainingDataAnalyser(2, path2SignDir)
		elif param1 == '3d':
			myTrainingDataAnalyser(3, path2SignDir)
		elif param1 == '4d':
			myTrainingDataAnalyser(4, path2SignDir)
	else:
		pass
			
			
########################################################################################################################

if __name__ == '__main__':
	main()

	####################################################################################################################		

	# dim = 2
	# path2SignDir = '../../results/functionApproximation/2d_19'
	
	################################################################################################################		
	
	# analyse slected training data
		# myTrainingDataAnalyser(dim, path2SignDir)
	
	################################################################################################################		
	
	# create comparative surface plots, compute the experimental order of convergence, and analyse the cpu times

		# path2ModDir = path2SignDir + '/model'
		# path2FigDir = path2SignDir + '/figures'
		
		# createComparativePerfPlot(dim, path2SignDir, ['2d_4_0', '2d_4_1', '2d_4_3', '2d_4_76', '2d_4_119'])
		# createComparativeSurfPlot(dim, path2SignDir, ['2d_11_17', '2d_11_59'], ['modelData22', 'modelData16'])
		# createComparativeSurfPlot(dim, path2SignDir, ['2d_1_55', '2d_1_0'], ['modelData3', 'modelData3'])

		# computeEOC(path2ModDir, 'modelData', path2FigDir, storeFig = True)

		# analyseCPUTimes(dim, path2SignDir)
