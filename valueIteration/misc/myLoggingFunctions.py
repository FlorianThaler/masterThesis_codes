"""
	Author:			Florian Thaler
	Email:			florian.thaler@edu.uni-graz.at
	Description:	Part of the code package corresponding to my master thesis. This file implements a 2d function
					optimal control problem example.
	Year:			2019/2020
"""

########################################################################################################################
# importing stuff
########################################################################################################################	

import datetime
import platform
import os

import logging

###########################################################
# definition of functions
###########################################################


def createSetupFile(sdeParamsDict, ctrlProbDict, valIterParamsDict, \
						alsParamsDict, cgParamsDict, cpdParamsDict, path2StpUpDataDir):
	"""
		@param[in] ### sdeParamsDict ### dictionary containing parameter values concerning the given sde
		@param[in] ### ctrlProbDict ### dictionary containing parameter values concerning the control problem
		@param[in] ### valIterParamsDict ### dictionary containing parameters regarding the value iteration
		@param[in] ### cpdParamsDict ### dictionary containing cpd parameters
		@param[in] ### alsParamsDict ### dictionary containing als parameters
		@param[in] ### cgParamsDict ### dictionary containing cg parameters
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
	stpFile.write('##### sde details #####\n')
	stpFile.write('\n')
	for key in sdeParamsDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(sdeParamsDict[key]) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('##### control details #####\n')
	stpFile.write('\n')
	for key in ctrlProbDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(ctrlProbDict[key]) + '\n')
	stpFile.write('\n')
	stpFile.write('######################################################################\n')
	stpFile.write('\n')
	stpFile.write('##### value iteration details #####\n')
	stpFile.write('\n')
	for key in valIterParamsDict.keys():
		stpFile.write(' + ' + key + ' = ' + str(valIterParamsDict[key]) + '\n')
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


def depositeIntermediateTrainingResults(modelFunc, path2ModDir, k):
	"""
		this function stores the parameter of a CPD approximation of the value function corresponding
		to a given optimal control problem to file.
		
		@param[in] ### modelFunc ### instance of class MyValueFunctionApproximation
		@param[in] ### path2ModDir ### string corresponding to the directory where model data should be stored
		@param[in] ### k ### integer corresponding to the current value iteration level
	"""
	logging.info('> write model parameters to file')
	os.makedirs(path2ModDir + '/iter_' + str(k))
	modelFunc.writeParams2File(path2ModDir + '/iter_' + str(k), 'modelData')
				

	
