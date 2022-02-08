def myImageExample():
	
	random.seed(0)
	np.random.seed(0)
	
	#################################################################
	# ### provide data
	#################################################################
	
	img = mpimg.imread('orange.png')
	
	# now transform image to greyscale image
	
	r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	
	# rgb 2 greyscale transformation following the mathworks/matlab approach!
	grScImg = 0.2989 * r + 0.5870 * g + 0.1140 * b
		
	nRows = grScImg.shape[0]
	nCols = grScImg.shape[1]
	
	L = 500
	
	a = 1
	b = (nRows / nCols) * a 
	
	dx1 = a / nCols
	dx2 = b / nRows
	
	boundingBox = []
	boundingBox.append((0, a))
	boundingBox.append((0, b))

	
	x1Inds = np.random.randint(0, nCols, L)
	x2Inds = np.random.randint(0, nRows, L)

	xData = np.zeros((2, L))
	yData = np.zeros(L)
	

	yData = grScImg[x2Inds, x1Inds]
	
	zz = grScImg[x2Inds, x1Inds.reshape(-1, 1)]
	
	x1Data = x1Inds * dx1
	x2Data = x2Inds * dx2
	
	xData[0, :] = x1Data
	xData[1, :] = x2Data
	
	#################################################################
	# ### start function approximation procedure
	#################################################################
	
	dim = 2
	rank = 5
	
	degrs = []
	degrs.append(20)
	degrs.append(20)
	
	modelFunc = MyCPDRadialAnsatzFunc(dim, rank, degrs)
		
	t0 = time.time()
	costFuncValList, l2ApprErrList, cgPerformanceList = rbfFunctionApproximation(L, boundingBox, xData, yData, modelFunc, verbose = False)
	t1 = time.time()
		
	print('#####################################################')
	print('#')
	print('# total approximation time = ' + str(t1 - t0) + ' s')
	print('#')
	print('#####################################################')
	
	#################################################################
	# ### visualise
	#################################################################
	
	# ### plot approximative function

	x1 = np.linspace(boundingBox[0][0], boundingBox[0][1], nCols)
	x2 = np.linspace(boundingBox[1][0], boundingBox[1][1], nRows)
	
	[xx1, xx2] = np.meshgrid(x1, x2)
	
	fig1 = plt.figure(1)

	zz = np.zeros((nRows, nCols))
	
	argsList = []
	argsList.append(xx1)
	argsList.append(xx2)
	zz = modelFunc.evaluate(argsList)
	
	plt.imshow(zz, cmap = 'gray', vmin = 0, vmax = 1)
	
	# ### plot diagnostic plots
	
	fig2 = plt.figure(2)
	plt.plot(np.arange(0, len(costFuncValList)), costFuncValList)
	plt.title('evolution of value of cost functional during optimisation process')
	plt.xlabel('ALS iterations')
	# plt.ylabel('')
	
	fig3 = plt.figure(3)
	plt.plot(np.arange(0, len(l2ApprErrList)), l2ApprErrList)
	plt.title('evolution of the l2 approximation error')
	plt.xlabel('ALS iterations')
	
	fig4 = plt.figure(4)
	plt.title('cg performance w.r.t. to each coordinate optimisation process')
	
	for d in range(0, dim):	
		tmpList1 = [elem[1] for elem in cgPerformanceList[d]]
		tmpList2 = [elem[0] for elem in cgPerformanceList[d]]
		
		plt.subplot(dim, 2, d * dim + 1)
		plt.plot(np.arange(0, len(tmpList1)), tmpList1)
		plt.title('norm of gradient')
		plt.xlabel('ALS iterations')
		
		plt.subplot(dim, 2, d * dim + 2)
		plt.plot(np.arange(0, len(tmpList2)), tmpList2)
		plt.title('number of cg iterations')
		plt.xlabel('ALS iterations')
		
	plt.show()
