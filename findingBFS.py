import csv
import math
import numpy as np
import scipy.signal
import sys


################
# IO Functions #
################

def loadData(fName):
	'''
	Loads the data. Assumes ordering is:
		DEPTH, GR, ..... idk

	:param fName: 	Whole path in the form of a string (can be relative).

	:returns GEPTH:	Depth of the well log
	:returns GR:	GR of the well log 
	'''
	DEPTH = []
	GR = []
	iFile = csv.reader(open(fName,'r'))
	for l in iFile:
		DEPTH.append(float(l[0]))
		GR.append(float(l[1]))
	return DEPTH, GR


def alignToZero(inDEPTH, inGR):
	'''
	Finds the max value of the GR (as of this version) and centres it at zero.
	Also parses out the ends of the data. A lot will be removed from the front, a little from the end.
	
	:*WARNING* : This works with the assumption that the max value is where we should zero. This is not accurate in general!!!!

	:param inDEPTH: 	Depth of the well log.
	:param inGR:		GR of the well log 

	:returns DEPTH: 	Depth of the well log.
	:returns GR:		GR of the well log 
	'''
	DEPTH = inDEPTH[[n for n,i in enumerate(inDEPTH) if i > (1120)][0]:]
	GR = inGR[[n for n,i in enumerate(inDEPTH) if i > (1120)][0]:]

	farBack = 40
	threshold = 50
	for i in range(farBack,len(DEPTH)):
		if np.mean(GR[i-farBack:i]) - GR[i] > threshold:
			BFSmarker = i
			break




	#maxValInd = np.argmax(GR[:])
	maxDepth = DEPTH[BFSmarker]
	DEPTH = [x - maxDepth for x in DEPTH]


	endInd = [n for n,i in enumerate(DEPTH) if i > (100)][0]
	DEPTH = DEPTH[:endInd]
	GR = GR[:endInd]

	return DEPTH, GR


def getStandardScores(inGR, inDEPTH):
	'''
	Converts the data into a Z-score/standard score version. Also truncates a few indicies past the max point. 
	
	:*WARNING* : This works with the assumption that the max value is where we should zero. This is not accurate in general!!!!

	:param inGR:		GR of the well log 
	:param inDEPTH:		Depth list corresponding to inGR

	:returns GR_Z:		Z-score/Standard score version of GR
	:returns offset:	Offset index of where the alignments will happen; Where the Z-scores/Standard Score start relative to inGR
	'''
	
	indOfDepth = [n for n,i in enumerate(inDEPTH) if i > (10)][0]
	indOfDepthEnd = [n for n,i in enumerate(inDEPTH) if i > (80)][0]
	maxValInd = np.argmax(inGR[indOfDepth:])
	GR_Z = scipy.stats.mstats.zscore(inGR[indOfDepth:indOfDepthEnd])
	#mean = np.mean(GR_Z)	
	#GR_Z = [x - mean for x in GR_Z]
	return GR_Z, indOfDepth
	


######################
# CODE DOWN HERE NOW #
######################

############
# IO stuff #
############

#1 data points: 6000 		(not align to 0)
#2 data points: 10000 		** GOOD **
#3 data points: 10000 		** GOOD **
#4 data points: 10000 		(not align to 0)
#5 data points: 10000 		(not align to 0)
#6 data points: 10000 		(not align to 0)

#load data in
dpth1, gr1 = loadData('./data/1.csv')
dpth2, gr2 = loadData('./data/2.csv')			#5, 6 are busted. I think something is wrong with the data
dpth3, gr3 = loadData('./data/3.csv')
dpth4, gr4 = loadData('./data/4.csv')
dpth5, gr5 = loadData('./data/5.csv')
dpth6, gr6 = loadData('./data/6.csv')

import matplotlib.pylab as plt


#align data
dpth1, gr1 = alignToZero(dpth1, gr1)
dpth2, gr2 = alignToZero(dpth2, gr2)
dpth3, gr3 = alignToZero(dpth3, gr3)
dpth4, gr4 = alignToZero(dpth4, gr4)
dpth5, gr5 = alignToZero(dpth5, gr5)
dpth6, gr6 = alignToZero(dpth6, gr6)

#plt.plot(dpth1,gr1)
#plt.plot(dpth2,gr2)
#plt.plot(dpth3,gr3)
plt.plot(dpth4,gr4)
plt.plot(dpth5,gr5)
plt.plot(dpth6,gr6)
plt.autoscale()
plt.show()


'''
#normalize the data
gr1_z, ind1 = getStandardScores(gr1, dpth1)
gr2_z, ind2 = getStandardScores(gr2, dpth2)
gr3_z, ind3 = getStandardScores(gr3, dpth3)
gr4_z, ind4 = getStandardScores(gr4, dpth4)
gr5_z, ind5 = getStandardScores(gr5, dpth5)
gr6_z, ind6 = getStandardScores(gr6, dpth6)

'''

