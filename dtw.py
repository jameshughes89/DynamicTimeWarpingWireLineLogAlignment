'''
This script will do dynamic time warping (but requires alterations for applications)

TODO
- get depth
- line up GR and RESIST (maybe do something with like, in 1, but NOT in the other)
- Figure out how to do more than 2 at a time (some multi-dimentional dynamic programming).
- Maybe find 6 big ones and then small one between easy of the 6 (1 at a time)?

'''


import csv
import math
import numpy as np
import scipy.signal
import sys

EPSILON = 0.0000001

def distance(x, y):
	'''
	Calculates some distance between two data (this can be a single point or even a list of points (depending on metric)). This method should be changed to fit the needs of the applications

	:param x: Some data 
	:param y: Some other data

	:returns: Calculated distance between the two data.
	'''
	return abs(x - y)
	#return (x-y)**2

def dtw(s, t, d):
	'''
	Performs basic dynamic time warping (no window on this one).

	WARNING! Need to have array 1 bigger because reasons

	:param s: Some data series 
	:param t: Some other data series
	:param d: Some distance measure

	:returns: best value in the bottom corner
	:returns: matrix after dtw was done (need this to actual do alignment).
	'''
	#setup DTW matrix
	DTW = np.zeros((len(s) + 1, len(t) + 1))	#+ 1 because of 0 needing the extra row and col
	DTW[1:,0] = sys.float_info.max
	DTW[0,1:] = sys.float_info.max

	#Calculate full DTW matrix
	for i in range(1, len(s) + 1):
		for j in range(1, len(t) + 1):
			sPoint = s[i - 1]			#this will change denending on how d works
			tPoint = t[j - 1]			#this will change denending on how d works
			cost = d(sPoint, tPoint)
			DTW[i,j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])	#insertion, deletion, match (in that order)

	return DTW[-1,-1], DTW



def dtw_window(s, t, w, d):
	'''
	Performs window dynamic time warping (window on this one). By using a window we can enforce some locality constraint.

	WARNING! Need to have array 1 bigger because reasons


	:param s: Some data series 
	:param t: Some other data series
	:param w: Window size
	:param d: Some distance measure

	:returns: best value in the bottom corner
	:returns: matrix after dtw was done (need this to actual do alignment).
	'''
	#setup DTW matrix
	DTW = np.zeros((len(s) + 1, len(t) + 1))	#+ 1 because of 0 needing the extra row and col
	DTW[:,:] = sys.float_info.max
	DTW[0,0] = 0

	w = max(w, abs(len(s) - len(t)))	#Makes sure the window isnt't bigger than the difference between lengths of data

	#Calculate full DTW matrix
	for i in range(1, len(s) + 1):
		for j in range(max(1, i-w), min(len(t) + 1, i + w)):
			sPoint = s[i - 1]			#this will change denending on how d works
			tPoint = t[j - 1]			#this will change denending on how d works
			cost = d(sPoint, tPoint)
			DTW[i,j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])	#insertion, deletion, match (in that order)

	return DTW[-1,-1], DTW


def backtrack(DTW):
	'''
	Backtracks and returns the points where there were matches.

	:param DTW: The dynamic time warping matrix

	:returns: A list of 'time' points where there were matches found in the matrix. Points will be ordered in the same way the matrix was generated.
	'''

	#setup
	points = []
	i= len(DTW) - 1
	j = len(DTW[0]) - 1
	#backtracking part
	while i != 0 and j != 0:
		minV = min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
		if abs(minV - DTW[i - 1, j]) < EPSILON:
			i = i - 1
		elif abs(minV - DTW[i, j - 1]) < EPSILON :
			j = j - 1
		else:						#must be an overlap then
			points.append([i, j])
			i = i - 1
			j = j - 1
	
	return points

def backtrack_overlapFirst(DTW):
	'''
	Backtracks but gives backtrack the first check (basically same as above but with different 'if' statement orderings. Returns the points where there were matches.

	:param DTW: The dynamic time warping matrix

	:returns: A list of 'time' points where there were matches found in the matrix. Points will be ordered in the same way the matrix was generated.
	'''

	#setup
	points = []
	i = len(DTW) - 1
	j = len(DTW[0]) - 1

	#backtracking part
	while i != 0 and j != 0:
		minV = min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
		if abs(minV - DTW[i - 1, j - 1]) < EPSILON:		#overlap
			points.append([i, j])
			i = i - 1
			j = j - 1
		elif abs(minV - DTW[i - 1, j]) < EPSILON:			#insertion
			i = i - 1
		else:								#must be a deletion
			j = j - 1

	
	return points


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
	DEPTH = inDEPTH[:]
	GR = inGR[:]

	whereToEnd = len(DEPTH)#[n for n,i in enumerate(DEPTH) if i > (1160)][0]
	indOfDepth = [n for n,i in enumerate(DEPTH[:whereToEnd]) if i > (1150)][0]	#find the index of the detph below 1150 (why 1150? because it's near-ish base fish scales!)

	maxValInd = np.argmax(GR[indOfDepth:])
	maxDepth = DEPTH[indOfDepth + maxValInd]
	DEPTH = [x - maxDepth for x in DEPTH]

	begInd = [n for n,i in enumerate(DEPTH) if i > (-10)][0]
	endInd = [n for n,i in enumerate(DEPTH) if i > (100)][0]
	DEPTH = DEPTH[begInd:endInd]
	GR = GR[begInd:endInd]

	return DEPTH, GR


def alignToZeroHAX(inDEPTH, inGR):
	'''
	Finds the max value of the GR (as of this version) and centres it at zero.
	Also parses out the ends of the data. A lot will be removed from the front, a little from the end.
	
	:*WARNING* : This works by seeing if the previous X (40) data points is greater than some threshold (60)

	:param inDEPTH: 	Depth of the well log.
	:param inGR:		GR of the well log 

	:returns DEPTH: 	Depth of the well log.
	:returns GR:		GR of the well log 
	'''
	DEPTH = inDEPTH[[n for n,i in enumerate(inDEPTH) if i > (1100)][0]:]
	GR = inGR[[n for n,i in enumerate(inDEPTH) if i > (1100)][0]:]

	farBack = 40
	threshold = 60
	for i in range(farBack,len(DEPTH)):
		if np.mean(GR[i-farBack:i]) - GR[i] > threshold:
			BFSmarker = i
			break




	#maxValInd = np.argmax(GR[:])
	maxDepth = DEPTH[BFSmarker]
	DEPTH = [x - maxDepth for x in DEPTH]


	begInd = [n for n,i in enumerate(DEPTH) if i > (-10)][0]
	endInd = [n for n,i in enumerate(DEPTH) if i > (100)][0]
	DEPTH = DEPTH[begInd:endInd]
	GR = GR[begInd:endInd]

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
	indOfDepthEnd = [n for n,i in enumerate(inDEPTH) if i > (75)][0]
	maxValInd = np.argmax(inGR[indOfDepth:])
	GR_Z = scipy.stats.mstats.zscore(inGR[indOfDepth:indOfDepthEnd])
	#mean = np.mean(GR_Z)	
	#GR_Z = [x - mean for x in GR_Z]
	return GR_Z, indOfDepth
	

##################
# Plot Functions #
##################

def makeLines(gr1, dpth1, ind1, gr2, dpth2, ind2, points, lineDraw=1, gap=200):
	'''
	Generates the lines that align the "timeseries's". Returned thing can be thrown at the plot. 
	
	:param gr1:			"Timeseries 1" (will be below)
	:param dpth1:		Depths for "timeseries" 1
	:param ind1:		offset index for "Timeseries" 1 of where to start aligninng (because points are based on Z-score/standard score
	:param gr2:			"Timeseries 2" (will be below)
	:param dpth2:		Depths for "timeseries" 2
	:param ind2:		offset index for "Timeseries" 2 of where to start aligninng (because points are based on Z-score/standard score
	:param points:		The points that are aligned based on DTW
	:param lineDraw:	Create every X lines. Default to 1, so it creates every line. If it was 2, it would print every other line 
	:param gap:			Gap between the two "Timeseries" to be plotted. Default to 200, because... why not?

	:returns lc:		Line Collection of the lines.
	'''
	import matplotlib.collections as clt
	lines = []
	for i in range(0,len(points), lineDraw):
		if i < 1 or abs(gr1[points[i][0] - 1 + ind1] - gr2[points[i][1] - 2 + ind2]) > 10:			#ONLY DRAW THE DIFFERENCE IF IT's GREATER THAN THIS VALUE
		#had to -1 because of DTW array starting at 0, but  important starting at 1	
			lines.append([(dpth1[points[i][0]-1 + ind1], gr1[points[i][0]-1 + ind1]),(dpth2[points[i][1]-1 + ind2], gr2[points[i][1]-1 + ind2] + gap)])

	lc = clt.LineCollection(lines, colors='r', linestyles='dotted')
	return lc

def makeLinesLists(grs, dpths, inds, points, lineDraw=1, gap=200):
	'''
	FIXME
	FIXME
	Generates the lines that align the "timeseries's" for a collection of graphs (this does not seem to work well...)
	
	:param grs:			Lists of "Timeseries"
	:param dpths:		Lists of Depths
	:param inds:		List of offset index for "Timeseries"s: where to start aligninng (because points are based on Z-score/standard score
		offset index for "Timeseries" 2 of where to start aligninng (because points are based on Z-score/standard score
	:param points:		List of the points that are aligned based on DTW (0 is for [0,1], 1 is for [1,2], etc.)
	:param lineDraw:	Create every X lines. Default to 1, so it creates every line. If it was 2, it would print every other line 
	:param gap:			Gap between the two "Timeseries" to be plotted. Default to 200, because... why not?

	:returns lc:		List of line Collection of the lines for each set of points (pair of "timeseries").
	'''
	import matplotlib.collections as clt
	lc = []
	gapOrig = gap
	
	for k in range(len(grs) - 1):
		lines = []
		for i in range(0,len(points[k]), lineDraw):
			if i < 1 or abs(grs[k][points[k][i][0] - 1 + inds[k]] - grs[k+1][points[k][i][1] - 2 + inds[k+1]]) > (80 * (dpths[k][i+1] - dpths[k][i])):			#ONLY DRAW THE DIFFERENCE IF IT's GREATER THAN THIS VALUE
			#had to -1 because of DTW array starting at 0, but  important starting at 1	
				lines.append([(dpths[k][points[k][i][0]-1 + inds[k]], grs[k][points[k][i][0]-1 + inds[k]]),(dpths[k+1][points[k][i][1]-1 + inds[k+1]], grs[k+1][points[k][i][1]-1 + inds[k+1]] + gap)])

		lc.append(clt.LineCollection(lines, colors='r', linestyles='dotted'))
		gap = gap + gapOrig

	return lc

def plotAlign(gr1, dpth1, gr2, dpth2, lines, gap=200):
	'''
	Plots the "Timeseries's" and draws the lines that align them. 
	
	:param gr1:			"Timeseries 1" (will be below)
	:param dpth1:		Depths for "timeseries" 1
	:param gr2:			"Timeseries 2" (will be below)
	:param dpth2:		Depths for "timeseries" 2
	:param lines:		Create every X lines. Default to 1, so it creates every line. If it was 2, it would print every other line 
	:param gap:			Gap between the two "Timeseries" to be plotted. Default to 200, because...

	'''
	import matplotlib.pylab as plt
	fig, ax = plt.subplots()
	plt.plot(dpth1, gr1)
	plt.plot(dpth2, [x + gap for x in gr2])
	ax.add_collection(lines)
	plt.autoscale()
	plt.show()


def plotAlignList(grs, dpths, lines, gap=200):
	'''
	Plots the "Timeseries's" and draws the lines that align them. 
	
	:param gr1:			"Timeseries 1" (will be below)
	:param dpth1:		Depths for "timeseries" 1
	:param gr2:			"Timeseries 2" (will be below)
	:param dpth2:		Depths for "timeseries" 2
	:param lines:		Create every X lines. Default to 1, so it creates every line. If it was 2, it would print every other line 
	:param gap:			Gap between the two "Timeseries" to be plotted. Default to 200, because...

	'''
	import matplotlib.pylab as plt
	fig, ax = plt.subplots()
	plt.plot(dpths[0], grs[0])
	
	for i in range(len(grs) - 1):
		plt.plot(dpths[i+1], [x + gap for x in grs[i+1]])
		ax.add_collection(lines[i])

	plt.autoscale()
	plt.show()




def makeLinesPlotLists(grs, dpths, inds, points, lineDraw=1, gap=200):
	'''
	FIXME
	FIXME
	Generates the lines that align the "timeseries's" for a collection of graphs (this does not seem to work well...)
	
	:param grs:			Lists of "Timeseries"
	:param dpths:		Lists of Depths
	:param inds:		List of offset index for "Timeseries"s: where to start aligninng (because points are based on Z-score/standard score
		offset index for "Timeseries" 2 of where to start aligninng (because points are based on Z-score/standard score
	:param points:		List of the points that are aligned based on DTW (0 is for [0,1], 1 is for [1,2], etc.)
	:param lineDraw:	Create every X lines. Default to 1, so it creates every line. If it was 2, it would print every other line 
	:param gap:			Gap between the two "Timeseries" to be plotted. Default to 200, because... why not?

	:returns lc:		List of line Collection of the lines for each set of points (pair of "timeseries").
	'''
	import matplotlib.pylab as plt
	import matplotlib.collections as clt
	
	gapPrev = 0	
	gapOrig = gap
	
	fig, ax = plt.subplots()
	plt.plot(dpths[0], grs[0], color='k')


	for k in range(len(grs) - 1):
		lines = []
		for i in range(0,len(points[k]), lineDraw):
			if i < 1 or abs(grs[k][points[k][i][0] - 1 + inds[k]] - np.mean(grs[k][points[k][i][1] - 5 + inds[k]:points[k][i][1] - 2 + inds[k]])) > (120 * (dpths[k][i+1] - dpths[k][i])):			#ONLY DRAW THE DIFFERENCE IF IT's GREATER THAN THIS VALUE
			#had to -1 because of DTW array starting at 0, but  important starting at 1	
				lines.append([(dpths[k][points[k][i][0]-1 + inds[k]], grs[k][points[k][i][0]-1 + inds[k]] + gapPrev),(dpths[k+1][points[k][i][1]-1 + inds[k+1]], grs[k+1][points[k][i][1]-1 + inds[k+1]] + gap)])

		lc = clt.LineCollection(lines, colors='r', linestyles='dotted')

		plt.plot(dpths[k+1], [x + gap for x in grs[k+1]], color='k')
		ax.add_collection(lc)

		gapPrev = gap
		gap = gap + gapOrig


	plt.autoscale()
	plt.show()

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






#align data
dpth1, gr1 = alignToZeroHAX(dpth1, gr1)
dpth2, gr2 = alignToZeroHAX(dpth2, gr2)
dpth3, gr3 = alignToZeroHAX(dpth3, gr3)
dpth4, gr4 = alignToZeroHAX(dpth4, gr4)
dpth5, gr5 = alignToZeroHAX(dpth5, gr5)
dpth6, gr6 = alignToZeroHAX(dpth6, gr6)

'''
import matplotlib.pylab as plt
plt.plot(dpth1,gr1)
plt.plot(dpth2,gr2)
plt.plot(dpth3,gr3)
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


##########
# Now Go #
##########

val, DTW = dtw(gr1_z, gr2_z, distance)
#val, DTW = dtw_window(gr1_z, gr2_z, 10, distance)
points1_2 = backtrack(DTW)

val, DTW = dtw(gr2_z, gr3_z, distance)
#val, DTW = dtw_window(gr1_z, gr2_z, 10, distance)
points2_3 = backtrack(DTW)

val, DTW = dtw(gr3_z, gr4_z, distance)
#val, DTW = dtw_window(gr1_z, gr2_z, 10, distance)
points3_4 = backtrack(DTW)

val, DTW = dtw(gr4_z, gr5_z, distance)
#val, DTW = dtw_window(gr1_z, gr2_z, 10, distance)
points4_5 = backtrack(DTW)

val, DTW = dtw(gr5_z, gr6_z, distance)
#val, DTW = dtw_window(gr1_z, gr2_z, 10, distance)
points5_6 = backtrack(DTW)

gap = 100

#normal
#lc = makeLines(gr1, dpth1, ind1, gr2, dpth2, ind2, points, lineDraw=1, gap=gap)
#plotAlign(gr1, dpth1, gr2, dpth2, lc, gap=gap)

#busted
#lc = makeLinesLists([gr1, gr2], [dpth1,dpth2], [ind1,ind2], [points], lineDraw=1, gap=gap)
#plotAlignList([gr1,gr2], [dpth1,dpth2], [lc], gap=gap)


#makeLinesPlotLists([gr1, gr2, gr3, gr4], [dpth1, dpth2, dpth3, dpth4], [ind1,ind2, ind3, ind4], [points1_2, points2_3, points3_4], lineDraw=1, gap=gap)


#with 5 & 6
makeLinesPlotLists([gr1, gr2, gr3, gr4, gr5, gr6], [dpth1, dpth2, dpth3, dpth4, dpth5, dpth6], [ind1,ind2, ind3, ind4, ind5, ind6], [points1_2, points2_3, points3_4, points4_5, points5_6], lineDraw=1, gap=gap)


