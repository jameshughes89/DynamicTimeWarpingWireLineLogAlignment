'''
This script will do dynamic time warping (but requires alterations for applications)

distance function needs to be set for the needs of the applciation 
	ideas for strato stuff
	- just euclidean distance?
	- squared distance?
	- pearson correlation
	- spearman correlation (would have to do a window thing)

	ideas for representing
	1. plot both on same plot (but have one be translated up by some value
	2. figure out where matches happen (like the 'time points' where there is a match)
	3. draw dotted lines between the points. 


	probably should preprocess matea's data
	- z-score probably. 
'''

import math
import matplotlib.pylab as plt
import matplotlib.collections as clt
import numpy as np
import sys


def distance(x, y):
	'''
	Calculates some distance between two data (this can be a single point or even a list of points (depending on metric)). This method should be changed to fit the needs of the applications

	:param x: Some data 
	:param y: Some other data

	:returns: Calculated distance between the two data.
	'''
	return abs(x - y)


def dtw(s, t, d):
	'''
	Performs basic dynamic time warping (no window on this one).

	WARNING! Need to have array 1 bigger because

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
	while i != 0 or j != 0:
		minV = min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
		if minV == DTW[i - 1, j]:
			i = i - 1
		elif minV == DTW[i, j - 1]:
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
	i= len(DTW) - 1
	j = len(DTW[0]) - 1
	
	#backtracking part
	while i != 0 or j != 0:
		minV = min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
		if minV == DTW[i - 1, j - 1]:		#overlap
			points.append([i, j])
			i = i - 1
			j = j - 1
		elif minV == DTW[i - 1, j]:			#insertion
			i = i - 1
		else:								#must be a deletion
			j = j - 1

	
	return points


######################
# CODE DOWN HERE NOW #
######################



d1 = [0,0,0,1,1,2,1,0,1,1,2,1,2,3,2,1,1,2,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,2,3,3,3,3,3,2,1,0]
d2 = [1,1,2,1,0,0,0,0,0,0,0,0,1,1,1,1,2,2,1,1,2,2,3,3,2,1,1,1,1,2,2,1,1,0,1,1,1,0,1,2,3,2,1,0]

#d1 = [math.sin(x)+np.random.normal(0,0.25) for x in range(15)]
#d2 = [math.sin(x)+np.random.normal(0,0.25) for x in range(5,25)]

val, DTW = dtw(d1, d2, distance)
points = backtrack(DTW)

#plt.plot(d1)
#plt.plot([x + 10 for x in d2])					#I added 10 because
#plt.ylim([-10,20])

lines = []
for i in range(len(points)):
	#had to -1 because of DTW array starting at 0, but  important starting at 1	
	lines.append([(points[i][0]-1,d1[points[i][0]-1]),(points[i][1]-1,d2[points[i][1]-1] + 10)]) 	#I added 10 because

lc = clt.LineCollection(lines, colors='r', linestyles='dotted')
fig, ax = plt.subplots()
plt.plot(d1)
plt.plot([x + 10 for x in d2])					#I added 10 because
ax.add_collection(lc)
plt.ylim([-10,20])
plt.xlim([-1,max(len(d1), len(d2))])
plt.show()

#np.savetxt('DTWmat.csv', DTW, delimiter=',')












