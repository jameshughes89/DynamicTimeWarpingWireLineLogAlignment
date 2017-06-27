import numpy as np
import csv
import matplotlib.pylab as plt
import scipy
import scipy.signal

d1 = []
r1 = []
dpth1 = []
d2 = []
r2 = []
dpth2 = []
d3 = []
r3 = []
dpth3 = []
d4 = []
r4 = []
dpth4 = []


iFile = csv.reader(open('./data/2.csv','r'))

for l in iFile:
	dpth1.append(float(l[0]))
	d1.append(float(l[1]))
	r1.append(float(l[2]))

iFile = csv.reader(open('./data/3.csv','r'))

for l in iFile:
	dpth2.append(float(l[0]))
	d2.append(float(l[1]))
	r2.append(float(l[2]))
	
	
d1 = d1[700*10:-100*10]
r1 = r1[700*10:-100*10]
dpth1 = dpth1[700*10:-100*10]

d2 = d2[700*10:-100*10]
r2 = r2[700*10:-100*10]
dpth2 = dpth2[700*10:-100*10]

d1_z = scipy.stats.mstats.zscore(d1)
d2_z = scipy.stats.mstats.zscore(d2)




maxD1 = dpth1[np.argmax(d1)]
maxD2 = dpth2[np.argmax(d2)]

dpth1 = [x - maxD1 for x in dpth1]
dpth2 = [x - maxD2 for x in dpth2]

plt.plot(dpth1, d1)
plt.plot(dpth2, d2)
plt.plot(dpth1, r1)
plt.plot(dpth2, r2)
plt.show()