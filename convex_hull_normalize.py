"""
Goal - to normalize convex hull area for number of individuals 
"""

import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D

in_dir1 = '../../output/temp_collective/roi/convex_hull_area.p'

values = pickle.load(open(in_dir1, 'rb')) # 'rb is for read binary
colors = plt.cm.viridis(np.linspace(0,1,6))
a = np.nanmean(values[0:1000, :, :, :], axis = 0)
b = np.nanmean(a, axis = 2)
b_std = np.nanstd(a, axis = 2)
c= np.log2(b)
plt.close('all') # always start by cleaning up
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
fs = 12
temp = [9,13,17,21,25,29]
z=np.empty([6,2])
for i in range(6):
	ax.plot([4,8,16,32], b[i,:], label = temp[i], color = colors[i])
	ax.fill_between([4,8,16,32], b[i,:] - b_std[i,:], b[i,:] + b_std[i,:], alpha = 0.2, color = colors[i])
plt.xlabel('group size', size = fs)
plt.ylabel('convex hull area', size = fs)
plt.xscale('log',basex=2) 
plt.yscale('log',basey=2) 

ax.legend()

plt.show()


