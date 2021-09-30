"""
Goal - to produce poster of convex hull area 
Date - Sep 17th 2021 
"""

import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import numpy as np

in_dir1 = '../../output/temp_collective/roi/convex_hull_area.p'

values = pickle.load(open(in_dir1, 'rb')) # 'rb is for read binary

groups = 3
replicates = 10
looms = 5
lw=np.linspace(0.75,1,6)
fs=16
ls = [ '-' , '--' , '-.' , ':' ]
frames = list(range(500,1100,25))
frames = frames + [1099]
colors = plt.cm.viridis(np.linspace(0,1,6))
alpha = np.linspace(0.2,1,6)
plt.close('all') # always start by cleaning up
fig = plt.figure()#figsize=(16,10))
ax = fig.add_subplot(111)

for i in range(0,looms):
    for j in range(0,replicates):
        for k in range(0,groups):
            for l in range(6):
                ax.plot(range(500,1500), values[(i*2000 + 500):(i*2000+1500),l,k,j],linewidth = 0.75, color = colors[l], alpha = 0.2)


plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()



out_dir = '../../output/temp_collective/roi_figures/poster.png'
fig.savefig(out_dir, dpi = 1200)
