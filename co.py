"""
Goal - to produce figures of convex hull area
"""

import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import numpy as np

in_dir1 = '../../output/temp_collective/roi/convex_hull_area.p'

values = pickle.load(open(in_dir1, 'rb')) # 'rb is for read binary

groups = 4
replicates = 10
looms = 5
lw=np.linspace(0.75,1,6)
fs=12
ls = [ '-' , '--' , '-.' , ':' ]
colors = plt.cm.viridis(np.linspace(0,1,6))
alpha = np.linspace(0.2,1,6)
plt.close('all') # always start by cleaning up
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
for i in range(0,looms):
    for j in range(0,replicates):
        for k in range(0,groups):
            for l in range(6):
                ax.plot(range(0,2000), values[(i*2000):(i*2000+2000),l,k,j]/(2**(k+2)),linewidth = 0.75, color = colors[l], alpha = 0.2)
#plt.axvline(500, color = 'k',alpha = 0.5)
#plt.axvline(1097, color = 'k', alpha = 0.5)      
plt.xlabel('Frame', size = fs)
plt.ylabel('Convex hull area', size = fs)
#plt.legend(fontsize=fs, loc='upper right', framealpha = 0.5)
ax.annotate('loom end', xy=(1097, 0.5), xytext=(1097, 2),arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='bottom')
ax.annotate('loom start', xy=(500, 0.5), xytext=(500, 2),arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='bottom')
plt.show()

custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4),
                Line2D([0], [0], color=colors[2], lw=4),
                Line2D([0], [0], color=colors[3], lw=4),
                Line2D([0], [0], color=colors[4], lw=4),
                Line2D([0], [0], color=colors[5], lw=4)]


ax.legend(custom_lines, ['9' + r'$^{\circ}$C', '13' + r'$^{\circ}$C', '17' + r'$^{\circ}$C', '21' + r'$^{\circ}$C', '25' + r'$^{\circ}$C', '29' + r'$^{\circ}$C'], loc='upper right')
"""
custom_lines1 = [Line2D([0], [0], ls = ls[0], lw=4),
                Line2D([0], [0], ls = ls[1], lw=4),
                Line2D([0], [0], ls = ls[2], lw=4),
                Line2D([0], [0], ls = ls[3], lw=4)]


ax.legend(custom_lines1, ['4', '8', '16', '32'], loc='lower left')
"""
out_dir = '../../output/temp_collective/roi_figures/convex_hull_area_rescaled.png'
fig.savefig(out_dir, dpi = 300)
