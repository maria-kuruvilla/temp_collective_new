"""
Goal - to produce figures of convex hull area with loom sizes

Date - May 18th 2021 
"""

import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import numpy as np

plt.close('all') # always start by cleaning up
fig = plt.figure()#figsize=(16,10))
ax = fig.add_subplot(111)


x = np.array(range(500,1100))
y = 44 +2 * np.sqrt(2*1000/(np.pi*(5000-(50/6)*(x-500))))
ax.plot(x,y)

plt.show()