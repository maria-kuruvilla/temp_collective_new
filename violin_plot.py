
"""

Goal - to test the violin plot
Date - Sep 7th 2021

"""


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import cm
import argparse
import seaborn as sns 

lw=2
fs=16
fig = plt.figure()
ax = fig.add_subplot(111)
dpi = 100
gs=[16]
colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))

data1 = pd.read_csv('../../data/temp_collective/roi/loom_speed_99_predictions_one_model.csv')

data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')

temp = [9,13,17,21,25,29]

for t in temp:
    parts = ax.violinplot(data2.speed_percentile99[data2.Groupsize == 16][data2.Loom == 1][data2.Temperature == t], [t])
    parts['bodies'][0].set_facecolor('gray')
    parts['bodies'][0].set_edgecolor('black')
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
count = 1
ax.plot(
    data1.temp[data1.gs ==16][data1.loom == 1], 
    (data1.speed99[data1.gs ==16][data1.loom == 1])**2, 
    color = colors[count], lw = lw)

ax.fill_between(
    data1.temp[data1.gs ==16][data1.loom == 1], 
    (data1.speed99_025[data1.gs ==16][data1.loom == 1])**2,  
    (data1.speed99_975[data1.gs ==16][data1.loom == 1])**2, alpha = 0.3, 
    color = colors[count], lw = 0,label = str(16))
count +=1
        
plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
plt.yticks(fontsize = fs)
plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
plt.ylabel('Near maximum speed (BL/s)', size = fs)


out_dir = '../../output/temp_collective/roi_figures/loom_speed_99_predictions_w_data_one_model.png'
fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")


plt.show()