
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

lw=3
fs=16
fig = plt.figure()
ax = fig.add_subplot(111)
dpi = 100
gs=[16]
colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))

data1 = pd.read_csv('../../data/temp_collective/roi/loom_speed_99_predictions_one_model.csv')

data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')

temp = [9,13,17,21,25,29]
count = 1
for t in temp:
    parts = ax.violinplot(data2.speed_percentile99[data2.Groupsize == 16][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
    parts['bodies'][0].set_facecolor(colors[count])
    parts['bodies'][0].set_edgecolor(colors[count])
    parts['bodies'][0].set_alpha(0.2)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(colors[count])
        vp.set_linewidth(1)
        vp.set_alpha(0.8)
    #ax = sns.violinplot(data2.Temperature, data2.speed_percentile99[data2.Groupsize == 16][data2.Loom == 1], width = 0.5, color = colors[count], inner = None, cut = 0)
    #plt.setp(ax.collections, alpha=0.2)   
    
ax.plot(
    data1.temp[data1.gs ==16][data1.loom == 1], 
    (data1.speed99[data1.gs ==16][data1.loom == 1])**2, 
    color = colors[count], lw = lw)

ax.fill_between(
    data1.temp[data1.gs ==16][data1.loom == 1], 
    (data1.speed99_025[data1.gs ==16][data1.loom == 1])**2,  
    (data1.speed99_975[data1.gs ==16][data1.loom == 1])**2, alpha = 0.2, 
    color = colors[count], lw = 0,label = str(16))
count +=1
        
plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
plt.yticks(ticks = [8,12,16,20], labels = [8,12,16,20],fontsize = fs)
plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
plt.ylabel('Maximum speed (BL/s)', size = fs)


#out_dir = '../../output/temp_collective/roi_figures/loom_speed_99_predictions_w_data_one_model.png'
#fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")


plt.show()
