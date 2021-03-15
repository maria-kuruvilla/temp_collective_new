"""
Goal - make figures of the predictions with and without data
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

data1 = pd.read_csv('../../data/temp_collective/roi/loom_speed_predictions.csv')

data2 = pd.read_csv('../../data/temp_collective/roi/stats_loom_low_pass_data.csv')


colors = plt.cm.viridis(np.linspace(0,1,5))
#Plotting
lw=1.25
fs=12
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
"""
ax.scatter(data2.Temperature[data2.Groupsize == 16][data2.loom == 1],
	data2["99_speed"][data2.Groupsize == 16][data2.loom == 1], alpha = 0.5, color = colors[0])

ax.plot(
	data1.Temperature[data1.Groupsize == 16][data1.loom == 1], 
	(data1.loom_speed[data1.Groupsize==16][data1.loom == 1])**2, color = colors[0])
ax.fill_between(
	data1.Temperature[data1.Groupsize == 16][data1.loom == 1], 
	(data1.loom_speed025[data1.Groupsize==16][data1.loom == 1])**2,  
	(data1.loom_speed975[data1.Groupsize==16][data1.loom == 1])**2, alpha = 0.3, color = colors[0])
plt.xlabel('Temperature', size = fs)
plt.ylabel('99th percentile of speed during loom', size = fs)
ax.set_title('Groupsize = 16, Loom = 1', fontsize = fs)
out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions_wo_data.png'
fig.savefig(out_dir, dpi = 300)
plt.show()

"""
####################################
gs = [1,2,4,8,16]
loom = [1,2,3,4,5]
count = 0
i = 16
for j in loom:
	"""
	ax.scatter(data2.Temperature[data2.Groupsize == i][data2.loom == j],
	data2["99_speed"][data2.Groupsize == i][data2.loom == j], alpha = 0.5, color = colors[count])
	"""
	ax.plot(
		data1.Temperature[data1.Groupsize == i][data1.loom == j], 
		(data1.loom_speed[data1.Groupsize==i][data1.loom == j])**2, color = colors[count], 
		label = str(j))
	ax.fill_between(
		data1.Temperature[data1.Groupsize == i][data1.loom == j], 
		(data1.loom_speed025[data1.Groupsize==i][data1.loom == j])**2,  
		(data1.loom_speed975[data1.Groupsize==i][data1.loom == j])**2, alpha = 0.3, color = colors[count])
	count += 1
plt.xlabel('Temperature', size = fs)
plt.ylabel('99th percentile of speed during loom', size = fs)
plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
ax.set_title('Groupsize= 16', fontsize = fs)
plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions_all_loom_wo_data.png'
fig.savefig(out_dir, dpi = 300)


plt.show()