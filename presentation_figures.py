
"""
Goal - to produce figures for presentation to show during loom, after loom 

Date - May 10th 2021 
"""

import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import numpy as np

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
# ax = fig.add_subplot(111)


# x = np.array(range(500,1099))
# y = 52 + 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20)) #changing mean from 44
# y2 = 52 - 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20)) #changing mean from 44
# # ax.plot(x,y, color = 'black',linewidth = 0.75)
# # ax.plot(x,y2, color = 'black',linewidth = 0.75)
# ax.fill_between(x, y2, y, alpha = 1, color = 'gray', linewidth = 0)
# # sec = ax.secondary_xaxis(0.8)#,functions = (loom,nonloom))
# # sec.set_xlabel('Loom radius')
# plt.xlabel('Time (s)', size = fs)
# plt.ylabel('Loom size ', size = fs)
# plt.ylim(40,58)
# #plt.xlim(-10,10)
# plt.xticks(ticks = [500,800,1100,1400,1700], labels = [-10,-5,0,5,10],fontsize = fs)
# plt.yticks([])

# plt.annotate(text='', xy=(1000,49), xytext=(1200,49), arrowprops=dict(arrowstyle='<->')) # changing from 49

# plt.annotate(text='', xy=(1400,49), xytext=(1200,49), arrowprops=dict(arrowstyle='<->')) # changing from 49
# #plt.annotate(text='Loom size', xy=(700,53), fontsize = fs) #chanign from 45
# plt.annotate(text='Loom', xy=(1025,47), fontsize = fs) # changing from 45
# plt.annotate(text='(Predation threat)', xy=(1000,45), fontsize = fs) # changing from 41
# plt.annotate(text='Post-loom', xy=(1225,47), fontsize = fs) # changing from 45
# plt.show()

# out_dir = '../../output/temp_collective/roi_figures/presentation_loom_size.png'
# fig.savefig(out_dir, dpi = 1200, bbox_inches="tight")




# prelooms figure


ax1 = fig.add_subplot(111)


x = np.array(range(500,1099))
y = 52 + 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20)) #changing mean from 44
y2 = 52 - 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20)) #changing mean from 44
# ax.plot(x,y, color = 'black',linewidth = 0.75)
# ax.plot(x,y2, color = 'black',linewidth = 0.75)
ax1.fill_between(x, y2, y, alpha = 1, color = 'gray', linewidth = 0)
# sec = ax.secondary_xaxis(0.8)#,functions = (loom,nonloom))
# sec.set_xlabel('Loom radius')
plt.xlabel('Time (s)', size = fs)
plt.ylabel('Loom size ', size = fs)
plt.ylim(40,58)
#plt.xlim(-10,10)
plt.xticks(ticks = [-30000,500,800,1100,1400], labels = [-1010,-10,-5,0,5],fontsize = fs)
plt.yticks([])

# plt.annotate(text='', xy=(1000,49), xytext=(1200,49), arrowprops=dict(arrowstyle='<->')) # changing from 49

# plt.annotate(text='', xy=(1400,49), xytext=(1200,49), arrowprops=dict(arrowstyle='<->')) # changing from 49
# #plt.annotate(text='Loom size', xy=(700,53), fontsize = fs) #chanign from 45
# plt.annotate(text='Loom', xy=(1025,47), fontsize = fs) # changing from 45
# plt.annotate(text='(Predation threat)', xy=(1000,45), fontsize = fs) # changing from 41
# plt.annotate(text='Post-loom', xy=(1225,47), fontsize = fs) # changing from 45
plt.show()

out_dir = '../../output/temp_collective/roi_figures/presentation_loom_size_preloom.png'
fig.savefig(out_dir, dpi = 1200, bbox_inches="tight")
