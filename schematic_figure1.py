"""
Goal - to produce figures of convex hull area with loom sizes

Date - May 10th 2021 
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

# for mm in range(500,700):
#     ax.plot(mm, 44, color = 'black')
#     print(44 + 2*1000/(5000-(50/6)*(mm-500)))
# for m in frames:
#     ax.scatter(m, 44, s = 2*1000/(5000-(50/6)*(m-500)), color = 'black')
     
def loom(x):
    return np.round(np.sqrt(2*1000/((5000-(50/6)*(x-500))*20))/2,2)

    

def nonloom(x):
    return (5000 - 2000/(4*x^2))*(6/(50*20)) + 500


x = np.array(range(500,1099))
y = 44 + 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20))
y2 = 44 - 2*np.sqrt(2*1000/((5000-(50/6)*(x-500))*20))
# ax.plot(x,y, color = 'black',linewidth = 0.75)
# ax.plot(x,y2, color = 'black',linewidth = 0.75)
ax.fill_between(x, y2, y, alpha = 1, color = 'gray', linewidth = 0)
# sec = ax.secondary_xaxis(0.8)#,functions = (loom,nonloom))
# sec.set_xlabel('Loom radius')
plt.xlabel('Frame', size = fs)
plt.ylabel('Convex hull area ', size = fs)
plt.ylim(0,52)
plt.xticks(ticks = [700,1100,1500], labels = [-400,0,400],fontsize = fs)
# sec.set_xticks([700,1000,1099])
# sec.set_xticklabels([loom(700),loom(1000), loom(1099)])#(ticks = [700,1100,1500], labels = [-400,0,400],fontsize = fs)
# plt.xticks(ticks = [])#,fontsize = fs)
plt.yticks([0,10,20,30,40], labels = [0,10,20,30,40],fontsize = fs)

plt.annotate(text='', xy=(1000,47), xytext=(1200,47), arrowprops=dict(arrowstyle='<->'))

plt.annotate(text='', xy=(1400,47), xytext=(1200,47), arrowprops=dict(arrowstyle='<->'))
plt.annotate(text='Loom size', xy=(700,45), fontsize = fs)
plt.annotate(text='Loom', xy=(1050,49), fontsize = fs)
plt.annotate(text='Post-loom', xy=(1225,49), fontsize = fs)
plt.show()

custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4),
                Line2D([0], [0], color=colors[2], lw=4),
                Line2D([0], [0], color=colors[3], lw=4),
                Line2D([0], [0], color=colors[4], lw=4),
                Line2D([0], [0], color=colors[5], lw=4)]

#plt.legend()
#plt.legend(fontsize=fs, loc='upper right', framealpha = 0.5)
ax.legend(custom_lines, ['9' + r'$^{\circ}$C', '13' + r'$^{\circ}$C', '17' + r'$^{\circ}$C', '21' + r'$^{\circ}$C', '25' + r'$^{\circ}$C', '29' + r'$^{\circ}$C'], loc='upper left')
"""
custom_lines1 = [Line2D([0], [0], ls = ls[0], lw=4),
                Line2D([0], [0], ls = ls[1], lw=4),
                Line2D([0], [0], ls = ls[2], lw=4),
                Line2D([0], [0], ls = ls[3], lw=4)]


ax.legend(custom_lines1, ['4', '8', '16', '32'], loc='lower left')
"""
out_dir = '../../output/temp_collective/roi_figures/schematic_figure_1.png'
fig.savefig(out_dir, dpi = 1200, bbox_inches="tight")
