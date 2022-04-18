"""

Goal - one code to make prediction figures for all corrected parameters but editing previous code so that there is one figure with all group sizes
Date - 18 April, 2022

"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import cm
import argparse
import seaborn as sns 

#argparse
def boolean_string(s):
    # this function helps with getting Boolean input
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True' # note use of ==

# create the parser object
parser = argparse.ArgumentParser()

# NOTE: argparse will throw an error if:
#     - a flag is given with no value
#     - the value does not match the type
# and if a flag is not given it will be filled with the default.
parser.add_argument('-a', '--a_string', default='annd_after_loom_predictions.csv', type=str)
#parser.add_argument('-s', '--a_string', default='annd_std', type=str)
parser.add_argument('-b', '--integer_b', default=3, type=int)
parser.add_argument('-c', '--float_c', default=1.5, type=float)
parser.add_argument('-v', '--verbose', default=True, type=boolean_string)
# Note that you assign a short name and a long name to each argument.
# You can use either when you call the program, but you have to use the
# long name when getting the values back from "args".

# get the arguments
args = parser.parse_args()

#################################################################################

data1 = pd.read_csv('../../data/temp_collective/roi/'+args.a_string)

data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_corrected.csv')



#Plotting
lw=3
fs=16
fig = plt.figure()
ax = fig.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
dpi = 100

temp = [9,13,17,21,25,29]

#speed during loom

if args.a_string=='loom_speed_99_predictions_one_model_corrected.csv':
    
    data_hull = data2.speed_percentile99
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #individual with data
    for i in gs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for t in temp:
            parts = ax.violinplot(data2.speed_percentile99[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99[data1.gs ==i][data1.loom == 1])**2, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99_025[data1.gs ==i][data1.loom == 1])**2,  
            (data1.speed99_975[data1.gs ==i][data1.loom == 1])**2, alpha = 0.2, 
            color = colors[count], lw = 0,label = str(i))
        count +=1
    
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed (BL/s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/revisions/loom_speed_99_predictions_w_data_one_model_corrected_'+str(i)+'.pdf'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
        plt.show()
    

    #all without data
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in gs:
        
        
        ax2.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99[data1.gs ==i][data1.loom == 1])**2, 
            color = colors[count], lw = lw,label = str(i))

        ax2.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99_025[data1.gs ==i][data1.loom == 1])**2,  
            (data1.speed99_975[data1.gs ==i][data1.loom == 1])**2, alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Maximum speed (BL/s)', size = fs)

    plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions/loom_speed_99_predictions_wo_data_one_model_corrected_all.pdf'
    fig2.savefig(out_dir, dpi = dpi, bbox_inches="tight")
    plt.show()

    # all with data
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    for i in gs:
        for t in temp:
            parts = ax3.violinplot(data2.speed_percentile99[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax3.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99[data1.gs ==i][data1.loom == 1])**2, 
            color = colors[count], lw = lw,label = str(i))

        ax3.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.speed99_025[data1.gs ==i][data1.loom == 1])**2,  
            (data1.speed99_975[data1.gs ==i][data1.loom == 1])**2, alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Maximum speed (BL/s)', size = fs)

    plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions/loom_speed_99_predictions_wo_data_one_model_corrected_all_w_data.pdf'
    fig3.savefig(out_dir, dpi = dpi, bbox_inches="tight")
    plt.show()


