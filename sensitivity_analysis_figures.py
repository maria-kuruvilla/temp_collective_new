"""

Goal - one code to make prediction figures for different thresholds to show sensitivity
Date - 26th October, 2021

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

data2 = pd.read_csv('../../data/temp_collective/roi/sensitivity_analysis_new.csv')

#Plotting
lw=3
fs=16
fig = plt.figure()
ax = fig.add_subplot(111)
dpi = 100

temp = [9,13,17,21,25,29]



#latency in seconds to loom 
if args.a_string=='latency_seconds_predictions_one_model_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency8_25_3000','latency10_25_3000','latency12_25_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t]/60 -10, [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*2 +2))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='lower left', title = 'Startle threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_w_data_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*2 +2))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Startle threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_wo_data_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    



#prop startles 
if args.a_string=='prop_startles_predictions_one_model_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','prop_startles8_25_3000','prop_startles10_25_3000','prop_startles12_25_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*2 +2))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Startle threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_w_data_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*2 +2))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Startle threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_wo_data_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#speed during loom

if args.a_string=='speed_predictions_one_model_speed_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','speed25_3000','speed30_3000','speed35_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])**2, 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])**2,  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])**2, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_predictions_w_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])**2, 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])**2,  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])**2, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_predictions_wo_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#latency in seconds to loom - speed threshold sensitivity
if args.a_string=='latency_seconds_predictions_one_model_speed_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency10_25_3000','latency10_30_3000','latency10_35_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t]/60 -10, [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='lower left', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_w_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_wo_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    



#latency in seconds to loom - acc threshold sensitivity
if args.a_string=='latency_seconds_predictions_one_model_acc_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency10_30_2000','latency10_30_3000','latency10_30_4000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t]/60 -10, [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='lower left', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_w_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_wo_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()




#prop startles - speed threshold
if args.a_string=='prop_startles_predictions_one_model_speed_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','prop_startles10_25_3000','prop_startles10_30_3000','prop_startles10_35_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_w_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_wo_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()



#prop startles - acc threshold
if args.a_string=='prop_startles_predictions_one_model_acc_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','prop_startles10_30_2000','prop_startles10_30_3000','prop_startles10_30_4000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_w_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1]), 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1]),  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_wo_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#speed during loom

if args.a_string=='speed_predictions_one_model_acc_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','speed30_2000','speed30_3000','speed30_4000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])**2, 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])**2,  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])**2, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_predictions_w_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])**2, 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                (data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])**2,  
                (data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])**2, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_predictions_wo_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()



#acc during loom - speed sensitivity

if args.a_string=='acc_predictions_one_model_speed_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','acc25_3000','acc30_3000','acc35_3000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_predictions_w_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw,label = str(i*5 +10))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Speed threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_predictions_wo_data_speed_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#acc during loom - acc sensitivity

if args.a_string=='acc_predictions_one_model_acc_threshold_sensitivity.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','acc30_2000','acc30_3000','acc30_4000']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    #new_data = new_data.drop([749,751,326,694])
    
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,3+1))
    if args.verbose==True:
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == 16][new_data.Loom == 1][new_data.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_predictions_w_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        for i in [3,4,5]:
            data_hull = new_data.iloc[:,i]
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-5][data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw,label = str(i*1000 -1000))

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.iloc[:,i*3-4][data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.iloc[:,i*3-3][data1.gs ==16][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0)
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Acceleration threshold', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_predictions_wo_data_acc_threshold_sensitivity.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
