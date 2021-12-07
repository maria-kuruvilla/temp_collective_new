"""

Goal - one code to make prediction figures for all corrected parameters. One model is used for all parameteres. Try plotting boxplot of data with predtictions.
Date - 1 Dec, 2021

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
dpi = 100

temp = [9,13,17,21,25,29]



#speed during loom

if args.a_string=='loom_speed_99_predictions_one_model_corrected.csv':
    
    data_hull = data2.speed_percentile99
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
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
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed (BL/s)', size = fs)


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_99_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
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
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed (BL/s)', size = fs)


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_99_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")
        plt.show()



#acceleration during loom

if args.a_string=='loom_acc_99_int_predictions_one_model_corrected.csv':
    
    data_hull = data2.acc_percentile99
    gs = [1,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
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
                np.exp(data1.acc99[data1.gs ==i][data1.loom == 1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.acc99_025[data1.gs ==i][data1.loom == 1])-1,  
                np.exp(data1.acc99_975[data1.gs ==i][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_int_predictions_w_data_one_model_corected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [1,16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.acc99[data1.gs ==i][data1.loom == 1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.acc99_025[data1.gs ==i][data1.loom == 1])-1,  
                np.exp(data1.acc99_975[data1.gs ==i][data1.loom == 1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum \n acceleration (BL/s'+r'$^2$)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)

        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_int_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#latency in seconds to loom 
if args.a_string=='latency_seconds_predictions_one_model_corrected.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    new_data = new_data.drop([740,742,322,685])
    data_hull = new_data.latency
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == i][new_data.Loom == 1][new_data.Temperature == t]/60 -10, [t], widths = 1)
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
                (data1.latency[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.latency_025[data1.gs ==i][data1.loom == 1]),  
                (data1.latency_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.latency[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.latency_025[data1.gs ==i][data1.loom == 1]),  
                (data1.latency_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_seconds_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()

#prop startles predictions

if args.a_string=='prop_startles_predictions_one_model_corrected.csv':
    
    data_hull = data2.prop_startles
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
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
                (data1.prop_startles[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.prop_startles025[data1.gs ==i][data1.loom == 1]),  
                (data1.prop_startles975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.prop_startles[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.prop_startles025[data1.gs ==i][data1.loom == 1]),  
                (data1.prop_startles975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")


       
        
        plt.show()

#annd after loom

if args.a_string=='annd_predictions_one_model_corrected.csv':
    
    data_hull = data2.annd
    gs = [2,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot((data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t]), [t], widths = 1)
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
                np.exp(data1.annd[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.annd_025[data1.gs ==i][data1.loom == 1]),  
                np.exp(data1.annd_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
        plt.yticks(fontsize = fs*2)
        #plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
        plt.ylabel('ANND (BL)', size = fs*2)

        plt.legend(fontsize=fs*1.3, loc='upper right', title = 'Groupsize', framealpha = 0.5, title_fontsize=fs*1.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_int_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [2,16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.annd[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.annd_025[data1.gs ==i][data1.loom == 1]),  
                np.exp(data1.annd_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('ANND (BL/s)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)

        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_int_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
        
        
#convex_hull_area after loom        
if args.a_string=='hull_predictions_one_model_corrected.csv':
    
    data_hull = data2.convex_hull_area
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot((data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t]), [t], widths = 1)
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
                (data1.hull[data1.gs ==i][data1.loom == 1])**2, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.hull_025[data1.gs ==i][data1.loom == 1])**2,  
                (data1.hull_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
        plt.yticks(fontsize = fs*2)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
        plt.ylabel('Convex hull \n area (BL'+r'$^2$)', size = fs*2)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/hull_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.hull[data1.gs ==i][data1.loom == 1])**2, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.hull_025[data1.gs ==i][data1.loom == 1])**2,  
                (data1.hull_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Convex hull area (BL'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/hull_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
        

if args.a_string=='pol_postloom_predictions_one_model.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_pol_corrected.csv')
    data_hull = data2.polarization_1_postloom
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot((np.abs(data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t])), [t], widths = 1)
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
                (data1.pol[data1.gs ==i][data1.loom == 1])**2, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.pol_025[data1.gs ==i][data1.loom == 1])**2,  
                (data1.pol_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
        plt.yticks(fontsize = fs*2)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
        plt.ylabel('Polarization', size = fs*2)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pol_postloom_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.pol[data1.gs ==i][data1.loom == 1])**2, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.pol_025[data1.gs ==i][data1.loom == 1])**2,  
                (data1.pol_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Polarization', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pol_postloom_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")
        
        plt.show()
       
        
        
#preloom speed


if args.a_string=='preloom_speed_99_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.speed_percentile99
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.speed99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.speed99_025[data1.gs ==i])-1,  
                np.exp(data1.speed99_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed (BL/s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_99_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.speed99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.speed99_025[data1.gs ==i])-1,  
                np.exp(data1.speed99_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum speed (BL/s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_99_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")



       
        
        plt.show()
        
        
        
#pre loom acc

if args.a_string=='preloom_acc_99_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.acc_percentile99
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.acc99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc99_025[data1.gs ==i])-1,  
                np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum acceleration (BL/s'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_99_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.acc99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc99_025[data1.gs ==i])-1,  
                np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Maximum \n acceleration (BL/s'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_99_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()



#appendix figures

#avg acc

if args.a_string=='avg_acc_preloom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.avg_acc
    gs = [1,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.avg_acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.avg_acc_025[data1.gs ==i])-1,  
                np.exp(data1.avg_acc_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_avg_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [1,16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.avg_acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.avg_acc_025[data1.gs ==i])-1,  
                np.exp(data1.avg_acc_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_avg_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#median acc

#median acceleration before loom

if args.a_string=='median_acc_preloom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.acc_percentile50
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.acc50[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc50_025[data1.gs ==i])-1,  
                np.exp(data1.acc50_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_50_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.acc50[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc50_025[data1.gs ==i])-1,  
                np.exp(data1.acc50_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_50_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#avg speed before loom

if args.a_string=='avg_speed_preloom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.avg_speed
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.avg_speed[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.avg_speed_025[data1.gs ==i])-1,  
                np.exp(data1.avg_speed_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_avg_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.avg_speed[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.avg_speed_025[data1.gs ==i])-1,  
                np.exp(data1.avg_speed_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_avg_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()

#median speed before loom

if args.a_string=='median_speed_preloom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.speed_percentile50
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.speed50[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.speed50_025[data1.gs ==i])-1,  
                np.exp(data1.speed50_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_50_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.speed50[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.speed50_025[data1.gs ==i])-1,  
                np.exp(data1.speed50_975[data1.gs ==i])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_50_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()





#avg speed during loom

if args.a_string=='avg_speed_loom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_corrected.csv')
    data_hull = data2.avg_speed
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t][data2.Loom == 1], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.avg_speed[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.avg_speed_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.avg_speed_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_avg_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1][data1.loom ==1],
                np.exp(data1.avg_speed[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.avg_speed_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.avg_speed_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_avg_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")
        
        plt.show()


#median speed before loom

if args.a_string=='median_speed_loom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_corrected.csv')
    data_hull = data2.speed_percentile50
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t][data2.Loom == 1], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.speed50[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.speed50_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.speed50_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_50_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.speed50[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.speed50_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.speed50_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median speed (BL/s)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_50_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
        

#avg acc during loom

if args.a_string=='average_acc_loom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_corrected.csv')
    data_hull = data2.avg_acc
    gs = [1,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t][data2.Loom == 1], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.avg_acc[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.avg_acc_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.avg_acc_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_avg_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [1,16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.avg_acc[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.avg_acc_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.avg_acc_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Average acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_avg_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()


#median acc before loom

if args.a_string=='median_acc_loom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_corrected.csv')
    data_hull = data2.acc_percentile50
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == i][data2.Temperature == t][data2.Loom == 1], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                parts['bodies'][0].set_alpha(0.2)
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
                    vp.set_alpha(0.8)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.acc50[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.acc50_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.acc50_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_50_predictions_w_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom ==1],
                np.exp(data1.acc50[data1.gs ==i][data1.loom ==1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom ==1], 
                np.exp(data1.acc50_025[data1.gs ==i][data1.loom ==1])-1,  
                np.exp(data1.acc50_975[data1.gs ==i][data1.loom ==1])-1, alpha = 0.2, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Median acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_50_predictions_wo_data_one_model_corrected.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        

        
        plt.show()
