"""

Goal - one code to make prediction figures for all corrected parameters but editing previous code so that there is one figure with all group sizes

second round of revision - editing previous code so that legend is there only in one panel
Date - 12 Jul, 2022

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
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
dpi = 300

temp = [9,13,17,21,25,29]

#speed during loom

if args.a_string=='loom_speed_99_predictions_one_model_corrected.csv':
    
    data_hull = data2.speed_percentile99

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

    #plt.legend(fontsize=fs*0.75, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/loom_speed_99_predictions_w_data_one_model_corrected_all_revision2.pdf'
    fig3.savefig(out_dir, dpi = dpi, bbox_inches="tight")
    plt.show()



##########################################################################################################################################################
#acceleration during loom

if args.a_string=='loom_acc_99_int_predictions_one_model_corrected.csv':
    
    data_hull = data2.acc_percentile99

    #all with data

    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    
    for i in gs:
        
        for t in temp:
            parts = ax3.violinplot(data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
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
            np.exp(data1.acc99[data1.gs ==i][data1.loom == 1])-1, 
            color = colors[count], lw = lw,label = str(i))

        ax3.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            np.exp(data1.acc99_025[data1.gs ==i][data1.loom == 1])-1,  
            np.exp(data1.acc99_975[data1.gs ==i][data1.loom == 1])-1, alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Maximum acceleration (BL/s'+r'$^2$)', size = fs)

    #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    plt.legend(fontsize=fs*0.75, title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/loom_acc_99_int_predictions_w_data_one_model_corected_all_w_data_revision2.pdf'
    fig3.savefig(out_dir, dpi = dpi, bbox_inches="tight")


    plt.show()
    


#####################################################################################################################################################

#latency in seconds to loom 
if args.a_string=='latency_seconds_predictions_one_model_corrected.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    new_data = new_data.drop([740,742,322,685])
    data_hull = new_data.latency
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    
    #all with data
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in gs:
        
        for t in temp:
            parts = ax2.violinplot(data_hull[new_data.Groupsize == i][new_data.Loom == 1][new_data.Temperature == t]/60 -10, [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax2.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.latency[data1.gs ==i][data1.loom == 1]), 
            color = colors[count], lw = lw,label = str(i))

        ax2.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.latency_025[data1.gs ==i][data1.loom == 1]),  
            (data1.latency_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Latency (s)', size = fs)
    #plt.legend(fontsize=fs*0.75, loc = 'lower right',title = 'Groupsize', framealpha = 0.5)
    #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    ax2.set_ylim((-1.65,0.58))
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/latency_seconds_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig2.savefig(out_dir, dpi = dpi, bbox_inches="tight")


    plt.show()
    
    
####################################################################################################################################################

#prop startles predictions

if args.a_string=='prop_startles_predictions_one_model_corrected.csv':
    
    data_hull = data2.prop_startles

    # all with data
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
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
            color = colors[count], lw = lw,label = str(i))

        ax.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.prop_startles025[data1.gs ==i][data1.loom == 1]),  
            (data1.prop_startles975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Proportion of individuals \n startling', size = fs)

    #plt.legend(fontsize=fs*0.75, loc='upper right', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/prop_startles_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()
    
    
    
    
####################################################################################################################################################
####################################################################################################################################################



###post loom collective behavior
#annd
if args.a_string=='annd_predictions_one_model_corrected.csv':
    
    data_hull = data2.annd
    gs = [2,4,8,16]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #all with data
    
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in gs:
        for t in temp:
            parts = ax2.violinplot((data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t]), [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax2.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            np.exp(data1.annd[data1.gs ==i][data1.loom == 1]), 
            color = colors[count], lw = lw,label = str(i))

        ax2.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            np.exp(data1.annd_025[data1.gs ==i][data1.loom == 1]),  
            np.exp(data1.annd_975[data1.gs ==i][data1.loom == 1]), alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.yscale('log')
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
    plt.yticks(fontsize = fs*2)
    #plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
    plt.ylabel('ANND (BL)', size = fs*2)

    #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5, title_fontsize=fs)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/annd_int_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig2.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()
    
    
    
    
####################################################################################################################################################


#convex_hull_area after loom        
if args.a_string=='hull_predictions_one_model_corrected.csv':
    
    data_hull = data2.convex_hull_area
    
    #all with data
    gs = [4,8,16]
    count = 3
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+3))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in gs:
        for t in temp:
            parts = ax2.violinplot((data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t]), [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax2.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.hull[data1.gs ==i][data1.loom == 1])**2, 
            color = colors[count], lw = lw,label = str(i))

        ax2.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.hull_025[data1.gs ==i][data1.loom == 1])**2,  
            (data1.hull_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    #plt.yscale('log')
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
    plt.yticks(fontsize = fs*2)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
    plt.ylabel('Convex hull \n area (BL'+r'$^2$)', size = fs*2)

    #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5, title_fontsize = fs)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/hull_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig2.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()    
    
    
    
####################################################################################################################################################


#pol post loom  
if args.a_string=='pol_postloom_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_pol_corrected.csv')
    data_hull = data2.polarization_1_postloom
    #all with data

    gs = [2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in gs:
        for t in temp:
            parts = ax2.violinplot((np.abs(data_hull[data2.Groupsize == i][data2.Loom == 1][data2.Temperature == t])), [t], widths = 1)
            parts['bodies'][0].set_facecolor(colors[count])
            parts['bodies'][0].set_edgecolor(colors[count])
            parts['bodies'][0].set_alpha(0.2)
            
            for partname in ('cbars','cmins','cmaxes'):
                vp = parts[partname]
                vp.set_edgecolor(colors[count])
                vp.set_linewidth(1)
                vp.set_alpha(0.8)
        
        ax2.plot(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.pol[data1.gs ==i][data1.loom == 1])**2, 
            color = colors[count], lw = lw,label = str(i))

        ax2.fill_between(
            data1.temp[data1.gs ==i][data1.loom == 1], 
            (data1.pol_025[data1.gs ==i][data1.loom == 1])**2,  
            (data1.pol_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    #plt.yscale('log')
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs*2)
    plt.yticks(fontsize = fs*2)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs*2)
    plt.ylabel('Polarization', size = fs*2)

    plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5,title_fontsize=fs)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/pol_postloom_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig2.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()    
    
    
####################################################################################################################################################
####################################################################################################################################################


#preloom speed


if args.a_string=='preloom_speed_99_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    data_hull = data2.speed_percentile99
    
    #all_with data
    gs = [1,2,4,8,16]
    count = 2
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
            color = colors[count], lw = lw,label = str(i))

        ax.fill_between(
            data1.temp[data1.gs ==i], 
            np.exp(data1.speed99_025[data1.gs ==i])-1,  
            np.exp(data1.speed99_975[data1.gs ==i])-1, alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Maximum speed (BL/s)', size = fs)

    #plt.legend(fontsize=fs*0.75, loc='upper left', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/preloom_speed_99_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()
    
    
####################################################################################################################################################

#pre loom acc

if args.a_string=='preloom_acc_99_predictions_one_model_corrected.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom_corrected.csv')
    
    #data2 = data2.reset_index(drop=True)
    data2 = data2.drop(125)
    data_hull = data2.acc_percentile99   
    
    #all_with_data
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
            color = colors[count], lw = lw,label = str(i))

        ax.fill_between(
            data1.temp[data1.gs ==i], 
            np.exp(data1.acc99_025[data1.gs ==i])-1,  
            np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.2, 
            color = colors[count], lw = 0)
        count +=1
    
    plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.locator_params(axis = 'y', nbins = 4)
    plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
    plt.ylabel('Maximum acceleration (BL/s'+r'$^2$)', size = fs)

    plt.legend(fontsize=fs*0.75, loc='upper left', title = 'Groupsize', framealpha = 0.5)
    out_dir = '../../output/temp_collective/roi_figures/predictions/revisions2/preloom_acc_99_predictions_w_data_one_model_corrected_all_revisions2.pdf'
    fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

    
    plt.show()
