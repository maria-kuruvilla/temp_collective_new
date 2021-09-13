"""

Goal - one code to make prediction figures for all parameters. One model is used for all parameteres. Try plotting boxplot of data with predtictions.
Date - 6th September, 2021

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

data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')



#Plotting
lw=2
fs=16
fig = plt.figure()
ax = fig.add_subplot(111)
dpi = 100

temp = [9,13,17,21,25,29]
### speed during loom

#sqrt(speed) ~ temp + I(temp^2) + log(gs,2) + loom
#$variable
#[1] "temp"       "I(temp^2)"  "log(gs, 2)" "loom"      

#$partial.rsq
#[1] 0.003723370 0.009238227 0.110646788 0.014898521

# (Intercept)  2.3527820  0.2279874  10.320  < 2e-16 ***
# temp         0.0518650  0.0252381   2.055   0.0401 *  
# I(temp^2)   -0.0021388  0.0006589  -3.246   0.0012 ** 
# log(gs, 2)   0.2284672  0.0192687  11.857  < 2e-16 ***
# loom        -0.0783119  0.0189434  -4.134 3.83e-05 ***

if args.a_string=='loom_speed_99_predictions_one_model.csv':
    
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
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


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_99_predictions_w_data_one_model.png'
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
                (data1.speed99_975[data1.gs ==16][data1.loom == 1])**2, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(16))
            count +=1
        
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum speed (BL/s)', size = fs)


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_99_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")
        plt.show()


#acceleration during loom 
# (Intercept)  4.7577198  0.1671428  28.465  < 2e-16 ***
#   temp         0.0731255  0.0185027   3.952 8.23e-05 ***
#   I(temp^2)   -0.0025823  0.0004831  -5.346 1.09e-07 ***
#   log(gs, 2)   0.1498198  0.0141263  10.606  < 2e-16 ***
#   loom        -0.0568259  0.0138878  -4.092 4.58e-05 ***
#   


# 
# variable
# [1] "temp"       "I(temp^2)"  "log(gs, 2)" "loom"      
# 
# $partial.rsq
# [1] 0.01363420 0.02466482 0.09052954 0.01460022
# 
if args.a_string=='loom_acc_99_predictions_one_model.csv':
    
    data_hull = data2.acc_percentile99
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[data2.Groupsize == 16][data2.Loom == 1][data2.Temperature == t], [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.acc99[data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.acc99_025[data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.acc99_975[data1.gs ==16][data1.loom == 1])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(16))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_predictions_w_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.acc99[data1.gs ==16][data1.loom == 1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==16][data1.loom == 1], 
                np.exp(data1.acc99_025[data1.gs ==16][data1.loom == 1])-1,  
                np.exp(data1.acc99_975[data1.gs ==16][data1.loom == 1])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(16))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)


        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()

#accleration during loom with interaction

# $variable
# [1] "temp"            "I(temp^2)"       "log(gs, 2)"      "loom"            "temp:log(gs, 2)"

# $partial.rsq
# [1] 0.016541258 0.025582519 0.027135200 0.014654451 0.003755625

# > rsq(model_lm_int)
# [1] 0.1823035
if args.a_string=='loom_acc_99_int_predictions_one_model.csv':
    
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.acc99[data1.gs ==i][data1.loom == 1])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.acc99_025[data1.gs ==i][data1.loom == 1])-1,  
                np.exp(data1.acc99_975[data1.gs ==i][data1.loom == 1])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_int_predictions_w_data_one_model.png'
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
                np.exp(data1.acc99_975[data1.gs ==i][data1.loom == 1])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)

        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_acc_99_int_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()



#latency
# > rsq(model_pois6)
# [1] 0.06258895

# $variable
# [1] "temp"       "loom"       "I(temp^2)"  "log(gs, 2)"
# 
# $partial.rsq
# [1] 0.017143434 0.004740903 0.016090973 0.041742987

if args.a_string=='latency_predictions_one_model.csv':
    
    new_data = data2[['Temperature','Groupsize','Loom','latency']]
    new_data = new_data.dropna()
    new_data = new_data.reset_index(drop=True)
    new_data = new_data.drop([749,751,326,694])
    data_hull = new_data.latency
    gs = [16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            for t in temp:
                parts = ax.violinplot(data_hull[new_data.Groupsize == i][new_data.Loom == 1][new_data.Temperature == t]/60, [t], widths = 1)
                parts['bodies'][0].set_facecolor(colors[count])
                parts['bodies'][0].set_edgecolor(colors[count])
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.pred[data1.gs ==i][data1.loom == 1])/60, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.lcb[data1.gs ==i][data1.loom == 1])/60,  
                (data1.ucb[data1.gs ==i][data1.loom == 1])/60, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_predictions2_w_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.pred[data1.gs ==i][data1.loom == 1])/60, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.lcb[data1.gs ==i][data1.loom == 1])/60,  
                (data1.ucb[data1.gs ==i][data1.loom == 1])/60, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency (s)', size = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)

        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()

#prop startles predictions

if args.a_string=='prop_startles_predictions_one_model.csv':
    
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.prop_startles[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.prop_startles025[data1.gs ==i][data1.loom == 1]),  
                (data1.prop_startles975[data1.gs ==i][data1.loom == 1]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_w_data_one_model.png'
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
                (data1.prop_startles975[data1.gs ==i][data1.loom == 1]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Proportion of individuals \n startling', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")


       
        
        plt.show()


#pre loom speed


if args.a_string=='preloom_speed_99_predictions_one_model.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.speed99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.speed99_025[data1.gs ==i])-1,  
                np.exp(data1.speed99_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum speed (BL/s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_99_predictions_w_data_one_model.png'
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
                np.exp(data1.speed99_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum speed (BL/s)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_speed_99_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")



       
        
        plt.show()



#pre loom acc

# > rsq(model_lm)
# [1] 0.0984856
# > rsq.partial(model_lm)
# $adjustment
# [1] FALSE

# $variable
# [1] "temp"       "I(temp^2)"  "log(gs, 2)"

# $partial.rsq
# [1] 0.01912525 0.01266149 0.05780389

if args.a_string=='preloom_acc_99_predictions_one_model.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i],
                np.exp(data1.acc99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc99_025[data1.gs ==i])-1,  
                np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_99_predictions_w_data_one_model.png'
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
                np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Near maximum \n acceleration (BL/s'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/preloom_acc_99_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()

#annnd after loom

# > rsq(model_lm_trans_int)
# [1] 0.4167516
# > rsq.partial(model_lm_trans_int)
# $adjustment
# [1] FALSE
# 
# $variable
# [1] "temp"            "log(gs, 2)"      "loom"            "I(temp^2)"       "temp:log(gs, 2)"
# 
# $partial.rsq
# [1] 1.104543e-02 1.134305e-01 2.800796e-06 3.398251e-03 6.344730e-03



if args.a_string=='annd_predictions_one_model.csv':
    
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.annd[data1.gs ==i][data1.loom == 1]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                np.exp(data1.annd_025[data1.gs ==i][data1.loom == 1]),  
                np.exp(data1.annd_975[data1.gs ==i][data1.loom == 1]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('ANND (BL)', size = fs)

        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_int_predictions_w_data_one_model.png'
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
                np.exp(data1.annd_975[data1.gs ==i][data1.loom == 1]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('ANND (BL/s)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)

        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_int_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
        
        
# hull after loom


# > rsq(model_lm)
# [1] 0.4178037
# > rsq.partial(model_lm)
# $adjustment
# [1] FALSE
# 
# $variable
# [1] "temp"       "I(temp^2)"  "loom"       "log(gs, 2)"
# 
# $partial.rsq


# [1] 0.0060876909 0.0011404016 0.0004368472 0.3924370723
if args.a_string=='hull_predictions_one_model.csv':
    
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
                for partname in ('cbars','cmins','cmaxes'):
                    vp = parts[partname]
                    vp.set_edgecolor(colors[count])
                    vp.set_linewidth(1)
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.hull[data1.gs ==i][data1.loom == 1])**2, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.loom == 1], 
                (data1.hull_025[data1.gs ==i][data1.loom == 1])**2,  
                (data1.hull_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Convex hull area (BL'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/hull_predictions_w_data_one_model.png'
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
                (data1.hull_975[data1.gs ==i][data1.loom == 1]**2), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        #plt.yscale('log')
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Convex hull area (BL'+r'$^2$)', size = fs)

        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/hull_predictions_wo_data_one_model.png'
        fig.savefig(out_dir, dpi = dpi, bbox_inches="tight")

        
        plt.show()
        