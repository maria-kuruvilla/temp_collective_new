"""
Goal - pass argument to make figure with and without data
Date - Mar 15 2021
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

gs = [1,2,4,8,16]
colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
#Plotting
lw=2
fs=30
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
count = 2
dpi = 100
text = '_low_res'


if args.a_string=='annd_after_loom_predictions.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/annd_after_loom_predictions.csv')

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    y_label = 'ANND (Body Length)'
    gs = [2,4,8,16]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #Plotting
    
    
    if args.verbose==True:
        
        for i in gs:
            
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data2["annd"][data2.Groupsize == i], alpha = 0.5, color = colors[count], s =10)
        
            ax.plot(
                data1.temp[data1.gs == i][data1.date == 18106][data1.trial == 10], 
                np.exp(data1.annd[data1.gs==i][data1.date == 18106][data1.trial == 10]), color = colors[count],
                 lw = lw)

            ax.fill_between(
                data1.temp[data1.gs == i][data1.date == 18106][data1.trial == 10], 
                np.exp(data1.annd025[data1.gs==i][data1.date == 18106][data1.trial == 10]),  
                np.exp(data1.annd975[data1.gs==i][data1.date == 18106][data1.trial == 10]), alpha = 0.3, color = colors[count], label = str(i),lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('ANND (BL)', size = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        legend = plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        plt.setp(legend.get_title(),fontsize='xx-large')
        ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_after_loom_predictions_w_data_all_low_res.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()
    else:
        gs = [2,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:
        
            ax.plot(
                data1.temp[data1.gs == i][data1.date == 18106][data1.trial == 10], 
                np.exp(data1.annd[data1.gs==i][data1.date == 18106][data1.trial == 10]), color = colors[count],
                 lw = lw)

            ax.fill_between(
                data1.temp[data1.gs == i][data1.date == 18106][data1.trial == 10], 
                np.exp(data1.annd025[data1.gs==i][data1.date == 18106][data1.trial == 10]),  
                np.exp(data1.annd975[data1.gs==i][data1.date == 18106][data1.trial == 10]), alpha = 0.3, color = colors[count],label = str(i), lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('ANND (BL)', size = fs)
        
        legend = plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        plt.setp(legend.get_title(),fontsize='xx-large')
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_after_loom_predictions_wo_data_all_low_res.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()


#model_glm_7 <- glm(startles_during_loom ~ I(Temperature^2) + Temperature + Groupsize + I(Groupsize^2) + Loom, family = poisson, data1)

if args.a_string=='number_startles_predictions.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/number_startles_predictions.csv')

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    
    gs = [1,2,4,8,16]
    loom = [1,5]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #Plotting
    
    
    if args.verbose==True:
        
        for i in gs:
            
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2["number_startles"][data2.Groupsize == i][data2.Loom == 1], alpha = 0.5, color = colors[count], s =10)
        
            ax.plot(
                data1.Temperature[data1.Groupsize == i][data1.Loom == 1], 
                (data1.startles[data1.Groupsize ==i][data1.Loom == 1]), color = colors[count],
                label = str(i), lw = lw)

            ax.fill_between(
                data1.Temperature[data1.Groupsize == i][data1.Loom == 1], 
                (data1.startles025[data1.Groupsize==i][data1.Loom == 1]),  
                (data1.startles975[data1.Groupsize==i][data1.Loom == 1]), alpha = 0.3, color = colors[count])
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Number of startles', size = fs)
        ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/startles_w_data_all.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:
        gs = [16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:
            
            ax.plot(
                data1.Temperature[data1.Groupsize == i][data1.Loom == 1], 
                (data1.startles[data1.Groupsize ==i][data1.Loom == 1]), color = colors[count],
                label = str(i), lw = lw)

            ax.fill_between(
                data1.Temperature[data1.Groupsize == i][data1.Loom == 1], 
                (data1.startles025[data1.Groupsize ==i][data1.Loom == 1]),  
                (data1.startles975[data1.Groupsize ==i][data1.Loom == 1]), alpha = 0.3, color = colors[count])
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Number of startles', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        ax.set_title('Loom = 1', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/startles_predictions_wo_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()

#model_pois6 <- glm(latency ~ temp*gs + I(temp^2) + loom, family = quasipoisson, my_new_data2)
if args.a_string=='latency_predictions.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/'+args.a_string)

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    
    gs = [1,2,4,8,16]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #Plotting
    
    
    
    if args.verbose==True:
        
        for i in gs:
            
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2.latency[data2.Groupsize == i][data2.Loom == 1], s = 10, alpha = 0.5, color = colors[count])
        
            ax.plot(
                data1.temp[data1.gs== i][data1.loom == 1], 
                (data1.pred[data1.gs==i][data1.loom == 1]), color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.gs== i][data1.loom == 1], 
                (data1.lcb[data1.gs==i][data1.loom == 1]),  
                (data1.ucb[data1.gs==i][data1.loom == 1]), alpha = 0.3, color = colors[count], label = str(i),lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.yticks(ticks = [580,585,590,595], labels = [580,585,590,595],fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_w_data.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()
    else:
        gs = [1,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:
        
            ax.plot(
                data1.temp[data1.gs== i][data1.loom == 1], 
                (data1.pred[data1.gs==i][data1.loom == 1]), color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.gs== i][data1.loom == 1], 
                (data1.lcb[data1.gs==i][data1.loom == 1]),  
                (data1.ucb[data1.gs==i][data1.loom == 1]), alpha = 0.3, color = colors[count], label = str(i),lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Latency', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        legend = plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        plt.setp(legend.get_title(),fontsize='xx-large')
        plt.yticks(ticks = [580,585,590,595], labels = [580,585,590,595],fontsize = fs)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/latency_wo_data.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()





#model_glm <- glm(hull ~ gs + temp*loom + I(temp^2) + date, my_data, family = "Gamma")
if args.a_string=='hull_ratio_600_650_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/convex_hull_ratio_600_650w_loom.csv')
    data_hull = data2.convex_hull_area_600_650
    gs = [16]
    loom = [1,5]
    colors = plt.cm.bone_r(np.linspace(0,1,len(loom)+2))
    if args.verbose==True:
        
        for i in gs:
            for j in loom:
                ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == j],
                    data_hull[data2.Groupsize == i][data2.Loom == j], s = 10, alpha = 0.5, 
                    color = colors[count])
                
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    (data1.hull[data1.gs==i][data1.loom == j][data1.date == 18106]), 
                    color = colors[count], label = str(j), lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    (data1.hull025[data1.gs==i][data1.loom == j][data1.date == 18106]),  
                    (data1.hull975[data1.gs==i][data1.loom == j][data1.date == 18106]), alpha = 0.3, 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Ratio of convex hull area during loom to \n convex hull area after loom', size = fs)
        ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_ratio_w_data_loom_gs16.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        for i in gs:
        
            for j in loom:
                
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    (data1.hull[data1.gs==i][data1.loom == j][data1.date == 18106]), 
                    color = colors[count], label = str(j), lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    (data1.hull025[data1.gs==i][data1.loom == j][data1.date == 18106]),  
                    (data1.hull975[data1.gs==i][data1.loom == j][data1.date == 18106]), alpha = 0.3, 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Ratio of convex hull area during loom to \n convex hull before after loom', size = fs)
        ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_ratio_wo_data_loom_gs16.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


#hull ratio - during/before

#model_lm <- lm(log(hull)~ log(gs,2) + I(temp^2) + loom + date, my_data)
if args.a_string=='hull_ratio_600_650_predictions2.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/convex_hull_ratio_600_650w_loom.csv')
    data_hull = data2.convex_hull_area_ratio_loom
    gs = [4,8,16]
    loom = [1]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    count = 1
    if args.verbose==True:
        
        for i in gs:
            for j in loom:
                ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == j],
                    data_hull[data2.Groupsize == i][data2.Loom == j], s = 10, alpha = 0.5, 
                    color = colors[count])
                
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.hull[data1.gs==i][data1.loom == j][data1.date == 18106]), 
                    color = colors[count], label = str(i), lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.hull025[data1.gs==i][data1.loom == j][data1.date == 18106]),  
                    np.exp(data1.hull975[data1.gs==i][data1.loom == j][data1.date == 18106]), alpha = 0.3, 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Ratio of convex hull area during loom to \n convex hull before loom', size = fs)
        ax.set_title('Loom = '+str(loom[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_ratio_loom_w_data_loom1_gs_all.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:
        gs = [16]
        loom = [1]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
        
            for j in loom:
                
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.hull[data1.gs==i][data1.loom == j][data1.date == 18106]), 
                    color = colors[count], label = str(j), lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.hull025[data1.gs==i][data1.loom == j][data1.date == 18106]),  
                    np.exp(data1.hull975[data1.gs==i][data1.loom == j][data1.date == 18106]), alpha = 0.3, 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Ratio of convex hull area during loom to \n convex hull area after loom', size = fs)
        ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_ratio_loom_wo_data_loom1_gs16.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()




#model_lm <- lm(log(speed+1) ~ temp,my_data)
if args.a_string=='speed99_before_loom_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data_hull = data2.speed_percentile99
    
    
    colors = plt.cm.bone_r(np.linspace(0,1,3))
    if args.verbose==True:
        ax.scatter(data2.Temperature,
            data_hull, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed \n before loom (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_w_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed \n before loom (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_wo_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
"""
#model_lm <- lm(log(speed+1) ~ temp + temp^2,my_data)
if args.a_string=='speed99_before_loom_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data_hull = data2.speed_percentile99
    
    
    colors = plt.cm.bone_r(np.linspace(0,1,3))
    if args.verbose==True:
        ax.scatter(data2.Temperature,
            data_hull, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_new_w_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_new_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
"""

#median speed during unperturbed swimming


#model_lm <- lm(log(speed+1) ~ temp ,my_data)
if args.a_string=='speed50_before_loom_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data_hull = data2.speed_percentile50
    
    
    colors = plt.cm.bone_r(np.linspace(0,1,3))
    if args.verbose==True:
        ax.scatter(data2.Temperature,
            data_hull, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Median speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile50_new_w_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Median speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile50_new_wo_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()


#average speed during unperturbed swimming


#model_lm <- lm(log(speed+1) ~ temp ,my_data)
if args.a_string=='speed_avg_before_loom_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data_hull = data2.avg_speed
    
    
    colors = plt.cm.bone_r(np.linspace(0,1,3))
    if args.verbose==True:
        ax.scatter(data2.Temperature,
            data_hull, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Mean speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_avg_new_w_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Mean speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_avg_new_wo_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()



## loom speed predictions
if args.a_string=='loom_speed_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    data_hull = data2.speed_percentile99
    
    
    if args.verbose==True:
        gs = [1,2,4,8,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2.speed_percentile99[data2.Groupsize == i][data2.Loom == 1], alpha = 0.5, color = colors[count], s =10)

            ax.plot(
                data1.Temperature[data1.Groupsize == i][data1.loom == 1], 
                (data1.loom_speed[data1.Groupsize==i][data1.loom == 1])**2, color = colors[count], label = str(i), lw = lw)
            ax.fill_between(
                data1.Temperature[data1.Groupsize == i][data1.loom == 1], 
                (data1.loom_speed025[data1.Groupsize==i][data1.loom == 1])**2,  
                (data1.loom_speed975[data1.Groupsize==i][data1.loom == 1])**2, alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('99th percentile of speed \n during loom (BL/s)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions_w_data_all.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()
    else:

        
        gs = [16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:
            ax.plot(
                data1.Temperature[data1.Groupsize == i][data1.loom == 1], 
                (data1.loom_speed[data1.Groupsize==i][data1.loom == 1])**2, color = colors[count], lw = lw)
            ax.fill_between(
                data1.Temperature[data1.Groupsize == i][data1.loom == 1], 
                (data1.loom_speed025[data1.Groupsize==i][data1.loom == 1])**2,  
                (data1.loom_speed975[data1.Groupsize==i][data1.loom == 1])**2, alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed \n during loom (BL/s)', size = fs)
        #ax.set_title('Groupsize = 16, Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions_wo_data.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()


## 99th percentile of loom speed predictions - 2 (after including t1)
# model_lm <- lm((speed)^0.5 ~ temp + I(temp^2) + log(gs,2) + loom + I(log(gs,2)^2) + t1,my_data)
# rsq 0.1872
if args.a_string=='99th_percentile_speed_during_loom_with_t1_predictions2.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    data_hull = data2.speed_percentile99
    
    
    if args.verbose==True:
        gs = [1,2,4,8,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2.speed_percentile99[data2.Groupsize == i][data2.Loom == 1], alpha = 0.5, color = colors[count], s =10)

            ax.plot(
                data1.temp[data1.gs == i][data1.loom == 1][data1.t1 == 1620244800], 
                (data1.speed99[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2, color = colors[count], label = str(i), lw = lw)
            ax.fill_between(
                data1.temp[data1.gs == i][data1.loom == 1][data1.t1 == 1620244800], 
                (data1.speed99_025[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2,  
                (data1.speed99_975[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2, alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('99th percentile of speed \n during loom (BL/s)', size = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        ax.set_title('Loom = 1', fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions2_w_data_all.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        
        gs = [16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
            ax.plot(
                data1.temp[data1.gs == i][data1.loom == 1][data1.t1 == 1620244800], 
                (data1.speed99[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2, color = colors[count], label = str(i), lw = lw)
            ax.fill_between(
                data1.temp[data1.gs == i][data1.loom == 1][data1.t1 == 1620244800], 
                (data1.speed99_025[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2,  
                (data1.speed99_975[data1.gs==i][data1.loom == 1][data1.t1 == 1620244800])**2, alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed \n during loom (BL/s)', size = fs)
        ax.set_title('Groupsize = 16, Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/loom_speed_predictions2_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


#model_glm <-  glm(prop_startles ~ temp + I(temp^2) + loom + t+date, family = binomial,my_data)
if args.a_string=='prop_startles_predictions.csv':
    
    if args.verbose==True:
        
        
        colors = plt.cm.bone_r(np.linspace(0,1,3))
        ax.scatter(data2.Temperature,
            data2.prop_startles, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200], 
            (data1.prop_startles[data1.loom == 1][data1.date == 18106][data1.t == 1200]), 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200], 
            (data1.prop_startles025[data1.loom == 1][data1.date == 18106][data1.t == 1200]),  
            (data1.prop_startles975[data1.loom == 1][data1.date == 18106][data1.t == 1200]), alpha = 0.3, 
            color = colors[count], lw = 0)
            
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Proportion of individuals that startle', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_w_data_loom_1_all.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        colors = plt.cm.bone_r(np.linspace(0,1,3))
        
        ax.plot(
            data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200], 
            (data1.prop_startles[data1.loom == 1][data1.date == 18106][data1.t == 1200]), 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200], 
            (data1.prop_startles025[data1.loom == 1][data1.date == 18106][data1.t == 1200]),  
            (data1.prop_startles975[data1.loom == 1][data1.date == 18106][data1.t == 1200]), alpha = 0.3, 
            color = colors[count], lw = 0)
            
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Proportion of individuals that startle', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29])
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_wo_data_loom_1_all.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()


#model_glm <-  glm(prop_startles ~ temp + I(temp^2) + loom +date, family = binomial,my_data)
if args.a_string=='prop_startles_predictions2.csv':
    
    if args.verbose==True:
        
        
        colors = plt.cm.bone_r(np.linspace(0,1,3))
        ax.scatter(data2.Temperature,
            data2.prop_startles, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp[data1.loom == 1][data1.date == 18106], 
            (data1.prop_startles[data1.loom == 1][data1.date == 18106]), 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp[data1.loom == 1][data1.date == 18106], 
            (data1.prop_startles025[data1.loom == 1][data1.date == 18106]),  
            (data1.prop_startles975[data1.loom == 1][data1.date == 18106]), alpha = 0.3, 
            color = colors[count], lw = 0)
            
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Proportion of individuals that startle', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29],fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_w_data_loom_1_all_new.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        colors = plt.cm.bone_r(np.linspace(0,1,3))
        
        ax.plot(
            data1.temp[data1.loom == 1][data1.date == 18106], 
            (data1.prop_startles[data1.loom == 1][data1.date == 18106]), 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp[data1.loom == 1][data1.date == 18106], 
            (data1.prop_startles025[data1.loom == 1][data1.date == 18106]),  
            (data1.prop_startles975[data1.loom == 1][data1.date == 18106]), alpha = 0.3, 
            color = colors[count], lw = 0)
            
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Proportion of individuals that startle', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29],fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/prop_startles_wo_data_loom_1_all_new.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()

## loom acceleration

#model_lm <- lm(log(acc+1) ~ temp + I(temp^2)*log(gs,2)*loom + date + t,my_data)
if args.a_string=='loom_acc_99_predictions.csv':
    
    if args.verbose==True:
        
        gs = [1,2,4,8,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
        for i in gs:

            ax.scatter(data2.Temperature[data2.Groupsize==i],
                data2.acc_percentile99[data2.Groupsize ==i], s = 10, alpha = 0.5, 
                color = colors[count])
            
            ax.plot(
                data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i], 
                np.exp(data1.acc99[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1, 
                color = colors[count], lw = lw, label = str(i))

            ax.fill_between(
                data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i], 
                np.exp(data1.acc99_025[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1,  
                np.exp(data1.acc99_975[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1,
                alpha = 0.3, color = colors[count], lw = 0)
            count += 1
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n during loom (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29],fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_w_data_loom_1_all.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()
    else:

        gs = [16]
        colors = plt.cm.bone_r(np.linspace(0,1,3))
        
        for i in gs:

            
            
            ax.plot(
                data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i], 
                np.exp(data1.acc99[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1, 
                color = colors[count], lw = lw, label = str(i))

            ax.fill_between(
                data1.temp[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i], 
                np.exp(data1.acc99_025[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1,  
                np.exp(data1.acc99_975[data1.loom == 1][data1.date == 18106][data1.t == 1200][data1.gs == i])-1,
                alpha = 0.3, color = colors[count], lw = 0)
            count += 1
            
            
            
                
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n during loom (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = 16, Loom = 1', fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29],fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_wo_data_loom_1.png'
        fig.savefig(out_dir, dpi = 100)
        plt.show()


#startle distance

#model_lm <- 
#lm(log(distance) ~ temp + gs + temp*gs + loom + I(temp^2)*gs + loom*I(temp^2) + loom*gs + date, my_data)
if args.a_string=='startle_distance_predictions2.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/'+args.a_string)

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_startle.csv')
    
    gs = [1,2,4,8,16]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #Plotting
    
    
    
    if args.verbose==True:
        gs = [1,2,4,8,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
            
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2.distance[data2.Groupsize == i][data2.Loom == 1]/60, s = 10, alpha = 0.5, color = colors[count])
        
            ax.plot(
                data1.temp[data1.gs== i][data1.loom == 1][data1.date == 18106], 
                np.exp(data1.distance[data1.gs==i][data1.loom == 1][data1.date == 18106])/60, 
                color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.gs== i][data1.loom == 1][data1.date == 18106], 
                np.exp(data1.distance_025[data1.gs==i][data1.loom == 1][data1.date == 18106])/60,  
                np.exp(data1.distance_975[data1.gs==i][data1.loom == 1][data1.date == 18106])/60, 
                label = str(i), alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Distance (BL)', size = fs)
        ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/distance_w_data_loom1.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:
        gs = [16]
        loom = [1,5]
        colors = plt.cm.bone_r(np.linspace(0,1,len(loom)+1))
        count = 1
        for i in gs:
            for j in loom:
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.distance[data1.gs==i][data1.loom == j][data1.date == 18106])/60, color = colors[count],
                    lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.distance_025[data1.gs==i][data1.loom == j][data1.date == 18106])/60,  
                    np.exp(data1.distance_975[data1.gs==i][data1.loom == j][data1.date == 18106])/60, 
                    alpha = 0.3, label = str(j), color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Distance (BL)', size = fs)
        ax.set_title('Groupsize = 16', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/distance_wo_data_gs16.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


### pre loom acceleration 

#model_lm <- lm(log(acc+1) ~ temp + log(gs,2) + I(log(gs,2)^2),my_new_data)
# r sq 0.1936
if args.a_string=='acc_99_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data2 = data2.drop(labels = 127)
    data_hull = data2.acc_percentile99
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
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
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n before loom (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_w_data.png'
        fig.savefig(out_dir, dpi = 300)
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
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n before loom (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()

"""

#model_lm <- lm(log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2),my_new_data)
# r sq 0.1962
if args.a_string=='acc_99_predictions_new_squared.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data2 = data2.drop(labels = 127)
    data_hull = data2.acc_percentile99
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
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
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_w_data_new_squared.png'
        fig.savefig(out_dir, dpi = 300)
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
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_wo_data_new_squared.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


"""

#startle distance corrected

#model_lm <- 
#log(distance) ~ temp + log(gs,2) + temp*log(gs,2) + loom*temp + I(temp^2)*log(gs,2) + loom*I(temp^2)

#+date + loom*log(gs,2), my_data
if args.a_string=='startle_distance_predictions3.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/'+args.a_string)

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_startle_corrected.csv')
    
    gs = [1,2,4,8,16]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+2))
    #Plotting
    
    
    
    if args.verbose==True:
        gs = [1,2,4,8,16]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
            
            ax.scatter(data2.Temperature[data2.Groupsize == i][data2.Loom == 1],
                data2.distance[data2.Groupsize == i][data2.Loom == 1]/60, s = 10, alpha = 0.5, color = colors[count])
        
            ax.plot(
                data1.temp[data1.gs== i][data1.loom == 1][data1.date == 18106], 
                np.exp(data1.distance[data1.gs==i][data1.loom == 1][data1.date == 18106])/60, 
                color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.gs== i][data1.loom == 1][data1.date == 18106], 
                np.exp(data1.distance_025[data1.gs==i][data1.loom == 1][data1.date == 18106])/60,  
                np.exp(data1.distance_975[data1.gs==i][data1.loom == 1][data1.date == 18106])/60, 
                label = str(i), alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Distance (BL)', size = fs)
        ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/distance_w_data_loom1.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:
        gs = [1,16]
        loom = [1]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
            for j in loom:
                ax.plot(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.distance[data1.gs==i][data1.loom == j][data1.date == 18106])/60, color = colors[count],
                    lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i][data1.loom == j][data1.date == 18106], 
                    np.exp(data1.distance_025[data1.gs==i][data1.loom == j][data1.date == 18106])/60,  
                    np.exp(data1.distance_975[data1.gs==i][data1.loom == j][data1.date == 18106])/60, 
                    alpha = 0.3, label = str(i), color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Distance (BL)', size = fs)
        ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/distance_wo_data_loom1.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()

### pre loom annd

#model_lm <- lm(log(annd) ~ temp + log(gs,2), my_data)
#r sq = 0.76 #residuals not good
if args.a_string=='annd_before_loom_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    #data2 = data2.drop(labels = 127)
    data_hull = data2.annd
    gs = [2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.annd[data1.gs ==i]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.annd025[data1.gs ==i]),  
                np.exp(data1.annd975[data1.gs ==i]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Average Nearest Neighbor Distance \n before loom (BL)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_before_loom_w_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        
        gs = [2,16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.annd[data1.gs ==i]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.annd025[data1.gs ==i]),  
                np.exp(data1.annd975[data1.gs ==i]), alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Average Nearest Neighbor Distance \n before loom (BL)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/annd_before_loom_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


#local polarization
#lm(pol ~ temp  + loom + t1, my_data)
#r sq 0.03

if args.a_string=='pol1_during_loom_predictions.csv':
    data1 = pd.read_csv('../../data/temp_collective/roi/'+args.a_string)

    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom_pol.csv')
    
    loom = [1,2,3,4,5]
    colors = plt.cm.bone_r(np.linspace(0,1,len(loom)+1))
    count = 1
    #Plotting
    
    
    
    if args.verbose==True:
        loom = [1,2,3,4,5]
        colors = plt.cm.bone_r(np.linspace(0,1,len(loom)+1))
        count = 1
        for i in loom:
            
            ax.scatter(data2.Temperature[data2.Loom == i],
                data2.polarization_1[data2.Loom == 1], s = 10, alpha = 0.5, color = colors[count])
        
            ax.plot(
                data1.temp[data1.loom == i][data1.t1 == 1618600800], 
                data1.pol_1[data1.loom == i][data1.t1 == 1618600800], 
                color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.loom == i][data1.t1== 1618600800], 
                data1.pol1_025[data1.loom == i][data1.t1 == 1618600800],  
                data1.pol1_975[data1.loom == i][data1.t1 == 1618600800], 
                label = str(i), alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Local Polarization', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/local_pol1_w_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:
        loom = [1,2,3,4,5]
        colors = plt.cm.bone_r(np.linspace(0,1,len(loom)+1))
        count = 1
        for i in loom:
            
            
        
            ax.plot(
                data1.temp[data1.loom == i][data1.t1 == 1618600800], 
                data1.pol_1[data1.loom == i][data1.t1 == 1618600800], 
                color = colors[count],
                lw = lw)

            ax.fill_between(
                data1.temp[data1.loom == i][data1.t1== 1618600800], 
                data1.pol1_025[data1.loom == i][data1.t1 == 1618600800],  
                data1.pol1_975[data1.loom == i][data1.t1 == 1618600800], 
                label = str(i), alpha = 0.3, color = colors[count], lw = 0)
            count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel('Local Polarization', size = fs)
        #ax.set_title('Loom = 1', fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/local_pol1_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


#hull after loom
#model_lm <- lm(hull ~ gs + temp , my_data)
if args.a_string=='hull_after_loom_predictions_700_900.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')
    data_hull = data2.convex_hull_area
    gs = [4,8,16]
    loom = [1]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    count = 1
    if args.verbose==True:
        
        for i in gs:
            for j in loom:
                ax.scatter(data2.Temperature[data2.Groupsize == i],
                    data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                    color = colors[count])
                
                ax.plot(
                    data1.temp[data1.gs== i], 
                    (data1.hull[data1.gs==i]), 
                    color = colors[count], lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i], 
                    (data1.hull025[data1.gs==i]),  
                    (data1.hull975[data1.gs==i]), alpha = 0.3, label = str(i), 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Convex hull area after loom', size = fs)
        #ax.set_title('Loom = '+str(loom[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_after_loom_w_data_gs_all.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:
        gs = [16]
        loom = [1]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
        
            for j in loom:
                
                ax.plot(
                    data1.temp[data1.gs== i], 
                    (data1.hull[data1.gs==i]), 
                    color = colors[count], lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i], 
                    (data1.hull025[data1.gs==i]),  
                    (data1.hull975[data1.gs==i]), alpha = 0.3, 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Convex hull area after loom', size = fs)
        #ax.set_title('Loom = '+str(loom[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_after_loom_wo_data_gs_16.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()

#hull during loom
#model_lm <- lm(hull ~ gs + temp , my_data)
if args.a_string=='hull_during_loom_predictions_500_700.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/convex_hull_during_loom.csv')
    data_hull = data2.convex_hull_area_500_700
    gs = [4,8,16]
    loom = [1]
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    count = 1
    if args.verbose==True:
        
        for i in gs:
            for j in loom:
                ax.scatter(data2.Temperature[data2.Groupsize == i],
                    data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                    color = colors[count])
                
                ax.plot(
                    data1.temp[data1.gs== i], 
                    (data1.hull[data1.gs==i]), 
                    color = colors[count], lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i], 
                    (data1.hull025[data1.gs==i]),  
                    (data1.hull975[data1.gs==i]), alpha = 0.3, label = str(i), 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Convex hull area during loom', size = fs)
        #ax.set_title('Loom = '+str(loom[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_during_loom_w_data_gs_all.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:
        gs = [16]
        loom = [1]
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        count = 1
        for i in gs:
        
            for j in loom:
                
                ax.plot(
                    data1.temp[data1.gs== i], 
                    (data1.hull[data1.gs==i]), 
                    color = colors[count], lw = lw)

                ax.fill_between(
                    data1.temp[data1.gs== i], 
                    (data1.hull025[data1.gs==i]),  
                    (data1.hull975[data1.gs==i]), alpha = 0.3, label = str(i), 
                    color = colors[count], lw = 0)
                count += 1
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Convex hull area during loom', size = fs)
        #ax.set_title('Loom = '+str(loom[0]), fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/convex_hull_during_loom_wo_data_gs_16.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()





### pre loom avg acceleration

#log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2)
# r sq 0.2786
if args.a_string=='acc_avg_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    #data2 = data2.drop(labels = 127)
    data_hull = data2.avg_acc
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc_025[data1.gs ==i])-1,  
                np.exp(data1.acc_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Average acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_avg_acc_w_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc_025[data1.gs ==i])-1,  
                np.exp(data1.acc_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Average acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_avg_acc_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()

### pre loom median acceleration

#log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2)
# r sq 0.301
if args.a_string=='acc_50_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    #data2 = data2.drop(labels = 127)
    data_hull = data2.acc_percentile50
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc50[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc50_025[data1.gs ==i])-1,  
                np.exp(data1.acc50_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Median acceleration (BL/s'+r'$^2$)', size = fs)

        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_50_acc_w_data.png'
        fig.savefig(out_dir, dpi = 300)
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
                np.exp(data1.acc50_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Median acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_50_acc_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


### pre loom avg acceleration

#log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2)
# r sq 0.2786
if args.a_string=='acc_avg_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    #data2 = data2.drop(labels = 127)
    data_hull = data2.avg_acc
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc_025[data1.gs ==i])-1,  
                np.exp(data1.acc_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0,label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Mean acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_avg_acc_w_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()
    else:

        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc_025[data1.gs ==i])-1,  
                np.exp(data1.acc_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Mean acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pre_loom_avg_acc_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()

### pca coeff - bernoulli

# model_glm <- glm(pca ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2) + date, family = binomial, data = my_data)
# summary(model_glm)
#aic = 834
if args.a_string=='pca_predictions.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/pca_coeff_bernoulli.csv')
    #data2 = data2.drop(labels = 127)
    data_hull = data2.pca_coeff
    gs = [4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==False:
        
        gs = [16]
        count = 1
        colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i][data1.date == 18106], 
                (data1.pca[data1.gs ==i][data1.date == 18106]), 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i][data1.date == 18106], 
                (data1.pca_025[data1.gs ==i][data1.date == 18106]),  
                (data1.pca_975[data1.gs ==i][data1.date == 18106]), alpha = 0.3, 
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            'Probability for pca coeff \n to be >= 0', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/pca_bernoulli_wo_data.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()








# acc before each loom

#model_lm <- lm(log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2),my_new_data)
# r sq 0.1962
if args.a_string=='acc_before_loom_99_predictions_new_squared.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data2 = data2.drop(labels = 127)
    data_hull = data2.acc_percentile99
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
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
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n before loom (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_before_loom_percentile99_w_data_new_squared.png'
        fig.savefig(out_dir, dpi = 300)
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
                color = colors[count], lw = 0, label = str(i))
            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration \n before loom (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_before_loom_percentile99_wo_data_new_squared.png'
        fig.savefig(out_dir, dpi = 300)
        plt.show()


#figure with both unperturbed swimming speed predictions - linear and quadratic

#model_lm <- lm(log(speed+1) ~ temp + temp^2,my_data)
if args.a_string=='speed99_before_loom_predictions_new.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data_hull = data2.speed_percentile99
    data3 = pd.read_csv('../../data/temp_collective/roi/speed99_before_loom_predictions.csv')
    
    colors = plt.cm.bone_r(np.linspace(0,1,3))
    if args.verbose==True:
        ax.scatter(data2.Temperature,
            data_hull, s = 10, alpha = 0.5, 
            color = colors[count])
        
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0)
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_new_w_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        
        colors = plt.cm.bone_r(np.linspace(0,1,3))
        ax.plot(
            data1.temp, 
            np.exp(data1.speed99)-1, 
            color = colors[count], lw = lw)

        ax.plot(
            data3.temp, 
            np.exp(data3.speed99)-1, 
            color = colors[count-1], lw = lw)

        ax.fill_between(
            data1.temp, 
            np.exp(data1.speed99_025)-1,  
            np.exp(data1.speed99_975)-1, alpha = 0.3, 
            color = colors[count], lw = 0, label = 'quadratic')

        ax.fill_between(
            data3.temp, 
            np.exp(data3.speed99_025)-1,  
            np.exp(data3.speed99_975)-1, alpha = 0.3, 
            color = colors[count - 1], lw = 0, label = 'linear')
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of speed (BL/s)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        legend = plt.legend(fontsize=fs, loc='lower right', title = 'Model', framealpha = 0.5)
        plt.setp(legend.get_title(),fontsize='xx-large')
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        out_dir = '../../output/temp_collective/roi_figures/predictions/speed_percentile99_together_wo_data.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()


#model_lm <- lm(log(acc+1) ~ temp + I(temp^2) + log(gs,2) + I(log(gs,2)^2),my_new_data)
# r sq 0.1962
if args.a_string=='acc_99_predictions_new_squared.csv':
    data2 = pd.read_csv('../../data/temp_collective/roi/all_params_wo_loom.csv')
    data2 = data2.drop(labels = 127)
    data_hull = data2.acc_percentile99
    data3 = pd.read_csv('../../data/temp_collective/roi/acc_99_predictions.csv')
    gs = [1,2,4,8,16]
    count = 1
    colors = plt.cm.bone_r(np.linspace(0,1,len(gs)+1))
    if args.verbose==True:
        for i in gs:
            ax.scatter(data2.Temperature[data2.Groupsize == i],
                data_hull[data2.Groupsize == i], s = 10, alpha = 0.5, 
                color = colors[count])
            
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
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize=fs, loc='lower right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_w_data_new_squared.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
    else:

        
        gs = [16]
        count = 2
        colors = plt.cm.bone_r(np.linspace(0,1,3))
        for i in gs:
            
            
            ax.plot(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc99[data1.gs ==i])-1, 
                color = colors[count], lw = lw)

            ax.fill_between(
                data1.temp[data1.gs ==i], 
                np.exp(data1.acc99_025[data1.gs ==i])-1,  
                np.exp(data1.acc99_975[data1.gs ==i])-1, alpha = 0.3, 
                color = colors[count], lw = 0, label = 'quadratic')
            ax.plot(
                data3.temp[data1.gs ==i], 
                np.exp(data3.acc99[data3.gs ==i])-1, 
                color = colors[count-1], lw = lw)

            ax.fill_between(
                data3.temp[data1.gs ==i], 
                np.exp(data3.acc99_025[data3.gs ==i])-1,  
                np.exp(data3.acc99_975[data3.gs ==i])-1, alpha = 0.3, 
                color = colors[count-1], lw = 0, label = 'linear')

            count +=1
        
        plt.xlabel('Temperature '+r'($^{\circ}$C)', size = fs)
        plt.ylabel(
            '99th percentile of acceleration (BL/s'+r'$^2$)', size = fs)
        #ax.set_title('Groupsize = '+str(gs[0]), fontsize = fs)
        #plt.legend(fontsize=fs, loc='upper right', title = 'Loom', framealpha = 0.5)
        #ax.set_title('Interaction of temperature and groupsize', fontsize = fs)
        #ax.set_title('Groupsize = 16', fontsize = fs)
        plt.xticks(ticks = [9,13,17,21,25,29], labels = [9,13,17,21,25,29], fontsize = fs)
        plt.yticks(fontsize = fs)
        legend = plt.legend(fontsize=fs, loc='lower right', title = 'Model', framealpha = 0.5)
        plt.setp(legend.get_title(),fontsize='xx-large')
        #plt.legend(fontsize=fs, loc='upper right', title = 'Groupsize', framealpha = 0.5)
        out_dir = '../../output/temp_collective/roi_figures/predictions/acc_percentile99_wo_data_together.png'
        fig.savefig(out_dir, dpi = dpi)
        plt.show()
