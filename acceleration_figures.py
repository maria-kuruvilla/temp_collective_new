

"""
Goal - to fix the colors on the acceleration data
Date - Jul 8 2021
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
fs=30
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)


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
