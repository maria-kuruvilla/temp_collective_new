
import os
import pathlib
from pprint import pprint

import numpy as np
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import trajectorytools as tt
import trajectorytools.plot as ttplot
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import dir_of_data
import csv
import pickle
import argparse
import pandas as pd


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
parser.add_argument('-a', '--a_string', default='hi', type=str)
parser.add_argument('-b1', '--integer_b1', default=29, type=int)
parser.add_argument('-b2', '--integer_b2', default=16, type=int)
parser.add_argument('-b3', '--integer_b3', default=3, type=int)
parser.add_argument('-f1', '--integer_f1', default=0, type=int)
parser.add_argument('-f2', '--integer_f2', default=10000, type=int)
parser.add_argument('-c', '--float_c', default=1.5, type=float)
parser.add_argument('-v', '--verbose', default=True, type=boolean_string)
# Note that you assign a short name and a long name to each argument.
# You can use either when you call the program, but you have to use the
# long name when getting the values back from "args".

# get the arguments
args = parser.parse_args()

parent_dir = '../../output/temp_collective/roi'

input_dir = parent_dir + '/' + str(args.integer_b1) + '/' + str(args.integer_b2) + '/' 
input_file = input_dir + str(args.integer_b3) + '_nosmooth.p'
#sigma_values = 1.5 #smoothing parameter
if args.integer_b2 == 1:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories.npy'
else:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories_wo_gaps.npy'
try:
    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path)
    #tr.new_time_unit(tr.params['frame_rate'], 'seconds')
except FileNotFoundError:
    print(args.integer_b1,args.integer_b2,args.integer_b3)
    print('File not found')
    pass 


def speed_histogram(x): 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #speed_pdf, speed = np.histogram(x,bins=10,density=True)
    #plt.plot(speed,np.log(speed_pdf))
    ax.hist(x, density=True, bins=100, log = False)
    #ax.set_xscale('log')
    #ax.set_xlim(left = 5)
    #ax.set_ylim([0,0.0002])
    ax.set_xlabel('Speed')
    ax.set_ylabel('Probability')
    #plt.show()
    #plt.xticks(ticks = [10,20,100,300], labels = [10,20,100,300])
    
    out_dir = parent_dir = '../../output/temp_collective/trial_hist4.png'
    fig.savefig(out_dir, dpi = 300)
    return(ax)
    

df = pd.read_csv('../../data/temp_collective/colin_trial.csv',names=["Frame", "Individual", "x1", "y1","x2","y2"])

x = np.minimum(df.x1,df.x2)+ abs(df.x1 - df.x2)/2
y = np.minimum(df.y1,df.y2)+ abs(df.y1 - df.y2)/2

xx = pd.Series(x, name = 'x')
yy = pd.Series(y, name = 'y') 

#xxx = pd.DataFrame(data = [xx.values], columns = xx.index) 
#yyy = pd.DataFrame(data = [yy.values], columns = yy.index) 

#x = np.reshape(tr.speed,tr.speed.shape[0]*tr.speed.shape[1])   
data = pd.concat([df,xx, yy], axis=1) 
grouped = data.groupby('Individual') 
for group in grouped: 
    print(group)

speed_histogram(x)
speed = []

for i in range(len(data)):
    for j in range(len(data)):
        if data['Frame'][j] == data['Frame'][i] + 1:
            if data['Individual'][j] == data['Individual'][i]:
                speed.append(np.sqrt((data['x'][j] - data['x'][i])**2 + (data['y'][j] - data['y'][i])**2)) 
