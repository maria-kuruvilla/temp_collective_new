"""
Mon Jan 25th
Goal - To have prelim code for individual based speed ratio - speed during loom:speed before loom
"""
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

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

#output parent directory
parent_dir = '../../data/temp_collective/roi'

def position(tr): #shape returns tr.s.shape
    return(tr.s)

def speed(tr): #speed(tr).shape returns tr.speed.shape - 2
    v = (position(tr)[2:] - position(tr)[:-2]) / 2
    b = np.linalg.norm(v, axis=-1)
    return(b*60)

def acceleration(tr): #shape returns tr.acceleration.shape - 2
    a = position(tr)[2:] - 2 * position(tr)[1:-1] + position(tr)[:-2]
    aa = np.linalg.norm(a, axis=-1)  
    return(aa*3600)
        
def e(tr): #e.shape returns tr.speed.shape - 2
    vel = (position(tr)[2:] - position(tr)[:-2]) / 2
    n = np.linalg.norm(v,axis = 2)  
    return(vel/n[...,np.newaxis])

def filter_low_pass(tr, roi1 = 30, roi2 = 3340): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,0],copy=False)
    position_mask1 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,1],copy=False)
    return(position_mask0,position_mask1)                                 

def filter_speed_low_pass(tr, roi = 30): 
    speed_mask = np.ma.masked_where((speed(tr) > roi), speed(tr),copy=False)
    
    return(speed_mask)         

def filter_acc_low_pass(tr, roi = 3340): 
    acc_mask = np.ma.masked_where((acceleration(tr) > roi), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)  

def spikes_position_new(tr,j): #uses filter_speed
    list1 = []
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr)[:,j]) if value > 10]
    list2.insert(0,100000000)
    list1 = list1 + [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def accurate_startles(tr, loom, ind): #uses filtered speed
    list1 = spikes_position_new(tr, ind)
    

    
    list2 = [i for i, value in enumerate(list1[:]) if value < (loom[0] + 700) and value > (loom[0]+500) ]
    list2 = list2 + [i for i, value in enumerate(list1[:]) if value < (loom[1] + 700) and value > (loom[1]+500) ]
    list2 = list2 + [i for i, value in enumerate(list1[:]) if value < (loom[2] + 700) and value > (loom[2]+500) ]
    list2 = list2 + [i for i, value in enumerate(list1[:]) if value < (loom[3] + 700) and value > (loom[3]+500) ]
    list2 = list2 + [i for i, value in enumerate(list1[:]) if value < (loom[4] + 700) and value > (loom[4]+500) ]
    
    return(len(list2))

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
    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
except FileNotFoundError:
    print(args.integer_b1,args.integer_b2,args.integer_b3)
    print('File not found')
    pass

looms = []

for m in range(len(met.Temperature)):
    if met.Temperature[m] == args.integer_b1 and met.Groupsize[m] == args.integer_b2 and met.Replicate[m] == args.integer_b3: 
        looms.append(met['Loom 1'][m]) 
        looms.append(met['Loom 2'][m]) 
        looms.append(met['Loom 3'][m]) 
        looms.append(met['Loom 4'][m]) 
        looms.append(met['Loom 5'][m])

frame_list = np.r_[(looms[0]+500):(looms[0]+700),(looms[1]+500):(looms[1]+700),(looms[2]+500):(looms[2]+700),(looms[3]+500):(looms[3]+700),(looms[4]+500):(looms[4]+700)]
max_speed  = np.empty([tr.number_of_individuals, 1])
max_speed.fill(np.nan)
for ind in range(tr.number_of_individuals):
	if accurate_startles(tr,looms, ind) != 0:
		max_speed[ind,0] = np.nanmax(filter_speed_low_pass(tr)[frame_list,ind])

print(np.nanmean(max_speed))

