"""
Goal - to enter the temp, group size and replicate and func should return plot with filtered speed as a function of time as well unfiltered speed as a function of time to evaluate the effectiveness of the filter.
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
fs = 16

def spikes_position(trajectory):
    list1 = []
    for j in range(trajectory.number_of_individuals):
        list2 = [i for i, value in enumerate(trajectory.speed[:,j]) if value > 10]
        list2.insert(0,100000000)
        list1 = list1 + [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def track_check(tr, temp, group, rep): #replicates start from 1
    frame_range = range(tr.s.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for focal in range(tr.number_of_individuals):
        ax.plot(np.asarray(frame_range),tr.speed[frame_range, focal])
    for j in range(5):
        plt.axvline(looms[j], color = 'k',alpha=0.3)   
    
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Speed (BL/s)')
    ax.set_title('Temp:' + str(temp) + ' Group:' + str(group) + ' Replicate:' + str(rep))
    return(ax)

def track_check_acc(tr, temp, group, rep): #replicates start from 1
    frame_range = range(tr.s.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for focal in range(tr.number_of_individuals):
        ax.plot(np.asarray(frame_range),tr.acceleration[frame_range, focal])
    for j in range(5):
        plt.axvline(looms[j], color = 'k',alpha=0.3)   
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Acceleration (BL/s^2)')
    ax.set_title('Temp:' + str(temp) + ' Group:' + str(group) + ' Replicate:' + str(rep))
    return(ax)
	


def track_check_position(tr, temp, group, rep): #replicates start from 1
    frame_range = range(tr.s.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(tr.number_of_individuals):
        ax.plot(tr.s[frame_range,i,0], tr.s[frame_range,i,1])
        
    
    ax.set_xlabel('X (BL)')
    ax.set_ylabel('Y (BL)')
    ax.set_title('Trajectories')
    return(ax)

def position(tr):
    return(tr.s)


def speed(tr):
    v = (position(tr)[2:] - position(tr)[:-2]) / 2
    b = np.linalg.norm(v, axis=-1)
    return(b*60)

def acceleration(tr):
    a = position(tr)[2:] - 2 * position(tr)[1:-1] + position(tr)[:-2]
    aa = np.linalg.norm(a, axis=-1)  
    return(aa*3600)
        

def filter(tr, roi = 5): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,0],copy=False)
    position_mask1 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,1],copy=False)
    return(position_mask0,position_mask1)  
    
def filter_speed(tr, roi = 5): 
    speed_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), speed(tr),copy=False)
    
    return(speed_mask)         



def filter_acc(tr, roi = 5): 
    acc_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)                                   

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



def track_check_position_masked(tr, temp, group, rep): #replicates start from 1
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(tr.number_of_individuals):
        
        ax.plot(filter_low_pass(tr)[0], filter_low_pass(tr)[1])
        
    
    ax.set_xlabel('X (BL)')
    ax.set_ylabel('Y (BL)')
    ax.set_title('Trajectories')
    return(ax)

def track_check_masked(tr, temp, group, rep): #replicates start from 1
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    frame_range = range(filter_speed(tr,5).shape[0])
    #frame_range = range(tr.s.shape[0])
    colors = plt.cm.viridis_r(np.linspace(0,1,tr.number_of_individuals))
    for i in range(tr.number_of_individuals):
        
        ax.plot(np.asarray(frame_range),filter_speed_low_pass(tr)[frame_range,i], color = colors[i])
        
    ax.set_xlim(0, filter_speed(tr,5).shape[0])
    for j in range(5):
        frames = [looms[j]]#range(looms[j],looms[j]+600,50)
        frames = list(frames) + [looms[j]+599]
        for k in frames:
            #plt.axvline(looms[j], color = 'k',alpha=0.3)    
            ax.scatter(k, 29, s = 500/(5000-(50/6)*(k-looms[j])), color = 'black')
    plt.annotate(s='', xy=(looms[0],29), xytext=(0,29), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(s='Pre-loom', xy=(4000,27), fontsize = fs)
    ax.set_xlabel('Time (s)', fontsize = fs)
    ax.set_ylabel('Speed (BL/s)', fontsize = fs)
    plt.xticks(ticks = [0,30000,60000,90000], labels = [0,500,1000,1500],fontsize = fs)
    plt.yticks([0,10,20,30], labels = [0,10,20,30],fontsize = fs)
    #ax.set_title('C')
    out_dir = '../../output/temp_collective/roi_figures/schematic_figure_2.png'
    fig.savefig(out_dir, dpi = 300, bbox_inches="tight")
    return(ax)


def track_check_acc_masked(tr, temp, group, rep): #replicates start from 1
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    frame_range = range(filter_acc(tr,5).shape[0])
    for i in range(tr.number_of_individuals):
        
        ax.plot(np.asarray(frame_range),filter_acc_low_pass(tr)[frame_range,i])
        
    ax.set_xlim(0, filter_speed(tr,5).shape[0])
    for j in range(5):
        plt.axvline(looms[j], color = 'k',alpha=0.3)    
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Acceleration (BL/s^2)')
    ax.set_title('Temp:' + str(temp) + ' Group:' + str(group) + ' Replicate:' + str(rep))
    return(ax)

    



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
met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')
for i in range(len(met.Temperature)):
    if met.Temperature[i] == args.integer_b1 and met.Groupsize[i] == args.integer_b2 and met.Replicate[i] == args.integer_b3 : 
        looms.append(met['Loom 1'][i]) 
        looms.append(met['Loom 2'][i]) 
        looms.append(met['Loom 3'][i]) 
        looms.append(met['Loom 4'][i]) 
        looms.append(met['Loom 5'][i]) 

#track_check(tr, args.integer_b1, args.integer_b2, args.integer_b3)
#track_check_acc(tr,args.integer_b1,args.integer_b2,args.integer_b3)
track_check_masked(tr, args.integer_b1, args.integer_b2, args.integer_b3)
#track_check_acc_masked(tr,args.integer_b1,args.integer_b2,args.integer_b3)
#track_check_own(tr, args.integer_b1, args.integer_b2, args.integer_b3)
#track_check_acc_own(tr,args.integer_b1,args.integer_b2,args.integer_b3)
#track_check_position_masked(tr,args.integer_b1,args.integer_b2,args.integer_b3)
#print(spikes_position(tr))
plt.show()

