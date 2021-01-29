"""
Tue Jan 26th

Goal - code to calculate total number of frames in each video to see max and min and also to calculate max number of frames masked
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

temperature = range(9,30,4)

group = [1,2,4,8,16,32]

replication = range(11) # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'

frames = []
frames_masked = []
temp_frames = []
gs_frames = []
rep_frames = []
ii = 0
for i in temperature:
    jj = 0
    for j in group:
        replicate_loom_startles = np.empty([len(replication), 1])
        replicate_loom_startles.fill(np.nan)
        for k in replication:
            if j == 1:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
            try:
                tr = tt.Trajectories.from_idtrackerai(trajectories_file_path,                  center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')
            except FileNotFoundError:
                print(i,j,k)
                print('File not found')
                continue
            frames.append(tr.speed.shape[0])
            frames_masked.append(tr.speed.shape[0]-filter_speed_low_pass(tr)[:,:].compressed().shape[0]/tr.number_of_individuals)
            temp_frames.append(i)
            gs_frames.append(j)
            rep_frames.append(k+1)

min_frames = np.min(frames)
max_frames = np.max(frames)
max_masked = np.max(frames_masked)

print(min_frames,max_frames,max_masked)
