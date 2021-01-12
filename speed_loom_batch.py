"""
Goal - To caluclate speed of all the treatments - before, during and after the loom
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
from scipy.spatial import ConvexHull
import pandas as pd

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

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

temperature = range(9,30,4)

group = [1,2,4,8,16,32]

replication = range(10) # number of replicates per treatment


speed_loom = np.empty([239*5,2004])
speed_loom.fill(np.nan)

count = 0

#output parent directory
parent_dir = '../../data/temp_collective/roi'

ii = 0 # to keep count of temperature

#frames = 5000 #number of frames for which annd is calculated


for i in temperature:
    jj = 0 # to keep count of groups
    for j in group:
        out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/'

        for k in replication:
            print(i,j,k+1)
            if j == 1:
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:   
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                
            

            
            
            try:
                tr = tt.Trajectories.from_idtrackerai(input_file,                  center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')        
            
            except FileNotFoundError:
                print(i,j,k+1)
                print('File not found')
                continue
             
            
            looms = []
            
            for m in range(len(met.Temperature)):
                if met.Temperature[m] == i and met.Groupsize[m] == j and met.Replicate[m] == (k+1): 
                    looms.append(met['Loom 1'][m]) 
                    looms.append(met['Loom 2'][m]) 
                    looms.append(met['Loom 3'][m]) 
                    looms.append(met['Loom 4'][m]) 
                    looms.append(met['Loom 5'][m])            
            
            
            frame_list = np.array([list(range(looms[0]-500, looms[0]+1500)), list(range(looms[1]-500, looms[1]+1500)) , list(range(looms[2]-500, looms[2]+1500)) ,list(range(looms[3]-500, looms[3]+1500)) ,list(range(looms[4]-500, looms[4]+1500))]) 
            for loom_number in range(5):

                speed_loom[count, 4:2004] = np.nanmean(filter_speed_low_pass(tr)[frame_list[loom_number],:], axis = 1)
                speed_loom[count, 0] = i
                speed_loom[count, 1] = j
                speed_loom[count, 2] = k+1
                speed_loom[count, 3] = loom_number + 1
                count += 1


        
        
        jj= jj + 1
        
    ii = ii + 1

out_dir = '../../output/temp_collective/roi/loom_speed.npy'
np.save(out_dir,speed_loom)

