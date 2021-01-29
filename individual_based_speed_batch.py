"""
Mon Jan 25th
Goal - To calculate individual based speed ratio - speed during loom for all individuals that do startle - for all temps & gs
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

temperature = range(9,30,4)

group = [1,2,4,8,16]

replication = range(11) # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'

ind_startle_speed = np.empty([len(temperature), len(group)])
ind_startle_speed.fill(np.nan)

std_ind_startle_speed = np.empty([len(temperature), len(group)])
std_ind_startle_speed.fill(np.nan)

ind_median_speed = np.empty([len(temperature), len(group)])
ind_median_speed.fill(np.nan)

std_ind_median_speed = np.empty([len(temperature), len(group)])
std_ind_median_speed.fill(np.nan)

ind_ratio_speed = np.empty([len(temperature), len(group)])
ind_ratio_speed.fill(np.nan)

std_ind_ratio_speed = np.empty([len(temperature), len(group)])
std_ind_ratio_speed.fill(np.nan)

ii = 0
for i in temperature:
    jj = 0
    for j in group:
        replicate_ind_startle_speed = np.empty([len(replication), 1])
        replicate_ind_startle_speed.fill(np.nan)
        replicate_ind_median_speed = np.empty([len(replication), 1])
        replicate_ind_median_speed.fill(np.nan)
        replicate_ind_ratio_speed = np.empty([len(replication), 1])
        replicate_ind_ratio_speed.fill(np.nan)
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
            looms = []
        
            for m in range(len(met.Temperature)):
                if met.Temperature[m] == i and met.Groupsize[m] == j and met.Replicate[m] == (k+1): 
                    looms.append(met['Loom 1'][m]) 
                    looms.append(met['Loom 2'][m]) 
                    looms.append(met['Loom 3'][m]) 
                    looms.append(met['Loom 4'][m]) 
                    looms.append(met['Loom 5'][m]) 
            frame_list = np.r_[looms[0]+500:looms[0]+700,looms[1]+500:looms[1]+700,looms[2]+500:looms[2]+700,looms[3]+500:looms[3]+700,looms[4]+500:looms[4]+700]           
            max_speed  = np.empty([tr.number_of_individuals, 1])
            max_speed.fill(np.nan)
            median_speed  = np.empty([tr.number_of_individuals, 1])
            median_speed.fill(np.nan)
            ratio_speed  = np.empty([tr.number_of_individuals, 1])
            ratio_speed.fill(np.nan)
            for ind in range(tr.number_of_individuals):
                if accurate_startles(tr,looms, ind) != 0:
                    max_speed[ind,0] = np.nanmax(filter_speed_low_pass(tr)[frame_list,ind])
                    median_speed[ind,0]=np.percentile(filter_speed_low_pass(tr)[0:looms[0],ind],50)
                    ratio_speed[ind,0]=max_speed[ind,0]/median_speed[ind,0]
            replicate_ind_startle_speed[k] = np.nanmean(max_speed)
            replicate_ind_median_speed[k] = np.nanmean(median_speed)
            replicate_ind_ratio_speed[k] = np.nanmean(ratio_speed)
        ind_startle_speed[ii,jj] = np.nanmean(replicate_ind_startle_speed)
        std_ind_startle_speed[ii,jj] = np.nanstd(replicate_ind_startle_speed)
        ind_median_speed[ii,jj] = np.nanmean(replicate_ind_median_speed)
        std_ind_median_speed[ii,jj] = np.nanstd(replicate_ind_median_speed)
        ind_ratio_speed[ii,jj] = np.nanmean(replicate_ind_ratio_speed)
        std_ind_ratio_speed[ii,jj] = np.nanstd(replicate_ind_ratio_speed)

        jj += 1
    ii += 1
            


out_dir = '../../output/temp_collective/roi/'
"""
fn1 = out_dir + 'ind_startle_speed.p'
pickle.dump(ind_startle_speed, open(fn1, 'wb')) # 'wb' is for write binary

fn2 = out_dir + 'ind_startle_speed_std.p'
pickle.dump(std_ind_startle_speed, open(fn2, 'wb')) # 'wb' is for write binary
"""

fn1 = out_dir + 'ind_median_speed.p'
pickle.dump(ind_median_speed, open(fn1, 'wb')) # 'wb' is for write binary

fn2 = out_dir + 'ind_median_speed_std.p'
pickle.dump(std_ind_median_speed, open(fn2, 'wb')) # 'wb' is for write binary

fn1 = out_dir + 'ind_ratio_speed.p'
pickle.dump(ind_ratio_speed, open(fn1, 'wb')) # 'wb' is for write binary

fn2 = out_dir + 'ind_ratio_speed_std.p'
pickle.dump(std_ind_ratio_speed, open(fn2, 'wb')) # 'wb' is for write binary



