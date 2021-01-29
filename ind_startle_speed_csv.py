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

def spikes_position_new(tr,j): #uses filter_speed
    list1 = []
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr)[:,j]) if value > 10]
    list2.insert(0,100000000)
    list1 = list1 + [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def accurate_startles(tr, loom, num, ind): #uses filtered speed
    list1 = spikes_position_new(tr, ind)
    

    
    list2 = [i for i, value in enumerate(list1[:]) if value < (loom[num] + 700) and value > (loom[num]+500) ]

    
    return(len(list2))

#temperatures used in the experiment - used for files naminf=g        
temperature = range(9,30,4)

#group sizes used in the experiment - used for naming files
group = [1,2,4,8,16]

#latest tracked replicate
replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')





with open('../../data/temp_collective/roi/stats_ind_startle_speed_loom.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial','Time_fish_in', 'Time_start_record','Loom','Individual','max_startle_speed','ratio'])

    for i in temperature:
        
        for j in group:
            

            for k in replication:
                print(i,j,k+1)
                if j == 1:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
                else:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                try:
                    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
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
                        for loom in range(5):
                            frame_list = np.r_[looms[loom]+500:looms[loom]+700]
                            for ind in range(tr.number_of_individuals):
                                if accurate_startles(tr,looms,loom,ind) != 0 :

                                    writer.writerow([i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],met.Time_fish_in[m],met.Time_start_record[m],loom+1,ind+1,np.nanmax(filter_speed_low_pass(tr)[frame_list,ind]),np.nanmax(filter_speed_low_pass(tr)[frame_list,ind])/np.percentile(filter_speed_low_pass(tr)[0:looms[0],ind],50)])        
                
