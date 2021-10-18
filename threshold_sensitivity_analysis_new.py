"""
Date - Mon 18 Oct 2021
Goal - For a range of speed thresholds and startle thresholds, calculate max speed, max acc, latency, prop ind startling
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

def filter_speed_low_pass(tr, roi1 = 30, roi2 = 3340):
    speed_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), speed(tr),copy=False)
    
    return(speed_mask)         

def filter_acc_low_pass(tr, roi1 = 30, roi2 = 3340):
    acc_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)  

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')


#prop of ind that startles

def prop_startles(tr, loom,n,s1=30,s2=3340, t=10):
    count = 0
    for j in range(tr.number_of_individuals):
        list2 = []
        list2 = [i for i, value in enumerate(filter_speed_low_pass(tr,s1,s2)[(loom[n]+500):(loom[n]+700),j]) if value > t]
        
        if list2:
            count = count + 1
    #if count == 0:
    #    return(np.nan)
    else:
        return(count/tr.number_of_individuals) 


#latency 

def spikes_position_new(tr,roi1 = 30, roi2 = 3340, t = 10): #uses filter_speed
    list1 = []
    for j in range(tr.number_of_individuals):
        list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value > t]
        list2.insert(0,100000000)
        list1 = list1 + [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def accurate_startles(tr, loom, n, roi1 = 30, roi2 = 3340, t = 10): #uses filtered speed # n starts from 0
    list1 = spikes_position_new(tr,roi1 , roi2 , t)
    
    list2 = [i for i, value in enumerate(list1[:]) if value < (loom[n] + 700) and value > (loom[n]+500) ]
    
    return(len(list2))

def accurate_startles_frame(tr, loom, n, roi1 = 30, roi2 = 3340, t = 10): #uses filtered speed # n starts from 0
    list1 = spikes_position_new(tr,roi1 , roi2 , t)
    
    list2 = [value for value in list1 if value < (loom[n] + 700) and value > (loom[n]+500) ]
    
    return(list2)

def startles_list_new(tr,roi1 = 30, roi2 = 3340, t = 10): #uses filtered data
    return(len(spikes_position_new(tr,roi1 , roi2 , t)))

def first_startle(tr, loom,n,roi1 = 30, roi2 = 3340, t = 10):
    a = accurate_startles_frame(tr,loom,n,roi1 , roi2 , t)
    if not a:
        return(accurate_startles_frame(tr,loom,n,roi1 , roi2 , t))
    else:
        return(min(a)) 
    

def latency(tr, loom,n,roi1 = 30, roi2 = 3340, t = 10):
    a = first_startle(tr,loom,n,roi1 , roi2 , t)
    if not a:
        return(float('nan'))
    else:
        return(a - loom[n])
        
#speed
def speed_percentile(tr, percentile, loom, n,roi1=30,roi2=3340):
    s_p = np.percentile(filter_speed_low_pass(tr,roi1,roi2)[(loom[n]+500):(loom[n]+700),:].compressed(),percentile)
    return(s_p)

#acc
def acc_percentile(tr, percentile, loom, n,roi1=30,roi2=3340):
    a_p = np.percentile(filter_acc_low_pass(tr,roi1,roi2)[(loom[n]+500):(loom[n]+700),:].compressed(),percentile)
    return(a_p)




startle_threshold = [5,6,7,8,9,10,11,12,13,14,15]

speed_threshold = [25,30,35]

acc_threshold = [2000, 3000, 4000]


replicate_startles_prop = np.empty([len(startle_threshold), len(speed_threshold), len(acc_threshold)])
replicate_startles_prop.fill(np.nan)

list1 = ['Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial','Time_fish_in', 'Time_start_record','Loom']


for i in startle_threshold:
    for j in speed_threshold:
        for k in acc_threshold:
            list1.append('latency' + str(i) + '_' + str(j)+ '_' + str(k))

for i in startle_threshold:
    for j in speed_threshold:
        for k in acc_threshold:
            list1.append('prop_startles' + str(i) + '_' + str(j)+ '_' + str(k))

for j in speed_threshold:
    for k in acc_threshold:
        list1.append('speed' + str(j) + '_' + str(k))

for j in speed_threshold:
    for k in acc_threshold:
        list1.append('acc' + str(j) + '_' + str(k))



temperature = range(9,30,4)

group = [1,2,4,8,16]

replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

with open('../../data/temp_collective/roi/sensitivity_analysis_new.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(list1)

    for i in temperature:
        
        for j in group:
            

            for k in replication:
                #print(i,j,k+1)
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
                            frame = looms[loom]
                            list2 = []
                            for ii in startle_threshold:
                                for jj in speed_threshold:
                                    for kk in acc_threshold:
                                        list2.append(latency(tr, looms, loom, jj, kk, ii))

                            for ii in startle_threshold:
                                for jj in speed_threshold:
                                    for kk in acc_threshold:
                                        list2.append(prop_startles(tr, looms, loom, jj, kk, ii))

                            for jj in speed_threshold:
                                for kk in acc_threshold:
                                    list2.append(speed_percentile(tr, 99, looms, loom, jj, kk))

                            for jj in speed_threshold:
                                for kk in acc_threshold:
                                    list2.append(acc_percentile(tr, 99, looms, loom, jj, kk))

                            writer.writerow([i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],met.Time_fish_in[m],met.Time_start_record[m],loom+1]+list2)        
                
