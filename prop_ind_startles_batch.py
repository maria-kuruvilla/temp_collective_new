"""
Created on Thu Dec 3 2020

@author: Maria Kuruvilla

Goal - proportion of individuals that startle (using only masked data)
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

    

def filter(tr, roi = 5): #ind (for individual) starts from 0, roi - edge of region of interest
#shape returns tr.s.shape -2 
    position_mask0 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,0],copy=False)
    position_mask1 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,1],copy=False)
    return(np.ma.dstack((position_mask0,position_mask1)))  
    
def filter_speed(tr, roi = 5): #shape returns tr.s.shape -2 
    speed_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), speed(tr),copy=False)
    
    return(speed_mask)         



def filter_acc(tr, roi = 5): #shape returns tr.s.shape -2 
    acc_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data) 

    
def filter_e(tr, roi =5):
    x = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), e(tr)[:,:,0],copy=False)
    y = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), e(tr)[:,:,1],copy=False)
    return(np.ma.dstack((x,y)))  


met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')




def prop_startles(tr, loom):
    count = 0
    for j in range(tr.number_of_individuals):
        list2 = []
        list2 = [i for i, value in enumerate(filter_speed(tr,5)[(loom+500):(loom+700),j]) if value > 10]
        
        if list2:
            count = count + 1
    if count == 0:
        return(np.nan)
    else:
        return(count/tr.number_of_individuals) 


temperature = range(9,30,4)

group = [2,4,8,16,32]

replication = range(10) # number of replicates per treatment


average_prop_startles = np.empty([len(temperature), len(group)])
average_prop_startles.fill(np.nan)

average_prop_startles_std = np.empty([len(temperature), len(group)])
average_prop_startles_std.fill(np.nan)

#output parent directory
parent_dir = '../../data/temp_collective/roi'

ii = 0 # to keep count of temperature

#frames = 5000 #number of frames for which annd is calculated


for i in temperature:
    jj = 0 # to keep count of groups
    for j in group:
        out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/'
        replicate_startles_prop = np.empty([len(replication), 1])
        replicate_startles_prop.fill(np.nan)
        for k in replication:
            print(i,j,k+1)
            
            if j == 1:
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:   
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                
            

            
            
            try:
                tr = tt.Trajectories.from_idtrackerai(input_file,center=True).normalise_by('body_length')
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
            startles_prop = np.empty([5, 1]) # empty array to calculate average no. of startles per treatment
            startles_prop.fill(np.nan)
            for l in range(5):  
                startles_prop[l] = prop_startles(tr,looms[l])
            replicate_startles_prop[k] = np.nanmean(startles_prop)


        
        average_prop_startles[ii,jj] = np.nanmean(replicate_startles_prop)
        average_prop_startles_std[ii,jj] = np.nanstd(replicate_startles_prop)
        jj= jj + 1
        
    ii = ii + 1

out_dir = '../../output/temp_collective/roi/'

# save it as a pickle file
fn1 = out_dir + 'prop_startles.p'
pickle.dump(average_prop_startles, open(fn1, 'wb')) # 'wb' is for write binary

fn2 = out_dir + 'prop_startles_std.p'
pickle.dump(average_prop_startles_std, open(fn2, 'wb')) # 'wb' is for write binary

 
