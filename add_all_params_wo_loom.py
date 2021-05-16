"""
Goal - Add polarization to the all params wo loom

Fri, Apr 16th 2021

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
from scipy.spatial import ConvexHull

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
        
def direction(tr): #e.shape returns tr.speed.shape - 2
    vel = (position(tr)[2:] - position(tr)[:-2]) / 2
    n = np.linalg.norm(vel,axis = 2)  
    return(vel/n[...,np.newaxis])


def filter_low_pass(tr, roi1 = 30, roi2 = 3340): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,0],copy=False)
    position_mask1 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,1],copy=False)
    b_new=np.ma.stack((position_mask0,position_mask1),axis=-1)    
    return(b_new)         

def filter_direction_low_pass(tr, roi1 = 30, roi2 = 3340):
    position_mask0 = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), direction(tr)[:,:,0],copy=False)
    position_mask1 = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), direction(tr)[:,:,1],copy=False)
    b_new=np.ma.stack((position_mask0,position_mask1),axis=-1)    
    return(b_new)       
                        

def filter_speed_low_pass(tr, roi1 = 30, roi2 = 3340):
    speed_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), speed(tr),copy=False)
    
    return(speed_mask)         

def filter_acc_low_pass(tr, roi1 = 30, roi2 = 3340):
    acc_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)  


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
        
#number of startles
#accurate startles()

#speed
def speed_percentile(tr, percentile, loom, n,roi1=30,roi2=3340):
    s_p = np.percentile(filter_speed_low_pass(tr,roi1,roi2)[(loom[n]+500):(loom[n]+700),:].compressed(),percentile)
    return(s_p)

#acc
def acc_percentile(tr, percentile, loom, n,roi1=30,roi2=3340):
    a_p = np.percentile(filter_acc_low_pass(tr,roi1,roi2)[(loom[n]+500):(loom[n]+700),:].compressed(),percentile)
    return(a_p)


#annd

def annd(trajectory,looms,n, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals==1:
        annd = np.nan
    else:
        nnd = np.empty([200, trajectory.number_of_individuals])
        nnd.fill(np.nan)
        
        nd = np.empty([200,trajectory.number_of_individuals])
        nd.fill(np.nan)
        
        for i in range(trajectory.number_of_individuals):
            for j in range(trajectory.number_of_individuals):
                if i!=j:
                    nd[:,j] = np.sqrt((filter_low_pass(tr, roi1, roi2)[0][(looms[n]+700):(looms[n]+900),i] - filter_low_pass(tr, roi1, roi2)[0][(looms[n]+700):(looms[n]+900),j])**2 + (filter_low_pass(tr, roi1, roi2)[1][(looms[n]+700):(looms[n]+900),i] - filter_low_pass(tr, roi1, roi2)[1][(looms[n]+700):(looms[n]+900),i])**2)
                
            nnd[:,i] = np.nanmin(nd,1)
            
        annd = np.nanmean(nnd)
    return(annd)
        
                
    

#convex hull area #does not use masked data because tracking errors will not affect convex hull area
def convex_hull(tr,looms,n, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals<4:
        return(np.nan)
    else:
        frame_list = list(range(looms[n]+700, looms[n]+900)) 
        convex_hull_area = np.empty([200])
        count = 0
        convex_hull_area.fill(np.nan)
        for n in frame_list :
            convex_hull_area[count]=ConvexHull(tr.s[n]).area
            count += 1
        return(np.nanmean(convex_hull_area))

#startle data
def startle_data(tr,t=10,roi1=30,roi2=3340):
    speed_mask2 = np.ma.masked_where((speed(tr) > roi1)|(speed(tr) < t)|(acceleration(tr) > roi2), speed(tr),copy=False)
    return(speed_mask2) 

#startle data percentile
def startle_data_percentile(tr, percentile, loom, n,t=10,roi1=30,roi2=3340):
    a = startle_data(tr,t,roi1,roi2)[(loom[n]+500):(loom[n]+700),:]
    if (a.all() is np.ma.masked) == True:
        s_p = np.nan
    else:
        s_p = np.percentile(a.compressed(),percentile)
    return(s_p)


#dtc #not masked
def dtc(tr,loom,n):
    dto = np.nanmean(tr.distance_to_origin[(loom[n]+700):(loom[n]+900),:])
    return(dto)


def local_polarization(tr,frames,n=1,x=1): #uses filtered data
    if n<x :
        return(np.nan)
    if tr.number_of_individuals == 1:
        return(np.nan)
    if x >= tr.number_of_individuals:
        return(np.nan)
    else:
        indices = ttsocial.give_indices(filter_low_pass(tr), n) # indices of the closest n neighbors
        a = ttsocial.restrict(filter_direction_low_pass(tr)[1:-1],indices) # normalized velocity (direction) vectors of focal individual and closest n neighbors 
        b = np.multiply(a[:,:,0,:],a[:,:,x,:]) #polarization between focal and xth closest individual
        b_mask = np.ma.array(b, mask = filter_direction_low_pass(tr)[1:-1].mask)
        c=np.nansum(b_mask,axis = 2) #dot product of vectors.
        #lp = tt.norm(tt.collective.polarization(a))
        #return(c)
        return(np.nanmean(c[frames]))

temperature = [9,13,17,21,25,29]#range(9,30,4)

group = [1,2,4,8,16]

replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

input = pd.read_csv('../../data/temp_collective/roi/all_params_w_loom.csv')

for random in range(1):

    with open('../../data/temp_collective/roi/all_params_wo_loom_pol.csv', mode='w') as csvoutput:
        
        writer = csv.writer(csvoutput, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(
            ['Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial',
            'Time_fish_in', 'Time_start_record','polarization_1', 'polarization_2'])

        for random2 in range(1):
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
                                frame_list=list(range(0,looms[0]+500))
                                writer.writerow(
                                    [i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],
                                    met.Time_fish_in[m],met.Time_start_record[m],
                                    local_polarization(tr,frame_list,1,1), 
                                    local_polarization(tr,frame_list,2,2)])