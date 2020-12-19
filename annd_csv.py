"""
Created on Sat Dec 18 2020

@author: Maria Kuruvilla

Goal - Code to analyse all the tracked videos and calculate annd and save it as csv file.
"""


import sys, os
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
import pickle
import pandas as pd
import csv

#temperatures used in the experiment - used for files naminf=g        
temperature = range(9,30,4)

#group sizes used in the experiment - used for naming files
group = [2,4,8,16,32]

#latest tracked replicate
replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')


def annd(trajectory):
    nnd = np.empty([trajectory.s.shape[0], trajectory.number_of_individuals])
    nnd.fill(np.nan)
    
    nd = np.empty([trajectory.s.shape[0],trajectory.number_of_individuals])
    nd.fill(np.nan)
    
    for i in range(trajectory.number_of_individuals):
        for j in range(trajectory.number_of_individuals):
            if i!=j:
                nd[:,j] = np.sqrt((trajectory.s[:,i,0] - trajectory.s[:,j,0])**2 + (trajectory.s[:,i,1] - trajectory.s[:,i,1])**2)
            
        nnd[:,i] = np.nanmin(nd,1)
        
    annd = np.nanmean(nnd)
        
                
    return(annd)

#output parent directory
parent_dir = '../../data/temp_collective/roi'



with open('../../data/temp_collective/roi/stats_annd_data.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial','Time_fish_in', 'Time_start_record','annd'])

    for i in temperature:
        
        for j in group:
            out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/'

            for k in replication:
                print(i,j,k+1)
                if j == 1:
                    input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
                else:   
                    input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                    
                

                
                
                try:
                    tr = tt.Trajectories.from_idtrackerai(input_file, center=True).normalise_by('body_length')
                    tr.new_time_unit(tr.params['frame_rate'], 'seconds')        
                
                except FileNotFoundError:
                    print(i,j,k+1)
                    print('File not found')
                    continue
                 
                
                
                annd_value = annd(tr)
                for m in range(len(met.Temperature)):
                    if met.Temperature[m] == i and met.Groupsize[m] == j and met.Replicate[m] == (k+1): 
                        writer.writerow([i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],met.Time_fish_in[m],met.Time_start_record[m],annd_value])        
                
                

                
         
