"""
Created on thu Jan 28 2021

@author: Maria Kuruvilla

Goal - Write csv file containing all covariates (including loom number) and distance to center
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

#temperatures used in the experiment - used for files naminf=g        
temperature = range(9,30,4)

#group sizes used in the experiment - used for naming files
group = [1,2,4,8,16]

#latest tracked replicate
replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')





with open('../../data/temp_collective/roi/stats_dtc_after_loom.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial','Time_fish_in', 'Time_start_record','Loom','dtc_after_loom','dtc_after_loom_norm'])

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
                    tr1 = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True)
                    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
                    tr1.new_time_unit(tr.params['frame_rate'], 'seconds')
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

                            frame_list = np.r_[looms[loom]+700:looms[loom]+1000]
                            writer.writerow([i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],met.Time_fish_in[m],met.Time_start_record[m],loom+1,np.nanmean(tr1.distance_to_origin[frame_list,:]),np.nanmean(tr.distance_to_origin[frame_list,:])])        
                
