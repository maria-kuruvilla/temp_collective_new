"""
Created on Jan 28 2021
Goal - To calculate the average distance to center for each treatment

@author: Maria Kuruvilla
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
import pickle



#functions
    

temperature = range(9,30,4)



group = [1,2,4,8,16]



replication = range(10) # number of replicates per treatment


distance_to_center = np.empty([len(temperature), len(group)])
distance_to_center.fill(np.nan)

std_distance_to_center = np.empty([len(temperature), len(group)])
std_distance_to_center.fill(np.nan)

distance_to_center_norm = np.empty([len(temperature), len(group)])
distance_to_center_norm.fill(np.nan)

std_distance_to_center_norm = np.empty([len(temperature), len(group)])
std_distance_to_center_norm.fill(np.nan)




#output parent directory
parent_dir = '../../output/temp_collective/roi'

ii = 0 # to keep count of temperature

#frames = 5000 #number of frames for which annd is calculated

for i in temperature:
    jj = 0 # to keep count of groups
    for j in group:
        out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/' 
        
        average_replicate_dtc = np.empty([len(replication), 1])
        average_replicate_dtc.fill(np.nan)

        average_replicate_dtc_norm = np.empty([len(replication), 1])
        average_replicate_dtc_norm.fill(np.nan)

        for k in replication:
            
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
             
            average_replicate_dtc[k] = np.nanmean(tr.distance_to_origin)
            average_replicate_dtc_norm[k] = np.nanmean(tr1.distance_to_origin)
            
            
            
        
        distance_to_center[ii, jj] = np.nanmean(average_replicate_dtc)
        std_distance_to_center[ii,jj] = np.nanstd(average_replicate_dtc)
        distance_to_center_norm[ii, jj] = np.nanmean(average_replicate_dtc_norm)
        std_distance_to_center_norm[ii,jj] = np.nanstd(average_replicate_dtc_norm)

        
        
        jj= jj + 1
        
    ii = ii + 1

out_dir = '../../output/temp_collective/roi/'

# save it as a pickle file
dtc_fn1 = out_dir + 'dtc_roi_norm.p'
pickle.dump(distance_to_center, open(dtc_fn1, 'wb')) # 'wb' is for write binary

dtc_fn2 = out_dir + 'dtc_roi_norm_std.p'
pickle.dump(std_distance_to_center, open(dtc_fn2, 'wb')) # 'wb' is for write binary

dtc_fn1 = out_dir + 'dtc_roi.p'
pickle.dump(distance_to_center_norm, open(dtc_fn1, 'wb')) # 'wb' is for write binary

dtc_fn2 = out_dir + 'dtc_roi_std.p'
pickle.dump(std_distance_to_center_norm, open(dtc_fn2, 'wb')) # 'wb' is for write binary