"""
Goal - To caluclate convex hull area of all the treatments - before, during and after the loom
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

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

temperature = range(9,30,4)

group = [4,8,16,32]

replication = range(10) # number of replicates per treatment


convex_hull_area = np.empty([10000,len(temperature), len(group), 10])
convex_hull_area.fill(np.nan)



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
                tr = tt.Trajectories.from_idtrackerai(input_file, 		           center=True).normalise_by('body_length')
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
            
            
            frame_list = list(range(looms[0]-500, looms[0]+1500))+ list(range(looms[1]-500, looms[1]+1500)) + list(range(looms[2]-500, looms[2]+1500)) + list(range(looms[3]-500, looms[3]+1500)) + list(range(looms[4]-500, looms[4]+1500)) 
        
            count = 0
            for n in frame_list :
                convex_hull_area[count,ii,jj,k]=ConvexHull(tr.s[n]).area
                count += 1


        
        
        jj= jj + 1
        
    ii = ii + 1

out_dir = '../../output/temp_collective/roi/'

# save it as a pickle file
fn1 = out_dir + 'convex_hull_area.p'
pickle.dump(convex_hull_area, open(fn1, 'wb')) # 'wb' is for write binary

 
