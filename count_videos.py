"""
Goal - To count the number of videos I have in total 
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


temperature = range(9,30,4)

group = [1,2,4,8,16,32]

replication = range(10) # number of replicates per treatment

count1 = 0
count2 = 0 

#output parent directory
parent_dir = '../../data/temp_collective/roi'




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
                tr = tt.Trajectories.from_idtrackerai(input_file, 		           center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')		
            
            except FileNotFoundError:
                print(i,j,k+1)
                print('File not found')
                if j < 4:
                    count1 = count1 + 1
                else:
                    count2 = count2 + 1
                continue

print("Total number of videos for gs 1 - 32 is") 
print(360 - count1 - count2)
print("Total number of videos for gs 4 - 32 is")
print(240 - count2)

             