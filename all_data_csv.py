"""
Goal - to produce a csv file with frame, individual, position, speed, acceleration for each temp,gs,replicate combination
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
import time

##
ii = 13
jj = 3
kk = 4
temperature = range(ii,30,4)

group = range(jj,6)

replication = range(kk,10) # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'



for j in group:

    
    for i in temperature:
        
        
        for k in replication:
            print(i,2**j,k+1, time.time())
            if 2**j == 1:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(2**j)+'/GS_'+str(2**j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(2**j)+'/GS_'+str(2**j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
            try:
                tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, 		           center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')
            except FileNotFoundError:
                print(i,2**j,k+1, time.time())
                print('File not found')
                continue
            with open('../../data/temp_collective/csv/GS_' + str(2**j) + '_T_' + str(i) + '_rep_'+str(k+1)+'.csv', mode='w') as stats_speed:
                writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
                writer.writerow(['Frame','Individual','x','y','speed','acceleration'])
                for m in range(tr.s.shape[0]):
                    for n in range(tr.s.shape[1]):
                        writer.writerow([m,n+1,tr.s[m,n,0],tr.s[m,n,1],tr.speed[m,n], tr.acceleration[m,n]])
            temperature = range(9,30,4)
            replication = range(0,10)
