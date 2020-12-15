"""
Goal - to produce a csv file with frame, temp, gs, individual, date of exp, trial # in day, 
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


rows1 = []
with open('../../data/temp_collective/roi/trial_date.csv', 'r') as csvfile:
    looms = csv.reader(csvfile)
    for row in looms:
        rows1.append(row)

rows2 = []
with open('../../data/temp_collective/roi/trial_replicate.csv', 'r') as csvfile:
    looms2 = csv.reader(csvfile)
    for row in looms2:
        rows2.append(row)
        

temperature = range(9,30,4)



group = [1,2,4,8,16,32]



replication = range(10) # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'

with open('../../data/temp_collective/roi/metadata.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Temperature', 'Groupsize', 'Replicate', 'Trial','Date','Subtrial','Time_fish_in','Time_start_record'])
    for i in range(len(rows2)):
        for j in range(len(rows1)):
            if (rows2[i][2] == rows1[j][0]):
                writer.writerow([int(rows2[i][0]), int(rows2[i][1]), int(rows2[i][3]), int(rows2[i][2]),rows1[j][1],int(rows1[j][2]),rows1[j][3],rows1[j][4]])

"""
    for i in temperature:
        print(i)
        jj = 0 # to keep count of groups
        for j in group:
            for k in replication:
                if j == 1:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
                else:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                try:
                    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path,center=True).normalise_by('body_length')
                    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
                except FileNotFoundError:
                    print(i,j,k)
                    print('File not found')
                    continue
                #perc_speed = np.percentile(filter_speed(tr,5).compressed(),90)
                #perc_acc = np.percentile(filter_acc(tr,5).compressed(),90)
                #for loom in range(5):
                    #lat = latency_loom(tr,i,j,k+1, loom)
                    #if np.isnan(lat) != True:
                        #writer.writerow([i, j, k+1, loom+1,lat])
"""
