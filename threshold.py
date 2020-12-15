"""
Goal - To calculate highest speed and acceleration for group size 1 individuals 
Created - 12/09/20
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

def position(tr):
    return(tr.s)


def speed(tr):
    v = (position(tr)[2:] - position(tr)[:-2]) / 2
    b = np.linalg.norm(v, axis=-1)
    return(b*60)

def acceleration(tr):
    a = position(tr)[2:] - 2 * position(tr)[1:-1] + position(tr)[:-2]
    aa = np.linalg.norm(a, axis=-1)  
    return(aa*3600)
        

max_speed = 0
max_acc = 0
max_i = 0
max_k = 0
max_j = 0
max_acc_i = 0
max_acc_k = 0
max_acc_j = 0

groups = [2,4,8,16,32]
temperature = range(9,30,4)
replication = range(10)

parent_dir = '../../data/temp_collective/roi'


for i in temperature:
    
    for j in groups:
        out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/' 
        

        for k in replication:
            
            if j == 1:
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:   
                input_file = out_dir + '/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                
            
            
            
            try:
                tr = tt.Trajectories.from_idtrackerai(input_file, center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')		
            
            except FileNotFoundError:
                print(i,j,k)
                print('File not found')
                continue
            s = speed(tr)
            acc = acceleration(tr)
            if np.max(s) > 30:
                max_speed = np.max(s)
                max_i = i
                max_k = k+1
                max_j = j
            if np.max(acc) > 3338:
                max_acc = np.max(acc)
                max_acc_i = i
                max_acc_k = k+1
                max_acc_j = j

            print(max_speed,max_acc, max_i, max_k, max_j, max_acc_i, max_acc_k, max_acc_j)



