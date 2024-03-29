"""
Goal - to write one csv file with average body length (in pixels) of all fish
Fri, April 1st 2022

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


body_length_gs1 = [] 
body_length_list = [] 
temperature = [9,13,17,21,25,29]#range(9,30,4)

group = [1,2,4,8,16]

replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

body_length = np.zeros((6,5))
body_length_std = np.zeros((6,5))
ii = 0
jj = 0

for i in temperature:
    #body_length_gs1 = []  

    for j in group:
            
        body_length_list=[]
        for k in replication:
            #print(i,j,k+1)
            
            if j == 1:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
            else:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
            try:
                tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
                    
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')
                body_length_gs1.append(tr.params['body_length_px'])
                body_length_list.append(tr.params['body_length_px'])
            except FileNotFoundError:
                print(i,j,k)
                print('File not found')
                continue
        body_length[ii,jj] = np.median(body_length_list)
        body_length_std[ii,jj] = np.std(body_length_list)
        
        jj = jj + 1
    jj=0
    ii = ii + 1
    #print(i)
    #print(np.mean(body_length_gs1)) 
    #print(np.std(body_length_gs1)) 
         
#print(np.mean(body_length_gs1)) 
#print(np.std(body_length_gs1))                     
print(body_length)   
print(body_length_std)                 
                 
a = np.array([[46.5/892]*5])  
b = np.array([[46.5/884.67]*5])

c = np.array([[46.5/915.4]*5])                                         

d = np.array([[46.5/828]*5])                                           

e = np.array([[46.5/922.5]*5])                                         

f = np.array([[46.5/917.9]*5])   

conversion = np.vstack((a,b,c,d,e,f))
np.multiply(body_length_std,conversion) 
np.multiply(body_length_std,conversion)           
