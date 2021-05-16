"""
Goal - to write one csv file with convex hull during loom (at frame 625 after loom), 
convex hull during loom (at frame 600 - 650 after loom), max convex hull area (between 500 and 700 of loom)
Tue, Apr 20th 2021
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


a = 600
b = 625
c = 650

#convex hull area #does not use masked data because tracking errors will not affect convex hull area
def convex_hull_during(tr,looms,n, a, b, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals<4:
        return(np.nan)
    else:
        
        frame_list2 = list(range(looms[n]+a, looms[n]+b)) 
        convex_hull_area2 = np.empty([(b-a)])
        count2 = 0
        convex_hull_area2.fill(np.nan)
        for n in frame_list2 :
            convex_hull_area2[count2]=ConvexHull(tr.s[n]).area
            count2 += 1
        return(np.nanmean(convex_hull_area2))

def convex_hull_max(tr,looms,n, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals<4:
        return(np.nan)
    else:
        
        frame_list2 = list(range(looms[n]+500, looms[n]+700)) 
        convex_hull_area2 = np.empty([200])
        count2 = 0
        convex_hull_area2.fill(np.nan)
        for n in frame_list2 :
            convex_hull_area2[count2]=ConvexHull(tr.s[n]).area
            count2 += 1
        return(np.nanmax(convex_hull_area2), np.nanmean(convex_hull_area2))

def convex_hull_before(tr,looms,n, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals<4:
        return(np.nan)
    else:
        frame_list1 = list(range(looms[n]-1000, looms[n])) 
        
        convex_hull_area = np.empty([1000])
        count = 0
        convex_hull_area.fill(np.nan)
        for n in frame_list1 :
            convex_hull_area[count]=ConvexHull(tr.s[n]).area
            count += 1
        
        return(np.nanmean(convex_hull_area))

def convex_hull(tr,looms, roi1 = 30, roi2 = 3340):
    if tr.number_of_individuals<4:
        return(np.nan)
    else:
        frame_list1 = list(range(0, looms[0])) 
        convex_hull_area = np.empty([len(frame_list1)])
        count = 0
        convex_hull_area.fill(np.nan)
        
        for n in frame_list1 :
            convex_hull_area[count]=ConvexHull(tr.s[n]).area
            count += 1
        
        return(np.nanmean(convex_hull_area))

temperature = [9,13,17,21,25,29]#range(9,30,4)

group = [4,8,16]

replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

with open('../../data/temp_collective/roi/convex_hull_during_loom.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow([
        'Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial',
        'Time_fish_in', 'Time_start_record','Loom','convex_hull_area_'+str(a)+'_'+str(c),
        'convex_hull_area_'+str(500)+'_'+str(700),'convex_hull_area_'+str(b),'max_convex_hull'])

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
                        initial = convex_hull(tr,looms)
                        for n in range(5):
                            convex_range = convex_hull_during(tr,looms,n,a,c)
                            convex_one = convex_hull_during(tr,looms,n,b-1,b)
                            convex_max = convex_hull_max(tr,looms,n)[0]
                            convex_range2 = convex_hull_max(tr,looms,n)[1]
                            writer.writerow([
                                i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],
                                met.Time_fish_in[m],met.Time_start_record[m],n+1,
                                convex_range, convex_range2, convex_one, convex_max])        
