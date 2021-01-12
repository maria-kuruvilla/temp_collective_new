"""
Total number of videos for gs 1 - 32 is
239
Total number of videos for gs 4 - 32 is
151
Goal - write npy file of convex hulls for fpca analysis 
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
import math

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

temperature = range(9,30,4)

group = [4,8,16,32]

replication = range(10) # number of replicates per treatment


convex_hull_area = np.empty([151,10003])
convex_hull_area.fill(np.nan)


in_dir1 = '../../output/temp_collective/convex_hull_area.p'

area = pickle.load(open(in_dir1, 'rb')) # 'rb is for read binary


count1 = 0
count2 = 0

ii = 0 # to keep count of temperature
for i in temperature:
    jj = 0 # to keep count of groups
    for j in group:
        

        for k in replication:
            
            if math.isnan(area[0,ii,jj,k]) == False: 
            	print(i,j,k)
            	convex_hull_area[count1,3:10003] = area[0:10000,ii,jj,k]
            	convex_hull_area[count1,0] = i
            	convex_hull_area[count1,1] = j
            	convex_hull_area[count1,2] = k+1
            	count1 = count1 + 1

            

        
        
        jj= jj + 1
        
    ii = ii + 1

out_dir = '../../output/temp_collective/roi/convex_hull_area.npy'
np.save(out_dir,convex_hull_area)