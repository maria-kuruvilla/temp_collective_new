"""
Goal - to add loom frame number to the metadata
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

rows_meta = []
with open('../../data/temp_collective/roi/metadata.csv', 'r') as csvfile:
    looms = csv.reader(csvfile)
    for row in looms:
        rows_meta.append(row)


rows_loom = []
with open('../../data/temp_collective/looms_roi.csv', 'r') as csvfile:
    looms = csv.reader(csvfile)
    for row in looms:
        rows_loom.append(row)



with open('../../data/temp_collective/roi/metadata_w_loom.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Temperature', 'Groupsize', 'Replicate', 'Trial','Date','Subtrial','Time_fish_in','Time_start_record','Loom 1','Loom 2','Loom 3','Loom 4','Loom 5'])
    for i in range(1,len(rows_meta)):
        for j in range(1,len(rows_loom)):
            if rows_loom[j][1]== 'Cam 7':
                temp = 29
            elif rows_loom[j][1]== 'Cam 8':
                temp = 25
            elif rows_loom[j][1]== 'Cam 9':
                temp = 17
            elif rows_loom[j][1]== 'Cam 10':
                temp = 13
            elif rows_loom[j][1]== 'Cam 11':
                temp = 21
            elif rows_loom[j][1]== 'Cam 12':
                temp = 9
            
            if (int(rows_meta[i][0]) == temp) and (rows_meta[i][1] == rows_loom[j][3]) and (rows_meta[i][2] == rows_loom[j][4]):
                print(i,j)
                writer.writerow([rows_meta[i][0], rows_meta[i][1], rows_meta[i][2], rows_meta[i][3],rows_meta[i][4],rows_meta[i][5],rows_meta[i][6],rows_meta[i][7], rows_loom[j][2], int(rows_loom[j][2]) + 11403, int(rows_loom[j][2]) + 2*11403, int(rows_loom[j][2]) + 3*11403, int(rows_loom[j][2]) + 4*11403])

