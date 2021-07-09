"""
Goal - to draw trajectories on video
Jun 9th 2021
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
import cv2
import cmapy
from scipy.spatial import ConvexHull
import pandas as pd

#argparse
def boolean_string(s):
    # this function helps with getting Boolean input
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True' # note use of ==

# create the parser object
parser = argparse.ArgumentParser()

# NOTE: argparse will throw an error if:
#     - a flag is given with no value
#     - the value does not match the type
# and if a flag is not given it will be filled with the default.
parser.add_argument('-a', '--a_string', default='/ssd/Trial_3/07-24-19_14-50-26.000_Cam02.avi', type=str)
parser.add_argument('-b1', '--integer_b1', default=29, type=int)
parser.add_argument('-b2', '--integer_b2', default=16, type=int)
parser.add_argument('-b3', '--integer_b3', default=3, type=int)
parser.add_argument('-f1', '--integer_f1', default=0, type=int)
parser.add_argument('-f2', '--integer_f2', default=10000, type=int)
parser.add_argument('-c', '--float_c', default=1.5, type=float)
parser.add_argument('-v', '--verbose', default=True, type=boolean_string)
# Note that you assign a short name and a long name to each argument.
# You can use either when you call the program, but you have to use the
# long name when getting the values back from "args".

# get the arguments
args = parser.parse_args()

parent_dir = '../../output/temp_collective/roi'

input_dir = parent_dir + '/' + str(args.integer_b1) + '/' + str(args.integer_b2) + '/' 
input_file = input_dir + str(args.integer_b3) + '_nosmooth.p'
#sigma_values = 1.5 #smoothing parameter
if args.integer_b2 == 1:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories.npy'
else:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories_wo_gaps.npy'
try:
    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path)
    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
except FileNotFoundError:
    print(args.integer_b1,args.integer_b2,args.integer_b3)
    print('File not found')
    pass 

FPS = 60
seconds = 10
radius = 4


vidcap = cv2.VideoCapture(args.a_string)

frame_width = int( vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( vidcap.get( cv2.CAP_PROP_FRAME_HEIGHT))  

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('../../output/temp_collective/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'_frame_'+str(args.integer_f1)+'_'+str(args.integer_f2)+'schematic_new.avi', fourcc, float(FPS), (frame_width, frame_height))
success,image = vidcap.read()
count = 0

"""
while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    image = cv2.circle(image, (int(tr.s[count-1,0,0]), int(tr.s[count-1,0,1])), radius, (255, 0, 0), -1)
    video.write(image)
video.release()
"""
colour = []
thickness = 2
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  

  
# fontScale 
fontScale = 0.8
#n_frames = args.integer_f2 - args.integer_f1
n_frames= 1000
#colors = plt.cm.bone(np.linspace(0,1,n_frames))
colour = []
for n in np.linspace(0,255,n_frames,dtype = int):
    #colour.append(list(np.random.random(size=3) * 256)) 
    colour.append(cmapy.color('bone',n))


#viridis = plt.cm.get_cmap('bone', 12)

#color_trial = viridis(np.linspace(0, 1, n_frames))

loom = 54495

#overlay = image.copy()
hull = ConvexHull(tr.s[54845]) 
pts = tr.s[54845][hull.vertices].reshape((-1,1,2)) 
hull2 = ConvexHull(tr.s[53845]) 
pts2 = tr.s[53845][hull2.vertices].reshape((-1,1,2))  
hull3 = ConvexHull(tr.s[loom]) 
pts3 = tr.s[loom][hull3.vertices].reshape((-1,1,2)) 
for i in range(args.integer_f2):#tr.s.shape[0]):
    
    success,image = vidcap.read()
    
    if i > args.integer_f1:
        print(i)
        for j in range(tr.number_of_individuals):
            
            for k in range(n_frames):
                image = cv2.circle(image, (int(tr.s[i-k,j,0]), int(tr.s[i-k,j,1])), radius, colour[k] , -1)
        cv2.polylines(image,np.int32([pts]),True,colour[0], thickness = 2)
        cv2.polylines(image,np.int32([pts2]),True,colour[n_frames-1], thickness = 2)
        cv2.polylines(image,np.int32([pts3]),True,colour[54845-loom], thickness = 2)
        org = (int(tr.s[i,0,0]), int(tr.s[i,0,1]) + 50)
        image = cv2.putText(image, 'convex hull' , org, font, fontScale, colour[j], thickness, cv2.LINE_AA, False) 
        
        #alpha = 0.4  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        #image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        video.write(image)
video.release()
