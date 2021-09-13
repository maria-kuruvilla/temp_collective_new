"""
Goal - To produce figure with trajectories and convex hull area
June 9th 2021
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
parser.add_argument('-a', '--a_string', default='hi', type=str)
parser.add_argument('-b1', '--integer_b1', default=29, type=int)
parser.add_argument('-b2', '--integer_b2', default=8, type=int)
parser.add_argument('-b3', '--integer_b3', default=3, type=int)
parser.add_argument('-b4', '--integer_b4', default=2, type=int)
parser.add_argument('-c', '--float_c', default=1.5, type=float)
parser.add_argument('-v', '--verbose', default=True, type=boolean_string)
# Note that you assign a short name and a long name to each argument.
# You can use either when you call the program, but you have to use the
# long name when getting the values back from "args".

# get the arguments
args = parser.parse_args()


if args.integer_b2 == 1:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories.npy'
else:
    trajectories_file_path = '../../data/temp_collective/roi/'+str(args.integer_b1)+'/' +str(args.integer_b2)+'/GS_'+str(args.integer_b2)+'_T_'+str(args.integer_b1)+'_roi_'+str(args.integer_b3)+'/trajectories_wo_gaps.npy'
try:
    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
except FileNotFoundError:
    print(args.integer_b1,args.integer_b2,args.integer_b3)
    print('File not found')
    pass

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
        
                                  

def filter_low_pass(tr, roi1 = 30, roi2 = 3340): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,0],copy=False)
    position_mask1 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,1],copy=False)
    return(position_mask0,position_mask1)                                 

def filter_speed_low_pass(tr, roi = 30): 
    speed_mask = np.ma.masked_where((speed(tr) > roi), speed(tr),copy=False)
    
    return(speed_mask)         



def filter_acc_low_pass(tr, roi = 3340): 
    acc_mask = np.ma.masked_where((acceleration(tr) > roi), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)  

looms = []
met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')
for i in range(len(met.Temperature)):
    if met.Temperature[i] == args.integer_b1 and met.Groupsize[i] == args.integer_b2 and met.Replicate[i] == args.integer_b3 : 
        looms.append(met['Loom 1'][i]) 
        looms.append(met['Loom 2'][i]) 
        looms.append(met['Loom 3'][i]) 
        looms.append(met['Loom 4'][i]) 
        looms.append(met['Loom 5'][i]) 

n_frames = 900
hull_frame= 600
colors = plt.cm.bone_r(np.linspace(0,1,n_frames+200))
loom_no = looms[args.integer_b4]

# colors = np.empty([ n_frames,tr.number_of_individuals,4])
# colors.fill(np.nan)

# for i in range(tr.number_of_individuals):
# 	colors[:,i,:] = plt.cm.bone_r(np.linspace(0,1,n_frames))

plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(tr.number_of_individuals):
#for i in range(loom_no,loom_no+10):
	ax.scatter(filter_low_pass(tr)[0][loom_no:loom_no+n_frames,i], filter_low_pass(tr)[1][loom_no:loom_no+n_frames,i], color = colors[200:(n_frames+200)], s = 10)
hull = ConvexHull(tr.s[loom_no+n_frames]) 
hull2 = ConvexHull(tr.s[loom_no+hull_frame]) 
hull3 = ConvexHull(tr.s[loom_no])
for simplex in hull.simplices:

    plt.plot(tr.s[loom_no+n_frames][simplex, 0], tr.s[loom_no+n_frames][simplex, 1], color = colors[n_frames+199])
pts = tr.s[loom_no+n_frames][hull.vertices].reshape((-1,1,2))
ax.fill(pts[:,:,0],pts[:,:,1], facecolor = colors[n_frames+199], alpha = 0.2)
for simplex in hull2.simplices:

    plt.plot(tr.s[loom_no+hull_frame][simplex, 0], tr.s[loom_no+hull_frame][simplex, 1], color = colors[hull_frame+199])
pts = tr.s[loom_no+hull_frame][hull2.vertices].reshape((-1,1,2))
ax.fill(pts[:,:,0],pts[:,:,1], facecolor = colors[hull_frame+199], alpha = 0.2)
for simplex in hull3.simplices:

    plt.plot(tr.s[loom_no][simplex, 0], tr.s[loom_no][simplex, 1], color = colors[199])
pts = tr.s[loom_no][hull3.vertices].reshape((-1,1,2))
ax.fill(pts[:,:,0],pts[:,:,1], facecolor = colors[199], alpha = 0.2)

fs = 16
plt.annotate(text='Convex \n hull', xy=(-4.5,-4.2), fontsize = fs)
plt.annotate(text='Post-\nloom (5 s)', xy=(-4.5,-2.5), fontsize = fs)
plt.annotate(text='Loom (0 s)', xy=(-2,-1.7), fontsize = fs)
plt.annotate(text='Pre-\nloom (-10 s)', xy=(2,-3.6), fontsize = fs)


plt.xlim([-5, 5])
plt.ylim([-4, 5.9])

plt.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

out_dir = '../../output/temp_collective/roi_figures/schematic_figure_4_no_background.png'
fig.savefig(out_dir, dpi = 1200, bbox_inches="tight")


plt.show()