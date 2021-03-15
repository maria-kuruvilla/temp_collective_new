"""
Goal - to enter the temp, group size and replicate and func should return a histogram of speed
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

def speed_histogram(x,x1,y,y1,y2): 
    
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
   
    n, bins, patches = ax1.hist(x, 20, color='green',log = False)
    
    
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #plt.hist(x, bins=logbins)
    #plt.xscale('log')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, density=True, bins=logbins, log = False)
    #ax.hist(np.log10(x), density=True, log = True)
    ax.set_xscale('log')
    ax.axvline(np.percentile(x.compressed(),99), color='red')
    ax.axvline(np.percentile(x.compressed(),99.9), color='red')
    ax.axvline(np.percentile(x.compressed(),90), color='red')
    ax.axvline(np.percentile(x.compressed(),50), color='red')
    ax.axvline(np.nanmean(x), color='green')
    plt.xticks(ticks = [1,10,30,np.percentile(x.compressed(),99),np.percentile(x.compressed(),99.9),np.percentile(x.compressed(),90),np.percentile(x.compressed(),50),np.nanmean(x)], labels = [1,10,30,str(round(np.percentile(x.compressed(),99),1)),str(round(np.percentile(x.compressed(),99.9),1)),str(round(np.percentile(x.compressed(),90),1)),str(round(np.percentile(x.compressed(),50),1)),str(round(np.nanmean(x),1))])
    ax.set_xlabel('Speed (BL/s)')
    ax.set_ylabel('Probability')
    ax.set_title('Temperature: ' + str(y)+' GS: ' + str(y1)+' Rep: ' + str(y2))
    #out_dir = parent_dir = '../../output/temp_collective/roi_figures/speed_histogram_temp' + str(y) +'.png'
    #fig.savefig(out_dir, dpi = 300)

    #### LOOM 

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
   
    n, bins, patches = ax1.hist(x1, 20, color='green',log = False)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #plt.hist(x, bins=logbins)
    #plt.xscale('log')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x1, density=True, bins=logbins, log = False)
    #ax.hist(np.log10(x), density=True, log = True)
    ax.set_xscale('log')
    ax.axvline(np.percentile(x1.compressed(),99), color='red')
    ax.axvline(np.percentile(x1.compressed(),99.9), color='red')
    ax.axvline(np.percentile(x1.compressed(),90), color='red')
    ax.axvline(np.percentile(x1.compressed(),50), color='red')
    ax.axvline(np.nanmean(x1), color='green')
    plt.xticks(ticks = [1,10,30,np.percentile(x1.compressed(),99),np.percentile(x1.compressed(),99.9),np.percentile(x1.compressed(),90),np.percentile(x1.compressed(),50),np.nanmean(x1)], labels = [1,10,30,str(round(np.percentile(x1.compressed(),99),1)),str(round(np.percentile(x1.compressed(),99.9),1)),str(round(np.percentile(x1.compressed(),90),1)),str(round(np.percentile(x1.compressed(),50),1)),str(round(np.nanmean(x1),1))])
    ax.set_xlabel('Speed (BL/s)')
    ax.set_ylabel('Probability')
    ax.set_title('Loom - Temperature: ' + str(y)+' GS: ' + str(y1)+' Rep: ' + str(y2))
    #out_dir = parent_dir = '../../output/temp_collective/roi_figures/speed_histogram_temp' + str(y)+ str(y1) + str(y2)+'.png'
    #fig.savefig(out_dir, dpi = 300)
    
    return(fig)
    #bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
      

    #col = bin_centers - min(bin_centers) 
    #col /= max(col)                                     


    # for c, p in zip(col, patches): 
    #     if c > 0.9: 
    #         plt.setp(p, 'facecolor', 'red') 

    #plt.show()
    
    
    
    
    

def acc_histogram(x,x1,y,y1,y2): #replicates start from 1
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
   
    n, bins, patches = ax1.hist(x, 20, color='green',log = False)
    
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #plt.hist(x, bins=logbins)
    #plt.xscale('log')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, density=True, bins=logbins, log = False)
    #ax.hist(np.log10(x), density=True, log = True)
    ax.set_xscale('log')
    ax.axvline(np.percentile(x.compressed(),99), color='red')
    ax.axvline(np.percentile(x.compressed(),99.9), color='red')
    ax.axvline(np.percentile(x.compressed(),90), color='red')
    ax.axvline(np.percentile(x.compressed(),50), color='red')
    ax.axvline(np.nanmean(x), color='green')
    plt.xticks(ticks = [1,10,np.percentile(x.compressed(),99),np.percentile(x.compressed(),99.9),np.percentile(x.compressed(),90),np.percentile(x.compressed(),50),np.nanmean(x)], labels = [1,10,str(round(np.percentile(x.compressed(),99),1)),str(round(np.percentile(x.compressed(),99.9),1)),str(round(np.percentile(x.compressed(),90),1)),str(round(np.percentile(x.compressed(),50),1)),str(round(np.nanmean(x),1))])
    ax.set_xlabel('Acceleration (BL/s'+r'$^2$)')
    ax.set_ylabel('Probability')
    ax.set_title(' Non Loom - Temperature: ' + str(y)+' GS: ' + str(y1)+' Rep: ' + str(y2))

    ## loom 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
   
    n, bins, patches = (ax1.hist(x1, 20, color='green',log = False))

    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #plt.hist(x, bins=logbins)
    #plt.xscale('log')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x1, density=True, bins=logbins, log = False)
    #ax.hist(np.log10(x), density=True, log = True)
    ax.set_xscale('log')
    ax.axvline(np.percentile(x1.compressed(),99), color='red')
    ax.axvline(np.percentile(x1.compressed(),99.9), color='red')
    ax.axvline(np.percentile(x1.compressed(),90), color='red')
    ax.axvline(np.percentile(x1.compressed(),50), color='red')
    ax.axvline(np.nanmean(x1), color='green')
    plt.xticks(ticks = [1,10,np.percentile(x1.compressed(),99),np.percentile(x1.compressed(),99.9),np.percentile(x1.compressed(),90),np.percentile(x1.compressed(),50),np.nanmean(x1)], labels = [1,10,str(round(np.percentile(x1.compressed(),99),1)),str(round(np.percentile(x1.compressed(),99.9),1)),str(round(np.percentile(x1.compressed(),90),1)),str(round(np.percentile(x1.compressed(),50),1)),str(round(np.nanmean(x1),1))])
    ax.set_xlabel('Acceleration (BL/s'+r'$^2$)')
    ax.set_ylabel('Probability')
    ax.set_title('Loom - Temperature: ' + str(y)+' GS: ' + str(y1)+' Rep: ' + str(y2))

    #out_dir = parent_dir = '../../output/temp_collective/roi_figures/acc_histogram_temp' + str(y) +'.png'
    #fig.savefig(out_dir, dpi = 300)

    return(fig)


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
        

def filter(tr, roi = 5): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,0],copy=False)
    position_mask1 = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), position(tr)[1:-1,:,1],copy=False)
    return(position_mask0,position_mask1)  
    
def filter_speed(tr, roi = 5): 
    speed_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), speed(tr),copy=False)
    
    return(speed_mask)         



def filter_acc(tr, roi = 5): 
    acc_mask = np.ma.masked_where((tr.distance_to_origin[1:-1] > roi)|(tr.distance_to_origin[0:-2] > roi)|(tr.distance_to_origin[2:] > roi), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data) 

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
parser.add_argument('-b1', '--integer_b1', default=9, type=int)
parser.add_argument('-b2', '--integer_b2', default=1, type=int)
parser.add_argument('-b3', '--integer_b3', default=1, type=int)
#parser.add_argument('-b3', '--integer_b4', default=1, type=int) # 0 for non loom data and #1 for loom
#parser.add_argument('-c', '--float_c', default=1.5, type=float)
parser.add_argument('-v', '--verbose', default=True, type=boolean_string)
# Note that you assign a short name and a long name to each argument.
# You can use either when you call the program, but you have to use the
# long name when getting the values back from "args".

# get the arguments
args = parser.parse_args()

temperature = [args.integer_b1]#range(9,30,4)



group = [args.integer_b2]



replication = [args.integer_b3] # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

ii = 0 # to keep count of temperature

#frames = 5000 #number of frames for which annd is calculated
x = [] #append speed values to this 
y=[]
x1 = [] #append speed values to this 
y1=[]
for i in temperature:
    jj = 0 # to keep count of groups
    for j in group:
        out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/' 
        for k in replication:
            
            input_file = out_dir + str(k+1) + '_nosmooth.p'
            if j == 1:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k)+'/trajectories.npy'
            else:
                trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k)+'/trajectories_wo_gaps.npy'
            try:
                tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
                tr.new_time_unit(tr.params['frame_rate'], 'seconds')		
            
            except FileNotFoundError:
                print(i,j,k)
                print('File not found')
                continue
            looms_frame = []
            for m in range(len(met.Temperature)):
                if met.Temperature[m] == i and met.Groupsize[m] == j and met.Replicate[m] == (k): 
                    looms_frame.append(met['Loom 1'][m])
                    looms_frame.append(met['Loom 2'][m])
                    looms_frame.append(met['Loom 3'][m])
                    looms_frame.append(met['Loom 4'][m])
                    looms_frame.append(met['Loom 5'][m])
            #for m in range(tr.speed.shape[1]):
             #   x= np.r_[x,tr.speed[:,m]]
            yy = np.ma.reshape(filter_acc_low_pass(tr)[0:looms_frame[0]],(filter_acc_low_pass(tr)[0:looms_frame[0]].shape[0]*filter_acc_low_pass(tr)[0:looms_frame[0]].shape[1]))
            xx = np.ma.reshape(filter_speed_low_pass(tr)[0:looms_frame[0]],(filter_speed_low_pass(tr)[0:looms_frame[0]].shape[0]*filter_speed_low_pass(tr)[0:looms_frame[0]].shape[1]))
            x = np.ma.mr_[x,xx] 
            y = np.ma.mr_[y,yy] 
            for looms in looms_frame:
                yy1 = np.ma.reshape(filter_acc_low_pass(tr)[(looms+500):(looms+700)],(filter_acc_low_pass(tr)[(looms+500):(looms+700)].shape[0]*filter_acc_low_pass(tr)[(looms+500):(looms+700)].shape[1]))
                xx1 = np.ma.reshape(filter_speed_low_pass(tr)[(looms+500):(looms+700)],(filter_speed_low_pass(tr)[(looms+500):(looms+700)].shape[0]*filter_speed_low_pass(tr)[(looms+500):(looms+700)].shape[1]))
                x1 = np.ma.mr_[x1,xx1] 
                y1 = np.ma.mr_[y1,yy1] 
""" 
            for m in range(tr.acceleration.shape[1]):
                y= np.r_[y,filter_acc(tr,5)[:,m]]
                x= np.r_[x,filter_speed(tr,5)[:,m]]
"""
acc_histogram(y+1, y1+1, args.integer_b1, args.integer_b2, args.integer_b3)
speed_histogram(x+1, x1+1, args.integer_b1, args.integer_b2, args.integer_b3)
#acc_histogram(tr, args.integer_b1, args.integer_b2, args.integer_b3)
plt.show()
