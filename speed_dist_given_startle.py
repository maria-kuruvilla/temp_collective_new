"""
Goal - Given that the speed is above the startle threshold, what is the distribution of speeds
Tue Mar 2 2021
edited on April 6th for another csv file (3) - startle data computed with second threshold = 5
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

def position(tr): #shape returns tr.s.shape
    return(tr.s)


def speed(tr): #speed(tr).shape returns tr.speed.shape - 2
    v = (position(tr)[2:] - position(tr)[:-2]) / 2
    b = np.linalg.norm(v, axis=-1)
    return(b*60)

def acceleration(tr): #shape returns tr.acceleration.shape - 2
    a = position(tr)[2:] - 2 * position(tr)[1:-1] + position(tr)[:-2]
    aa = np.linalg.norm(a, axis=-1)  
    return(aa*3600)
        
def e(tr): #e.shape returns tr.speed.shape - 2
    vel = (position(tr)[2:] - position(tr)[:-2]) / 2
    n = np.linalg.norm(v,axis = 2)  
    return(vel/n[...,np.newaxis])


def filter_low_pass(tr, roi1 = 30, roi2 = 3340): #ind (for individual) starts from 0, roi - edge of region of interest
    position_mask0 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,0],copy=False)
    position_mask1 = np.ma.masked_where((speed(tr)[1:-1] > roi1)|(speed(tr)[0:-2] > roi1)|(speed(tr)[2:] > roi1)|(acceleration(tr)[1:-1] > roi2)|(acceleration(tr)[0:-2] > roi2)|(acceleration(tr)[2:] > roi2), position(tr)[2:-2,:,1],copy=False)
    return(position_mask0,position_mask1)                                 

def filter_speed_low_pass(tr, roi1 = 30, roi2 = 3340):
    speed_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), speed(tr),copy=False)
    
    return(speed_mask)         

def filter_acc_low_pass(tr, roi1 = 30, roi2 = 3340):
    acc_mask = np.ma.masked_where((speed(tr) > roi1)|(acceleration(tr) > roi2), acceleration(tr),copy=False)
    
    return(acc_mask)#[~acc_mask.mask].data)  


#prop of ind that startles

def prop_startles(tr, loom, t=10,s1=30,s2=3340):
    count = 0
    for j in range(tr.number_of_individuals):
        list2 = []
        list2 = [i for i, value in enumerate(filter_speed_low_pass(tr,s1,s2)[(loom+500):(loom+700),j]) if value > t]
        
        if list2:
            count = count + 1
    #if count == 0:
    #    return(np.nan)
    else:
        return(count/tr.number_of_individuals) 

#latency - need to edit

def spikes_position_new(tr,roi1 = 30, roi2 = 3340, t = 10): #uses filter_speed
    list1 = []
    for j in range(tr.number_of_individuals):
        list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value > t]
        list2.insert(0,100000000)
        list1 = list1 + [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def accurate_startles(tr, loom, n, roi1 = 30, roi2 = 3340, t = 10): #uses filtered speed # n starts from 0
    list1 = spikes_position_new(tr,roi1 , roi2 , t)
    
    list2 = [i for i, value in enumerate(list1[:]) if value < (loom[n] + 700) and value > (loom[n]+500) ]
    
    return(len(list2))

def startle_data(tr,t=10,roi1=30,roi2=3340):
    speed_mask2 = np.ma.masked_where((speed(tr) > roi1)|(speed(tr) < t)|(acceleration(tr) > roi2), speed(tr),copy=False)
    
    return(speed_mask2) 

##### new startle data

def threshold0(tr, j, roi1 = 30, roi2 = 3340, t = 10):
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value > t]
    list2.insert(0,100000000)
    list1 = [value for i,value in enumerate(list2[1:]) if  (value != (list2[i]+1))]
        
    return(list1)

def threshold1(tr,j,loom, n=0, roi1 = 30, roi2 = 3340, t = 10):
    list1 = threshold0(tr, j, roi1 , roi2 , t)
    
    list2 = [value for i, value in enumerate(list1[:]) if value < (loom[n] + 700) and value > (loom[n]+500) ]
    
    return(list2)

def threshold2(tr, j, roi1 = 30, roi2 = 3340, t = 5):
    
    
    list2 = [i for i, value in enumerate(filter_speed_low_pass(tr, roi1, roi2)[:,j]) if value < t]
        
        
    return(list2)
    
def startle_size(tr, loom, n, roi1 = 30, roi2 = 3340, t1 = 10, t2 = 5):

    distance = np.empty([tr.number_of_individuals, 1])
    distance.fill(np.nan)

    perc99 = np.empty([tr.number_of_individuals, 1])
    perc99.fill(np.nan)

    perc90 = np.empty([tr.number_of_individuals, 1])
    perc90.fill(np.nan)

    perc50 = np.empty([tr.number_of_individuals, 1])
    perc50.fill(np.nan)

    avg = np.empty([tr.number_of_individuals, 1])
    avg.fill(np.nan)



    for ind in range(tr.number_of_individuals):
        speed_data = np.empty([1,])  
        speed_data.fill(np.nan)
        a = threshold1(tr,ind, loom,n)
        b = threshold2(tr,ind)

        if not a:
            distance[ind] = np.nan
            perc99[ind] = np.nan
            perc90[ind] = np.nan
            perc50[ind] = np.nan
            avg[ind] = np.nan
        else:

            c = []
            for i in a:
                for j in b:
                    if j>i:
                        c.append(j)
                        break

            
            
            for k in range(len(a)):
                speed_data = np.r_[speed_data,filter_speed_low_pass(tr, roi1, roi2)[a[k]:c[k],ind].compressed()]
                #distance_mult[k] = np.sum(speed_data)
            #print(speed_data)
            distance[ind] = np.nansum(speed_data)
            perc99[ind] = np.nanpercentile(speed_data,99)
            perc90[ind] = np.nanpercentile(speed_data,90)
            perc50[ind] = np.nanpercentile(speed_data,50)
            avg[ind] = np.nanmean(speed_data)
    #print(distance)
    return(
        np.nanmean(distance), np.nanmean(perc99), np.nanmean(perc90), 
        np.nanmean(perc50), np.nanmean(avg))


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

def speed_histogram(x1,y,t,s): 
    
    
    #### LOOM 

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
   
    n, bins, patches = ax1.hist(x1, 20, color='green',log = False)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #plt.hist(x, bins=logbins)
    #plt.xscale('log')
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x1, density=True, bins=logbins, log = False)
    #ax.hist(np.log10(x), density=True, log = True)
    ax.set_xscale('log')
    ax.axvline(np.percentile(x1.compressed(),99), color='red')
    ax.axvline(np.percentile(x1.compressed(),99.9), color='red')
    ax.axvline(np.percentile(x1.compressed(),90), color='red')
    ax.axvline(np.percentile(x1.compressed(),50), color='red')
    ax.axvline(np.nanmean(x1), color='green')
    plt.xticks(ticks = [t,s,np.percentile(x1.compressed(),99),np.percentile(x1.compressed(),99.9),np.percentile(x1.compressed(),90),np.percentile(x1.compressed(),50),np.nanmean(x1)], labels = [t,s,str(round(np.percentile(x1.compressed(),99),1)),str(round(np.percentile(x1.compressed(),99.9),1)),str(round(np.percentile(x1.compressed(),90),1)),str(round(np.percentile(x1.compressed(),50),1)),str(round(np.nanmean(x1),1))])
    ax.set_xlabel('Speed (BL/s)')
    ax.set_ylabel('Probability')
    ax.set_title('Startle speeds - Temperature: ' + str(y) +', Threshold: ' + str(t) +'BL/s')
    out_dir = parent_dir = '../../output/temp_collective/roi_figures/startle_speed_histogram_temp' + str(y)+'.png'
    fig.savefig(out_dir, dpi = 300)
    
    return(fig)


temperature = range(9,30,4)#[29]#[args.integer_b1]#



group = [1,2,4,8,16]#[args.integer_b2]



replication = range(10)#[args.integer_b3] # number of replicates per treatment

#output parent directory
parent_dir = '../../output/temp_collective/roi'

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

with open('../../data/temp_collective/roi/all_params_w_loom_startle_corrected3.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow([
        'Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial',
        'Time_fish_in', 'Time_start_record','Loom','distance','startle_data_percentile99',
        'startle_data_percentile90','startle_data_percentile50','avg_startle_speed'])

    for i in temperature:
        jj = 0 # to keep count of groups
        for j in group:
            out_dir = parent_dir + '/' + str(i) + '/' + str(j) + '/' 
            for k in replication:
                
                input_file = out_dir + str(k+1) + '_nosmooth.p'
                if j == 1:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories.npy'
                else:
                    trajectories_file_path = '../../data/temp_collective/roi/'+str(i)+'/' +str(j)+'/GS_'+str(j)+'_T_'+str(i)+'_roi_'+str(k+1)+'/trajectories_wo_gaps.npy'
                try:
                    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
                    tr.new_time_unit(tr.params['frame_rate'], 'seconds')        
                
                except FileNotFoundError:
                    print(i,j,k+1)
                    print('File not found')
                    continue
                looms_frame = []
                for m in range(len(met.Temperature)):
                    if met.Temperature[m] == i and met.Groupsize[m] == j and met.Replicate[m] == (k+1): 
                        looms_frame.append(met['Loom 1'][m])
                        looms_frame.append(met['Loom 2'][m])
                        looms_frame.append(met['Loom 3'][m])
                        looms_frame.append(met['Loom 4'][m])
                        looms_frame.append(met['Loom 5'][m])
                
                        for n in range(5):
                            startle_data=startle_size(tr,looms_frame,n)
                            
                            writer.writerow([
                                        i,j,k+1,met.Trial[m],met.Date[m],met.Subtrial[m],
                                        met.Time_fish_in[m],met.Time_start_record[m], n+1, 
                                        startle_data[0],startle_data[1], startle_data[2], 
                                        startle_data[3], startle_data[4]])