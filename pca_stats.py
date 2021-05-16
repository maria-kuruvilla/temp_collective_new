"""

Goal = write coeff of first pca components into csv file to do stats
Date - 27th April 2021
edited on 28th for another csv file - 1 if the pca coeff is positive, 0 if not

"""

#import stuffs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import pathlib
from pprint import pprint


from scipy import stats
from scipy.spatial import distance

from matplotlib.pyplot import figure

import trajectorytools as tt
import trajectorytools.plot as ttplot
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import dir_of_data
import csv
import pickle
import argparse
import pandas as pd


#load data
all_data = np.load('../../output/temp_collective/roi/convex_hull_area_loom.npy')

temperature = [9,13,17,21,25,29]#range(9,30,4)

group = [4,8,16]

replication = range(10) # number of replicates per treatment

met = pd.read_csv('../../data/temp_collective/roi/metadata_w_loom.csv')

with open('../../data/temp_collective/roi/pca_coeff_bernoulli.csv', mode='w') as stats_speed:
    writer = csv.writer(stats_speed, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow([
        'Temperature', 'Groupsize', 'Replicate', 'Trial', 'Date', 'Subtrial',
        'Time_fish_in', 'Time_start_record','Loom','pca_coeff'])

    for j in group:
        trial_info = all_data[:,0:4] #this is the temp, group_size, trial, loom_rep
        convex_hull_area = all_data[:,4:]

        #select data to use

        #First, let's get a specific group size
        group_size = j
        convex_hull_area_gs = convex_hull_area[trial_info[:,1]==group_size,:]
        trial_info_gs = trial_info[trial_info[:,1]==group_size,:]

        ##The data is the 1000 frames before and after the loom
        #let's look at the 300 frames after the loom
        frames_to_use = list(range(1000,1500))
        data_to_use = np.transpose(convex_hull_area_gs[:,frames_to_use])

        ##pre processing the data
        scaler = StandardScaler()   
        data_scaled = scaler.fit_transform(data_to_use)

        # see how many components we need

        pca = PCA().fit(data_scaled)
        """
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.xlim([0,10])
        """
        # Let's see how it works with 3

        num_pca_components = 1
        # this part is for second csv
        pca = PCA(n_components=num_pca_components)
        principalComponents = pca.fit_transform(data_scaled)  #find the PCAs or "loadings"
        PCAFit = scaler.inverse_transform(pca.inverse_transform(principalComponents))
        co_Effs = np.transpose(pca.components_)
        for i in range(len(co_Effs)):
            for m in range(len(met.Temperature)):
                if met.Temperature[m] == trial_info_gs[i,0] and met.Groupsize[m] == j and met.Replicate[m] == trial_info_gs[i,2]: 
                    if co_Effs[i][0] >= 0 :
                        a = 1
                    else:
                        a = 0
                    writer.writerow([
                        trial_info_gs[i,0],j,trial_info_gs[i,2],met.Trial[m],met.Date[m],met.Subtrial[m],
                        met.Time_fish_in[m],met.Time_start_record[m], trial_info_gs[i,3], a])
""" this part id for the first csv 

        pca = PCA(n_components=num_pca_components)
        principalComponents = pca.fit_transform(data_scaled)  #find the PCAs or "loadings"
        PCAFit = scaler.inverse_transform(pca.inverse_transform(principalComponents))
        co_Effs = np.transpose(pca.components_)
        for i in range(len(co_Effs)):
            for m in range(len(met.Temperature)):
                if met.Temperature[m] == trial_info_gs[i,0] and met.Groupsize[m] == j and met.Replicate[m] == trial_info_gs[i,2]: 
                    writer.writerow([
                        trial_info_gs[i,0],j,trial_info_gs[i,2],met.Trial[m],met.Date[m],met.Subtrial[m],
                        met.Time_fish_in[m],met.Time_start_record[m], trial_info_gs[i,3], co_Effs[i][0]])

"""


