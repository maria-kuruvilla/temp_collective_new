This is a set of python scripts that analyses the trajectories generated by idtracker.ai.

________________________________________________________________________________

batch_analysis.py 

input: ../../data/temp_collective/roi/
output: ../../output/temp_collective/roi/
description: It loads all the trajectories in the experiment and calculates
             the average and standard deviation of speed, acceleration, average 
             nearest neighbor distance, latency, number of startles. It plots
             these parameters as a function of temperature and group size.
latency =
array([[574.        , 598.        , 591.33333333, 590.33333333,
        587.33333333],
       [593.        , 519.66666667, 582.5       , 580.33333333,
        549.33333333],
       [597.33333333, 799.33333333, 584.66666667, 588.66666667,
        463.66666667],
       [573.        , 387.33333333, 593.5       , 544.        ,
        579.        ],
       [596.        , 588.        , 456.        , 589.33333333,
        552.33333333],
       [         nan, 684.66666667, 385.66666667, 574.66666667,
        458.66666667]])

Each column is for group size and each row is for temperature.
________________________________________________________________________________

func_return_trajectory.py

input: G:/My Drive/CollectiveBehavior_Thermal_Experiments/Tracked
description: It a function that loads the trajectory if you input the temperature, 
             group size and replicate number.

_________________________________________________________________________________

func_test_spikes.py

input: G:/My Drive/CollectiveBehavior_Thermal_Experiments/Tracked
description: It loads all the trajectories and plots the speed as a function of
             frame number for one video. This is to check for errors in tracking

_________________________________________________________________________________

latency.py

input: G:/My Drive/CollectiveBehavior_Thermal_Experiments/Tracked
       C:/Users/Maria Kuruvilla/Documents/data/looms.csv
description: For a given video identified by a temperature, group size and
             replicate, it calculates the frame numbers between the start of the 
             loom and the startle of the first fish.

_________________________________________________________________________________

test_annd.py

input: G:/My Drive/CollectiveBehavior_Thermal_Experiments/Tracked
description: For a given video, it calculates the average distance between every
             fish and its nearest neighbor throughout the video. It returns the 
             average for all fish

________________________________________________________________________________

batch_analysis_masked.py 

input: ../../data/temp_collective/roi/
output: ../../output/temp_collective/roi/
description: It loads all the trajectories from the tracked videos and calculates
             the average and standard deviation of speed, acceleration, average 
             nearest neighbor distance, latency, number of startles, etc. with data where values 		     greater than a certain distance from the center of tank has been masked to reduce   		     edge effects. 
_________________________________________________________________________________

all_data_csv.py 

input: ../../data/temp_collective/roi/
      no arg parse

output: ../../output/temp_collective/csv/

description: to produce a csv file with frame, individual, position, speed, acceleration for each temp,gs,replicate combination - for john grady
_________________________________________________________________________________

all_params_w_loom.py 

input: ../../data/temp_collective/roi/
      no arg parse
      
output: ../../output/temp_collective/roi/

output type: csv file

description: to write one csv file with all params that have loom as covariate and that have both speed and acc masks
_________________________________________________________________________________

all_params_wo_loom.py 

input: ../../data/temp_collective/roi/
      
input type: no arg parse
      
output: ../../output/temp_collective/roi/all_params_wo_loom.csv

output type: csv file

description: to write one csv file with all params wo loom as covariate and that have both speed and acc masks
_________________________________________________________________________________


annd_batch.py 

input: ../../output/temp_collective/
      
input type: arg parse for replicate number
      
output: ../../output/temp_collective/annd.p

output type: pickled file

description: Code to analyse all the tracked videos and calculate annd and save it as pickled file.
_________________________________________________________________________________

annd_csv.py 

input: ../../data/temp_collective/roi
      ../../data/temp_collective/roi/metadata_w_loom.csv
      
input type: no arg parse 
      
output: ../../data/temp_collective/roi/stats_annd_data.csv

output type: csv file

description: Code to analyse all the tracked videos and calculate annd and save it as csv file.
_________________________________________________________________________________

batch_analysis_masked.py 

input: ../../data/temp_collective/roi
      
input type: arg parse for replicate number 
      
output: ../../output/temp_collective/roi/

output type: pickled files

description: to calculate average and std of many parameters with masked trajectory data
_________________________________________________________________________________

colin_tracking.py 


description: code to take output from colin's (andrew's colleague) tracking code and make speed histograms
_________________________________________________________________________________

convex_hull.py 

input: ../../data/temp_collective/roi
../../data/temp_collective/roi/metadata_w_loom.csv
      
input type: arg parse for temp,gs and replicate number 
      
output: 

output type: figure

description: To caluclate convex hull area and plot it
_________________________________________________________________________________

convex_hull_batch.py 

input: ../../data/temp_collective/roi
../../data/temp_collective/roi/metadata_w_loom.csv
      
input type: no arg parse 
      
output: ../../output/temp_collective/roi/convex_hull_area.p

output type: pickled file

description: To caluclate convex hull area of all the treatments - before, during and after the loom
_________________________________________________________________________________


