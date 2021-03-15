"""
Mon Jan 25th
Goal - To try fft with speed data
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
from scipy.fft import fft, fftfreq

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
parser.add_argument('-b2', '--integer_b2', default=16, type=int)
parser.add_argument('-b3', '--integer_b3', default=3, type=int)
parser.add_argument('-b4', '--integer_b4', default=0, type=int)
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
    tr = tt.Trajectories.from_idtrackerai(trajectories_file_path, center=True).normalise_by('body_length')
    tr.new_time_unit(tr.params['frame_rate'], 'seconds')
except FileNotFoundError:
    print(args.integer_b1,args.integer_b2,args.integer_b3)
    print('File not found')
    pass
"""
# Number of sample points

N = 600

# sample spacing

T = 1.0 / 800.0

x = np.linspace(0.0, N*T, N, endpoint=False)

y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

yf = fft(y)

xf = fftfreq(N, T)[:N//2]

import matplotlib.pyplot as plt

plt.plot(xf,  np.abs(yf[0:N//2]))

plt.grid()

plt.show()

N = tr.speed.shape[0]
for i in range(tr.number_of_individuals):
	yf = fft(tr.speed[:,i])
	
	xf = fftfreq(tr.speed.shape[0],1)[:tr.speed.shape[0]//2]
	plt.plot(xf,  np.abs(yf[0:N//2]))

plt.grid()

plt.show()


N = tr.speed.shape[0]
yf = fft(tr.speed)
xf = fftfreq(tr.speed.shape[0],1)[:tr.speed.shape[0]//2]
plt.plot(xf,  np.abs(yf[0:N//2]))
plt.show()
"""
#N = 1000
"""
N_total = 1500
ryf = np.fft.rfft(np.log(tr.speed[:N_total]))
rxf = np.fft.rfftfreq(N_total)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rxf*60,ryf[:(N_total//2 + 1)])
plt.ylabel('log(speed)')#+r'$^2$')
plt.xlabel('Hz')
plt.show()
"""
ind = args.integer_b4
temp = args.integer_b1
group = args.integer_b2
rep = args.integer_b3
lw = 0.75
N1 = 0
N = 1500
fig1 = plt.figure(figsize=(20,8))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(range(N1,N),np.log(tr.speed[N1:N,ind]), linewidth = lw, c = 'k', alpha = 0.75)

a = 352
b = 392
ax1.axvline(a, color = 'r',alpha=0.3) 
ax1.axvline(b, color = 'r',alpha=0.3) 
ax1.text(a+(b-a)/2,-2, str(b-a), c = 'r')
a = 809
b = 858
ax1.axvline(a, color = 'r',alpha=0.3) 
ax1.axvline(b, color = 'r',alpha=0.3) 
ax1.text(a+(b-a)/2,-2, str(b-a), c = 'r')

ax1.set_title('Temp:' + str(temp) + ' Group:' + str(group) + ' Replicate:' + str(rep))
plt.xlabel('Frames')
plt.ylabel('log(speed)')
plt.show()
