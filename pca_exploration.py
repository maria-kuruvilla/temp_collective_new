#import stuffs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#load data
all_data = np.load('../../output/temp_collective/roi/convex_hull_area_loom.npy')

trial_info = all_data[:,0:4] #this is the temp, group_size, trial, loom_rep
convex_hull_area = all_data[:,4:]

#select data to use

#First, let's get a specific group size
group_size = 16
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

pca = PCA(n_components=num_pca_components)
principalComponents = pca.fit_transform(data_scaled)  #find the PCAs or "loadings"
PCAFit = scaler.inverse_transform(pca.inverse_transform(principalComponents))

for i in range(num_pca_components):
    plt.plot(principalComponents[:,i])
plt.fill_between([100,150],-20,20,alpha=0.2, lw= 0, label = 'during loom')
plt.fill_between([200,400],-20,20,alpha=0.2, lw= 0, label = 'after loom')
# plt.axvline(149)
# plt.axvline(100)
# plt.axvline(200)
#plt.axvline(400)
plt.xlabel('Frame number')
plt.ylabel('Empirical Orthoginal Function (EOF)')
plt.legend(loc='upper right', framealpha = 0.5)
out_dir = '../../output/temp_collective/roi_figures/pca1_500.png'
plt.savefig(out_dir, dpi = 300)
co_Effs = np.transpose(pca.components_)
plt.show()