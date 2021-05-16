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
group_size = 8
convex_hull_area_gs = convex_hull_area[trial_info[:,1]==group_size,:]
trial_info_gs = trial_info[trial_info[:,1]==group_size,:]

##The data is the 1000 frames before and after the loom
#let's look at the 300 frames after the loom
frames_to_use = list(range(1000,1300))
data_to_use = np.transpose(convex_hull_area_gs[:,frames_to_use])

##pre processing the data
scaler = StandardScaler()   
data_scaled = scaler.fit_transform(data_to_use)

# see how many components we need

pca = PCA().fit(data_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim([0,10])

# Let's see how it works with 3

num_pca_components = 3

pca = PCA(n_components=num_pca_components)
principalComponents = pca.fit_transform(data_scaled)  #find the PCAs or "loadings"
PCAFit = scaler.inverse_transform(pca.inverse_transform(principalComponents))

for i in range(num_pca_components):
    plt.plot(principalComponents[:,i])
    
plt.xlabel('Frame number')
plt.ylabel('Empirical Orthoginal Function (EOF)')

co_Effs = np.transpose(pca.components_)

#Let's look at the co-efficents
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

colors = plt.cm.viridis(np.linspace(0,1,6))
i = 0
for temp in np.unique(all_data[:,0]):

    #data_2_use = all_data[all_data[:,1]==group_size]
    ww = co_Effs[trial_info_gs[:,0]==temp]
    ax.scatter(ww[:,0], ww[:,1], c=colors[i], alpha = 0.5)
    i += 1


out_dir = '../../output/temp_collective/roi_figures/pca_coeff_gs_8.png'
fig.savefig(out_dir, dpi = 300)

plt.show()
#interesting that they sort of lie on a ring?