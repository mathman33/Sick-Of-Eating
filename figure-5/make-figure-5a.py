from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import pickle
import gc
import scipy.ndimage
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

from os import listdir
from os.path import isfile, join

pickle_filenames = [join('results', f) for f in listdir('results') if (isfile(join('results', f)) and 'h2=' in f)]
pickle_filenames.sort()

def tau_to_tau_index(tau):
    return (200/.4)*(tau - 0.05)

def h2_to_h2_index(h2):
    return (200/.96)*(h2 - 0.04)

data = []
m = 0
M = 0
for ind, f in enumerate(pickle_filenames):
    d = pickle.load(open(f,'r'))
    LEs = [i['LE'] for i in d[0]]
    m_temp = min(LEs)
    M_temp = max(LEs)
    m = min(m,m_temp)
    M = max(M,M_temp)
    data.append(LEs)

# make colormap based on min and max lyapunov exponents
cvals = [m,0,M]
colors = ['darkblue','white','darkred']
norm = plt.Normalize(m,M)
tuples = list(zip(map(norm,cvals),colors))
cmap = LinearSegmentedColormap.from_list("", tuples)

fig = plt.figure()
ax = plt.subplot(111)
ax.set_xticks([0,50,100,150,200])
ax.set_yticks([h2_to_h2_index(0.1*i) for i in range(1,11)])
ax.set_xticklabels([0.05,0.15,0.25,0.35,0.45])
ax.set_yticklabels([0.1*i for i in range(1,11)])
ax.set_xlabel(r'Strength of Immune Tradeoff $\tau_1,\tau_2$',fontsize=15)
ax.set_ylabel(r'Immune Heritability $h_y^2 = \frac{\sigma_{y,G}^2}{\sigma_y^2}$',fontsize=15)
plt.plot([0,200],[h2_to_h2_index(0.1),h2_to_h2_index(0.1)],'k--',linewidth=1)
plt.plot([0,200],[h2_to_h2_index(0.5),h2_to_h2_index(0.5)],'k--',linewidth=1)
plt.plot([0,200],[h2_to_h2_index(0.9),h2_to_h2_index(0.9)],'k--',linewidth=1)
plt.scatter(tau_to_tau_index(0.3),h2_to_h2_index(0.9),color='black',marker='X',s=75)
im = ax.imshow(data,cmap=cmap,origin='lower',vmin=m,vmax=M)
ax2 = fig.colorbar(im)
ax2.set_label('Lyapunov Exponent',fontsize=15,labelpad=-5)
# plt.show()
plt.savefig('results-left-column-continuation.png',dpi=400)
plt.close()
gc.collect();gc.collect()

