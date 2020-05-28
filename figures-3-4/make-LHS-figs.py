from __future__ import division

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d,spline
from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import os

from scipy import stats

gist_stern = mpl.cm.get_cmap('gist_stern')
truncated_gist_stern = LinearSegmentedColormap.from_list(
    'trunc(gist_stern,0,0.8)',
    gist_stern(np.linspace(0,0.8,100)))

def plot_fig3(pickle_filename,title,png_filename):
    with open(pickle_filename,'rb') as f:
        data = pickle.load(f)

    x = data['x']
    y = data['y']
    P = data['P']
    plt.figure()
    plt.scatter(x,y,s=1,c=data['P'],cmap=truncated_gist_stern,vmin=0)
    plt.title(title,fontsize=15)
    plt.xlabel(r'Foraging Trait $\overline{x}$',fontsize=15)
    plt.ylabel(r'Immune Trait $\overline{y}$',fontsize=15)
    cb = plt.colorbar()
    cb.set_label('Predator Density',fontsize=15)
    # plt.show()
    plt.savefig(os.path.join('fig3',png_filename),dpi=400)
    plt.close()

def plot_fig4(pickle_filename,title,png_filename):
    with open(pickle_filename,'rb') as f:
        data = pickle.load(f)

    intake = np.array(data['intake1'])/(np.array(data['intake1']) + np.array(data['intake2']))
    exposure = np.array(data['exposure1'])/(np.array(data['exposure1']) + np.array(data['exposure2']))
    infection = np.array(data['infection1'])/(np.array(data['infection1']) + np.array(data['infection2']))

    intake_sorted1,exposure_sorted1,infection_sorted1 = zip(*sorted(zip(intake,exposure,infection)))
    exposure_sorted2,infection_sorted2 = zip(*sorted(zip(exposure,infection)))

    fig, axs = plt.subplots(2,2,gridspec_kw={'hspace':0.15,'wspace':0.125},figsize=(9.6,7.2))
    (ax11,ax12),(ax21,ax22) = axs
    cbax = fig.add_axes([ax22.get_position().x0,ax22.get_position().y0,ax22.get_position().x1-ax22.get_position().x0,0.05])

    tck_intake_infection = interpolate.splrep(intake_sorted1,infection_sorted1,s=100)
    tck_intake_exposure = interpolate.splrep(intake_sorted1,exposure_sorted1,s=100)
    tck_exposure_infection = interpolate.splrep(exposure_sorted2,infection_sorted2,s=100)
    X = np.linspace(0,1,100)
    Y_intake_infection = interpolate.splev(X,tck_intake_infection,der=0)
    Y_intake_exposure = interpolate.splev(X,tck_intake_exposure,der=0)
    Y_exposure_infection = interpolate.splev(X,tck_exposure_infection,der=0)

    sc = ax11.scatter(intake,infection,s=1,c=data['P'],cmap=truncated_gist_stern)
    ax11.set_xlim([0,1])
    ax11.set_ylim([0,1])
    ax21.set_xlim([0,1])
    ax21.set_ylim([0,1])
    ax12.set_xlim([0,1])
    ax12.set_ylim([0,1])
    ax11.set_xticks([])
    ax11.set_ylabel('Relative Parasite Infection', fontsize=15)
    ax11.plot(X,Y_intake_infection,'k--')
    ax12.set_yticks([])
    ax12.scatter(exposure,infection,s=1,c=data['P'],cmap=truncated_gist_stern)
    ax12.set_xlabel('Relative Parasite Exposure', fontsize=15)
    ax12.plot(X,Y_exposure_infection,'k--')
    ax21.scatter(intake,exposure,s=1,c=data['P'],cmap=truncated_gist_stern)
    ax21.set_xlabel('Relative Prey Intake', fontsize=15)
    ax21.set_ylabel('Relative Parasite Exposure', fontsize=15)
    ax21.plot(X,Y_intake_exposure,'k--')
    ax22.axis('off')
    ax22.set_title(title,y=0.45,fontsize=20)
    cbar = mpl.colorbar.ColorbarBase(ax=cbax,norm=plt.Normalize(0,max(data['P'])),cmap=truncated_gist_stern,orientation='horizontal')
    cbar.set_label('Predator Density',fontsize=15)
    # plt.show()
    plt.savefig(os.path.join('fig4',png_filename),dpi=400)
    plt.close()

plot_fig3('strong-foraging-strong-immune-LHS-results.pickle','Strong Foraging Tradeoff\nStrong Immune Tradeoff','strong-strong-LHS-xy.png')
plot_fig3('weak-foraging-strong-immune-LHS-results.pickle','Weak Foraging Tradeoff\nStrong Immune Tradeoff','weak-strong-LHS-xy.png')
plot_fig3('strong-foraging-weak-immune-LHS-results.pickle','Strong Foraging Tradeoff\nWeak Immune Tradeoff','strong-weak-LHS-xy.png')
plot_fig3('weak-foraging-weak-immune-LHS-results.pickle','Weak Foraging Tradeoff\nWeak Immune Tradeoff','weak-weak-LHS-xy.png')

plot_fig4('strong-foraging-strong-immune-LHS-results.pickle','Strong Foraging Tradeoff\nStrong Immune Tradeoff','strong-strong-LHS-iei.png')
plot_fig4('weak-foraging-strong-immune-LHS-results.pickle','Weak Foraging Tradeoff\nStrong Immune Tradeoff','weak-strong-LHS-iei.png')
plot_fig4('strong-foraging-weak-immune-LHS-results.pickle','Strong Foraging Tradeoff\nWeak Immune Tradeoff','strong-weak-LHS-iei.png')
plot_fig4('weak-foraging-weak-immune-LHS-results.pickle','Weak Foraging Tradeoff\nWeak Immune Tradeoff','weak-weak-LHS-iei.png')

