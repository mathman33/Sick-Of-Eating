from __future__ import division

from SickOfEatingSystem import SickOfEatingSystem
import json
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy.linalg as linalg


x_mesh = np.arange(-0.1,1.1,0.01)
y_mesh = np.arange(-0.1,1.1,0.01)
X, Y = np.meshgrid(x_mesh, y_mesh)

def get_plot_data(parameters, initial_conditions, trajectory_t=np.linspace(0,1000000,100000)):
    system = SickOfEatingSystem(slow_evo=True,**parameters)
    model = system.model

    # Find the nullclines
    nullclines = system.get_nullclines(X,Y)

    # Find the stable and unstable equilibria, solve for
    # the separatrices (stable manifolds of the saddles)
    eq = system.get_slow_evo_model_equilibria()
    [st_eq, unst_eq] = system.determine_stability(eq)
    seps = system.get_separatrices(unst_eq)

    trajectories = []
    for ind, ic in enumerate(initial_conditions):
        trajectories.append(odeint(model,ic,trajectory_t))

    return (nullclines,st_eq,unst_eq,seps,trajectories)

def plot(ax, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices):
    (xsurface,ysurface) = nullclines
    ax.contour(X,Y,xsurface,[0],linewidths=3,colors="c",zorder=1)
    ax.contour(X,Y,ysurface,[0],linewidths=3,colors="m",zorder=1)
    for i in st_eq:
        ax.scatter(i[0],i[1],marker='o',s=75,edgecolors='k',facecolors='k',zorder=3)
    for i in unst_eq:
        ax.scatter(i[0],i[1],marker='o',s=75,edgecolors='k',facecolors='none',zorder=3)
    for i in separatrix_trajectories:
        ax.plot(i[:,0],i[:,1],'k--',linewidth=1,zorder=2)
    for ind, t in enumerate(trajectories):
        ax.plot(t[:,0],t[:,1],'b-')
        for (a,speed) in arrow_indices[ind]:
            arrow_x = t[a[0],0]
            arrow_y = t[a[0],1]
            arrow_dest_x = t[a[1],0]
            arrow_dest_y = t[a[1],1]
            arrow_dx = arrow_dest_x - arrow_x
            arrow_dy = arrow_dest_y - arrow_y
            if abs(arrow_dy) > abs(arrow_dx):
                ax.arrow(arrow_x,arrow_y,arrow_dx,arrow_dy,width=0.004,color='b',head_width=0.06)
            else:
                ax.arrow(arrow_x,arrow_y,arrow_dx,arrow_dy,width=0.005,color='b',head_width=0.08,head_length=0.08)
            if speed == 'fast':
                if abs(arrow_dy) > abs(arrow_dx):
                    ax.arrow(arrow_x+0.5*arrow_dx,arrow_y+0.5*arrow_dy,arrow_dx,arrow_dy,width=0.004,color='b',head_width=0.06)
                else:
                    ax.arrow(arrow_x+0.5*arrow_dx,arrow_y+0.5*arrow_dy,arrow_dx,arrow_dy,width=0.005,color='b',head_width=0.08,head_length=0.08)

    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])


# import parameters
baseline_parameters_filename = "baseline-parameters.json"
with open(baseline_parameters_filename,'r') as f:
    baseline_parameters = json.load(f)

# initialize the figure
fig, axs = plt.subplots(4,3,sharex='col',sharey='row',gridspec_kw={'hspace':0.15,'wspace':0.125},figsize=(9.6,7.2))
# name each of the twelve subplots
(ax11, ax12, ax13), (ax21, ax22, ax23), (ax31, ax32, ax33), (ax41, ax42, ax43) = axs


##### ax11 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmaxG'] = 0.005
parameters['sigmayG'] = 0.25

initial_conditions = [[0.35,0.5],[0.35,0.8],[0.65,0.2],[0.65,0.5]]
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions)

arrow_indices = [[([3,6],'fast'),([100,200],'slow')],[([860,870],'fast')],[([860,870],'fast')],[([3,6],'fast'),([100,200],'slow')]]
plot(ax11, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax11'
del parameters

##### ax12 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmaxG'] = 0.25
parameters['sigmayG'] = 0.005

initial_conditions = [[0.35,0.4],[0.35,0.8],[0.65,0.2],[0.65,0.6]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([2,4],'fast')],[([2,4],'fast'),([800000,1600000],'slow')],[([2,4],'fast'),([800000,1600000],'slow')],[([2,4],'fast')]]
plot(ax12, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax12'
del parameters

##### ax13 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmayG'] = 0.1
parameters['sigmaxG'] = 0.1
parameters['m1'] = 0.1
parameters['m2'] = 0.1
parameters['c1'] = 0.1
parameters['c2'] = 0.1

initial_conditions = [[0.35,0.4],[0.35,0.8],[0.65,0.2],[0.65,0.6],[0.46,0.4],[0.46,0.6],[0.54,0.4],[0.54,0.6]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions,trajectory_t)

arrow_indices = [[([10,25],'fast')],[([10,25],'fast'),([640000,1300000],'slow')],[([10,25],'fast'),([640000,1300000],'slow')],[([10,25],'fast')],[([80000,220000],'slow')],[([80000,220000],'slow')],[],[]]
plot(ax13, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax13'
del parameters

##### ax21 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmaxG'] = 0.005
parameters['sigmayG'] = 0.25

initial_conditions = [[0.1,0.1],[0.9,0.9]]
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions)

arrow_indices = [[([116,120],'fast'),([1000,2000],'slow')],[([116,120],'fast'),([1000,2000],'slow')]]
plot(ax21, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax21'
del parameters

##### ax22 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmaxG'] = 0.25
parameters['sigmayG'] = 0.005

initial_conditions = [[0.1,0.35],[0.1,0.65],[0.9,0.35],[0.9,0.65]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([10,20],'fast')],[([10,20],'fast'),([100000,200000],'slow')],[([10,20],'fast'),([100000,200000],'slow')],[([10,20],'fast')]]
plot(ax22, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax22'
del parameters

##### ax23 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 0.01
parameters['tau2'] = 0.01
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmayG'] = 0.1
parameters['sigmaxG'] = 0.1
parameters['m1'] = 0.1
parameters['m2'] = 0.1
parameters['c1'] = 0.1
parameters['c2'] = 0.1

initial_conditions = [[0.1,0.35],[0.1,0.65],[0.9,0.35],[0.9,0.65]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([50,115],'fast')],[([50,115],'fast'),([100000,200000],'slow')],[([50,115],'fast'),([100000,200000],'slow')],[([50,115],'fast')]]
plot(ax23, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax23'
del parameters

##### ax31 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmaxG'] = 0.005
parameters['sigmayG'] = 0.25

initial_conditions = [[0.45,0.05],[0.3,0.9],[0.55,0.95],[0.7,0.1]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions,trajectory_t)

arrow_indices = [[([0,70],'fast')],[([10,35],'fast'),([4000,5000],'slow')],[([0,70],'fast')],[([10,35],'fast'),([4000,5000],'slow')]]
plot(ax31, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax31'
del parameters

##### ax32 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmaxG'] = 0.25
parameters['sigmayG'] = 0.005

initial_conditions = [[0.45,0.3],[0.3,0.9],[0.55,0.7],[0.7,0.1]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions,trajectory_t)

arrow_indices = [[([5,7],'fast')],[([0,3],'fast'),([60000,100000],'slow')],[([5,7],'fast')],[([0,3],'fast'),([60000,100000],'slow')]]
plot(ax32, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax32'
del parameters

##### ax33 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 0.01
parameters['zeta2'] = 0.01
parameters['sigmayG'] = 0.1
parameters['sigmaxG'] = 0.1
parameters['m1'] = 0.1
parameters['m2'] = 0.1
parameters['c1'] = 0.1
parameters['c2'] = 0.1

initial_conditions = [[0.4,0.2],[0.4,0.9],[0.6,0.1],[0.6,0.8],[0.46,0.1],[0.46,0.9],[0.54,0.1],[0.54,0.9]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([20,35],'fast')],[([20,35],'fast'),([10000,25000],'slow')],[([20,35],'fast'),([10000,25000],'slow')],[([20,35],'fast')],[([10000,35000],'slow')],[([10000,35000],'slow')],[],[]]
plot(ax33, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax33'
del parameters

##### ax41 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmaxG'] = 0.005
parameters['sigmayG'] = 0.25

initial_conditions = [[0.1,0.1],[0.9,0.9]]
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions)

arrow_indices = [[([1,2],'fast'),([1000,2000],'slow'),([6150,6225],'fast')],[([1,2],'fast'),([1000,2000],'slow'),([6150,6225],'fast')]]
plot(ax41, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax41'
del parameters

##### ax42 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmaxG'] = 0.25
parameters['sigmayG'] = 0.005

initial_conditions = [[0.1,0.1],[0.9,0.1],[0.1,0.9],[0.9,0.9]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([5,15],'fast')],[([5,15],'fast'),([50000,150000],'slow')],[([5,15],'fast'),([50000,150000],'slow')],[([5,15],'fast')]]
plot(ax42, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax42'
del parameters

##### ax43 #####
parameters = baseline_parameters.copy()
parameters['tau1'] = 1
parameters['tau2'] = 1
parameters['zeta1'] = 1
parameters['zeta2'] = 1
parameters['sigmayG'] = 0.1
parameters['sigmaxG'] = 0.1
parameters['m1'] = 0.1
parameters['m2'] = 0.1
parameters['c1'] = 0.1
parameters['c2'] = 0.1

initial_conditions = [[0.1,0.1],[0.9,0.1],[0.1,0.9],[0.9,0.9]]
trajectory_t = np.linspace(0,2000000,4000000)
(nullclines,st_eq,unst_eq,separatrix_trajectories,trajectories) = get_plot_data(parameters,initial_conditions, trajectory_t)

arrow_indices = [[([40,90],'fast')],[([40,90],'fast'),([20000,55000],'slow')],[([40,90],'fast'),([20000,55000],'slow')],[([40,90],'fast')]]
plot(ax43, nullclines, st_eq, unst_eq, separatrix_trajectories, trajectories, arrow_indices)
print 'done with ax43'
del parameters





##### labels, titles, headers, etc. #####
print 'setting axes labels and titles'
ax11.set_ylabel(r'$\overline{y}$',fontsize=15,rotation=0,labelpad=8)
ax21.set_ylabel(r'$\overline{y}$',fontsize=15,rotation=0,labelpad=8)
ax31.set_ylabel(r'$\overline{y}$',fontsize=15,rotation=0,labelpad=8)
ax41.set_ylabel(r'$\overline{y}$',fontsize=15,rotation=0,labelpad=8)
ax41.set_xlabel(r'$\overline{x}$',fontsize=15,rotation=0)
ax42.set_xlabel(r'$\overline{x}$',fontsize=15,rotation=0)
ax43.set_xlabel(r'$\overline{x}$',fontsize=15,rotation=0)
ax11.set_title(r'$h_y^2 \gg h_x^2$',fontsize=15)
ax12.set_title(r'$h_x^2 \gg h_y^2$',fontsize=15)
ax13.set_title(r'$0<m_i,c_i\ll1$, $i=1,2$',fontsize=15)
ax132 = ax13.twinx()
ax132.set_ylabel('Strong immune\nstrong foraging',fontsize=12)
ax132.set_yticks([])
ax232 = ax23.twinx()
ax232.set_ylabel('Strong immune\nweak foraging',fontsize=12)
ax232.set_yticks([])
ax332 = ax33.twinx()
ax332.set_ylabel('Weak immune\nstrong foraging',fontsize=12)
ax332.set_yticks([])
ax432 = ax43.twinx()
ax432.set_ylabel('Weak immune\nweak foraging',fontsize=12)
ax432.set_yticks([])






##### saving figure #####
print 'saving figure: three-timescale-trajectories.png'
# plt.show()
plt.savefig('three-timescale-trajectories.png',dpi=400)
plt.close()

