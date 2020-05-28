from __future__ import division

import numpy as np
import numpy.linalg as npla
import scipy.spatial.distance
import scipy.integrate
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from SickOfEatingSystem import SickOfEatingSystem
import argparse

def in_equilibrium(LIST):
    return LIST[-1] == LIST[-2]

def main(h2):
    initial_initial_condition = [2.96374495, 2.41263081, 2.9048171, 0.50702777, 0.9992738]

    # resolution
    tau_res = 201

    # constant parameters
    r = 1
    K = 100
    d = 0.4

    b = 1
    m = 0.9
    c = 0.9
    alpha = 0.7
    beta = 0.95
    gamma = 0.05

    sigmax = 0.25
    sigmaxG = 0.1
    sigmay = 0.25
    sigmayG = np.sqrt(h2*sigmay**2)

    zeta = 0.1
    taus = np.linspace(0.05,0.45,tau_res)

    theta1 = 0
    theta2 = 1
    phi1 = 0
    phi2 = 1

    baseline_kwargs = {
        'r1':r,'r2':r,
        'K1':K,'K2':K,
        'd':d,
        'b1':b,'b2':b,
        'm1':m,'m2':m,
        'c1':c,'c2':c,
        'alpha1':alpha,'alpha2':alpha,
        'beta1':beta,'beta2':beta,
        'gamma1':gamma,'gamma2':gamma,
        'sigmax':sigmax,'sigmay':sigmay,'sigmaxG':sigmaxG,'sigmayG':sigmayG,
        'zeta1':zeta,'zeta2':zeta,
        'theta1':theta1,'theta2':theta2,
        'phi1':phi1,'phi2':phi2
    }

    models = []
    IC = initial_initial_condition
    l = 10000
    t = np.linspace(0,l,l)
    extrema = []
    for index,tau in enumerate(taus):
        print index,tau
        model_dict = baseline_kwargs
        model_dict['tau1'] = tau
        model_dict['tau2'] = tau
        system = SickOfEatingSystem(log=True,**model_dict)
        model = system.model
        solution = scipy.integrate.odeint(model, IC, t)
        IC = solution[-1,:]
        last_half_x_solution = solution[int(l/2):,3]
        if in_equilibrium(last_half_x_solution):
            extrema.append([last_half_x_solution[-1]])
        else:
            temp_extrema = []
            for i in range(1,len(last_half_x_solution)-1):
                past = last_half_x_solution[i-1]
                pres = last_half_x_solution[i]
                futu = last_half_x_solution[i+1]
                if (pres < past and pres < futu) or (pres > past and pres > futu):
                    if not any([abs(pres-a)<1e-4 for a in temp_extrema]):
                        temp_extrema.append(pres)
            extrema.append(temp_extrema)

    plt.figure()
    for index,tau in enumerate(taus):
        for e in extrema[index]:
            plt.scatter(h2,e,s=1,c='k')
    plt.xlabel(r'Strength of Immune Tradeoff, $\tau_1$, $\tau_2$',fontsize=13,labelpad=2)
    plt.ylabel(r'Extrema of Foraging Trait $\bar{x}$ Trajectory',fontsize=15)
    plt.ylim([0,1])
    # plt.show()
    filename = 'h2=' + str(h2).remove('.') + '.png'
    plt.savefig(filename,dpi=400)
    plt.close()

if __name__ == '__main__':
    main(0.1)
    main(0.5)
    main(0.9)



