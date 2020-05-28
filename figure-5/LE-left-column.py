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

experiment_timestep = 0.1
t_experiment = np.linspace(0,experiment_timestep,10)

def iterate_experiment(y,y_perturbed,model):
    solution = scipy.integrate.odeint(model, y, t_experiment)
    perturbed_solution = scipy.integrate.odeint(model, y_perturbed, t_experiment)

    y_new = solution[-1,:]
    y_perturbed_new = perturbed_solution[-1,:]

    return (y_new, y_perturbed_new)

def replace_values(y_new, y_perturbed_new, d0):
    y_diff = y_new - y_perturbed_new
    d1 = npla.norm(y_diff)
    y_diff_normalized = y_diff*(d0/d1)
    y_perturbed = y_new + y_diff_normalized
    y = [k for k in y_new]
    return (y, y_perturbed, d1)

def main():
    # LE experiment parameters
    experiment_start_time = 10000
    d0=1e-6
    initial_perturbation_direction = np.array([1.0,1.0,1.0,0.0,0.0])
    initial_perturbation_direction_norm = npla.norm(initial_perturbation_direction)    
    normed_initial_perturbation_direction = initial_perturbation_direction*(d0/initial_perturbation_direction_norm)    
    num_experiment_steps = 1000
    buffer_iterations = 100

    # resolution
    h2_res = 201

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

    zeta = 0.1
    tau = 0.05

    theta1 = 0
    theta2 = 1
    phi1 = 0
    phi2 = 1

    # variable parameters
    y_heritabilities = np.linspace(0.04,1,h2_res)
    sigmayGs = [np.sqrt(h*sigmay**2) for h in y_heritabilities]

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
        'sigmax':sigmax,'sigmay':sigmay,'sigmaxG':sigmaxG,
        'zeta1':zeta,'zeta2':zeta,'tau1':tau,'tau2':tau,
        'theta1':theta1,'theta2':theta2,
        'phi1':phi1,'phi2':phi2
    }

    models = []
    for h2 in y_heritabilities:
        model_dict = baseline_kwargs
        model_dict['sigmayG'] = np.sqrt(h2*sigmay**2)
        system = SickOfEatingSystem(log=True,**model_dict)
        model = system.model
        models.append({'h2':h2,'model':model})

    # initial condition chosen because it is a stable equilibrium for the first y heritability - tau combination (N1, N2, P) are log(density)
    IC = [2.96374495, 2.41263081, 2.9048171, 0.50702777, 0.9992738]

    LE_experiment_data = []
    for h2_index, h2 in enumerate(y_heritabilities):
        print 'h2 =',h2
        model = models[h2_index]['model']

        print 'buffering dynamics'
        pre_buffer_t = np.linspace(0, experiment_start_time - buffer_iterations*experiment_timestep,100000)
        pre_buffer_soln = scipy.integrate.odeint(model, IC, pre_buffer_t)

        y = pre_buffer_soln[-1,:]
        y_perturbed = IC + normed_initial_perturbation_direction

        print 'buffering perturbation direction'
        for buffer_it in range(buffer_iterations):
            (y_new,y_perturbed_new) = iterate_experiment(y,y_perturbed,model)
            (y,y_perturbed,d1) = replace_values(y_new,y_perturbed_new,d0)

        L = np.zeros(num_experiment_steps)

        print 'running experiment'
        for step in range(num_experiment_steps):
            (y_new,y_perturbed_new) = iterate_experiment(y,y_perturbed,model)
            (y,y_perturbed,d1) = replace_values(y_new,y_perturbed_new,d0)
            L[step] = np.log(d1/d0)

        LE = np.mean(L)
        print LE

        LE_experiment_data.append({'h2':h2,'LE':LE,'IC':IC})

        IC = y

    filename = 'left-column.pickle'
    pickle.dump(LE_experiment_data,open(filename,'wb'))
    print 'saved data'

if __name__ == '__main__':
    main()



