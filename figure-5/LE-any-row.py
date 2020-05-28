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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-h2", type=float, dest="h2")
    parser.add_argument("-M10", type=float, dest="M10")
    parser.add_argument("-M20", type=float, dest="M20")
    parser.add_argument("-Q0", type=float, dest="Q0")
    parser.add_argument("-x0", type=float, dest="x0")
    parser.add_argument("-y0", type=float, dest="y0")
    return parser.parse_args()

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
    args = parse_args()
    h2 = args.h2
    M10 = args.M10
    M20 = args.M20
    Q0 = args.Q0
    x0 = args.x0
    y0 = args.y0

    # LE experiment parameters
    experiment_start_time = 10000
    d0=1e-6
    initial_perturbation_direction = np.array([1.0,1.0,1.0,0.0,0.0])
    initial_perturbation_direction_norm = npla.norm(initial_perturbation_direction)    
    normed_initial_perturbation_direction = initial_perturbation_direction*(d0/initial_perturbation_direction_norm)    
    num_experiment_steps = 10000
    buffer_iterations = 100

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

    theta1 = 0
    theta2 = 1
    phi1 = 0
    phi2 = 1

    # variable parameters
    taus = np.linspace(0.05,0.45,tau_res)

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
    for tau in taus:
        model_dict = baseline_kwargs
        model_dict['tau1'] = tau
        model_dict['tau2'] = tau
        system = SickOfEatingSystem(log=True,**model_dict)
        model = system.model
        models.append({'tau':tau,'model':model})

    # initial condition chosen because it is a stable equilibrium for the first y heritability - tau combination (N1, N2, P) are log(density)
    IC = [M10,M20,Q0,x0,y0]

    LE_experiment_data = []
    for tau_index, tau in enumerate(taus):
        print 'tau =',tau
        model = models[tau_index]['model']

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

        LE_experiment_data.append({'tau':tau,'LE':LE,'IC':IC})

        previous_final_state = y
        if tau_index == 0:
            first_final_state_in_previous_column = y

        IC = y

    filename = os.path.join("results",'h2=%5.5f.pickle' % h2)
    pickle.dump(LE_experiment_data,open(filename,'wb'))
    print 'saved data'

if __name__ == '__main__':
    main()



