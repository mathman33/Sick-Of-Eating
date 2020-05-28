from __future__ import division

from SickOfEatingSystem import SickOfEatingSystem
import json
import numpy as np
from scipy.integrate import odeint
import numpy.linalg as linalg
import pickle

# Seed the RNG
MAX_NUMPY_SEED = 4294967295
SEED = 'Sick of Eating 2020'
NUMBER = int(''.join([str(ord(i)) for i in SEED])) % MAX_NUMPY_SEED
np.random.seed(NUMBER)

LHS_parameters_filename = "LHS-parameters.json"
with open(LHS_parameters_filename,'r') as f:
    LHS_parameters = json.load(f)

baseline_parameters = {}
for key, value in LHS_parameters.items():
    if type(value) != list:
        baseline_parameters[key] = value

LHS_resolution = 4001
# parameters boundaries...
K1s = np.linspace(LHS_parameters['K1'][0],LHS_parameters['K1'][1],LHS_resolution)
K2s = np.linspace(LHS_parameters['K2'][0],LHS_parameters['K2'][1],LHS_resolution)
b1s = np.linspace(LHS_parameters['b1'][0],LHS_parameters['b1'][1],LHS_resolution)
b2s = np.linspace(LHS_parameters['b2'][0],LHS_parameters['b2'][1],LHS_resolution)
m1s = np.linspace(LHS_parameters['m1'][0],LHS_parameters['m1'][1],LHS_resolution)
m2s = np.linspace(LHS_parameters['m2'][0],LHS_parameters['m2'][1],LHS_resolution)
c1s = np.linspace(LHS_parameters['c1'][0],LHS_parameters['c1'][1],LHS_resolution)
c2s = np.linspace(LHS_parameters['c2'][0],LHS_parameters['c2'][1],LHS_resolution)
r1s = np.linspace(LHS_parameters['r1'][0],LHS_parameters['r1'][1],LHS_resolution)
r2s = np.linspace(LHS_parameters['r2'][0],LHS_parameters['r2'][1],LHS_resolution)
ds = np.linspace(LHS_parameters['d'][0],LHS_parameters['d'][1],LHS_resolution)
alpha1s = np.linspace(LHS_parameters['alpha1'][0],LHS_parameters['alpha1'][1],LHS_resolution)
alpha2s = np.linspace(LHS_parameters['alpha2'][0],LHS_parameters['alpha2'][1],LHS_resolution)
beta1s = np.linspace(LHS_parameters['beta1'][0],LHS_parameters['beta1'][1],LHS_resolution)
beta2s = np.linspace(LHS_parameters['beta2'][0],LHS_parameters['beta2'][1],LHS_resolution)
gamma1s = np.linspace(LHS_parameters['gamma1'][0],LHS_parameters['gamma1'][1],LHS_resolution)
gamma2s = np.linspace(LHS_parameters['gamma2'][0],LHS_parameters['gamma2'][1],LHS_resolution)
# initial conditions
x0s = np.linspace(0,1,LHS_resolution)
y0s = np.linspace(0,1,LHS_resolution)

# randomly sample between boundaries, then permute the samples.
K1s = list(np.random.permutation([np.random.random()*(K1s[i]-K1s[i-1]) + K1s[i-1] for i in range(1,LHS_resolution)]))
K2s = list(np.random.permutation([np.random.random()*(K2s[i]-K2s[i-1]) + K2s[i-1] for i in range(1,LHS_resolution)]))
b1s = list(np.random.permutation([np.random.random()*(b1s[i]-b1s[i-1]) + b1s[i-1] for i in range(1,LHS_resolution)]))
b2s = list(np.random.permutation([np.random.random()*(b2s[i]-b2s[i-1]) + b2s[i-1] for i in range(1,LHS_resolution)]))
m1s = list(np.random.permutation([np.random.random()*(m1s[i]-m1s[i-1]) + m1s[i-1] for i in range(1,LHS_resolution)]))
m2s = list(np.random.permutation([np.random.random()*(m2s[i]-m2s[i-1]) + m2s[i-1] for i in range(1,LHS_resolution)]))
c1s = list(np.random.permutation([np.random.random()*(c1s[i]-c1s[i-1]) + c1s[i-1] for i in range(1,LHS_resolution)]))
c2s = list(np.random.permutation([np.random.random()*(c2s[i]-c2s[i-1]) + c2s[i-1] for i in range(1,LHS_resolution)]))
r1s = list(np.random.permutation([np.random.random()*(r1s[i]-r1s[i-1]) + r1s[i-1] for i in range(1,LHS_resolution)]))
r2s = list(np.random.permutation([np.random.random()*(r2s[i]-r2s[i-1]) + r2s[i-1] for i in range(1,LHS_resolution)]))
ds = list(np.random.permutation([np.random.random()*(ds[i]-ds[i-1]) + ds[i-1] for i in range(1,LHS_resolution)]))
alpha1s = list(np.random.permutation([np.random.random()*(alpha1s[i]-alpha1s[i-1]) + alpha1s[i-1] for i in range(1,LHS_resolution)]))
alpha2s = list(np.random.permutation([np.random.random()*(alpha2s[i]-alpha2s[i-1]) + alpha2s[i-1] for i in range(1,LHS_resolution)]))
beta1s = list(np.random.permutation([np.random.random()*(beta1s[i]-beta1s[i-1]) + beta1s[i-1] for i in range(1,LHS_resolution)]))
beta2s = list(np.random.permutation([np.random.random()*(beta2s[i]-beta2s[i-1]) + beta2s[i-1] for i in range(1,LHS_resolution)]))
gamma1s = list(np.random.permutation([np.random.random()*(gamma1s[i]-gamma1s[i-1]) + gamma1s[i-1] for i in range(1,LHS_resolution)]))
gamma2s = list(np.random.permutation([np.random.random()*(gamma2s[i]-gamma2s[i-1]) + gamma2s[i-1] for i in range(1,LHS_resolution)]))
x0s = list(np.random.permutation([np.random.random()*(x0s[i]-x0s[i-1]) + x0s[i-1] for i in range(1,LHS_resolution)]))
y0s = list(np.random.permutation([np.random.random()*(y0s[i]-y0s[i-1]) + y0s[i-1] for i in range(1,LHS_resolution)]))

experiments = [
    [0.01,0.01,0.01,0.01,'strong-foraging-strong-immune-LHS-results.pickle'],
    [1,1,0.01,0.01,'weak-foraging-strong-immune-LHS-results.pickle'],
    [0.01,0.01,1,1,'strong-foraging-weak-immune-LHS-results.pickle'],
    [1,1,1,1,'weak-foraging-weak-immune-LHS-results.pickle']
]
for [zeta1, zeta2, tau1, tau2, filename] in experiments:
    ##### Strong foraging, strong immune #####
    parameters = baseline_parameters.copy()
    parameters['zeta1'] = zeta1
    parameters['zeta2'] = zeta2
    parameters['tau1'] = tau1
    parameters['tau2'] = tau2

    results = {
        'x': [],
        'y': [],
        'N1': [],
        'N2': [],
        'P': [],
        'intake1': [],
        'intake2': [],
        'infection1': [],
        'infection2': [],
        'exposure1': [],
        'exposure2': []
    }
    skipped_simulations = 0
    for i in range(LHS_resolution-1):
        if i % 25 == 0:
            print 'Running through Latin Hypercube Sample... ', i, '/', (LHS_resolution-1), '(',skipped_simulations,'simulations skipped )'
        parameters['K1'] = K1s[i]
        parameters['K2'] = K2s[i]
        parameters['b1'] = b1s[i]
        parameters['b2'] = b2s[i]
        parameters['m1'] = m1s[i]
        parameters['m2'] = m2s[i]
        parameters['c1'] = c1s[i]
        parameters['c2'] = c2s[i]
        parameters['r1'] = r1s[i]
        parameters['r2'] = r2s[i]
        parameters['d'] = ds[i]
        parameters['alpha1'] = alpha1s[i]
        parameters['alpha2'] = alpha2s[i]
        parameters['beta1'] = beta1s[i]
        parameters['beta2'] = beta2s[i]
        parameters['gamma1'] = gamma1s[i]
        parameters['gamma2'] = gamma2s[i]

        IC = [x0s[i],y0s[i]]

        system = SickOfEatingSystem(slow_evo=True,**parameters)
        model = system.model

        try:
            soln = odeint(model, IC, np.linspace(0,1000000,100000))
            evo_state = soln[-1:]
            x = evo_state[-1,0]
            y = evo_state[-1,1]
            eco_state = system.saturated_equilibrium([x,y])
            [N1, N2, P] = eco_state
            intake1 = system.a1(x)*N1
            intake2 = system.a2(x)*N2
            exposure1 = intake1*system.c1
            exposure2 = intake2*system.c2
            infection1 = exposure1*system.S1(y)
            infection2 = exposure2*system.S2(y)
            results['x'].append(x)
            results['y'].append(y)
            results['N1'].append(N1)
            results['N2'].append(N2)
            results['P'].append(P)
            results['intake1'].append(intake1)
            results['intake2'].append(intake2)
            results['exposure1'].append(exposure1)
            results['exposure2'].append(exposure2)
            results['infection1'].append(infection1)
            results['infection2'].append(infection2)
        except:
            skipped_simulations += 1
            for key, value in parameters.items():
                print key, value
            print 'initial condition', IC

    print skipped_simulations,'simulations skipped due to numerical computation complications...'
    print 'saving results...'
    with open(filename,'wb') as f:
        pickle.dump(results,f)



















