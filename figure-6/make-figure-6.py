from __future__ import division

from SickOfEatingSystem import SickOfEatingSystem

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

h2 = 0.9
parameters = {
    'r1' : 1,
    'r2' : 1,
    'K1' : 100,
    'K2' : 100,
    'd' : 0.4,

    'b1' : 1,
    'b2' : 1,
    'm1' : 0.9,
    'm2' : 0.9,
    'c1' : 0.9,
    'c2' : 0.9,
    'alpha1' : 0.7,
    'alpha2' : 0.7,
    'beta1' : 0.95,
    'beta2' : 0.95,
    'gamma1' : 0.05,
    'gamma2' : 0.05,

    'sigmax' : 0.25,
    'sigmaxG' : 0.1,
    'sigmay' : 0.25,
    'sigmayG': np.sqrt(h2*0.25**2),

    'zeta1' : 0.1,
    'zeta2' : 0.1,
    'tau1' : 0.3,
    'tau2' : 0.3,

    'theta1' : 0,
    'theta2' : 1,
    'phi1' : 0,
    'phi2' : 1
}

system = SickOfEatingSystem(log=True,**parameters)
model = system.model
a1 = system.a1
a2 = system.a2
S1 = system.S1
S2 = system.S2
initial_condition = [1,1,1,0.3,0.6]
t = np.linspace(0,10000,100000)
soln = odeint(model,initial_condition,t)
M1 = soln[:,0]
N1 = np.exp(M1)
M2 = soln[:,1]
N2 = np.exp(M2)
Q = soln[:,2]
P = np.exp(Q)
x = soln[:,3]
y = soln[:,4]
prey1intake = a1(x)*N1
parasite1exposure = prey1intake*parameters['c1']
parasite1infection = parasite1exposure*S1(y)
prey2intake = a2(x)*N2
parasite2exposure = prey2intake*parameters['c2']
parasite2infection = parasite2exposure*S2(y)
intake = prey1intake/(prey1intake+prey2intake)
exposure = parasite1exposure/(parasite1exposure+parasite2exposure)
infection = parasite1infection/(parasite1infection+parasite2infection)

plt.figure()
plt.plot(t,N1,label='Prey 1')
plt.plot(t,N2,label='Prey 2')
plt.plot(t,P,label='Predator')
plt.xlim([9500,10000])
plt.legend(loc=0,fontsize=13,ncol=3)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Population Density',fontsize=15)
plt.savefig('densities.png',dpi=400)
# plt.show()
plt.close()

plt.figure()
plt.plot(t,x,label=r'Foraging Trait $\overline{x}$')
plt.plot(t,y,label=r'Immune Trait $\overline{y}$')
plt.xlim([9500,10000])
plt.legend(loc=0,fontsize=13,ncol=2)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Mean Trait Value',fontsize=15)
plt.savefig('traits.png',dpi=400)
# plt.show()
plt.close()

plt.figure()
plt.plot(intake[95000:],infection[95000:],linewidth=1)
left,right = plt.xlim()
down,up = plt.ylim()
plt.plot([-1,2],[-1,2],'k--',linewidth=0.5,zorder=-1)
plt.xlim(min(left,down),max(right,up))
plt.ylim(min(left,down),max(right,up))
plt.xlabel('Relative Prey Intake',fontsize=15)
plt.ylabel('Relative Parasite Infection',fontsize=15)
plt.savefig('intake-infection.png',dpi=400)
# plt.show()
plt.close()

plt.figure()
plt.plot(x[95000:],y[95000:],linewidth=1)
left,right = plt.xlim()
down,up = plt.ylim()
plt.plot([-1,2],[-1,2],'k--',linewidth=0.5,zorder=-1)
plt.xlim(min(left,down),max(right,up))
plt.ylim(min(left,down),max(right,up))
plt.xlabel(r'Foraging Trait $\overline{x}$',fontsize=15)
plt.ylabel(r'Immune Trait $\overline{y}$',fontsize=15)
plt.savefig('traits-phase.png',dpi=400)
# plt.show()
plt.close()

