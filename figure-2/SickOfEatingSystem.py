from __future__ import division

import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import norm
from scipy.integrate import odeint

class SickOfEatingSystem:
    def __init__(self,
                 r1=0,r2=0,
                 K1=0,K2=0,
                 zeta1=0,zeta2=0,
                 alpha1=0,alpha2=0,
                 sigmax=0,sigmaxG=0,
                 sigmay=0,sigmayG=0,
                 theta1=0,theta2=0,
                 b1=0,b2=0,
                 d=0,
                 m1=0,m2=0,
                 c1=0,c2=0,
                 beta1=0,beta2=0,
                 gamma1=0,gamma2=0,
                 tau1=0,tau2=0,
                 phi1=0,phi2=0,
                 log=False,slow_evo=False):
        self.r1 = r1
        self.r2 = r2
        self.K1 = K1
        self.K2 = K2
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sigmax = sigmax
        self.sigmaxG = sigmaxG
        self.sigmay = sigmay
        self.sigmayG = sigmayG
        self.theta1 = theta1
        self.theta2 = theta2
        self.b1 = b1
        self.b2 = b2
        self.d = d
        self.m1 = m1
        self.m2 = m2
        self.c1 = c1
        self.c2 = c2
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.tau1 = tau1
        self.tau2 = tau2
        self.phi1 = phi1
        self.phi2 = phi2

        self.A1 = sigmax**2 + zeta1**2
        self.A2 = sigmax**2 + zeta2**2
        self.B1 = sigmay**2 + tau1**2
        self.B2 = sigmay**2 + tau2**2

        self.log = log

        self.slow_evo = slow_evo

        if self.slow_evo:
            self.model = self.make_slow_evo_model()
        else:
            self.model = self.make_full_model()

    def S1(self,y):
        return self.beta1 - (self.beta1 - self.gamma1)*(self.tau1/np.sqrt(self.B1))*np.exp((-(y-self.phi1)**2)/(2*self.B1))
    def S2(self,y):
        return self.beta2 - (self.beta2 - self.gamma2)*(self.tau2/np.sqrt(self.B2))*np.exp((-(y-self.phi2)**2)/(2*self.B2))
    def a1(self,x):
        return (self.alpha1*self.zeta1/np.sqrt(self.A1))*np.exp((-(x-self.theta1)**2)/(2*self.A1))
    def a2(self,x):
        return (self.alpha2*self.zeta2/np.sqrt(self.A2))*np.exp((-(x-self.theta2)**2)/(2*self.A2))
    def make_full_model(self):
        def model(Y,t):
            x = Y[3]
            y = Y[4]
            a1x = self.a1(x)
            a2x = self.a2(x)
            S1y = self.S1(y)
            S2y = self.S2(y)

            f = [0]*5

            if self.log:
                N1 = np.exp(Y[0])
                N2 = np.exp(Y[1])
                P = np.exp(Y[2])
                f[0] = self.r1*(1 - N1/self.K1) - a1x*P
                f[1] = self.r2*(1 - N2/self.K2) - a2x*P
                f[2] = (self.b1 - self.m1*self.c1*S1y)*a1x*N1 + (self.b2 - self.m2*self.c2*S2y)*a2x*N2 - self.d
            else:
                N1 = Y[0]
                N2 = Y[1]
                P = Y[2]
                f[0] = N1*(self.r1*(1 - N1/self.K1) - a1x*P)
                f[1] = N2*(self.r2*(1 - N2.self.K2) - a2x*P)
                f[2] = P*((self.b1 - self.m1*self.c1*S1y)*a1x*N1 + (self.b2 - self.m2*self.c2*S2y)*a2x*N2 - self.d)
            f[3] = (self.sigmaxG**2)*((self.b1 - self.m1*self.c1*S1y)*a1x*N1*((self.theta1-x)/self.A1) + (self.b2 - self.m2*self.c2*S2y)*a2x*N2*((self.theta2-x)/self.A2))
            f[4] = (self.sigmayG**2)*(self.m1*self.c1*a1x*N1*(self.beta1-S1y)*((self.phi1-y)/self.B1) + self.m2*self.c2*a2x*N2*(self.beta2-S2y)*((self.phi2-y)/self.B2))

            return f
        return model

    def per_capita_grow_rate(self,N1,N2,P,x,y):
        f1 = self.r1*(1 - N1/self.K1) - self.a1(x)*P
        f2 = self.r2*(1 - N2/self.K2) - self.a2(x)*P
        f3 = (self.b1 - self.m1*self.c1*self.S1(y))*self.a1(x)*N1 + (self.b2 - self.m2*self.c2*self.S2(y))*self.a2(x)*N2 - self.d
        return (f1,f2,f3)

    def not_negative(self,a,b,c):
        return ((a >= 0) and (b >= 0) and (c >= 0))

    # Checks if all three values are non-positive (allowed error is 1e-10)
    def not_positive(self,v,cutoff):
        return ((v[0] <= cutoff) and (v[1] <= cutoff) and (v[2] <= cutoff))

    def saturated_equilibrium(self,v):
        x = v[0]
        y = v[1]

        # Coexistence Equilibrium
        P_coexistence_numerator = (self.b1 - self.m1*self.c1*self.S1(y))*self.a1(x)*self.K1 + (self.b2 - self.m2*self.c2*self.S2(y))*self.a2(x)*self.K2 - self.d
        P_coexistence_denominator = (self.b1 - self.m1*self.c1*self.S1(y))*((self.a1(x))**2)*(self.K1/self.r1) + (self.b2 - self.m2*self.c2*self.S2(y))*((self.a2(x))**2)*(self.K2/self.r2)
        P_coexistence = P_coexistence_numerator/P_coexistence_denominator
        N1_coexistence = self.K1*(1 - (self.a1(x)/self.r1)*P_coexistence)
        N2_coexistence = self.K2*(1 - (self.a2(x)/self.r2)*P_coexistence)

        # Exclude N1 equilibrium
        N2_exclude_N1 = self.d/((self.b2 - self.m2*self.c2*self.S2(y))*self.a2(x))
        P_exclude_N1 = (self.r2/self.a2(x))*(1 - N2_exclude_N1/self.K2)
        N1_exclude_N1 = 0

        # Exclude N2 equilibrium
        N1_exclude_N2 = self.d/((self.b1 - self.m1*self.c1*self.S1(y))*self.a1(x))
        P_exclude_N2 = (self.r1/self.a1(x))*(1 - N1_exclude_N2/self.K1)
        N2_exclude_N2 = 0

        # Exclude P equilibrium
        N1_exclude_P = self.K1
        N2_exclude_P = self.K2
        P_exclude_P = 0

        sum_bool_check = 0
        cutoff = 1e-10
        while (sum_bool_check != 1):
            # Boolean values of feasibility
            feas_coexistence = self.not_negative(N1_coexistence,N2_coexistence,P_coexistence)
            feas_exclude_N1 = self.not_negative(N1_exclude_N1,N2_exclude_N1,P_exclude_N1)
            feas_exclude_N2 = self.not_negative(N1_exclude_N2,N2_exclude_N2,P_exclude_N2)
            feas_exclude_P = self.not_negative(N1_exclude_P,N2_exclude_P,P_exclude_P)

            # Boolean values of saturation
            satu_coexistence = self.not_positive(self.per_capita_grow_rate(N1_coexistence,N2_coexistence,P_coexistence,x,y),cutoff)
            satu_exclude_N1 = self.not_positive(self.per_capita_grow_rate(N1_exclude_N1,N2_exclude_N1,P_exclude_N1,x,y),cutoff)
            satu_exclude_N2 = self.not_positive(self.per_capita_grow_rate(N1_exclude_N2,N2_exclude_N2,P_exclude_N2,x,y),cutoff)
            satu_exclude_P = self.not_positive(self.per_capita_grow_rate(N1_exclude_P,N2_exclude_P,P_exclude_P,x,y),cutoff)

            feas_and_satu_coexistence = feas_coexistence and satu_coexistence
            feas_and_satu_exclude_N1 = feas_exclude_N1 and satu_exclude_N1
            feas_and_satu_exclude_N2 = feas_exclude_N2 and satu_exclude_N2
            feas_and_satu_exclude_P = feas_exclude_P and satu_exclude_P

            bool_check = [feas_and_satu_coexistence,
                          feas_and_satu_exclude_N1,
                          feas_and_satu_exclude_N2,
                          feas_and_satu_exclude_P]

            sum_bool_check = np.sum(bool_check)

            if (cutoff <= 1e-20):
                print "Saturated Equilibrium function not working correctly..."
                print N1_coexistence,N2_coexistence,P_coexistence
                print N1_exclude_N1,N2_exclude_N1,P_exclude_N1
                print N1_exclude_N2,N2_exclude_N2,P_exclude_N2
                print N1_exclude_P,N2_exclude_P,P_exclude_P
                print feas_coexistence, feas_exclude_N1, feas_exclude_N2, feas_exclude_P
                print per_capita_grow_rate(N1_coexistence,N2_coexistence,P_coexistence,x,y)
                print per_capita_grow_rate(N1_exclude_N1,N2_exclude_N1,P_exclude_N1,x,y)
                print per_capita_grow_rate(N1_exclude_N2,N2_exclude_N2,P_exclude_N2,x,y)
                print per_capita_grow_rate(N1_exclude_P,N2_exclude_P,P_exclude_P,x,y)
                print bool_check

                raise
            cutoff = cutoff/1.01

        if feas_and_satu_coexistence:
            return [N1_coexistence,N2_coexistence,P_coexistence]
        elif feas_and_satu_exclude_N1:
            return [N1_exclude_N1,N2_exclude_N1,P_exclude_N1]
        elif feas_and_satu_exclude_N2:
            return [N1_exclude_N2,N2_exclude_N2,P_exclude_N2]
        elif feas_and_satu_exclude_P:
            return [N1_exclude_P,N2_exclude_P,P_exclude_P]

    def get_nullclines(self,X,Y):
        N1 = np.zeros(np.shape(X))
        N2 = np.zeros(np.shape(X))
        for ind1 in xrange(len(X)):
            for ind2 in xrange(len(X[ind1])):
                v = self.saturated_equilibrium([X[ind1][ind2], Y[ind1][ind2]])
                N1[ind1][ind2] = v[0]
                N2[ind1][ind2] = v[1]

        a1X = self.a1(X)
        a2X = self.a2(X)
        S1Y = self.S1(Y)
        S2Y = self.S2(Y)

        xsurface = (self.b1 - self.m1*self.c1*S1Y)*N1*a1X*((self.theta1 - X)/self.A1) + (self.b2 - self.m2*self.c2*S2Y)*N2*a2X*((self.theta2 - X)/self.A2)
        ysurface = self.m1*self.c1*a1X*N1*(self.beta1 - S1Y)*((self.phi1 - Y)/self.B1) + self.m2*self.c2*a2X*N2*(self.beta2 - S2Y)*((self.phi2 - Y)/self.B2)

        return [xsurface,ysurface]

    def make_x_dot(self,N1,N2):
        def x_dot(x,y):
            one = (self.b1 - self.m1*self.c1*self.S1(y))*self.a1(x)*N1*((self.theta1 - x)/(self.A1))
            two = (self.b2 - self.m2*self.c2*self.S2(y))*self.a2(x)*N2*((self.theta2 - x)/(self.A2))
            return (self.sigmaxG**2)*(one + two)
        return x_dot

    def make_y_dot(self,N1,N2):
        def y_dot(x,y):
            one = -self.m1*self.c1*self.a1(x)*N1*(self.S1(y) - self.beta1)*((self.phi1 - y)/(self.B1))
            two = -self.m2*self.c2*self.a2(x)*N2*(self.S2(y) - self.beta2)*((self.phi2 - y)/(self.B2))
            return (self.sigmayG**2)*(one + two)
        return y_dot

    def get_slow_evo_model_equilibria(self):
        if not self.slow_evo:
            print 'this function only works for the slow evo model.'
            return

        def clines_independent_of_heritability(Y,t):
            f = [0]*2
            m = self.model(Y,t)
            f[0] = m[0]/(self.sigmaxG**2)
            f[1] = m[1]/(self.sigmayG**2)
            return f

        equilibria = []
        for i in np.linspace(0,1,15):
            for j in np.linspace(0,1,15):
                # solve for dx/dt = 0, dy/dt = 0
                eq, infodict, ier, mesg = fsolve(clines_independent_of_heritability,[i,j],args=0,full_output=True,xtol=1e-12)
                # check if solve was unsuccessful
                if ier != 1:
                    continue
                # check if solution is unique (up to some tolerance)
                if all([norm(eq - e) > 1e-7 for e in equilibria]):
                    equilibria.append(eq)
        relevant_equilibria = [eq for eq in equilibria if eq[0] >= 0 and eq[0] <= 1 and eq[1] >= 0 and eq[1] <= 1]
        return relevant_equilibria

    def determine_stability(self,equilibria):
        st = []
        unst = []
        for eq in equilibria:
            ics = [eq + 1e-4*np.array([np.cos(2*np.pi*(i/5)+0.1),np.sin(2*np.pi*(i/5)+0.1)]) for i in range(5)]
            return_trajectories = []
            for ic in ics:
                final = odeint(self.model,ic,np.linspace(0,50000000,100000))[-1,:]
                if norm(final-eq) < 1e-8:
                    return_trajectories.append(True)
                else:
                    return_trajectories.append(False)
            if all(return_trajectories):
                st.append(eq)
            else:
                unst.append(eq)
        return [st, unst]

    def get_separatrices(self,unst):
        solns = []
        for eq in unst:
            for i in range(2):
                soln = odeint(self.model,eq + 1e-4*np.array([np.cos(2*np.pi*(i/2)+0.1),np.sin(2*np.pi*(i/2)+0.1)]),np.linspace(0,-500000000,100000))
                solns.append(soln)
        return solns

    def jacobian_approximation(self,e,epsilon=1e-10):
        if not self.slow_evo:
            print 'this function only works for the slow evo model.'
            return
        (N1star, N2star, Pstar) = self.saturated_equilibrium(e)
        x = e[0]
        y = e[1]
        x_dot = self.make_x_dot(N1star,N2star)
        y_dot = self.make_y_dot(N1star,N2star)

        # 4th-order accurate derivative approximation
        Jxx = (-x_dot(x+2*epsilon,y) + 8*x_dot(x+epsilon,y) - 8*x_dot(x-epsilon,y) + x_dot(x-2*epsilon,y))/(12*epsilon)
        Jxy = (-x_dot(x,y+2*epsilon) + 8*x_dot(x,y+epsilon) - 8*x_dot(x,y-epsilon) + x_dot(x,y-2*epsilon))/(12*epsilon)
        Jyx = (-y_dot(x+2*epsilon,y) + 8*y_dot(x+epsilon,y) - 8*y_dot(x-epsilon,y) + y_dot(x-2*epsilon,y))/(12*epsilon)
        Jyy = (-y_dot(x,y+2*epsilon) + 8*y_dot(x,y+epsilon) - 8*y_dot(x,y-epsilon) + y_dot(x,y-2*epsilon))/(12*epsilon)
        J = np.array([[Jxx,Jxy],[Jyx,Jyy]])
        return J

    def make_slow_evo_model(self):
        def model(Y,t):
            (N1star, N2star, Pstar) = self.saturated_equilibrium(Y)
            x = Y[0]
            y = Y[1]
            x_dot = self.make_x_dot(N1star,N2star)
            y_dot = self.make_y_dot(N1star,N2star)

            f = [0]*2
            f[0] = x_dot(x,y)
            f[1] = y_dot(x,y)
            return f

        return model