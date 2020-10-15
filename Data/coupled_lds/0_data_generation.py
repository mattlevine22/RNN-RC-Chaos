#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from Utils.differential_equation import f_ode
import pickle
import random as rand
from scipy.integrate import solve_ivp, ode
import sys
import pdb

A = np.array([[0, 5],[-5,0]])
dimensions = 2*A.shape[0] # 2 coupled oscillators governed by A

T1 = 1000; T2 = 2000; dt = 0.01;


# INTEGRATION
u0 = np.random.randn(dimensions)
t0 = 0

print("Initial transients...")
t_span = [t0, T1]
t_eval = np.array([t0+T1])
# sol = solve_ivp(fun=lambda t, y: self.rhs(t0, y0), t_span=t_span, y0=u0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='Radau')

print("Integration...")
u0 = np.squeeze(sol.y)
t_span = [t0, T2]
t_eval_tmp = np.arange(t0, T2, dt)
t_eval = np.zeros(len(t_eval_tmp)+1)
t_eval[:-1] = t_eval_tmp
t_eval[-1] = T2
# sol = solve_ivp(fun=lambda t, y: self.rhs(t0, y0), t_span=t_span, y0=u0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='Radau')
u = sol.y.T

data = {
    "A":A,
    "T1":T1,
    "T2":T2,
    "dt":dt,
    "u":u,
}


with open("./Simulation_Data/coupled_lds_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
