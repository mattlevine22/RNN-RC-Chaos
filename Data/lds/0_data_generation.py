#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from Utils.lds import *
import pickle
import random as rand
from scipy.integrate import ode
import sys

dimensions = 2
A = np.array([[0, 5],[-5,0]])

T1 = 1000; T2 = 2000; dt = 0.01;


# INTEGRATION
u0 = np.random.randn(dimensions,1)
t0 = 0

r = ode(lds)
r.set_initial_value(u0, t0).set_f_params(A)


print("Initial transients...")
while r.successful() and r.t < T1:
    r.integrate(r.t+dt)
    sys.stdout.write("\r Time={:.2f}".format(r.t)+ " " * 10)
    sys.stdout.flush()

print("\n")

u0 = r.y
t0 = 0
r.set_initial_value(u0, t0).set_f_params(A)

u = np.zeros((dimensions, int(T2/dt)+1))
u[:,0] = np.reshape(u0, (-1))

print("Integration...")
i=1
while r.successful() and r.t < T2 - dt:
    r.integrate(r.t+dt);
    u[:,i] = np.reshape(r.y, (-1))
    i=i+1
    sys.stdout.write("\r Time={:.2f}".format(r.t)+ " " * 10)
    sys.stdout.flush()
print("\n")

u = np.transpose(u)

data = {
    "A":A,
    "T1":T1,
    "T2":T2,
    "dt":dt,
    "u":u,
}


with open("./Simulation_Data/lds_data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
