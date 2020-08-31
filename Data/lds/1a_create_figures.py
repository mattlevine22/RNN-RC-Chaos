#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm

plt.rcParams["text.usetex"] = True
# plt.rcParams['xtick.major.pad']='20'
# plt.rcParams['ytick.major.pad']='20'
# mpl.rcParams['legend.fontsize'] = 10
font = {'weight':'normal', 'size':12}
plt.rc('font', **font)

import pickle


with open("./Simulation_Data/lds_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["u"]
    A = data["A"]

N_plot = 20000
fig = plt.figure()
ax = fig.gca(projection='2d')
ax.plot(u[:N_plot,0], u[:N_plot,1], "b--", label='LDS attractor')
ax.legend()
#plt.show()
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.savefig("./Figures/Trajectories_plot_{:d}.pdf".format(N_plot), dpi=300)
plt.close()
