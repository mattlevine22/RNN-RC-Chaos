#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Matthew Levine, CMS, Caltech
"""
#!/usr/bin/env python
import numpy as np
import io
import os
import pdb

class Physics(object):
	def __init__(self, name='l63', params=None):
		self.rhs = getattr(self, name)

	def l63(self, t0, u0, sigma=10, rho=28, beta=8./3):
	    dudt = np.zeros(np.shape(u0))
	    dudt[0] = sigma * (u0[1]-u0[0])
	    dudt[1] = u0[0] * (rho-u0[2]) - u0[1]
	    dudt[2] = u0[0] * u0[1] - beta*u0[2]
	    return dudt

	def lds(self, t0, u0, A=np.array([[0, 5],[-5,0]])):
	    dudt = A @ u0
	    return dudt

	def set_rhs(self):
		return
