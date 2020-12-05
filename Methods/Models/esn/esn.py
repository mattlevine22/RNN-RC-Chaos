#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
				Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
from scipy import signal # for periodogram
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from scipy.integrate import solve_ivp, ode
from scipy.interpolate import CubicSpline
# from scipy.linalg import lstsq as scipylstsq
# from numpy.linalg import lstsq as numpylstsq
from utils import *
import os
from plotting_utils import *
from global_utils import *
from ode_utils import *
import pickle
import time
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#MATT hacking in ODE SOLVER
import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
base_dir = '/Users/matthewlevine/code_projects/RNN-RC-Chaos'
lorenz_dir = os.path.join(base_dir, "Data/Lorenz3D/Utils")
sys.path.insert(0, lorenz_dir)
from lorenz3D import lorenz
lds_dir = os.path.join(base_dir,"Data/lds/Utils")
sys.path.insert(0, lds_dir)
from lds import lds



# MEMORY TRACKING
import psutil

class esn(object):
	def delete(self):
		return 0

	def __init__(self, params):
		self.display_output = params["display_output"]

		print("RANDOM SEED: {:}".format(params["worker_id"]))
		np.random.seed(params["worker_id"])

		self.worker_id = params["worker_id"]
		self.input_dim = params["RDIM"]
		self.N_used = params["N_used"]
		self.approx_reservoir_size = params["approx_reservoir_size"]
		self.degree = params["degree"]
		self.radius = params["radius"]
		self.sigma_input = params["sigma_input"]
		self.dynamics_length = params["dynamics_length"]
		self.iterative_prediction_length = params["iterative_prediction_length"]
		self.num_test_ICS = params["num_test_ICS"]
		self.train_data_path = params["train_data_path"]
		self.test_data_path = params["test_data_path"]
		self.fig_dir = params["fig_dir"]
		self.model_dir = params["model_dir"]
		self.logfile_dir = params["logfile_dir"]
		self.write_to_log = params["write_to_log"]
		self.results_dir = params["results_dir"]
		self.saving_path = params["saving_path"]
		self.regularization_RC = params["regularization_RC"]
		self.regularization_RF = params["regularization_RF"]
		self.scaler_tt = params["scaler"]
		self.scaler_tt_derivatives = params["scaler_derivatives"]
		self.learning_rate = params["learning_rate"]
		self.number_of_epochs = params["number_of_epochs"]
		self.solver = str(params["solver"])
		##########################################
		self.component_wise = params["component_wise"]
		self.scaler = scaler(tt=self.scaler_tt, tt_derivative=self.scaler_tt_derivatives, component_wise=self.component_wise)
		self.noise_level = params["noise_level"]
		self.model_name = self.createModelName(params)
		self.dt = self.get_dt()
		self.dt_fast_frac = params["dt_fast_frac"]
		self.test_integrator = params["test_integrator"]
		self.hidden_dynamics = params["hidden_dynamics"]
		self.output_dynamics = params["output_dynamics"]
		self.gamma = params["gamma"]
		self.lam = params["lambda"]
		self.plot_matrix_spectrum = params["plot_matrix_spectrum"]
		self.use_tilde = params["use_tilde"]
		self.dont_redo = params["dont_redo"]
		self.bias_var = params["bias_var"]
		self.learn_markov = params["learn_markov"]
		self.learn_memory = params["learn_memory"]
		self.rf_dim = params["rf_dim"]
		self.rf_Win_bound = params["rf_Win_bound"]
		self.rf_bias_bound = params["rf_bias_bound"]

		####### Add physical mechanistic rhs "f0" ##########
		self.use_f0 = params["use_f0"]
		if self.use_f0:
			physics = Physics(name=params["f0_name"])
			self.f0 = physics.rhs
			self.rc_error_input = params["rc_error_input"]
			self.rc_state_input = params["rc_state_input"]
			self.rf_error_input = 0 #params["rf_error_input"] #THIS doesnt work yet
		else:
			self.f0 = 0
			self.rc_error_input = 0
			self.rf_error_input = 0
		# count the augmented input dimensions (i.e. [x(t); f0(x(t))-x(t)])
		if self.component_wise:
			self.input_dim_rc = self.rc_state_input + self.rc_error_input
			self.input_dim_rf = 1 + self.rf_error_input
		else:
			self.input_dim_rc = (self.rc_state_input + self.rc_error_input)*self.input_dim
			self.input_dim_rf = (1 + self.rf_error_input)*self.input_dim

		self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
		# print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

		os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

		print('FIGURE PATH:', self.saving_path + self.results_dir + self.model_name)
	# def solve(self, r0, x0, solver='Euler'):

	def predict_next(self, x_input, h_reservoir):
		solver = self.test_integrator

		x_input = np.squeeze(x_input)
		h_reservoir = np.squeeze(h_reservoir)

		u0 = np.hstack((x_input, h_reservoir))
		t0 = 0

		if solver=='Euler':
			rhs = self.rhs(t0, u0)
			u_next = u0 + self.dt * rhs
			x_next = u_next[:self.input_dim,None]
			h_next = u_next[self.input_dim:,None]
		elif solver=='Euler_fast':
			dt_fast = self.dt_fast_frac * self.dt
			t_end = t0 + self.dt
			t = np.float(t0)
			u = np.copy(u0)
			while t < t_end:
				rhs = self.rhs(t, u)
				u += dt_fast * rhs
				t += dt_fast
			x_next = u[:self.input_dim,None]
			h_next = u[self.input_dim:,None]
		elif solver=='RK45':
			t_span = [t0, t0+self.dt]
			t_eval = np.array([t0+self.dt])
			# sol = solve_ivp(fun=lambda t, y: self.rhs(t0, y0), t_span=t_span, y0=u0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
			sol = solve_ivp(fun=self.rhs, t_span=t_span, y0=u0, t_eval=t_eval, max_step=self.dt/2)
			u_next = sol.y
			x_next = u_next[:self.input_dim]
			h_next = u_next[self.input_dim:]
		elif solver=='Euler_old':
			rhs = self.rhs(t0, u0)
			if self.learn_markov and self.learn_memory:
				raise ValueError('Havent dealt with this case yet')
			elif self.learn_memory:
				if self.use_f0:
					y0 = self.scaler.descaleData(x_input)
					f0 = self.f0(t0=t0, u0=y0)
					if self.scaler_tt in ['Standard', 'standard']:
						f0 = f0 / self.scaler.data_std
						f0 = np.reshape(f0, (-1,1))
					else:
						raise ValueError('not set up to undo other types of normalizations!')
				else:
					f0 = 0
				h_next = h_reservoir[:,None] + self.dt * rhs[self.input_dim:,None]
				x_next = x_input[:,None] + self.dt * (f0 + self.W_out_memory @ self.augmentHidden(h_next))
			elif self.learn_markov:
				h_next = h_reservoir[:,None] # should be empty
				x_next = x_input[:,None] + self.dt * rhs[:,None]

		elif solver=='Euler_old_fast':
			dt_fast = self.dt_fast_frac * self.dt
			t_end = t0 + self.dt
			t = np.float(t0)
			u = np.copy(u0)
			while t < t_end:
				rhs = self.rhs(t, u)
				u[self.input_dim:] = u[self.input_dim:] + dt_fast * rhs[self.input_dim:]
				if self.component_wise:
					for k in range(self.input_dim):
						kth_inds = np.arange((self.input_dim+k*self.reservoir_size),(self.input_dim+(k+1)*self.reservoir_size))
						u[k] += dt_fast * self.W_out_memory @ self.augmentHidden(u[kth_inds])
				else:
					u[:self.input_dim] = u[:self.input_dim]  + dt_fast * self.W_out_memory @ self.augmentHidden(u[self.input_dim:])
				t += dt_fast
			x_next = u[:self.input_dim,None]
			h_next = u[self.input_dim:,None]

		elif solver=='Euler_old_fast_v2':
			dt_fast = self.dt_fast_frac * self.dt
			t_end = t0 + self.dt
			t = np.float(t0)
			u = np.copy(u0)
			while t < t_end:
				rhs = self.rhs(t, u)
				# update reservoir state
				u[self.input_dim:] = u[self.input_dim:] + dt_fast * rhs[self.input_dim:]

				# update output state
				fcorr = np.zeros(self.input_dim)
				if self.learn_memory:
					if self.component_wise:
						for k in range(self.input_dim):
							kth_inds = np.arange((self.input_dim+k*self.reservoir_size),(self.input_dim+(k+1)*self.reservoir_size))
							fcorr[k] += self.W_out_memory @ self.augmentHidden(u[kth_inds])
					else:
						fcorr += self.W_out_memory @ self.augmentHidden(u[self.input_dim:])
				if self.learn_markov:
					if self.component_wise:
						for k in range(self.input_dim):
							kth_inds = np.arange((self.input_dim+k*self.reservoir_size),(self.input_dim+(k+1)*self.reservoir_size))
							foo = np.tanh(self.W_in_markov @ u[k,None] + np.squeeze(self.b_h_markov))
							fcorr[k] += self.W_out_memory @ foo
					else:
						foo = np.tanh(self.W_in_markov @ u[:self.input_dim] + np.squeeze(self.b_h_markov))
						fcorr += self.W_out_markov @ foo

				u[:self.input_dim] = u[:self.input_dim]  + dt_fast*fcorr
				t += dt_fast
			x_next = u[:self.input_dim,None]
			h_next = u[self.input_dim:,None]

			# check with old (using dt_fast=self.dt)
			# rhs_old = self.rhs(t0, u0)
			# h_next_old = h_reservoir[:,None] + self.dt * rhs_old[self.input_dim:,None]
			# x_next_old = x_input[:,None] + self.dt * self.W_out @ self.augmentHidden(h_next_old)
			# if not (np.array_equal(h_next_old, h_next) and np.array_equal(x_next_old, x_next)):
			# 	raise ValueError('New euler setup is not same as old!')

		return x_next, h_next


	def getKeysInModelName(self):
		keys = {
		'RDIM':'RDIM',
		'N_used':'N_used',
		'approx_reservoir_size':'SIZE',
		'degree':'D',
		'radius':'RADIUS',
		'sigma_input':'SIGMA',
		'dynamics_length':'DL',
		'noise_level':'NL',
		'iterative_prediction_length':'IPL',
		'regularization_RC':'REGRC',
		'regularization_RF':'REGRF',
		#'num_test_ICS':'NICS',
		# 'worker_id':'WID',
		'hidden_dynamics': 'HD',
		'output_dynamics': 'OD',
		'gamma': 'GAM',
		# 'lambda': 'LAM',
		'use_tilde': 'USETILDE',
		'scaler': 'SCALER',
		# 'scaler_derivatives': 'DSCALER',
		'bias_var': 'BVAR',
		'rf_dim': 'N_RF',
		'learn_markov': 'MARKOV',
		'learn_memory': 'MEMORY',
		'use_f0': 'f0',
		'test_integrator': 'INT',
		# 'dt_fast_frac': 'DTF'
		'component_wise': 'COMP',
		'rc_error_input': 'RCE',
		'rc_state_input': 'RCS'
		# 'rf_error_input': 'RFI'
		}
		return keys


	def createModelName(self, params):
		keys = self.getKeysInModelName()
		str_ = "RNN-esn_" + self.solver
		for key in keys:
			str_ += "-" + keys[key] + "_{:}".format(params[key])
		return str_

	def getSparseWeights(self, sizex, sizey, radius, sparsity, worker_id=1):
		# W = np.zeros((sizex, sizey))
		# Sparse matrix with elements between 0 and 1
		# print("WEIGHT INIT")
		W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id)
		# W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id, data_rvs=np.random.randn)
		# Sparse matrix with elements between -1 and 1
		# W.data *=2
		# W.data -=1
		# to print the values do W.A
		# print("EIGENVALUE DECOMPOSITION")
		eigenvalues, eigvectors = splinalg.eigs(W)
		eigenvalues = np.abs(eigenvalues)
		W = (W/np.max(eigenvalues))*radius
		return W

	def augmentHidden(self, h):
		h_aug = h.copy()
		# h_aug = pow(h_aug, 2.0)
		# h_aug = np.concatenate((h,h_aug), axis=0)
		if self.use_tilde:
			h_aug[::2]=pow(h_aug[::2],2.0)
		return h_aug

	def getAugmentedStateSize(self):
		return self.reservoir_size

	# def augmentHidden(self, h):
	#     h_aug = h.copy()
	#     h_aug = pow(h_aug, 2.0)
	#     h_aug = np.concatenate((h,h_aug), axis=0)
	#     return h_aug
	# def getAugmentedStateSize(self):
	#     return 2*self.reservoir_size

	def set_h_zeros(self, type='wide'):
		self.h_zeros = np.array([])
		if self.learn_memory:
			if self.component_wise:
				if type=='wide':
					self.h_zeros = np.zeros((self.reservoir_size, self.input_dim))
				elif type=='tall':
					self.h_zeros = np.zeros((self.input_dim*self.reservoir_size, 1))
				else:
					raise ValueError('type of structure for hidden reservoir state not recognized. Choose tall or wide.')
			else:
				self.h_zeros = np.zeros((self.reservoir_size, 1))

	def set_random_weights(self):

		# First initialize everything to be None
		self.W_in_markov = None
		self.b_h_markov = None
		self.b_h = None
		self.W_in = None
		self.W_h = None
		self.W_h_effective = None
		self.W_out_memory = None
		self.W_out_markov = None

		# initialize markovian random terms for Random Feature Maps
		if self.learn_markov:
			b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rf_dim, 1))
			W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rf_dim, self.input_dim_rf))

			self.W_in_markov = W_in_markov
			self.b_h_markov = b_h_markov

		# initialize Reservoir random terms for RC
		if self.learn_memory:
			# print("Initializing the reservoir weights...")
			if self.component_wise:
				self.reservoir_size = self.approx_reservoir_size
			else:
				nodes_per_input = int(np.ceil(self.approx_reservoir_size/self.input_dim_rc))
				self.reservoir_size = int(self.input_dim_rc*nodes_per_input)
			self.sparsity = self.degree/self.reservoir_size;
			print("NETWORK SPARSITY: {:}".format(self.sparsity))
			# print("Computing sparse hidden to hidden weight matrix...")
			W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)

			# Initializing the input weights
			# print("Initializing the input weights...")
			W_in = np.zeros((self.reservoir_size, self.input_dim_rc))
			q = int(self.reservoir_size/self.input_dim_rc)
			for i in range(0, W_in.shape[1]):
				W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))

			# Initialize the hidden bias term
			b_h = self.bias_var * np.random.randn(self.reservoir_size, 1)

			# Set the diffusion term
			gammaI = self.gamma * sparse.eye(self.reservoir_size)

			if self.hidden_dynamics in ['ARNN', 'LARNN_forward']:
				W_h_effective = (W_h-W_h.T - gammaI)
			else:
				W_h_effective = W_h

			# add to self
			self.b_h = b_h
			self.W_in = W_in
			self.W_h = W_h
			self.W_h_effective = W_h_effective

			plotMatrix(self, self.W_h.todense() , 'W_h')
			plotMatrix(self, self.W_h_effective.todense() , 'W_h_effective')


	def x_t(self, t, t0=0):
		#linearly interpolate self.x_vec at time t
		ind_mid = (t-t0) / self.dt
		ind_low = max(0, min( int(np.floor(ind_mid)), self.x_vec.shape[0]-1) )
		ind_high = min(self.x_vec.shape[0]-1, int(np.ceil(ind_mid)))
		v0 = self.x_vec[ind_low,:]
		v1 = self.x_vec[ind_high,:]
		tmid = ind_mid - ind_low

		return (1 - tmid) * v0 + tmid * v1

	def xdot_t(self, t):
		'''differentiate self.x_vec at time t using stored component-wise spline interpolant'''

		# initialize output
		xdot = np.zeros(self.input_dim)
		for k in range(self.input_dim):
			xdot[k] = self.xdot_spline[k](t)
		return xdot

	def mdag(self, t, x, xdot):
		if self.use_f0:
			m = xdot - self.f0(t0=t, u0=x)
		else:
			m = xdot
		return m

	def q_t(self, x_t):
		q = np.tanh(self.W_in_markov @ x_t + np.squeeze(self.b_h_markov))
		return q

	def rcrf_rhs(self, t, S, k=None):
		'''k is the component when doing component-wise models'''
		x = self.x_t(t=t)
		xdot = self.xdot_t(t=t)
		m = self.mdag(t, x, xdot)
		if self.component_wise:
			x = x[k,None]
			xdot = xdot[k,None]
			m = m[k,None]

		if self.rc_error_input and self.rc_state_input:
			rc_input = np.hstack((x,m))
		elif self.rc_state_input:
			rc_input = x
		elif self.rc_error_input:
			rc_input = m

		if self.rf_error_input:
			rf_input = np.hstack((x,m))
		else:
			rf_input = x

		if self.learn_memory:
			r = S[:self.reservoir_size]
			r_aug = self.augmentHidden(r)
			dr = np.tanh( self.W_h_effective @ r + self.W_in @ rc_input + np.squeeze(self.b_h) )
			dZrr = np.outer(r_aug, r_aug).reshape(-1)
			dYr = np.outer(r_aug, m).reshape(-1)
		if self.learn_markov:
			q = self.q_t(rf_input)
			dZqq = np.outer(q, q).reshape(-1)
			dYq = np.outer(q, m).reshape(-1)

		if self.learn_memory and self.learn_markov:
			dZrq = np.outer(r_aug, q).reshape(-1)
			S = np.hstack( (dr, dZrr, dZrq, dZqq, dYq, dYr) )
		elif self.learn_memory:
			S = np.hstack( (dr, dZrr, dYr) )
		elif self.learn_markov:
			S = np.hstack( (dZqq, dYq) )
		return S

	def newMethod_getIC(self, T_warmup, T_train):
		# generate ICs for training integration
		yall = []

		if self.learn_memory:
			x0 = self.x_t(t=0)
			xdot0 = self.xdot_t(t=0)
			m0 = self.mdag(t=0, x=x0, xdot=xdot0)
			if self.component_wise:
				print('Integrating over warmup data...')
				timer_start = time.time()
				t_span = [0, T_warmup]
				t_eval = np.array([T_warmup])
				for k in range(self.input_dim):
					x0k = x0[k,None]
					m0k = m0[k,None]
					# warm up reservoir state if learning memory
					r0 = np.zeros(self.reservoir_size)
					r_aug0 = self.augmentHidden(r0)
					Zrr0 = np.outer(r_aug0, r_aug0).reshape(-1)
					Yr0 = np.outer(r_aug0, m0k).reshape(-1)
					if self.learn_markov:
						if self.rf_error_input:
							rf_input = np.hstack((x0k,m0k))
						else:
							rf_input = x0k
						q0k = self.q_t(rf_input)
						Zqq0 = np.outer(q0k, q0k).reshape(-1)
						Yq0 = np.outer(q0k, m0k).reshape(-1)
						Zrq0 = np.outer(r_aug0, q0k).reshape(-1)
						y0 = np.hstack( (r0, Zrr0, Zrq0, Zqq0, Yq0, Yr0) )
					else:
						y0 = np.hstack( (r0, Zrr0, Yr0) )

					sol = solve_ivp(fun=lambda t, y: self.rcrf_rhs(t, y, k=k), t_span=t_span, y0=y0, t_eval=t_eval, max_step=self.dt/2)
					y0 = np.squeeze(sol.y)
					yall.append(y0)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
			else:
				# warm up reservoir state if learning memory
				r0 = np.zeros(self.reservoir_size)
				r_aug0 = self.augmentHidden(r0)
				Zrr0 = np.outer(r_aug0, r_aug0).reshape(-1)
				Yr0 = np.outer(r_aug0, m0).reshape(-1)
				if self.learn_markov:
					if self.rf_error_input:
						rf_input = np.hstack((x0,m0))
					else:
						rf_input = x0
					q0 = self.q_t(rf_input)
					Zqq0 = np.outer(q0, q0).reshape(-1)
					Yq0 = np.outer(q0, m0).reshape(-1)
					Zrq0 = np.outer(r_aug0, q0).reshape(-1)
					y0 = np.hstack( (r0, Zrr0, Zrq0, Zqq0, Yq0, Yr0) )
				else:
					y0 = np.hstack( (r0, Zrr0, Yr0) )

				print('Integrating over warmup data...')
				timer_start = time.time()
				t_span = [0, T_warmup]
				t_eval = np.array([T_warmup])
				sol = solve_ivp(fun=self.rcrf_rhs, t_span=t_span, y0=y0, t_eval=t_eval, max_step=self.dt/2)
				yall = np.squeeze(sol.y)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))
		elif self.learn_markov:
			x0 = self.x_t(t=T_warmup)
			xdot0 = self.xdot_t(t=T_warmup)
			m0 = self.mdag(t=T_warmup, x=x0, xdot=xdot0)
			if self.component_wise:
				for k in range(self.input_dim):
					x0k = x0[k,None]
					m0k = m0[k,None]
					if self.rf_error_input:
						rf_input = np.hstack((x0k,m0k))
					else:
						rf_input = x0k
					q0k = self.q_t(rf_input)
					Zqq0 = np.outer(q0k, q0k).reshape(-1)
					Yq0 = np.outer(q0k, m0k).reshape(-1)
					y0 = np.hstack( (Zqq0, Yq0) )
					yall.append(y0)
			else:
				if self.rf_error_input:
					rf_input = np.hstack((x0,m0))
				else:
					rf_input = x0
				q0 = self.q_t(rf_input)
				Zqq0 = np.outer(q0, q0).reshape(-1)
				Yq0 = np.outer(q0, m0).reshape(-1)
				yall = np.hstack( (Zqq0, Yq0) )
		else:
			yall = []

		yall = np.array(yall)

		return yall

	def newMethod_saveYZ(self, yend, T_train):

		if self.component_wise:
			in_dim = 1
		else:
			in_dim = self.input_dim

		if self.learn_memory and self.learn_markov:
			# y0 = np.hstack( (r0, Zrr0, Zrq0, Zqq0, Yq0, Yr0) )
			r = yend[:self.reservoir_size]

			# Zrr
			i_start = self.reservoir_size
			i_end = i_start + self.reservoir_size**2
			Zrr = yend[i_start:i_end]

			#Zrq
			i_start = i_end
			i_end = i_start + self.reservoir_size*self.rf_dim
			Zrq = yend[i_start:i_end]

			#Zqq
			i_start = i_end
			i_end = i_start + self.rf_dim**2
			Zqq = yend[i_start:i_end]

			#Yq
			i_start = i_end
			i_end = i_start + self.rf_dim*in_dim
			Yq = yend[i_start:i_end]

			#Yr
			Yr = yend[i_end:]

		elif self.learn_markov:
			# y0 = np.hstack( (Zqq0, Yq0) )
			Zqq = yend[:self.rf_dim**2]
			Yq = yend[self.rf_dim**2:]
		elif self.learn_memory:
			# y0 = np.hstack( (r0, Zrr0, Yr0) )
			r = yend[:self.reservoir_size]
			i_start = self.reservoir_size
			i_end = i_start + self.reservoir_size**2
			Zrr = yend[i_start:i_end]
			Yr = yend[i_end:]

		# Concatenate solutions into big matrices Y, Z
		if self.learn_memory and self.learn_markov:
			Zrq = Zrq.reshape(self.reservoir_size, self.rf_dim)
		if self.learn_memory:
			Zrr = Zrr.reshape(self.reservoir_size, self.reservoir_size)
			Yr = Yr.reshape(self.reservoir_size, in_dim)
		if self.learn_markov:
			Zqq = Zqq.reshape(self.rf_dim, self.rf_dim)
			Yq = Yq.reshape(self.rf_dim, in_dim)

		if self.learn_memory and self.learn_markov:
			Y = np.vstack( (Yr, Yq) )
			Z = np.vstack( ( np.hstack( (Zrr, Zrq) ), np.hstack( (Zrq.T, Zqq) ) ) )
		elif self.learn_markov:
			Y = Yq
			Z = Zqq
		elif self.learn_memory:
			Y = Yr
			Z = Zrr
		else:
			Y = np.zeros(1)
			Z = np.zeros(1)

		# save Time-Normalized Y,Z
		if self.component_wise:
			self.Y.append(Y / T_train)
			self.Z.append(Z / T_train)
		else:
			self.Y = Y / T_train
			self.Z = Z / T_train

		# store Z size for building regularization identity matrix
		self.reg_dim = Z.shape[0]


	def newMethod(self, tl, dynamics_length, train_input_sequence):

		self.x_vec = train_input_sequence

		# get spline derivative
		t_vec = self.dt*np.arange(self.x_vec.shape[0])
		self.xdot_spline = [CubicSpline(x=t_vec, y=self.x_vec[:,k]).derivative() for k in range(self.input_dim)]

		T_warmup = self.dt*dynamics_length
		T_train = self.dt*tl

		y0 = self.newMethod_getIC(T_warmup=T_warmup, T_train=T_train)

		if self.component_wise:
			self.Y = []
			self.Z = []
			for k in range(self.input_dim):
				# Perform training integration using IC y0
				print('Integrating over training data...')
				timer_start = time.time()
				t_span = [T_warmup, T_warmup + T_train]
				# t_eval = np.array([T_warmup + T_train])
				# sol = solve_ivp(fun=self.rcrf_rhs, t_span=t_span, y0=y0, t_eval=t_eval, max_step=self.dt/2)
				sol = solve_ivp(fun=lambda t, y: self.rcrf_rhs(t, y, k=k), t_span=t_span, y0=y0[k], max_step=self.dt/2)
				print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))

				# plot the integration
				if self.learn_memory:
					# print('Plotting reservoir state...')
					newMethod_plotting(model=self, hidden=sol.y[:self.reservoir_size,:].T, set_name='TRAINORIGINAL_component{k}'.format(k=k), dt=self.dt)
					# self.r_aug = self.vstack(self.r_aug, self.augmentHidden(sol.y[:self.reservoir_size,:].T))

				# allocate, reshape, normalize, and save solutions
				print('Compute final Y,Z component...')
				self.newMethod_saveYZ(yend=sol.y[:,-1], T_train=T_train)
			self.Y = np.vstack(self.Y)
			self.Z = np.vstack(self.Z)
		else:
			# Perform training integration using IC y0
			print('Integrating over training data...')
			timer_start = time.time()
			t_span = [T_warmup, T_warmup + T_train]
			# t_eval = np.array([T_warmup + T_train])
			# sol = solve_ivp(fun=self.rcrf_rhs, t_span=t_span, y0=y0, t_eval=t_eval, max_step=self.dt/2)
			sol = solve_ivp(fun=self.rcrf_rhs, t_span=t_span, y0=y0, max_step=self.dt/2)
			print('...took {:2.2f} minutes'.format((time.time() - timer_start)/60))

			# plot the integration
			if self.learn_memory:
				# print('Plotting reservoir state...')
				newMethod_plotting(model=self, hidden=sol.y[:self.reservoir_size,:].T, set_name='TRAINORIGINAL', dt=self.dt)
				# self.r_aug = self.augmentHidden(sol.y[:self.reservoir_size,:].T)

			# allocate, reshape, normalize, and save solutions
			print('Compute final Y,Z...')
			self.newMethod_saveYZ(yend=sol.y[:,-1], T_train=T_train)
			## FORGOT TO AUGMENT R STATE?


	def doNewSolving(self):
		# regI = np.identity(self.Z.shape[0])
		regI = np.identity(self.reg_dim)
		if self.learn_memory and self.learn_markov:
			regI[:self.reservoir_size,:self.reservoir_size] *= self.regularization_RC
			regI[self.reservoir_size:,self.reservoir_size:] *= self.regularization_RF
		elif self.learn_markov:
			regI *= self.regularization_RF
		elif self.learn_memory:
			regI *= self.regularization_RC

		if self.component_wise:
			# stack regI K times
			regI = np.tile(regI,(self.input_dim,1))

		pinv_ = scipypinv2(self.Z + regI)
		# W_out_all = self.Y.T @ pinv_ # old code
		W_out_all = (pinv_ @ self.Y).T # basically the same...very slight differences due to numerics
		if self.learn_memory and self.learn_markov:
			self.W_out_memory = W_out_all[:,:self.reservoir_size]
			self.W_out_markov = W_out_all[:,self.reservoir_size:]
		elif self.learn_markov:
			self.W_out_markov = W_out_all
		elif self.learn_memory:
			self.W_out_memory = W_out_all

		# print("FINALISING WEIGHTS...")
		if self.learn_markov:
			plotMatrix(self, self.W_out_markov, 'W_out_markov')
		if self.learn_memory:
			plotMatrix(self, self.W_out_memory, 'W_out_memory')
		return

	def getPrediction(self):
		pred = np.zeros( (self.X.shape[0], self.input_dim) )

		if self.component_wise:
			for k in range(self.input_dim):
				if self.learn_memory:
					pred[:,k,None] += self.H_aug_memory_big[:,:,k] @ self.W_out_memory.T
				if self.learn_markov:
					pred[:,k,None] += self.H_markov_big[:,:,k] @ self.W_out_markov.T
		else:
			if self.learn_memory:
				pred += self.H_aug_memory_big @ self.W_out_memory.T
			if self.learn_markov:
				pred += self.H_markov_big @ self.W_out_markov.T

		self.pred = pred

	def makeNewPlots(self, true_state, true_residual, predicted_residual, H_memory_big=None, set_name='', ic_idx=''):
		n_times = true_state.shape[0] # self.X
		time_vec = np.arange(n_times)*self.dt
		# true_residual: self.Y_all

		## Plot original hidden dynamics
		if self.learn_memory and H_memory_big is not None:
			## Plot H
			fig_path = self.saving_path + self.fig_dir + self.model_name + "/hidden_raw_{}_{}.png".format(set_name, ic_idx)
			fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
			ax.set_xlabel(r"Time $t$", fontsize=12)
			ax.set_ylabel(r"State $h$", fontsize=12)
			n_hidden = H_memory_big.shape[1]
			for n in range(n_hidden):
				if self.component_wise:
					for k in range(self.input_dim):
						ax.plot(time_vec, H_memory_big[:,n,k])
				else:
					ax.plot(time_vec, H_memory_big[:,n])
			plt.savefig(fig_path)
			plt.close()

		## Plot the learned markovian function
		# Treat states as interchangeable:
		# Plot X_k vs (Y_k prediction, Y_k truth)
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/statewise_fits_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8, 6))
		ax.set_xlabel(r"State $X_k$", fontsize=12)
		ax.set_ylabel(r"Correction $Y_k$", fontsize=12)
		X_unlist = np.reshape(true_state, (-1,1))
		ax.scatter(X_unlist, np.reshape(true_residual, (-1,1)), color='gray', s=10, alpha=0.8, label='inferred residuals')
		ax.scatter(X_unlist, np.reshape(predicted_residual, (-1,1)), color='red', marker='+', s=3, label='fitted residuals')
		ax.legend()
		plt.savefig(fig_path)
		plt.close()

		# Treat states independently:
		for k in range(self.input_dim):
			fig_path = self.saving_path + self.fig_dir + self.model_name + "/statewise_fits_{set_name}_state{k}.png".format(set_name=set_name,k=k)
			fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8, 6))
			ax.set_xlabel(r"State $X_{k}$".format(k=k), fontsize=12)
			ax.set_ylabel(r"Correction $Y_{k}$".format(k=k), fontsize=12)
			ax.scatter(true_state[:,k], true_residual[:,k], color='gray', s=10, alpha=0.8, label='inferred residuals')
			ax.scatter(true_state[:,k], predicted_residual[:,k], color='red', marker='+', s=3, label='fitted residuals')
			ax.scatter(true_state[0,k], predicted_residual[0,k], color='red', marker='x', s=100, label='START fit')
			ax.legend()
			plt.savefig(fig_path)
			plt.close()

		mse = np.mean( (true_residual - predicted_residual)**2)

		# plot over time
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/timewise_fits_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].scatter(time_vec, true_residual[:,k], color='gray', s=10, alpha=0.8, label='inferred residuals')
			ax[k].scatter(time_vec, predicted_residual[:,k], color='red', marker='+', s=3, label='fitted residuals')
		ax[-1].legend()
		plt.suptitle('Timewise fits with total MSE {mse:.5}'.format(mse=mse))
		plt.savefig(fig_path)
		plt.close()

		# plot over time
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/timewise_errors_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].plot(time_vec, true_residual[:,k] - predicted_residual[:,k], linewidth=2)
		plt.suptitle('Timewise errors with total MSE {mse:.5}'.format(mse=mse))
		plt.savefig(fig_path)
		plt.close()

		# Plot PSD
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/PSD_errors_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))

		err_mat = (true_residual - predicted_residual)**2
		# normalize assuming each component statistically the same
		err_mat -= np.mean(err_mat,0) # subtract scalar avg error
		err_mat /= np.std(err_mat,0) # divide by scalar SD

		for k in range(self.input_dim):
			# errs = true_residual[:,k] - predicted_residual[:,k]
			f, Pxx_den = signal.periodogram(err_mat[:,k], fs=1/self.dt)
			# pdb.set_trace()
			if k==0:
				Pxx_den_SUM = Pxx_den
			else:
				Pxx_den_SUM += Pxx_den
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].semilogx(f, Pxx_den)
		ax[-1].set_xlabel('frequency [Hz]')
		fig.suptitle('PSD [V**2/Hz]')
		plt.savefig(fig_path)
		plt.close()

		# Plot avg PSD across trajectories
		Pxx_den_SUM /= self.input_dim
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/PSDavg_errors_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
		ax.set_ylabel('PSD [V**2/Hz]', fontsize=12)
		ax.set_xlabel('frequency [Hz]')
		ax.loglog(f, Pxx_den_SUM)
		ax.set_ylim([1e-8, 2])
		ax.set_title('Average PSD of normalized squared error of predicted sequence')
		plt.savefig(fig_path)
		plt.close()


		# Plot CROSS-CORRELATION between true data sequence and prediction error sequence
		true_mat = (true_state - np.mean(true_state,0)) / np.std(true_state,0)
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/XCORR_truth_vs_error_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=self.input_dim, ncols=1,figsize=(12, 12))
		for k in range(self.input_dim):
			xcorr = matt_xcorr(x=true_mat[:,k], y=err_mat[:,k])
			n_lags = (xcorr.shape[0]-1) / 2
			lag_vec = self.dt*np.arange(-n_lags,n_lags+1)
			ax[k].plot(lag_vec, xcorr, linewidth=2)
			ax[k].set_ylabel(r"$Y_{k}$".format(k=k), fontsize=12)
			ax[k].set_ylim([-1,1])
			ax[k].axvline(x=0, linewidth=1, linestyle='--', color='black')
			ax[k].axhline(y=0, linewidth=1, linestyle='--', color='black')
			if k==0:
				xcorr_SUM = xcorr
			else:
				xcorr_SUM += xcorr
		fig.suptitle('Cross correlation: Lag of true data correlating to errors')
		plt.savefig(fig_path)
		plt.close()

		# Plot avgCROSS-CORRELATION between true data sequence and prediction error sequence
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/XCORRavg_truth_vs_error_{}_{}.png".format(set_name, ic_idx)
		fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
		xcorr_SUM /= self.input_dim
		ax.set_ylabel(r"Correlation", fontsize=12)
		ax.plot(lag_vec, xcorr_SUM, linewidth=2)
		ax.set_ylim([-1,1])
		ax.axvline(x=0, linewidth=1, linestyle='--', color='black')
		ax.axhline(y=0, linewidth=1, linestyle='--', color='black')
		fig.suptitle('avg Cross correlation: Lag of true data correlating to errors')
		plt.savefig(fig_path)
		plt.close()

	def train(self):
		if self.dont_redo and os.path.exists(self.saving_path + self.model_dir + self.model_name + "/data.pickle"):
			raise ValueError('Model has already run for this configuration. Exiting with an error.')
		self.start_time = time.time()
		dynamics_length = self.dynamics_length
		input_dim = self.input_dim
		N_used = self.N_used

		with open(self.train_data_path, "rb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"]
			print("Adding {:} noise to the training data.".format(self.noise_level))
			train_input_sequence = addNoise(train_input_sequence, self.noise_level)
			N_all, dim = np.shape(train_input_sequence)
			if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
			train_input_sequence = train_input_sequence[:N_used, :input_dim]
			dt = data["dt"]
			del data

		print("Using {:}/{:} dimensions and {:}/{:} samples".format(input_dim, dim, N_used, N_all))
		if N_used > N_all: raise ValueError("Not enough samples in the training data.")

		# print("SCALING")
		train_input_sequence = self.scaler.scaleData(train_input_sequence)

		N, input_dim = np.shape(train_input_sequence)

		# Setting the reservoir variables
		self.set_random_weights()

		# TRAINING LENGTH
		tl = N - dynamics_length

		# print("TRAINING: Dynamics prerun...")
		# H_dyn = np.zeros((dynamics_length, self.getAugmentedStateSize(), 1))

		# Initialize reservoir state
		# self.set_h_zeros()
		# h = self.get_h_DL(dynamics_length=dynamics_length, train_input_sequence=train_input_sequence)

		# alternative method for training!!!
		self.newMethod(tl=tl, dynamics_length=dynamics_length, train_input_sequence=train_input_sequence)

		# alternative method for solving!!!
		print('Solving inverse problem W = (Z+rI)^-1 Y...')
		self.doNewSolving()

		# Teacher Forcing
		# self.doTeacherForcing(h=h, tl=tl, dynamics_length=dynamics_length, train_input_sequence=train_input_sequence)

		# STore something useful for plotting
		self.first_train_vec = train_input_sequence[(dynamics_length+1),:]

		# solve
		# self.doSolving()

		# acquire predictions implied by the solve
		# self.getPrediction()

		# plot things
		# self.makeNewPlots(true_state=self.X, true_residual=self.Y_all, predicted_residual=self.pred, H_memory_big=self.H_memory_big, set_name='TRAINORIGINAL')

		# raise ValueError('shortcut stoppage!')

		# print("COMPUTING PARAMETERS...")
		self.n_trainable_parameters = self.learn_memory*np.size(self.W_out_memory) + self.learn_markov*np.size(self.W_out_markov)
		self.n_model_parameters = self.learn_memory*(np.size(self.W_in) + np.size(self.W_h) + np.size(self.b_h))
		self.n_model_parameters += self.learn_markov*(np.size(self.W_in_markov) + np.size(self.b_h_markov))
		self.n_model_parameters += self.n_trainable_parameters
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))
		# print("SAVING MODEL...")
		self.saveModel()

		# plot matrix spectrum
		if self.plot_matrix_spectrum:
			plotMatrixSpectrum(self, self.W_h_effective , 'W_h_effective')
			plotMatrixSpectrum(self, self.W_in, 'B')
			plotMatrixSpectrum(self, self.W_out_markov, 'C_markov')
			plotMatrixSpectrum(self, self.W_out_memory, 'C_memory')
			# plotMatrixSpectrum(self, self.W_in @self.W_out, 'BC')

	def isWallTimeLimit(self):
		training_time = time.time() - self.start_time
		if training_time > self.reference_train_time:
			print("## Maximum train time reached. ##")
			return True
		else:
			return False

	def predictSequence(self, input_sequence):

		b_h_markov = self.b_h_markov
		W_in_markov = self.W_in_markov
		b_h = self.b_h
		W_h = self.W_h
		W_out_markov = self.W_out_markov
		W_out_memory = self.W_out_memory
		W_in = self.W_in
		W_h_effective = self.W_h_effective
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		if self.learn_memory:
			self.reservoir_size, _ = np.shape(W_h)

		# Initialize reservoir state
		self.set_h_zeros(type='tall')

		N = np.shape(input_sequence)[0]

		# PREDICTION LENGTH
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

		prediction_warm_up = []
		hidden_warm_up = []
		h = np.copy(self.h_zeros)
		for t in range(1,dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t-1], (-1,1))

			if  (not self.learn_memory or self.hidden_dynamics in ['ARNN', 'naiveRNN', 'LARNN_forward']) and self.output_dynamics in ["simpleRHS"]:
				out, h = self.predict_next(i, h)
			else:
				raise ValueError('TAKE HEED: back to the old way of doing things.')
				if self.learn_memory:
					if self.hidden_dynamics in ['ARNN', 'naiveRNN']:
						h = h + self.dt * np.tanh(W_h_effective @ h + W_in @ i + b_h)
					elif self.hidden_dynamics=='LARNN_forward':
						h = (I + self.dt*W_h_effective) @ h + self.dt * np.tanh(W_in @ i)
					else:
						h = np.tanh(W_h @ h + W_in @ i)
				if self.learn_markov:
					h_markov = np.tanh(W_in_markov @ i + b_h_markov)

				if self.learn_markov and self.learn_memory:
					h_all = np.vstack((h_markov, self.augmentHidden(h)))
				elif self.learn_markov:
					h_all = h_markov
				elif self.learn_memory:
					h_all = self.augmentHidden(h)

				if self.output_dynamics=="simpleRHS":
					out = i + self.dt * self.scaler.descaleDerivatives((W_out @ h_all).T).T
				elif self.output_dynamics=="andrewRHS":
					out = i - self.lam * self.dt * ( i - W_out @ h_all )
				else:
					out = W_out @ h_all

			prediction_warm_up.append(out)
			hidden_warm_up.append(h)

		print("\n")

		i = np.reshape(input_sequence[dynamics_length-1], (-1,1))
		target = input_sequence[dynamics_length:]
		prediction = []
		hidden = []
		for t in range(iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")

			if  (not self.learn_memory or self.hidden_dynamics in ['ARNN', 'naiveRNN', 'LARNN_forward']) and self.output_dynamics in ["simpleRHS"]:
				out, h = self.predict_next(i, h)
			else:
				raise ValueError('TAKE HEED: back to the old way of doing things.')
				if self.learn_memory:
					if self.hidden_dynamics in ['ARNN', 'naiveRNN']:
						h = h + self.dt * np.tanh(W_h_effective @ h + W_in @ i + b_h)
					elif self.hidden_dynamics=='LARNN_forward':
						h = (I + self.dt*W_h_effective) @ h + self.dt * np.tanh(W_in @ i)
					else:
						h = np.tanh(W_h @ h + W_in @ i)
				if self.learn_markov:
					h_markov = np.tanh(W_in_markov @ i + b_h_markov)

				if self.learn_markov and self.learn_memory:
					h_all = np.vstack((h_markov, self.augmentHidden(h)))
				elif self.learn_markov:
					h_all = h_markov
				elif self.learn_memory:
					h_all = self.augmentHidden(h)

				if self.output_dynamics=="simpleRHS":
					out = out + self.dt * self.scaler.descaleDerivatives((W_out @ h_all).T).T
				elif self.output_dynamics=="andrewRHS":
					out = out - self.lam * self.dt * ( out - W_out @ h_all )
				else:
					out = W_out @ h_all
			prediction.append(out)
			hidden.append(h)
			i = out
		print("\n")

		prediction = np.array(prediction)[:,:,0]
		prediction_warm_up = np.array(prediction_warm_up)[:,:,0]

		hidden = np.array(hidden)[:,:,0]
		hidden_warm_up = np.array(hidden_warm_up)[:,:,0]

		target_augment = input_sequence
		prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)
		hidden_augment = np.concatenate((hidden_warm_up, hidden), axis=0)

		return prediction, target, prediction_augment, target_augment, hidden, hidden_augment

	def predictSequenceMemoryCapacity(self, input_sequence, target_sequence):
		b_h = self.b_h
		W_h = self.W_h
		W_out = self.W_out
		W_in = self.W_in
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		self.reservoir_size, _ = np.shape(W_h)
		N = np.shape(input_sequence)[0]
		gammaI = self.gamma * sparse.eye(self.reservoir_size)
		I = sparse.eye(self.reservoir_size)

		# PREDICTION LENGTH
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

		h = np.zeros((self.reservoir_size, 1))
		for t in range(dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t], (-1,1))
			if self.hidden_dynamics=='ARNN':
				h = h + self.dt * np.tanh((W_h-W_h.T - gammaI) @ h + W_in @ i + b_h)
			elif self.hidden_dynamics=='naiveRNN':
				h = h + self.dt * np.tanh(W_h @ h + W_in @ i)
			elif self.hidden_dynamics=='LARNN_forward':
				h = (I + self.dt*(W_h-W_h.T - gammaI)) @ h + self.dt * np.tanh(W_in @ i)
			else:
				h = np.tanh(W_h @ h + W_in @ i)

		out = i
		target = target_sequence
		prediction = []
		signal = []
		for t in range(dynamics_length, dynamics_length+iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
			signal.append(i)
			if self.output_dynamics=="simpleRHS":
				out = out + self.dt * self.scaler.descaleDerivatives((W_out @ self.augmentHidden(h)).T).T
			elif self.output_dynamics=="andrewRHS":
				out = out - self.lam * self.dt * ( out - W_out @ self.augmentHidden(h) )
			else:
				out = W_out @ self.augmentHidden(h)
			prediction.append(out)

			# TEACHER FORCING
			i = np.reshape(input_sequence[t], (-1,1))
			# i = out
			if self.hidden_dynamics=='ARNN':
				h = h + self.dt * np.tanh((W_h-W_h.T - gammaI) @ h + W_in @ i + b_h)
			elif self.hidden_dynamics=='naiveRNN':
				h = h + self.dt * np.tanh(W_h @ h + W_in @ i)
			elif self.hidden_dynamics=='LARNN_forward':
				h = (I + self.dt*(W_h-W_h.T - gammaI)) @ h + self.dt * np.tanh(W_in @ i)
			else:
				h = np.tanh(W_h @ h + W_in @ i)

		prediction = np.array(prediction)[:,:,0]
		target = np.array(target)
		signal = np.array(signal)
		return prediction, target, signal

	def get_dt(self):
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			dt = data["dt"]
			del data
		return dt

	def testing(self):
		if self.loadModel()==0:
			self.testingOnTrainingSet()
			self.testingOnTestingSet()
			self.saveResults()
		return 0

	def testingOnTrainingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			testing_ic_indexes = data["testing_ic_indexes"]
			dt = data["dt"]
			del data

		with open(self.train_data_path, "rb") as file:
			data = pickle.load(file)
			train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
			del data

		if num_test_ICS==1:
			training_ic_indexes = [2*self.dynamics_length+1]
		else:
			training_ic_indexes = testing_ic_indexes
		rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.predictIndexes(train_input_sequence, training_ic_indexes, dt, "TRAIN")

		for var_name in getNamesInterestingVars():
			exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))

		# add Dt_specific performance data
		# for dt_fast in [self.dt*x for x in [2, 5, 10, 100, 200] ]:
		# 	rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.predictIndexes(train_input_sequence, training_ic_indexes, dt, "TRAIN", dt_fast=dt_fast, make_plots=False)
		# 	for var_name in getNamesInterestingVars():
		# 		exec("self.fast_dt['{:s}']['{:s}_TRAIN'] = {:s}".format(fast_dt, var_name, var_name))

		return 0

	def testingOnTestingSet(self):
		num_test_ICS = self.num_test_ICS
		with open(self.test_data_path, "rb") as file:
			data = pickle.load(file)
			testing_ic_indexes = data["testing_ic_indexes"]
			test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
			dt = data["dt"]
			del data

		rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all = self.predictIndexes(test_input_sequence, testing_ic_indexes, dt, "TEST")

		for var_name in getNamesInterestingVars():
			exec("self.{:s}_TEST = {:s}".format(var_name, var_name))

		# add Dt_specific performance data
		return 0


	def predictIndexes(self, input_sequence, ic_indexes, dt, set_name):
		num_test_ICS = self.num_test_ICS
		input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
		predictions_all = []
		hidden_all = []
		truths_all = []
		rmse_all = []
		rmnse_all = []
		num_accurate_pred_005_all = []
		num_accurate_pred_050_all = []
		for ic_num in range(num_test_ICS):
			if self.display_output == True:
				print("\n##### {:} IC {:}/{:}, {:2.3f}% #####".format(set_name, ic_num, num_test_ICS, ic_num/num_test_ICS*100), end='\r')
			ic_idx = ic_indexes[ic_num]
			input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
			prediction, target, prediction_augment, target_augment, hidden, hidden_augment = self.predictSequence(input_sequence_ic)
			prediction = self.scaler.descaleData(prediction)
			target = self.scaler.descaleData(target)
			rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, self.scaler.data_std)
			predictions_all.append(prediction)
			hidden_all.append(hidden)
			truths_all.append(target)
			rmse_all.append(rmse)
			rmnse_all.append(rmnse)
			num_accurate_pred_005_all.append(num_accurate_pred_005)
			num_accurate_pred_050_all.append(num_accurate_pred_050)
			# PLOTTING ONLY THE FIRST THREE PREDICTIONS
			# print('First target:', target_augment[0])
			if num_test_ICS==1 and set_name=='TRAIN' and not all(target_augment[0]==self.first_train_vec) and self.noise_level==0:
				raise ValueError('Training trajectories are not aligned')
			if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt, target_augment, prediction_augment, warm_up=self.dynamics_length, hidden=hidden, hidden_augment=hidden_augment)
			# plot more things
			self.makeNewPlots(true_state=target, true_residual=target, predicted_residual=prediction, set_name=set_name, ic_idx=ic_idx)


		predictions_all = np.array(predictions_all)
		hidden_all = np.array(hidden_all)
		truths_all = np.array(truths_all)
		rmse_all = np.array(rmse_all)
		rmnse_all = np.array(rmnse_all)
		num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
		num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

		# print("TRAJECTORIES SHAPES:")
		# print(np.shape(truths_all))
		# print(np.shape(predictions_all))
		rmnse_avg = np.mean(rmnse_all)
		print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
		num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
		print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
		num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
		print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))
		freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, truths_all, dt)
		print("FREQUENCY ERROR: {:}".format(error_freq))

		plotSpectrum(self, sp_true, sp_pred, freq_true, freq_pred, set_name)
		return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred, hidden_all

	def saveResults(self):

		if self.write_to_log == 1:
			logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/test.txt"
			self.results_logfile_path = logfile_test
			writeToTestLogFile(logfile_test, self)

		data = {}
		for var_name in getNamesInterestingVars():
			exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
			exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
		data["model_name"] = self.model_name
		data["num_test_ICS"] = self.num_test_ICS
		data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
		self.results_pickle_path = data_path
		self.results_data = data
		with open(data_path, "wb") as file:
			# Pickle the "data" dictionary using the highest protocol available.
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

	def loadModel(self):
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
		try:
			with open(data_path, "rb") as file:
				data = pickle.load(file)
				self.W_in_markov = data["W_in_markov"]
				self.b_h_markov = data["b_h_markov"]
				self.W_out_markov = data["W_out_markov"]
				self.W_out_memory = data["W_out_memory"]
				self.W_in = data["W_in"]
				self.W_h = data["W_h"]
				self.W_h_effective = data["W_h_effective"]
				self.b_h = data["b_h"]
				self.gamma = data["gamma"]
				self.scaler = data["scaler"]
				self.first_train_vec = data["first_train_vec"]
				del data
			return 0
		except:
			print("MODEL {:s} NOT FOUND.".format(data_path))
			return 1

	def saveModel(self):
		# print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:2.2f} minutes".format(self.total_training_time/60))

		# print("MEMORY TRACKING IN MB...")
		process = psutil.Process(os.getpid())
		memory = process.memory_info().rss/1024/1024
		self.memory = memory
		print("Script used {:} MB".format(self.memory))
		# print("SAVING MODEL...")

		if self.write_to_log == 1:
			logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
			writeToTrainLogFile(logfile_train, self)

		data = {
		"memory":self.memory,
		"n_trainable_parameters":self.n_trainable_parameters,
		"n_model_parameters":self.n_model_parameters,
		"total_training_time":self.total_training_time,
		"W_in_markov":self.W_in_markov,
		"b_h_markov":self.b_h_markov,
		"W_out_markov":self.W_out_markov,
		"W_out_memory":self.W_out_memory,
		"W_in":self.W_in,
		"W_h":self.W_h,
		"W_h_effective":self.W_h_effective,
		"b_h":self.b_h,
		"gamma":self.gamma,
		"scaler":self.scaler,
		"first_train_vec": self.first_train_vec
		}
		data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
		with open(data_path, "wb") as file:
			pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
			del data
		return 0

	def get_h_DL(self, dynamics_length, train_input_sequence):
		h = np.copy(self.h_zeros)
		for t in range(dynamics_length):
			if self.display_output == True:
				print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(train_input_sequence[t], (-1,1))
			if self.learn_memory:
				if self.hidden_dynamics in ['ARNN', 'naiveRNN']:
					if self.component_wise:
						for k in range(self.input_dim):
							h[:,k] += self.dt * np.tanh(self.W_h_effective @ h[:,k] + self.W_in @ i[k] + np.squeeze(self.b_h))
					else:
						h = h + self.dt * np.tanh(self.W_h_effective @ h + self.W_in @ i + self.b_h)
				elif self.hidden_dynamics=='LARNN_forward':
					if self.component_wise:
						for k in range(self.input_dim):
							h[:,k] = (I + self.dt*self.W_h_effective) @ h[:,k] + self.dt * np.tanh(self.W_in @ i[k])
					else:
						h = (I + self.dt*self.W_h_effective) @ h + self.dt * np.tanh(self.W_in @ i)
				else:
					if self.component_wise:
						for k in range(self.input_dim):
							h[:,k] = np.tanh(self.W_h @ h[:,k] + self.W_in @ i[k])
					else:
						h = np.tanh(self.W_h @ h + self.W_in @ i)
		# print("\n")
		return h

	def rhs(self, t0, u0):

		x_input = u0[:self.input_dim]
		h_reservoir = u0[self.input_dim:]

		f_error_markov = np.zeros(self.input_dim)
		f_error_memory = np.zeros(self.input_dim)
		if self.learn_markov:
			if self.component_wise:
				for k in range(self.input_dim):
					f_error_markov[k] = self.W_out_markov @ self.q_t(x_input[k,None])
			else:
				f_error_markov = self.W_out_markov @ self.q_t(x_input)
		if self.learn_memory:
			if self.component_wise:
				for k in range(self.input_dim):
					kth_inds = np.arange((k*self.reservoir_size),((k+1)*self.reservoir_size))
					f_error_memory[k] = self.W_out_memory @ self.augmentHidden(h_reservoir[kth_inds])
			else:
				f_error_memory = self.W_out_memory @ self.augmentHidden(h_reservoir)

		# total predicted error of rhs for x
		m = f_error_markov + f_error_memory

		# add mechanistic rhs
		if self.use_f0:
			y0 = self.scaler.descaleData(x_input)
			f0 = self.f0(t0=t0, u0=y0)
			if self.scaler_tt in ['Standard', 'standard']:
				f0 = f0 / self.scaler.data_std
			else:
				raise ValueError('not set up to undo other types of normalizations!')
		else:
			f0 = 0

		# total rhs for x
		dx = f0 + m

		# get RHS for \dot{h} hidden/reservoir state
		if self.learn_memory:
			if self.component_wise:
				dh_reservoir = np.zeros(self.input_dim*self.reservoir_size)
				for k in range(self.input_dim):
					kth_inds = np.arange((k*self.reservoir_size),((k+1)*self.reservoir_size))
					xk = x_input[k,None]
					mk = m[k,None]
					if self.rc_state_input and self.rc_error_input:
						rc_input = np.hstack((xk, mk))
					elif self.rc_state_input:
						rc_input = xk
					elif self.rc_error_input:
						rc_input = mk
					dh_reservoir[kth_inds] = np.tanh(self.W_h_effective @ h_reservoir[kth_inds] + self.W_in @ rc_input + np.squeeze(self.b_h))
			else:
				if self.rc_state_input and self.rc_error_input:
					rc_input = np.hstack((x_input, m))
				elif self.rc_state_input:
					rc_input = x_input
				elif self.rc_error_input:
					rc_input = m
				dh_reservoir = np.tanh(self.W_h_effective @ h_reservoir + self.W_in @ rc_input + np.squeeze(self.b_h))
			return np.hstack((dx, dh_reservoir))
		else:
			return dx



	def rhs_old(self, t0, u0):

		# if self.learn_markov and self.learn_memory:
		# 	raise ValueError('Cant do both markov and memory yet!!!')

		x_input = u0[:self.input_dim]
		h_reservoir = u0[self.input_dim:]

		# get RHS values for \dot{x} = C*[f_markov; f_memory]
		f_error_markov = np.zeros(self.input_dim)
		f_error_memory = np.zeros(self.input_dim)
		if self.learn_markov:
			if self.component_wise:
				for k in range(self.input_dim):
					foo = np.tanh(self.W_in_markov @ x_input[k,None] + np.squeeze(self.b_h_markov))
					f_error_markov[k] = self.W_out_markov @ foo
			else:
				foo = np.tanh(self.W_in_markov @ x_input + np.squeeze(self.b_h_markov))
				f_error_markov = self.W_out_markov @ foo
		if self.learn_memory:
			if self.component_wise:
				for k in range(self.input_dim):
					kth_inds = np.arange((k*self.reservoir_size),((k+1)*self.reservoir_size))
					f_error_memory[k] = self.W_out_memory @ self.augmentHidden(h_reservoir[kth_inds])
			else:
				f_error_memory = self.W_out_memory @ self.augmentHidden(h_reservoir)

		# concatenate RHS for \dot{x} using RC-memory terms and RF-markovian terms
		# fcorrection = np.zeros(self.input_dim)
		# if self.learn_markov and self.learn_memory:
		# 	f_all = np.vstack((f_markov, f_memory))
		# 	fcorrection = self.W_out @ f_all
		# elif self.learn_markov:
		# 	if self.component_wise:
		# 		for k in range(self.input_dim):
		# 			kth_inds = np.arange((k*self.rf_dim),((k+1)*self.rf_dim))
		# 			fcorrection[kth_inds] = self.W_out @ f_markov[kth_inds]
		# 	else:
		# 		fcorrection = self.W_out @ f_markov
		# elif self.learn_memory:
		# 	if self.component_wise:
		# 		for k in range(self.input_dim):
		# 			kth_inds = np.arange((k*self.reservoir_size),((k+1)*self.reservoir_size))
		# 			fcorrection[kth_inds] = self.W_out @ f_memory[kth_inds]
		# 	else:
		# 		fcorrection = self.W_out @ f_markov
		# else:
		# 	raise ValueError('need to learn SOMETHING duh.')

		# add mechanistic rhs
		if self.use_f0:
			y0 = self.scaler.descaleData(x_input)
			f0 = self.f0(t0=t0, u0=y0)
			if self.scaler_tt in ['Standard', 'standard']:
				f0 = f0 / self.scaler.data_std
			else:
				raise ValueError('not set up to undo other types of normalizations!')
		else:
			f0 = 0

		dx = f0 + f_error_markov + f_error_memory

		# get RHS for \dot{h} hidden/reservoir state
		if self.learn_memory:
			if self.component_wise:
				# i = np.reshape(x_input, (-1,1))
				dh_reservoir = np.zeros(self.input_dim*self.reservoir_size)
				for k in range(self.input_dim):
					kth_inds = np.arange((k*self.reservoir_size),((k+1)*self.reservoir_size))
					dh_reservoir[kth_inds] = np.tanh(self.W_h_effective @ h_reservoir[kth_inds] + self.W_in @ x_input[k,None] + np.squeeze(self.b_h))
			else:
				dh_reservoir = np.tanh(self.W_h_effective @ h_reservoir + self.W_in @ x_input + np.squeeze(self.b_h))
			return np.hstack((dx, dh_reservoir))
		else:
			return dx
