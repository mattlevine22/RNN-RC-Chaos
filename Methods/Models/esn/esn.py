#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
				Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from scipy.integrate import solve_ivp, ode
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
		else:
			self.f0 = 0

		self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
		print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

		os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

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
			sol = solve_ivp(fun=self.rhs, t_span=t_span, y0=u0, t_eval=t_eval, max_step=self.dt)
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

	def rhs(self, t0, u0):

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
		'dt_fast_frac': 'DTF'
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
		print("WEIGHT INIT")
		W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id)
		# W = sparse.random(sizex, sizey, density=sparsity, random_state=worker_id, data_rvs=np.random.randn)
		# Sparse matrix with elements between -1 and 1
		# W.data *=2
		# W.data -=1
		# to print the values do W.A
		print("EIGENVALUE DECOMPOSITION")
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
			if self.component_wise:
				W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rf_dim, 1))
			else:
				W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rf_dim, self.input_dim))
			b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rf_dim, 1))

			self.W_in_markov = W_in_markov
			self.b_h_markov = b_h_markov

		# initialize Reservoir random terms for RC
		if self.learn_memory:
			print("Initializing the reservoir weights...")
			if self.component_wise:
				self.reservoir_size = self.approx_reservoir_size
			else:
				nodes_per_input = int(np.ceil(self.approx_reservoir_size/self.input_dim))
				self.reservoir_size = int(self.input_dim*nodes_per_input)
			self.sparsity = self.degree/self.reservoir_size;
			print("NETWORK SPARSITY: {:}".format(self.sparsity))
			print("Computing sparse hidden to hidden weight matrix...")
			W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)

			# Initializing the input weights
			print("Initializing the input weights...")
			if self.component_wise:
				W_in = np.zeros((self.reservoir_size, 1))
				q = self.reservoir_size
				for i in range(0, 1):
					W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))
			else:
				W_in = np.zeros((self.reservoir_size, self.input_dim))
				q = int(self.reservoir_size/self.input_dim)
				for i in range(0, self.input_dim):
					W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))
			# if self.output_dynamics:
			# 	W_in = W_in / dt

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
		print("\n")
		return h

	def doTeacherForcing(self, train_input_sequence, tl, dynamics_length, h):

		matt_offset = 1 + int(self.output_dynamics in ["simpleRHS", "andrewRHS"])
		n_range = tl - matt_offset

		self.H_markov = []
		self.H_memory = []
		H_memory = []
		H_markov = []
		Y = []

		Y_true = []
		Y_all = []
		X = [] # true ground states

		HTH = None
		YTH = None
		HmarkTHmark = None
		YTHmark = None
		HmarkTHmem = None

		# if self.solver == "pinv":
		NORMEVERY = 10
		if self.learn_memory:
			HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
			if self.component_wise:
				YTH = np.zeros((1, self.getAugmentedStateSize()))
				self.H_aug_memory_big = np.zeros( (n_range, self.getAugmentedStateSize(), self.input_dim) )
				self.H_memory_big = np.zeros( (n_range, self.getAugmentedStateSize(), self.input_dim) )
			else:
				YTH = np.zeros((self.input_dim, self.getAugmentedStateSize()))
				self.H_aug_memory_big = np.zeros( (n_range, self.getAugmentedStateSize()) )
				self.H_memory_big = np.zeros( (n_range, self.getAugmentedStateSize()) )
		if self.learn_markov:
			HmarkTHmark = np.zeros( (self.rf_dim, self.rf_dim))
			if self.component_wise:
				YTHmark = np.zeros( (1, self.rf_dim))
				self.H_markov_big = np.zeros( (n_range, self.rf_dim, self.input_dim) )
			else:
				YTHmark = np.zeros( (self.input_dim, self.rf_dim))
				self.H_markov_big = np.zeros( (n_range, self.rf_dim) )
		if self.learn_markov and self.learn_memory:
			HmarkTHmem = np.zeros( (self.rf_dim, self.getAugmentedStateSize()))


		print("TRAINING: Teacher forcing...")

		for t in range(n_range):
			if self.display_output == True:
				print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t, tl, t/tl*100), end="\r")
			i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))

			# MEMORY / RC SECTION
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
							h[:,k] = np.tanh(W_h @ h[:,k] + self.W_in @ i[k])
					else:
						h = np.tanh(W_h @ h + self.W_in @ i)
				# AUGMENT THE HIDDEN STATE
				if self.component_wise:
					for k in range(self.input_dim):
						h_aug = self.augmentHidden(h[:,k,None])
						H_memory.append(h_aug[:,0])
						self.H_aug_memory_big[t,:,k] = h_aug[:,0]
						self.H_memory_big[t,:,k] = h[:,k]
				else:
					h_aug = self.augmentHidden(h)
					H_memory.append(h_aug[:,0])
					self.H_aug_memory_big[t,:] = h_aug[:,0]
					self.H_memory_big[t,:] = h[:,0]

			# MARKOV / RF SECTION
			if self.learn_markov:
				if self.component_wise:
					for k in range(self.input_dim):
						h_markov = np.tanh(self.W_in_markov @ i[k,None] + self.b_h_markov)
						H_markov.append(h_markov[:,0])
						self.H_markov_big[t,:,k] = h_markov[:,0]
				else:
					h_markov = np.tanh(self.W_in_markov @ i + self.b_h_markov)
					H_markov.append(h_markov[:,0])
					self.H_markov_big[t,:] = h_markov[:,0]

			# TARGET DATA SECTION
			if self.output_dynamics=='simpleRHS':
				# FORWARD DIFFERENCE---infer derivative at time t+dl
				t0=0
				u0 = self.scaler.descaleData(train_input_sequence[t+dynamics_length])
				if self.use_f0:
					f0 = self.f0(t0=t0, u0=u0)
					f0 = f0 / self.scaler.data_std
					f0 = np.reshape(f0, (-1,1))
				else:
					f0 = 0
				target = (np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1)) - np.reshape(train_input_sequence[t+dynamics_length], (-1,1))) / self.dt - f0
				try:
					true_derivative = lorenz(t0=t0, u0=u0)
					Y_true.append(true_derivative)
				except:
					# fails when RDIM is not full state space
					pass
			elif self.output_dynamics=='andrewRHS':
				# FORWARD DIFFERENCE
				target = np.reshape(train_input_sequence[t+dynamics_length], (-1,1)) + ((np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1)) - np.reshape(train_input_sequence[t+dynamics_length], (-1,1))) / (self.dt*self.lam) )
			else:
				target = np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1))

			# STORE COLLECTED DATA IN NEW STRUCTURES
			if self.component_wise:
				for k in range(self.input_dim):
					Y.append(target[k,0])
			else:
				Y.append(target[:,0])

			Y_all.append(target[:,0])
			X.append(train_input_sequence[t+dynamics_length,:])



			self.H_markov += H_markov
			self.H_memory += H_memory

			last_batch = t==(n_range - 1)
			if self.solver == "pinv" and ((t % NORMEVERY == 0) or last_batch):
				# Batched approach used in the pinv case
				Y = np.array(Y)
				# RC-only stuff
				if self.learn_memory:
					H_memory = np.array(H_memory)
					HTH += H_memory.T @ H_memory
					YTH += Y.T @ H_memory

				# RF-only stuff
				if self.learn_markov:
					H_markov = np.array(H_markov)
					HmarkTHmark += H_markov.T @ H_markov
					YTHmark += Y.T @ H_markov

				# RC + RF joint stuff
				if self.learn_memory and self.learn_markov:
					HmarkTHmem += H_markov.T @ H_memory

				Y = []
				H_markov = []
				H_memory = []


		print("TEACHER FORCING ENDED.")
		print('\nmax(x)=',np.max(train_input_sequence))
		print('mean(x)=',np.mean(train_input_sequence))
		print('std(x)=',np.std(train_input_sequence))

		Y_all = np.array(Y_all)
		print('\nmax(Y)=',np.max(Y_all))
		print('mean(Y)=',np.mean(Y_all))
		print('std(Y)=',np.std(Y_all))


		# save things to self
		self.HTH = HTH
		self.YTH = YTH
		self.HmarkTHmark = HmarkTHmark
		self.YTHmark = YTHmark
		self.HmarkTHmem = HmarkTHmem
		self.Y_all = np.array(Y_all)
		self.X = np.array(X)
		self.Y_true = Y_true

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

	def doSolving(self):

		print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))

		HTH = self.HTH
		YTH = self.YTH
		HmarkTHmem = self.HmarkTHmem
		YTHmark = self.YTHmark
		HmarkTHmark = self.HmarkTHmark

		if self.solver == "pinv":
			"""
			Learns mapping H -> Y with Penrose Pseudo-Inverse
			"""
			if self.learn_memory and self.learn_markov:
				Z = np.hstack( ( np.vstack( (HTH, HmarkTHmem) ), np.vstack( (HmarkTHmem.T, HmarkTHmark) ) ) )
				regI = np.identity(Z.shape[0])
				regI[:self.reservoir_size,:self.reservoir_size] *= self.regularization_RC
				regI[self.reservoir_size:,self.reservoir_size:] *= self.regularization_RF
				pinv_ = scipypinv2(Z + regI)

				YTH_stack = np.hstack( (YTH, YTHmark) )
				W_out_all = YTH_stack @ pinv_
				W_out_memory = W_out_all[:,:self.reservoir_size]
				W_out_markov = W_out_all[:,self.reservoir_size:]

			elif self.learn_markov:
				I = np.identity(np.shape(HmarkTHmark)[1])
				pinv_ = scipypinv2(HmarkTHmark + self.regularization_RF*I)
				W_out_markov = YTHmark @ pinv_

			elif self.learn_memory:
				I = np.identity(np.shape(HTH)[1])
				pinv_ = scipypinv2(HTH + self.regularization_RC*I)
				W_out_memory = YTH @ pinv_

		elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
			"""
			Learns mapping H -> Y with Ridge Regression
			"""
			Y = np.array(self.Y_all)
			if self.output_dynamics=='simpleRHS':
				Y = self.scaler.scaleDerivatives(Y)
				print('\nmax(Y_norm)=',np.max(Y))
				print('mean(Y_norm)=',np.mean(Y))
				print('std(Y_norm)=',np.std(Y))

			# group Markovian-RFs and reservoir states
			if self.learn_markov and self.learn_memory:
				raise ValueError('Not yet ready to jointly learn both markov and memory terms with ridge regression')
				# H_all = np.hstack((H,H_markov))
			elif self.learn_markov:
				ridge = Ridge(alpha=self.regularization_RF, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
				ridge.fit(self.H_markov, Y)
				W_out_markov = ridge.coef_
			elif self.learn_memory:
				ridge = Ridge(alpha=self.regularization_RC, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
				ridge.fit(self.H_memory, Y)
				W_out_memory = ridge.coef_

		else:
			raise ValueError("Undefined solver.")

		print("FINALISING WEIGHTS...")
		if self.learn_markov:
			self.W_out_markov = W_out_markov
			plotMatrix(self, self.W_out_markov, 'W_out_markov')
		if self.learn_memory:
			self.W_out_memory = W_out_memory
			plotMatrix(self, self.W_out_memory, 'W_out_memory')


	def makeTrainPlots(self):
		H_markov = np.array(self.H_markov)
		H_memory = np.array(self.H_memory)

		## Plot original hidden dynamics
		if self.learn_memory:
			## Plot H
			fig_path = self.saving_path + self.fig_dir + self.model_name + "/hidden_TRAIN_raw.png"
			fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
			ax.set_xlabel(r"Time $t$", fontsize=12)
			ax.set_ylabel(r"State $h$", fontsize=12)
			n_times = self.H_memory_big.shape[0]
			n_hidden = self.H_memory_big.shape[1]
			for n in range(n_hidden):
				if self.component_wise:
					for k in range(self.input_dim):
						ax.plot(np.arange(n_times)*self.dt, self.H_memory_big[:,n,k])
				else:
					ax.plot(np.arange(n_times)*self.dt, self.H_memory_big[:,n])
			plt.savefig(fig_path)

		## Plot the learned markovian function
		# Treat states as interchangeable:
		# Plot X_k vs (Y_k prediction, Y_k truth)
		fig_path = self.saving_path + self.fig_dir + self.model_name + "/statewise_training_fits.png"
		fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
		ax.set_xlabel(r"State $X_k$", fontsize=12)
		ax.set_ylabel(r"Correction $Y_k$", fontsize=12)
		X_unlist = np.reshape(self.X, (-1,1))
		ax.scatter(X_unlist, np.reshape(self.Y_all, (-1,1)), color='gray', s=10, alpha=0.8, label='inferred residuals')
		ax.scatter(X_unlist, np.reshape(self.pred, (-1,1)), color='red', marker='+', s=3, label='fitted residuals')
		ax.legend()
		plt.savefig(fig_path)
		plt.close()

		if self.solver == "pinv":
			"""
			Learns mapping H -> Y with Penrose Pseudo-Inverse
			"""


		elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
			if self.learn_memory:
				## Plot H
				fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
				axes = [axes]
				axes[0].set_xlabel(r"Time $t$", fontsize=12)
				axes[0].set_ylabel(r"State $h$", fontsize=12)
				n_times, n_hidden = H_memory.shape
				fig_path = self.saving_path + self.fig_dir + self.model_name + "/hidden_TRAIN_raw.png"
				for n in range(n_hidden):
					axes[0].plot(np.arange(n_times)*self.dt, H_memory[:,n])
				plt.savefig(fig_path)
				plt.close()


			try:
				## Plot ridge-regression quality
				Y = np.array(self.Y_all)
				Y_true = np.array(self.Y_true)
				ridge_predict = ridge.predict(H_all)
				fig_path = self.saving_path + self.fig_dir + self.model_name + "/ridge_fit_TRAIN_true.png"
				fig, axes = plt.subplots(nrows=1, ncols=1+Y.shape[1],figsize=(5*(1+Y.shape[1]), 6))
				for ax_ind in range(Y.shape[1]):
					axes[ax_ind].plot(Y[:,ax_ind], 'o', color='green', label='data')
					axes[ax_ind].plot(ridge_predict[:,ax_ind], '+', color='magenta', label='Ridge Predictions')
					axes[ax_ind].legend(loc="lower right")
				if self.output_dynamics=='simpleRHS':
					prediction = self.scaler.descaleDerivatives(ridge_predict)
					target = self.scaler.descaleDerivatives(Y)
					std = self.scaler.derivative_std
				else:
					prediction = self.scaler.descaleData(ridge_predict)
					target = self.scaler.descaleData(Y)
					std = self.scaler.data_std
				rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, std)
				axes[-1].plot(rmnse)
				axes[-1].set_title('Training Error Sequence')
				plt.savefig(fig_path)
				plt.close()

				# Plot quality of forward-difference
				try:
					if self.output_dynamics=='simpleRHS':
						fig_path = self.saving_path + self.fig_dir + self.model_name + "/forward_difference_eval.png"
						fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
						if self.output_dynamics=='simpleRHS':
							Y_descaled = self.scaler.descaleDerivatives(Y)
							Y_true_descaled = self.scaler.descaleDerivatives(Y_true)
						else:
							Y_descaled = self.scaler.descaleData(Y)
							Y_true_descaled = self.scaler.descaleData(Y_true)
						axes[0].plot(np.linalg.norm(Y_descaled - Y_true_descaled, axis=1))
						axes[0].set_title('MSE sequence')
						axes[1].plot(np.sum(Y_descaled - Y_true_descaled, axis=1))
						axes[1].set_title('Absolute Error sequence')
						fig.suptitle('Forward-Difference Evaluation')
						plt.savefig(fig_path)
						plt.close()
				except:
					# fails when RDIM is not full state space
					pass

				## Plot fitted trajectories from ridge fit
				n_times, n_states = Y.shape
				if self.output_dynamics in ["simpleRHS", "andrewRHS"]:
					ridge_predict_traj = np.zeros((n_times, n_states))
					out = train_input_sequence[dynamics_length]
					for t in range(n_times):
						if self.output_dynamics=="simpleRHS":
							out += self.dt * self.scaler.descaleDerivatives((W_out @ H_all[t,:]).T).T
						elif self.output_dynamics=="andrewRHS":
							out = out - self.lam * self.dt * ( out - W_out @ H_all[t,:] )
						ridge_predict_traj[t,:] = out
				else:
					ridge_predict_traj = ridge_predict

				true_traj = train_input_sequence[(dynamics_length+1):]

				print('First true traj:', true_traj[0,:])
				self.first_train_vec = true_traj[0,:]
				fig_path = self.saving_path + self.fig_dir + self.model_name + "/ridge_trajectories_TRAIN_true.png"
				fig, axes = plt.subplots(nrows=n_states, ncols=1,figsize=(12, 12), squeeze=False)
				for n in range(n_states):
					axes[n,0].plot(true_traj[:n_times,n], 'o', label='data')
					axes[n,0].plot(ridge_predict_traj[:n_times,n], '+', label='Ridge Predictions')
					axes[n,0].legend(loc="lower right")
				fig.suptitle('Training Trajectories Sequence')
				plt.savefig(fig_path)
				plt.close()
			except:
				print('Unable to plot matt extra stuff')

		# if self.component_wise:
		# 	pdb.set_trace()
		# 	fig_path = self.saving_path + self.fig_dir + self.model_name + "/component_wise_fit.png"
		# 	fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8, 8))
		# 	X = np.array(X)
		# 	Y = np.array(Y)
		# 	ax.scatter(X, Y, color='gray', label='data')
		# 	if self.learn_markov:
		# 		ax.plot(W_out_markov @ H_markov, Y, color='blue', label='RF')
		# 	ax.legend()
		# 	plt.savefig(fig_path)
		# 	plt.close()


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
			print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
			train_input_sequence = addNoise(train_input_sequence, self.noise_level)
			N_all, dim = np.shape(train_input_sequence)
			if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
			train_input_sequence = train_input_sequence[:N_used, :input_dim]
			dt = data["dt"]
			del data

		print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
		if N_used > N_all: raise ValueError("Not enough samples in the training data.")
		print("SCALING")

		train_input_sequence = self.scaler.scaleData(train_input_sequence)

		N, input_dim = np.shape(train_input_sequence)

		# Setting the reservoir variables
		self.set_random_weights()

		# TRAINING LENGTH
		tl = N - dynamics_length

		print("TRAINING: Dynamics prerun...")
		# H_dyn = np.zeros((dynamics_length, self.getAugmentedStateSize(), 1))

		# Initialize reservoir state
		self.set_h_zeros()
		h = self.get_h_DL(dynamics_length=dynamics_length, train_input_sequence=train_input_sequence)

		# Teacher Forcing
		self.doTeacherForcing(h=h, tl=tl, dynamics_length=dynamics_length, train_input_sequence=train_input_sequence)

		# STore something useful for plotting
		self.first_train_vec = train_input_sequence[(dynamics_length+1),:]

		# solve
		self.doSolving()

		# acquire predictions implied by the solve
		self.getPrediction()

		# plot things
		self.makeTrainPlots()



		print("COMPUTING PARAMETERS...")
		self.n_trainable_parameters = np.size(self.W_out_memory) + np.size(self.W_out_markov)
		self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out_memory) + np.size(self.W_out_markov) + np.size(self.W_in_markov) + np.size(self.b_h_markov)
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))
		print("SAVING MODEL...")
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
		print("\n")

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

		print("\n")
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
				print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num/num_test_ICS*100))
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
			print('First target:', target_augment[0])
			if num_test_ICS==1 and set_name=='TRAIN' and not all(target_augment[0]==self.first_train_vec) and self.noise_level==0:
				raise ValueError('Training trajectories are not aligned')
			if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt, target_augment, prediction_augment, warm_up=self.dynamics_length, hidden=hidden, hidden_augment=hidden_augment)

		predictions_all = np.array(predictions_all)
		hidden_all = np.array(hidden_all)
		truths_all = np.array(truths_all)
		rmse_all = np.array(rmse_all)
		rmnse_all = np.array(rmnse_all)
		num_accurate_pred_005_all = np.array(num_accurate_pred_005_all)
		num_accurate_pred_050_all = np.array(num_accurate_pred_050_all)

		print("TRAJECTORIES SHAPES:")
		print(np.shape(truths_all))
		print(np.shape(predictions_all))
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
		print("Recording time...")
		self.total_training_time = time.time() - self.start_time
		print("Total training time is {:}".format(self.total_training_time))

		print("MEMORY TRACKING IN MB...")
		process = psutil.Process(os.getpid())
		memory = process.memory_info().rss/1024/1024
		self.memory = memory
		print("Script used {:} MB".format(self.memory))
		print("SAVING MODEL...")

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
