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
# from scipy.linalg import lstsq as scipylstsq
# from numpy.linalg import lstsq as numpylstsq
from utils import *
import os
from plotting_utils import *
from global_utils import *
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
		self.regularization = params["regularization"]
		self.scaler_tt = params["scaler"]
		self.scaler_tt_derivatives = params["scaler_derivatives"]
		self.learning_rate = params["learning_rate"]
		self.number_of_epochs = params["number_of_epochs"]
		self.solver = str(params["solver"])
		##########################################
		self.scaler = scaler(tt=self.scaler_tt, tt_derivative=self.scaler_tt_derivatives)
		self.noise_level = params["noise_level"]
		self.model_name = self.createModelName(params)
		self.dt = self.get_dt()
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

		self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
		print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

		os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
		os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

	# def solve(self, r0, x0, solver='Euler'):

	def rhs(self, t, u):
		x_input = u[:input_dim,:]
		h_reservoir = u[input_dim:,:]

		# get RHS values for \dot{x} = C*[f_markov; f_memory]
		if self.learn_markov:
			f_markov = np.tanh(self.W_in_markov @ x_input + self.b_h_markov)
		if self.learn_memory:
			f_memory = self.augmentHidden(h_reservoir)

		# concatenate RHS for \dot{x} using RC-memory terms and RF-markovian terms
		if self.learn_markov and self.learn_memory:
			f_all = np.vstack((f_markov, f_memory))
		elif self.learn_markov:
			f_all = f_markov
		elif self.learn_memory:
			f_all = f_memory
		else:
			raise ValueError('need to learn SOMETHING duh.')
		dx = self.W_out @ f_all

		# get RHS for \dot{h} hidden/reservoir state
		if self.learn_memory:
			dh_reservoir = np.tanh(self.A @ h_reservoir + self.W_in @ x_input + self.b_h)
			return np.stack((dx, dh_reservoir))
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
		'regularization':'REG',
		#'num_test_ICS':'NICS',
		'worker_id':'WID',
		'hidden_dynamics': 'HD',
		'output_dynamics': 'OD',
		'gamma': 'GAM',
		'lambda': 'LAM',
		'use_tilde': 'USETILDE',
		'scaler': 'SCALER',
		'scaler_derivatives': 'DSCALER',
		'bias_var': 'BVAR',
		'rf_dim': 'N_RF',
		'learn_markov': 'MARKOV',
		'learn_memory': 'MEMORY'
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

		# Setting the reservoir size automatically to avoid overfitting
		if self.learn_markov:
			W_in_markov = np.random.uniform(low=-self.rf_Win_bound, high=self.rf_Win_bound, size=(self.rf_dim, self.input_dim))
			b_h_markov = np.random.uniform(low=-self.rf_bias_bound, high=self.rf_bias_bound, size=(self.rf_dim, 1))

		h_zeros = np.array([])
		if self.learn_memory:
			print("Initializing the reservoir weights...")
			nodes_per_input = int(np.ceil(self.approx_reservoir_size/input_dim))
			self.reservoir_size = int(input_dim*nodes_per_input)
			self.sparsity = self.degree/self.reservoir_size;
			print("NETWORK SPARSITY: {:}".format(self.sparsity))
			print("Computing sparse hidden to hidden weight matrix...")
			W_h = self.getSparseWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity, self.worker_id)

			# Initializing the input weights
			print("Initializing the input weights...")
			W_in = np.zeros((self.reservoir_size, input_dim))
			q = int(self.reservoir_size/input_dim)
			for i in range(0, input_dim):
				W_in[i*q:(i+1)*q,i] = self.sigma_input * (-1 + 2*np.random.rand(q))
			# if self.output_dynamics:
			# 	W_in = W_in / dt

			# Initialize the hidden bias term
			b_h = self.bias_var * np.random.randn(self.reservoir_size, 1)

			# Set the diffusion term
			gammaI = self.gamma * sparse.eye(self.reservoir_size)
			I = sparse.eye(self.reservoir_size)

			# Initialize reservoir state
			h_zeros = np.zeros((self.reservoir_size, 1))

			if self.hidden_dynamics in ['ARNN', 'LARNN_forward']:
				W_h_effective = (W_h-W_h.T - gammaI)
			else:
				W_h_effective = W_h

		# TRAINING LENGTH
		tl = N - dynamics_length

		print("TRAINING: Dynamics prerun...")
		# H_dyn = np.zeros((dynamics_length, self.getAugmentedStateSize(), 1))
		h = np.copy(h_zeros)
		for t in range(dynamics_length):
			if self.display_output == True:
				print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(train_input_sequence[t], (-1,1))
			if self.learn_memory:
				if self.hidden_dynamics in ['ARNN', 'naiveRNN']:
					h = h + self.dt * np.tanh(W_h_effective @ h + W_in @ i + b_h)
				elif self.hidden_dynamics=='LARNN_forward':
					h = (I + self.dt*W_h_effective) @ h + self.dt * np.tanh(W_in @ i)
				else:
					h = np.tanh(W_h @ h + W_in @ i)
				# H_dyn[t] = self.augmentHidden(h)

		print("\n")

		# # FOR PLOTTING THE DYNAMICS
		# dyn_plot_max = 1000
		# H_dyn_plot = H_dyn[:,:dyn_plot_max,0]
		# fig_path = self.saving_path + self.fig_dir + self.model_name + "/H_dyn_prerun_plot.png"
		# plt.plot(H_dyn_plot)
		# plt.title('Dynamics prerun')
		# plt.savefig(fig_path)
		# plt.close()

		if self.solver == "pinv":
			NORMEVERY = 10
			HTH = np.zeros((self.getAugmentedStateSize(), self.getAugmentedStateSize()))
			YTH = np.zeros((input_dim, self.getAugmentedStateSize()))

		H = []
		H_markov = []
		Y = []
		Y_true = []

		print("TRAINING: Teacher forcing...")

		matt_offset = 1 + int(self.output_dynamics in ["simpleRHS", "andrewRHS"])
		for t in range(tl - matt_offset):
			if self.display_output == True:
				print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format(t, tl, t/tl*100), end="\r")
			i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
			if self.learn_memory:
				if self.hidden_dynamics in ['ARNN', 'naiveRNN']:
					h = h + self.dt * np.tanh(W_h_effective @ h + W_in @ i + b_h)
				elif self.hidden_dynamics=='LARNN_forward':
					h = (I + self.dt*W_h_effective) @ h + self.dt * np.tanh(W_in @ i)
				else:
					h = np.tanh(W_h @ h + W_in @ i)
				# AUGMENT THE HIDDEN STATE
				h_aug = self.augmentHidden(h)
				H.append(h_aug[:,0])
			if self.learn_markov:
				h_markov = np.tanh(W_in_markov @ i + b_h_markov)
				H_markov.append(h_markov[:,0])

			# if self.learn_markov and self.learn_memory:
			# 	h_all = np.vstack((h_markov, h_aug))
			# elif self.learn_markov:
			# 	h_all = h_markov
			# elif self.learn_memory:
			# 	h_all = h_aug

			if self.output_dynamics=='simpleRHS':
				# FORWARD DIFFERENCE---infer derivative at time t+dl
				target = (np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1)) - np.reshape(train_input_sequence[t+dynamics_length], (-1,1))) / self.dt
				u0 = self.scaler.descaleData(train_input_sequence[t+dynamics_length])
				try:
					true_derivative = lorenz(t0=0, u0=u0)
					Y_true.append(true_derivative)
				except:
					# fails when RDIM is not full state space
					pass
			elif self.output_dynamics=='andrewRHS':
				# FORWARD DIFFERENCE
				target = np.reshape(train_input_sequence[t+dynamics_length], (-1,1)) + ((np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1)) - np.reshape(train_input_sequence[t+dynamics_length], (-1,1))) / (self.dt*self.lam) )
			else:
				target = np.reshape(train_input_sequence[t+dynamics_length+1], (-1,1))

			Y.append(target[:,0])
			if self.solver == "pinv" and (t % NORMEVERY == 0):
				# Batched approach used in the pinv case
				H = np.array(H)
				Y = np.array(Y)
				HTH += H.T @ H
				YTH += Y.T @ H
				H = []
				Y = []

		self.first_train_vec = train_input_sequence[(dynamics_length+1),:]
		if self.solver == "pinv" and (len(H) != 0):
			# ADDING THE REMAINING BATCH
			H = np.array(H)
			Y = np.array(Y)
			HTH+=H.T @ H
			YTH+=Y.T @ H
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))
			print(np.shape(HTH))
			print(np.shape(YTH))
		else:
			print("TEACHER FORCING ENDED.")
			print(np.shape(H))
			print(np.shape(Y))
			# print('\nmax(H)=',np.max(H))
			# print('mean(H)=',np.mean(H))
			# print('std(H)=',np.std(H))

			print('\nmax(x)=',np.max(train_input_sequence))
			print('mean(x)=',np.mean(train_input_sequence))
			print('std(x)=',np.std(train_input_sequence))

			print('\nmax(Y)=',np.max(Y))
			print('mean(Y)=',np.mean(Y))
			print('std(Y)=',np.std(Y))

		print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))

		if self.solver == "pinv":
			"""
			Learns mapping H -> Y with Penrose Pseudo-Inverse
			"""
			I = np.identity(np.shape(HTH)[1])
			pinv_ = scipypinv2(HTH + self.regularization*I)
			W_out = YTH @ pinv_

		elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
			"""
			Learns mapping H -> Y with Ridge Regression
			"""
			Y = np.array(Y)
			if self.output_dynamics=='simpleRHS':
				Y = self.scaler.scaleDerivatives(Y)
				print('\nmax(Y_norm)=',np.max(Y))
				print('mean(Y_norm)=',np.mean(Y))
				print('std(Y_norm)=',np.std(Y))

			# group Markovian-RFs and reservoir states
			H = np.array(H)
			H_markov = np.array(H_markov)
			if self.learn_markov and self.learn_memory:
				H_all = np.hstack((H,H_markov))
			elif self.learn_markov:
				H_all = H_markov
			elif self.learn_memory:
				H_all = H


			ridge = Ridge(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
			# ridge = Lasso(alpha=self.regularization, fit_intercept=False, normalize=False, copy_X=True)
			# print(np.shape(H))
			# print(np.shape(Y))
			# print("##")

			ridge.fit(H_all, Y)
			W_out = ridge.coef_
			H_all = np.array(H_all)

			if self.learn_memory:
				## Plot H
				fontsize = 12
				# Plotting the contour plot
				fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
				axes = [axes]
				axes[0].set_xlabel(r"Time $t$", fontsize=fontsize)
				axes[0].set_ylabel(r"State $h$", fontsize=fontsize)
				n_times, n_hidden = H.shape
				fig_path = self.saving_path + self.fig_dir + self.model_name + "/hidden_TRAIN_raw.png"
				for n in range(n_hidden):
					axes[0].plot(np.arange(n_times)*self.dt, H[:,n])
				plt.savefig(fig_path)
				plt.close()



			try:
				## Plot ridge-regression quality
				Y = np.array(Y)
				Y_true = np.array(Y_true)
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







		else:
			raise ValueError("Undefined solver.")

		print("FINALISING WEIGHTS...")
		self.W_out = W_out
		if self.learn_markov:
			self.b_h_markov = b_h_markov
			self.W_in_markov = W_in_markov
		else:
			self.b_h_markov = None
			self.W_in_markov = None

		if self.learn_memory:
			self.b_h = b_h
			self.W_in = W_in
			self.W_h = W_h
			self.W_h_effective = W_h_effective
		else:
			self.b_h = None
			self.W_in = None
			self.W_h = None
			self.W_h_effective = None


		print("COMPUTING PARAMETERS...")
		self.n_trainable_parameters = np.size(self.W_out)
		self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out) + np.size(self.W_in_markov) + np.size(self.b_h_markov)
		print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
		print("Total number of parameters: {}".format(self.n_model_parameters))
		print("SAVING MODEL...")
		self.saveModel()

		# plot matrix spectrum
		plotMatrix(self, self.W_out, 'W_out')
		if self.plot_matrix_spectrum:
			plotMatrixSpectrum(self, self.W_h_effective , 'W_h_effective')
			plotMatrixSpectrum(self, self.W_in, 'B')
			plotMatrixSpectrum(self, self.W_out, 'C')
			plotMatrixSpectrum(self, self.W_in @self.W_out, 'BC')

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
		W_out = self.W_out
		W_in = self.W_in
		W_h_effective = self.W_h_effective
		dynamics_length = self.dynamics_length
		iterative_prediction_length = self.iterative_prediction_length

		if self.learn_memory:
			self.reservoir_size, _ = np.shape(W_h)
			I = sparse.eye(self.reservoir_size)
			h_zeros = np.zeros((self.reservoir_size, 1))
		else:
			h_zeros = np.array([])

		N = np.shape(input_sequence)[0]

		# PREDICTION LENGTH
		if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N != iterative_prediction_length + dynamics_length")

		prediction_warm_up = []
		hidden_warm_up = []
		h = np.copy(h_zeros)
		for t in range(1,dynamics_length):
			if self.display_output == True:
				print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format(t, dynamics_length, t/dynamics_length*100), end="\r")
			i = np.reshape(input_sequence[t-1], (-1,1))

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
			hidden_warm_up.append(h_all)

		print("\n")

		i = np.reshape(input_sequence[dynamics_length-1], (-1,1))
		target = input_sequence[dynamics_length:]
		prediction = []
		hidden = []
		for t in range(iterative_prediction_length):
			if self.display_output == True:
				print("PREDICTION: T {:}/{:}, {:2.3f}%".format(t, iterative_prediction_length, t/iterative_prediction_length*100), end="\r")
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
			hidden.append(h_all)
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
				self.W_out = data["W_out"]
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
		"W_out":self.W_out,
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
