#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import socket
from scipy.linalg import svdvals, eigvals
from scipy.sparse.linalg import svds as sparse_svds
from scipy.sparse.linalg import eigs as sparse_eigs


# Plotting parameters
import pdb
import matplotlib
import pandas as pd
hostname = socket.gethostname()
print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2]=='eu')  or (hostname[:5]=='daint') or (hostname[:3]=='nid')) else False
if CLUSTER: matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d

from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)


def plotTrainingLosses(model, loss_train, loss_val, min_val_error,additional_str=""):
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total_log"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), np.log(loss_train), color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), np.log(loss_val), color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, np.log(min_val_error), "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Log-Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    else:
        print("## Empty losses. Not printing... ##")



def plotAttractor(model, set_name, latent_states, ic_idx):

    print(np.shape(latent_states))
    if np.shape(latent_states)[1] >= 2:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        X = latent_states[:, 0]
        Y = latent_states[:, 1]
        epsilon = 1e-7
        # for i in range(0, len(X)-1):
        for i in range(len(X)-1):
            if np.abs(X[i+1]-X[i]) > epsilon and np.abs(Y[i+1]-Y[i]) > epsilon:
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', lw=0, length_includes_head=True, zorder=2, linestyle='')
                plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', length_includes_head=True, zorder=2)
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', shape='full', zorder=2)
        plt.plot(X, Y, 'k', linewidth = 1, label='output', zorder=1)
        plt.autoscale(enable=True, axis='both')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        plt.plot(latent_states[:-1, 0], latent_states[1:, 0], 'b', linewidth = 2.0, label='output')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()



def plotIterativePrediction(model, set_name, target, prediction, error, nerror, ic_idx, dt, truth_augment=None, prediction_augment=None, hidden=None, hidden_augment=None, warm_up=None, latent_states=None):


    if latent_states is not None:
        plotAttractor(model, set_name, latent_states, ic_idx)

    if hidden_augment is not None and hidden_augment.shape[1]>0:
        fontsize = 12
        vmin = hidden_augment.min()
        vmax = hidden_augment.max()
        # Plotting the contour plot
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
        axes = [axes]
        # fig.subplots_adjust(hspace=0.4, wspace = 0.4)
        # axes[1].set_ylabel(r"Time $t$", fontsize=fontsize)
        axes[0].set_xlabel(r"Time $t$", fontsize=fontsize)
        axes[0].set_ylabel(r"State $h$", fontsize=fontsize)
        # createContour_(fig, axes[1], hidden_augment, "Hidden", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)

        n_times, n_hidden = hidden_augment.shape
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/hidden_augment_{:}_{:}.png".format(set_name, ic_idx)
        for n in range(n_hidden):
            axes[0].plot(np.arange(n_times)*dt, hidden_augment[:,n])
        axes[0].plot(np.ones((100,1))*warm_up*dt, np.linspace(np.min(hidden_augment[:,0]), np.max(hidden_augment[:,0]), 100), 'g--', linewidth = 2.0, label='warm-up')
        plt.savefig(fig_path)
        plt.close()

    if hidden is not None and hidden.shape[1]>0:
        fontsize = 12
        vmin = hidden.min()
        vmax = hidden.max()
        # Plotting the contour plot
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
        axes = [axes]
        # fig.subplots_adjust(hspace=0.4, wspace = 0.4)
        # axes[1].set_ylabel(r"Time $t$", fontsize=fontsize)
        axes[0].set_xlabel(r"Time $t$", fontsize=fontsize)
        axes[0].set_ylabel(r"State $h$", fontsize=fontsize)
        # createContour_(fig, axes[1], hidden, "Hidden", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)

        n_times, n_hidden = hidden.shape
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/hidden_{:}_{:}.png".format(set_name, ic_idx)
        for n in range(n_hidden):
            axes[0].plot(np.arange(n_times)*dt, hidden[:,n])
        plt.savefig(fig_path)
        plt.close()

    if ((truth_augment is not None) and (prediction_augment is not None)):
        for state_idx in range(prediction_augment.shape[1]):
            fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_augment_{:}_{:}_state{:}.png".format(set_name, ic_idx, state_idx)
            plt.plot(np.arange(np.shape(prediction_augment)[0]), prediction_augment[:,state_idx], 'b', linewidth = 2.0, label='output')
            plt.plot(np.arange(np.shape(truth_augment)[0]), truth_augment[:,state_idx], 'r', linewidth = 2.0, label='target')
            plt.plot(np.ones((100,1))*warm_up, np.linspace(np.min(truth_augment[:,state_idx]), np.max(truth_augment[:,state_idx]), 100), 'g--', linewidth = 2.0, label='warm-up')
            plt.legend(loc="lower right")
            plt.savefig(fig_path)
            plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}.png".format(set_name, ic_idx)
    plt.plot(prediction, 'r--', label='prediction')
    plt.plot(target, 'g--', label='target')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_error.png".format(set_name, ic_idx)
    plt.plot(error, label='error')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_log_error.png".format(set_name, ic_idx)
    plt.plot(np.log(np.arange(np.shape(error)[0])), np.log(error), label='log(error)')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_nerror.png".format(set_name, ic_idx)
    plt.plot(nerror, label='nerror')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

    if model.input_dim >=3: createTestingContours(model, target, prediction, dt, ic_idx, set_name)


def createTestingContours(model, target, output, dt, ic_idx, set_name):
    fontsize = 12
    error = np.abs(target-output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()

    print("VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))

    # Plotting the contour plot
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    axes[0].set_ylabel(r"Time $t$", fontsize=fontsize)
    createContour_(fig, axes[0], target, "Target", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[1], output, "Output", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[2], error, "Error", fontsize, vmin_error, vmax_error, plt.get_cmap("Reds"), dt)
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_contour.png".format(set_name, ic_idx)
    plt.savefig(fig_path)
    plt.close()

def createContour_(fig, ax, data, title, fontsize, vmin, vmax, cmap, dt):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0])*dt, np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"$State$", fontsize=fontsize)
    return mp

def plotSpectrum(model, sp_true, sp_pred, freq_true, freq_pred, set_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/frequencies_{:}.png".format(set_name)
    plt.plot(freq_pred, sp_pred, 'r--', label="prediction")
    plt.plot(freq_true, sp_true, 'g--', label="target")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum [dB]')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()

def plotMatrixSpectrum(model, A, mat_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/singular_values_{:}.png".format(mat_name)
    try:
        s = svdvals(A)
    except:
        s = -np.sort(-sparse_svds(A, return_singular_vectors=False, k=min(100,min(A.shape))))
    plt.plot(s,'o')
    plt.ylabel(r'$\sigma$')
    plt.title('Singular values of {:}'.format(mat_name))
    plt.savefig(fig_path)
    plt.close()

    if A.shape[0]==A.shape[1]: #is square
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/eigenvalues_{:}.png".format(mat_name)
        try:
            eig = eigvals(A)
        except:
            eig = sparse_eigs(A, return_eigenvectors=False, k=min(1000,min(A.shape)))
        plt.plot(eig.real, eig.imag, 'o')
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.title('Eigenvalues of {:}'.format(mat_name))
        plt.savefig(fig_path)
        plt.close()

def plotMatrix(model, A, mat_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/matrix_{:}.png".format(mat_name)
    # plot matrix visualizations
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
    foo = ax[0].matshow(A, vmin=np.min(A), vmax=np.max(A), aspect='auto')
    # ax[0].axes.xaxis.set_visible(False)
    # ax[0].axes.yaxis.set_visible(False)
    ax[0].set_title('W_out')
    fig.colorbar(foo, ax=ax[0])

    sns.ecdfplot(data=np.abs(A).reshape(-1,1), ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_title('Distribution of matrix entries')
    plt.savefig(fig_path)
    plt.close()


def plotHyperparamContour(df, fig_path, xkey=None, ykey=None, zkey=None):
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6, 6))
    sc = ax.scatter(df[xkey], df[ykey], c=df[zkey])
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_title(zkey)
    plt.colorbar(sc)
    plt.savefig(fig_path)
    plt.close()

def newMethod_plotting(model, hidden, set_name, dt):
    fig, ax = plt.subplots()
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/newMethod_hidden{:}.png".format(set_name)
    n_times, n_hidden = hidden.shape
    for n in range(n_hidden):
        ax.plot(np.arange(n_times)*dt, hidden[:,n])
    plt.savefig(fig_path)
    plt.close()
