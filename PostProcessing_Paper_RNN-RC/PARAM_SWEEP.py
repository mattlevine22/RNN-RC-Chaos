#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import pickle
import glob, os
import numpy as np
import argparse
import time
import pandas as pd
import pdb

# ADDING PARENT DIRECTORY TO PATH
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
methods_dir = os.path.dirname(current_dir)+"/Methods"
sys.path.insert(0, methods_dir)
from Config.global_conf import global_params
global_utils_path = methods_dir + "/Models/Utils"
sys.path.insert(0, global_utils_path)
from global_utils import *
from plotting_utils import *


# PLOTTING
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from Utils.utils import *

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)
linewidth = 2
markersize = 10



# python3 POSTPROCESS_12b.py --system_name Lorenz96_F8GP40R40 --Experiment_Name Experiment_Daint_Large

parser = argparse.ArgumentParser()
parser.add_argument("--system_name", help="system", type=str, required=True)
parser.add_argument("--Experiment_Name", help="Experiment_Name", type=str, required=False, default=None)
args = parser.parse_args()
system_name = args.system_name
Experiment_Name = args.Experiment_Name
# system_name="Lorenz3D"
# Experiment_Name="Experiment_Daint_Large"

if system_name == "Lorenz3D":
    dt=0.01
    lambda1=1
    RDIM_VALUES = [3]
elif system_name == "Lorenz96_F8GP40R40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [35,40]
elif system_name == "Lorenz96_F10GP40R40":
    dt=0.01
    lambda1=2.27
    RDIM_VALUES = [35,40]
elif system_name == "Lorenz96_F8GP40":
    dt=0.01
    lambda1=1.68
    RDIM_VALUES = [40]

def stripVal(str, token, delim='-'):
    splitlist = str.split(delim)
    substr = [z for z in splitlist if token in z][0]
    val = substr.strip(token + '_')
    try:
        val = float(val)
    except:
        pass
    return val


def getAllModelsTrainDict(saving_path):
    modeldict = {}
    for subdir, dirs, files in os.walk(saving_path):
        for filename in files:
            # print(filename)
            if filename == "train.txt":
                filedir = os.path.join(subdir, filename)
                # print(filedir)
                with open(filedir, 'r') as file_object:
                    for line in file_object:
                        # print(line)
                        modeldict=parseLineDict(line, filename, modeldict)
    return modeldict

def getAllModelsTestDict(saving_path):
    modeldict = {}
    for subdir, dirs, files in os.walk(saving_path):
        for filename in files:
            # print(filename)
            if filename == "test.txt":
                filedir = os.path.join(subdir, filename)
                # print(filedir)
                with open(filedir, 'r') as file_object:
                    for line in file_object:
                        # print(line)
                        modeldict=parseLineDict(line, filename, modeldict)
    return modeldict


if Experiment_Name is None or Experiment_Name=="None" or global_params.cluster == "daint":
    saving_path = global_params.saving_path.format(system_name)
else:
    saving_path = global_params.saving_path.format(Experiment_Name +"/"+system_name)
logfile_path=saving_path+"/Logfiles"
print(system_name)
print(logfile_path)
val_results_path = saving_path + global_params.results_dir
train_results_path = saving_path + global_params.model_dir

fig_path = saving_path + "/Total_Results_Figures"
os.makedirs(fig_path, exist_ok=True)

model_test_dict = getAllModelsTestDict(logfile_path)
model_train_dict = getAllModelsTrainDict(logfile_path)

master_list_of_dicts = []
keep_keys = ['rmnse_avg_TEST',
                'rmnse_avg_TRAIN',
                'num_accurate_pred_005_avg_TEST',
                'num_accurate_pred_005_avg_TRAIN',
                'num_accurate_pred_050_avg_TEST',
                'num_accurate_pred_050_avg_TRAIN',
                'error_freq_TEST',
                'error_freq_TRAIN']

param_list = ['SIGMA', 'GAM', 'OD', 'HD']
NUM_ACC_PRED_STR = "num_accurate_pred_005_avg_TEST"

for model_name_key in model_test_dict:
    val_file = val_results_path + model_name_key + "/results.pickle"
    train_file = train_results_path + model_name_key + "/data.pickle"

    # print("Loading validation results...")
    val_result = pickle.load(open(val_file, "rb" ))
    predictions_all = val_result["predictions_all_TEST"]
    truths_all = val_result["truths_all_TEST"]

    # print("Loading training results...")
    train_result = pickle.load(open(train_file, "rb" ))
    scaler = train_result["scaler"]
    num_ics_deviating, num_ics_not_deviating = getNumberOfDivergentTrajectories(truths_all, predictions_all, scaler.data_std)
    new_dict = {key: val_result[key] for key in keep_keys}
    splitlist = model_name_key.split()
    for token in param_list:
        try:
            new_dict[token] = stripVal(str=model_name_key, token=token)
        except:
            pdb.set_trace()
    new_dict['name'] = model_name_key
    new_dict['t_valid'] = new_dict[NUM_ACC_PRED_STR]*dt
    new_dict['num_ics_deviating'] = num_ics_deviating
    master_list_of_dicts.append(new_dict)

df = pd.DataFrame(master_list_of_dicts)

df = df[(df['HD']=='ARNN') & (df['OD']=='simpleRHS')]

eval_keys = keep_keys + ['t_valid', 'num_ics_deviating']
for eval_type in eval_keys:
    plotHyperparamContour(df.sort_values(by=['SIGMA','GAM']), fig_path=os.path.join(fig_path,'sigma_vs_gamma' + eval_type), xkey='SIGMA', ykey='GAM', zkey=eval_type)
