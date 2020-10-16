#!/usr/bin/env python
import os
import glob
import pdb
import pandas as pd

# Plotting parameters
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from matplotlib import colors
from mpl_toolkits import mplot3d

import six
color_dict = dict(six.iteritems(colors.cnames))

font = {'size'   : 20}
matplotlib.rc('font', **font)


def main():
    base_dir = '/Users/matthewlevine/code_projects/RNN-RC-Chaos/Results/coupled_lds'
    log_dir = os.path.join(base_dir,'Logfiles')
    summary_dir = os.path.join(base_dir,'Summary_Figures')
    os.makedirs(summary_dir, exist_ok=True)
    str_frame = 'RNN-esn_auto-RDIM_2-N_used_5000-SIZE_{Dr}-D_10.0-RADIUS_0.8-SIGMA_1.0-DL_2000-NL_0-IPL_2000-REG_1e-05-WID_0-HD_ARNN-OD_simpleRHS-GAM_1.0-LAM_0-USETILDE_1-SCALER_Standard-DSCALER_no-BVAR_1-N_RF_2000-MARKOV_0-MEMORY_1-f0_{use_f0}'
    df = get_experiment_summary(log_dir=log_dir, str_frame=str_frame)

    metrics = [c for c in df.columns if c not in ['Dr', 'f0']]
    for y_metric in metrics:
        fig_path = os.path.join(summary_dir, y_metric)
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 12))
        sns.lineplot(ax=ax, data=df, x="Dr", y=y_metric, hue="f0")
        plt.savefig(fig_path)
        plt.close()
    return

def get_experiment_summary(log_dir, str_frame):

    my_glob = str_frame.format(Dr='*', use_f0='*')

    dirlist = glob.glob(os.path.join(log_dir,my_glob))
    master_list = []
    for dir in dirlist:
        info = dir.split('-')
        use_f0 = [x for x in info if 'f0' in x][0].split('_')[1]
        Dr = [x for x in info if 'SIZE' in x][0].split('_')[1]
        test_eval_fname = os.path.join(dir, 'test.txt')
        if not os.path.exists(test_eval_fname):
            print('Skipping', test_eval_fname)
            continue
        modeldict = {}
        with open(test_eval_fname, 'r') as file_object:
            for line in file_object:
                modeldict=parseLineDict(line, modeldict)
        foo = modeldict
        modeldict['Dr'] = int(Dr)
        modeldict['f0'] = int(use_f0)
        master_list.append(modeldict)
    df = pd.DataFrame(master_list)
    return df

def parseLineDict(line, modeldict):
    temp = line[:-1].split(":")
    modeldict={}
    for i in range(2, len(temp), 2):
        modeldict[str(temp[i])] = float(temp[i+1])
    return modeldict


if __name__ == "__main__":
    main()
