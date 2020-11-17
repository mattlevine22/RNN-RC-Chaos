#!/usr/bin/env python
import os
import argparse
import subprocess
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--use_f0', type=int, default=1, help='boolean to use f0 mechanistic rhs')
parser.add_argument('--learn_markov', type=int, default=0, help='boolean to use learn markovian correction term')
parser.add_argument('--learn_memory', type=int, default=0, help='boolean to use learn memoryfull correction term')
parser.add_argument('--component_wise', type=int, default=0, help='boolean to choose whether correction terms are component-wise or defined over full state space')
parser.add_argument('--test_integrator', type=str, default='RK45')
parser.add_argument('--system_name', type=str, default='coupled_lds_less_memory')
parser.add_argument('--f0_name', type=str, default='lds')
parser.add_argument('--regularization_RF', type=float, default=0)
parser.add_argument('--regularization_RC', type=float, default=0)
parser.add_argument('--approx_reservoir_size', type=int, default=2000)

FLAGS = parser.parse_args()

BASE_STR = '''python3 RUN.py esn \
  --dont_redo 0 \
  --gamma 1 \
  --rf_dim 300 \
  --solver pinv \
  --hidden_dynamics ARNN \
  --output_dynamics simpleRHS \
  --mode all \
  --display_output 1 \
  --write_to_log 1 \
  --N 1000004 \
  --N_used 50000 \
  --RDIM 2 \
  --noise_level 0 \
  --scaler Standard \
  --scaler_derivatives no \
  --dynamics_length 12000 \
  --iterative_prediction_length 20000 \
  --num_test_ICS 2 \
  --number_of_epochs 1000000 \
  --learning_rate 0.001 \
  --reference_train_time 10 \
  --buffer_train_time 0.5'''


def main(base_str=BASE_STR):
    os.chdir('../../../Methods')
    for key, val in FLAGS.__dict__.items():
        base_str += ' --{key} {val}'.format(key=key, val=val)
    subprocess.run(base_str, shell=True)
    return


if __name__ == "__main__":
    main()
