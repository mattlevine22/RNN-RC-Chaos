#!/bin/bash

# parser.add_argument('--f0', type=int, default=1, help='boolean to use f0 mechanistic rhs')
# parser.add_argument('--learn_markov', type=int, default=0, help='boolean to use learn markovian correction term')
# parser.add_argument('--learn_memory', type=int, default=0, help='boolean to use learn memoryfull correction term')
# parser.add_argument('--component_wise', type=int, default=0, help='boolean to choose whether correction terms are component-wise or defined over full state space')
# parser.add_argument('--test_integrator', type=str, default='RK45')
# parser.add_argument('--system_name', type=str, default='coupled_lds_less_memory')
# parser.add_argument('--f0_name', type=str, default='lds')

REG_RF=0.01
for INT in Euler_old_fast_v2 RK45
do
  for SYS_NAME in coupled_lds_fast_memory coupled_lds_some_memory
  do
    for DR in 30 50 200
    do
      # RC only
      USE_F0=1
      REG_RC=0.0000001
      python3 pyrunner.py --use_f0=$USE_F0 --learn_markov=0 --learn_memory=1 --component_wise=0 --test_integrator=$INT --system_name=$SYS_NAME --regularization_RF=$REG_RF --regularization_RC=$REG_RC --approx_reservoir_size=$DR

      USE_F0=0
      REG_RC=0.0000001
      python3 pyrunner.py --use_f0=$USE_F0 --learn_markov=0 --learn_memory=1 --component_wise=0 --test_integrator=$INT --system_name=$SYS_NAME --regularization_RF=$REG_RF --regularization_RC=$REG_RC --approx_reservoir_size=$DR

    done
  done
done
