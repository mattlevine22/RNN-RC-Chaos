#!/bin/bash

cd ../../../Methods

python3 RUN_Bayesopt.py esn \
--use_tilde 1 \
--hidden_dynamics ARNN \
--output_dynamics simpleRHS \
--gamma 20 \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM 3 \
--noise_level 0 \
--scaler standard \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.5 \
--sigma_input 1 \
--regularization 0.000001 \
--dynamics_length 1000 \
--iterative_prediction_length 2000 \
--num_test_ICS 1 \
--solver auto \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
