#!/bin/bash

cd ../../../Methods

# Observations:
# 1. Without bias, standard-normalization fails, whereas no-norm is good
# 2. With bias, standard normalization is rescued.
# 3. With bias, no-norm is even better than before, and is better than standard-norm w/ bias.
for BV in 0.01 1 0
do
python3 RUN.py esn \
--bias_var $BV \
--scaler standard \
--sigma_input 8 \
--gamma 20 \
--dont_redo 1 \
--use_tilde 1 \
--hidden_dynamics ARNN \
--output_dynamics simpleRHS \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM 3 \
--noise_level 0 \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.5 \
--regularization 0.0000001 \
--dynamics_length 1000 \
--iterative_prediction_length 2000 \
--num_test_ICS 1 \
--solver auto \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5

python3 RUN.py esn \
--dont_redo 1 \
--bias_var $BV \
--scaler no \
--sigma_input 1 \
--gamma 20 \
--dont_redo 1 \
--use_tilde 1 \
--hidden_dynamics ARNN \
--output_dynamics simpleRHS \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM 3 \
--noise_level 0 \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.5 \
--regularization 0.0000001 \
--dynamics_length 1000 \
--iterative_prediction_length 2000 \
--num_test_ICS 1 \
--solver auto \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
