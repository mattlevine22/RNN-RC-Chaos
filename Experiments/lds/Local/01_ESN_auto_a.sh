#!/bin/bash

cd ../../../Methods


python3 RUN.py esn \
--hidden_dynamics 0 \
--output_dynamics 0 \
--gamma 0 \
--lambda 0 \
--sigma_input 1 \
--regularization 0.0000001 \
--dynamics_length 2500 \
--iterative_prediction_length 400 \
--solver auto \
--noise_level 0 \
--RDIM 2 \
--system_name lds \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--mode all \
--display_output 1 \
--scaler no \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.8 \
--num_test_ICS 1 \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
