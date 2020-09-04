#!/bin/bash

cd ../../../Methods



python3 RUN.py esn \
--euler_hidden 1 \
--euler_output 1 \
--gamma 0.1 \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM 3 \
--noise_level 1 \
--scaler Standard \
--approx_reservoir_size 1000 \
--degree 10 \
--radius 0.9 \
--sigma_input 1 \
--regularization 0.000001 \
--dynamics_length 200 \
--iterative_prediction_length 2000 \
--num_test_ICS 1 \
--solver pinv \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
