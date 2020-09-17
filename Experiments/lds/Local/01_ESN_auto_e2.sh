#!/bin/bash

cd ../../../Methods

for LAM in 0.5 0.8
do
  for HID in 0
  do
python3 RUN.py esn \
--hidden_dynamics $HID \
--output_dynamics andrewRHS \
--gamma 0 \
--lambda $LAM \
--sigma_input 1 \
--regularization 0.0000001 \
--dynamics_length 2500 \
--iterative_prediction_length 2000 \
--solver auto \
--noise_level 0 \
--RDIM 2 \
--system_name lds \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--mode all \
--display_output 1 \
--scaler standard \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.8 \
--num_test_ICS 1 \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done
