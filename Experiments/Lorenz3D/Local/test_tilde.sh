#!/bin/bash

cd ../../../Methods

for RDIM in 3 2
do
for TIL in 0 1
do
python3 RUN.py esn \
--dont_redo 1 \
--use_tilde $TIL \
--hidden_dynamics 0 \
--output_dynamics 0 \
--gamma 0.0001 \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM $RDIM \
--noise_level 1 \
--scaler Standard \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.5 \
--sigma_input 1 \
--regularization 0.000001 \
--dynamics_length 1000 \
--iterative_prediction_length 1000 \
--num_test_ICS 1 \
--solver auto \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done
