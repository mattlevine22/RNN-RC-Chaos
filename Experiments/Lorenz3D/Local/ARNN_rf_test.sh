#!/bin/bash

cd ../../../Methods

for DT in 1 0.5 0.1 0.2 0.05 0.01 0.005
do
python3 RUN.py esn \
--dont_redo 1 \
--dt_fast_frac $DT \
--learn_markov 1 \
--learn_memory 0 \
--rf_Win_bound 0.005 \
--rf_bias_bound 4 \
--rf_dim 2000 \
--output_dynamics simpleRHS \
--mode all \
--display_output 1 \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--RDIM 3 \
--noise_level 0 \
--scaler standard \
--scaler_derivatives no \
--regularization 0 \
--dynamics_length 1000 \
--iterative_prediction_length 2000 \
--num_test_ICS 1 \
--solver auto \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
