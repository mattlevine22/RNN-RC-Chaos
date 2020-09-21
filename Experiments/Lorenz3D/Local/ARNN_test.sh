#!/bin/bash

cd ../../../Methods

for RDIM in 3
do
for TIL in 1
do
for HID in ARNN
do
for OUT in simpleRHS
do
for SIG in 2 3.5 5 7.5 10
do
for GAM in 2 5 10 15 20 25
do
python3 RUN.py esn \
--dont_redo 1 \
--use_tilde $TIL \
--hidden_dynamics $HID \
--output_dynamics $OUT \
--gamma $GAM \
--lambda 0 \
--sigma_input $SIG \
--regularization 0.0000001 \
--dynamics_length 1000 \
--iterative_prediction_length 1000 \
--solver auto \
--noise_level 0 \
--RDIM $RDIM \
--system_name Lorenz3D \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--mode all \
--display_output 1 \
--scaler standard \
--approx_reservoir_size 2000 \
--degree 10 \
--radius 0.6 \
--num_test_ICS 3 \
--number_of_epochs 1000000 \
--learning_rate 0.001 \
--reference_train_time 10 \
--buffer_train_time 0.5
done
done
done
done
done
done
