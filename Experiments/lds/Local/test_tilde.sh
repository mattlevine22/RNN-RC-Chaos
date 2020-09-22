#!/bin/bash

cd ../../../Methods

for TIL in 1 0
do
for HID in ARNN
do
for OUT in simpleRHS
do
for RDIM in 2 1
do
  for DR in 2000
  do
    for GAM in 0 1
    do
python3 RUN.py esn \
--dont_redo 1 \
--use_tilde $TIL \
--hidden_dynamics $HID \
--output_dynamics $OUT \
--gamma $GAM \
--lambda 0 \
--sigma_input 1 \
--regularization 0.0000001 \
--dynamics_length 2500 \
--iterative_prediction_length 2000 \
--solver auto \
--noise_level 0 \
--RDIM $RDIM \
--system_name lds \
--write_to_log 1 \
--N 100000 \
--N_used 5000 \
--mode all \
--display_output 1 \
--scaler standard \
--approx_reservoir_size $DR \
--degree 10 \
--radius 0.8 \
--num_test_ICS 1 \
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
