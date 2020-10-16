#!/bin/bash

cd ../../../Methods

for DR in {20..200..5}
do
  for F0 in 0 1
  do
  python3 RUN.py esn \
  --f0_name lds \
  --learn_markov 0 \
  --learn_memory 1 \
  --use_f0 $F0 \
  --use_tilde 1 \
  --hidden_dynamics ARNN \
  --output_dynamics simpleRHS \
  --gamma 1 \
  --mode all \
  --display_output 1 \
  --system_name coupled_lds \
  --write_to_log 1 \
  --N 100000 \
  --N_used 5000 \
  --RDIM 2 \
  --noise_level 0 \
  --scaler Standard \
  --scaler_derivatives no \
  --approx_reservoir_size $DR \
  --degree 10 \
  --radius 0.8 \
  --sigma_input 1 \
  --regularization 0.00001 \
  --dynamics_length 2000 \
  --iterative_prediction_length 2000 \
  --num_test_ICS 10 \
  --solver auto \
  --number_of_epochs 1000000 \
  --learning_rate 0.001 \
  --reference_train_time 10 \
  --buffer_train_time 0.5

  # python3 RUN.py esn \
  # --f0_name lds \
  # --learn_markov 1 \
  # --learn_memory 0 \
  # --use_f0 $F0 \
  # --hidden_dynamics ARNN \
  # --output_dynamics simpleRHS \
  # --mode all \
  # --display_output 1 \
  # --system_name coupled_lds \
  # --write_to_log 1 \
  # --N 100000 \
  # --N_used 5000 \
  # --RDIM 2 \
  # --noise_level 1 \
  # --scaler Standard \
  # --scaler_derivatives no \
  # --approx_reservoir_size 2000 \
  # --regularization 0 \
  # --dynamics_length 1000 \
  # --iterative_prediction_length 2000 \
  # --num_test_ICS 1 \
  # --solver auto \
  # --number_of_epochs 1000000 \
  # --learning_rate 0.001 \
  # --reference_train_time 10 \
  # --buffer_train_time 0.5
done
done
