#!/bin/bash

cd ../../../Methods

# learn component-wise RF and integrate with implicit Euler
  python3 RUN.py esn \
  --test_integrator Euler_old_fast_v2 \
  --component_wise 1 \
  --learn_memory 1 \
  --learn_markov 1 \
  --use_f0 1 \
  --solver pinv \
  --f0_name l96slow \
  --regularization_RF 0 \
  --regularization_RC 0.00001 \
  --hidden_dynamics ARNN \
  --output_dynamics simpleRHS \
  --mode all \
  --display_output 1 \
  --system_name L96M \
  --write_to_log 1 \
  --N 1000004 \
  --N_used 50000 \
  --RDIM 9 \
  --noise_level 0 \
  --scaler Standard \
  --scaler_derivatives no \
  --dynamics_length 2000 \
  --iterative_prediction_length 6000 \
  --num_test_ICS 1 \
  --number_of_epochs 1000000 \
  --learning_rate 0.001 \
  --reference_train_time 10 \
  --buffer_train_time 0.5

  # learn component-wise RF and integrate with Euler
    # python3 RUN.py esn \
    # --test_integrator Euler_fast \
    # --component_wise 1 \
    # --learn_markov 1 \
    # --use_f0 1 \
    # --rf_dim 300 \
    # --solver pinv \
    # --f0_name l96slow \
    # --learn_memory 0 \
    # --regularization_RF 0 \
    # --regularization_RC 0 \
    # --hidden_dynamics ARNN \
    # --output_dynamics simpleRHS \
    # --mode all \
    # --display_output 1 \
    # --system_name L96M \
    # --write_to_log 1 \
    # --N 1000004 \
    # --N_used 50000 \
    # --RDIM 9 \
    # --noise_level 0 \
    # --scaler Standard \
    # --scaler_derivatives no \
    # --dynamics_length 2000 \
    # --iterative_prediction_length 6000 \
    # --num_test_ICS 1 \
    # --number_of_epochs 1000000 \
    # --learning_rate 0.001 \
    # --reference_train_time 10 \
    # --buffer_train_time 0.5

    # Run f0 (only) and solve with RK45
      # python3 RUN.py esn \
      # --test_integrator RK45 \
      # --component_wise 1 \
      # --learn_markov 0 \
      # --use_f0 1 \
      # --rf_dim 300 \
      # --solver pinv \
      # --f0_name l96slow \
      # --learn_memory 0 \
      # --regularization_RF 0 \
      # --regularization_RC 0 \
      # --hidden_dynamics ARNN \
      # --output_dynamics simpleRHS \
      # --mode all \
      # --display_output 1 \
      # --system_name L96M \
      # --write_to_log 1 \
      # --N 1000004 \
      # --N_used 50000 \
      # --RDIM 9 \
      # --noise_level 0 \
      # --scaler Standard \
      # --scaler_derivatives no \
      # --dynamics_length 2000 \
      # --iterative_prediction_length 6000 \
      # --num_test_ICS 1 \
      # --number_of_epochs 1000000 \
      # --learning_rate 0.001 \
      # --reference_train_time 10 \
      # --buffer_train_time 0.5
