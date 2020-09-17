#!/bin/bash


# RC: discrete
source 01_ESN_auto_a.sh
cd -

# RC: naiveRNN reservoir, discrete output
source 01_ESN_auto_b0.sh
cd -

# RC: ARNN reservoir, discrete output
source 01_ESN_auto_b.sh
cd -

# RC: LARNN_forward reservoir, discrete output
source 01_ESN_auto_b2.sh
cd -

# RC: LARNN_forward reservoir, andrewRHS output (lambda=1)
source 01_ESN_auto_c.sh
cd -

# RC: LARNN_forward reservoir, simpleRHS output
source 01_ESN_auto_c2.sh
cd -

# RC: ARNN reservoir, andrewRHS output (lambda=1), longer prediction length
source 01_ESN_auto_d.sh
cd -

# RC: ARNN reservoir, simpleRHS output
source 01_ESN_auto_d2.sh
cd -

# RC: discrete reservoir, simpleRHS output [SHOULD BE VERY BASIC!!!]
source 01_ESN_auto_e.sh
cd -

# RC: discrete reservoir, andrewRHS output (lambda sweep)
source 01_ESN_auto_e2.sh
cd -
