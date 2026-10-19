#!/bin/bash

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2

for seed in 10 20 30 40 50
do
  python algorithms/RX/rx_cleanup_4a.py SEED=$seed ENV_NAME="clean_up_4a_Alow" MAP_NAME="easy"
done
