#!/bin/bash

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2

for seed in 10 20 30 40 50
do
  python algorithms/RX/rx_harvest_4a.py SEED=$seed ENV_NAME="harvest_unfair" MAP_NAME="unfair"
done
