#!/bin/bash
# LASE + AGiR (C=1), two-agent Cleanup, Spawn gap  (paper Section 6.2)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/agir_lase_cleanup_2a.py SEED=$seed ENV_NAME="clean_up_2a_spawn_gap" MAP_NAME="spawn_gap" REFUSE_KWARGS.refusing_method="scRw" REFUSE_KWARGS.w_init="random"
done
