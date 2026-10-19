#!/bin/bash
# LASE, two-agent Cleanup, No gap (random spawn, equal competence)  (paper Section 6.2)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_cleanup_2a.py SEED=$seed ENV_NAME="clean_up_2a_no_gap" MAP_NAME="no_gap"
done
