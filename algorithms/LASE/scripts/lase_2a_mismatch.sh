#!/bin/bash
# LASE, two-agent Cleanup, Mismatch (fixed spawn against competence; 0.25x)  (paper Section 6.2)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_cleanup_2a.py SEED=$seed ENV_NAME="clean_up_2a_mismatch" MAP_NAME="mismatch"
done
