#!/bin/bash
# LASE, two-agent Cleanup, Competence gap (random spawn; talented cleaner apple reward 0.1x)  (paper Section 6.2)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_cleanup_2a.py SEED=$seed ENV_NAME="clean_up_2a_competence_gap_r01" MAP_NAME="competence_gap_r01"
done
