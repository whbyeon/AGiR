#!/bin/bash
# LASE (vanilla, no imagination module), Unfair Cleanup-Hard  (paper Table 1)
# Original paper wandb group: env=cleanup_4a_Amid_algo=LASE-NI_slg=-1_nb=4_lr=0.0005_num_envs=256_num_steps=100_envsteps=500_gradsteps=50
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_cleanup_4a.py SEED=$seed ENV_NAME="clean_up_4a_Amid" MAP_NAME="hard"
done
