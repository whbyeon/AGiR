#!/bin/bash
# LASE + AGiR, Unfair Cleanup-Hard, C=100  (paper Table 1, Fig 3, Fig 5)
# Original paper wandb group: env=cleanup_4a_Amid_algo=scRw-LASE_winit=random_re=50_b=1.0_nb=4_lr=0.0005_env=256_ns=100_es=500_gs=50
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/agir_lase_cleanup_hard_4a.py SEED=$seed ENV_NAME="clean_up_4a_Amid" MAP_NAME="hard" REFUSE_KWARGS.refusing_method="scRw" REFUSE_KWARGS.w_init="random"
done
