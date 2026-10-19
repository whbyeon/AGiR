#!/bin/bash
# RX + AGiR, Unfair Cleanup-Hard, C=100  (paper Table 1)
# Original paper wandb group: env=cleanup_4a_Amid_algo=scRw-RX_winit=random_re=50_b=1.0_wind50_gr=0.5_env=256_ns=100_es=500_nb=4
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/RX/agir_rx_cleanup_hard_4a.py SEED=$seed ENV_NAME="clean_up_4a_Amid" MAP_NAME="hard"\
  REFUSE_KWARGS.refusing_method="scRw" REFUSE_KWARGS.w_init="random"
done
