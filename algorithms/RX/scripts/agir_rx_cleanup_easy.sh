#!/bin/bash
# RX + AGiR, Unfair Cleanup-Easy, C=1  (paper Table 9 / Appendix G)
# Original paper wandb group: env=cleanup_4a_Alow_algo=Rw-RX_winit=random_re=50_b=1.0_wind50_gr=0.5_env=256_ns=100_es=500_nb=4
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/RX/agir_rx_cleanup_easy_4a.py SEED=$seed ENV_NAME="clean_up_4a_Alow" MAP_NAME="easy"\
  REFUSE_KWARGS.refusing_method="Rw" REFUSE_KWARGS.w_init="random"
done
