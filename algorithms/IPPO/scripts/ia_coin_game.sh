#!/bin/bash
# IA (inequity aversion, alpha=0.05 beta=0.0005), Unfair Coin Game (asym2)  (paper Table 3)
# Original paper wandb group: env=coin_asym2_algo=IA_alpha0.05_beta0.0005_env=256_ns=500_es=500
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/IPPO/ia_cnn_coins_asym.py SEED=$seed
done
