#!/bin/bash
# IPPO-I (individual reward), Unfair Coin Game (asym2)  (paper Table 3)
# Original paper wandb group: env=coin_asym2_algo=ippo_reward=individual_env=256_ns=500_es=500
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/IPPO/ippo_ind_cnn_coins_asym.py SEED=$seed
done
