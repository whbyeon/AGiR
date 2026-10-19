#!/bin/bash
# RX (Reward Exchange, vanilla), Unfair Coin Game (asym2)  (paper Table 3)
# Original paper wandb group: env=coin_asym2_asym2_RX_algo=RX_gr=0.5_env=64_ns=500_es=500_nb=500
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/RX/rx_coin_game.py SEED=$seed MAP_NAME="asym2"
done
