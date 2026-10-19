#!/bin/bash
# RX + AGiR, Unfair Coin Game (asym2), C=100  (paper Table 3)
# Original paper wandb group: env=coin_asym2_algo=Rw-RXsc_winit=random_re=50_b=1.0_wind50_gr=0.5_env=64_ns=500_es=500_nb=500
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/RX/agir_rx_coin_game.py SEED=$seed MAP_NAME="asym2" REFUSE_KWARGS.refusing_method="Rw" REFUSE_KWARGS.w_init="random"
done
