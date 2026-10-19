#!/bin/bash
# LASE + AGiR, Unfair Coin Game (asym2), C=50  (paper Table 3)
# Original paper wandb group: env=coin_asym2_algo=scRw50-LASE_winit=random_re=50_b=1.0,0.0,1.0_slg=-1_nb=500_lr=0.0005_env=64_ns=500_es=500_gradsteps=50
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/agir_lase_coin_game.py SEED=$seed MAP_NAME="asym2" REFUSE_KWARGS.refusing_method="scRw50" REFUSE_KWARGS.w_init="random"
done
