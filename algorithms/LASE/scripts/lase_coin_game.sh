#!/bin/bash
# LASE (vanilla, no imagination module), Unfair Coin Game (asym2)  (paper Table 3)
# Original paper wandb group: env=coin_asym2_algo=LASE-NI_slg=-1_nb=500_lr=0.0005_num_envs=64_num_steps=500_envsteps=500_gradsteps=50
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_coin_game.py SEED=$seed
done
