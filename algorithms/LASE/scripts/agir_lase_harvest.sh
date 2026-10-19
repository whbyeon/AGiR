#!/bin/bash
# LASE + AGiR, Unfair Harvest, C=100  (paper Table 2, Fig 4)
# Original paper wandb group: env=harvest_u_4a_map2_algo=scRw-LASE_winit=random_re=50_b=1.0_nb=4_lr=0.0005_env=64_ns=500_es=500_gradsteps=50
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/LASE/agir_lase_harvest_4a.py SEED=$seed ENV_NAME="harvest_unfair" MAP_NAME="unfair" REFUSE_KWARGS.refusing_method="scRw" REFUSE_KWARGS.w_init="random"
done
