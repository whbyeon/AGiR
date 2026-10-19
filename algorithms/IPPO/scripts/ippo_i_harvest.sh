#!/bin/bash
# IPPO-I (individual reward), Unfair Harvest  (paper Table 2)
# Original paper wandb group: env=harvest_u_4a_map2_algo=ippo_reward=individual_num_envs=64_num_steps=500_envsteps=500
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/IPPO/ippo_cnn_harvest_4a.py SEED=$seed ENV_NAME="harvest_unfair" MAP_NAME="unfair" ENV_KWARGS.shared_rewards=False REWARD="individual"
done
