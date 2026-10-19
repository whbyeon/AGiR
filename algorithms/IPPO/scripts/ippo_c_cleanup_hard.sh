#!/bin/bash
# IPPO-C (common reward), Unfair Cleanup-Hard  (paper Table 1)
# Original paper wandb group: env=cleanup_4a_Amid_algo=ippo_reward=common_PS=F_num_envs=256_num_steps=100_envsteps=500_nb4
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=2
for seed in 10 20 30 40 50
do
  python algorithms/IPPO/ippo_cnn_cleanup_4a.py SEED=$seed NUM_ENVS=256 NUM_STEPS=100 NUM_MINIBATCHES=4 TOTAL_TIMESTEPS=3e8 \
  ENV_NAME="clean_up_4a_Amid" MAP_NAME="hard" ENV_KWARGS.shared_rewards=True REWARD="common" WANDB_TAGS=[COMMON_REWARD]
done
