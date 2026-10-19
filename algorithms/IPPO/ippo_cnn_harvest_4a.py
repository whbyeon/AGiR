"""Based on PureJaxRL and JaxMARL. PPO baselines (IPPO and inequity aversion)."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import pdb

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    gift_reward: jnp.ndarray
    gift_given: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        if config["PARAMETER_SHARING"]:
            pi, value = network.apply(params, obs_batch)
            action = pi.sample(seed=key_a0)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )
        else:
            env_act = {}
            for i in range(env.num_agents):
                pi, value = network[i].apply(params[i], obs_batch)
                action = pi.sample(seed=key_a0)
                env_act[env.agents[i]] = action


        env_act = {k: v.squeeze() for k, v in env_act.items()}


        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # Initialize policy and value networks.
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # Initialize parallel environments.
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Training loop.
        def _update_step(runner_state, unused):
            # Collect rollout transitions.
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state


                rng, _rng = jax.random.split(rng)

                if config["PARAMETER_SHARING"]:

                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)


                    print("input_obs_shape", obs_batch.shape)       # batchified order (A*E, obs_dim) due to transpose above

                    pi, value = network.apply(train_state.params, obs_batch)        # value: batchified order, since obs_batch is batchified
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    for i in range(env.num_agents):
                        print("input_obs_shape", obs_batch[i].shape)
                        pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)


                env_act = [v for v in env_act.values()]


                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                if config["PARAMETER_SHARING"]:


                    reward_flat = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()


                    gift_reward = jnp.zeros_like(reward_flat)
                    gift_given = jnp.zeros_like(reward_flat)


                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)      # (E*A,)

                    transition = Transition(
                        done=batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),    # (NUM_ACTORS,)
                        action=action,
                        value=value,
                        reward=reward_flat,
                        log_prob=log_prob,
                        obs=obs_batch,
                        info=info,
                        gift_reward=gift_reward,
                        gift_given=gift_given,
                    )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        # Without parameter sharing, each agent processes E environments.
                        # info_i: (E,A)->(E,)->(E,1)
                        gift_reward = jnp.zeros_like(reward)        # (E,A)
                        gift_given = jnp.zeros_like(reward)         # (E,A)
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],        # (E,A)->(E,)
                            log_prob[i],
                            obs_batch[i],
                            info_i,                 # (E,1)
                            gift_reward[:,i],
                            gift_given[:,i],
                        ))


                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition


            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )


            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.stack([last_obs[:,a,...] for a in env.agents]).reshape(-1, *(env.observation_space()[0]).shape)    # (num_agents * num_envs, obs_shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)


            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)

            # Compute GAE using extrinsic rewards.
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )


                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                advantages = []
                targets = []
                for i in range(env.num_agents):
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)


            # Update policy and value networks.

            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):

                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )


                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []

                metric = {}
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]

                    metric_i = dict(traj_batch[i].info)
                    metric_i['loss'] = loss_info[0]
                    for k, v in metric_i.items():
                        metric[f"agent{i}_{k}"] = v


                    rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1


            AppleCount_save = []
            for i in range(env.num_agents):
                AppleCount_save.append(metric[f"agent{i}_AppleCount_info"][:,:1,0])   # (T,E,1)->(T,1)


            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)


            num_steps = traj_batch[0].reward.shape[0]
            num_envs = config["NUM_ENVS"]

            if config["ALGO"]=="ippo" and config["ENV_KWARGS"]["inequity_aversion"]:

                ori_reward_return_average = 0.0
                for i in range(env.num_agents):
                    ori_reward = traj_batch[i].info["original_rewards"].reshape((num_steps, num_envs))  # (T,E,1)->(T,E)
                    ori_reward_return_per_episode = jnp.sum(ori_reward, axis=0)                         # (E,)
                    ori_reward_return_mean_per_agent = jnp.mean(ori_reward_return_per_episode)
                    metric[f"agent{i}_env_return"] = ori_reward_return_mean_per_agent
                    metric[f"agent{i}_total_return"] = ori_reward_return_mean_per_agent
                    ori_reward_return_average = ori_reward_return_average + ori_reward_return_mean_per_agent
                metric["average_env_return"] = ori_reward_return_average/env.num_agents

                # 2. shaped reward
                returned_episode_returns_average = 0.0
                for i in range(env.num_agents):
                    returned_episode_returns = traj_batch[i].info["returned_episode_returns"].reshape((num_steps, num_envs))    # (T,E,1)->(T,E)
                    returned_episode_returns_per_episode = returned_episode_returns[-1]                     # (E,)
                    returned_episode_returns_per_agent = jnp.mean(returned_episode_returns_per_episode)
                    metric[f"agent{i}_received_reward_return"] = returned_episode_returns_per_agent      # env is already giving "shaped" reward
                    metric[f"agent{i}_total_return_using_reward"] = returned_episode_returns_per_agent   # env is already giving "shaped" reward
                    returned_episode_returns_average = returned_episode_returns_average + returned_episode_returns_per_agent
                metric["average_env_return_using_reward"] = returned_episode_returns_average/env.num_agents


                for i in range(env.num_agents):
                    metric[f"agent{i}_received_gift_return"] = 0.0


                for i in range(env.num_agents):
                    metric[f"agent{i}_given_gift_return"] = 0.0

            elif config["ALGO"]=="ippo" and config["ENV_KWARGS"]["shared_rewards"]:

                ori_reward_return_average = 0.0
                for i in range(env.num_agents):
                    ori_reward = traj_batch[i].info["original_rewards"].reshape((num_steps, num_envs))  # (T,E,1)->(T,E)
                    ori_reward_return_per_episode = jnp.sum(ori_reward, axis=0)                         # (E,)
                    ori_reward_return_mean_per_agent = jnp.mean(ori_reward_return_per_episode)
                    metric[f"agent{i}_env_return"] = ori_reward_return_mean_per_agent
                    metric[f"agent{i}_total_return"] = ori_reward_return_mean_per_agent
                    ori_reward_return_average = ori_reward_return_average + ori_reward_return_mean_per_agent
                metric["average_env_return"] = ori_reward_return_average/env.num_agents

                # 2. shaped reward
                returned_episode_returns_average = 0.0
                for i in range(env.num_agents):
                    returned_episode_returns = traj_batch[i].info["returned_episode_returns"].reshape((num_steps, num_envs))    # (T,E,1)->(T,E)
                    returned_episode_returns_per_episode = returned_episode_returns[-1]                     # (E,)
                    returned_episode_returns_per_agent = jnp.mean(returned_episode_returns_per_episode)
                    metric[f"agent{i}_received_reward_return"] = returned_episode_returns_per_agent      # env is already giving "shaped" reward
                    metric[f"agent{i}_total_return_using_reward"] = returned_episode_returns_per_agent   # env is already giving "shaped" reward
                    returned_episode_returns_average = returned_episode_returns_average + returned_episode_returns_per_agent
                metric["average_env_return_using_reward"] = returned_episode_returns_average/env.num_agents


                for i in range(env.num_agents):
                    metric[f"agent{i}_received_gift_return"] = 0.0


                for i in range(env.num_agents):
                    metric[f"agent{i}_given_gift_return"] = 0.0

            else:


                returned_episode_returns_average = 0.0
                returns_using_reward_average = 0.0
                for i in range(env.num_agents):
                    returned_episode_returns = traj_batch[i].info["returned_episode_returns"].reshape((num_steps, num_envs))    # (T,E,1)->(T,E)
                    returned_episode_returns_per_episode = returned_episode_returns[-1]                     # (E,)
                    returned_episode_returns_per_agent = jnp.mean(returned_episode_returns_per_episode)
                    metric[f"agent{i}_env_return"] = returned_episode_returns_per_agent
                    returned_episode_returns_average = returned_episode_returns_average + returned_episode_returns_per_agent

                    reward_from_traj_batch = traj_batch[i].reward    # (T,E)
                    episode_returns_per_episode_using_reward = jnp.sum(reward_from_traj_batch, axis=0)                    # (E,)
                    episode_returns_per_agent_using_reward = jnp.mean(episode_returns_per_episode_using_reward)
                    metric[f"agent{i}_received_reward_return"] = episode_returns_per_agent_using_reward      # env is already giving "shaped" reward
                    returns_using_reward_average = returns_using_reward_average + episode_returns_per_agent_using_reward

                    gift_reward = traj_batch[i].gift_reward  # (T,E)
                    gift_return_per_episode = jnp.sum(gift_reward, axis=0)  # shape: (E,)
                    gift_return_mean_per_agent = jnp.mean(gift_return_per_episode)
                    metric[f"agent{i}_received_gift_return"] = gift_return_mean_per_agent

                    gift_given = traj_batch[i].gift_given    # shape (T, E)
                    gift_given_per_episode = jnp.sum(gift_given, axis=0)  # shape: (E,)
                    gift_given_mean_per_agent = jnp.mean(gift_given_per_episode)
                    metric[f"agent{i}_given_gift_return"] = gift_given_mean_per_agent

                    metric[f"agent{i}_total_return"] = (
                        returned_episode_returns_per_agent + gift_return_mean_per_agent - gift_given_mean_per_agent
                    )
                    metric[f"agent{i}_total_return_using_reward"] = (
                        episode_returns_per_agent_using_reward + gift_return_mean_per_agent - gift_given_mean_per_agent
                    )
                metric["average_env_return"] = returned_episode_returns_average/env.num_agents
                metric["average_env_return_using_reward"] = returns_using_reward_average/env.num_agents


            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            for i in range(env.num_agents):
                metric[f"agent{i}_AppleCount_info"] = metric[f"agent{i}_AppleCount_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(callback, metric)


            last_extrinsic_reward_env0 = []
            for i in range(env.num_agents):
                last_extrinsic_reward_env0.append(traj_batch[i].reward[:,:1])     # (T, 1)
            metric["last_extrinsic_reward_env0"] = jnp.stack(last_extrinsic_reward_env0, axis=0)  # (A,T,E=10)

            metric["last_AppleCount_env0"] = jnp.stack(AppleCount_save, axis=0)   # (A,T,E=10)


            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}


    return train

def single_run(config):
    config = OmegaConf.to_container(config)


    wandb.login()
    if config["ALGO"] == 'ippo':
        if config["ENV_KWARGS"]["inequity_aversion"]:
            group_name = f'env={str(config["ENV_NAME"])[:9]}_4a_{config["MAP_NAME"]}_algo=IA_alpha{config["ENV_KWARGS"]["inequity_aversion_alpha"]}_beta{config["ENV_KWARGS"]["inequity_aversion_beta"]}_num_envs={config["NUM_ENVS"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}'
            run_name=f'{str(config["ENV_NAME"])[:9]}_4a_{config["MAP_NAME"]} seed={config["SEED"]} ia,{config["ENV_KWARGS"]["inequity_aversion_alpha"]},{config["ENV_KWARGS"]["inequity_aversion_beta"]} {config["NUM_ENVS"]},{config["NUM_STEPS"]},{config["ENV_KWARGS"]["num_inner_steps"]}'
        else:
            group_name = f'env={str(config["ENV_NAME"])[:9]}_4a_{config["MAP_NAME"]}_algo={config["ALGO"]}_reward={config["REWARD"]}_num_envs={config["NUM_ENVS"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}'
            run_name=f'{str(config["ENV_NAME"])[:9]}_4a_{config["MAP_NAME"]} seed={config["SEED"]} ippo-{config["REWARD"]} {config["NUM_ENVS"]},{config["NUM_STEPS"]},{config["ENV_KWARGS"]["num_inner_steps"]}'
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=run_name,
        group=group_name,
        job_type="harvest",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'harvest_{config["MAP_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])


    import pandas as pd
    excel_dir = f"last_harvest/harvest_{config['MAP_NAME']}_{config['ENV_KWARGS']['num_agents']}a"
    path = Path(excel_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename_excel = f"{config['MAP_NAME']}_{config['SEED']}_reward_{config['REWARD']}.xlsx"
    filepath = path / filename_excel

    def save_arrays_to_excel(array1, array2, array3, filename=str(filepath)):
        with pd.ExcelWriter(filename) as writer:
            for name, array in zip(
                ["last_w", "last_extrinsic_reward", "last_AppleCount"],
                [array1, array2, array3]
            ):
                # shape: (A, T, E)
                reshaped = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
                df = pd.DataFrame(reshaped)
                df.to_excel(writer, sheet_name=name, index=False)

        print(f"Saved to {filename}")
    save_arrays_to_excel(
        jnp.ones_like(out["metrics"]["last_extrinsic_reward_env0"][0, -1, :, :, :]),
        out["metrics"]["last_extrinsic_reward_env0"][0, -1, :, :, :],
        out["metrics"]["last_AppleCount_env0"][0, -1, :, :, :],
    )


    print(f'shape of last_extrinsic_reward before: {out["metrics"]["last_extrinsic_reward_env0"].shape}')
    last_extrinsic_reward_env0 = out["metrics"]["last_extrinsic_reward_env0"][0, -1, :, :, 0]   # (1,num_updates,A,T,E) -> (A,T)
    print(f"shape of last_extrinsic_reward: {last_extrinsic_reward_env0.shape}")   # (A,T)
    A, T = last_extrinsic_reward_env0.shape


    last_AppleCount_env0 = out["metrics"]["last_AppleCount_env0"][0, -1, :, :, 0]
    print(f"shape of last_AppleCount: {last_AppleCount_env0.shape}")

    for i in range(A):
        wandb.define_metric(f"last_w_env0_agent{i}", step_metric="last_w_step")
        wandb.define_metric(f"last_ext_reward_env0_agent{i}", step_metric="last_w_step")
        wandb.define_metric(f"last_AppleCount_env0_agent{i}", step_metric="last_w_step")

    for t in range(T):
        log_dict = {"last_w_step": t}
        for i in range(A):
            log_dict[f"last_w_env0_agent{i}"] = 1.0
            log_dict[f"last_ext_reward_env0_agent{i}"] = last_extrinsic_reward_env0[i, t]
            log_dict[f"last_AppleCount_env0_agent{i}"] = last_AppleCount_env0[i, t]
        wandb.log(log_dict)


    save_path = f"./checkpoints/individual/{filename}.pkl"
    if config["PARAMETER_SHARING"]:
        save_path = f"./checkpoints/indvidual/{filename}.pkl"
        save_params(train_state, save_path)
        params = load_params(save_path)
    else:
        params = []
        for i in range(config['ENV_KWARGS']['num_agents']):
            save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
            save_params(train_state[i], save_path)
            params.append(load_params(save_path))

    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/harvest_4a"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):


        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation="relu")  # Use the same network configuration as during training.
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)

            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i],axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action


        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]


        img = env.render(state)
        pics.append(img)

        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')


        print("###################")


    from datetime import datetime
    import pytz


    korea_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(korea_tz)

    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    if config["ALGO"] == 'ippo':
        if config["ENV_KWARGS"]["inequity_aversion"]:
            gif_path = f'{root_dir}/algo=IA_{formatted_time}_frame_{o_t+1}_{config["MAP_NAME"]}_alpha={config["ENV_KWARGS"]["inequity_aversion_alpha"]}_beta={config["ENV_KWARGS"]["inequity_aversion_beta"]}_reward={config["REWARD"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}_seed={config["SEED"]}.gif'
        else:
            gif_path = f'{root_dir}/algo={config["ALGO"]}_reward={config["REWARD"]}_{formatted_time}_frame_{o_t+1}_{config["MAP_NAME"]}_num_steps={config["NUM_STEPS"]}_envsteps={config["ENV_KWARGS"]["num_inner_steps"]}_seed={config["SEED"]}.gif'


    print(f"Saving Episode GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(

    gif_path,
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=200,
    loop=1,
    )


def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "cleanup",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {


            "SEED": {"values": [42, 52, 62]},

        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)

        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])


    wandb.login()
    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_harvest_4a")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
